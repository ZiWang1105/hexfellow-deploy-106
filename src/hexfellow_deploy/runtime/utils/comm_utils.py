from contextlib import contextmanager
from typing import Generator, List
from multiprocessing.synchronize import Lock as LockType
import os
import errno
import logging


@contextmanager
def acquire_with_timeout(
    lock: LockType, timeout: float, raise_timeout: bool = False
) -> Generator[bool, None, None]:
    """Context manager that acquires a multiprocessing.Lock with a timeout."""
    acquired = lock.acquire(timeout=timeout)
    if not acquired:
        if raise_timeout:
            raise TimeoutError("Failed to acquire lock within timeout.")
        else:
            yield False
            return
    try:
        yield True
    finally:
        # Always release if (and only if) we actually acquired it
        lock.release()


def check_locked(lock: LockType) -> bool:
    """Check if the lock is locked."""
    acquired = lock.acquire(timeout=0)
    if acquired:
        lock.release()
        return False
    else:
        return True


def set_rt(policy="fifo", prio=50):
    pol = {"fifo": os.SCHED_FIFO, "rr": os.SCHED_RR}[policy]
    try:
        os.sched_setscheduler(0, pol, os.sched_param(prio))
        cur_pol = os.sched_getscheduler(0)
        cur_pri = os.sched_getparam(0).sched_priority
        logging.info(f"RT set: policy={cur_pol} priority={cur_pri}")
        return True
    except PermissionError as e:
        logging.warning(
            f"No permission (check PAM limits, session, or container caps): {e}"
        )
        return False
    except OSError as e:
        if e.errno == errno.EINVAL:
            logging.warning("Invalid priority/policy (out of range or unsupported).")
        else:
            logging.warning(f"sched_setscheduler failed: {e}")
        return False


def set_affinity(affinity: int | List[int]) -> bool:
    """Set the affinity of the current process."""
    try:
        if isinstance(affinity, int):
            affinity = [affinity]
        pid = os.getpid()
        os.sched_setaffinity(pid, affinity)
        logging.info(f"Set affinity of process {pid} to {affinity}")
        return True
    except PermissionError as e:
        logging.warning(
            f"No permission (check PAM limits, session, or container caps): {e}"
        )
    except OSError as e:
        logging.warning(f"sched_setaffinity failed: {e}")
    return False
