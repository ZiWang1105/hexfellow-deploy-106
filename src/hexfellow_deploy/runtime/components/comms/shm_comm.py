import struct
import sys
from multiprocessing import shared_memory
from typing import Tuple, Optional, Generator
from typing_extensions import override
from contextlib import contextmanager
import logging

import numpy as np
import posix_ipc

from hexfellow_deploy.runtime.components.comms.base_comm import DABaseComm

# Python 3.13+ lets us opt a SharedMemory out of the resource_tracker (we manage
# unlink ourselves), which avoids the "N leaked shared_memory objects" warnings
# under fork. No-op on older runtimes (e.g. the 3.10 ROS robot env).
_TRACK_KW = {"track": False} if sys.version_info >= (3, 13) else {}


class DASHMComm(DABaseComm):
    """Shared memory communication component.

    A topic owns exactly one shared-memory segment **and** one POSIX named
    semaphore (a per-topic mutex), both keyed off the topic name. Because the
    semaphore is *named*, it is shared by fork (the manager's component
    processes) and openable by name from an unrelated process — so
    :meth:`DASHMNumPyComm.attach` yields a fully-functional comm (buffer + lock),
    not just a raw buffer. The owner (``create=True``) creates both and unlinks
    them on cleanup; an attacher (``create=False``) opens both by name.

    Caveat: a POSIX semaphore is not released if its holder is killed mid-write,
    so a crash between ``acquire`` and ``release`` wedges blocking readers until
    the next restart (the owner's create-path unlinks stale locks). Consumers in
    a real-time loop should pass a finite ``timeout`` rather than block forever.
    """

    _SEM_PREFIX = "/da_"
    _SEM_SUFFIX = "_lock"

    def __init__(self, name: str, size: int, *, create: bool = True):
        super().__init__(name)
        self._size = size
        self._create = create
        self._sem_name = f"{self._SEM_PREFIX}{self._name}{self._SEM_SUFFIX}"
        self._shm = None
        self._sem = None

    @override
    def initialize(self) -> bool:
        """Create (or attach to) the shared memory segment and its named lock.

        The owner creates the semaphore (the per-topic mutex, initial value 1)
        *before* the segment, so any process that can open the segment is
        guaranteed the lock already exists.
        """
        if self._create:
            logging.getLogger("DASHMComm").info(f"Creating shared memory {self._name}")
            # remove any stale segment / semaphore from a previous crashed run
            try:
                stale = shared_memory.SharedMemory(name=self._name)
                stale.close()
                stale.unlink()
            except FileNotFoundError:
                pass
            try:
                posix_ipc.unlink_semaphore(self._sem_name)
            except posix_ipc.ExistentialError:
                pass
            self._sem = posix_ipc.Semaphore(
                self._sem_name, flags=posix_ipc.O_CREAT, initial_value=1
            )
        else:
            logging.getLogger("DASHMComm").info(
                f"Attaching to shared memory {self._name}"
            )
            try:
                self._sem = posix_ipc.Semaphore(self._sem_name)
            except posix_ipc.ExistentialError as e:
                raise FileNotFoundError(
                    f"attach: lock semaphore for topic '{self._name}' not found "
                    f"({self._sem_name}) — is the owner (DAManager) running?"
                ) from e
        self._shm = shared_memory.SharedMemory(
            name=self._name, create=self._create, size=self._size, **_TRACK_KW
        )
        # When attaching, SharedMemory ignores the requested size and adopts the
        # segment's actual size, so an undersized declaration would otherwise
        # silently read/write only a prefix. Fail loudly on any mismatch.
        if not self._create and self._shm.size != self._size:
            raise ValueError(
                f"attach size mismatch for '{self._name}': declared {self._size} bytes "
                f"but the live segment is {self._shm.size} bytes — the attaching process "
                f"must supply the same shape/dtype the owner created."
            )
        logging.getLogger("DASHMComm").info(f"Shared memory {self._name} initialized")
        self._initialized = True
        return True

    @contextmanager
    def _hold(self, timeout: Optional[float]) -> Generator[bool, None, None]:
        """Acquire the per-topic named semaphore (blocking if timeout is None).

        Yields True if acquired (and releases on exit), False on timeout. NOTE: a
        positive ``timeout`` uses ``sem_timedwait``, which is Linux-only — on
        macOS pass ``None`` (block) or ``0`` (try); a positive timeout would
        raise ``NotImplementedError``.
        """
        try:
            self._sem.acquire(timeout)  # None=block, 0=try, >0=timed (Linux)
            acquired = True
        except posix_ipc.BusyError:
            acquired = False
        if not acquired:
            yield False
            return
        try:
            yield True
        finally:
            self._sem.release()

    @override
    def run(self) -> None:
        """Run the shared memory communication component."""
        pass  # nothing to do here

    @override
    def clean_up(self) -> bool:
        """Close the segment + semaphore; the owner also unlinks them."""
        if self._shm is not None:
            self._shm.close()
            if self._create:
                self._shm.unlink()
        if self._sem is not None:
            # Unlink BEFORE close: posix_ipc's Semaphore.unlink() raises on a
            # closed handle, so close-then-unlink silently leaks the segment.
            # Do not guard it — a failure here is a real bug, not expected noise.
            if self._create:
                self._sem.unlink()
            self._sem.close()
        self._cleaned_up = True
        return True


class DASHMNumPyComm(DASHMComm):
    """Shared memory communication component for numpy arrays."""

    def __init__(
        self,
        name: str,
        size: Tuple[int, ...],
        dtype: np.dtype,
        *,
        create: bool = True,
    ):
        # compute the size of the shared memory. Normalize the shape to a tuple so
        # `.shape` is always iterable regardless of how the config expressed it
        # (e.g. `15`, `[15]`, or `${tuple:[15]}`).
        self._dtype = np.dtype(dtype)
        self._timestamp_size = 8
        self._shape = (int(size),) if isinstance(size, int) else tuple(size)
        shm_size = int(np.prod(self._shape) * self._dtype.itemsize) + self._timestamp_size
        super().__init__(name, shm_size, create=create)
        self._array = None

    @classmethod
    def attach(cls, *args, **kwargs) -> "DASHMNumPyComm":
        """Map a segment created by another (non-forked) process.

        Builds the wrapper with ``create=False`` and initializes it, so an
        external consumer (e.g. a model runner in its own process) can read/write
        a topic owned by a running :class:`DAManager`. The positional/keyword
        args are the wrapper's own (e.g. ``joint_names``, ``height``/``width``);
        pass exactly the shape/dtype the owner created — a size mismatch raises
        in ``initialize``. The per-topic named semaphore is opened by name, so
        ``send``/``receive`` on an attached comm are lock-protected against the
        owner's writes.
        """
        if "create" in kwargs:
            raise TypeError("attach() always uses create=False; do not pass 'create'")
        comm = cls(*args, create=False, **kwargs)
        comm.initialize()
        return comm

    @override
    def initialize(self) -> bool:
        """Initialize the shared memory."""
        super().initialize()

        self._array = np.ndarray(
            self._shape, dtype=self._dtype, buffer=self._shm.buf[self._timestamp_size :]
        )
        return True

    @override
    def send(
        self, data: Tuple[float, np.ndarray], timeout: Optional[float] = None
    ) -> bool:
        timestamp, data = data
        assert isinstance(timestamp, float)
        assert (
            data.shape == self._array.shape
        ), f"Shape mismatch: {data.shape} != {self._array.shape}"
        assert (
            data.dtype == self._array.dtype
        ), f"Dtype mismatch: {data.dtype} != {self._array.dtype}"
        with self._hold(timeout) as acquired:
            if not acquired:
                return False
            self._array[:] = data
            self._shm.buf[0 : self._timestamp_size] = struct.pack(">d", timestamp)
        return True

    @override
    def receive(self, timeout: Optional[float] = None) -> Tuple[float, np.ndarray]:
        with self._hold(timeout) as acquired:
            if not acquired:
                return 0.0, np.zeros(self._shape, dtype=self._dtype)
            timestamp = struct.unpack(">d", self._shm.buf[0 : self._timestamp_size])[0]
            return timestamp, self._array.copy()

    @property
    def dtype(self) -> np.dtype:
        """Get the dtype of the shared memory."""
        return self._dtype

    @property
    def shape(self) -> Tuple[int, ...]:
        """Get the shape of the shared memory."""
        return self._shape

    @property
    def array(self) -> np.ndarray:
        """Get the array of the shared memory.
        Warning: this is not lock-protected. Do not modify the array directly.
        """
        return self._array

    @contextmanager
    def zero_copy_receive(
        self, timeout: Optional[float] = None
    ) -> Generator[Tuple[float, np.ndarray], None, None]:
        """Context manager that allows zero-copy receive of the shared memory.
        Writing to the array is not allowed.
        """
        with self._hold(timeout) as acquired:
            if not acquired:
                yield 0.0, np.zeros(self._shape, dtype=self._dtype)
                return
            timestamp = struct.unpack(">d", self._shm.buf[0 : self._timestamp_size])[0]
            try:
                self._array.flags.writeable = False
                yield timestamp, self._array
            finally:
                self._array.flags.writeable = True
