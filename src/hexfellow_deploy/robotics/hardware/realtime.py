"""Scoped CPython thread fairness; never changes deadlines or motor limits."""

import sys
from contextlib import contextmanager


@contextmanager
def sdk_thread_fairness(interval_s=.001):
    previous = sys.getswitchinterval()
    sys.setswitchinterval(interval_s)
    try:
        yield {"previous_gil_switch_s": previous, "gil_switch_s": sys.getswitchinterval(),
               "realtime_guarantee": False}
    finally:
        sys.setswitchinterval(previous)


@contextmanager
def control_runtime(*, realtime=True):
    """Scoped GC/GIL policy, restored even after controller or logger failure."""
    import gc
    enabled = gc.isenabled()
    gc.collect()
    gc.disable()
    try:
        with sdk_thread_fairness() as timing, control_cpu('compute') as cpu, control_priority(enabled=realtime) as priority:
            yield {**timing, **cpu, **priority}
    finally:
        if enabled:
            gc.enable()
        else:
            gc.disable()


def realtime_run(function):
    from functools import wraps
    @wraps(function)
    def wrapped(*args, **kwargs):
        arguments = args[0] if args else kwargs.get("args")
        mode = getattr(arguments, "mode", None)
        with control_runtime(realtime=mode != "sim"):
            return function(*args, **kwargs)
    return wrapped


@contextmanager
def control_cpu(role):
    """Keep compute and SDK on separate fast physical cores on hybrid hosts.

    Compute and the SDK thread group each get one different physical core. Scoped
    to this process/thread and its future children; never changes the CPU
    governor or another application's affinity.
    """
    import os
    from pathlib import Path
    if not hasattr(os, 'sched_getaffinity'):
        yield {}
        return
    previous = os.sched_getaffinity(0)
    responsive = os.environ.get('POLICY_SIM_RESPONSIVE_CPUS')
    if responsive:
        import json
        try:
            selected = json.loads(responsive)[role]
            os.sched_setaffinity(0, selected)
            yield {'role': role, 'affinity': sorted(os.sched_getaffinity(0)),
                   'realtime_guarantee': False, 'responsive_pools': True}
        finally:
            os.sched_setaffinity(0, previous)
        return
    previous_pool = os.environ.get('_POLICY_SIM_CPU_POOL')
    pool = previous if previous_pool is None else {int(v) for v in previous_pool.split(',')}
    os.environ['_POLICY_SIM_CPU_POOL'] = ','.join(map(str, sorted(pool)))
    try:
        cores = {}
        for cpu in sorted(pool):
            root = Path(f'/sys/devices/system/cpu/cpu{cpu}')
            try:
                frequency = int((root / 'cpufreq/cpuinfo_max_freq').read_text())
                core = ((root / 'topology/physical_package_id').read_text().strip(),
                        (root / 'topology/core_id').read_text().strip())
            except OSError:
                continue
            cores.setdefault(core, (frequency, cpu))
        ranked = sorted(cores.values(), key=lambda v: (-v[0], v[1]))
        # No assumption about CPU numbering or P/E layout on other machines.
        if len(ranked) >= 2:
            selected = ranked[:1] if role == 'compute' else ranked[1:2]
            os.sched_setaffinity(0, {cpu for _, cpu in selected})
        yield {'role': role, 'affinity': sorted(os.sched_getaffinity(0)), 'realtime_guarantee': False}
    finally:
        os.sched_setaffinity(0, previous)
        if previous_pool is None:
            os.environ.pop('_POLICY_SIM_CPU_POOL', None)
        else:
            os.environ['_POLICY_SIM_CPU_POOL'] = previous_pool


@contextmanager
def control_priority(priority=10, *, enabled=True):
    """Use the user's existing RLIMIT_RTPRIO allowance; no sudo/global changes.

    RESET_ON_FORK keeps loggers and simulated peers at normal priority. Gateway
    workers opt in separately before creating their SDK threads. Failure to get
    priority is reported explicitly, not mistaken for a realtime guarantee.
    """
    import os
    if not enabled:
        yield {'scheduler': 'unchanged_for_simulation', 'realtime_guarantee': False}
        return
    if not hasattr(os, 'sched_setscheduler'):
        yield {'scheduler': 'unsupported', 'realtime_guarantee': False}
        return
    import resource
    previous = os.sched_getscheduler(0)
    parameter = os.sched_getparam(0)
    changed = False
    result = {'requested_priority': priority, 'realtime_guarantee': False,
              'reset_on_fork_until_process_exit': True}
    try:
        allowance = resource.getrlimit(resource.RLIMIT_RTPRIO)[0]
        if allowance < priority and allowance != resource.RLIM_INFINITY:
            if os.environ.get('POLICY_SIM_RESPONSIVE_CPUS'):
                from .responsive_scheduling import request_priority
                import threading
                result.update(request_priority(threading.get_native_id(), priority))
                changed = True
            else:
                result['priority_unavailable'] = f'RLIMIT_RTPRIO={allowance}'
        else:
            try:
                os.sched_setscheduler(0, os.SCHED_RR | os.SCHED_RESET_ON_FORK, os.sched_param(priority))
                changed = True
            except PermissionError as error:
                result['priority_unavailable'] = str(error)
        result.update(scheduler=os.sched_getscheduler(0), priority=os.sched_getparam(0).sched_priority)
        yield result
    finally:
        if changed:
            try:
                os.sched_setscheduler(0, previous, parameter)
            except PermissionError:
                # Linux lets an unprivileged process enable RESET_ON_FORK but
                # not clear it. Restore normal policy/priority while keeping that
                # child-safety flag until this CLI process exits.
                os.sched_setscheduler(0, previous | os.SCHED_RESET_ON_FORK, parameter)


@contextmanager
def ordinary_priority():
    """Unpaced offline previews must not consume Linux's realtime CPU quota."""
    import os
    if not hasattr(os, 'sched_setscheduler'):
        yield
        return
    previous, parameter = os.sched_getscheduler(0), os.sched_getparam(0)
    base = previous & ~os.SCHED_RESET_ON_FORK
    changed = base in (os.SCHED_RR, os.SCHED_FIFO)
    try:
        if changed:
            os.sched_setscheduler(0, os.SCHED_OTHER | (previous & os.SCHED_RESET_ON_FORK), os.sched_param(0))
        yield
    finally:
        if changed:
            os.sched_setscheduler(0, previous, parameter)


def offline_work(function):
    from functools import wraps
    @wraps(function)
    def wrapped(*args, **kwargs):
        with ordinary_priority():
            return function(*args, **kwargs)
    return wrapped
