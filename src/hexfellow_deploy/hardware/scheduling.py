"""Scoped UMI scheduling, prepared before ownership rather than during motion."""
from contextlib import contextmanager
import gc

from ..robot.upstream import load_umi


@contextmanager
def control_memory():
    """Avoid cyclic-GC pauses across control/SDK threads; restore on exit."""
    load_umi()
    from hexfellow_deploy.robotics.hardware.realtime import sdk_thread_fairness
    enabled = gc.isenabled()
    gc.collect()  # Caller must enter BEFORE acquiring physical devices.
    gc.disable()
    try:
        with sdk_thread_fairness(.0002) as timing:
            yield {**timing, 'cyclic_gc_during_control': False}
    finally:
        if enabled:
            gc.enable()


@contextmanager
def control_scheduling(role):
    """Select separate compute/gateway cores using upstream's scoped helpers."""
    load_umi()
    from hexfellow_deploy.robotics.hardware.realtime import control_cpu, control_priority
    import os
    selector = runtime_cpu if os.environ.get("MOBILE_RUNTIME_CPU_POOLS") else control_cpu
    with selector(role) as cpu, control_priority() as priority:
        yield {**cpu, **priority}


def configure_runtime_cpus(config=None):
    """Reserve disjoint *application* CPU pools before spawning any workers.

    This excludes our background workers, not other desktop applications or IRQs.
    No global scheduler/CPU setting is modified. Explicit lists support Thor.
    """
    import json
    import os
    from pathlib import Path
    if not hasattr(os, 'sched_getaffinity'):
        return {'supported': False}
    allowed = sorted(os.sched_getaffinity(0))
    config = config or {}
    physical = {}
    for cpu in allowed:
        path = Path(f'/sys/devices/system/cpu/cpu{cpu}/topology')
        try:
            identity = ((path / 'physical_package_id').read_text().strip(),
                        (path / 'core_id').read_text().strip())
        except OSError:
            identity = ('cpu', cpu)
        physical.setdefault(identity, []).append(cpu)
    groups = list(physical.values())
    if config.get('cpus'):
        pools = config['cpus']
    elif len(groups) >= 4:
        pools = dict(compute=groups[0], gateway=groups[1] + groups[2],
                     background=[c for group in groups[3:] for c in group])
    elif len(groups) >= 3:
        pools = dict(compute=groups[0], gateway=groups[1],
                     background=[c for group in groups[2:] for c in group])
    else:
        pools = dict(compute=allowed, gateway=allowed, background=allowed)
    if set(pools) != {'compute', 'gateway', 'background'}:
        raise ValueError('CPU pools require compute, gateway and background')
    for role, cpus in pools.items():
        if not cpus or any(type(c) is not int or c not in allowed for c in cpus):
            raise ValueError(f'Invalid {role} CPU pool for this process cpuset')
    overlap = any(set(pools[a]) & set(pools[b]) for a,b in
                  [('compute','gateway'), ('compute','background'), ('gateway','background')])
    if config.get('cpus') and overlap:
        raise ValueError('Explicit CPU pools must be disjoint')
    os.environ['MOBILE_RUNTIME_CPU_POOLS'] = json.dumps(pools)
    os.sched_setaffinity(0, pools['background'])
    return dict(pools=pools, shared_fallback=overlap, exclusive_system_cpus=False,
                parent_affinity=sorted(os.sched_getaffinity(0)))


@contextmanager
def runtime_cpu(role):
    import os
    import json
    previous = os.sched_getaffinity(0) if hasattr(os, 'sched_getaffinity') else None
    pools = json.loads(os.environ.get('MOBILE_RUNTIME_CPU_POOLS', '{}'))
    try:
        if previous is not None and role in pools:
            os.sched_setaffinity(0, pools[role])
        yield dict(role=role, affinity=sorted(os.sched_getaffinity(0)) if previous else [],
                   realtime_guarantee=False)
    finally:
        if previous is not None:
            os.sched_setaffinity(0, previous)


def gateway_thread_pools(cpus):
    """Keep SDK workers off the command owner's physical core when possible."""
    from pathlib import Path
    groups = {}
    for cpu in sorted(cpus):
        root = Path(f'/sys/devices/system/cpu/cpu{cpu}/topology')
        try:
            identity = ((root/'physical_package_id').read_text().strip(),
                        (root/'core_id').read_text().strip())
        except OSError:
            # Unknown topology cannot establish physical-core separation.
            return dict(owner=sorted(cpus), sdk=sorted(cpus), shared_fallback=True)
        groups.setdefault(identity, []).append(cpu)
    cores = list(groups.values())
    if len(cores) < 2:
        return dict(owner=sorted(cpus), sdk=sorted(cpus), shared_fallback=True)
    return dict(owner=cores[0], sdk=[cpu for core in cores[1:] for cpu in core],
                shared_fallback=False)


@contextmanager
def thread_affinity(cpus):
    """Scope the calling thread's affinity; newly created SDK threads inherit it."""
    import os
    if not hasattr(os, 'sched_getaffinity'):
        yield
        return
    previous = os.sched_getaffinity(0)
    try:
        os.sched_setaffinity(0, cpus)
        yield
    finally:
        os.sched_setaffinity(0, previous)
