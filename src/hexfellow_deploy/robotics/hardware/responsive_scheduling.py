"""Opt-in desktop scheduling, established before robot ownership.

CPU pools exclude our background workers, not other applications or IRQs.
RealtimeKit grants only this user's threads; no global scheduler changes.
"""
from contextlib import contextmanager
import json
import os
from pathlib import Path
import resource
import subprocess
import threading

POOL_ENV = 'POLICY_SIM_RESPONSIVE_CPUS'


def configure():
    allowed = sorted(os.sched_getaffinity(0))
    cores = {}
    for cpu in allowed:
        root = Path(f'/sys/devices/system/cpu/cpu{cpu}')
        key = ((root/'topology/physical_package_id').read_text().strip(),
               (root/'topology/core_id').read_text().strip())
        try:
            frequency = int((root/'cpufreq/cpuinfo_max_freq').read_text())
        except OSError:
            frequency = 0
        entry = cores.setdefault(key, [frequency, []])
        entry[1].append(cpu)
    ranked = sorted(cores.values(), key=lambda row: (-row[0], min(row[1])))
    if len(ranked) < 4:
        raise RuntimeError('Responsive calibration requires four physical CPU cores')
    pools = dict(compute=ranked[0][1], owner=ranked[1][1], sdk=ranked[2][1],
                 background=[cpu for _, group in ranked[3:] for cpu in group])
    pools['gateway'] = pools['owner'] + pools['sdk']
    os.environ[POOL_ENV] = json.dumps(pools)
    os.sched_setaffinity(0, pools['background'])
    return dict(pools=pools, exclusive_system_cpus=False, realtime_guarantee=False)


@contextmanager
def thread_cpu(cpus):
    previous = os.sched_getaffinity(0)
    try:
        os.sched_setaffinity(0, cpus)
        yield
    finally:
        os.sched_setaffinity(0, previous)


def request_priority(tid, priority=10):
    """Startup only: promote a live thread belonging to the calling process.

RESET_ON_FORK also resets new SDK threads, so each must be promoted explicitly.
The broker's bounded RTTIME hard limit persists until this process exits.
"""
    if not Path(f'/proc/{os.getpid()}/task/{tid}').exists():
        raise RuntimeError(f'Scheduling target thread {tid} is not in this process')
    allowance = resource.getrlimit(resource.RLIMIT_RTPRIO)[0]
    backend = 'rlimit'
    if allowance == resource.RLIM_INFINITY or allowance >= priority:
        os.sched_setscheduler(tid, os.SCHED_RR | os.SCHED_RESET_ON_FORK, os.sched_param(priority))
    else:
        backend = 'rtkit'
        prefix = ['busctl', '--timeout=3', '--allow-interactive-authorization=no', 'call',
                  'org.freedesktop.RealtimeKit1', '/org/freedesktop/RealtimeKit1']
        result = subprocess.run(prefix + ['org.freedesktop.DBus.Properties', 'Get', 'ss',
            'org.freedesktop.RealtimeKit1', 'RTTimeUSecMax'], capture_output=True, text=True, timeout=4)
        if result.returncode:
            raise RuntimeError('RealtimeKit unavailable: '+result.stderr.strip())
        cap = int(result.stdout.split()[-1])
        if not 0 < cap <= 1_000_000:
            raise RuntimeError('Invalid RealtimeKit RTTIME bound')
        soft, hard = resource.getrlimit(resource.RLIMIT_RTTIME)
        resource.setrlimit(resource.RLIMIT_RTTIME,
            (min(soft, cap) if soft != resource.RLIM_INFINITY else cap,
             min(hard, cap) if hard != resource.RLIM_INFINITY else cap))
        result = subprocess.run(prefix + ['org.freedesktop.RealtimeKit1',
            'MakeThreadRealtimeWithPID', 'ttu', str(os.getpid()), str(tid), str(priority)],
            capture_output=True, text=True, timeout=4)
        if result.returncode:
            raise RuntimeError('RealtimeKit priority denied: '+result.stderr.strip())
    policy, actual = os.sched_getscheduler(tid), os.sched_getparam(tid).sched_priority
    if policy & ~os.SCHED_RESET_ON_FORK != os.SCHED_RR or actual != priority:
        raise RuntimeError('Requested SDK scheduling was not applied')
    return dict(tid=tid, scheduler=policy, priority=actual, backend=backend,
                affinity=sorted(os.sched_getaffinity(tid)), realtime_guarantee=False,
                rttime_limit_us=list(resource.getrlimit(resource.RLIMIT_RTTIME)))


def promote_new_threads(previous):
    rows = []
    for thread in threading.enumerate():
        if thread.native_id is not None and thread.native_id not in previous and thread.is_alive():
            rows.append(dict(name=thread.name, **request_priority(thread.native_id)))
    return rows
