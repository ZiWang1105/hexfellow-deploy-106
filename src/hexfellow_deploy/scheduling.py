"""Process-local Python scheduling settings for independently clocked DA workers."""
from contextlib import contextmanager
import gc
import math
import os
import sys


def validate_settings(settings):
    if set(settings)-{'gil_switch_s', 'cyclic_gc'}:
        raise ValueError('Unknown Python scheduling setting')
    interval = settings.get('gil_switch_s', sys.getswitchinterval())
    if not isinstance(interval, (int, float)) or not math.isfinite(interval) or interval <= 0:
        raise ValueError('gil_switch_s must be finite and positive')
    if 'cyclic_gc' in settings and type(settings['cyclic_gc']) is not bool:
        raise ValueError('cyclic_gc must be a boolean')


def runtime_state():
    return dict(gil_switch_s=sys.getswitchinterval(), cyclic_gc=gc.isenabled(),
                affinity=sorted(os.sched_getaffinity(0)), scheduler=os.sched_getscheduler(0),
                priority=os.sched_getparam(0).sched_priority, realtime_guarantee=False)


@contextmanager
def python_scheduling(settings):
    """Enter before device construction; restore after the sender threads stop."""
    validate_settings(settings)
    previous_interval, previous_gc = sys.getswitchinterval(), gc.isenabled()
    try:
        if settings.get('cyclic_gc') is False:
            # Collect before hardware streaming, never from its periodic loop.
            gc.collect()
            gc.disable()
        elif settings.get('cyclic_gc') is True:
            gc.enable()
        if 'gil_switch_s' in settings:
            sys.setswitchinterval(settings['gil_switch_s'])
        yield
    finally:
        sys.setswitchinterval(previous_interval)
        (gc.enable if previous_gc else gc.disable)()
