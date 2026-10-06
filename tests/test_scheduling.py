"""Python runtime settings apply inside DA children before device construction."""
import gc
import multiprocessing as mp
import sys

import pytest
from hexfellow_deploy.scheduling import python_scheduling, runtime_state


def test_settings_are_process_local_and_restored_after_failure():
    parent = runtime_state()
    receiver, sender = mp.Pipe(duplex=False)
    def worker():
        before = runtime_state()
        try:
            with python_scheduling(dict(gil_switch_s=.0002, cyclic_gc=False)):
                sender.send(runtime_state())
                raise RuntimeError('driver failed')
        except RuntimeError:
            pass
        sender.send(dict(before=before, after=runtime_state()))
    process = mp.Process(target=worker)
    process.start()
    try:
        assert receiver.poll(5)
        during = receiver.recv()
        assert during['gil_switch_s'] == pytest.approx(.0002)
        assert during['cyclic_gc'] is False
        assert receiver.poll(5)
        restored = receiver.recv()
        assert restored['before'] == restored['after']
        assert runtime_state() == parent
    finally:
        process.join(5)
        if process.is_alive():
            process.terminate()
            process.join()
        receiver.close()
        sender.close()
    assert process.exitcode == 0


def test_disabled_automatic_gc_does_not_collect_cycles_during_control():
    previous = gc.get_threshold()
    finalized = []
    class Cycle:
        def __init__(self):
            self.cycle = self
        def __del__(self):
            finalized.append(1)
    try:
        with python_scheduling(dict(gil_switch_s=.0002, cyclic_gc=False)):
            gc.set_threshold(1, 1, 1)
            for _ in range(1000):
                Cycle()
            assert not finalized
        gc.collect()
        assert len(finalized) == 1000
    finally:
        gc.set_threshold(*previous)


@pytest.mark.parametrize('settings', [dict(gil_switch_s=0), dict(gil_switch_s=float('nan')),
                                     dict(cyclic_gc='false'), dict(typo=True)])
def test_invalid_configuration_cannot_change_runtime(settings):
    before = (gc.isenabled(), sys.getswitchinterval())
    with pytest.raises(ValueError):
        with python_scheduling(settings):
            pytest.fail('Invalid settings entered')
    assert (gc.isenabled(), sys.getswitchinterval()) == before
