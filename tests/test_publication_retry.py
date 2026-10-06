"""Exercise real DA semaphore contention without acquiring robot devices."""
import json
import threading
import time
import uuid

import pytest

from hexfellow_deploy.backend import BackendProxy
from hexfellow_deploy.reference import ReferenceOwner, Receipt
from hexfellow_deploy.control_topics import ReferenceTopic
from mobile_manipulator.robot.joints import JOINTS
from hexfellow_deploy.actuation import decode_mit
from test_actuation import example


class Status:
    def get(self):
        return None


def pending(frames):
    proxy = object.__new__(BackendProxy)
    proxy.frames, proxy.status = frames, Status()
    frame, measured = example(time.monotonic())
    proxy.selected = dict(measured=measured,
                          received=dict.fromkeys(('left','right','theta','base'), measured['timestamp_s']))
    return proxy, frame, decode_mit(frame['mit'])


def test_real_shm_lock_delay_retries_exact_reference_without_advancing_ik():
    frames = ReferenceTopic('retry_'+uuid.uuid4().hex)
    frames.initialize()
    locked = threading.Event()
    def reader():
        with frames._hold(None) as acquired:
            assert acquired
            locked.set()
            time.sleep(.020)
    thread = threading.Thread(target=reader)
    thread.start()
    try:
        assert locked.wait(1.)
        proxy, frame, mit = pending(frames)
        owner = ReferenceOwner()
        solves = []
        sequence, _ = owner.generate(lambda: solves.append(1))
        waits = []
        result = proxy.submit(sequence, frame['command'], mit, dict.fromkeys(JOINTS,0.), on_wait=waits.append)
        assert result['retries'] > 0 and waits
        assert solves == [1] and owner.pending == sequence and owner.accepted == 0
        assert owner.generate(lambda: solves.append(2)) is None
        actual = frames.get(timeout=.01)
        assert actual['sequence'] == sequence
        assert actual['created_s'] == frame['created_s'] == actual['mit']['timestamp_s']
        assert actual['command'] == frame['command'] and actual['mit'] == json.loads(json.dumps(frame['mit']))
        owner.acknowledge(Receipt(sequence, time.monotonic(), 'simulator_applied'))
        assert owner.pending is None and owner.accepted == sequence and solves == [1]
    finally:
        thread.join(1.)
        frames.clean_up()


def test_wedged_shm_still_expires_with_no_partial_publication():
    frames = ReferenceTopic('wedged_'+uuid.uuid4().hex)
    frames.initialize()
    try:
        proxy, frame, mit = pending(frames)
        started = time.monotonic()
        with frames._hold(None):
            with pytest.raises(RuntimeError, match='publication deadline expired'):
                proxy.submit(1, frame['command'], mit, dict.fromkeys(JOINTS,0.), timeout_s=.025)
        assert time.monotonic()-started < .2
        assert frames.get() is None
    finally:
        frames.clean_up()


@pytest.mark.parametrize('condition', ['stop', 'backend_fault', 'unknown_write_failure'])
def test_retry_does_not_bypass_stop_fault_or_unknown_publication(condition):
    class Frames:
        calls = 0
        def put(self, *args, **kwargs):
            self.calls += 1
            if condition == 'unknown_write_failure':
                raise OSError('unknown partial write')
            return False
    frames = Frames()
    proxy, frame, mit = pending(frames)
    if condition == 'backend_fault':
        proxy.status.get = lambda: {'fault': 'device ownership lost'}
    with pytest.raises((RuntimeError, OSError)):
        proxy.submit(1, frame['command'], mit, {}, stopped=lambda: condition == 'stop')
    assert frames.calls == (1 if condition == 'unknown_write_failure' else 0)
