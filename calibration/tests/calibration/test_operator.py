import pytest
from policy_sim.calibration.operator import Rehearsal


def machine():
    r = Rehearsal([{'time_s': 1.0}, {'time_s': 2.0}], 3.0)
    r.heartbeat(0)
    return r


def arrive(r, now=0):
    for _ in range(11):
        r.heartbeat(now)
        r.tick(.1, now)


def test_operator_must_capture_before_next_and_never_captures_in_motion():
    r = machine()
    r.command('next', now=0, version=0)
    with pytest.raises(ValueError, match='stationary'):
        r.command('capture', now=0)
    arrive(r)
    with pytest.raises(ValueError, match='Capture this pose'):
        r.command('next', now=0)
    r.command('capture', now=0)
    assert r.phase == 'at_pose'
    r.command('next', now=0)
    assert r.phase == 'moving' and r.pose == 2


def test_pause_overrides_stale_version_resume_is_explicit():
    r = machine()
    r.command('next', now=0)
    r.tick(.1, 0)
    r.command('pause', now=0, version=-1)
    saved = r.time
    r.tick(.1, 0)
    assert r.time == saved
    with pytest.raises(ValueError, match='stationary'):
        r.command('capture', now=0)
    r.command('resume', now=0)
    assert r.phase == 'moving'


def test_browser_loss_pauses_and_reconnect_does_not_resume():
    r = machine()
    r.command('next', now=0)
    r.tick(.1, 3)
    assert r.phase == 'paused'
    with pytest.raises(ValueError, match='heartbeat'):
        r.command('resume', now=3)
    r.heartbeat(3)
    r.tick(.1, 3)
    assert r.phase == 'paused'
    r.command('resume', now=3)
    assert r.phase == 'moving'


def test_duplicate_next_does_not_queue_another_pose():
    r = machine()
    r.command('next', now=0, version=0)
    with pytest.raises(ValueError, match='State changed'):
        r.command('next', now=0, version=0)
    assert r.pose == 1


def test_return_requires_complete_captures_and_can_pause():
    r = machine()
    with pytest.raises(ValueError, match='all poses'):
        r.command('return', now=0)
    for _ in range(2):
        r.command('next', now=0)
        arrive(r)
        r.command('capture', now=0)
    assert r.status()['can_solve']
    r.command('return', now=0)
    r.command('pause', now=0)
    assert r.phase == 'paused_return'
    r.command('resume', now=0)
    arrive(r)
    assert r.phase == 'complete' and r.status()['can_solve']
