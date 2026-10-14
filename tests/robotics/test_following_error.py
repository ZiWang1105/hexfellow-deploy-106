import numpy as np
import pytest

from hexfellow_deploy.robotics.control.following_error import FollowingErrorGovernor, FollowingErrorMonitor


def test_growing_lag_slows_before_moderate_limit_and_recovers_gradually():
    g = FollowingErrorGovernor()
    q, v = np.zeros(14), np.zeros(14)
    assert g.update(q, q, v, v, .02) == 1.
    command = q.copy()
    command[8] = .025
    velocity = v.copy()
    velocity[8] = .4
    scale = g.update(command, q, velocity, v, .02)
    assert 0. < scale < .5  # Lag forecast includes another 40 ms of divergence.
    assert g.observation()['joint_index'] == 8
    command[8] = .055
    assert g.update(command, q, velocity, v, .02) == 0.
    assert g.update(q, q, v, v, .02) == pytest.approx(.04)
    for _ in range(30):
        g.update(q, q, v, v, .02)
    assert g.scale == 1.
    g.reset()
    assert g.observation()['error_rad'] == 0.


@pytest.mark.parametrize('sign', [-1, 1])
def test_slowdown_is_symmetric_and_does_not_treat_catching_up_as_worsening(sign):
    g = FollowingErrorGovernor()
    q, v = np.zeros(14), np.zeros(14)
    q[0] = sign * .035
    v[0] = sign * .2
    growing = g.update(q, np.zeros(14), v, np.zeros(14), .02)
    g.reset()
    catching_up = g.update(q, np.zeros(14), np.zeros(14), v, .02)
    assert catching_up > growing


def test_moderate_lag_gets_time_to_recover_but_blockage_and_large_errors_still_fault():
    monitor = FollowingErrorMonitor()
    for t in [0., .12, .3, 1.]:
        assert monitor.check(.065, t, governed=True) is None
    assert monitor.check(.02, 1.1, governed=True) is None
    assert monitor.check(.065, 1.2, governed=True) is None
    assert monitor.check(.065, 3.3, governed=True) == 'arm following error did not recover within timeout'
    monitor = FollowingErrorMonitor()
    assert monitor.check(.085, 0., governed=True) is None
    assert monitor.check(.085, .11, governed=True) == 'sustained arm following error'
    assert FollowingErrorMonitor().check(.101, 0., governed=True) == 'hard arm following error'


def test_preparation_and_legacy_control_keep_original_monitor():
    monitor = FollowingErrorMonitor()
    assert monitor.check(.06, 0., governed=False) is None
    assert monitor.check(.06, .11, governed=False) == 'sustained arm following error'
    assert monitor.check(.06, .12, governed=True) is None
