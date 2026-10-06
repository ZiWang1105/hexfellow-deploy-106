import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'tools/camera'))
from calibration_runtime import CalibrationController, CalibrationSource, describe_arm_arrival_timeout
from policy_sim.hardware import session
from policy_sim.hardware.x7 import HardwareSafetyError


def recorded_failure():
    # Actual endpoint from 20260930T200227.090837Z, in model coordinates.
    target = np.zeros(18)
    target[11:] = [-.4384770822144887, 1.8379999999994097, .5023656145117945,
                  .10866674168212855, 1.5594822962260302, .503460037191385, -.618]
    actual = target.copy()
    actual[11:] = [-.4607870560243439, 1.8307183857626217, .49934291839387335,
                  .11045803027529508, 1.558064286255446, .5021390235344404, -.6183090064713821]
    state = SimpleNamespace(q=actual, v=np.zeros(18), received={s: 0. for s in ('left','right','theta','base')})
    arrival = session.Arrival()
    return state, target, arrival


def test_recorded_stationary_shoulder_still_fails_original_arrival_guard():
    state, target, arrival = recorded_failure()
    with pytest.raises(HardwareSafetyError) as error:
        for tick in range(510):
            state.received = dict.fromkeys(state.received, tick * .02)
            arrival.update(state, target, .02, timeout_active=True)
    detail = describe_arm_arrival_timeout(error.value, arrival, state, phase='calibration', pose_id=1)
    assert detail['stationary']
    assert [j['joint'] for j in detail['joints'] if j['position_failed']] == ['right J1']
    assert 'right J1 position error -1.278 deg' in detail['message']
    assert 'allowed +/-1.146' in detail['message']
    assert not any(j['speed_failed'] for j in detail['joints'])
    np.testing.assert_array_equal(arrival.tolerance, session.TOLERANCE)


def test_speed_failure_is_distinguished_from_position_failure():
    state, target, arrival = recorded_failure()
    state.q = target.copy()
    state.v[16] = .03
    arrival.details['right'] = {'target': target[11:].tolist()}
    error = HardwareSafetyError('right did not settle in 10 seconds of new observations')
    detail = describe_arm_arrival_timeout(error, arrival, state, phase='calibration', pose_id=2)
    assert not detail['stationary']
    assert not any(j['position_failed'] for j in detail['joints'])
    assert 'right J6 speed 1.719 deg/s' in detail['message']


def test_other_fault_is_not_relabelled_as_settling_failure():
    state, target, arrival = recorded_failure()
    assert describe_arm_arrival_timeout(HardwareSafetyError('network unreachable'), arrival, state,
                                       phase='calibration', pose_id=1) is None


def test_default_keeps_fault_shutdown_and_never_marks_arrival(monkeypatch):
    state, target, arrival = recorded_failure()
    source = CalibrationSource()
    source.target_pose = 1
    controller = object.__new__(CalibrationController)
    controller.source = source
    controller.phase = 'default'
    controller.arrival = arrival
    arrival.details['right'] = {'target': target[11:].tolist()}
    def fail(*args):
        raise HardwareSafetyError('right did not settle in 10 seconds of new observations')
    monkeypatch.setattr(controller, '_sample', fail)
    with pytest.raises(HardwareSafetyError, match='right J1 position error'):
        controller.sample(state, 10.)
    assert source.failure_detail['pose_id'] == 1
    assert source.paused and source.interruption_count == 1
    assert source.arrived_pose == 0 and source.commands.empty()
    assert source.summary()['failure_detail'] == source.failure_detail
