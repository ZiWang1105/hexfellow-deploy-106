import json
from pathlib import Path

import qpsolvers
import numpy as np
import pinocchio as pin
import pytest

from hexfellow_deploy.benchmark.diverse import DiverseTrajectory
from hexfellow_deploy.benchmark.run import initial_state
from hexfellow_deploy.robot.upstream import PROJECT_ROOT
from hexfellow_deploy.robot.joints import LIFT
from hexfellow_deploy.robotics.control.ik_core import PinkIKCore
from hexfellow_deploy.robotics.control.motion_profile import controller_config_for_profile

SPECS = sorted((PROJECT_ROOT/'configs/benchmarks/generalization').glob('*.json'))
SPECS = [p for p in SPECS if p.name != 'suite.json']


@pytest.fixture(scope='module')
def model():
    return PinkIKCore(config=controller_config_for_profile('x7-gp80-fast-trajectory-sim-v1',
        lift_position_margin=0.), base_mode='fixed', locked_joint_names=(LIFT,)).model


@pytest.mark.parametrize('path', SPECS, ids=lambda p: p.stem)
def test_diverse_derivatives_return_and_time_scaling(model, path):
    spec = json.loads(path.read_text())
    trajectory = DiverseTrajectory(model, initial_state(spec['posture']), spec, 1.2)
    slower = DiverseTrajectory(model, initial_state(spec['posture']), spec, .6)
    assert slower.travel_s == pytest.approx(2*trajectory.travel_s)
    eps = 1e-6
    for t in np.linspace(0, trajectory.duration_s, 73):
        poses, velocity, acceleration = trajectory.sample(t)
        after, next_velocity, _ = trajectory.sample(t+eps)
        before, prev_velocity, _ = trajectory.sample(t-eps)
        for side in poses:
            finite = pin.log6(before[side].inverse()*after[side]).vector/(2*eps)
            np.testing.assert_allclose(velocity[side], finite, atol=6e-5, rtol=1e-4)
            np.testing.assert_allclose(acceleration[side], (next_velocity[side]-prev_velocity[side])/(2*eps), atol=.002, rtol=1e-3)
            if side not in trajectory.moving_arms:
                np.testing.assert_allclose(poses[side].homogeneous, trajectory.initial[side].homogeneous, atol=1e-12)
                np.testing.assert_allclose(velocity[side], 0., atol=1e-12)
    final, v, a = trajectory.sample(trajectory.duration_s)
    for side in final:
        np.testing.assert_allclose(final[side].homogeneous, trajectory.initial[side].homogeneous, atol=1e-12)
        np.testing.assert_allclose(v[side], 0., atol=1e-12)
        np.testing.assert_allclose(a[side], 0., atol=1e-12)


def test_left_envelope_is_not_mistaken_for_stationary_right(model):
    from hexfellow_deploy.benchmark.trajectories import envelope_report
    from hexfellow_deploy.robotics.control.motion_profile import load_motion_profile
    spec = json.loads((PROJECT_ROOT/'configs/benchmarks/generalization/left_sweep.json').read_text())
    path = DiverseTrajectory(model, initial_state(spec['posture']), spec, 1.2)
    result = envelope_report(path, load_motion_profile('x7-gp80-fast-trajectory-sim-v1'))
    assert result['maxima']['linear_velocity'] == pytest.approx(1.2, rel=.001)
