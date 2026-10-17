"""Independent geometry, derivative and actuator-clock checks for the benchmark."""
import qpsolvers  # Pin/Pink shared-library load order.
import numpy as np
import pinocchio as pin
import pytest

from hexfellow_deploy.benchmark.profiles import experiment_profile
from hexfellow_deploy.benchmark.run import initial_state, poses, scene_for_run
from hexfellow_deploy.benchmark.trajectories import Trajectory
from hexfellow_deploy.robot.joints import LIFT, JOINTS
from hexfellow_deploy.robotics.control.ik_core import PinkIKCore
from hexfellow_deploy.robotics.control.motion_profile import load_motion_profile, controller_config_for_profile


@pytest.fixture
def core():
    return PinkIKCore(config=controller_config_for_profile('x7-gp80-highspeed-fixed-base-v1',
                      lift_position_margin=0.), base_mode='fixed', locked_joint_names=(LIFT,))


def test_symbolic_fk_and_jacobian_match_independent_pinocchio(core):
    ca = pytest.importorskip('casadi')
    from hexfellow_deploy.benchmark.nonlinear import symbolic_fk
    q, frames = symbolic_fk(core.model)
    f = ca.Function('geometry', [q], [frames['right'], ca.jacobian(frames['right'][:3, 3], q)])
    rng, data = np.random.default_rng(128), core.model.createData()
    for _ in range(40):
        state = rng.uniform(core.model.lowerPositionLimit, core.model.upperPositionLimit)
        actual = poses(core.model, data, state)['right']
        frame, jac = f(state)
        expected_jac = pin.computeFrameJacobian(core.model, data, state,
            core.model.getFrameId('right_tcp'), pin.LOCAL_WORLD_ALIGNED)
        np.testing.assert_allclose(frame, actual.homogeneous, atol=1e-12, rtol=0)
        np.testing.assert_allclose(jac, expected_jac[:3], atol=1e-12, rtol=0)


def test_symbolic_se3_log_matches_pinocchio_and_is_differentiable_at_identity():
    ca = pytest.importorskip('casadi')
    from hexfellow_deploy.benchmark.nonlinear import se3_log
    transform = ca.SX.sym('transform', 4, 4)
    log = se3_log(transform)
    f = ca.Function('log', [transform], [log, ca.jacobian(log, ca.vec(transform))])
    rng = np.random.default_rng(93)
    for angle in (0., 1e-8, .0001, .2, 1., 2.9):
        axis = rng.normal(size=3)
        T = pin.SE3(pin.exp3(axis/np.linalg.norm(axis)*angle), rng.normal(size=3))
        result, derivative = f(T.homogeneous)
        np.testing.assert_allclose(np.asarray(result).ravel(), pin.log6(T).vector, atol=2e-9, rtol=0)
        assert np.all(np.isfinite(derivative))


@pytest.mark.parametrize('kind', ['vertical', 'lateral', 'forward', 'circle', 'figure8', 'rotate', 'rounded_rectangle'])
def test_target_derivatives_match_finite_differences_and_end_at_rest(core, kind):
    start = poses(core.model, core.model.createData(), initial_state())
    trajectory = Trajectory(start, kind, 1.2, dual=True)
    eps = 1e-6
    peak = 0.
    for t in np.linspace(0, trajectory.duration_s, 181):
        transforms, twists, acceleration = trajectory.sample(t)
        next_transforms, next_twists, _ = trajectory.sample(t+eps)
        for side in start:
            finite = pin.log6(transforms[side].inverse()*next_transforms[side]).vector/eps
            np.testing.assert_allclose(twists[side], finite, atol=5e-5, rtol=1e-4)
            np.testing.assert_allclose(acceleration[side], (next_twists[side]-twists[side])/eps,
                                       atol=.002, rtol=1e-3)
        peak = max(peak, np.linalg.norm(twists['right'][:3]))
    assert 1.17 < peak <= 1.2001
    final, twist, _ = trajectory.sample(trajectory.duration_s)
    np.testing.assert_allclose(final['right'].homogeneous, start['right'].homogeneous, atol=1e-12)
    np.testing.assert_allclose(twist['right'], 0., atol=1e-12)


def test_derived_profile_keeps_joint_limits_and_motor_gains(core, tmp_path):
    original = load_motion_profile('x7-gp80-highspeed-fixed-base-v1')
    derived = load_motion_profile(experiment_profile(tmp_path, hz=100, fast=True,
        tuning={'frame_gain': .5, 'low_acceleration_cost': 2.}))
    assert derived.sha256 != original.sha256
    assert derived.control_dt_s == .01
    for name in ('arm_velocity_rad_s', 'arm_acceleration_rad_s2', 'arm_jerk_rad_s3'):
        np.testing.assert_array_equal(getattr(derived, name), getattr(original, name))
    assert derived.mit_servo == original.mit_servo
    controller_config_for_profile(derived)


def test_public_adapter_transforms_derivatives_at_the_tool_point_and_clears_them():
    from types import SimpleNamespace
    from hexfellow_deploy.control.umi_ik import UMIPinkIK
    # Exercise the public method independently of either kinematics implementation.
    obj = SimpleNamespace(pin=pin, policy_T_umi={s: pin.SE3(np.eye(3), np.array([0., 0., .017]))
                                               for s in ('left', 'right')})
    pose = pin.SE3(pin.exp3(np.array([.2, -.3, .4])), np.array([.6, .3, .8]))
    targets = {s: (pose.translation, pose.rotation) for s in obj.policy_T_umi}
    v = np.array([.1, .2, -.1, .3, -.4, .2])
    UMIPinkIK.track(obj, targets, body_twists={s: v for s in targets}, sample_timestamp_s=1.)
    dt = 1e-6
    for side, offset in obj.policy_T_umi.items():
        before, after = pose*offset, pose*pin.exp6(v*dt)*offset
        expected = pin.log6(before.inverse()*after).vector/dt
        np.testing.assert_allclose(obj.target_twists[side], expected, atol=1e-8)
    assert obj.target_accelerations is None
    obj.dt = .01
    UMIPinkIK.check_target_time(obj, 1.01)
    with pytest.raises(RuntimeError, match='stale or future-dated'):
        UMIPinkIK.check_target_time(obj, 1.021)
    with pytest.raises(RuntimeError, match='stale or future-dated'):
        UMIPinkIK.check_target_time(obj, .99)
    UMIPinkIK.track(obj, targets)
    assert obj.target_twists is None
    with pytest.raises(ValueError, match='require target twists'):
        UMIPinkIK.track(obj, targets, body_accelerations={s: v for s in targets})
    with pytest.raises(ValueError, match='finite 6-vectors'):
        UMIPinkIK.track(obj, targets, body_twists={s: np.full(6, np.nan) for s in targets}, sample_timestamp_s=1.)


def test_screw_trajectory_is_an_independently_reachable_joint_motion(core):
    from hexfellow_deploy.benchmark.trajectories import ScrewTrajectory, quintic
    q = initial_state('extended')
    data = core.model.createData()
    start = poses(core.model, data, q)
    screws = {s: pin.computeFrameJacobian(core.model, data, q,
        core.model.getFrameId(s+'_tcp'), pin.LOCAL)[:, i]*sign
        for s, i, sign in [('left', 1, -1), ('right', 8, 1)]}
    path = ScrewTrajectory(start, screws, 1.3, dual=True)
    for t in np.linspace(1., 1.+path.travel_s, 30):
        h, dh, _ = quintic((t-1.)/path.travel_s)
        joints = q.copy()
        joints[1] -= h*.8
        joints[8] += h*.8
        desired, twists, _ = path.sample(t)
        independent = poses(core.model, data, joints)
        for side in start:
            np.testing.assert_allclose(desired[side].homogeneous, independent[side].homogeneous, atol=1e-12)
        assert abs(dh*.8/path.travel_s) < core.model.velocityLimit[8]


def test_multi_joint_cartesian_derivatives_match_independent_differences(core):
    from hexfellow_deploy.benchmark.trajectories import JointPathTrajectory
    trajectory = JointPathTrajectory(core.model, initial_state('extended'), 1.3, dual=True)
    eps = 1e-6
    for t in np.linspace(0, trajectory.duration_s, 97):
        transforms, twists, derivatives = trajectory.sample(t)
        next_transforms, next_twists, _ = trajectory.sample(t+eps)
        for side in transforms:
            finite = pin.log6(transforms[side].inverse()*next_transforms[side]).vector/eps
            np.testing.assert_allclose(twists[side], finite, atol=5e-5, rtol=1e-4)
            np.testing.assert_allclose(derivatives[side], (next_twists[side]-twists[side])/eps,
                                       atol=.002, rtol=1e-3)


def test_public_tcp_evaluation_velocity_matches_rigid_pose_derivative():
    from hexfellow_deploy.benchmark.public_tcp import transform_trace
    pose = pin.SE3(pin.exp3(np.array([.2, -.1, .3])), np.array([.6, -.2, .8]))
    offset = pin.SE3(pin.exp3(np.array([-.05, .1, .02])), np.array([0., 0., .017]))
    velocity, dt = np.array([.1, -.2, .3, .6, .1, -.3]), 1e-6
    trace = dict(requested=np.tile(pose.homogeneous, (1, 2, 1, 1)),
                 measured=np.tile(pose.homogeneous, (1, 2, 1, 1)),
                 tcp_velocity=np.tile(velocity, (1, 2, 1)))
    public = transform_trace(trace, {s: offset.homogeneous for s in ('left', 'right')})
    before, after = pose*offset.inverse(), pose*pin.exp6(velocity*dt)*offset.inverse()
    finite = pin.log6(before.inverse()*after).vector/dt
    np.testing.assert_allclose(public['tcp_velocity'][0, 1], finite, atol=1e-8)
    np.testing.assert_allclose(public['measured'][0, 1], before.homogeneous, atol=1e-12)


@pytest.mark.parametrize('speed,distance', [(0., 1.), (np.nan, 1.), (1., 0.), (1., np.inf)])
def test_invalid_trajectory_scaling_is_rejected(speed, distance):
    from hexfellow_deploy.benchmark.trajectories import ScrewTrajectory
    with pytest.raises(ValueError):
        Trajectory({}, 'vertical', speed, distance=distance)
    with pytest.raises(ValueError):
        ScrewTrajectory({}, {}, speed, distance=distance)


def test_deterministic_plant_clock_feedback_and_receipt(core, tmp_path):
    from hexfellow_deploy.simulation import SimulationPlant
    with pytest.raises(ValueError, match='zero submission delay'):
        SimulationPlant(None, tmp_path/'invalid', realtime=False, submit_delay_s=.01)
    profile = load_motion_profile('x7-gp80-highspeed-fixed-base-v1')
    q = initial_state()
    targets = poses(core.model, core.model.createData(), q)
    core.reset(q, targets, measured_velocity=np.zeros(15), state_timestamp=0.)
    result = core.step(q, targets, measured_velocity=np.zeros(15), state_timestamp=0., control_timestamp=0.,
                       feedback_received={s: 0. for s in ('left', 'right', 'theta', 'base')}, feedback_policy='strict')
    command = dict(joint_positions=dict(zip(JOINTS, result.positions)), base_velocity_body=[0., 0., 0.],
                   gripper_open=dict(left=None, right=None))
    scene = scene_for_run(tmp_path, q, .02)
    endpoints = []
    for index in range(2):
        plant = SimulationPlant(scene, tmp_path/str(index), profile=profile,
            effort_limits=profile.mit_servo.torque_limit_nm, realtime=False)
        try:
            assert plant.thread is None
            with pytest.raises(RuntimeError, match='Acquire'):
                plant.advance(.02)
            plant.acquire()
            plant.submit(1, command, result.mit)
            measured = plant.advance(.02)
            assert measured['timestamp_s'] == pytest.approx(.02)
            assert plant.acknowledgement(1).sequence == 1
            endpoints.append(plant.data.qpos.copy())
            with pytest.raises(ValueError, match='whole number'):
                plant.advance(.003)
        finally:
            plant.close()
    np.testing.assert_array_equal(*endpoints)
