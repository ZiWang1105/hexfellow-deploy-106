"""Check trajectory derivatives in the measured-pose objective independently."""
from types import SimpleNamespace

import qpsolvers
import numpy as np
import pinocchio as pin
import pytest

from hexfellow_deploy.robotics.control.measured_tracking import MeasuredPoseTracking


def setup_tracker():
    config = SimpleNamespace(frame_gain=.2, target_translation_deadband=1e-5,
        target_orientation_deadband=1e-5, target_linear_velocity_limit=2.,
        target_linear_acceleration_limit=8., target_linear_jerk_limit=120.,
        target_angular_velocity_limit=3., target_angular_acceleration_limit=14.,
        target_angular_jerk_limit=160., target_tracking_bandwidth_s_inv=12., measured_error_derivative_gain=0.)
    pose = pin.SE3(pin.exp3(np.array([.2, -.1, .3])), np.zeros(3))
    configuration = SimpleNamespace(model=SimpleNamespace(nv=6),
        get_transform_frame_to_world=lambda frame: pose,
        get_frame_jacobian=lambda frame: np.eye(6))
    task = SimpleNamespace(frame='tcp', cost=np.ones(6), lm_damping=0.)
    return MeasuredPoseTracking(config), configuration, task, pose


def test_zero_pose_error_does_not_discard_moving_target_derivatives():
    tracker, configuration, task, pose = setup_tracker()
    twist = np.array([.01, -.02, .005, .01, .005, -.01])
    acceleration = np.array([.01, .02, -.01, .005, -.004, .002])
    dt = .01
    tracker.objective('right', task, configuration, pose, dt,
                      target_twist=twist, target_acceleration=acceleration)
    velocity = np.r_[pose.rotation@twist[:3], pose.rotation@twist[3:]]
    world_acceleration = np.r_[pose.rotation@(acceleration[:3]+np.cross(twist[3:],twist[:3])),
                              pose.rotation@acceleration[3:]]
    expected_jerk = 12.**2*velocity+2*12.*world_acceleration
    np.testing.assert_allclose(tracker.observations['right']['jerk_world'], expected_jerk, atol=1e-12)
    np.testing.assert_allclose(tracker.velocity['right'], .5*dt**2*expected_jerk, atol=1e-12)
    legacy, configuration, task, pose = setup_tracker()
    legacy.objective('right', task, configuration, pose, dt)
    np.testing.assert_array_equal(legacy.velocity['right'], np.zeros(6))


def test_invalid_derivatives_do_not_advance_history():
    tracker, configuration, task, pose = setup_tracker()
    for kwargs in (dict(target_acceleration=np.zeros(6)),
                   dict(target_twist=np.full(6,np.nan)),
                   dict(target_twist=np.zeros(6),target_acceleration=np.zeros(5))):
        with pytest.raises(ValueError):
            tracker.objective('right', task, configuration, pose, .01, **kwargs)
        assert not tracker.velocity and not tracker.acceleration


def test_error_rate_matches_independent_pose_difference_and_damps_motion():
    tracker, configuration, task, current = setup_tracker()
    tracker.config.measured_error_derivative_gain = 1.
    target = pin.SE3(pin.exp3(np.array([.1, .2, -.2])) @ current.rotation, np.array([.002, -.001, .001]))
    joint_velocity = np.array([.01, -.005, .002, .005, -.003, .002])
    target_twist = np.array([.003, .002, -.001, -.001, .002, .003])
    dt = .01
    tracker.objective('right', task, configuration, target, dt,
                      target_twist=target_twist, target_acceleration=np.zeros(6),
                      measured_joint_velocity=joint_velocity)
    # Independent central difference of SE(3) trajectories. The fake local
    # Jacobian is identity, so joint_velocity is the measured body twist.
    epsilon = 1e-6
    errors = []
    for sign in [-1, 1]:
        c = current * pin.exp6(sign * epsilon * joint_velocity)
        t = target * pin.exp6(sign * epsilon * target_twist)
        errors.append(np.r_[t.translation-c.translation, pin.log3(t.rotation @ c.rotation.T)])
    numerical = (errors[1]-errors[0]) / (2 * epsilon)
    np.testing.assert_allclose(tracker.observations['right']['error_rate_world'], numerical, atol=1e-9)


def test_derivative_feedback_requires_measured_velocity_without_advancing_history():
    tracker, configuration, task, pose = setup_tracker()
    tracker.config.measured_error_derivative_gain = 1.
    for velocity in [None, np.zeros(5), np.full(6, np.nan)]:
        with pytest.raises(ValueError, match='measured joint velocity'):
            tracker.objective('right', task, configuration, pose, .01,
                              measured_joint_velocity=velocity)
        assert not tracker.velocity and not tracker.acceleration
