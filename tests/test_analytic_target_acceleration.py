"""Analytic trajectory acceleration must retain its physical frame convention."""
import qpsolvers  # Preserve native shared-library initialization order.
import numpy as np
import pinocchio as pin
import pytest

from hexfellow_deploy.robotics.control.target_motion import (
    CartesianMotionLimits, ContinuousTargetMotionPlanner,
)


@pytest.mark.parametrize('gain', [0., 1.])
def test_analytic_acceleration_preserves_world_linear_acceleration(gain):
    planner = ContinuousTargetMotionPlanner(
        CartesianMotionLimits(2., 8., 120., 3., 14., 160.),
        continuous_tracking=True, source_acceleration_gain=gain)
    pose = pin.SE3(pin.exp3(np.array([.2, -.3, .1])), np.zeros(3))
    twist = np.array([.2, -.1, .05, .1, .2, -.1])
    acceleration = np.array([.3, .1, -.2, .02, -.03, .01])
    planner.reset(pose, twist, acceleration)
    dt = .01
    actual, _ = planner.step(pose, dt, desired_body_twist=twist,
                             desired_body_acceleration=acceleration)
    world_a = pose.rotation @ (acceleration[:3]+np.cross(twist[3:], twist[:3]))
    # With matching source/state, source feedforward cancels the acceleration
    # feedback term. With gain zero the established v4 damping law is retained.
    jerk = -3*planner.TRACKING_BANDWIDTH_S*(1-gain)*world_a
    expected = pose.translation + pose.rotation@twist[:3]*dt + .5*world_a*dt**2 + jerk*dt**3/6
    np.testing.assert_allclose(actual.pose.translation, expected, atol=1e-11)
    assert planner._has_explicit_acceleration is False


def test_invalid_acceleration_does_not_leak_into_next_pose_only_step():
    planner = ContinuousTargetMotionPlanner(CartesianMotionLimits(2., 8., 120., 3., 14., 160.),
        continuous_tracking=True, source_acceleration_gain=1.)
    planner.reset(pin.SE3.Identity())
    with pytest.raises(ValueError):
        planner.step(pin.SE3.Identity(), .01, desired_body_acceleration=np.full(6, np.nan))
    assert planner._has_explicit_acceleration is False
    motion, _ = planner.step(pin.SE3.Identity(), .01)
    np.testing.assert_allclose(motion.pose.homogeneous, np.eye(4), atol=1e-12)
