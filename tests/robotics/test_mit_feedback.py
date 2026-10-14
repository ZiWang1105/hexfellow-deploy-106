"""Measured-state and fault contracts for the experimental MIT core."""

import dataclasses
import time

import numpy as np
import pinocchio as pin
import pytest

from hexfellow_deploy.robotics.control.ik_core import (
    TCP_FRAMES,
    ControlDeadlineExceeded,
    PinkControllerConfig,
    PinkIKCore,
)
from hexfellow_deploy.robotics.control.mit import GravityPayload, MitServoConfig
from hexfellow_deploy.robotics.control.motion_profile import controller_config_for_profile


def make_core():
    core = PinkIKCore(config=dataclasses.replace(
        PinkControllerConfig(), mit_servo=MitServoConfig(kp=(160.,) * 14, kd=(3.,) * 14)
    ))
    q = core.home_q.copy()
    core.configuration.update(q)
    targets = {
        side: core.configuration.get_transform_frame_to_world(frame).copy()
        for side, frame in TCP_FRAMES.items()
    }
    core.reset(q, targets, measured_velocity=np.zeros(core.model.nv), state_timestamp=0.)
    return core, q, targets


def make_v5_core():
    core = PinkIKCore(config=controller_config_for_profile("x7-mit-highspeed-v10"))
    q = core.home_q.copy()
    core.configuration.update(q)
    targets = {
        side: core.configuration.get_transform_frame_to_world(frame).copy()
        for side, frame in TCP_FRAMES.items()
    }
    core.reset(q, targets, measured_velocity=np.zeros(core.model.nv), state_timestamp=0.0)
    return core, q, targets


def test_v5_dispatch_deadline_rejects_expired_compute():
    core, _, _ = make_v5_core()

    with pytest.raises(ControlDeadlineExceeded, match="control_deadline_exceeded"):
        core._validate_dispatch_freshness(time.perf_counter() - 0.021, 0.0)

    assert core._deadline_reject_count == 1


def test_v5_late_dispatch_rolls_back_and_returns_non_executable_fault(monkeypatch):
    core, q, targets = make_v5_core()
    targets["left"].translation[0] += 0.01
    reference_state = tuple(
        value.copy()
        for value in (
            core._last_safe_positions,
            core._last_velocity,
            core._last_acceleration,
            core._last_jerk,
        )
    )
    calls = 0

    def reject_second_dispatch(started, feedback_age_s):
        nonlocal calls
        calls += 1
        if calls == 2:
            core._deadline_reject_count += 1
            raise ControlDeadlineExceeded("test late dispatch")

    monkeypatch.setattr(core, "_validate_dispatch_freshness", reject_second_dispatch)
    command = core.step(
        q,
        targets,
        measured_velocity=np.zeros(core.model.nv),
        state_timestamp=0.0,
    )

    assert calls == 2
    assert not command.executable
    assert command.mit is None
    assert core.control_state == "FAULT"
    assert "ControlDeadlineExceeded" in command.diagnostics.fault
    assert core._deadline_reject_count == 1
    for actual, expected in zip(
        (
            core._last_safe_positions,
            core._last_velocity,
            core._last_acceleration,
            core._last_jerk,
        ),
        reference_state,
        strict=True,
    ):
        np.testing.assert_array_equal(actual, expected)


def test_raw_measurement_is_not_replaced_by_planning_projection():
    core, q, targets = make_core()
    q[1] = core.model.lowerPositionLimit[1] - .001
    assert q[1] > core._physical_lower_position_limit[1]
    core.step(q, targets, measured_velocity=np.zeros(core.model.nv), state_timestamp=0.)
    assert core._measured_configuration.q[1] == q[1]
    assert core.reference_limit.measured_q[0] == q[1]


def test_expired_feedback_latches_fault_without_erasing_history():
    core, q, targets = make_core()
    core._last_velocity[1] = .1
    core._last_acceleration[1] = .2
    old_v = core._last_velocity.copy()
    old_a = core._last_acceleration.copy()
    command = core.step(q, targets, measured_velocity=np.zeros(core.model.nv),
                        state_timestamp=0., control_timestamp=.041)
    assert not command.executable
    assert command.mit is None
    assert core.control_state == "FAULT"
    np.testing.assert_array_equal(core._last_velocity, old_v)
    np.testing.assert_array_equal(core._last_acceleration, old_a)
    command = core.step(q, targets, measured_velocity=np.zeros(core.model.nv),
                        state_timestamp=.06, control_timestamp=.06)
    assert not command.executable
    core.reset(q, targets, measured_velocity=np.zeros(core.model.nv), state_timestamp=.06)
    assert core._mit_fault_latched is None


def test_finite_overspeed_is_observed_then_explicitly_faulted():
    core, q, targets = make_core()
    velocity = np.zeros(core.model.nv)
    velocity[1] = core.model.velocityLimit[1] + 1.
    np.testing.assert_array_equal(core._validate_velocity_vector(velocity), velocity)
    command = core.step(q, targets, measured_velocity=velocity, state_timestamp=0.)
    assert not command.executable
    assert "overspeed" in command.diagnostics.fault
    np.testing.assert_array_equal(core._last_measured_velocity, velocity)


def test_stop_uses_continuous_reference_history():
    # Kinematic unit replay, not a physical tracking benchmark.
    core, q, targets = make_core()
    targets["left"].translation[0] += .02
    velocity = np.zeros(core.model.nv)
    previous_q = q.copy()
    previous_v = velocity.copy()
    previous_a = velocity.copy()
    for i in range(80):
        if i == 12:
            core.request_stop()
        result = core.step(q, targets, measured_velocity=velocity,
                           state_timestamp=i * .02)
        assert result.executable, result.diagnostics.fault
        np.testing.assert_allclose(
            (result.positions[1:] - previous_q[1:]) / .02,
            result.velocity[1:], atol=1e-9,
        )
        a = (result.velocity - previous_v) / .02
        j = (a - previous_a) / .02
        assert np.all(np.abs(a) <= core._acceleration_limits + .011)
        assert np.all(np.abs(j) <= core._jerk_limits + .011)
        previous_q, previous_v, previous_a = result.positions.copy(), result.velocity.copy(), a
        q, velocity = result.positions.copy(), result.velocity.copy()
    assert core.control_state == "HOLD"
    core.resume_tracking()
    assert core.control_state == "TRACKING"


def test_hold_requires_two_hundred_ms_of_continuous_rest():
    core, q, targets = make_core()
    core.request_stop()
    for i in range(10):
        result = core.step(q, targets, measured_velocity=np.zeros(core.model.nv),
                           state_timestamp=i * .02)
        assert result.executable
        assert core.control_state == "BRAKING"
    result = core.step(q, targets, measured_velocity=np.zeros(core.model.nv), state_timestamp=.2)
    assert result.executable
    assert core.control_state == "HOLD"
    assert result.diagnostics.stop_reason == "requested"


@pytest.mark.parametrize("scale", [1.05, 1.2])
def test_infeasible_overspeed_braking_faults_without_dispatch(scale):
    from hexfellow_deploy.robotics.control.motion_profile import controller_config_for_profile
    core = PinkIKCore(config=controller_config_for_profile("x7-mit-highspeed-v10"))
    q = core.home_q.copy()
    targets = {side: core.configuration.get_transform_frame_to_world(frame).copy()
               for side, frame in TCP_FRAMES.items()}
    core.reset(q, targets, measured_velocity=np.zeros(core.model.nv), state_timestamp=0.)
    velocity = np.zeros(core.model.nv)
    velocity[1] = scale * core.model.velocityLimit[1]
    command = core.step(q, targets, measured_velocity=velocity, state_timestamp=0.)
    # The current effort/jerk constraints cannot brake this discontinuous
    # measurement safely. Preserve the reference and reject dispatch.
    assert not command.executable
    assert command.mit is None
    assert core.control_state == "FAULT"
    assert "NoSolutionFound" in command.diagnostics.fault
    np.testing.assert_array_equal(core._last_safe_positions, q)
    np.testing.assert_array_equal(core._last_measured_velocity, velocity)


def test_cold_reset_cannot_seed_an_overspeed_reference():
    core, q, targets = make_core()
    velocity = np.zeros(core.model.nv)
    velocity[1] = 1.1 * core.model.velocityLimit[1]
    before = core._last_velocity.copy()
    with pytest.raises(ValueError, match="initialize"):
        core.reset(q, targets, measured_velocity=velocity)
    np.testing.assert_array_equal(core._last_velocity, before)


def test_requested_brake_does_not_replan_a_cartesian_target(monkeypatch):
    core, q, targets = make_core()
    def forbidden(*args, **kwargs):
        raise AssertionError("a stop must not retarget Cartesian motion")
    for planner in core._target_planners.values():
        monkeypatch.setattr(planner, "step", forbidden)
    core.request_stop()
    command = core.step(q, targets, measured_velocity=np.zeros(core.model.nv), state_timestamp=0.)
    assert command.executable, command.diagnostics.fault


def test_infeasible_cartesian_segment_attempts_a_constrained_stop(monkeypatch):
    from hexfellow_deploy.robotics.control.motion_profile import controller_config_for_profile
    from hexfellow_deploy.robotics.control.target_motion import TargetTrajectoryError
    core = PinkIKCore(config=controller_config_for_profile("x7-mit-highspeed-v10"))
    q = core.home_q.copy()
    targets = {side: core.configuration.get_transform_frame_to_world(frame).copy()
               for side, frame in TCP_FRAMES.items()}
    core.reset(q, targets, measured_velocity=np.zeros(core.model.nv), state_timestamp=0.)
    def infeasible(*args, **kwargs):
        raise TargetTrajectoryError("test: valid pose, infeasible segment")
    monkeypatch.setattr(core._target_planners["left"], "step", infeasible)
    command = core.step(q, targets, measured_velocity=np.zeros(core.model.nv), state_timestamp=0.)
    assert command.executable, command.diagnostics.fault
    assert command.diagnostics.recovery_reason == "target_trajectory_infeasible"
    assert core.control_state == "TRACKING"  # Recovery is transient, not a latched stop.


def test_command_timestamp_is_control_time_not_feedback_time():
    core, q, targets = make_core()
    command = core.step(q, targets, measured_velocity=np.zeros(core.model.nv),
                        state_timestamp=0., control_timestamp=.02)
    assert command.executable, command.diagnostics.fault
    assert command.mit.timestamp_s == .02


def test_gravity_uses_gripper_feedback_and_declared_payload():
    core, q, _ = make_core()
    gravity = core.gravity_model
    closed = gravity.evaluate(q, np.zeros(12))
    opened = gravity.evaluate(q, np.full(12, .3))
    assert np.max(np.abs(closed - opened)) > 1e-5
    # Independent full-model evaluation checks controlled joint selection.
    full_q = gravity.nominal_q.copy()
    full_q[gravity.full_q] = q[gravity.control_q]
    full_q[gravity.auxiliary_q] = .3
    expected = pin.computeGeneralizedGravity(gravity.model, gravity.model.createData(), full_q)
    np.testing.assert_allclose(opened, expected[gravity.full_v], atol=1e-12)
    loaded = PinkIKCore(config=dataclasses.replace(
        core.config, gravity_payloads=(GravityPayload("left_tcp", .2, (.01, 0., 0.)),)
    ))
    effort = loaded.gravity_model.evaluate(q, np.zeros(12))
    assert np.max(np.abs(effort - closed)) > .01
    np.testing.assert_allclose(effort[8:], closed[8:], atol=1e-12)


@pytest.mark.parametrize("mass,com", [(-1., (0., 0., 0.)), (float("nan"), (0., 0., 0.)),
                                    (1., (0., float("inf"), 0.))])
def test_invalid_declared_payload(mass, com):
    with pytest.raises(ValueError):
        GravityPayload("left_tcp", mass, com)


@pytest.mark.parametrize("physics_hz", [500, 600])
def test_integration_profile_motion_then_physical_stop(tmp_path, physics_hz):
    import mujoco

    from hexfellow_deploy.robotics.control.ik import MaverX4PinkController, snap_targets_to_tcp
    from hexfellow_deploy.robotics.control.motion_profile import controller_config_for_profile
    from hexfellow_deploy.robotics.scene.builder import build_tabletop_scene

    scene = build_tabletop_scene(
        tmp_path / "scene.xml", preserve_fixed_inertias=True, base_mode="fixed",
        object_ids=("apple",), layout_seed=0, include_box=False, include_trashcan=False,
        dynamic_table=False, timestep=1. / physics_hz,
    )
    model = mujoco.MjModel.from_xml_path(str(scene))
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, model.key("home").id)
    mujoco.mj_forward(model, data)
    core = MaverX4PinkController(
        model, config=controller_config_for_profile("x7-mit-highspeed-v10")
    )
    targets = snap_targets_to_tcp(model, data)
    core.reset(core.measured_configuration(data), targets,
               measured_velocity=core.measured_velocity(data),
               measured_gripper_positions=core.measured_gripper_positions(data),
               state_timestamp=0.)
    assert core.self_collision_system.collision_envelope.requires_dynamic_travel
    for i in range(4 * physics_hz):
        if i == physics_hz:
            targets["left"].translation[0] += .02
        if i == 2 * physics_hz:
            core.request_stop()
        if i % core.control_period_steps == 0:
            command = core.step(
                core.measured_configuration(data), targets,
                measured_velocity=core.measured_velocity(data),
                measured_gripper_positions=core.measured_gripper_positions(data),
                state_timestamp=float(data.time),
            )
            assert command.executable, command.diagnostics.fault
            core.apply_command(data, command)
        mujoco.mj_step(model, data)
    assert core.control_state == "HOLD"
    assert max(core._task_errors()[:2]) < .002
    assert max(core._task_errors()[2:]) < np.deg2rad(1.)
    before = data.ctrl.copy()
    invalid = dataclasses.replace(command, positions=np.full(core.model.nq, np.nan))
    with pytest.raises(ValueError):
        core.apply_command(data, invalid)
    np.testing.assert_array_equal(data.ctrl, before)


def test_repeated_feedback_brakes_then_expires():
    core, q, targets = make_core()
    velocity = np.zeros(core.model.nv)
    first = core.step(q, targets, measured_velocity=velocity, state_timestamp=0.)
    assert first.executable
    repeated = core.step(q, targets, measured_velocity=velocity,
                         state_timestamp=0., control_timestamp=.02)
    assert repeated.executable, repeated.diagnostics.fault
    assert core.control_state in {"BRAKING", "HOLD"}
    expired = core.step(q, targets, measured_velocity=velocity,
                        state_timestamp=0., control_timestamp=.06)
    assert not expired.executable
    assert expired.diagnostics.timings.total_ms > 0.
