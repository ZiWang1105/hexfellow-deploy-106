"""Offline MIT primitives and nominal asset parity; no hardware dependencies."""

import dataclasses
import os
import subprocess
import sys

import numpy as np
import pinocchio as pin
import pytest

from hexfellow_deploy.robotics.control.command_limits import stoppable_speed, stopping_distance
from hexfellow_deploy.robotics.control.mit import MitCommand, MitServoConfig, ModelGravity, mit_torque
from hexfellow_deploy.robotics.control.robot import DEFAULT_URDF, load_nominal_mujoco_model


def command():
    return MitCommand(("j1", "j2"), [1, -1], [2, -2], [3, -3],
                      [10, 10], [2, 2], 0.0)


def test_command_is_snapshot_and_arm_selection_is_ordered():
    source = np.array([1., -1.])
    packet = dataclasses.replace(command(), position=source)
    source[:] = 100
    np.testing.assert_array_equal(packet.position, [1, -1])
    with pytest.raises(ValueError):
        packet.position[0] = 2
    selected = packet.select(("j2", "j1"))
    np.testing.assert_array_equal(selected.position, [-1, 1])
    assert not np.shares_memory(selected.position, packet.position)


@pytest.mark.parametrize("field,value", [
    ("position", [float("nan"), 0]), ("velocity", [0]),
    ("kp", [-1, 1]), ("timestamp_s", float("inf")),
    ("joint_names", ("j1", "j1")), ("joint_names", (1, 2)),
])
def test_invalid_packets(field, value):
    with pytest.raises(ValueError):
        dataclasses.replace(command(), **{field: value})


def test_total_pd_and_feedforward_are_saturated_together():
    clipped, requested = mit_torque(command(), [0, 0], [0, 0], [5, 6])
    np.testing.assert_array_equal(requested, [17, -17])
    np.testing.assert_array_equal(clipped, [5, -6])


@pytest.mark.parametrize("field,value", [
    ("kp", (0.,) * 14), ("kd", (-1.,) * 14),
    ("torque_limit_nm", (float("inf"),) * 14),
])
def test_invalid_servo(field, value):
    with pytest.raises(ValueError):
        MitServoConfig(**{field: value})


def test_jerk_braking_matches_numerical_integration():
    # Integrate the stated braking law independently, including acceleration
    # initially pointing along travel. This is a reference, not plant proof.
    for velocity in (0.1, 1.0, 2.8, 9.4):
        for acceleration in (-18., 0., 18.):
            dt = 1e-5
            v, a, distance = velocity, acceleration, 0.
            for _ in range(200000):
                next_a = max(-18., a - 600. * dt)
                next_v = v + (a + next_a) * 0.5 * dt
                distance += max(0., (v + max(0., next_v)) * 0.5 * dt)
                v, a = next_v, next_a
                if v <= 0:
                    break
            exact = stopping_distance(velocity, acceleration, 18., 600.)
            assert distance == pytest.approx(float(exact), abs=2e-7)
            assert stopping_distance(velocity, acceleration, 18., 600., .02) >= exact


def test_stoppable_speed_respects_distance():
    distance = np.array([.001, .02, .1, 10.])
    speed = stoppable_speed(distance, 0., np.full(4, 2.8), 24., 600., .02)
    feasible = stopping_distance(0., 0., 24., 600., .02) <= distance
    assert np.all(stopping_distance(speed[feasible], 0., 24., 600., .02)
                  <= distance[feasible] + 1e-10)
    assert speed[-1] == 2.8


def test_nominal_asset_gravity_matches_independent_engine():
    import mujoco

    model = load_nominal_mujoco_model()
    data = mujoco.MjData(model)
    nominal = pin.buildModelFromUrdf(str(DEFAULT_URDF))
    gravity = ModelGravity(nominal)
    rng = np.random.default_rng(7)
    for _ in range(5):
        q = rng.uniform(-.15, .15, nominal.nq)
        for joint_id in range(1, nominal.njoints):
            joint = nominal.joints[joint_id]
            data.qpos[model.joint(nominal.names[joint_id]).qposadr[0]] = q[joint.idx_q]
        data.qvel[:] = 0
        mujoco.mj_forward(model, data)
        actual = np.array([
            data.qfrc_bias[model.joint(nominal.names[j]).dofadr[0]]
            for j in range(1, nominal.njoints)
        ])
        np.testing.assert_allclose(gravity.evaluate(q), actual, atol=1e-10)
    # Engine bias is accessed ONLY here as a validation oracle.


def test_portable_core_import_and_creation_without_mujoco():
    code = """
import sys
class RejectSimulator:
    def find_spec(self, fullname, path=None, target=None):
        if fullname == 'mujoco' or fullname.startswith('mujoco.'):
            raise ImportError('simulator forbidden in portable control')
sys.meta_path.insert(0, RejectSimulator())
from hexfellow_deploy.robotics.control.ik_core import PinkIKCore
from hexfellow_deploy.robotics.control.mit import ModelGravity
core = PinkIKCore()
assert ModelGravity(core.model).evaluate(core.configuration.q).shape == (core.model.nv,)
assert 'mujoco' not in sys.modules
"""
    subprocess.run([sys.executable, "-c", code], check=True, env=os.environ.copy())

@pytest.mark.parametrize("physics_hz", [500, 600])
def test_mit_physical_hold_and_reference_contract(tmp_path, physics_hz):
    import mujoco

    from hexfellow_deploy.robotics.control.ik import (
        MaverX4PinkController,
        PinkControllerConfig,
        snap_targets_to_tcp,
    )
    from hexfellow_deploy.robotics.control.robot import build_scene
    from hexfellow_deploy.robotics.scene.builder import build_tabletop_scene

    robot = build_scene(output_path=tmp_path / "robot.xml", base_mode="fixed",
                        preserve_fixed_inertias=True)
    scene = build_tabletop_scene(
        tmp_path / "scene.xml", robot_scene_path=robot, base_mode="fixed",
        object_ids=("apple",), layout_seed=0, dynamic_table=False,
        include_box=False, include_trashcan=False, timestep=1. / physics_hz,
    )
    model = mujoco.MjModel.from_xml_path(str(scene))
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, model.key("home").id)
    mujoco.mj_forward(model, data)
    physical = {name: getattr(model, name).copy()
                for name in ("body_mass", "body_inertia", "body_ipos", "dof_damping",
                             "dof_frictionloss", "jnt_actfrcrange")}
    config = dataclasses.replace(
        PinkControllerConfig(), mit_servo=MitServoConfig(kp=(160.,) * 14, kd=(3.,) * 14)
    )
    controller = MaverX4PinkController(model, config=config)
    for name, original in physical.items():
        np.testing.assert_array_equal(getattr(model, name), original)
    targets = snap_targets_to_tcp(model, data)
    q = controller.measured_configuration(data)
    controller.reset(q, targets, measured_velocity=controller.measured_velocity(data),
                     state_timestamp=0.)
    previous_q = q[1:].copy()
    previous_v = np.zeros(14)
    previous_a = np.zeros(14)
    for i in range(2 * physics_hz):
        if i % controller.control_period_steps == 0:
            result = controller.step(
                controller.measured_configuration(data), targets,
                measured_velocity=controller.measured_velocity(data),
                state_timestamp=float(data.time),
            )
            assert result.diagnostics.fault is None
            packet = result.mit
            assert packet is not None
            finite_difference = (packet.position - previous_q) / config.control_dt
            np.testing.assert_allclose(finite_difference, packet.velocity, atol=1e-10)
            acceleration = (packet.velocity - previous_v) / config.control_dt
            jerk = (acceleration - previous_a) / config.control_dt
            assert np.all(np.abs(packet.velocity) <= controller.model.velocityLimit[1:] + 1e-6)
            assert np.all(np.abs(acceleration) <= controller._acceleration_limits[1:] + 1e-6)
            assert np.all(np.abs(jerk) <= controller._jerk_limits[1:] + 1e-6)
            previous_q, previous_v, previous_a = packet.position, packet.velocity, acceleration
            actual_q = data.qpos.copy()
            controller.apply_command(data, result)
            np.testing.assert_array_equal(data.qpos, actual_q)
            assert not np.any(data.qfrc_applied)
        mujoco.mj_step(model, data)
        for actuator in controller._mj_upper_actuator_ids[1:]:
            assert abs(data.actuator_force[actuator]) <= 100. + 1e-8
    assert max(controller._task_errors()[:2]) < 2e-5
    assert not controller.actuation_model_assumptions["body_gravity_compensation"]
