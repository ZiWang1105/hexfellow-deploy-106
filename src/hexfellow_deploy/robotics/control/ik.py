# ruff: noqa: I001 -- load the core's qpsolvers before Pinocchio shared libraries.
"""MuJoCo adapter and backwards-compatible exports for the portable IK core."""

from __future__ import annotations

from pathlib import Path
import json

import mujoco
import numpy as np
from .robot_model import load_robot_model

from .ik_core import (
    ARM_JOINTS as ARM_JOINTS,
    TCP_FRAMES as TCP_FRAMES,
    TARGET_BODIES as TARGET_BODIES,
    BASE_ACTUATORS as BASE_ACTUATORS,
    BASE_JOINTS as BASE_JOINTS,
    CONTROLLED_JOINTS as CONTROLLED_JOINTS,
    DEFAULT_URDF as DEFAULT_URDF,
    LIFT_JOINT as LIFT_JOINT,
    NullPose as NullPose,
    PinkIKCore as PinkIKCore,
    PinkControllerConfig as PinkControllerConfig,
    IKCommand as IKCommand,
    IKDiagnostics as IKDiagnostics,
    IKPhaseTimings as IKPhaseTimings,
    PlanarBaseCommand as PlanarBaseCommand,
    TransformError as TransformError,
    TrackingErrorChain as TrackingErrorChain,
    MotionUtilization as MotionUtilization,
    FeedforwardFrameTask as FeedforwardFrameTask,
    JerkLimit as JerkLimit,
    qpsolvers as qpsolvers,
)
import pinocchio as pin

from .gripper import WIRE_MAX_WIDTH_M, aperture_to_angle


def _object_id(model: mujoco.MjModel, object_type: mujoco.mjtObj, name: str) -> int:
    object_id = mujoco.mj_name2id(model, object_type, name)
    if object_id < 0:
        raise RuntimeError(f"MuJoCo model is missing {name!r}")
    return object_id


def _rotation_from_quaternion(quaternion: np.ndarray) -> np.ndarray:
    rotation = np.empty(9)
    mujoco.mju_quat2Mat(rotation, quaternion)
    return rotation.reshape(3, 3)


def _quaternion_from_rotation(rotation: np.ndarray) -> np.ndarray:
    quaternion = np.empty(4)
    mujoco.mju_mat2Quat(quaternion, np.asarray(rotation).reshape(9))
    return quaternion


def site_pose(model: mujoco.MjModel, data: mujoco.MjData, site_name: str) -> pin.SE3:
    """Read a named MuJoCo site as a world-frame Pinocchio transform."""
    site_id = _object_id(model, mujoco.mjtObj.mjOBJ_SITE, site_name)
    return pin.SE3(data.site_xmat[site_id].reshape(3, 3).copy(), data.site_xpos[site_id].copy())


def mocap_pose(model: mujoco.MjModel, data: mujoco.MjData, body_name: str) -> pin.SE3:
    """Read a mocap body's target transform in the world frame."""
    body_id = _object_id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
    mocap_id = int(model.body_mocapid[body_id])
    if mocap_id < 0:
        raise RuntimeError(f"MuJoCo body {body_name!r} is not a mocap body")
    return pin.SE3(_rotation_from_quaternion(data.mocap_quat[mocap_id]), data.mocap_pos[mocap_id].copy())


def set_mocap_pose(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    body_name: str,
    pose: pin.SE3,
) -> None:
    """Write a world-frame target transform to a MuJoCo mocap body."""
    body_id = _object_id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
    mocap_id = int(model.body_mocapid[body_id])
    if mocap_id < 0:
        raise RuntimeError(f"MuJoCo body {body_name!r} is not a mocap body")
    data.mocap_pos[mocap_id] = pose.translation
    data.mocap_quat[mocap_id] = _quaternion_from_rotation(pose.rotation)


def snap_targets_to_tcp(model: mujoco.MjModel, data: mujoco.MjData) -> dict[str, pin.SE3]:
    """Place both mocap targets on the robot's current TCP poses."""
    targets: dict[str, pin.SE3] = {}
    for side, site_name in TCP_FRAMES.items():
        pose = site_pose(model, data, site_name)
        set_mocap_pose(model, data, TARGET_BODIES[side], pose)
        targets[side] = pose
    return targets


def read_mocap_targets(model: mujoco.MjModel, data: mujoco.MjData) -> dict[str, pin.SE3]:
    return {side: mocap_pose(model, data, body) for side, body in TARGET_BODIES.items()}


def apply_gripper_apertures(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    left: float,
    right: float,
) -> None:
    """Command symmetric left and right finger apertures in meters."""
    for side, aperture, joint_names in (
        ("left", left, tuple(f"robot_joint{i}" for i in range(1, 7))),
        ("right", right, tuple(f"robot2_joint{i}" for i in range(1, 7))),
    ):
        if not np.isfinite(aperture) or not 0.0 <= aperture <= WIRE_MAX_WIDTH_M:
            raise ValueError(f"{side} gripper aperture must be in [0, {WIRE_MAX_WIDTH_M:g}] m")
        actuator_ids = [
            _object_id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, f"position_{joint_name}") for joint_name in joint_names
        ]
        angle = aperture_to_angle(aperture)
        data.ctrl[actuator_ids] = (angle, -angle, -angle, angle, angle, -angle)


class MaverX4PinkController(PinkIKCore):
    """Map dual world-frame TCP targets to safe whole-body commands."""

    def __init__(
        self,
        mj_model: mujoco.MjModel,
        urdf_path: Path | None = None,
        config: PinkControllerConfig | None = None,
        nominal_pose: NullPose | None = None,
    ) -> None:
        self.mj_model = mj_model
        config = config or PinkControllerConfig()
        self.config = config
        self.auxiliary_joints = load_robot_model(config.robot_model).auxiliary_joints
        if config.solver not in qpsolvers.available_solvers:
            raise RuntimeError(
                f"Pink QP solver {config.solver!r} is unavailable; installed solvers: "
                f"{', '.join(qpsolvers.available_solvers) or 'none'}"
            )
        ratio = config.control_dt / float(mj_model.opt.timestep)
        self.control_period_steps = round(ratio)
        if self.control_period_steps < 1 or not np.isclose(ratio, self.control_period_steps, atol=1e-9):
            raise RuntimeError(
                f"IK period {config.control_dt:g}s must be an integer multiple of the "
                f"MuJoCo timestep {mj_model.opt.timestep:g}s"
            )

        base_present = [mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_JOINT, name) >= 0 for name in BASE_JOINTS]
        if any(base_present) and not all(base_present):
            raise RuntimeError("MuJoCo model contains only part of the planar-base joint triplet")
        self.base_mode = "planar" if all(base_present) else "fixed"
        self._base_dofs = 3 if self.base_mode == "planar" else 0
        self._mj_qpos_indices: list[int] = []
        self._mj_qvel_indices: list[int] = []
        self._mj_base_actuator_ids: list[int] = []
        self._mj_upper_actuator_ids: list[int] = []
        self._mj_gripper_qpos_indices: list[int] = []
        self._validate_mujoco_model()
        self._apply_simulation_servo_tuning()
        super().__init__(
            urdf_path=urdf_path, config=config, nominal_pose=nominal_pose,
            base_mode=self.base_mode,
        )
        if config.mit_servo is not None:
            self._configure_mit_actuators()

    def _configure_mit_actuators(self) -> None:
        """Model an MIT motor's local PD, with finite *total* output effort.

        MuJoCo only evaluates the motor law at physics substeps. References
        and model gravity are sampled by the portable controller, not here.
        Non-arm lift support represents the existing position servo's load
        balance and is explicitly charged against its joint force budget.
        """
        model = self.mj_model
        if self.config.gravity_urdf_sha256 is not None:
            text_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_TEXT, "mit_asset_contract")
            if text_id < 0:
                raise ValueError("MIT scene has no nominal asset contract; rebuild the scene")
            address = int(model.text_adr[text_id])
            size = int(model.text_size[text_id])
            contract = json.loads(model.text_data[address:address + size].rstrip(b"\x00"))
            if (contract.get("urdf_sha256") != self.config.gravity_urdf_sha256
                    or contract.get("conversion") != "unfused_urdf_v1"
                    or contract.get("mujoco_version") != mujoco.__version__):
                raise ValueError("MIT scene asset contract is stale; rebuild the scene")
            if any(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name) < 0
                   for name in ("robot_base", "robot2_base")):
                raise ValueError("MIT profile requires a fresh unfused robot scene; do not reuse a legacy cache")
        servo = self.effective_mit_servo
        arm_actuators = self._mj_upper_actuator_ids[1:]
        lift_joint = int(model.actuator_trnid[self._mj_upper_actuator_ids[0], 0])
        if not model.jnt_actfrclimited[lift_joint]:
            raise ValueError("lift position-servo support requires an existing force budget")
        # Validate every actuator before changing any actuator's command units.
        for index, actuator in enumerate(arm_actuators):
            joint = int(model.actuator_trnid[actuator, 0])
            if not model.jnt_actfrclimited[joint]:
                raise ValueError("MIT simulation requires a finite existing joint force bound")
            inherited = min(abs(model.jnt_actfrcrange[joint]))
            if servo.torque_limit_nm[index] > inherited + 1e-9:
                raise ValueError("MIT torque bound cannot increase the existing actuator force budget")
        for index, actuator in enumerate(arm_actuators):
            # ctrl = kp*q_des + kd*v_des + tau_ff. The affine bias supplies
            # -kp*q - kd*v inside the finite actuator, not as an external force.
            model.actuator_gainprm[actuator, :] = 0.0
            model.actuator_gainprm[actuator, 0] = 1.0
            model.actuator_biasprm[actuator, :] = 0.0
            model.actuator_biasprm[actuator, 1:3] = (-servo.kp[index], -servo.kd[index])
            model.actuator_ctrllimited[actuator] = False
            model.actuator_forcelimited[actuator] = True
            bound = servo.torque_limit_nm[index]
            model.actuator_forcerange[actuator] = (-bound, bound)
        lift_body = int(model.jnt_bodyid[lift_joint])
        supported_bodies = {lift_body}
        for body in range(lift_body, model.nbody):
            if int(model.body_parentid[body]) in supported_bodies:
                supported_bodies.add(body)
        model.body_gravcomp[list(supported_bodies)] = 0.0
        self.actuation_model_assumptions = {
            "arms": "sampled MIT references; local joint PD; nominal-model gravity feedforward",
            "lift": "position servo with bounded nominal-model gravity load support",
            "lift_support_is_hardware_command": False,
            "force_bounds": ("manufacturer peak joint bounds; MIT continuous operating budget; "
                             "firmware clamp unverified"),
            "body_gravity_compensation": False,
        }

    def _validate_mujoco_model(self) -> None:
        gravity = np.asarray(self.mj_model.opt.gravity)
        if not np.allclose(gravity, (0.0, 0.0, -9.81), atol=1e-8):
            raise RuntimeError(f"Pink mode requires MuJoCo gravity [0, 0, -9.81], found {gravity}")
        if self.base_mode == "planar":
            expected_ranges = (
                self.config.base.x_limits_m,
                self.config.base.y_limits_m,
                self.config.base.yaw_limits_rad,
            )
            for joint_name, actuator_name, expected_range in zip(
                BASE_JOINTS,
                BASE_ACTUATORS,
                expected_ranges,
                strict=True,
            ):
                joint_id = _object_id(self.mj_model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
                if self.mj_model.jnt_type[joint_id] not in (
                    mujoco.mjtJoint.mjJNT_HINGE,
                    mujoco.mjtJoint.mjJNT_SLIDE,
                ):
                    raise RuntimeError(f"Planar-base joint {joint_name!r} is not one-DoF")
                bounded = self.config.base.workspace_mode == "bounded"
                if bool(self.mj_model.jnt_limited[joint_id]) != bounded:
                    raise RuntimeError("Planar-base position-limit mode does not match controller config")
                if bounded and not np.allclose(self.mj_model.jnt_range[joint_id], expected_range, atol=1e-8):
                    raise RuntimeError(f"Planar-base joint {joint_name!r} range does not match controller config")
                self._mj_qpos_indices.append(int(self.mj_model.jnt_qposadr[joint_id]))
                self._mj_qvel_indices.append(int(self.mj_model.jnt_dofadr[joint_id]))
                actuator_id = _object_id(self.mj_model, mujoco.mjtObj.mjOBJ_ACTUATOR, actuator_name)
                if int(self.mj_model.actuator_trnid[actuator_id, 0]) != joint_id:
                    raise RuntimeError(f"Actuator transmission mismatch for {joint_name!r}")
                self._mj_base_actuator_ids.append(actuator_id)

        for joint_name in CONTROLLED_JOINTS:
            joint_id = _object_id(self.mj_model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
            if self.mj_model.jnt_type[joint_id] not in (
                mujoco.mjtJoint.mjJNT_HINGE,
                mujoco.mjtJoint.mjJNT_SLIDE,
            ):
                raise RuntimeError(f"Controlled joint {joint_name!r} is not one-DoF")
            body_id = int(self.mj_model.jnt_bodyid[joint_id])
            if not np.isclose(self.mj_model.body_gravcomp[body_id], 1.0):
                raise RuntimeError(f"Controlled body for {joint_name!r} does not have gravcomp=1")
            self._mj_qpos_indices.append(int(self.mj_model.jnt_qposadr[joint_id]))
            self._mj_qvel_indices.append(int(self.mj_model.jnt_dofadr[joint_id]))

            actuator_id = _object_id(
                self.mj_model,
                mujoco.mjtObj.mjOBJ_ACTUATOR,
                f"position_{joint_name}",
            )
            if int(self.mj_model.actuator_trnid[actuator_id, 0]) != joint_id:
                raise RuntimeError(f"Actuator transmission mismatch for {joint_name!r}")
            self._mj_upper_actuator_ids.append(actuator_id)

        for site_name in TCP_FRAMES.values():
            site_id = _object_id(self.mj_model, mujoco.mjtObj.mjOBJ_SITE, site_name)
            body_id = int(self.mj_model.site_bodyid[site_id])
            if not np.isclose(self.mj_model.body_gravcomp[body_id], 1.0):
                raise RuntimeError(f"TCP body for {site_name!r} does not have gravcomp=1")
        for body_name in TARGET_BODIES.values():
            body_id = _object_id(self.mj_model, mujoco.mjtObj.mjOBJ_BODY, body_name)
            if self.mj_model.body_mocapid[body_id] < 0:
                raise RuntimeError(f"IK target {body_name!r} is not a mocap body")
        self._mj_gripper_qpos_indices = [
            int(self.mj_model.jnt_qposadr[
                _object_id(self.mj_model, mujoco.mjtObj.mjOBJ_JOINT, name)
            ])
            for name in self.auxiliary_joints
        ]

    def _apply_simulation_servo_tuning(self) -> None:
        """Apply profile-selected MuJoCo servo damping without changing the scene."""

        kv = self.config.simulation_arm_servo_kv
        for joint_name in ARM_JOINTS:
            actuator_id = _object_id(
                self.mj_model,
                mujoco.mjtObj.mjOBJ_ACTUATOR,
                f"position_{joint_name}",
            )
            # Include zero: selecting the safe profile on a reused model must
            # undo damping left by a previously selected simulation profile.
            # MuJoCo position actuators encode -kp and -kv in biasprm[1:3].
            self.mj_model.actuator_biasprm[actuator_id, 2] = -kv

    def measured_configuration(self, mj_data: mujoco.MjData) -> np.ndarray:
        return np.asarray(mj_data.qpos[self._mj_qpos_indices]).copy()

    def measured_velocity(self, mj_data: mujoco.MjData) -> np.ndarray:
        return np.asarray(mj_data.qvel[self._mj_qvel_indices]).copy()

    def base_pose(self, mj_data: mujoco.MjData) -> np.ndarray | None:
        if self.base_mode == "fixed":
            return None
        return self.measured_configuration(mj_data)[:3]

    def measured_gripper_positions(self, mj_data: mujoco.MjData) -> np.ndarray:
        """Joint encoder equivalents in the portable gravity model's order."""
        return np.asarray(mj_data.qpos[self._mj_gripper_qpos_indices]).copy()

    def base_velocity(self, mj_data: mujoco.MjData) -> np.ndarray | None:
        if self.base_mode == "fixed":
            return None
        return self.measured_velocity(mj_data)[:3]

    def lift_position(self, mj_data: mujoco.MjData) -> float:
        return float(mj_data.qpos[self._mj_qpos_indices[self._base_dofs]])

    def apply_command(self, mj_data: mujoco.MjData, command: IKCommand) -> None:
        if not command.executable:
            raise RuntimeError(f"Refusing non-executable IK command: {command.diagnostics.fault}")
        if (command.mit is None) != (self.config.mit_servo is None):
            raise ValueError("position and MIT actuator command types cannot be interchanged")
        positions = np.asarray(command.positions)
        velocity = np.asarray(command.velocity)
        if (positions.shape != (self.model.nq,) or velocity.shape != (self.model.nv,)
                or not np.all(np.isfinite(positions)) or not np.all(np.isfinite(velocity))):
            raise ValueError("IK command must contain complete finite position and velocity vectors")
        pending = mj_data.ctrl.copy()
        if command.mit is not None:
            if command.mit.joint_names != ARM_JOINTS:
                raise ValueError("MIT command joint order does not match the robot")
            servo = self.config.mit_servo
            if (not np.array_equal(command.mit.kp, servo.kp)
                    or not np.array_equal(command.mit.kd, servo.kd)):
                raise ValueError("MIT command gains do not match the configured motor model")
            arm_slice = slice(self._base_dofs + 1, None)
            if (not np.array_equal(command.mit.position, positions[arm_slice])
                    or not np.array_equal(command.mit.velocity, velocity[arm_slice])):
                raise ValueError("MIT packet and whole-body command disagree")
        if self.base_mode == "planar":
            if command.base is None:
                raise ValueError("planar controller command is missing its base command")
            pending[self._mj_base_actuator_ids] = command.base.velocity_world
        pending[self._mj_upper_actuator_ids] = positions[self._base_dofs :]
        if command.mit is not None:
            mit = command.mit
            pending[self._mj_upper_actuator_ids[1:]] = mit.motor_input()
        if not np.all(np.isfinite(pending)):
            raise ValueError("nonfinite actuator command")
        mj_data.ctrl[:] = pending
        if command.mit is not None:
            # This is a documented plant assumption for the non-MIT lift, not
            # a torque field added to its hardware position command contract.
            lift_actuator = self._mj_upper_actuator_ids[0]
            self.mj_model.actuator_biasprm[lift_actuator, 0] = (
                self._gravity_effort[self._base_dofs]
            )

    def check_fk_parity(self, mj_data: mujoco.MjData, tolerance: float = 1e-6) -> dict[str, tuple[float, float]]:
        """Assert that Pinocchio TCP frames and MuJoCo TCP sites agree."""
        q = self.measured_configuration(mj_data)
        self.configuration.update(q)
        errors: dict[str, tuple[float, float]] = {}
        for side, frame_name in TCP_FRAMES.items():
            pin_pose = self.configuration.get_transform_frame_to_world(frame_name)
            mj_pose = site_pose(self.mj_model, mj_data, frame_name)
            position_error = float(np.linalg.norm(pin_pose.translation - mj_pose.translation))
            orientation_error = float(np.linalg.norm(pin.log3(pin_pose.rotation.T @ mj_pose.rotation)))
            errors[side] = (position_error, orientation_error)
            if position_error > tolerance or orientation_error > tolerance:
                raise RuntimeError(
                    f"{side} TCP differs between Pinocchio and MuJoCo: "
                    f"position={position_error:.3g}m, orientation={orientation_error:.3g}rad"
                )
        return errors
