"""Thin encoder-only adapter around the editable upstream PinkIKCore.

The policy's pad-centre TCP is converted to the upstream fingertip TCP with
one rigid transform. All optimization, reference integration, joint limits,
effort feasibility and braking are performed by the submodule's controller.
The mobile base is absent from its state vector; this is robot-base-frame IK.
The lift remains a measured coordinate but is hard-locked in the QP; only the
fourteen arm coordinates can move.
"""
from pathlib import Path
from dataclasses import replace

import numpy as np
import yaml

from ..interfaces.contract import validate_sensor_packet
from ..robot.joints import JOINTS, LEFT, RIGHT, LIFT, SIDES
from ..robot.upstream import PROJECT_ROOT, load_umi, umi_revision
from .velocity_limits import arm_velocity_bounds
from .lift_limits import FIXED_LIFT_MAX_M


DEFAULT_CONFIG = PROJECT_ROOT / "configs/controller/umi_pink.yaml"
LOCKED_JOINTS = (LIFT,)


class UMIPinkIK:
    def __init__(self, kinematics, config_path=DEFAULT_CONFIG, *, runtime_effort_limits_nm=None,
                 runtime_velocity_bounds=None):
        upstream = load_umi()
        try:
            import _hexfellow_collision_kernel as native
        except ImportError as error:
            raise RuntimeError(
                "Build the owned native extension with ./scripts/setup_sim.sh"
            ) from error
        if Path(native.__file__).resolve().parent != PROJECT_ROOT / "src":
            raise RuntimeError("The native extension must come from this hexfellow-deploy installation")
        # Preserve the upstream shared-library import order (qpsolvers first).
        from hexfellow_deploy.robotics.control.ik_core import PinkIKCore
        from hexfellow_deploy.robotics.control.motion_profile import controller_config_for_profile, load_motion_profile
        import pinocchio as pin

        self.pin = pin
        self.kinematics = kinematics
        self.config_path = Path(config_path).resolve()
        settings = yaml.safe_load(self.config_path.read_text())
        if set(settings) != {"motion_profile", "measured_pose_tracking", "arm_velocity_limit_rad_s",
                             "lift_velocity_limit_m_s"}:
            raise ValueError("Unexpected UMI adapter configuration")
        for name in ("lift_velocity_limit_m_s",):
            value = settings[name]
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not np.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        profile = load_motion_profile(settings["motion_profile"])
        config = controller_config_for_profile(profile)
        # A hard-locked lift has no commanded travel or stopping distance.
        # Keep endpoint FK at the measured height, not a 2-mm inward projection.
        config = replace(config, lift_position_margin=0.)
        if config.base_control_enabled or config.robot_model is None:
            raise ValueError("The encoder-only adapter requires a fixed-base robot profile")
        bounds = arm_velocity_bounds(settings["arm_velocity_limit_rad_s"], profile)
        bounds[LIFT] = settings["lift_velocity_limit_m_s"]
        if runtime_velocity_bounds is not None:
            if (set(runtime_velocity_bounds) != set(JOINTS)
                    or any(isinstance(v, bool) or not isinstance(v, (int, float))
                           or not np.isfinite(v) or v <= 0 for v in runtime_velocity_bounds.values())):
                raise ValueError('Runtime velocity bounds require all 15 physical joints')
            bounds = {n: min(bounds[n], runtime_velocity_bounds[n]) for n in JOINTS}
        self.core = PinkIKCore(config=config, base_mode="fixed",
                              locked_joint_names=LOCKED_JOINTS,
                              runtime_position_bounds={LIFT: (0., FIXED_LIFT_MAX_M)},
                              runtime_velocity_bounds=bounds,
                              runtime_effort_limits_nm=runtime_effort_limits_nm,
                              measured_pose_tracking=settings["measured_pose_tracking"])
        self.indices = {n: self.core.model.joints[self.core.model.getJointId(n)].idx_q for n in JOINTS}
        if self.core.model.nq != len(JOINTS) or self.core.model.nv != len(JOINTS):
            raise RuntimeError("UMI model contains non-encoder coordinates")
        self.data = self.core.model.createData()
        self.frame_ids = {s: self.core.model.getFrameId(s + "_tcp") for s in ("left", "right")}
        # Derive the static offset from robot geometry, without scene state.
        neutral = {n: 0. for n in JOINTS}
        kinematics.update(neutral)
        umi_poses = self.poses(neutral)
        self.policy_T_umi = {side: kinematics.pose(side).inverse() * umi_poses[side] for side in SIDES}
        self.initialized = False
        self.control_clock = None  # Hardware supplies monotonic time; simulation uses sensor time.
        self.feedback_provider = None  # Private per-device timestamps from hardware.
        self.last_result = None
        self.last_source = None
        self.tracking_requested = False
        self.targets = None
        self.report = {
            "implementation": "hexfellow_deploy.robotics.control.ik_core.PinkIKCore",
            "upstream_commit": umi_revision(), "motion_profile": settings["motion_profile"],
            "motion_profile_sha256": config.motion_profile_sha256,
            "base_mode": "fixed", "state_coordinates": list(JOINTS),
            "locked_joints": list(LOCKED_JOINTS), "controlled_arms": list(SIDES),
            "lift_control": "hold_initial_reference",
            "measured_pose_tracking": settings["measured_pose_tracking"],
            "runtime_velocity_bounds": bounds,
            "lift_position_max_m": FIXED_LIFT_MAX_M,
            "policy_T_umi_tcp": {side: pose.homogeneous.tolist() for side, pose in self.policy_T_umi.items()},
            "self_collision_enabled": config.self_collision_enabled,
            "actuation": "Transport selects position references (simulation) or native UMI MIT (hardware)",
            "upstream_path": str(upstream.relative_to(PROJECT_ROOT)),
        }

    @property
    def dt(self):
        return self.core.config.control_dt

    def vector(self, values):
        q = np.empty(len(JOINTS))
        for name, index in self.indices.items():
            q[index] = values[name]
        return q

    def poses(self, joints):
        self.pin.framesForwardKinematics(self.core.model, self.data, self.vector(joints))
        return {s: self.data.oMf[i].copy() for s, i in self.frame_ids.items()}

    def track(self, targets):
        if set(targets) != set(SIDES):
            raise ValueError("UMI tracking requires both left and right targets")
        self.requested_targets = {side: self.pin.SE3(rotation, position) * self.policy_T_umi[side]
                                  for side, (position, rotation) in targets.items()}
        self.tracking_requested = True

    def hold(self):
        self.tracking_requested = False
        if self.initialized:
            self.core.request_stop()

    def initialize_reference(self, measured, *, stationary_reference=False):
        """One-shot command-history seed; hardware shares the native guard seed."""
        if self.initialized:
            raise RuntimeError('UMI reference is already initialized; cannot rearm/reset history')
        validate_sensor_packet(measured)
        poses = self.poses(measured['joint_positions'])
        self.core.reset(self.vector(measured['joint_positions']), poses,
                        measured_velocity=self.vector(measured['joint_velocities']),
                        state_timestamp=measured['timestamp_s'],
                        stationary_reference=stationary_reference)
        self.targets = poses
        self.initialized = True
        if not self.tracking_requested:
            self.core.request_stop()

    def step(self, measured):
        validate_sensor_packet(measured)
        control_timestamp = self.control_clock() if self.control_clock else measured['timestamp_s']
        q = self.vector(measured["joint_positions"])
        velocity = self.vector(measured["joint_velocities"])
        if not self.initialized:
            self.initialize_reference(measured)
        if self.tracking_requested:
            if self.core.control_state == "HOLD":
                self.core.resume_tracking()
            self.targets = self.requested_targets
        received = self.feedback_provider(measured) if self.feedback_provider else None
        # The public epoch includes hands; IK uses the original native group
        # clocks, with joint changes checked against their own device clock.
        state_timestamp = min(received.values()) if received is not None else measured['timestamp_s']
        result = self.core.step(q, self.targets, measured_velocity=velocity,
                                state_timestamp=state_timestamp,
                                control_timestamp=control_timestamp,
                                feedback_received=received, feedback_policy="strict")
        if not result.executable or result.diagnostics.fault:
            excess = [f'{name}={velocity[i]:.6f} (limit {self.core.model.velocityLimit[i]:.6f})'
                      for name, i in self.indices.items()
                      if abs(velocity[i]) > self.core.model.velocityLimit[i] + self.core.config.constraint_tolerance]
            detail = '; measured joint speeds: ' + ', '.join(excess) if excess else ''
            raise RuntimeError(f"UMI IK stopped: {result.diagnostics.fault or result.diagnostics.status}{detail}")
        self.last_result = result
        self.last_source = dict(timestamp_s=measured['timestamp_s'],
                                joint_positions=dict(measured['joint_positions']),
                                feedback_received=dict(received) if received is not None else None)
        return {n: float(result.positions[i]) for n, i in self.indices.items()}, result.diagnostics
