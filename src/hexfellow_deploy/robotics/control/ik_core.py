# ruff: noqa: I001 -- qpsolvers must load before Pink/Pinocchio 4 shared libraries.
"""Simulator-independent Pink differential-IK controller for the Maver X4."""

from __future__ import annotations

import time
import hashlib
from collections.abc import Mapping
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import qpsolvers
import pink
import pinocchio as pin
from pink.exceptions import NoSolutionFound
from pink.limits import AccelerationLimit, ConfigurationLimit, Limit, VelocityLimit
from pink.tasks import DampingTask, FrameTask, LowAccelerationTask, PostureTask

from .adaptive_share import AdaptiveShare
from .allocator import (
    ADAPTIVE_SHARE_MODES,
    DEFAULT_WHOLE_BODY_ALLOCATOR_CONFIG,
    POSTURE_SHARE_MODES,
    PREDICTIVE_SHARE_MODES,
    WholeBodyAllocationDiagnostics,
    WholeBodyAllocatorConfig,
    WholeBodyAllocatorState,
    arm_limit_pressures,
    filtered_activation,
)
from .allocator import (
    smoothstep as _smoothstep,
)
from .allocator import (
    zero_based_activation as _zero_based_activation,
)
from .qp_pipeline import assemble_problem
from .posture_policy import HumanPosturePolicy, PostureTransition
from .posture import (
    ArmKinematicsCache,
    ArmManipulabilityMetrics,
    ArmNullspaceManipulabilityTask,
    ArmNullspacePostureTask,
    NullPose,
)
from .robot import (
    BASE_ACTUATORS as BASE_ACTUATORS,
    BASE_FRAME,
    BASE_JOINTS as BASE_JOINTS,
    CONTROLLED_JOINTS,
    DEFAULT_PLANAR_BASE_CONFIG,
    DEFAULT_URDF as DEFAULT_URDF,
    LEFT_ARM_JOINTS,
    LEFT_IK_TARGET,
    LEFT_TCP_SITE,
    LIFT_JOINT,
    READY_QPOS,
    RIGHT_ARM_JOINTS,
    RIGHT_IK_TARGET,
    RIGHT_TCP_SITE,
    PlanarBaseConfig,
)
from .robot_model import RobotModel, load_robot_model
from .self_collision import (
    DEFAULT_SELF_COLLISION_CONFIG,
    CollisionPairDiagnostic,
    SelfCollisionSystem,
)
from .target_motion import (
    C2TargetMotionPlanner,
    ContinuousTargetMotionPlanner,
    GuardedTargetMotionPlanner,
    PrecisionTargetMotionPlanner,
    CartesianMotionLimits,
    CartesianMotionUtilization,
    TargetTrajectoryError,
)
from .mit import (
    DynamicFeedforward, GravityCorrection, GravityPayload, JointStateGravity, MitCommand, MitServoConfig,
)
from .command_limits import (
    DirectionalVelocityLimit, JointBoundaryGovernor, JointReferenceLimit,
    MeasuredCollisionLimit, jerk_reaction_stopping_distance, joint_boundary_damper,
    stopping_distance,
)
from .qp_diagnostics import QPFailureSnapshot

TCP_FRAMES = {"left": LEFT_TCP_SITE, "right": RIGHT_TCP_SITE}
TARGET_BODIES = {"left": LEFT_IK_TARGET, "right": RIGHT_IK_TARGET}
ARM_JOINTS = (*LEFT_ARM_JOINTS, *RIGHT_ARM_JOINTS)



@dataclass(frozen=True)
class PinkControllerConfig:
    """Tuning and safety limits for the 50 Hz differential-IK loop."""

    robot_model: RobotModel | None = None
    preparation_lift_velocity_m_s: float = .02
    base_control_enabled: bool = True
    stationary_redundancy_hold: bool = False
    control_dt: float = 0.02
    solver: str = "daqp"
    qp_primal_tolerance: float | None = None
    qp_regularization: float = 1e-9
    position_cost: float = 50.0
    orientation_cost: float = 5.0
    frame_lm_damping: float = 1e-3
    # The longer V3 gripper TCP benefits from a gentler differential-IK gain;
    # 0.8 excites a persistent arm-servo oscillation at the new tool length.
    orientation_gain: float = 0.1
    effort_regularization_cost: float = 0.0
    stationary_cost_blend_s: float = 0.0
    frame_gain: float = 0.55
    base_anchor_gain: float = 1.0
    simulation_arm_servo_kv: float = 0.0
    mit_servo: MitServoConfig | None = None
    trajectory_kernel_version: int | None = None
    posture_policy: HumanPosturePolicy | None = None
    tracking_objective: str = "legacy"
    stationary_arrival: str = "legacy"
    mit_effort_preview: str = "none"
    mit_gravity_correction: GravityCorrection | None = None
    mit_dynamic_feedforward: DynamicFeedforward | None = None
    mit_planning_effort_fraction: float = 1.0
    lift_reference_max_error_m: float | None = None
    reference_velocity_fraction: float = 1.
    lift_reference_velocity_fraction: float = 1.
    maximum_feedback_age_s: float = 0.04
    motion_envelope: str = "none"
    gravity_payloads: tuple[GravityPayload, ...] = ()
    gravity_urdf_sha256: str | None = None
    measured_collision_method: str = "reference_velocity_v1"
    boundary_policy: str = "legacy"
    boundary_gain_s_inv: float = 1.0
    maximum_command_delay_s: float = 0.02
    collision_sweep_max_joint_step_rad: float | None = None
    collision_sweep_max_intervals: int = 0
    collision_activation_policy: str = "static_worst_case_v2"
    target_feedforward_gain: float = 1.0
    target_velocity_feedback_gain: float = 0.0
    target_tracking_bandwidth_s_inv: float = 12.0
    target_source_velocity_gain: float = 1.0
    target_source_acceleration_gain: float = 0.0
    target_source_acceleration_tau_s: float = 0.06
    arm_posture_cost: float = 0.4
    arm_posture_min_fraction: float = 0.0
    lift_posture_cost: float = 5.0
    # Penalize lift velocity without weakening its anchor or limit escape.
    # Zero reproduces the pre-integration controller for comparisons.
    lift_velocity_damping_scale: float = 3.0
    low_acceleration_cost: float = 0.04
    arm_manipulability_cost: float = 0.03
    arm_manipulability_rate: float = 0.01
    arm_manipulability_floor_fraction: float = 0.65
    arm_manipulability_exit_fraction: float = 0.72
    arm_manipulability_critical_sigma_fraction: float = 0.3
    arm_manipulability_activation_ramp_s: float = 0.1
    arm_manipulability_angular_length_scale_m: float = 0.25
    arm_jacobian_rank_rtol: float = 1e-6
    arm_velocity_limit: float = 0.8
    lift_velocity_limit: float = 0.05
    arm_acceleration_limit: float = 6.0
    lift_acceleration_limit: float = 0.2
    arm_jerk_limit: float = 1200.0
    lift_jerk_limit: float = 40.0
    arm_velocity_limits: tuple[float, ...] | None = None
    arm_acceleration_limits: tuple[float, ...] | None = None
    arm_jerk_limits: tuple[float, ...] | None = None
    motion_limit_source: str = "assumed"
    motion_profile_id: str | None = None
    motion_profile_sha256: str | None = None
    motion_profile_path: Path | None = None
    self_collision_enabled: bool = True
    self_collision_config: Path | None = DEFAULT_SELF_COLLISION_CONFIG
    self_collision_envelope_config: Path | None = None
    self_collision_pair_certificate_config: Path | None = None
    self_collision_pair_certificate_sha256: str | None = None
    self_collision_runtime: str = "auto_compatible_v1"
    native_collision_kernel_abi: int | None = None
    # Candidate loading is confined to deterministic review/benchmark tools.
    # Production callers keep the approval gate enabled by default.
    self_collision_require_approved: bool = True
    # Leave a physical guard band without making ordinary gripper contact push
    # a measured servo outside the planner's artificial interval.
    arm_position_margin: float = 0.012
    arm_position_lower_limits: tuple[float, ...] | None = None
    arm_position_upper_limits: tuple[float, ...] | None = None
    lift_position_margin: float = 0.002
    target_linear_velocity_limit: float = 0.15
    target_angular_velocity_limit: float = 0.45
    target_linear_acceleration_limit: float = 0.95
    target_linear_jerk_limit: float = 20.0
    target_angular_acceleration_limit: float = 2.8
    target_angular_jerk_limit: float = 60.0
    target_planner: str = "legacy_rate_clip_v1"
    # Deadbands coalesce device jitter around the last distinct request. Their
    # maximum bounded pose error is explicitly part of the motion profile.
    target_translation_deadband: float = 0.0
    target_orientation_deadband: float = 0.0
    target_stationary_dwell: float = 0.0
    target_limit_check_runtime: str = "auto_compatible_v1"
    settled_low_acceleration_cost: float = 0.04
    settled_arm_damping_cost: float = 0.0
    settling_blend_s: float = 0.15
    settling_linear_velocity_m_s: float = 0.01
    settling_angular_velocity_rad_s: float = 0.05
    settling_position_error_m: float = 0.005
    settling_orientation_error_rad: float = np.deg2rad(3.0)
    large_position_error: float = 0.03
    large_orientation_error: float = 0.15
    large_error_delay: float = 0.5
    constraint_tolerance: float = 1e-2
    base: PlanarBaseConfig = DEFAULT_PLANAR_BASE_CONFIG
    allocator: WholeBodyAllocatorConfig = DEFAULT_WHOLE_BODY_ALLOCATOR_CONFIG

    def __post_init__(self) -> None:
        if self.tracking_objective not in {"legacy", "normalized_v1"} or self.stationary_arrival not in {"legacy", "continuous_v1"}:
            raise ValueError("invalid balanced tracking mode")
        if not np.isfinite(self.orientation_gain) or not 0 < self.orientation_gain <= 1:
            raise ValueError("orientation_gain must be in (0, 1]")
        if not np.isfinite(self.effort_regularization_cost) or self.effort_regularization_cost < 0:
            raise ValueError("effort_regularization_cost must be non-negative")
        if not np.isfinite(self.stationary_cost_blend_s) or self.stationary_cost_blend_s < 0:
            raise ValueError("stationary_cost_blend_s must be non-negative")
        if (not np.isfinite(self.target_source_velocity_gain) or not 0 <= self.target_source_velocity_gain <= 1
                or (self.target_source_velocity_gain != 1 and self.robot_model is None)):
            raise ValueError("low-speed source velocity gain requires an explicit robot model")
        if not np.isfinite(self.base_anchor_gain) or not 0 < self.base_anchor_gain <= 1:
            raise ValueError("base_anchor_gain must be in (0, 1]")
        if (not np.isfinite(self.target_tracking_bandwidth_s_inv)
                or not (2.0 if self.robot_model is not None else 8.0) <= self.target_tracking_bandwidth_s_inv <= 20.0
                or (self.target_tracking_bandwidth_s_inv != 12.0
                    and self.target_planner not in {"c2_tracking_v4", "c2_tracking_v5", "c2_tracking_v6"})):
            raise ValueError("profile tracking bandwidth requires v4/v5; GP80 [2, 20], legacy [8, 20] s^-1")
        if self.allocator.mode == "directional_body_share_v6" and (
                self.lift_reference_max_error_m is None
                or self.target_planner not in {"c2_tracking_v5", "c2_tracking_v6"}):
            raise ValueError("directional body share requires persistent lift reference and v5 target planner")
        if (self.mit_effort_preview not in {"none", "jerk_prefix_v1"}
                or (self.mit_effort_preview != "none" and self.mit_servo is None)):
            raise ValueError("invalid MIT effort preview")
        if self.mit_gravity_correction is not None and (
                not isinstance(self.mit_gravity_correction, GravityCorrection) or self.mit_servo is None):
            raise ValueError("MIT gravity correction requires an MIT servo configuration")
        if self.mit_dynamic_feedforward is not None and (
                not isinstance(self.mit_dynamic_feedforward, DynamicFeedforward) or self.mit_servo is None):
            raise ValueError("MIT dynamic feedforward requires an MIT servo configuration")
        if (not np.isfinite(self.mit_planning_effort_fraction)
                or not 0 < self.mit_planning_effort_fraction <= 1
                or (self.mit_planning_effort_fraction != 1 and self.mit_servo is None)):
            raise ValueError("invalid MIT planning effort fraction")
        if self.lift_reference_max_error_m is not None:
            if (not np.isfinite(self.lift_reference_max_error_m)
                    or self.lift_reference_max_error_m <= 0 or self.mit_servo is None
                    or self.boundary_policy != "jerk_stoppable_directional_v2"):
                raise ValueError("persistent lift reference requires MIT and jerk-stoppable boundaries")
        if self.posture_policy is not None and not isinstance(self.posture_policy, HumanPosturePolicy):
            raise ValueError("posture policy must be a validated HumanPosturePolicy")
        if self.trajectory_kernel_version is not None:
            try:
                from _hexfellow_collision_kernel import TRAJECTORY_KERNEL_VERSION
            except ImportError as error:
                raise RuntimeError("fast IK requires rebuilt native trajectory kernel v1") from error
            if self.trajectory_kernel_version != 1 or TRAJECTORY_KERNEL_VERSION != 1:
                raise RuntimeError("unsupported native trajectory kernel version")
        if not np.isfinite(self.lift_reference_velocity_fraction) or not 0 < self.lift_reference_velocity_fraction <= 1:
            raise ValueError("invalid lift reference velocity fraction")
        if not np.isfinite(self.reference_velocity_fraction) or not 0 < self.reference_velocity_fraction <= 1:
            raise ValueError("reference velocity fraction must be in (0,1]")
        if self.motion_envelope not in {"none", "first-trial"}:
            raise ValueError("invalid motion envelope")
        if self.motion_envelope == "first-trial" and (
                self.motion_limit_source != "profile"
                or self.motion_profile_id not in {
                    "x7-mit-highspeed-v10", "x7-mit-highspeed-v10-base-v1", "x7-mit-highspeed-v10-base-lift-v2", "x7-mit-highspeed-v10-base-lift-v3", "x7-mit-highspeed-v10-base-lift-v4"}):
            raise ValueError("first-trial requires profile-bound v10 limits")
        if any(not isinstance(payload, GravityPayload) for payload in self.gravity_payloads):
            raise ValueError("gravity payloads must be GravityPayload declarations")
        object.__setattr__(self, "gravity_payloads", tuple(self.gravity_payloads))
        if not np.isfinite(self.maximum_feedback_age_s) or self.maximum_feedback_age_s <= 0:
            raise ValueError("maximum feedback age must be finite and positive")
        if self.mit_servo is not None and not isinstance(self.mit_servo, MitServoConfig):
            raise ValueError("mit_servo must be a MitServoConfig")
        if self.measured_collision_method not in {"reference_velocity_v1", "reference_endpoint_v2"}:
            raise ValueError("unknown measured-reference collision method")
        if self.measured_collision_method != "reference_velocity_v1" and self.mit_servo is None:
            raise ValueError("reference endpoint collision method requires MIT")
        if self.self_collision_runtime not in {"auto_compatible_v1", "native_fused_v2", "disabled"}:
            raise ValueError("unknown self-collision runtime")
        if self.self_collision_runtime == "disabled" and self.self_collision_enabled:
            raise ValueError("disabled collision runtime cannot enable collision checking")
        if self.self_collision_runtime == "native_fused_v2":
            if self.native_collision_kernel_abi not in {2, 3} or self.mit_servo is None:
                raise ValueError("native_fused_v2 requires MIT and collision kernel ABI 2+")
        elif self.self_collision_runtime != "disabled" and self.native_collision_kernel_abi is not None:
            raise ValueError("native collision ABI requires native_fused_v2")
        if self.target_limit_check_runtime not in {
            "auto_compatible_v1", "native_full_so3_v1"
        }:
            raise ValueError("unknown target limit-check runtime")
        if self.target_limit_check_runtime == "native_full_so3_v1" and (
            self.native_collision_kernel_abi is None
            or self.native_collision_kernel_abi < 3
        ):
            raise ValueError("native_full_so3_v1 requires collision kernel ABI 3")
        if self.boundary_policy not in {
            "legacy", "directional_velocity_damper_v1", "jerk_stoppable_directional_v2"
        }:
            raise ValueError("unknown boundary policy")
        if (not np.isfinite(self.boundary_gain_s_inv) or self.boundary_gain_s_inv <= 0
                or not np.isfinite(self.maximum_command_delay_s) or self.maximum_command_delay_s < 0):
            raise ValueError("invalid boundary gain or command delay")
        if self.boundary_policy != "legacy" and (
                self.mit_servo is None
                or self.measured_collision_method != "reference_endpoint_v2"):
            raise ValueError("directional boundaries require MIT endpoint constraints")
        if self.boundary_policy == "jerk_stoppable_directional_v2":
            if (
                self.collision_sweep_max_joint_step_rad is None
                or not np.isfinite(self.collision_sweep_max_joint_step_rad)
                or self.collision_sweep_max_joint_step_rad <= 0.0
                or not isinstance(self.collision_sweep_max_intervals, int)
                or not 1 <= self.collision_sweep_max_intervals <= 32
            ):
                raise ValueError("v2 boundaries require a finite collision sweep contract")
        elif self.collision_sweep_max_joint_step_rad is not None or self.collision_sweep_max_intervals:
            raise ValueError("collision sweep settings require the v2 boundary policy")
        if self.collision_activation_policy not in {
            "static_worst_case_v2",
            "causal_reachable_v3",
        }:
            raise ValueError("unknown collision activation policy")
        if (
            self.collision_activation_policy == "causal_reachable_v3"
            and self.boundary_policy != "jerk_stoppable_directional_v2"
        ):
            raise ValueError("causal collision activation requires v2 directional boundaries")
        if self.mit_servo is not None and self.simulation_arm_servo_kv != 0.0:
            raise ValueError("MIT gains cannot be combined with simulation-only servo tuning")
        if self.qp_primal_tolerance is not None and (
            self.solver != "daqp" or not np.isfinite(self.qp_primal_tolerance)
            or self.qp_primal_tolerance <= 0.
        ):
            raise ValueError("QP primal tolerance requires DAQP and a finite positive value")
        if not np.isfinite(self.qp_regularization) or self.qp_regularization <= 0.0:
            raise ValueError("QP regularization must be finite and positive")
        if not np.isfinite(self.lift_velocity_damping_scale) or self.lift_velocity_damping_scale < 0.0:
            raise ValueError("lift velocity damping scale must be finite and non-negative")
        if self.allocator.arm_limit_cost < self.arm_posture_cost:
            raise ValueError("near-limit arm cost must not be lower than the comfortable cost")
        if not 0.0 <= self.arm_posture_min_fraction <= 1.0:
            raise ValueError("minimum arm posture fraction must lie in [0, 1]")
        if self.allocator.lift_active_cost > self.lift_posture_cost:
            raise ValueError("active lift cost must not exceed its default cost")
        if self.allocator.base_active_cost > min(
            self.base.anchor_position_cost,
            self.base.anchor_yaw_cost,
        ):
            raise ValueError("active base cost must not exceed its default costs")
        if (
            self.arm_manipulability_cost <= 0.0
            or self.arm_manipulability_rate <= 0.0
            or not 0.0
            < self.arm_manipulability_floor_fraction
            < self.arm_manipulability_exit_fraction
            < 1.0
            or not 0.0 < self.arm_manipulability_critical_sigma_fraction < 1.0
            or self.arm_manipulability_activation_ramp_s <= 0.0
            or self.arm_manipulability_angular_length_scale_m <= 0.0
            or self.arm_jacobian_rank_rtol <= 0.0
        ):
            raise ValueError("invalid arm manipulability thresholds, scale, ramp, or rank tolerance")
        for label, values, fallback in (
            ("velocity", self.arm_velocity_limits, self.arm_velocity_limit),
            ("acceleration", self.arm_acceleration_limits, self.arm_acceleration_limit),
            ("jerk", self.arm_jerk_limits, self.arm_jerk_limit),
        ):
            resolved = (fallback,) * 14 if values is None else values
            if len(resolved) != 14 or any(not np.isfinite(value) or value <= 0.0 for value in resolved):
                raise ValueError(f"arm {label} limits must contain fourteen finite positive values")
        if (self.arm_position_lower_limits is None) != (self.arm_position_upper_limits is None):
            raise ValueError("arm position lower and upper limits must be supplied together")
        if self.arm_position_lower_limits is not None:
            lower = np.asarray(self.arm_position_lower_limits, dtype=float)
            upper = np.asarray(self.arm_position_upper_limits, dtype=float)
            if (
                lower.shape != (14,)
                or upper.shape != (14,)
                or not np.all(np.isfinite((lower, upper)))
                or np.any(lower >= upper)
            ):
                raise ValueError("arm position limits must contain fourteen ordered intervals")
        if self.lift_jerk_limit <= 0.0 or self.constraint_tolerance <= 0.0:
            raise ValueError("lift jerk limit and constraint tolerance must be positive")
        if not np.isfinite(self.simulation_arm_servo_kv) or self.simulation_arm_servo_kv < 0.0:
            raise ValueError("simulation arm servo kv must be finite and non-negative")
        target_limits = (
            self.target_linear_velocity_limit,
            self.target_angular_velocity_limit,
            self.target_linear_acceleration_limit,
            self.target_linear_jerk_limit,
            self.target_angular_acceleration_limit,
            self.target_angular_jerk_limit,
        )
        if any(not np.isfinite(value) or value <= 0.0 for value in target_limits):
            raise ValueError("target velocity, acceleration, and jerk limits must be finite and positive")
        if not np.isfinite(self.target_feedforward_gain) or not 0. <= self.target_feedforward_gain <= 1.:
            raise ValueError("target feedforward gain must be between zero and one")
        if not np.isfinite(self.target_velocity_feedback_gain) or not 0. <= self.target_velocity_feedback_gain <= 2.:
            raise ValueError("target velocity feedback gain must be between zero and two")
        if self.target_planner not in {
            "legacy_rate_clip_v1",
            "c2_stop_aware_v1",
            "c2_tracking_v2",
            "c2_tracking_v3",
            "c2_tracking_v4", "c2_tracking_v5", "c2_tracking_v6",
        }:
            raise ValueError("unknown Cartesian target planner")
        if (
            not np.isfinite(self.target_translation_deadband)
            or self.target_translation_deadband < 0.0
            or not np.isfinite(self.target_orientation_deadband)
            or self.target_orientation_deadband < 0.0
            or not np.isfinite(self.target_stationary_dwell)
            or self.target_stationary_dwell < 0.0
        ):
            raise ValueError("target deadbands must be finite and non-negative")
        settling_values = (
            self.settling_blend_s,
            self.settling_linear_velocity_m_s,
            self.settling_angular_velocity_rad_s,
            self.settling_position_error_m,
            self.settling_orientation_error_rad,
        )
        if any(not np.isfinite(value) or value <= 0.0 for value in settling_values):
            raise ValueError("settling thresholds and blend time must be finite and positive")
        if (
            not np.isfinite(self.settled_low_acceleration_cost)
            or self.settled_low_acceleration_cost < 0.0
            or not np.isfinite(self.settled_arm_damping_cost)
            or self.settled_arm_damping_cost < 0.0
        ):
            raise ValueError("settled regularization costs must be finite and non-negative")
        if self.motion_limit_source not in {"assumed", "vendor", "measured", "profile"}:
            raise ValueError("motion_limit_source must be assumed, vendor, measured, or profile")
        if self.motion_limit_source == "profile":
            if not self.motion_profile_id or not self.motion_profile_sha256:
                raise ValueError("profile motion limits require a profile ID and SHA-256")
            if len(self.motion_profile_sha256) != 64:
                raise ValueError("motion profile SHA-256 must contain 64 hexadecimal characters")
            try:
                int(self.motion_profile_sha256, 16)
            except ValueError as error:
                raise ValueError("motion profile SHA-256 is not hexadecimal") from error
            if self.self_collision_enabled and self.self_collision_envelope_config is None:
                raise ValueError("profile motion limits require a profile-bound collision envelope")
            from .motion_profile import load_motion_profile

            profile = load_motion_profile(self.motion_profile_path or self.motion_profile_id)
            from .first_trial import apply_envelope

            profile = apply_envelope(profile, self.motion_envelope)
            profile_fields_match = (
                self.robot_model == profile.robot_model
                and self.preparation_lift_velocity_m_s == profile.preparation_lift_velocity_m_s
                and self.stationary_redundancy_hold == profile.stationary_redundancy_hold
                and self.target_source_velocity_gain == profile.controller_tuning.get("target_source_velocity_gain", 1.)
                and self.target_source_acceleration_gain == profile.controller_tuning.get(
                    "target_source_acceleration_gain", 0.)
                and self.target_source_acceleration_tau_s == profile.controller_tuning.get(
                    "target_source_acceleration_tau_s", 0.06)
                and (profile.schema_version != 21 or self.self_collision_enabled == profile.self_collision_enabled)
                and self.base_anchor_gain == profile.controller_tuning.get("base_anchor_gain", 1.0)
                and profile.sha256 == self.motion_profile_sha256
                and self.target_tracking_bandwidth_s_inv == profile.controller_tuning.get(
                    "target_tracking_bandwidth_s_inv", 12.0)
                and self.trajectory_kernel_version == profile.trajectory_kernel_version
                and self.posture_policy == profile.posture_policy
                and self.lift_reference_max_error_m == profile.lift_reference_max_error_m
                and self.mit_planning_effort_fraction == profile.mit_planning_effort_fraction
                and self.mit_effort_preview == profile.mit_effort_preview
                and self.mit_gravity_correction == profile.mit_gravity_correction
                and self.mit_dynamic_feedforward == profile.mit_dynamic_feedforward
                and self.tracking_objective == profile.tracking_objective
                and self.stationary_arrival == profile.stationary_arrival
                and self.mit_servo == profile.mit_servo
                and self.reference_velocity_fraction == profile.reference_velocity_fraction
                and self.lift_reference_velocity_fraction == profile.lift_reference_velocity_fraction
                and self.gravity_payloads == profile.gravity_payloads
                and self.maximum_feedback_age_s == profile.maximum_feedback_age_s
                and self.gravity_urdf_sha256 == profile.gravity_urdf_sha256
                and self.measured_collision_method == profile.measured_collision_method
                and self.base.workspace_mode == profile.base_workspace_mode
                and self.base.boundary_governor == profile.base_boundary_governor
                and self.boundary_policy == profile.boundary_policy
                and self.boundary_gain_s_inv == profile.boundary_gain_s_inv
                and self.maximum_command_delay_s == profile.maximum_command_delay_s
                and self.collision_sweep_max_joint_step_rad
                == profile.collision_sweep_max_joint_step_rad
                and self.collision_sweep_max_intervals == profile.collision_sweep_max_intervals
                and self.collision_activation_policy == profile.collision_activation_policy
                and self.self_collision_runtime == profile.collision_runtime
                and self.native_collision_kernel_abi == profile.native_collision_kernel_abi
                and self.target_limit_check_runtime == profile.target_limit_check_runtime
                and all(
                    getattr(self, key) is not None
                    and np.isclose(getattr(self, key), value, atol=1e-12, rtol=0.)
                    for key, value in profile.controller_tuning.items()
                )
                and np.isclose(self.arm_position_margin, profile.arm_position_margin_rad, atol=1e-12, rtol=0.)
                and np.isclose(self.control_dt, profile.control_dt_s, atol=1e-12)
                and (
                    profile.schema_version < 15
                    or (
                        np.isclose(self.base.linear_velocity_limit, profile.base_linear_velocity_m_s)
                        and np.isclose(self.base.angular_velocity_limit, profile.base_angular_velocity_rad_s)
                        and np.isclose(self.base.linear_acceleration_limit, profile.base_linear_acceleration_m_s2)
                        and np.isclose(self.base.angular_acceleration_limit, profile.base_angular_acceleration_rad_s2)
                        and np.isclose(self.base.linear_jerk_limit, profile.base_linear_jerk_m_s3)
                        and np.isclose(self.base.angular_jerk_limit, profile.base_angular_jerk_rad_s3)
                        and np.isclose(self.lift_velocity_limit, profile.lift_velocity_m_s)
                        and np.isclose(self.lift_acceleration_limit, profile.lift_acceleration_m_s2)
                        and np.isclose(self.lift_jerk_limit, profile.lift_jerk_m_s3)
                    )
                )
                and np.allclose(
                    self.resolved_arm_limits("velocity"), profile.arm_velocity_rad_s, atol=1e-12
                )
                and np.allclose(
                    self.resolved_arm_limits("acceleration"), profile.arm_acceleration_rad_s2, atol=1e-12
                )
                and np.allclose(
                    self.resolved_arm_limits("jerk"), profile.arm_jerk_rad_s3, atol=1e-12
                )
                and (
                    profile.sdk_position_lower_rad is None
                    or (
                        np.allclose(
                            self.arm_position_lower_limits,
                            np.maximum(
                                profile.urdf_position_lower_rad,
                                profile.sdk_position_lower_rad,
                            ),
                            atol=1e-12,
                        )
                        and np.allclose(
                            self.arm_position_upper_limits,
                            np.minimum(
                                profile.urdf_position_upper_rad,
                                profile.sdk_position_upper_rad,
                            ),
                            atol=1e-12,
                        )
                    )
                )
                and np.isclose(
                    self.target_linear_velocity_limit,
                    profile.target_linear_velocity_m_s,
                    atol=1e-12,
                )
                and np.isclose(
                    self.target_angular_velocity_limit,
                    profile.target_angular_velocity_rad_s,
                    atol=1e-12,
                )
                and self.target_planner == profile.target_planner
                and np.isclose(
                    self.target_translation_deadband,
                    profile.target_translation_deadband_m,
                    atol=1e-12,
                )
                and np.isclose(
                    self.target_orientation_deadband,
                    profile.target_orientation_deadband_rad,
                    atol=1e-12,
                )
                and np.isclose(
                    self.target_stationary_dwell,
                    profile.target_stationary_dwell_s,
                    atol=1e-12,
                )
                and self.allocator == profile.allocator
                and (
                    self.target_planner == "legacy_rate_clip_v1"
                    or (
                        np.isclose(
                            self.target_linear_acceleration_limit,
                            profile.target_linear_acceleration_m_s2,
                            atol=1e-12,
                        )
                        and np.isclose(
                            self.target_linear_jerk_limit,
                            profile.target_linear_jerk_m_s3,
                            atol=1e-12,
                        )
                        and np.isclose(
                            self.target_angular_acceleration_limit,
                            profile.target_angular_acceleration_rad_s2,
                            atol=1e-12,
                        )
                        and np.isclose(
                            self.target_angular_jerk_limit,
                            profile.target_angular_jerk_rad_s3,
                            atol=1e-12,
                        )
                    )
                )
                and (self.self_collision_config is None and profile.collision_geometry_config is None
                     or self.self_collision_config is not None
                     and Path(self.self_collision_config).resolve() == profile.collision_geometry_config)
                and (
                    not self.self_collision_enabled
                    or Path(self.self_collision_envelope_config).resolve()
                    == profile.collision_envelope_config
                )
                and (
                    (
                        self.self_collision_pair_certificate_config is None
                        and profile.collision_pair_certificate_config is None
                    )
                    or (
                        self.self_collision_pair_certificate_config is not None
                        and Path(self.self_collision_pair_certificate_config).resolve()
                        == profile.collision_pair_certificate_config
                    )
                )
                and self.self_collision_pair_certificate_sha256
                == profile.collision_pair_certificate_sha256
            )
            if not profile_fields_match:
                raise ValueError("Pink limits do not match the immutable motion-profile content")

    def resolved_arm_limits(self, kind: str) -> np.ndarray:
        values = {
            "velocity": (self.arm_velocity_limits, self.arm_velocity_limit),
            "acceleration": (self.arm_acceleration_limits, self.arm_acceleration_limit),
            "jerk": (self.arm_jerk_limits, self.arm_jerk_limit),
        }
        try:
            per_axis, fallback = values[kind]
        except KeyError as error:
            raise ValueError(f"unknown arm limit kind {kind!r}") from error
        return np.asarray((fallback,) * 14 if per_axis is None else per_axis, dtype=float)


@dataclass(frozen=True)
class IKDiagnostics:
    status: str
    solve_time_ms: float
    left_position_error: float
    right_position_error: float
    left_orientation_error: float
    right_orientation_error: float
    target_limited: bool
    target_settled: bool
    target_motion_utilization: CartesianMotionUtilization | None
    base_mode: str
    base_pose: tuple[float, float, float] | None
    base_velocity_command: tuple[float, float, float] | None
    base_at_limit: bool
    allocation: WholeBodyAllocationDiagnostics
    nullspace_posture_error_rad: dict[str, float]
    arm_manipulability: dict[str, ArmManipulabilityMetrics]
    collision_pairs: tuple[CollisionPairDiagnostic, ...] = ()
    tracking_chain: dict[str, TrackingErrorChain] | None = None
    motion_utilization: MotionUtilization | None = None
    timings: IKPhaseTimings | None = None
    fault: str | None = None
    fault_kind: str | None = None
    fault_details: dict | None = None
    stop_reason: str | None = None
    feedback_age_s: float | None = None
    dispatch_state_age_s: float | None = None
    deadline_reject_count: int = 0
    boundary_limiting_reasons: tuple[str, ...] = ()
    braking_retries: int = 0
    target_planner_failures: int = 0
    last_target_planner_error: str | None = None
    recovery_reason: str | None = None
    minimum_collision_reaction_clearance_m: float | None = None
    minimum_joint_reaction_clearance_rad: float | None = None


@dataclass(frozen=True)
class TransformError:
    position_m: float
    orientation_rad: float


@dataclass(frozen=True)
class TrackingErrorChain:
    raw_to_limited: TransformError
    limited_to_command: TransformError
    command_to_measured: TransformError


@dataclass(frozen=True)
class MotionUtilization:
    position: float
    velocity: float
    acceleration: float
    jerk: float


@dataclass(frozen=True)
class IKPhaseTimings:
    gravity_ms: float = 0.0
    output_checks_ms: float = 0.0
    state_update_ms: float = 0.0
    collision_distance_ms: float = 0.0
    collision_reference_ms: float = 0.0
    collision_measured_ms: float = 0.0
    collision_jacobian_ms: float = 0.0
    collision_row_pack_ms: float = 0.0
    task_construction_ms: float = 0.0
    planner_ms: float = 0.0
    allocator_ms: float = 0.0
    qp_build_ms: float = 0.0
    qp_solve_ms: float = 0.0
    diagnostics_ms: float = 0.0
    swept_collision_ms: float = 0.0
    total_ms: float = 0.0
    critical_wall_ms: float = 0.0
    thread_cpu_ms: float = 0.0
    scheduler_delay_ms: float = 0.0


@dataclass(frozen=True)
class PlanarBaseCommand:
    next_pose_world: np.ndarray
    velocity_world: np.ndarray
    velocity_body: np.ndarray


@dataclass(frozen=True)
class IKCommand:
    positions: np.ndarray
    velocity: np.ndarray
    diagnostics: IKDiagnostics
    base: PlanarBaseCommand | None = None
    mit: MitCommand | None = None
    executable: bool = True


class ControlDeadlineExceeded(RuntimeError):
    """A completed MIT command is too old to be executable."""

    def __init__(
        self, reason: str, *, stage: str = "unknown", elapsed_s: float | None = None,
        limit_s: float | None = None, feedback_age_s: float | None = None,
    ) -> None:
        super().__init__(reason)
        self.stage = stage
        self.elapsed_s = elapsed_s
        self.limit_s = limit_s
        self.feedback_age_s = feedback_age_s


LIFT_REFERENCE_TOP_RESERVE_M = 0.003


class FeedforwardFrameTask(FrameTask):
    """Pink frame task with a body-twist feedforward displacement."""

    def __init__(self, *args, **kwargs) -> None:
        self._feedforward_displacement = np.zeros(6)
        super().__init__(*args, **kwargs)

    def set_target(self, transform_target_to_world: pin.SE3) -> None:
        super().set_target(transform_target_to_world)
        self._feedforward_displacement = np.zeros(6)

    def set_target_motion(
        self,
        transform_target_to_world: pin.SE3,
        body_twist: np.ndarray,
        dt: float,
    ) -> None:
        body_twist = np.asarray(body_twist, dtype=float)
        if body_twist.shape != (6,) or not np.all(np.isfinite(body_twist)):
            raise ValueError("frame feedforward twist must be a finite 6-vector")
        super().set_target(transform_target_to_world)
        self._feedforward_displacement = body_twist * float(dt)

    def compute_error(self, configuration: pink.Configuration) -> np.ndarray:
        error = super().compute_error(configuration)
        if self.gain <= 0.0:
            return error
        return error + self._feedforward_displacement / self.gain

    def compute_geometric_error(self, configuration: pink.Configuration) -> np.ndarray:
        """Return pose error without the feedforward displacement."""

        return super().compute_error(configuration)


class TrackingDampingTask(DampingTask):
    """Penalize reference velocity plus measured tracking-velocity error."""

    lift_tracking_displacement = 0.0

    def compute_error(self, configuration):
        error = super().compute_error(configuration)
        error[0] = self.lift_tracking_displacement
        return error


class WholeBodyDampingTask(DampingTask):
    """Include planar base DOFs excluded by Pink's floating-base damping."""

    def compute_error(self, configuration):
        return np.zeros(configuration.model.nv)

    def compute_jacobian(self, configuration):
        return np.eye(configuration.model.nv)


class JerkLimit(Limit):
    """Affine per-axis jerk envelope on Pink's displacement decision."""

    def __init__(self, model: pin.Model, jerk_limit: np.ndarray) -> None:
        jerk_limit = np.asarray(jerk_limit, dtype=float).reshape(-1)
        if jerk_limit.shape != (model.nv,) or np.any(~np.isfinite(jerk_limit)) or np.any(jerk_limit <= 0.0):
            raise ValueError(f"jerk limit must be a finite positive ({model.nv},) vector")
        self.model = model
        self.jerk_limit = jerk_limit
        self.projection_matrix = np.eye(model.nv)
        self.previous_velocity = np.zeros(model.nv)
        self.previous_acceleration = np.zeros(model.nv)

    def reset(
        self,
        velocity: np.ndarray | None = None,
        acceleration: np.ndarray | None = None,
    ) -> None:
        velocity = np.zeros(self.model.nv) if velocity is None else np.asarray(velocity, dtype=float)
        acceleration = (
            np.zeros(self.model.nv) if acceleration is None else np.asarray(acceleration, dtype=float)
        )
        if velocity.shape != (self.model.nv,) or not np.all(np.isfinite(velocity)):
            raise ValueError("jerk-limit reset velocity has the wrong shape or non-finite values")
        if acceleration.shape != (self.model.nv,) or not np.all(np.isfinite(acceleration)):
            raise ValueError("jerk-limit reset acceleration has the wrong shape or non-finite values")
        self.previous_velocity = velocity.copy()
        self.previous_acceleration = acceleration.copy()

    def set_last_integration(self, velocity: np.ndarray, dt: float) -> None:
        velocity = np.asarray(velocity, dtype=float)
        if velocity.shape != (self.model.nv,) or not np.all(np.isfinite(velocity)):
            raise ValueError("jerk-limit velocity has the wrong shape or non-finite values")
        acceleration = (velocity - self.previous_velocity) / float(dt)
        self.previous_velocity = velocity.copy()
        self.previous_acceleration = acceleration

    def compute_qp_inequalities(
        self,
        configuration: pink.Configuration,
        dt: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        del configuration
        centre_velocity = self.previous_velocity + self.previous_acceleration * dt
        velocity_radius = self.jerk_limit * dt**2
        upper_displacement = dt * (centre_velocity + velocity_radius)
        lower_displacement = dt * (centre_velocity - velocity_radius)
        return (
            np.vstack((self.projection_matrix, -self.projection_matrix)),
            np.concatenate((upper_displacement, -lower_displacement)),
        )


class PinkIKCore:
    """Closed-loop IK using named robot models and measured joint state only."""

    def _solve_active_problem(self, problem, **options):
        from .qp_pipeline import reduced_problem
        self._solved_problem = (reduced_problem(problem, self.active_indices)
                                if len(self.active_indices) != self.model.nv else problem)
        result = qpsolvers.solve_problem(self._solved_problem, solver=self.config.solver, **options)
        if result.x is not None and len(self.active_indices) != self.model.nv:
            expanded = np.zeros(self.model.nv)
            expanded[self.active_indices] = result.x
            result.x = expanded
        return result

    def __init__(
        self,
        urdf_path: Path | None = None,
        config: PinkControllerConfig | None = None,
        nominal_pose: NullPose | None = None,
        *,
        base_mode: str = "fixed",
        locked_joint_names: tuple[str, ...] = (),
        runtime_position_bounds: Mapping[str, tuple[float, float]] | None = None,
        runtime_velocity_bounds: Mapping[str, float] | None = None,
        runtime_effort_limits_nm: tuple[float, ...] | None = None,
        allow_upper_lift_retreat: bool = False,
        measured_pose_tracking: bool = False,
    ) -> None:
        config = config or PinkControllerConfig()
        self.config = config
        if type(measured_pose_tracking) is not bool:
            raise ValueError('measured_pose_tracking must be boolean')
        self.measured_pose_tracking = measured_pose_tracking
        self._measured_tracking_active = False
        from .measured_tracking import MeasuredPoseTracking
        self.measured_tracker = MeasuredPoseTracking(config) if measured_pose_tracking else None
        if base_mode not in {"fixed", "planar"}:
            raise ValueError("base_mode must be fixed or planar")
        self.base_mode = base_mode
        self.active_indices = np.arange(3 if base_mode == "planar" and not config.base_control_enabled else 0,
                                        18 if base_mode == "planar" else 15)
        self._base_dofs = 3 if base_mode == "planar" else 0
        if config.solver not in qpsolvers.available_solvers:
            raise RuntimeError(
                f"Pink QP solver {config.solver!r} is unavailable; installed solvers: "
                f"{', '.join(qpsolvers.available_solvers) or 'none'}"
            )
        self.robot_model = load_robot_model(config.robot_model)
        self.auxiliary_joints = self.robot_model.auxiliary_joints
        self.tcp_offset = self.robot_model.tcp_offset_m
        self.model = self._build_pinocchio_model(Path(urdf_path or self.robot_model.urdf))
        self.effective_mit_servo = config.mit_servo
        if runtime_effort_limits_nm is not None:
            effort = np.asarray(runtime_effort_limits_nm, dtype=float)
            if (config.mit_servo is None or effort.shape != (14,)
                    or not np.all(np.isfinite(effort) & (effort > 0))):
                raise ValueError("runtime MIT effort limits require fourteen positive finite bounds")
            self.effective_mit_servo = replace(
                config.mit_servo,
                torque_limit_nm=tuple(np.minimum(effort, config.mit_servo.torque_limit_nm)),
            )
        for name, bound in (runtime_velocity_bounds or {}).items():
            if not np.isfinite(bound) or bound <= 0:
                raise ValueError(f"invalid runtime velocity bound: {name}")
            if base_mode == "planar" and name in BASE_JOINTS:
                index = BASE_JOINTS.index(name)
            else:
                joint_id = self.model.getJointId(name)
                if joint_id >= self.model.njoints or self.model.joints[joint_id].nv != 1:
                    raise ValueError(f"invalid runtime velocity joint: {name}")
                index = self.model.joints[joint_id].idx_v
            self.model.velocityLimit[index] = min(self.model.velocityLimit[index], bound)
        # Device-reported physical ranges may only narrow the profile domain.
        # Apply before constructing any limit/collision/gravity model copies.
        for name, bounds in (runtime_position_bounds or {}).items():
            joint_id = self.model.getJointId(name)
            values = np.asarray(bounds, dtype=float)
            if (joint_id >= self.model.njoints or self.model.joints[joint_id].nq != 1
                    or values.shape != (2,) or not np.all(np.isfinite(values)) or values[0] >= values[1]):
                raise ValueError(f"invalid runtime physical position bounds: {name}")
            index = self.model.joints[joint_id].idx_q
            low_margin = self.model.lowerPositionLimit[index] - self._physical_lower_position_limit[index]
            high_margin = self._physical_upper_position_limit[index] - self.model.upperPositionLimit[index]
            self._physical_lower_position_limit[index] = max(self._physical_lower_position_limit[index], values[0])
            self._physical_upper_position_limit[index] = min(self._physical_upper_position_limit[index], values[1])
            self.model.lowerPositionLimit[index] = self._physical_lower_position_limit[index] + low_margin
            self.model.upperPositionLimit[index] = self._physical_upper_position_limit[index] - high_margin
            if self.model.lowerPositionLimit[index] >= self.model.upperPositionLimit[index]:
                raise ValueError(f"device/profile position ranges do not overlap: {name}")
        # The measured-lift boundary check keeps the planning bound; the reference itself
        # stays LIFT_REFERENCE_TOP_RESERVE_M lower so position-servo overshoot at the top
        # cannot trip that check. Only planning is tightened; no check is relaxed.
        self._lift_measured_upper = float(self.model.upperPositionLimit[3])
        if config.mit_servo is not None and base_mode == "planar":
            self.model.upperPositionLimit[3] -= LIFT_REFERENCE_TOP_RESERVE_M
            if self.model.lowerPositionLimit[3] >= self.model.upperPositionLimit[3]:
                raise ValueError("lift reference reserve eliminates the lift interval")
        self._normal_lift_upper = None
        if allow_upper_lift_retreat:
            if base_mode != "planar" or config.mit_servo is None:
                raise ValueError("upper lift retreat requires planar MIT")
            self._normal_lift_upper = float(self.model.upperPositionLimit[3])
            self.model.upperPositionLimit[3] = self._physical_upper_position_limit[3]
        # Commissioning locks are hard displacement equalities, not posture
        # costs. Keep them in the hard problem reused by every braking retry.
        if len(set(locked_joint_names)) != len(locked_joint_names):
            raise ValueError("duplicate locked joint names")
        locked_indices = []
        for name in locked_joint_names:
            if base_mode == "planar" and name in BASE_JOINTS:
                # Pinocchio represents these named runtime coordinates as
                # one 3-DoF composite root rather than three named joints.
                locked_indices.append(BASE_JOINTS.index(name))
                continue
            joint_id = self.model.getJointId(name)
            if joint_id >= self.model.njoints or self.model.joints[joint_id].nv != 1:
                raise ValueError(f"unknown or nonscalar locked joint: {name}")
            locked_indices.append(self.model.joints[joint_id].idx_v)
        if locked_indices and config.mit_servo is None:
            raise ValueError("commissioning locks require the fail-closed MIT path")
        self.locked_joint_names = tuple(locked_joint_names)
        self._locked_indices = np.asarray(locked_indices, dtype=int)
        self._lock_rows = np.eye(self.model.nv)[self._locked_indices]
        self.active_indices = np.setdiff1d(self.active_indices, self._locked_indices)
        self.data = self.model.createData()
        self._diagnostic_data = self.model.createData()
        def collision_system():
            return SelfCollisionSystem(
                self.model,
                config.self_collision_config,
                require_approved=config.self_collision_require_approved,
                envelope_config=config.self_collision_envelope_config,
                motion_profile_id=config.motion_profile_id,
                motion_profile_sha256=config.motion_profile_sha256,
                pair_certificate_config=config.self_collision_pair_certificate_config,
                pair_certificate_sha256=config.self_collision_pair_certificate_sha256,
                runtime=config.self_collision_runtime,
                native_kernel_abi=config.native_collision_kernel_abi,
            )

        self.self_collision_system = collision_system() if config.self_collision_enabled else None
        self._reference_collision_system = (
            collision_system()
            if self.self_collision_system is not None and config.mit_servo is not None
            else self.self_collision_system
        )
        self._collision_sweep_system = (
            collision_system()
            if self.self_collision_system is not None
            and config.boundary_policy == "jerk_stoppable_directional_v2"
            else None
        )
        self.home_q = self._named_pinocchio_configuration(READY_QPOS)
        self.nominal_q = self.home_q.copy()
        if nominal_pose is not None:
            self.nominal_q = self._named_pinocchio_configuration(
                nominal_pose.joint_positions(LIFT_JOINT, LEFT_ARM_JOINTS, RIGHT_ARM_JOINTS)
            )
            if np.any(self.nominal_q < self.model.lowerPositionLimit) or np.any(
                self.nominal_q > self.model.upperPositionLimit
            ):
                raise ValueError("nominal null pose is outside controller planning limits")

        from .precision_tasks import PrecisionFrameTask, EffortHeadroomTask
        precision = config.target_planner == "c2_tracking_v6" or config.tracking_objective == "normalized_v1"
        frame_type = PrecisionFrameTask if precision else FeedforwardFrameTask
        self.frame_tasks = {
            side: frame_type(
                frame,
                position_cost=config.position_cost,
                orientation_cost=config.orientation_cost,
                lm_damping=config.frame_lm_damping,
                gain=config.frame_gain,
                **({"orientation_gain": config.orientation_gain} if precision else {}),
            )
            for side, frame in TCP_FRAMES.items()
        }
        self.posture_task = PostureTask(cost=config.arm_posture_cost, gain=1.0)
        self.posture_task.cost = np.array([config.lift_posture_cost, *([0.0] * 14)], dtype=float)
        self.posture_task.set_target(self.nominal_q)
        arm_velocity_indices = {
            side: np.asarray(
                [
                    self.model.joints[self.model.getJointId(name)].idx_v
                    for name in (LEFT_ARM_JOINTS if side == "left" else RIGHT_ARM_JOINTS)
                ],
                dtype=int,
            )
            for side in TCP_FRAMES
        }
        self.arm_kinematics_cache = ArmKinematicsCache(
            self.model,
            frame_names=TCP_FRAMES,
            velocity_indices=arm_velocity_indices,
            angular_length_scale_m=config.arm_manipulability_angular_length_scale_m,
            rank_rtol=config.arm_jacobian_rank_rtol,
        )
        self.arm_posture_task = ArmNullspacePostureTask(
            self.model,
            frame_names=TCP_FRAMES,
            arm_joint_names={"left": LEFT_ARM_JOINTS, "right": RIGHT_ARM_JOINTS},
            target_q=self.nominal_q,
            cost=np.full(14, config.arm_posture_cost, dtype=float),
            gain=1.0,
            kinematics_cache=self.arm_kinematics_cache,
        )
        self.arm_manipulability_tasks = {
            side: ArmNullspaceManipulabilityTask(
                self.model,
                frame_name=frame,
                arm_joint_names=(LEFT_ARM_JOINTS if side == "left" else RIGHT_ARM_JOINTS),
                reference_q=self.nominal_q,
                cost=config.arm_manipulability_cost,
                rate=config.arm_manipulability_rate,
                floor_fraction=config.arm_manipulability_floor_fraction,
                exit_fraction=config.arm_manipulability_exit_fraction,
                critical_sigma_fraction=config.arm_manipulability_critical_sigma_fraction,
                activation_ramp_s=config.arm_manipulability_activation_ramp_s,
                angular_length_scale_m=config.arm_manipulability_angular_length_scale_m,
                side=side,
                kinematics_cache=self.arm_kinematics_cache,
            )
            for side, frame in TCP_FRAMES.items()
        }
        self.base_anchor_task: FrameTask | None = None
        if self.base_mode == "planar" and config.base_control_enabled:
            self.base_anchor_task = FrameTask(
                BASE_FRAME,
                position_cost=[config.base.anchor_position_cost, config.base.anchor_position_cost, 0.0],
                orientation_cost=[0.0, 0.0, config.base.anchor_yaw_cost],
                lm_damping=config.frame_lm_damping,
                gain=config.base_anchor_gain,
            )
        if config.tracking_objective == "normalized_v1":
            from .precision_tasks import NormalizedAccelerationTask
            self.low_acceleration_task = NormalizedAccelerationTask(
                config.low_acceleration_cost, np.r_[config.lift_acceleration_limit,
                                                     config.resolved_arm_limits("acceleration")], config.control_dt)
        else:
            self.low_acceleration_task = LowAccelerationTask(cost=config.low_acceleration_cost)
        self.lift_damping_task = (TrackingDampingTask(cost=0.0) if config.allocator.mode in ADAPTIVE_SHARE_MODES
                                  else DampingTask(cost=0.0))
        self.lift_damping_task.cost = np.zeros(15)
        self.tasks = [*self.frame_tasks.values()]
        self.stationary_damping_task = WholeBodyDampingTask(cost=0.)
        if config.stationary_redundancy_hold:
            self.tasks.append(self.stationary_damping_task)
        if self.base_anchor_task is not None:
            self.tasks.append(self.base_anchor_task)
        self.tasks.extend(
            (
                self.posture_task,
                self.lift_damping_task,
                self.arm_posture_task,
                *self.arm_manipulability_tasks.values(),
                self.low_acceleration_task,
            )
        )
        self.configuration_limit = ConfigurationLimit(self.model, config_limit_gain=0.9)
        self.velocity_limit = VelocityLimit(self.model)
        self._acceleration_limits = np.array(
            [
                *(
                    (
                        config.base.linear_acceleration_limit,
                        config.base.linear_acceleration_limit,
                        config.base.angular_acceleration_limit,
                    )
                    if self.base_mode == "planar"
                    else ()
                ),
                config.lift_acceleration_limit,
                *config.resolved_arm_limits("acceleration"),
            ],
            dtype=float,
        )
        self._jerk_limits = np.array(
            [
                *(
                    (
                        config.base.linear_jerk_limit,
                        config.base.linear_jerk_limit,
                        config.base.angular_jerk_limit,
                    )
                    if self.base_mode == "planar"
                    else ()
                ),
                config.lift_jerk_limit,
                *config.resolved_arm_limits("jerk"),
            ],
            dtype=float,
        )
        self.acceleration_limit = AccelerationLimit(self.model, self._acceleration_limits)
        self.jerk_limit = JerkLimit(self.model, self._jerk_limits)
        self.limits = [self.configuration_limit, self.velocity_limit, self.acceleration_limit, self.jerk_limit]
        if config.mit_servo is not None:
            from .command_limits import JerkSafeVelocityLimit

            fractions = np.full(self.model.nv, config.reference_velocity_fraction)
            fractions[self._base_dofs] = config.lift_reference_velocity_fraction
            self.limits.append(JerkSafeVelocityLimit(
                self.model, self.jerk_limit, self._acceleration_limits, config.control_dt,
                fractions))
        from .motion_profile import load_motion_profile
        self._base_resultant_constraint = bool(config.motion_profile_id and load_motion_profile(
            config.motion_profile_path or config.motion_profile_id).velocity_limit_authority == "manufacturer")
        if self.base_mode == "planar" and self._base_resultant_constraint:
            from .command_limits import PlanarResultantVelocityLimit
            self.limits.append(PlanarResultantVelocityLimit(
                self.model, self.jerk_limit, self._acceleration_limits, config.control_dt,
                config.reference_velocity_fraction))
        from .hardware_capabilities import load_hardware_capabilities
        self._manufacturer_capabilities = load_hardware_capabilities()
        self.gravity_model = (
            JointStateGravity(
                self._gravity_source_model, self.model, self.auxiliary_joints,
                self._gravity_nominal_q, config.gravity_payloads,
                correction=config.mit_gravity_correction, arm_joint_names=ARM_JOINTS,
            ) if config.mit_servo is not None else None
        )
        self.reference_limit = (
            JointReferenceLimit(self.model, self._base_dofs + 1, self.effective_mit_servo,
                                self._acceleration_limits, self._jerk_limits,
                                planning_effort_fraction=config.mit_planning_effort_fraction,
                                closed_form=config.boundary_policy in {
                                    "directional_velocity_damper_v1",
                                    "jerk_stoppable_directional_v2",
                                })
            if config.mit_servo is not None else None
        )
        if self.reference_limit is not None:
            self.limits.append(self.reference_limit)
        self.effort_preview = None
        if config.mit_effort_preview == "jerk_prefix_v1":
            from .command_limits import JerkEffortPreviewLimit
            self.effort_preview = JerkEffortPreviewLimit(self.reference_limit)
            self.limits.append(self.effort_preview)
        arm_indices = np.arange(self._base_dofs + 1, self.model.nv)
        self._joint_boundary_governor = (
            JointBoundaryGovernor(
                self.model,
                (np.arange(self._base_dofs, self.model.nv)
                 if (config.motion_envelope == "first-trial" or config.lift_reference_max_error_m is not None)
                 else arm_indices),
                self._acceleration_limits, self._jerk_limits
            )
            if config.boundary_policy == "jerk_stoppable_directional_v2"
            else None
        )
        from .command_limits import FiniteBaseBoundaryGovernor

        self._base_boundary_governor = (
            FiniteBaseBoundaryGovernor(
                self.model, np.arange(3), self._acceleration_limits, self._jerk_limits
            ) if self.base_mode == "planar" and config.base.workspace_mode == "bounded"
            and config.base.boundary_governor == "finite_base_v1" else None
        )
        from .lift_reference import LiftReferenceGovernor

        # A locked lift must keep its reset reference even in profiles whose
        # moving lift normally integrates from measured feedback. Otherwise a
        # zero-displacement equality follows encoder drift instead of holding.
        persistent_lift = (config.lift_reference_max_error_m is not None
                           or self._base_dofs in self._locked_indices)
        self._persistent_start = self._base_dofs + (0 if persistent_lift else 1)
        self._lift_reference_governor = (
            LiftReferenceGovernor(self.model, self._base_dofs, self._acceleration_limits,
                                  self._jerk_limits, config.lift_reference_max_error_m)
            if config.lift_reference_max_error_m is not None else None)
        # Measured-origin lift commands still retain reference acceleration and
        # jerk history. Pink's continuous sqrt(2*a*distance) bound alone cannot
        # keep that history feasible against the next discrete position bound.
        # A hard-locked lift does not integrate from measured feedback and cannot
        # execute the governor's recovery motion. Keep its persistent reference
        # and lock equality instead. Physical range/velocity validation remains
        # active; the native fixed-lift owner independently supervises the brake,
        # measured drift and feedback age. Do not apply this exception to a
        # movable lift merely because its current requested velocity is zero.
        self._measured_lift_boundary_governor = (
            FiniteBaseBoundaryGovernor(self.model, [self._base_dofs],
                                       self._acceleration_limits, self._jerk_limits)
            if (config.mit_servo is not None and self._lift_reference_governor is None
                and self._base_dofs not in self._locked_indices) else None)
        if self._measured_lift_boundary_governor is not None:
            self._measured_lift_boundary_governor.labels = ("lift_measured_lower", "lift_measured_upper")
        if precision or config.effort_regularization_cost > 0:
            self.tasks.append(EffortHeadroomTask(self.reference_limit, config.effort_regularization_cost,
                                                 config.maximum_feedback_age_s + config.maximum_command_delay_s))
        self._preparation_limit = None
        self._gravity_effort = np.zeros(self.model.nv)
        self.last_mit_effort = None

        self.configuration = pink.Configuration(
            self.model,
            self.data,
            self.home_q,
            copy_data=False,
        )
        self._measured_configuration = (
            pink.Configuration(self.model, self.model.createData(), self.home_q, copy_data=False)
            if self.reference_limit is not None else self.configuration
        )
        self._collision_sweep_configuration = (
            pink.Configuration(self.model, self.model.createData(), self.home_q, copy_data=False)
            if self._collision_sweep_system is not None
            else None
        )
        self.barriers = []
        self._limited_targets: dict[str, pin.SE3] = {}
        target_motion_limits = CartesianMotionLimits(
            linear_velocity_m_s=config.target_linear_velocity_limit,
            linear_acceleration_m_s2=config.target_linear_acceleration_limit,
            linear_jerk_m_s3=config.target_linear_jerk_limit,
            angular_velocity_rad_s=config.target_angular_velocity_limit,
            angular_acceleration_rad_s2=config.target_angular_acceleration_limit,
            angular_jerk_rad_s3=config.target_angular_jerk_limit,
        )
        planner_type = {"c2_tracking_v6": PrecisionTargetMotionPlanner,
                        "c2_tracking_v5": GuardedTargetMotionPlanner,
                        "c2_tracking_v4": ContinuousTargetMotionPlanner}.get(
                            config.target_planner, C2TargetMotionPlanner)
        self._target_planners = (
            {
                side: planner_type(
                    target_motion_limits,
                    continuous_tracking=config.target_planner
                    in {"c2_tracking_v2", "c2_tracking_v3", "c2_tracking_v4", "c2_tracking_v5", "c2_tracking_v6"},
                    translation_deadband_m=config.target_translation_deadband,
                    orientation_deadband_rad=config.target_orientation_deadband,
                    stationary_dwell_s=config.target_stationary_dwell,
                    limit_check_runtime=config.target_limit_check_runtime,
                    **({"tracking_bandwidth_s_inv": config.target_tracking_bandwidth_s_inv,
                        "source_velocity_gain": config.target_source_velocity_gain,
                        "source_acceleration_gain": config.target_source_acceleration_gain,
                        "source_acceleration_tau_s": config.target_source_acceleration_tau_s,
                        "continuous_arrival": config.stationary_arrival == "continuous_v1"}
                       if issubclass(planner_type, ContinuousTargetMotionPlanner) else {}),
                )
                for side in TCP_FRAMES
            }
            if config.target_planner != "legacy_rate_clip_v1"
            else {}
        )
        self._last_safe_positions = self.home_q.copy()
        from .following_error import FollowingErrorGovernor, governed_profile
        self.following_governor = (FollowingErrorGovernor() if self.reference_limit is not None and not measured_pose_tracking
                                   and governed_profile(config.motion_profile_id) else None)
        self._last_velocity = np.zeros(self.model.nv)
        self._last_acceleration = np.zeros(self.model.nv)
        self._last_jerk = np.zeros(self.model.nv)
        self._last_measured_q: np.ndarray | None = None
        self._last_measured_velocity: np.ndarray | None = None
        self._last_state_timestamp: float | None = None
        self._allow_initial_timestamp_repeat = False
        self._recovering = False
        self._transient_recovery_reason: str | None = None
        self._mit_fault_latched: str | None = None
        self._fault_info = (None, None)
        self._last_control_timestamp: float | None = None
        self._stop_requested = False
        self.control_state = "HOLD"
        self._stop_reason: str | None = None
        self._feedback_age_s: float | None = None
        self._dispatch_state_age_s: float | None = None
        self._deadline_reject_count = 0
        self._hold_since_s: float | None = None
        self.target_planner_failures = 0
        self.last_target_planner_error = None
        self.last_qp_failure: QPFailureSnapshot | None = None
        self._large_error_time = 0.0
        self._base_anchor_target: pin.SE3 | None = None
        self._lift_anchor_position = float(self.home_q[self._base_dofs])
        self._base_activation = 0.0
        self._lift_activation = 0.0
        self._base_engaged = False
        self._lift_engaged = False
        self._base_enter_elapsed_s = 0.0
        self._lift_enter_elapsed_s = 0.0
        self._base_release_elapsed_s = 0.0
        self._lift_release_elapsed_s = 0.0
        self._base_activation_reason = "none"
        self._lift_activation_reason = "none"
        self._settling_blend = 0.0
        self._stationary_cost_blend = 0.0
        self._allocation_reference_targets: dict[str, pin.SE3] = {}
        self._previous_allocation_targets: dict[str, pin.SE3] = {}
        self._adaptive_share = AdaptiveShare()
        self._allocation_diagnostics = WholeBodyAllocationDiagnostics(
            common_translation_error=(0.0, 0.0, 0.0),
            differential_translation_error=(0.0, 0.0, 0.0),
            base_activation=0.0,
            lift_activation=0.0,
            arm_pressure=0.0,
            max_arm_joint_pressure=0.0,
            arm_joint_pressures={name: 0.0 for name in ARM_JOINTS},
            base_weights=(
                (
                    (
                        config.allocator.base_inactive_cost
                        if config.allocator.mode != "continuous_v1"
                        else config.base.anchor_position_cost
                    ),
                    (
                        config.allocator.base_inactive_cost
                        if config.allocator.mode != "continuous_v1"
                        else config.base.anchor_position_cost
                    ),
                    (
                        config.allocator.base_inactive_cost
                        if config.allocator.mode != "continuous_v1"
                        else config.base.anchor_yaw_cost
                    ),
                )
                if self.base_mode == "planar"
                else None
            ),
            lift_weight=(
                config.allocator.lift_inactive_cost
                if config.allocator.mode != "continuous_v1"
                else config.lift_posture_cost
            ),
            arm_weights={name: config.arm_posture_cost for name in ARM_JOINTS},
            mode=config.allocator.mode,
            base_anchor_position=None,
            lift_anchor_position=float(self.home_q[self._base_dofs]),
            allocation_reference_translation={},
        )
        self._initialized = False

    def _build_pinocchio_model(self, urdf_path: Path) -> pin.Model:
        if (self.config.gravity_urdf_sha256 is not None
                and hashlib.sha256(urdf_path.read_bytes()).hexdigest() != self.config.gravity_urdf_sha256):
            raise ValueError("nominal gravity URDF does not match the profile hash")
        if self.base_mode == "planar":
            root_joint = pin.JointModelComposite()
            for joint in (pin.JointModelPX(), pin.JointModelPY(), pin.JointModelRZ()):
                root_joint.addJoint(joint)
            full_model = pin.buildModelFromUrdf(str(urdf_path.resolve()), root_joint)
        else:
            full_model = pin.buildModelFromUrdf(str(urdf_path.resolve()))
        reference = pin.neutral(full_model)
        for joint_name, value in zip(self.auxiliary_joints, self.robot_model.fixed_auxiliary_positions, strict=True):
            joint_id = full_model.getJointId(joint_name)
            if joint_id == 0:
                raise RuntimeError(f"URDF is missing gripper joint {joint_name!r}")
            joint = full_model.joints[joint_id]
            reference[joint.idx_q] = value
        lock_ids = [full_model.getJointId(name) for name in self.auxiliary_joints]
        model = pin.buildReducedModel(full_model, lock_ids, reference)

        actual_names = tuple(model.names[joint_id] for joint_id in range(1, model.njoints))
        expected_names = ("root_joint", *CONTROLLED_JOINTS) if self.base_mode == "planar" else CONTROLLED_JOINTS
        expected_dofs = 18 if self.base_mode == "planar" else 15
        if model.nq != expected_dofs or model.nv != expected_dofs or actual_names != expected_names:
            raise RuntimeError(
                f"Unexpected reduced Pinocchio model: nq={model.nq}, nv={model.nv}, joints={actual_names}"
            )

        if self.base_mode == "planar":
            model.lowerPositionLimit[:3] = (
                self.config.base.x_limits_m[0],
                self.config.base.y_limits_m[0],
                self.config.base.yaw_limits_rad[0],
            )
            model.upperPositionLimit[:3] = (
                self.config.base.x_limits_m[1],
                self.config.base.y_limits_m[1],
                self.config.base.yaw_limits_rad[1],
            )
        if self.base_mode == "planar" and self.config.base.workspace_mode == "unbounded":
            model.lowerPositionLimit[:3] = -np.inf
            model.upperPositionLimit[:3] = np.inf
        if self.config.arm_position_lower_limits is not None:
            arm_start = self._base_dofs + 1
            model.lowerPositionLimit[arm_start:] = np.maximum(
                model.lowerPositionLimit[arm_start:],
                np.asarray(self.config.arm_position_lower_limits),
            )
            model.upperPositionLimit[arm_start:] = np.minimum(
                model.upperPositionLimit[arm_start:],
                np.asarray(self.config.arm_position_upper_limits),
            )
        margins = np.array(
            [
                *(
                    (
                        self.config.base.position_margin_m,
                        self.config.base.position_margin_m,
                        self.config.base.yaw_margin_rad,
                    )
                    if self.base_mode == "planar"
                    else ()
                ),
                self.config.lift_position_margin,
                *([self.config.arm_position_margin] * 14),
            ]
        )
        self._physical_lower_position_limit = model.lowerPositionLimit.copy()
        self._physical_upper_position_limit = model.upperPositionLimit.copy()
        model.lowerPositionLimit = model.lowerPositionLimit + margins
        model.upperPositionLimit = model.upperPositionLimit - margins
        if np.any(model.lowerPositionLimit >= model.upperPositionLimit):
            raise RuntimeError("Configured joint margins eliminate a valid position interval")
        model.velocityLimit = np.array(
            [
                *(
                    (
                        self.config.base.linear_velocity_limit,
                        self.config.base.linear_velocity_limit,
                        self.config.base.angular_velocity_limit,
                    )
                    if self.base_mode == "planar"
                    else ()
                ),
                self.config.lift_velocity_limit,
                *self.config.resolved_arm_limits("velocity"),
            ],
            dtype=float,
        )

        for side, link_frame_name in (
            ("left", "maver_x4_asm_7_urdf_SLDASM_link_7"),
            ("right", "maver_x4_asm_7_urdf_SLDASM2_link_7"),
        ):
            frame_id = model.getFrameId(link_frame_name)
            if frame_id >= model.nframes:
                raise RuntimeError(f"URDF is missing end-effector frame {link_frame_name!r}")
            link_frame = model.frames[frame_id]
            parent_joint = (
                link_frame.parentJoint if hasattr(link_frame, "parentJoint") else link_frame.parent
            )
            parent_frame = (
                link_frame.parentFrame if hasattr(link_frame, "parentFrame") else link_frame.previousFrame
            )
            placement = link_frame.placement * pin.SE3(np.eye(3), np.asarray(self.tcp_offset))
            tcp_frame = pin.Frame(
                TCP_FRAMES[side],
                parent_joint,
                parent_frame,
                placement,
                pin.FrameType.OP_FRAME,
            )
            model.addFrame(tcp_frame, False)
            full_link = full_model.frames[full_model.getFrameId(link_frame_name)]
            full_parent = full_link.parentJoint if hasattr(full_link, "parentJoint") else full_link.parent
            full_previous = full_link.parentFrame if hasattr(full_link, "parentFrame") else full_link.previousFrame
            full_model.addFrame(pin.Frame(
                TCP_FRAMES[side], full_parent, full_previous,
                full_link.placement * pin.SE3(np.eye(3), np.asarray(self.tcp_offset)),
                pin.FrameType.OP_FRAME,
            ), False)
        self._gravity_source_model = full_model
        self._gravity_nominal_q = reference.copy()
        return model

    def _named_pinocchio_configuration(self, values: Mapping[str, float]) -> np.ndarray:
        q = pin.neutral(self.model)
        for joint_name in CONTROLLED_JOINTS:
            joint_id = self.model.getJointId(joint_name)
            joint = self.model.joints[joint_id]
            if joint.nq != 1:
                raise RuntimeError(f"Unsupported Pinocchio joint layout for {joint_name!r}")
            q[joint.idx_q] = values[joint_name]
        return q

    def reset(
        self,
        measured_q: np.ndarray,
        targets: Mapping[str, pin.SE3],
        *,
        measured_velocity: np.ndarray | None = None,
        measured_gripper_positions: np.ndarray | None = None,
        state_timestamp: float | None = None,
        stationary_reference: bool = False,
    ) -> None:
        self._last_feedback_received = None
        self._feedback_position_uncertainty = np.zeros(self.model.nv)
        self._validate_targets(targets)
        raw_measured_q = np.asarray(measured_q, dtype=float).reshape(-1).copy()
        measured_q = self._validate_configuration_vector(measured_q, recover_margin_excursion=True)
        if measured_velocity is None:
            measured_velocity = np.zeros(self.model.nv)
        measured_velocity = self._validate_velocity_vector(measured_velocity)
        if self.config.mit_servo is not None and np.any(
            np.abs(measured_velocity) > self.model.velocityLimit + self.config.constraint_tolerance
        ):
            raise ValueError("cannot initialize an MIT reference from measured overspeed")
        if self._base_dofs and self._base_resultant_constraint and np.linalg.norm(measured_velocity[:2]) > (
                min(self.model.velocityLimit[:2]) + self.config.constraint_tolerance):
            raise ValueError("cannot initialize an MIT reference from base resultant overspeed")
        reference_velocity = measured_velocity.copy()
        if stationary_reference:
            from .reference_initialization import stationary_reference_velocity

            reference_velocity = stationary_reference_velocity(measured_velocity, base_dofs=self._base_dofs)
        if state_timestamp is not None and (not np.isfinite(state_timestamp) or state_timestamp < 0.0):
            raise ValueError("state timestamp must be finite and non-negative")
        self.configuration.update(measured_q)
        if self._measured_configuration is not self.configuration:
            self._measured_configuration.update(raw_measured_q)
            self._gravity_effort = self.gravity_model.evaluate(raw_measured_q, measured_gripper_positions)
            gravity = self._gravity_effort[self._base_dofs + 1:]
            bounds = (np.asarray(self.effective_mit_servo.torque_limit_nm)
                      * self.config.mit_planning_effort_fraction)
            exceeded = np.flatnonzero(np.abs(gravity) >= bounds)
            if len(exceeded):
                names = ", ".join(f"{ARM_JOINTS[i]} ({gravity[i]:.3f}/{bounds[i]:.3f} Nm)" for i in exceeded)
                raise ValueError(f"initial gravity exhausts MIT operating torque budget: {names}")
        if self.self_collision_system is not None:
            initial_travel = np.zeros(self.model.nv) if self.reference_limit is not None else None
            reference_system = self._reference_collision_system
            self.barriers = list(
                reference_system.update(
                    self.configuration, self.config.control_dt,
                    joint_safety_travel=initial_travel,
                )
            )
            if self.reference_limit is not None:
                self.self_collision_system.update(
                    self._measured_configuration,
                    self.config.control_dt,
                    joint_safety_travel=initial_travel,
                )
            if self._collision_sweep_system is not None:
                self._collision_sweep_system.update(
                    self._measured_configuration,
                    self.config.control_dt,
                    joint_safety_travel=initial_travel,
                )
        self._posture_transition = None
        if self.config.posture_policy is not None:
            target = self.configuration.q.copy()
            target[self._base_dofs + 1:] = self.config.posture_policy.arm_position
            if np.any(target < self.model.lowerPositionLimit) or np.any(target > self.model.upperPositionLimit):
                raise ValueError("human posture outside effective joint bounds")
            self.nominal_q = target.copy()
            self._posture_transition = PostureTransition(
                self.configuration.q, target, self.config.posture_policy.blend_s)
            self.arm_posture_task.set_target(self.configuration.q)
            for task in self.arm_manipulability_tasks.values():
                task.set_reference_configuration(target)
        self._limited_targets = {side: targets[side].copy() for side in TCP_FRAMES}
        for side, planner in self._target_planners.items():
            planner.reset(self._limited_targets[side])
        for side, task in self.frame_tasks.items():
            task.set_target(self._limited_targets[side])
        self._base_activation = 0.0
        self._lift_activation = 0.0
        self._base_engaged = False
        self._lift_engaged = False
        self._base_enter_elapsed_s = 0.0
        self._lift_enter_elapsed_s = 0.0
        self._base_release_elapsed_s = 0.0
        self._lift_release_elapsed_s = 0.0
        self._base_activation_reason = "none"
        self._lift_activation_reason = "none"
        self._settling_blend = 1.0
        self._stationary_cost_blend = 0.0
        self._allocation_reference_targets = {
            side: self._limited_targets[side].copy() for side in TCP_FRAMES
        }
        self._previous_allocation_targets = {
            side: self._limited_targets[side].copy() for side in TCP_FRAMES
        }
        self._lift_anchor_position = float(measured_q[self._base_dofs])
        self._adaptive_share = AdaptiveShare()
        if self.base_anchor_task is not None:
            self._base_anchor_target = self.configuration.get_transform_frame_to_world(BASE_FRAME).copy()
            self.base_anchor_task.set_target(self._base_anchor_target)
        else:
            self._base_anchor_target = None
        self.arm_kinematics_cache.refresh(self.configuration)
        for task in self.arm_manipulability_tasks.values():
            task.reset_activation()
            task.begin_cycle(self.configuration, self.config.control_dt)
        self._update_redundancy_allocation(
            self._limited_targets
            if self.config.allocator.mode != "continuous_v1"
            else targets,
            self.config.control_dt,
        )
        self._last_safe_positions = measured_q.copy()
        if self.following_governor is not None:
            self.following_governor.reset()
        if self.measured_tracker is not None:
            self.measured_tracker.reset()
        self._last_velocity = reference_velocity
        self._last_acceleration = np.zeros(self.model.nv)
        self._last_jerk = np.zeros(self.model.nv)
        self._last_measured_q = measured_q.copy()
        self._last_state_timestamp = state_timestamp
        self._allow_initial_timestamp_repeat = state_timestamp is not None
        self._recovering = False
        self._large_error_time = 0.0
        self.low_acceleration_task.set_last_integration(self._last_velocity[self._base_dofs :], self.config.control_dt)
        self.low_acceleration_task.cost = self.config.settled_low_acceleration_cost
        self.lift_damping_task.cost[1:] = self.config.settled_arm_damping_cost
        self.acceleration_limit.set_last_integration(self._last_velocity, self.config.control_dt)
        self.jerk_limit.reset(self._last_velocity)
        self._initialized = True
        self._last_measured_velocity = measured_velocity.copy()
        self._mit_fault_latched = None
        self._fault_info = (None, None)
        self._last_control_timestamp = None
        self._stop_requested = False
        self._transient_recovery_reason = None
        self.control_state = "TRACKING"
        self._stop_reason = None
        self._feedback_age_s = None
        self._dispatch_state_age_s = None
        self._deadline_reject_count = 0
        self._hold_since_s = None
        self.last_qp_failure = None
        self.last_qp_capture_error = None

    def set_nominal_pose(self, pose: NullPose) -> None:
        """Replace the arm posture reference used by the null-space task."""
        nominal_q = self._named_pinocchio_configuration(
            pose.joint_positions(LIFT_JOINT, LEFT_ARM_JOINTS, RIGHT_ARM_JOINTS)
        )
        if np.any(nominal_q < self.model.lowerPositionLimit) or np.any(
            nominal_q > self.model.upperPositionLimit
        ):
            raise ValueError("nominal null pose is outside controller planning limits")
        self.nominal_q = nominal_q
        self.arm_posture_task.set_target(nominal_q)
        for task in self.arm_manipulability_tasks.values():
            task.set_reference_configuration(nominal_q)

    def limited_targets(self) -> dict[str, pin.SE3]:
        """Return current IK goals (raw viewer poses in measured tracking mode)."""
        if not self._initialized:
            raise RuntimeError("Call reset() before requesting controller targets")
        return {side: target.copy() for side, target in self._limited_targets.items()}

    def request_stop(self) -> None:
        """Brake using the existing history; do not reset a moving reference."""
        if self.config.mit_servo is None:
            raise RuntimeError("explicit constrained stop is an MIT operation")
        self._stop_requested = True
        if self._stop_reason is None:
            self._stop_reason = "requested"

    def reanchor_tracking_tasks(self, measured_q, measured_velocity):
        """Align stationary soft tasks without resetting MIT integration history."""
        if (not self._initialized or self._mit_fault_latched is not None
                or self.config.mit_servo is None or self.control_state != "TRACKING" or self._stop_requested):
            raise ValueError("tracking reanchor requires an initialized, fault-free controller")
        velocity = self._validate_velocity_vector(measured_velocity)
        settled_speed = np.full(self.model.nv, .02)
        settled_speed[self._base_dofs] = .003
        if self._base_dofs:
            settled_speed[:2] = .003
        if np.any(np.abs(velocity) > settled_speed) or np.any(np.abs(self._last_velocity) > settled_speed):
            raise ValueError("tracking reanchor requires stationary feedback and reference")
        q = self._validate_configuration_vector(measured_q).copy()
        measured_targets = self._command_frame_poses(q) if self.measured_pose_tracking else None
        q[self._persistent_start:] = self._last_safe_positions[self._persistent_start:]
        targets = measured_targets if measured_targets is not None else self._command_frame_poses(q)
        self.configuration.update(q)
        if self.config.posture_policy is not None:
            self._posture_transition = PostureTransition(q, self.nominal_q, self.config.posture_policy.blend_s)
        self.arm_posture_task.set_target(q)
        self._lift_anchor_position = float(q[self._base_dofs])
        if self.base_anchor_task is not None:
            self._base_anchor_target = self.configuration.get_transform_frame_to_world(BASE_FRAME).copy()
            self.base_anchor_task.set_target(self._base_anchor_target)
        self._base_activation = self._lift_activation = 0.0
        self._base_engaged = self._lift_engaged = False
        self._base_enter_elapsed_s = self._lift_enter_elapsed_s = 0.0
        self._base_release_elapsed_s = self._lift_release_elapsed_s = 0.0
        self._base_activation_reason = self._lift_activation_reason = "none"
        self._adaptive_share = AdaptiveShare()
        self._limited_targets = {s: p.copy() for s, p in targets.items()}
        self._stationary_cost_blend = 0.0
        self._allocation_reference_targets = {s: p.copy() for s, p in targets.items()}
        self._previous_allocation_targets = {s: p.copy() for s, p in targets.items()}
        for side, planner in self._target_planners.items():
            planner.reset(targets[side])
        for side, task in self.frame_tasks.items():
            task.set_target(targets[side])
        if self.measured_tracker is not None:
            self.measured_tracker.reset()
            self._large_error_time = 0.0
        return targets

    def resume_tracking(self) -> None:
        """Resume only after a completed stop; faults require an explicit reset."""
        if self.control_state != "HOLD" or self._mit_fault_latched is not None:
            raise RuntimeError("tracking may resume only from HOLD")
        if self.measured_tracker is not None:
            self.measured_tracker.reset()
            self._large_error_time = 0.0
        self._limited_targets = self._command_frame_poses(self._last_safe_positions)
        for side, planner in self._target_planners.items():
            planner.reset(self._limited_targets[side])
        for side, task in self.frame_tasks.items():
            task.set_target(self._limited_targets[side])
        self._stop_requested = False
        self._transient_recovery_reason = None
        self.control_state = "TRACKING"
        self._stop_reason = None
        self._hold_since_s = None

    def finish_lift_retreat(self):
        """Restore the artificial margin after both measurement and reference enter it."""
        if self._normal_lift_upper is None:
            return
        if max(self._last_measured_q[3], self._last_safe_positions[3]) > self._normal_lift_upper - .001:
            raise ValueError("lift has not entered the normal planning interval")
        self.model.upperPositionLimit[3] = self._normal_lift_upper
        if self._lift_reference_governor is not None:
            self._lift_reference_governor.upper = self._normal_lift_upper
        old = self.configuration_limit
        self.configuration_limit = ConfigurationLimit(self.model, config_limit_gain=0.9)
        self.limits[self.limits.index(old)] = self.configuration_limit
        self._normal_lift_upper = None

    def _validate_configuration_vector(self, q: np.ndarray, *, recover_margin_excursion: bool = False) -> np.ndarray:
        q = np.asarray(q, dtype=float).reshape(-1)
        if q.shape != (self.model.nq,) or not np.all(np.isfinite(q)):
            raise ValueError(f"Expected a finite ({self.model.nq},) configuration, got {q}")
        outside_margin = np.any(q < self.model.lowerPositionLimit - 1e-6) or np.any(
            q > self.model.upperPositionLimit + 1e-6
        )
        if outside_margin:
            if not recover_margin_excursion:
                raise ValueError("Measured configuration is outside the controller's joint margins")
            if np.any(q < self._physical_lower_position_limit - 1e-6) or np.any(
                q > self._physical_upper_position_limit + 1e-6
            ):
                raise ValueError("Measured configuration is outside the physical joint limits")
            # Contact can push a servo a few milliradians through the artificial
            # planning margin. Solve from the closest safe configuration so
            # the next command drives back inward instead of holding a target
            # that keeps loading the contact forever.
            q = np.clip(q, self.model.lowerPositionLimit, self.model.upperPositionLimit)
        if recover_margin_excursion:
            # Pink's acceleration limit takes a square root of the braking
            # distance to each position bound. Keep its computational state a
            # nanometre/radian inside the artificial planning interval so
            # round-off cannot turn a zero distance into a tiny negative value.
            # This does not modify MuJoCo state or the physical joint limits.
            numerical_inset = 1e-9
            q = np.clip(
                q,
                self.model.lowerPositionLimit + numerical_inset,
                self.model.upperPositionLimit - numerical_inset,
            )
        return q

    def _validate_velocity_vector(self, velocity: np.ndarray) -> np.ndarray:
        velocity = np.asarray(velocity, dtype=float).reshape(-1)
        if velocity.shape != (self.model.nv,) or not np.all(np.isfinite(velocity)):
            raise ValueError(f"Expected a finite ({self.model.nv},) measured velocity")
        tolerance = self.config.constraint_tolerance
        if self.config.mit_servo is None and np.any(np.abs(velocity) > self.model.velocityLimit + tolerance):
            raise ValueError("Measured velocity is outside the configured velocity profile")
        return velocity

    def _braking_velocity(self, dt: float) -> tuple[np.ndarray, np.ndarray]:
        if self.reference_limit is not None:
            from .command_limits import jerk_braking_step

            return jerk_braking_step(self._last_velocity, self._last_acceleration,
                                     self._acceleration_limits, self._jerk_limits, dt)
        desired_acceleration = np.clip(
            -self._last_velocity / dt,
            -self._acceleration_limits,
            self._acceleration_limits,
        )
        acceleration = np.clip(
            desired_acceleration,
            self._last_acceleration - self._jerk_limits * dt,
            self._last_acceleration + self._jerk_limits * dt,
        )
        velocity = self._last_velocity + acceleration * dt
        return velocity, acceleration

    @staticmethod
    def _validate_targets(targets: Mapping[str, pin.SE3]) -> None:
        """Reject malformed target poses before they can contaminate controller state."""
        for side in TCP_FRAMES:
            if side not in targets:
                raise ValueError(f"Missing {side!r} TCP target")
            pose = targets[side]
            try:
                translation = np.asarray(pose.translation, dtype=float)
                rotation = np.asarray(pose.rotation, dtype=float)
            except AttributeError as error:
                raise ValueError(f"{side} TCP target must be a Pinocchio SE3 pose") from error
            if translation.shape != (3,) or not np.all(np.isfinite(translation)):
                raise ValueError(f"{side} TCP target has a non-finite or malformed translation")
            if rotation.shape != (3, 3) or not np.all(np.isfinite(rotation)):
                raise ValueError(f"{side} TCP target has a non-finite or malformed rotation")
            if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-6) or not np.isclose(
                np.linalg.det(rotation), 1.0, atol=1e-6
            ):
                raise ValueError(f"{side} TCP target rotation is not in SO(3)")

    def _limit_target(self, previous: pin.SE3, desired: pin.SE3, dt: float) -> tuple[pin.SE3, bool]:
        delta_position = desired.translation - previous.translation
        distance = float(np.linalg.norm(delta_position))
        max_distance = self.config.target_linear_velocity_limit * dt
        position_scale = min(1.0, max_distance / distance) if distance > 1e-12 else 1.0
        translation = previous.translation + position_scale * delta_position

        rotation_vector = pin.log3(previous.rotation.T @ desired.rotation)
        angle = float(np.linalg.norm(rotation_vector))
        max_angle = self.config.target_angular_velocity_limit * dt
        rotation_scale = min(1.0, max_angle / angle) if angle > 1e-12 else 1.0
        rotation = previous.rotation @ pin.exp3(rotation_scale * rotation_vector)
        limited = position_scale < 1.0 - 1e-12 or rotation_scale < 1.0 - 1e-12
        return pin.SE3(rotation, translation), limited

    def _arm_pressures(self) -> np.ndarray:
        """Return smooth per-arm-joint pressure from the planning limits."""
        start = self._base_dofs + 1
        return arm_limit_pressures(
            self.configuration.q[start:],
            self.model.lowerPositionLimit[start:],
            self.model.upperPositionLimit[start:],
            self.config.allocator,
        )

    @staticmethod
    def _limit_escape_costs(pressures: np.ndarray, arm_weights: np.ndarray) -> np.ndarray:
        """Add a smooth, emergency-only inward bias before hard limits conflict.

        Human-pose regularization remains strictly projected into each TCP's
        null space.  Very near a joint boundary that one-dimensional null
        direction can be poorly aligned with the saturated joint, so this
        separate soft term fades in using the existing continuous pressure.
        It prevents the position and acceleration limits from becoming
        mutually infeasible without relaxing either hard constraint.
        """
        pressure = np.asarray(pressures, dtype=float)
        pressure_gate = pressure * pressure * (3.0 - 2.0 * pressure)
        return np.asarray(arm_weights, dtype=float) * pressure_gate

    def _filtered_activation(self, current: float, target: float, dt: float) -> float:
        return filtered_activation(current, target, dt, self.config.allocator)

    @staticmethod
    def _smooth_band(value: float, lower: float, upper: float) -> float:
        """Map a bounded ergonomic signal to a C1 severity in [0, 1]."""

        return _smoothstep((float(value) - lower) / (upper - lower))

    @staticmethod
    def _hysteresis_axis_state(
        *,
        engaged: bool,
        enter_elapsed_s: float,
        release_elapsed_s: float,
        activation_reason: str,
        distance: float,
        dominance: float,
        pressure: float,
        enter_distance: float,
        exit_distance: float,
        dt: float,
        config: WholeBodyAllocatorConfig,
        assist_demand: bool = False,
        assist_release_ready: bool = True,
        assist_reason: str = "ergonomic_strain",
    ) -> tuple[bool, float, float, str]:
        """Advance one arm-first allocation latch without hidden wall-clock state."""

        cartesian_demand = distance >= enter_distance and dominance >= 0.5
        pressure_demand = (
            pressure >= config.pressure_enter
            and distance >= exit_distance
            and dominance >= 0.25
        )
        if not engaged:
            release_elapsed_s = 0.0
            if pressure_demand:
                return True, 0.0, 0.0, "joint_pressure"
            pending_demand = cartesian_demand or assist_demand
            enter_elapsed_s = enter_elapsed_s + dt if pending_demand else 0.0
            if enter_elapsed_s + 1e-12 >= config.engage_dwell_s:
                reason = "persistent_cartesian_error" if cartesian_demand else assist_reason
                return True, 0.0, 0.0, reason
            return False, enter_elapsed_s, 0.0, "pending" if pending_demand else "none"

        enter_elapsed_s = 0.0
        release_ready = (
            distance <= exit_distance
            and pressure <= config.pressure_exit
            and assist_release_ready
        )
        release_elapsed_s = release_elapsed_s + dt if release_ready else 0.0
        if release_elapsed_s + 1e-12 >= config.release_dwell_s:
            return False, 0.0, 0.0, "release"
        return True, 0.0, release_elapsed_s, activation_reason

    def _allocator_state(self) -> WholeBodyAllocatorState:
        return WholeBodyAllocatorState(
            adaptive_share=self._adaptive_share.copy(),
            base_activation=self._base_activation,
            lift_activation=self._lift_activation,
            base_anchor_target=(self._base_anchor_target.copy() if self._base_anchor_target is not None else None),
            lift_anchor_position=self._lift_anchor_position,
            diagnostics=self._allocation_diagnostics,
            base_engaged=self._base_engaged,
            lift_engaged=self._lift_engaged,
            base_enter_elapsed_s=self._base_enter_elapsed_s,
            lift_enter_elapsed_s=self._lift_enter_elapsed_s,
            base_release_elapsed_s=self._base_release_elapsed_s,
            lift_release_elapsed_s=self._lift_release_elapsed_s,
            base_activation_reason=self._base_activation_reason,
            lift_activation_reason=self._lift_activation_reason,
            settling_blend=self._settling_blend,
            stationary_cost_blend=self._stationary_cost_blend,
            allocation_reference_targets={
                side: target.copy()
                for side, target in self._allocation_reference_targets.items()
            },
            previous_allocation_targets={
                side: target.copy()
                for side, target in self._previous_allocation_targets.items()
            },
        )

    def _restore_allocator_state(self, state: WholeBodyAllocatorState) -> None:
        """Roll back soft costs and anchors after a rejected QP update."""
        self._adaptive_share = state.adaptive_share.copy() if state.adaptive_share is not None else AdaptiveShare()
        self._base_activation = state.base_activation
        self._lift_activation = state.lift_activation
        self._base_anchor_target = state.base_anchor_target
        self._lift_anchor_position = state.lift_anchor_position
        self._allocation_diagnostics = state.diagnostics
        self._base_engaged = state.base_engaged
        self._lift_engaged = state.lift_engaged
        self._base_enter_elapsed_s = state.base_enter_elapsed_s
        self._lift_enter_elapsed_s = state.lift_enter_elapsed_s
        self._base_release_elapsed_s = state.base_release_elapsed_s
        self._lift_release_elapsed_s = state.lift_release_elapsed_s
        self._base_activation_reason = state.base_activation_reason
        self._lift_activation_reason = state.lift_activation_reason
        self._settling_blend = state.settling_blend
        self._stationary_cost_blend = state.stationary_cost_blend
        self._allocation_reference_targets = {
            side: target.copy()
            for side, target in state.allocation_reference_targets.items()
        }
        self._previous_allocation_targets = {
            side: target.copy()
            for side, target in state.previous_allocation_targets.items()
        }
        self.low_acceleration_task.cost = (
            self.config.low_acceleration_cost
            + self._settling_blend
            * (self.config.settled_low_acceleration_cost - self.config.low_acceleration_cost)
        )

        if self.base_anchor_task is not None and self._base_anchor_target is not None:
            self.base_anchor_task.set_target(self._base_anchor_target)
            assert state.diagnostics.base_weights is not None
            linear_x, linear_y, yaw = state.diagnostics.base_weights
            self.base_anchor_task.cost = np.array(
                [linear_x, linear_y, 0.0, 0.0, 0.0, yaw],
                dtype=float,
            )
        posture_target = self.nominal_q.copy()
        posture_target[self._base_dofs] = self._lift_anchor_position
        self.posture_task.set_target(posture_target)
        pressures = np.array(
            [state.diagnostics.arm_joint_pressures[name] for name in ARM_JOINTS],
            dtype=float,
        )
        arm_weights = np.array(
            [state.diagnostics.arm_weights[name] for name in ARM_JOINTS],
            dtype=float,
        )
        self.posture_task.cost = np.concatenate(
            ([state.diagnostics.lift_weight], self._limit_escape_costs(pressures, arm_weights))
        )
        self.lift_damping_task.cost[0] = (
            self.config.lift_velocity_damping_scale * state.diagnostics.lift_weight
        )
        self.lift_damping_task.cost[1:] = (
            self._settling_blend * self.config.settled_arm_damping_cost
        )
        self.arm_posture_task.cost = np.array(
            arm_weights,
        )


    def _update_redundancy_allocation(self, targets: Mapping[str, pin.SE3], dt: float) -> None:
        """Update soft QP costs and local references from the selected TCP demand."""
        allocator = self.config.allocator
        current_poses = {
            side: self.configuration.get_transform_frame_to_world(frame)
            for side, frame in TCP_FRAMES.items()
        }
        tracking_errors = {
            side: np.asarray(targets[side].translation)
            - current_poses[side].translation
            for side in TCP_FRAMES
        }
        if self.config.allocator.mode != "continuous_v1":
            if set(self._allocation_reference_targets) != set(TCP_FRAMES):
                self._allocation_reference_targets = {
                    side: current_poses[side].copy() for side in TCP_FRAMES
                }
            errors = {
                side: np.asarray(targets[side].translation)
                - self._allocation_reference_targets[side].translation
                for side in TCP_FRAMES
            }
        else:
            errors = tracking_errors
        common = 0.5 * (errors["left"] + errors["right"])
        differential = 0.5 * (errors["left"] - errors["right"])
        tracking_common = 0.5 * (tracking_errors["left"] + tracking_errors["right"])

        common_target_delta = np.zeros(3, dtype=float)
        common_target_yaw_delta = 0.0
        translation_deltas = []
        if allocator.mode in PREDICTIVE_SHARE_MODES:
            if set(self._previous_allocation_targets) == set(TCP_FRAMES):
                translation_deltas = []
                yaw_deltas = []
                for side in TCP_FRAMES:
                    previous = self._previous_allocation_targets[side]
                    current = targets[side]
                    translation_deltas.append(current.translation - previous.translation)
                    local_rotation_delta = pin.log3(previous.rotation.T @ current.rotation)
                    yaw_deltas.append(float((previous.rotation @ local_rotation_delta)[2]))
                common_target_delta = 0.5 * (
                    np.asarray(translation_deltas[0]) + np.asarray(translation_deltas[1])
                )
                common_target_yaw_delta = 0.5 * (yaw_deltas[0] + yaw_deltas[1])
            else:
                self._previous_allocation_targets = {
                    side: targets[side].copy() for side in TCP_FRAMES
                }

        pressures = self._arm_pressures()
        arm_pressure = 1.0 - float(np.prod(1.0 - pressures))
        pressure_blend = _smoothstep(arm_pressure)
        posture_errors = self.arm_posture_task.projected_error_norms(self.configuration)
        maximum_posture_error = max(posture_errors.values())
        manipulability_activation = max(
            task.activation for task in self.arm_manipulability_tasks.values()
        )
        posture_severity = self._smooth_band(
            maximum_posture_error,
            allocator.posture_assist_exit_rad,
            allocator.posture_assist_enter_rad,
        )
        rotation_errors = {}
        for side in TCP_FRAMES:
            reference = (
                self._allocation_reference_targets[side]
                if self.config.allocator.mode != "continuous_v1"
                else current_poses[side]
            )
            local_error = pin.log3(reference.rotation.T @ targets[side].rotation)
            rotation_errors[side] = reference.rotation @ local_error
        common_yaw_error = abs(
            0.5 * float(rotation_errors["left"][2] + rotation_errors["right"][2])
        )
        yaw_severity = self._smooth_band(
            common_yaw_error,
            allocator.common_yaw_exit_rad,
            allocator.common_yaw_enter_rad,
        )
        ergonomic_strain = max(arm_pressure, manipulability_activation, posture_severity)
        base_distance = float(
            allocator.base_activation_distances_m[0]
            + pressure_blend * (allocator.base_activation_distances_m[1] - allocator.base_activation_distances_m[0])
        )
        lift_distance = float(
            allocator.lift_activation_distances_m[0]
            + pressure_blend * (allocator.lift_activation_distances_m[1] - allocator.lift_activation_distances_m[0])
        )

        common_xy = float(np.linalg.norm(common[:2]))
        differential_xy = float(np.linalg.norm(differential[:2]))
        common_z = abs(float(common[2]))
        differential_z = abs(float(differential[2]))
        epsilon_sq = allocator.dominance_epsilon_m**2
        horizontal_dominance = common_xy**2 / (common_xy**2 + differential_xy**2 + epsilon_sq)
        vertical_dominance = common_z**2 / (common_z**2 + differential_z**2 + epsilon_sq)
        horizontal_dominance += arm_pressure * (1.0 - horizontal_dominance)
        vertical_dominance += arm_pressure * (1.0 - vertical_dominance)
        if allocator.mode != "continuous_v1":
            posture_assist = allocator.mode in {
                "posture_assist_hysteresis_v3",
                "predictive_body_share_v4", "adaptive_body_share_v5", "directional_body_share_v6",
            }
            (
                self._base_engaged,
                self._base_enter_elapsed_s,
                self._base_release_elapsed_s,
                self._base_activation_reason,
            ) = self._hysteresis_axis_state(
                engaged=self._base_engaged,
                enter_elapsed_s=self._base_enter_elapsed_s,
                release_elapsed_s=self._base_release_elapsed_s,
                activation_reason=self._base_activation_reason,
                distance=common_xy,
                dominance=horizontal_dominance,
                pressure=arm_pressure,
                enter_distance=allocator.base_enter_distance_m,
                exit_distance=allocator.base_exit_distance_m,
                dt=dt,
                config=allocator,
                assist_demand=posture_assist and (
                    maximum_posture_error >= allocator.posture_assist_enter_rad
                    or common_yaw_error >= allocator.common_yaw_enter_rad
                    or manipulability_activation >= 0.5
                ),
                assist_release_ready=(
                    not posture_assist
                    or (
                        maximum_posture_error <= allocator.posture_assist_exit_rad
                        and common_yaw_error <= allocator.common_yaw_exit_rad
                        and manipulability_activation <= 0.1
                    )
                ),
            )
            (
                self._lift_engaged,
                self._lift_enter_elapsed_s,
                self._lift_release_elapsed_s,
                self._lift_activation_reason,
            ) = self._hysteresis_axis_state(
                engaged=self._lift_engaged,
                enter_elapsed_s=self._lift_enter_elapsed_s,
                release_elapsed_s=self._lift_release_elapsed_s,
                activation_reason=self._lift_activation_reason,
                distance=common_z,
                dominance=vertical_dominance,
                pressure=arm_pressure,
                enter_distance=allocator.lift_enter_distance_m,
                exit_distance=allocator.lift_exit_distance_m,
                dt=dt,
                config=allocator,
                assist_demand=posture_assist and (
                    maximum_posture_error >= allocator.posture_assist_enter_rad
                    or manipulability_activation >= 0.5
                ),
                assist_release_ready=(
                    not posture_assist
                    or (
                        maximum_posture_error <= allocator.posture_assist_exit_rad
                        and manipulability_activation <= 0.1
                    )
                ),
            )
            if posture_assist:
                base_soft = allocator.soft_assist_activation * max(
                    posture_severity, yaw_severity, manipulability_activation
                )
                lift_soft = allocator.soft_assist_activation * max(
                    posture_severity, manipulability_activation
                )
                base_target_activation = (
                    1.0
                    if self._base_activation_reason in {
                        "joint_pressure", "persistent_cartesian_error"
                    }
                    else base_soft
                ) if self._base_engaged else 0.0
                lift_target_activation = (
                    1.0
                    if self._lift_activation_reason in {
                        "joint_pressure", "persistent_cartesian_error"
                    }
                    else lift_soft
                ) if self._lift_engaged else 0.0
            else:
                base_target_activation = float(self._base_engaged)
                lift_target_activation = float(self._lift_engaged)
        else:
            base_target_activation = horizontal_dominance * _zero_based_activation(
                common_xy,
                base_distance,
                allocator.activation_steepness,
            )
            lift_target_activation = vertical_dominance * _zero_based_activation(
                common_z,
                lift_distance,
                allocator.activation_steepness,
            )
            self._base_engaged = base_target_activation > 0.0
            self._lift_engaged = lift_target_activation > 0.0
            self._base_activation_reason = "continuous" if self._base_engaged else "none"
            self._lift_activation_reason = "continuous" if self._lift_engaged else "none"
        if self.base_mode == "planar" and self.config.base_control_enabled:
            self._base_activation = self._filtered_activation(self._base_activation, base_target_activation, dt)
        else:
            self._base_activation = 0.0
            self._base_engaged = False
            self._base_enter_elapsed_s = 0.0
            self._base_release_elapsed_s = 0.0
            self._base_activation_reason = "none"
        self._lift_activation = self._filtered_activation(self._lift_activation, lift_target_activation, dt)

        adaptive_delta = None
        if allocator.mode in ADAPTIVE_SHARE_MODES:
            if allocator.mode == "directional_body_share_v6":
                from .adaptive_share import directional_lift_anchor

                self._lift_anchor_position, rebased = directional_lift_anchor(
                    self._lift_anchor_position, self._measured_configuration.q[self._base_dofs],
                    common_target_delta[2],
                    common_target_delta[2] ** 2 / (
                        common_target_delta[2] ** 2
                        + (.5 * (translation_deltas[0][2] - translation_deltas[1][2])) ** 2 + 1e-12
                    ) if self._previous_allocation_targets is not None and translation_deltas else 0.,
                )
                if rebased and self._adaptive_share.goal is not None:
                    self._adaptive_share.goal[3] = self._lift_anchor_position
            base_pose = self._base_anchor_target
            base_anchor = np.zeros(3) if base_pose is None else np.r_[
                base_pose.translation[:2], np.arctan2(base_pose.rotation[1, 0], base_pose.rotation[0, 0])]
            anchor = np.r_[base_anchor, self._lift_anchor_position]
            measured = self._measured_configuration.q
            body_q = self.configuration.q
            # Base planning coordinates are measured, so their position gap
            # is identically zero. Use velocity tracking error over one cycle.
            base_gap = (dt * (self._last_velocity[:3] - self._last_measured_velocity[:3])
                        if self._base_dofs and self._last_measured_velocity is not None else np.zeros(3))
            gap = np.r_[base_gap, body_q[self._base_dofs] - measured[self._base_dofs]]
            differential_yaw = .5 * (rotation_errors['left'][2] - rotation_errors['right'][2])
            yaw_coherence = common_yaw_error**2 / (
                common_yaw_error**2 + differential_yaw**2 + max(self.config.target_orientation_deadband, 1e-6)**2)
            lower = np.r_[[-np.inf] * 3, self.model.lowerPositionLimit[self._base_dofs]]
            upper = np.r_[[np.inf] * 3, self.model.upperPositionLimit[self._base_dofs]]
            if self._base_dofs and self.config.base.workspace_mode == "bounded":
                lower[:3], upper[:3] = self.model.lowerPositionLimit[:3], self.model.upperPositionLimit[:3]
            base_share_enabled = float(self.config.base_control_enabled)
            adaptive_delta = self._adaptive_share.step(
                anchor, np.r_[common_target_delta[:2], common_target_yaw_delta, common_target_delta[2]],
                [horizontal_dominance, horizontal_dominance, yaw_coherence, vertical_dominance],
                ergonomic_strain, gap, lower, upper,
                [allocator.base_anchor_follow_linear_m_s] * 2 + [allocator.base_anchor_follow_angular_rad_s, allocator.lift_anchor_follow_m_s],
                [base_share_enabled * allocator.base_common_translation_share] * 2
                + [base_share_enabled * allocator.base_common_yaw_share, allocator.lift_common_translation_share],
                dt, stopped=self._stop_requested)

        base_weights: tuple[float, float, float] | None = None
        applied_base_share = np.zeros(3, dtype=float)
        if self.base_anchor_task is not None:
            current_base = self.configuration.get_transform_frame_to_world(BASE_FRAME)
            if self._base_anchor_target is None:
                self._base_anchor_target = current_base.copy()
            reference = self._base_anchor_target
            if allocator.mode in PREDICTIVE_SHARE_MODES:
                # Preserve the v3 ergonomic escape path in addition to the
                # explicit common-motion share.  Following the measured base
                # while active relaxes the anchor and lets the QP allocate
                # extra motion when arm posture or manipulability demands it.
                requested_xy = (
                    allocator.base_common_translation_share * common_target_delta[:2]
                    + 0.5 * self._base_activation
                    * (current_base.translation[:2] - reference.translation[:2])
                )
                if adaptive_delta is not None:
                    requested_xy = adaptive_delta[:2] + 0.5 * self._base_activation * (
                        current_base.translation[:2] - reference.translation[:2])
                xy_norm = float(np.linalg.norm(requested_xy))
                maximum_xy_step = allocator.base_anchor_follow_linear_m_s * dt
                applied_xy = requested_xy * min(1.0, maximum_xy_step / max(xy_norm, 1e-12))
                current_yaw = float(np.arctan2(
                    current_base.rotation[1, 0], current_base.rotation[0, 0]
                ))
                reference_yaw = float(np.arctan2(
                    reference.rotation[1, 0], reference.rotation[0, 0]
                ))
                yaw_error = current_yaw - reference_yaw
                if self.config.base.workspace_mode == "unbounded":
                    yaw_error = float(np.arctan2(np.sin(yaw_error), np.cos(yaw_error)))
                requested_yaw = (
                    allocator.base_common_yaw_share * common_target_yaw_delta
                    + 0.5 * self._base_activation * yaw_error
                )
                if adaptive_delta is not None:
                    requested_yaw = adaptive_delta[2] + 0.5 * self._base_activation * yaw_error
                applied_yaw = float(np.clip(
                    requested_yaw,
                    -allocator.base_anchor_follow_angular_rad_s * dt,
                    allocator.base_anchor_follow_angular_rad_s * dt,
                ))
                translation = reference.translation.copy()
                translation[:2] += applied_xy
                desired_yaw = reference_yaw + applied_yaw
                if self.config.base.workspace_mode == "bounded":
                    translation[0] = np.clip(
                        translation[0],
                        self.config.base.x_limits_m[0] + self.config.base.position_margin_m,
                        self.config.base.x_limits_m[1] - self.config.base.position_margin_m,
                    )
                    translation[1] = np.clip(
                        translation[1],
                        self.config.base.y_limits_m[0] + self.config.base.position_margin_m,
                        self.config.base.y_limits_m[1] - self.config.base.position_margin_m,
                    )
                    applied_xy = translation[:2] - reference.translation[:2]
                    desired_yaw = float(np.clip(
                        reference_yaw + applied_yaw,
                        self.config.base.yaw_limits_rad[0] + self.config.base.yaw_margin_rad,
                        self.config.base.yaw_limits_rad[1] - self.config.base.yaw_margin_rad,
                    ))
                applied_xy = translation[:2] - reference.translation[:2]
                applied_yaw = desired_yaw - reference_yaw
                rotation = pin.exp3(np.array([0.0, 0.0, desired_yaw]))
                applied_base_share = np.array([applied_xy[0], applied_xy[1], applied_yaw])
                inactive_position_cost = allocator.base_inactive_cost
                inactive_yaw_cost = allocator.base_inactive_cost
            elif allocator.mode != "continuous_v1":
                translation_delta = current_base.translation - reference.translation
                translation_step = min(
                    1.0,
                    allocator.base_anchor_follow_linear_m_s
                    * dt
                    * self._base_activation
                    / max(1e-12, float(np.linalg.norm(translation_delta[:2]))),
                )
                translation = reference.translation + translation_step * translation_delta
                rotation_delta = pin.log3(reference.rotation.T @ current_base.rotation)
                rotation_step = min(
                    1.0,
                    allocator.base_anchor_follow_angular_rad_s
                    * dt
                    * self._base_activation
                    / max(1e-12, float(np.linalg.norm(rotation_delta))),
                )
                rotation = reference.rotation @ pin.exp3(rotation_step * rotation_delta)
                inactive_position_cost = allocator.base_inactive_cost
                inactive_yaw_cost = allocator.base_inactive_cost
            else:
                translation = reference.translation + self._base_activation * (
                    current_base.translation - reference.translation
                )
                rotation_delta = pin.log3(reference.rotation.T @ current_base.rotation)
                rotation = reference.rotation @ pin.exp3(self._base_activation * rotation_delta)
                inactive_position_cost = self.config.base.anchor_position_cost
                inactive_yaw_cost = self.config.base.anchor_yaw_cost
            self._base_anchor_target = pin.SE3(rotation, translation)
            self.base_anchor_task.set_target(self._base_anchor_target)
            if allocator.mode in POSTURE_SHARE_MODES:
                linear_weight = float(np.exp(
                    np.log(inactive_position_cost)
                    + self._base_activation
                    * (np.log(allocator.base_active_cost) - np.log(inactive_position_cost))
                ))
                yaw_weight = float(np.exp(
                    np.log(inactive_yaw_cost)
                    + self._base_activation
                    * (np.log(allocator.base_active_cost) - np.log(inactive_yaw_cost))
                ))
            else:
                linear_weight = inactive_position_cost + self._base_activation * (
                    allocator.base_active_cost - inactive_position_cost
                )
                yaw_weight = inactive_yaw_cost + self._base_activation * (
                    allocator.base_active_cost - inactive_yaw_cost
                )
            self.base_anchor_task.cost = np.array(
                [linear_weight, linear_weight, 0.0, 0.0, 0.0, yaw_weight],
                dtype=float,
            )
            base_weights = (float(linear_weight), float(linear_weight), float(yaw_weight))

        lift_index = self._base_dofs
        lift_reference_goal = np.clip(
            float(self.configuration.q[lift_index])
            + allocator.lift_reference_error_gain * float(tracking_common[2]),
            self.model.lowerPositionLimit[lift_index],
            self.model.upperPositionLimit[lift_index],
        )
        # Acquire a local reference while lift participation is clearly active,
        # then freeze it through the small sigmoid release tail. This avoids
        # both rebound to reset height and slow integral drift at zero error.
        # A true deadband is required here: ``zero_based_activation`` is zero
        # only at an input of zero, so using 0.12 as its scale let tiny
        # one-arm motions integrate into a fast lift reference.
        lift_anchor_blend = _smoothstep(
            np.clip((self._lift_activation - 0.12) / (1.0 - 0.12), 0.0, 1.0)
        )
        if allocator.mode in PREDICTIVE_SHARE_MODES:
            lift_anchor_delta = float(np.clip(
                allocator.lift_common_translation_share * float(common_target_delta[2]),
                -allocator.lift_anchor_follow_m_s * dt,
                allocator.lift_anchor_follow_m_s * dt,
            ))
            if adaptive_delta is not None:
                lift_anchor_delta = float(adaptive_delta[3])
            self._lift_anchor_position += lift_anchor_delta
        else:
            lift_anchor_delta = lift_reference_goal - self._lift_anchor_position
        if allocator.mode != "continuous_v1" and allocator.mode not in PREDICTIVE_SHARE_MODES:
            lift_anchor_delta = float(
                np.clip(
                    lift_anchor_delta,
                    -allocator.lift_anchor_follow_m_s * dt * lift_anchor_blend,
                    allocator.lift_anchor_follow_m_s * dt * lift_anchor_blend,
                )
            )
            self._lift_anchor_position += lift_anchor_delta
        elif allocator.mode == "continuous_v1":
            self._lift_anchor_position += lift_anchor_blend * lift_anchor_delta
        self._lift_anchor_position = float(
            np.clip(
                self._lift_anchor_position,
                self.model.lowerPositionLimit[lift_index],
                self.model.upperPositionLimit[lift_index],
            )
        )
        posture_target = self.nominal_q.copy()
        posture_target[lift_index] = self._lift_anchor_position
        self.posture_task.set_target(posture_target)
        inactive_lift_cost = (
            allocator.lift_inactive_cost
            if allocator.mode != "continuous_v1"
            else self.config.lift_posture_cost
        )
        if allocator.mode in POSTURE_SHARE_MODES:
            lift_weight = float(np.exp(
                np.log(inactive_lift_cost)
                + self._lift_activation
                * (np.log(allocator.lift_active_cost) - np.log(inactive_lift_cost))
            ))
        else:
            lift_weight = inactive_lift_cost + self._lift_activation * (
                allocator.lift_active_cost - inactive_lift_cost
            )
        arm_weights = self.config.arm_posture_cost + pressures * (
            allocator.arm_limit_cost - self.config.arm_posture_cost
        )
        self.posture_task.cost = np.concatenate(
            ([lift_weight], self._limit_escape_costs(pressures, arm_weights))
        )
        self.lift_damping_task.cost[0] = self.config.lift_velocity_damping_scale * lift_weight
        posture_activation = np.concatenate(
            tuple(
                np.full(
                    7,
                    np.sqrt(max(
                        self.config.arm_posture_min_fraction,
                        1.0 - self.arm_manipulability_tasks[side].activation,
                    )),
                )
                for side in ("left", "right")
            )
        )
        effective_arm_weights = np.asarray(arm_weights, dtype=float) * posture_activation
        self.arm_posture_task.cost = effective_arm_weights

        if allocator.mode != "continuous_v1":
            maximum_tracking_error = max(
                float(np.linalg.norm(error)) for error in tracking_errors.values()
            )
            if maximum_tracking_error <= allocator.rebase_position_error_m:
                for side in TCP_FRAMES:
                    reference = self._allocation_reference_targets[side]
                    if self._base_engaged:
                        reference.translation[:2] = targets[side].translation[:2]
                    if self._lift_engaged:
                        reference.translation[2] = targets[side].translation[2]

        if allocator.mode in PREDICTIVE_SHARE_MODES:
            self._previous_allocation_targets = {
                side: targets[side].copy() for side in TCP_FRAMES
            }

        pressure_by_joint = {name: float(pressure) for name, pressure in zip(ARM_JOINTS, pressures, strict=True)}
        weight_by_joint = {
            name: float(weight) for name, weight in zip(ARM_JOINTS, effective_arm_weights, strict=True)
        }
        self._allocation_diagnostics = WholeBodyAllocationDiagnostics(
            common_translation_error=tuple(float(value) for value in common),
            differential_translation_error=tuple(float(value) for value in differential),
            base_activation=self._base_activation,
            lift_activation=self._lift_activation,
            arm_pressure=arm_pressure,
            max_arm_joint_pressure=float(np.max(pressures)),
            arm_joint_pressures=pressure_by_joint,
            base_weights=base_weights,
            lift_weight=float(lift_weight),
            arm_weights=weight_by_joint,
            mode=allocator.mode,
            base_engaged=self._base_engaged,
            lift_engaged=self._lift_engaged,
            base_activation_reason=self._base_activation_reason,
            lift_activation_reason=self._lift_activation_reason,
            base_enter_elapsed_s=self._base_enter_elapsed_s,
            lift_enter_elapsed_s=self._lift_enter_elapsed_s,
            base_release_elapsed_s=self._base_release_elapsed_s,
            lift_release_elapsed_s=self._lift_release_elapsed_s,
            base_anchor_position=(
                None
                if self._base_anchor_target is None
                else tuple(float(value) for value in self._base_anchor_target.translation)
            ),
            lift_anchor_position=self._lift_anchor_position,
            allocation_reference_translation={
                side: tuple(float(value) for value in target.translation)
                for side, target in self._allocation_reference_targets.items()
            },
            common_yaw_error_rad=common_yaw_error,
            posture_error_rad={
                side: float(value) for side, value in posture_errors.items()
            },
            manipulability_activation=float(manipulability_activation),
            ergonomic_strain=float(ergonomic_strain),
            adaptive_shares=(None if adaptive_delta is None else tuple(self._adaptive_share.shares)),
            body_goal=(None if adaptive_delta is None else tuple(self._adaptive_share.goal)),
            common_target_delta=tuple(float(value) for value in common_target_delta),
            requested_base_share_delta=(
                allocator.base_common_translation_share * float(common_target_delta[0]),
                allocator.base_common_translation_share * float(common_target_delta[1]),
                allocator.base_common_yaw_share * float(common_target_yaw_delta),
            ),
            base_share_delta=tuple(float(value) for value in applied_base_share),
            requested_lift_share_delta_m=(
                allocator.lift_common_translation_share * float(common_target_delta[2])
            ),
            lift_share_delta_m=float(
                lift_anchor_delta if allocator.mode in PREDICTIVE_SHARE_MODES else 0.0
            ),
        )

    def _task_errors(self) -> tuple[float, float, float, float]:
        errors: dict[str, tuple[float, float]] = {}
        for side, task in self.frame_tasks.items():
            error = task.compute_geometric_error(self._measured_configuration)
            errors[side] = (float(np.linalg.norm(error[:3])), float(np.linalg.norm(error[3:])))
        return errors["left"][0], errors["right"][0], errors["left"][1], errors["right"][1]

    def tracking_observation(self):
        """Small per-cycle causal evidence; no additional solve or geometry query."""
        return dict(tracking_state='measured' if self._measured_tracking_active else 'reference',
                    measured_tracking=(None if self.measured_tracker is None else dict(self.measured_tracker.observations)),
                    following_governor=(None if self.following_governor is None else self.following_governor.observation()),
                    settling_blend=self._settling_blend, stationary_cost_blend=self._stationary_cost_blend,
                    arm_posture_weights=self.arm_posture_task.cost.copy(),
                    posture_weights=self.posture_task.cost.copy(),
                    low_acceleration_weight=float(self.low_acceleration_task.cost),
                    reference_acceleration=self._last_acceleration.copy(),
                    manipulability_weights={s: float(t.cost) for s,t in self.arm_manipulability_tasks.items()},
                    planners={s:dict(settled=p.settled, input_event=getattr(p, 'input_event', None),
                                    body_twist=p.motion().body_twist.copy(),
                                    body_acceleration=p.motion().body_acceleration.copy())
                              for s,p in self._target_planners.items()})

    def _update_settling_regularization(self, dt: float) -> None:
        """Increase command smoothness only after the shaped reference settles."""

        if self._measured_tracking_active:
            desired_blend = float(self.measured_tracker.settled)
        elif not self._target_planners:
            desired_blend = 0.0
        else:
            motions = tuple(planner.motion() for planner in self._target_planners.values())
            linear_speed = max(float(np.linalg.norm(motion.body_twist[:3])) for motion in motions)
            angular_speed = max(float(np.linalg.norm(motion.body_twist[3:])) for motion in motions)
            left_pos, right_pos, left_rot, right_rot = self._task_errors()
            desired_blend = float(
                linear_speed <= self.config.settling_linear_velocity_m_s
                and angular_speed <= self.config.settling_angular_velocity_rad_s
                and max(left_pos, right_pos) <= self.config.settling_position_error_m
                and max(left_rot, right_rot) <= self.config.settling_orientation_error_rad
                and (not self.config.stationary_redundancy_hold
                     or all(planner.settled for planner in self._target_planners.values()))
            )
        blend = 1.0 - float(np.exp(-dt / self.config.settling_blend_s))
        self._settling_blend = float(
            np.clip(self._settling_blend + blend * (desired_blend - self._settling_blend), 0.0, 1.0)
        )
        self.low_acceleration_task.cost = (
            self.config.low_acceleration_cost
            + self._settling_blend
            * (self.config.settled_low_acceleration_cost - self.config.low_acceleration_cost)
        )
        self.lift_damping_task.cost[1:] = (
            self._settling_blend * self.config.settled_arm_damping_cost
        )

        if self.config.stationary_redundancy_hold:
            stationary = (self.measured_tracker.settled if self._measured_tracking_active else
                          bool(self._target_planners) and all(
                              planner.settled for planner in self._target_planners.values()))
            self.stationary_damping_task.cost = self._settling_blend * self.config.settled_arm_damping_cost
            if self.config.stationary_cost_blend_s > 0:
                alpha = -np.expm1(-dt / self.config.stationary_cost_blend_s)
                self._stationary_cost_blend += alpha * (float(stationary) - self._stationary_cost_blend)
                scale = 1. - self._stationary_cost_blend
                self.posture_task.cost *= scale
                self.arm_posture_task.cost *= scale
                if self.base_anchor_task is not None:
                    self.base_anchor_task.cost *= scale
                for task in self.arm_manipulability_tasks.values():
                    task.cost *= scale
            elif stationary:
                # Stop autonomous posture optimization once the operator's
                # reference is stationary. Full-body velocity regularization
                # includes the planar base; TCP objectives and every hard
                # joint/effort/acceleration/jerk constraint remain active.
                self.posture_task.cost[:] = 0.
                self.arm_posture_task.cost[:] = 0.
                if self.base_anchor_task is not None:
                    self.base_anchor_task.cost[:] = 0.
                for task in self.arm_manipulability_tasks.values():
                    task.cost = 0.


    @staticmethod
    def _transform_error(reference: pin.SE3, actual: pin.SE3) -> TransformError:
        return TransformError(
            position_m=float(np.linalg.norm(reference.translation - actual.translation)),
            orientation_rad=float(np.linalg.norm(pin.log3(reference.rotation.T @ actual.rotation))),
        )

    def _command_frame_poses(self, positions: np.ndarray) -> dict[str, pin.SE3]:
        pin.forwardKinematics(self.model, self._diagnostic_data, positions)
        pin.updateFramePlacements(self.model, self._diagnostic_data)
        return {
            side: self._diagnostic_data.oMf[self.model.getFrameId(frame)].copy()
            for side, frame in TCP_FRAMES.items()
        }

    def _tracking_chain(
        self,
        raw_targets: Mapping[str, pin.SE3],
        command_positions: np.ndarray,
    ) -> dict[str, TrackingErrorChain]:
        command_poses = self._command_frame_poses(command_positions)
        return {
            side: TrackingErrorChain(
                raw_to_limited=self._transform_error(raw_targets[side], self._limited_targets[side]),
                limited_to_command=self._transform_error(
                    self._limited_targets[side], command_poses[side]
                ),
                command_to_measured=self._transform_error(
                    command_poses[side],
                    self._measured_configuration.get_transform_frame_to_world(TCP_FRAMES[side]),
                ),
            )
            for side in TCP_FRAMES
        }

    def _motion_utilization(self) -> MotionUtilization:
        finite = np.isfinite(self.model.lowerPositionLimit) & np.isfinite(self.model.upperPositionLimit)
        lower, upper = self.model.lowerPositionLimit[finite], self.model.upperPositionLimit[finite]
        half_range = 0.5 * (upper - lower)
        midpoint = 0.5 * (upper + lower)
        return MotionUtilization(
            position=float(np.max(np.abs(self.configuration.q[finite] - midpoint) / half_range, initial=0.)),
            velocity=float(np.max(np.abs(self._last_velocity) / self.model.velocityLimit)),
            acceleration=float(np.max(np.abs(self._last_acceleration) / self._acceleration_limits)),
            jerk=float(np.max(np.abs(self._last_jerk) / self._jerk_limits)),
        )

    def _target_motion_utilization(self) -> CartesianMotionUtilization | None:
        if self._measured_tracking_active and self.measured_tracker.observations:
            values = {}
            for part, label in ((slice(0, 3), 'linear'), (slice(3, 6), 'angular')):
                for order in ('velocity', 'acceleration', 'jerk'):
                    limit = getattr(self.config, f'target_{label}_{order}_limit')
                    values[f'{label}_{order}'] = max(float(np.linalg.norm(o[f'{order}_world'][part])/limit)
                        for o in self.measured_tracker.observations.values())
            return CartesianMotionUtilization(**values)
        if not self._target_planners:
            return None
        utilizations = [planner.utilization() for planner in self._target_planners.values()]
        return CartesianMotionUtilization(
            **{
                field: max(float(getattr(utilization, field)) for utilization in utilizations)
                for field in CartesianMotionUtilization.__dataclass_fields__
            }
        )

    def _diagnostics(
        self,
        status: str,
        solve_time_ms: float,
        target_limited: bool,
        *,
        raw_targets: Mapping[str, pin.SE3] | None = None,
        command_positions: np.ndarray | None = None,
        timings: IKPhaseTimings | None = None,
        fault: str | None = None,
    ) -> IKDiagnostics:
        left_pos, right_pos, left_rot, right_rot = self._task_errors()
        base_pose = None
        base_velocity = None
        base_at_limit = False
        if self.base_mode == "planar":
            base_pose = tuple(float(value) for value in self.configuration.q[:3])
            base_velocity = tuple(float(value) for value in self._last_velocity[:3])
            lower_margin = self.configuration.q[:3] - self.model.lowerPositionLimit[:3]
            upper_margin = self.model.upperPositionLimit[:3] - self.configuration.q[:3]
            base_at_limit = bool(np.min(np.minimum(lower_margin, upper_margin)) <= 1e-5)
        nullspace_error = self.arm_posture_task.projected_error_norms(self.configuration)
        manipulability = {
            side: task.metrics(self.configuration) for side, task in self.arm_manipulability_tasks.items()
        }
        collision_pairs = (
            ()
            if self.self_collision_system is None
            else self.self_collision_system.diagnostics(max_pairs=8)
        )
        return IKDiagnostics(
            status=status,
            target_planner_failures=self.target_planner_failures,
            last_target_planner_error=self.last_target_planner_error,
            solve_time_ms=solve_time_ms,
            left_position_error=left_pos,
            right_position_error=right_pos,
            left_orientation_error=left_rot,
            right_orientation_error=right_rot,
            target_limited=target_limited,
            target_settled=(
                self.measured_tracker.settled if self._measured_tracking_active else (
                    not target_limited if not self._target_planners
                    else all(planner.settled for planner in self._target_planners.values()))
            ),
            target_motion_utilization=self._target_motion_utilization(),
            base_mode=self.base_mode,
            base_pose=base_pose,
            base_velocity_command=base_velocity,
            base_at_limit=base_at_limit,
            allocation=self._allocation_diagnostics,
            nullspace_posture_error_rad=nullspace_error,
            arm_manipulability=manipulability,
            collision_pairs=collision_pairs,
            tracking_chain=(
                None
                if raw_targets is None or command_positions is None
                else self._tracking_chain(raw_targets, command_positions)
            ),
            motion_utilization=self._motion_utilization(),
            timings=timings,
            fault=fault,
            fault_kind=self._fault_info[0] if fault else None,
            fault_details=self._fault_info[1] if fault else None,
            stop_reason=self._stop_reason,
            feedback_age_s=self._feedback_age_s,
            dispatch_state_age_s=self._dispatch_state_age_s,
            deadline_reject_count=self._deadline_reject_count,
        )

    def _base_command(self, positions: np.ndarray, velocity: np.ndarray) -> PlanarBaseCommand | None:
        if self.base_mode == "fixed":
            return None
        yaw = float(self.configuration.q[2])
        cosine = float(np.cos(yaw))
        sine = float(np.sin(yaw))
        vx, vy, wz = (float(value) for value in velocity[:3])
        body = np.array((cosine * vx + sine * vy, -sine * vx + cosine * vy, wz))
        return PlanarBaseCommand(
            next_pose_world=positions[:3].copy(),
            velocity_world=velocity[:3].copy(),
            velocity_body=body,
        )

    def _collision_activation_travel(
        self,
        safety_travel: np.ndarray | None,
        velocity: np.ndarray,
        dt: float,
    ) -> np.ndarray | None:
        """Bound travel until the next collision refresh without using plant physics."""

        if self.config.collision_activation_policy != "causal_reachable_v3":
            return None
        if safety_travel is None:
            raise ValueError("causal collision activation requires a stopping-travel bound")
        velocity = np.abs(np.asarray(velocity, dtype=float))
        one_step = np.minimum(
            self.model.velocityLimit * dt,
            velocity * dt + 0.5 * self._acceleration_limits * dt**2,
        )
        activation_travel = np.asarray(safety_travel, dtype=float) + one_step
        if not np.all(np.isfinite(activation_travel)) or np.any(activation_travel < 0.0):
            raise ValueError("causal collision activation produced an invalid travel bound")
        return activation_travel

    def _swept_collision_is_clear(
        self, measured_q: np.ndarray, command_q: np.ndarray, dt: float
    ) -> bool:
        """Sample the executable measured-to-command chord with portable proxies."""
        if self._collision_sweep_system is None:
            return True
        assert self._collision_sweep_configuration is not None
        step = self.config.collision_sweep_max_joint_step_rad
        assert step is not None
        tangent = pin.difference(self.model, measured_q, command_q)
        arm_tangent = tangent[self._base_dofs + 1 :]
        intervals = max(1, int(np.ceil(np.max(np.abs(arm_tangent)) / step)))
        if intervals > self.config.collision_sweep_max_intervals:
            return False
        # A single reviewed proxy step is already covered by the hard endpoint
        # CBF and each pair's configured maximum-relative-step guard. Sweeping
        # is the additional check for commands that cross multiple such steps.
        uncertainty = getattr(self, "_feedback_position_uncertainty", np.zeros(self.model.nv))
        if intervals == 1 and not np.any(uncertainty):
            return True
        if (
            self.config.collision_activation_policy == "causal_reachable_v3"
            and not np.any(uncertainty)
            and self.self_collision_system is not None
            and self.self_collision_system.certifies_tangent_motion(tangent)
        ):
            return True
        zero_travel = np.zeros(self.model.nv)
        for sample_index in range(1, intervals + 1):
            sample_q = pin.integrate(
                self.model, measured_q, tangent * (sample_index / intervals)
            )
            self._collision_sweep_configuration.update(sample_q)
            sweep_barriers = self._collision_sweep_system.update(
                self._collision_sweep_configuration,
                dt,
                joint_safety_travel=zero_travel,
                joint_position_uncertainty=uncertainty,
                joint_activation_travel=(
                    zero_travel
                    if self.config.collision_activation_policy == "causal_reachable_v3"
                    else None
                ),
            )
            if any(np.any(b.compute_barrier(self._collision_sweep_configuration) < -1e-9)
                   for b in sweep_barriers):
                return False
            if any(
                diagnostic.residual_m < -1e-9
                for diagnostic in self._collision_sweep_system.diagnostics(all_pairs=True)
            ):
                return False
        return True

    def _validate_dispatch_freshness(
        self, started: float, feedback_age_s: float, *, stage: str | None = None
    ) -> None:
        """Reject v5 commands whose source state expired during synchronous compute."""

        stage = getattr(self, "_deadline_stage", "unknown") if stage is None else stage
        if self.config.self_collision_runtime != "native_fused_v2":
            return
        elapsed_s = time.perf_counter() - started
        dispatch_age_s = feedback_age_s + elapsed_s
        self._dispatch_state_age_s = dispatch_age_s
        if elapsed_s >= self.config.maximum_command_delay_s:
            self._deadline_reject_count += 1
            raise ControlDeadlineExceeded(
                f"control_deadline_exceeded: {elapsed_s:.6f}s >= "
                f"{self.config.maximum_command_delay_s:.6f}s",
                stage=stage, elapsed_s=elapsed_s,
                limit_s=self.config.maximum_command_delay_s,
                feedback_age_s=feedback_age_s,
            )
        if dispatch_age_s > self.config.maximum_feedback_age_s:
            self._deadline_reject_count += 1
            raise ControlDeadlineExceeded(
                f"dispatch_state_expired: {dispatch_age_s:.6f}s > "
                f"{self.config.maximum_feedback_age_s:.6f}s",
                stage=stage, elapsed_s=elapsed_s,
                limit_s=self.config.maximum_feedback_age_s,
                feedback_age_s=feedback_age_s,
            )


    def _frame_feedforward(self, task, target, desired_twist, reference_twist, measured_velocity):
        """Versioned reference-to-Pink error mapping shared by sim and hardware."""
        damping = self.config.target_velocity_feedback_gain if measured_velocity is not None else 0.
        if self.config.target_planner == "c2_tracking_v6" or self.config.tracking_objective == "normalized_v1":
            current = self.configuration.get_transform_frame_to_world(task.frame)
            relative = current.rotation.T @ target.rotation
            result = self.config.target_feedforward_gain * np.r_[
                target.rotation @ desired_twist[:3], pin.Jlog3(relative) @ desired_twist[3:]]
            if damping:
                measured_pose = self._measured_configuration.get_transform_frame_to_world(task.frame)
                velocity = self._measured_configuration.get_frame_jacobian(task.frame) @ measured_velocity
                result += damping * np.r_[
                    target.rotation @ reference_twist[:3] - measured_pose.rotation @ velocity[:3],
                    pin.Jlog3(relative) @ reference_twist[3:]
                    - pin.Jlog3(relative.T) @ (current.rotation.T @ measured_pose.rotation @ velocity[3:])]
            return result
        exact = self.config.target_planner in {
            "c2_tracking_v4", "c2_tracking_v5", "c2_tracking_v6"
        }
        if not exact and not damping:
            return self.config.target_feedforward_gain * desired_twist
        current = self.configuration.get_transform_frame_to_world(task.frame)
        error_transform = current.actInv(target)
        if exact:
            # For e=log(current^-1 target), target motion contributes Jlog(T)v.
            reference_rate = pin.Jlog6(error_transform) @ reference_twist
            feedforward = self.config.target_feedforward_gain * (pin.Jlog6(error_transform) @ desired_twist)
        else:
            # Preserve the dynamics of previously tuned schema <=16 profiles.
            rotation = current.rotation.T @ target.rotation
            reference_rate = np.r_[rotation @ reference_twist[:3], rotation @ reference_twist[3:]]
            feedforward = self.config.target_feedforward_gain * np.r_[
                rotation @ desired_twist[:3], rotation @ desired_twist[3:]]
        if damping:
            measured = self._measured_configuration.get_frame_jacobian(task.frame) @ measured_velocity
            if self.reference_limit is not None:
                measured_rotation = self._measured_configuration.get_transform_frame_to_world(task.frame).rotation
                rotation = current.rotation.T @ measured_rotation
                measured = np.r_[rotation @ measured[:3], rotation @ measured[3:]]
            if exact:
                measured = pin.Jlog6(error_transform.inverse()) @ measured
            feedforward += damping * (reference_rate - measured)
        # Keep the position residual on the bounded persistent reference.
        # Replacing it with measured TCP error regressed both saved grasps;
        # physical velocity feedback above remains active.
        return feedforward

    def step(
        self,
        measured_q: np.ndarray,
        targets: Mapping[str, pin.SE3],
        dt: float | None = None,
        *,
        measured_velocity: np.ndarray | None = None,
        measured_gripper_positions: np.ndarray | None = None,
        state_timestamp: float | None = None,
        control_timestamp: float | None = None,
        target_sample=None,
        target_twists: Mapping[str, np.ndarray] | None = None,
        target_accelerations: Mapping[str, np.ndarray] | None = None,
        posture_target: np.ndarray | None = None,
        posture_velocity: np.ndarray | None = None,
        feedback_received: Mapping[str, float] | None = None,
        feedback_policy: str = "theta-80ms-first-trial",
        control_phase: str = "tracking",
    ) -> IKCommand:
        from .preparation import PHASES

        self.last_mit_effort = None
        self._measured_tracking_active = (self.measured_pose_tracking and control_phase == 'tracking'
                                         and posture_target is None)
        if control_phase not in PHASES or (control_phase != "tracking" and self.base_mode != "planar"):
            raise ValueError("invalid control phase")
        if not self._initialized:
            raise RuntimeError("Call reset() before the first Pink IK step")
        if self._normal_lift_upper is not None and control_phase != "lift-retreat":
            raise ValueError("upper endpoint initialization permits lift-retreat only")
        if self._mit_fault_latched is not None:
            return self._mit_fault_command(self._mit_fault_latched)
        dt = self.config.control_dt if dt is None else float(dt)
        if not np.isfinite(dt) or dt <= 0:
            raise ValueError("IK timestep must be finite and positive")
        if self.config.motion_limit_source == "profile" and not np.isclose(
            dt, self.config.control_dt, atol=1e-12, rtol=0.0
        ):
            raise ValueError("profile-bound IK timestep must match the motion profile")
        if posture_velocity is not None and posture_target is None:
            raise ValueError("posture_velocity requires posture_target")
        if posture_target is not None:
            if self.reference_limit is None:
                raise ValueError("posture preparation requires MIT")
            posture_target = self._validate_configuration_vector(posture_target)
            posture_velocity = (np.zeros(self.model.nv) if posture_velocity is None
                                else self._validate_velocity_vector(posture_velocity))
        target_limited = False
        solve_start = time.perf_counter()
        thread_cpu_start = time.thread_time()
        state_update_ms = 0.0
        gravity_ms = 0.0
        output_checks_ms = 0.0
        braking_retries = 0
        boundary_limits = []
        warnings = []
        directional_boundaries = self.config.boundary_policy in {
            "directional_velocity_damper_v1",
            "jerk_stoppable_directional_v2",
        }
        v2_boundaries = self.config.boundary_policy == "jerk_stoppable_directional_v2"
        cycle_recovery_reason = self._transient_recovery_reason
        minimum_collision_reaction_clearance = None
        minimum_joint_reaction_clearance = None
        reference_state = tuple(value.copy() for value in (
            self._last_safe_positions, self._last_velocity, self._last_acceleration, self._last_jerk
        ))
        collision_distance_ms = 0.0
        collision_reference_ms = 0.0
        collision_measured_ms = 0.0
        collision_jacobian_ms = 0.0
        collision_row_pack_ms = 0.0
        task_construction_ms = 0.0
        planner_ms = 0.0
        allocator_ms = 0.0
        qp_build_ms = 0.0
        qp_solve_ms = 0.0
        swept_collision_ms = 0.0
        allocator_state = self._allocator_state()
        limited_target_state = {side: target.copy() for side, target in self._limited_targets.items()}
        target_planner_state = {
            side: planner.snapshot() for side, planner in self._target_planners.items()
        }
        validated_measured_q: np.ndarray | None = None
        try:
            state_phase_start = time.perf_counter()
            age = 0.0
            self._feedback_position_uncertainty = np.zeros(self.model.nv)
            if feedback_received is not None:
                from .feedback_policy import (
                    THETA_POLICY, THETA_RECORD_ONLY, THETA_BOUNDED, theta_uncertainty, coupled_groups,
                    validate_received, validate_group_progress,
                )

                if (feedback_policy not in {"strict", THETA_POLICY, THETA_RECORD_ONLY, THETA_BOUNDED}
                        or (feedback_policy == THETA_POLICY and self.config.motion_envelope != "first-trial")
                        or self.base_mode not in {"planar", "fixed"}
                        or (self.base_mode == "fixed" and feedback_policy != "strict")
                        or measured_velocity is None or control_timestamp is None):
                    raise ValueError("asynchronous feedback requires measured velocity and a supported base/policy mode")
                validate_received(feedback_received, control_timestamp, feedback_policy)
                expected = min(feedback_received[n] for n in coupled_groups(feedback_policy))
                if state_timestamp != expected:
                    raise ValueError("MIT epoch must match actual policy receive times")
                lift_speed = (self.model.velocityLimit[self._base_dofs]
                              if feedback_policy in {"strict", THETA_RECORD_ONLY, THETA_BOUNDED} else .02)
                validate_group_progress(self._last_feedback_received, feedback_received,
                                        self._last_measured_q, np.asarray(measured_q),
                                        self._last_measured_velocity, np.asarray(measured_velocity),
                                        lift_speed=lift_speed, base_dofs=self._base_dofs)
                self._feedback_position_uncertainty[self._base_dofs] = (0.0 if feedback_policy == "strict" else theta_uncertainty(
                    feedback_received, control_timestamp, feedback_policy, lift_speed=lift_speed))
                reserve = self._feedback_position_uncertainty[self._base_dofs]
                upper_reserve = (0. if self._normal_lift_upper is not None and control_phase == "lift-retreat"
                                 else reserve)
                measured_upper = (self.model.upperPositionLimit[self._base_dofs] if self._normal_lift_upper is not None
                                  else self._lift_measured_upper)
                if feedback_policy != THETA_RECORD_ONLY and not (
                        self.model.lowerPositionLimit[self._base_dofs] + reserve <= measured_q[self._base_dofs]
                        <= measured_upper - upper_reserve):
                    raise ValueError("Theta uncertainty reaches position boundary")
                self._last_feedback_received = dict(feedback_received)
            if self.reference_limit is not None:
                # Omitting the host timestamp denotes synchronous feedback.
                # Delayed-feedback callers must supply the independent clock.
                control_timestamp = state_timestamp if control_timestamp is None else control_timestamp
                if (state_timestamp is None or control_timestamp is None
                        or not np.isfinite(control_timestamp)
                        or not np.isfinite(state_timestamp)):
                    raise ValueError("MIT requires finite feedback and control timestamps")
                age = float(control_timestamp) - float(state_timestamp)
                self._feedback_age_s = age
                if age < -1e-12 or age > self.config.maximum_feedback_age_s + 1e-12:
                    raise ValueError("MIT feedback is future-dated or expired")
                if (self._last_control_timestamp is not None
                        and control_timestamp <= self._last_control_timestamp):
                    raise ValueError("MIT control timestamp must increase monotonically")
                self._last_control_timestamp = float(control_timestamp)
            self._validate_targets(targets)
            raw_measured_q = np.asarray(measured_q, dtype=float).reshape(-1).copy()
            measured_q = self._validate_configuration_vector(measured_q, recover_margin_excursion=True)
            validated_measured_q = measured_q
            # Validate before rate limiting: the legacy limiter can replace
            # supplied feedforward, which must not hide malformed input.
            for label, vectors in (
                ("target twist", target_twists),
                ("target acceleration", target_accelerations),
            ):
                if vectors is None:
                    continue
                if set(vectors) != set(TCP_FRAMES):
                    raise ValueError(f"{label} must cover exactly both TCP frames")
                for side in TCP_FRAMES:
                    vector = np.asarray(vectors[side], dtype=float)
                    if vector.shape != (6,) or not np.all(np.isfinite(vector)):
                        raise ValueError(f"{side} {label} must be a finite 6-vector")
            if state_timestamp is not None:
                state_timestamp = float(state_timestamp)
                if not np.isfinite(state_timestamp) or state_timestamp < 0.0:
                    raise ValueError("state timestamp must be finite and non-negative")
                if (feedback_received is None and self._last_state_timestamp is not None
                        and state_timestamp <= self._last_state_timestamp):
                    initial_repeat = (
                        self._allow_initial_timestamp_repeat
                        and state_timestamp == self._last_state_timestamp
                    )
                    if not initial_repeat:
                        if self.reference_limit is not None and state_timestamp == self._last_state_timestamp:
                            if not np.allclose(raw_measured_q, self._last_measured_q, atol=1e-12, rtol=0):
                                raise ValueError("repeated timestamp has changed feedback positions")
                            if v2_boundaries:
                                cycle_recovery_reason = "repeated_feedback"
                                self._transient_recovery_reason = cycle_recovery_reason
                            else:
                                self._stop_requested = True
                                self._stop_reason = "repeated_feedback"
                        else:
                            raise ValueError("state timestamp must increase monotonically")
            resolved_measured_velocity: np.ndarray | None = None
            if measured_velocity is not None:
                resolved_measured_velocity = self._validate_velocity_vector(measured_velocity)
            elif (
                self._last_measured_q is not None
                and state_timestamp is not None
                and self._last_state_timestamp is not None
            ):
                measured_dt = state_timestamp - self._last_state_timestamp
                if measured_dt == 0 and self.reference_limit is not None:
                    resolved_measured_velocity = self._last_measured_velocity
                elif measured_dt <= 0:
                    raise ValueError("velocity estimation requires a positive feedback interval")
                else:
                    resolved_measured_velocity = self._validate_velocity_vector(
                        pin.difference(self.model, self._last_measured_q, raw_measured_q)
                        / measured_dt
                    )
            if self._recovering and resolved_measured_velocity is not None and self.reference_limit is None:
                self._last_velocity = resolved_measured_velocity.copy()
                self._last_acceleration = np.zeros(self.model.nv)
                self.low_acceleration_task.set_last_integration(
                    self._last_velocity[self._base_dofs :], dt
                )
                self.acceleration_limit.set_last_integration(self._last_velocity, dt)
                self.jerk_limit.reset(self._last_velocity)
            if (self.reference_limit is not None
                    and feedback_received is None
                    and not self._allow_initial_timestamp_repeat
                    and state_timestamp == self._last_state_timestamp
                    and resolved_measured_velocity is not None
                    and self._last_measured_velocity is not None
                    and not np.allclose(resolved_measured_velocity, self._last_measured_velocity,
                                        atol=1e-12, rtol=0)):
                raise ValueError("repeated timestamp has changed feedback velocity")
            self._last_measured_q = (
                raw_measured_q.copy() if self.reference_limit is not None else measured_q.copy()
            )
            if resolved_measured_velocity is not None:
                self._last_measured_velocity = resolved_measured_velocity.copy()
            if state_timestamp is not None:
                self._last_state_timestamp = state_timestamp
                self._allow_initial_timestamp_repeat = False
            state_update_ms += 1000.0 * (time.perf_counter() - state_phase_start)
            configuration_update_start = time.perf_counter()
            if self.reference_limit is None:
                self.configuration.update(measured_q)
            else:
                self._measured_configuration.update(raw_measured_q)
                reference_q = measured_q.copy()
                reference_q[self._persistent_start:] = self._last_safe_positions[self._persistent_start:]
                self.configuration.update(reference_q)
            if self.reference_limit is not None:
                if resolved_measured_velocity is None:
                    raise ValueError("MIT control requires measured or causally estimated joint velocity")
                gravity_start = time.perf_counter()
                self._gravity_effort = self.gravity_model.evaluate(raw_measured_q, measured_gripper_positions)
                gravity_ms = 1000.0 * (time.perf_counter() - gravity_start)
                caps = self._manufacturer_capabilities
                physical = np.r_[([caps.base_linear_speed_m_s] * 2 +
                                 [self.config.base.angular_velocity_limit]) if self._base_dofs else [],
                                 caps.lift_velocity_m_s, caps.velocity_rad_s]
                bad = np.flatnonzero(np.abs(resolved_measured_velocity) > physical + self.config.constraint_tolerance)
                if len(bad):
                    i = int(bad[0])
                    raise ValueError(f"MIT measured overspeed axis={i} value={resolved_measured_velocity[i]} "
                                     f"manufacturer={physical[i]} effective={self.model.velocityLimit[i]} "
                                     f"reference={self.model.velocityLimit[i] * self.config.reference_velocity_fraction}")
                resultant_limit = min(caps.base_linear_speed_m_s, *self.model.velocityLimit[:2]) if (
                    self._base_dofs and self._base_resultant_constraint) else caps.base_linear_speed_m_s
                if self._base_dofs and np.linalg.norm(resolved_measured_velocity[:2]) > (
                        resultant_limit + self.config.constraint_tolerance):
                    raise ValueError("MIT measured base resultant overspeed")
                if np.any(np.abs(resolved_measured_velocity) >
                          self.model.velocityLimit + self.config.constraint_tolerance):
                    arm_start = self._base_dofs + 1
                    speed = np.abs(resolved_measured_velocity)
                    tolerance = self.config.constraint_tolerance
                    # Candidate-only recovery domain. Keep reference limits
                    # unchanged and let the ordinary constrained QP reject
                    # infeasible braking. This is NOT a physical guarantee.
                    recoverable = (
                        self.config.motion_profile_id in (
                            "x7-mit-integration-v1", "x7-mit-integration-load-02-v1",
                            "x7-mit-integration-v2", "x7-mit-integration-load-02-v2",
                            "x7-mit-integration-v3", "x7-mit-integration-load-02-v3")
                        or v2_boundaries
                    ) and not self._base_resultant_constraint and (
                        np.all(
                            speed[:self._base_dofs]
                            <= self.model.velocityLimit[:self._base_dofs] + tolerance
                        )
                        and speed[self._base_dofs]
                        <= 1.3 * self.model.velocityLimit[self._base_dofs] + tolerance
                        and np.all(speed[arm_start:] <= 1.2 * self.model.velocityLimit[arm_start:] + tolerance)
                        and np.all(np.abs(self._last_velocity) <= self.model.velocityLimit + tolerance)
                    )
                    if not recoverable:
                        raise ValueError("MIT measured overspeed outside candidate braking domain")
                    self._stop_requested = True
                    self._stop_reason = "measured_overspeed"
                self.reference_limit.update(
                    self._last_safe_positions, self._last_acceleration,
                    raw_measured_q, resolved_measured_velocity, self._gravity_effort, dt,
                    reference_velocity=self._last_velocity,
                )
            configuration_update_ms = 1000.0 * (time.perf_counter() - configuration_update_start)
            state_update_ms += configuration_update_ms - gravity_ms
            measured_collision_limits = []
            reaction_s = age + self.config.maximum_command_delay_s + dt
            if self.self_collision_system is not None:
                collision_start = time.perf_counter()
                measured_safety_travel = None
                reference_safety_travel = None
                if self.reference_limit is not None:
                    tracking_gap = np.abs(pin.difference(
                        self.model, self._measured_configuration.q, self.configuration.q
                    ))
                    if v2_boundaries:
                        # SDK velocity differentiation is too noisy to serve as
                        # a portable acceleration signal. Use the bounded
                        # reference acceleration with measured velocity/gap;
                        # plant lag is therefore still charged explicitly.
                        measured_acceleration = np.abs(self._last_acceleration)
                        measured_safety_travel = jerk_reaction_stopping_distance(
                            np.abs(resolved_measured_velocity),
                            measured_acceleration,
                            self._acceleration_limits,
                            self._jerk_limits,
                            reaction_s,
                        ) + tracking_gap
                        reference_safety_travel = jerk_reaction_stopping_distance(
                            np.abs(self._last_velocity),
                            np.abs(self._last_acceleration),
                            self._acceleration_limits,
                            self._jerk_limits,
                            reaction_s,
                        ) + tracking_gap
                    else:
                        measured_safety_travel = stopping_distance(
                            np.abs(resolved_measured_velocity), self._acceleration_limits,
                            self._acceleration_limits, self._jerk_limits,
                            age + dt + (
                                self.config.maximum_command_delay_s
                                if directional_boundaries else 0.0
                            ),
                        ) + tracking_gap
                        reference_safety_travel = measured_safety_travel
                reference_activation_travel = self._collision_activation_travel(
                    reference_safety_travel,
                    self._last_velocity,
                    dt,
                )
                measured_activation_travel = self._collision_activation_travel(
                    measured_safety_travel,
                    resolved_measured_velocity,
                    dt,
                )
                reference_collision_start = time.perf_counter()
                self.barriers = list(
                    self._reference_collision_system.update(
                        self.configuration,
                        dt,
                        joint_safety_travel=reference_safety_travel,
                        joint_position_uncertainty=self._feedback_position_uncertainty,
                        joint_activation_travel=reference_activation_travel,
                    )
                )
                collision_reference_ms = 1000.0 * (
                    time.perf_counter() - reference_collision_start
                )
                if self.reference_limit is not None:
                    offset = pin.difference(
                        self.model, self._measured_configuration.q, self.configuration.q
                    )
                    measured_collision_start = time.perf_counter()
                    measured_collision_limits = [
                        MeasuredCollisionLimit(
                            barrier, self._measured_configuration, offset,
                            method=self.config.measured_collision_method,
                        )
                        for barrier in self.self_collision_system.update(
                            self._measured_configuration,
                            dt,
                            joint_safety_travel=measured_safety_travel,
                            joint_position_uncertainty=self._feedback_position_uncertainty,
                            joint_activation_travel=measured_activation_travel,
                        )
                    ]
                    collision_measured_ms = 1000.0 * (
                        time.perf_counter() - measured_collision_start
                    )
                if directional_boundaries:
                    envelope = self.self_collision_system.collision_envelope
                    if envelope is None or not envelope.activation_only:
                        raise ValueError("lightweight control requires an activation-only envelope")
                    reference_limits = [
                        MeasuredCollisionLimit(
                            barrier, self.configuration, np.zeros(self.model.nv),
                            method="reference_endpoint_v2",
                        ) for barrier in self.barriers
                    ]
                    for name, limits, boundary_velocity in (
                        ("measured", measured_collision_limits, resolved_measured_velocity),
                        ("reference", reference_limits, self._last_velocity),
                    ):
                        for limit in limits:
                            barrier = limit.barrier
                            configuration = limit.measured_configuration
                            jacobian_start = time.perf_counter()
                            jacobian = barrier.compute_jacobian(configuration)
                            collision_jacobian_ms += 1000.0 * (
                                time.perf_counter() - jacobian_start
                            )
                            row_pack_start = time.perf_counter()
                            clearance = barrier.compute_barrier(configuration)
                            share_geometry = (
                                self.config.self_collision_runtime == "native_fused_v2"
                            )
                            qp_matrix = -jacobian / dt if share_geometry else None
                            limit.freeze_endpoint(
                                jacobian,
                                clearance,
                                copy=not share_geometry,
                                qp_matrix=qp_matrix,
                            )
                            boundary_limits.append(DirectionalVelocityLimit(
                                jacobian, clearance,
                                boundary_velocity, reaction_s,
                                gain_s_inv=self.config.boundary_gain_s_inv,
                                guard_distance=envelope.guard_distance_m,
                                labels=(
                                    ()
                                    if share_geometry
                                    else tuple(
                                        f"{name}:{item.pair.first}:{item.pair.second}"
                                        for item in barrier.distances
                                    )
                                ),
                                copy_geometry=not share_geometry,
                                qp_matrix=qp_matrix,
                                label_factory=(
                                    (lambda row, active=barrier, prefix=name:
                                     active.pair_label(row, prefix))
                                    if share_geometry
                                    else None
                                ),
                            ))
                            collision_row_pack_ms += 1000.0 * (
                                time.perf_counter() - row_pack_start
                            )
                    measured_collision_limits.extend(reference_limits)
                    self.barriers = []
                    collision_clearances = [
                        float(np.min(limit.reaction_clearance))
                        for limit in boundary_limits
                        if len(limit.reaction_clearance)
                    ]
                    minimum_collision_reaction_clearance = (
                        min(collision_clearances) if collision_clearances else None
                    )
                collision_distance_ms = 1000.0 * (time.perf_counter() - collision_start)
            if directional_boundaries:
                if self.self_collision_system is None and not v2_boundaries:
                    tracking_gap = np.abs(pin.difference(
                        self.model, self._measured_configuration.q, self.configuration.q
                    ))
                    measured_safety_travel = stopping_distance(
                        np.abs(resolved_measured_velocity), self._acceleration_limits,
                        self._acceleration_limits, self._jerk_limits, reaction_s,
                    ) + tracking_gap
                if v2_boundaries:
                    measured_joint_limit = self._joint_boundary_governor.limit(
                        raw_measured_q, resolved_measured_velocity, reaction_s,
                        self._feedback_position_uncertainty,
                    )
                    reference_joint_limit = self._joint_boundary_governor.limit(
                        self.configuration.q, self._last_velocity, reaction_s,
                        self._feedback_position_uncertainty,
                    )
                    boundary_limits.extend((measured_joint_limit, reference_joint_limit))
                    joint_clearances = [
                        float(np.min(limit.reaction_clearance))
                        for limit in (
                            measured_joint_limit,
                            reference_joint_limit,
                        )
                        if len(limit.reaction_clearance)
                    ]
                    minimum_joint_reaction_clearance = (
                        min(joint_clearances) if joint_clearances else None
                    )
                else:
                    boundary_limits.append(joint_boundary_damper(
                        self.model, np.arange(self._base_dofs + 1, self.model.nv),
                        raw_measured_q, self.configuration.q, resolved_measured_velocity,
                        measured_safety_travel, reaction_s, self.config.boundary_gain_s_inv,
                    ))
                if any(limit.reaction_clearance_exhausted for limit in boundary_limits):
                    self._stop_requested = True
                    self._stop_reason = self._stop_reason or "boundary_reaction_clearance"
            if self._lift_reference_governor is not None:
                measured_collision_limits.extend(self._lift_reference_governor.limits(
                    raw_measured_q, self._last_safe_positions, resolved_measured_velocity,
                    self._last_velocity, self._last_acceleration, reaction_s,
                    self._feedback_position_uncertainty, dt,
                    upper_retreat=self._normal_lift_upper is not None and control_phase == "lift-retreat"))
            if self._measured_lift_boundary_governor is not None:
                # Use the same validated origin as the next lift integration.
                # Raw physical feedback remains in the measured configuration
                # and the physical-limit/overspeed checks. At an existing soft
                # margin excursion this origin permits inward recovery only.
                # The previous position command is also reserved because the
                # position servo may still be pursuing it between IK updates.
                measured_collision_limits.extend(self._measured_lift_boundary_governor.limits(
                    self.configuration.q, self._last_safe_positions, resolved_measured_velocity,
                    self._last_velocity, self._last_acceleration, reaction_s,
                    self._feedback_position_uncertainty, dt))
            if self._base_boundary_governor is not None:
                # These coordinate the persistent acceleration/jerk history.
                # Preserve them in the hard problem, including braking retries.
                measured_collision_limits.extend(self._base_boundary_governor.limits(
                    raw_measured_q, self._last_safe_positions,
                    resolved_measured_velocity if resolved_measured_velocity is not None else self._last_velocity,
                    self._last_velocity, self._last_acceleration, reaction_s,
                    self._feedback_position_uncertainty, dt,
                ))
            if isinstance(self.lift_damping_task, TrackingDampingTask):
                # A position servo can still be travelling after the reference
                # has slowed. Damp that mismatch without resetting its position.
                self.lift_damping_task.lift_tracking_displacement = dt * (
                    self._last_velocity[self._base_dofs] - resolved_measured_velocity[self._base_dofs])
            task_phase_start = time.perf_counter()
            if self._posture_transition is not None:
                preference = self._posture_transition.advance(
                    dt, active=not self._stop_requested and posture_target is None)
                self.arm_posture_task.set_target(preference)

            task_configuration = (self._measured_configuration if self._measured_tracking_active
                                  else self.configuration)
            self.arm_kinematics_cache.refresh(task_configuration)
            for task in self.arm_manipulability_tasks.values():
                task.begin_cycle(task_configuration, dt)
            if self.config.allocator.mode == "continuous_v1":
                self._update_redundancy_allocation(targets, dt)
            measured_objectives = {}
            if self._measured_tracking_active and not self._stop_requested:
                self.measured_tracker.check_sample(target_sample, dt)
            for side, task in self.frame_tasks.items():
                if posture_target is not None and self.config.robot_model is not None:
                    # Preparation solves a joint-space objective. Its TCP path
                    # need not be feasible under the independent teleoperation
                    # shaper; an unused Cartesian planner must not brake D/I.
                    # Keep joint reference/acceleration/jerk and all hard rows.
                    break
                if self.reference_limit is not None and self._stop_requested:
                    # Braking uses joint history and the same safety rows,
                    # not a newly retargeted Cartesian trajectory.
                    break
                if self._measured_tracking_active:
                    measured_objectives[task] = self.measured_tracker.objective(
                        side, task, task_configuration, targets[side], dt)
                    task.set_target(targets[side])
                    self._limited_targets[side] = targets[side].copy()
                    continue
                previous_target = self._limited_targets[side]
                requested_twist = np.zeros(6)
                if target_twists is not None:
                    if side not in target_twists:
                        raise ValueError(f"missing {side!r} target twist")
                    requested_twist = np.asarray(target_twists[side], dtype=float)
                requested_acceleration = np.zeros(6)
                if target_accelerations is not None:
                    if side not in target_accelerations:
                        raise ValueError(f"missing {side!r} target acceleration")
                    requested_acceleration = np.asarray(target_accelerations[side], dtype=float)
                if side in self._target_planners:
                    try:
                        planner_start = time.perf_counter()
                        motion, limited = self._target_planners[side].step(
                            targets[side],
                            dt,
                            **({"sample": target_sample}
                               if isinstance(self._target_planners[side], ContinuousTargetMotionPlanner) else {}),
                            desired_body_twist=(None if target_twists is None else requested_twist),
                            desired_body_acceleration=(
                                None if target_accelerations is None else requested_acceleration
                            ),
                        )
                        planner_ms += 1000.0 * (time.perf_counter() - planner_start)
                    except TargetTrajectoryError as error:
                        self.target_planner_failures += 1
                        self.last_target_planner_error = f"{side}: {error}"
                        if self.reference_limit is None:
                            raise
                        if v2_boundaries:
                            cycle_recovery_reason = "target_trajectory_infeasible"
                            self._transient_recovery_reason = cycle_recovery_reason
                        else:
                            self._stop_requested = True
                            self._stop_reason = "target_trajectory_infeasible"
                        self._limited_targets = limited_target_state
                        for name, snapshot in target_planner_state.items():
                            self._target_planners[name].restore(snapshot)
                            self.frame_tasks[name].set_target(self._limited_targets[name])
                        self._restore_allocator_state(allocator_state)
                        break
                    limited_target = motion.pose
                    # Feed forward the bounded reference while it is moving,
                    # including its approach to a now-stationary raw target.
                    desired_twist = (
                        motion.body_twist if self.config.target_planner
                        in {"c2_tracking_v4", "c2_tracking_v5", "c2_tracking_v6"}
                        or not limited or self._target_planners[side].request_is_moving else np.zeros(6)
                    )
                else:
                    limited_target, limited = self._limit_target(
                        previous_target,
                        targets[side],
                        dt,
                    )
                    desired_twist = requested_twist
                    if limited:
                        desired_twist = pin.log(previous_target.actInv(limited_target)).vector / dt
                self._limited_targets[side] = limited_target
                feedforward = self._frame_feedforward(
                    task, limited_target, desired_twist,
                    motion.body_twist if side in self._target_planners else desired_twist,
                    resolved_measured_velocity,
                )
                task.set_target_motion(limited_target, feedforward, dt)
                target_limited = target_limited or limited

            if (
                not self._stop_requested
                and cycle_recovery_reason is None
            ):
                if self.config.allocator.mode != "continuous_v1":
                    self._update_redundancy_allocation(self._limited_targets, dt)
                self._update_settling_regularization(dt)

            task_construction_ms = 1000.0 * (time.perf_counter() - task_phase_start)
            self.configuration.check_limits(safety_break=True)
            qp_build_start = time.perf_counter()
            phase_limits = []
            lock_rows = self._lock_rows
            if control_phase != "tracking":
                from .preparation import PreparationLimit

                if self._preparation_limit is None:
                    self._preparation_limit = PreparationLimit(self.model, self.jerk_limit, dt,
                                                                lift_velocity=self.config.preparation_lift_velocity_m_s)
                self._preparation_limit.inward = control_phase == "lift-retreat"
                phase_limits.append(self._preparation_limit)
                indices = np.r_[np.arange(3), np.arange(4, 18)] if control_phase == "lift-retreat" else np.arange(3)
                lock_rows = np.vstack((lock_rows, np.eye(self.model.nv)[indices]))
            problem, hard_problem, warnings = assemble_problem(
                self.configuration, self.tasks, dt, self.config.qp_regularization,
                [*self.limits, *measured_collision_limits, *phase_limits], self.barriers, lock_rows,
                boundary_limits if directional_boundaries else [], posture_target, posture_velocity,
                task_configuration=task_configuration if self._measured_tracking_active else None,
                task_objectives=measured_objectives)
            if self.following_governor is not None:
                if control_phase == 'tracking' and not self._stop_requested and cycle_recovery_reason is None:
                    arm_start = self._base_dofs + 1
                    scale = self.following_governor.update(
                        self._last_safe_positions[arm_start:], raw_measured_q[arm_start:],
                        self._last_velocity[arm_start:], resolved_measured_velocity[arm_start:], dt)
                    # For an unconstrained QP this scales its desired joint
                    # displacement. Keep every constraint, including command
                    # acceleration/jerk and measured-state effort bounds.
                    # Never scale an already-solved command after validation.
                    problem.q = problem.q * scale
                else:
                    self.following_governor.reset()
            if self.reference_limit is not None and (
                self._stop_requested or cycle_recovery_reason is not None
            ):
                braking_velocity, _ = self._braking_velocity(dt)
                # Project the braking proposal onto exactly the same limits
                # as tracking. This is not an unconstrained emergency reset.
                problem = qpsolvers.Problem(
                    P=np.eye(self.model.nv), q=-braking_velocity * dt,
                    G=(hard_problem.G if cycle_recovery_reason is not None else problem.G),
                    h=(hard_problem.h if cycle_recovery_reason is not None else problem.h),
                    A=hard_problem.A,
                    b=hard_problem.b,
                    lb=hard_problem.lb,
                    ub=hard_problem.ub,
                )
            qp_build_ms = 1000.0 * (time.perf_counter() - qp_build_start)
            qp_solve_start = time.perf_counter()
            # Pink solves for displacement, so residuals are amplified by dt^-3
            # when checking jerk. Tighten QP feasibility, not the safety check.
            solver_options = {}
            if self.config.qp_primal_tolerance is not None:
                solver_options["primal_tol"] = self.config.qp_primal_tolerance
            result = self._solve_active_problem(problem, **solver_options)
            if (directional_boundaries and warnings
                    and (not result.found or result.x is None)):
                # Only the early-slowdown rows are removed. Every hard row is
                # reused verbatim; no second geometry query or IK build.
                if v2_boundaries:
                    cycle_recovery_reason = "boundary_constraints_infeasible"
                    self._transient_recovery_reason = cycle_recovery_reason
                else:
                    self._stop_requested = True
                    self._stop_reason = self._stop_reason or "boundary_constraints_infeasible"
                braking_retries = 1
                braking_velocity, _ = self._braking_velocity(dt)
                problem = qpsolvers.Problem(
                    P=np.eye(self.model.nv), q=-braking_velocity * dt,
                    G=hard_problem.G, h=hard_problem.h,
                    A=hard_problem.A, b=hard_problem.b, lb=hard_problem.lb, ub=hard_problem.ub,
                )
                self._limited_targets = limited_target_state
                for name, snapshot in target_planner_state.items():
                    self._target_planners[name].restore(snapshot)
                    self.frame_tasks[name].set_target(self._limited_targets[name])
                self._restore_allocator_state(allocator_state)
                result = self._solve_active_problem(problem, **solver_options)
            qp_solve_ms = 1000.0 * (time.perf_counter() - qp_solve_start)
            output_check_start = time.perf_counter()
            if not result.found or result.x is None:
                try:
                    labels = []
                    for limit in [*self.limits, *measured_collision_limits, *phase_limits]:
                        inequalities = limit.compute_qp_inequalities(self.configuration, dt)
                        if inequalities is not None:
                            barrier = getattr(limit, "barrier", None)
                            distances = getattr(barrier, "distances", ())
                            if len(distances) == len(inequalities[1]):
                                labels.extend(f"measured_collision:{item.pair.first}:{item.pair.second}"
                                              for item in distances)
                            else:
                                labels.extend([type(limit).__name__] * len(inequalities[1]))
                    for barrier in self.barriers:
                        labels.extend(f"reference_collision:{item.pair.first}:{item.pair.second}"
                                      for item in barrier.distances)
                    # The retry can remove warning rows; label only rows in the failed problem.
                    if len(labels) < len(problem.h):
                        labels.extend(["boundary_warning"] * (len(problem.h) - len(labels)))
                    self.last_qp_failure = QPFailureSnapshot.capture(
                        self._solved_problem,
                        (np.asarray(labels)[np.any(np.asarray(problem.G)[:, self.active_indices] != 0., axis=1)]
                         if len(self.active_indices) != self.model.nv else labels),
                        active_indices=self.active_indices,
                        joint_names=np.asarray([*(BASE_JOINTS if self._base_dofs else ()),
                                                LIFT_JOINT, *ARM_JOINTS]),
                        reference_q=self.configuration.q,
                        measured_q=raw_measured_q,
                        measured_velocity=resolved_measured_velocity,
                        reference_velocity=self._last_velocity,
                        reference_acceleration=self._last_acceleration,
                        control_dt_s=dt, feedback_age_s=age,
                        locked_indices=self._locked_indices,
                        effort_preview_velocity_bounds=(None if self.effort_preview is None
                                                        else self.effort_preview.bounds),
                    )
                except Exception as evidence_error:
                    # Diagnostic errors cannot replace the original solver failure.
                    self.last_qp_capture_error = f"{type(evidence_error).__name__}: {evidence_error}"
                raise NoSolutionFound(problem, result)
            velocity = np.asarray(result.x) / dt
            if not np.all(np.isfinite(velocity)):
                raise ValueError("Pink returned non-finite joint commands")
            positions = pin.integrate(self.model, measured_q, velocity * dt)
            if self.reference_limit is not None:
                arm_slice = slice(self._persistent_start, None)
                positions[arm_slice] = self._last_safe_positions[arm_slice] + velocity[arm_slice] * dt
            if v2_boundaries and self._collision_sweep_system is not None:
                sweep_start = time.perf_counter()
                sweep_clear = self._swept_collision_is_clear(raw_measured_q, positions, dt)
                swept_collision_ms += 1000.0 * (time.perf_counter() - sweep_start)
                if not sweep_clear:
                    cycle_recovery_reason = "swept_collision_rejected"
                    self._transient_recovery_reason = cycle_recovery_reason
                    braking_retries += 1
                    braking_velocity, _ = self._braking_velocity(dt)
                    recovery_problem = qpsolvers.Problem(
                        P=np.eye(self.model.nv),
                        q=-braking_velocity * dt,
                        G=hard_problem.G,
                        h=hard_problem.h,
                        A=hard_problem.A,
                        b=hard_problem.b,
                        lb=hard_problem.lb,
                        ub=hard_problem.ub,
                    )
                    retry_start = time.perf_counter()
                    result = self._solve_active_problem(recovery_problem, **solver_options)
                    qp_solve_ms += 1000.0 * (time.perf_counter() - retry_start)
                    problem = recovery_problem
                    self._limited_targets = limited_target_state
                    for name, snapshot in target_planner_state.items():
                        self._target_planners[name].restore(snapshot)
                        self.frame_tasks[name].set_target(self._limited_targets[name])
                    self._restore_allocator_state(allocator_state)
                    if not result.found or result.x is None:
                        raise NoSolutionFound(recovery_problem, result)
                    velocity = np.asarray(result.x) / dt
                    positions = pin.integrate(self.model, measured_q, velocity * dt)
                    positions[arm_slice] = (
                        self._last_safe_positions[arm_slice] + velocity[arm_slice] * dt
                    )
                    sweep_start = time.perf_counter()
                    sweep_clear = self._swept_collision_is_clear(
                        raw_measured_q, positions, dt
                    )
                    swept_collision_ms += 1000.0 * (time.perf_counter() - sweep_start)
                    if not sweep_clear:
                        self._stop_requested = True
                        self._stop_reason = "swept_collision_exhausted"
                        raise ValueError("hard-constrained braking failed swept collision validation")
            if not np.all(np.isfinite(positions)):
                raise ValueError("Pink returned non-finite joint commands")
            if np.any(positions < self.model.lowerPositionLimit - 1e-7) or np.any(
                positions > self.model.upperPositionLimit + 1e-7
            ):
                raise ValueError("Pink returned a command outside joint margins")
            tolerance = self.config.constraint_tolerance
            acceleration = (velocity - self._last_velocity) / dt
            jerk = (acceleration - self._last_acceleration) / dt
            if np.any(np.abs(velocity) > self.model.velocityLimit + tolerance):
                raise ValueError("Pink returned a command outside velocity limits")
            if np.any(np.abs(acceleration) > self._acceleration_limits + tolerance):
                raise ValueError("Pink returned a command outside acceleration limits")
            if np.any(np.abs(jerk) > self._jerk_limits + tolerance):
                raise ValueError("Pink returned a command outside jerk limits")

            # Validate the wire command before committing the reference history.
            mit_command = self._mit_command(
                positions, velocity,
                control_timestamp if self.reference_limit is not None else state_timestamp,
                acceleration=acceleration, measured_q=raw_measured_q,
                measured_velocity=resolved_measured_velocity,
            )
            if mit_command is not None:
                from .hardware_capabilities import effort_breakdown

                self.last_mit_effort = effort_breakdown(
                    mit_command, raw_measured_q[self._base_dofs + 1:],
                    resolved_measured_velocity[self._base_dofs + 1:],
                    self.effective_mit_servo.torque_limit_nm,
                )
                planning_bounds = (np.asarray(self.effective_mit_servo.torque_limit_nm)
                                   * self.config.mit_planning_effort_fraction)
                self.last_mit_effort["planning_bounds_nm"] = planning_bounds
                self.last_mit_effort["planning_reserve_nm"] = (
                    np.asarray(self.effective_mit_servo.torque_limit_nm) - planning_bounds)
                self.last_mit_effort["planning_remaining_nm"] = (
                    planning_bounds - np.abs(self.last_mit_effort["requested_nm"]))
                if self.effort_preview is not None:
                    self.last_mit_effort["braking_preview"] = self.effort_preview.report(velocity, dt)
                bad = np.flatnonzero(self.last_mit_effort["remaining_nm"] < -1e-7)
                if len(bad):
                    raise ValueError("MIT total operating torque exceeded: " + ", ".join(ARM_JOINTS[i] for i in bad))
            base_command = self._base_command(positions, velocity)
            output_checks_ms = 1000.0 * (time.perf_counter() - output_check_start)
            self._deadline_stage = "command_ready"
            self._validate_dispatch_freshness(solve_start, age)
            self._last_safe_positions = positions.copy()
            self.low_acceleration_task.set_last_integration(velocity[self._base_dofs :], dt)
            self.acceleration_limit.set_last_integration(velocity, dt)
            self.jerk_limit.set_last_integration(velocity, dt)
            self._last_velocity = velocity.copy()
            self._last_acceleration = acceleration.copy()
            self._last_jerk = jerk.copy()
            if cycle_recovery_reason is not None:
                acceleration_utilization = float(
                    np.max(np.abs(acceleration) / self._acceleration_limits)
                )
                if acceleration_utilization <= 0.05:
                    self._transient_recovery_reason = None
            self._recovering = False
            solve_time_ms = qp_solve_ms

            errors = self._task_errors()
            if (
                max(errors[:2]) > self.config.large_position_error
                or max(errors[2:]) > self.config.large_orientation_error
            ):
                progress = self._measured_tracking_active and self.measured_tracker.making_progress(
                    max(max(errors[:2])/self.config.large_position_error,
                        max(errors[2:])/self.config.large_orientation_error), targets)
                self._large_error_time = 0. if progress else self._large_error_time + dt
            else:
                self._large_error_time = 0.0
                if self._measured_tracking_active:
                    self.measured_tracker.progress_error = float('inf')
            if self._large_error_time >= self.config.large_error_delay:
                status = "large/unreachable target error"
            elif target_limited:
                status = (
                    "target braking"
                    if self.config.target_planner == "c2_stop_aware_v1"
                    else "target rate-limited"
                )
            else:
                status = "tracking"
            if self.reference_limit is not None:
                if self._stop_requested:
                    stopped = (
                        np.max(np.abs(velocity)) <= 1e-5
                        and np.max(np.abs(resolved_measured_velocity)) <= 1e-3
                    )
                    if not stopped:
                        self._hold_since_s = None
                    elif self._hold_since_s is None:
                        self._hold_since_s = float(control_timestamp)
                    stable = (
                        self._hold_since_s is not None
                        and float(control_timestamp) - self._hold_since_s >= 0.2 - 1e-12
                    )
                    self.control_state = "HOLD" if stable else "BRAKING"
                else:
                    self.control_state = "TRACKING"
                # Do not overwrite a measured-pose tracking failure with the
                # motor state: drag_control checks this before dispatch.
                if not (self._measured_tracking_active and not self._stop_requested
                        and status == "large/unreachable target error"):
                    status = (
                        f"RECOVERING: {cycle_recovery_reason}"
                        if cycle_recovery_reason is not None and not self._stop_requested
                        else self.control_state
                    )
            timings = IKPhaseTimings(
                state_update_ms=state_update_ms,
                collision_distance_ms=collision_distance_ms,
                collision_reference_ms=collision_reference_ms,
                collision_measured_ms=collision_measured_ms,
                collision_jacobian_ms=collision_jacobian_ms,
                collision_row_pack_ms=collision_row_pack_ms,
                task_construction_ms=task_construction_ms,
                planner_ms=planner_ms,
                allocator_ms=allocator_ms,
                qp_build_ms=qp_build_ms,
                qp_solve_ms=qp_solve_ms,
                swept_collision_ms=swept_collision_ms,
            )
            diagnostics_start = time.perf_counter()
            diagnostics = self._diagnostics(
                status,
                solve_time_ms,
                target_limited,
                raw_targets=targets,
                command_positions=positions,
                timings=timings,
            )
            diagnostics_ms = 1000.0 * (time.perf_counter() - diagnostics_start)
            boundary_reasons = tuple(label for limit in boundary_limits
                                     for label in limit.binding_labels(velocity))
            diagnostics = replace(
                diagnostics, boundary_limiting_reasons=boundary_reasons,
                braking_retries=braking_retries,
                recovery_reason=cycle_recovery_reason,
                minimum_collision_reaction_clearance_m=minimum_collision_reaction_clearance,
                minimum_joint_reaction_clearance_rad=minimum_joint_reaction_clearance,
                target_limited=target_limited or bool(boundary_reasons),
                target_settled=diagnostics.target_settled and not boundary_reasons,
            )
            wall_ms = 1000.0 * (time.perf_counter() - solve_start)
            cpu_ms = 1000.0 * (time.thread_time() - thread_cpu_start)
            timings = replace(
                timings, gravity_ms=gravity_ms, output_checks_ms=output_checks_ms,
                diagnostics_ms=diagnostics_ms,
                total_ms=wall_ms, critical_wall_ms=wall_ms,
                thread_cpu_ms=cpu_ms, scheduler_delay_ms=max(0.0, wall_ms - cpu_ms),
            )
            diagnostics = replace(diagnostics, timings=timings)
            self._deadline_stage = "return_ready"
            self._validate_dispatch_freshness(solve_start, age)
            diagnostics = replace(
                diagnostics,
                dispatch_state_age_s=self._dispatch_state_age_s,
                deadline_reject_count=self._deadline_reject_count,
            )
            return IKCommand(positions, velocity, diagnostics, base_command, mit_command)
        except Exception as error:  # noqa: BLE001 -- controller boundary returns an explicit fault.
            from .faults import describe_control_fault

            self._fault_info = describe_control_fault(error)
            if isinstance(error, NoSolutionFound) and self.last_qp_failure is not None:
                conflicts = self.last_qp_failure.scalar_conflicts()
                self._fault_info[1]["scalar_conflicts"] = conflicts
                if conflicts:
                    self._fault_info[1]["qp_feasibility"] = "infeasible_scalar_intervals"
            solve_time_ms = qp_solve_ms
            self._restore_allocator_state(allocator_state)
            self._limited_targets = limited_target_state
            for side, snapshot in target_planner_state.items():
                self._target_planners[side].restore(snapshot)
            for side, task in self.frame_tasks.items():
                task.set_target(self._limited_targets[side])
            if self.reference_limit is not None:
                (self._last_safe_positions, self._last_velocity,
                 self._last_acceleration, self._last_jerk) = reference_state
                self.low_acceleration_task.set_last_integration(self._last_velocity[self._base_dofs :], dt)
                self.acceleration_limit.set_last_integration(self._last_velocity, dt)
                self.jerk_limit.reset(self._last_velocity, self._last_acceleration)
                # Until a fallback has passed the same output constraints as a
                # normal QP, it is not an executable braking command. Preserve
                # histories rather than fabricating an instantaneous stop.
                self._mit_fault_latched = f"{type(error).__name__}: {error}"
                fault_wall_ms = 1000.0 * (time.perf_counter() - solve_start)
                fault_cpu_ms = 1000.0 * (time.thread_time() - thread_cpu_start)
                fault_timings = IKPhaseTimings(
                    gravity_ms=gravity_ms, output_checks_ms=output_checks_ms,
                    state_update_ms=state_update_ms,
                    collision_distance_ms=collision_distance_ms,
                    collision_reference_ms=collision_reference_ms,
                    collision_measured_ms=collision_measured_ms,
                    collision_jacobian_ms=collision_jacobian_ms,
                    collision_row_pack_ms=collision_row_pack_ms,
                    task_construction_ms=task_construction_ms,
                    planner_ms=planner_ms, allocator_ms=allocator_ms,
                    qp_build_ms=qp_build_ms, qp_solve_ms=qp_solve_ms,
                    swept_collision_ms=swept_collision_ms,
                    total_ms=fault_wall_ms, critical_wall_ms=fault_wall_ms,
                    thread_cpu_ms=fault_cpu_ms,
                    scheduler_delay_ms=max(0.0, fault_wall_ms - fault_cpu_ms),
                )
                return self._mit_fault_command(
                    self._mit_fault_latched, started=solve_start, timings=fault_timings
                )
            if validated_measured_q is not None:
                previous_acceleration = self._last_acceleration.copy()
                braking_velocity, braking_acceleration = self._braking_velocity(dt)
                braking_positions = pin.integrate(
                    self.model, validated_measured_q, braking_velocity * dt
                )
                if self.reference_limit is not None:
                    arm_slice = slice(self._persistent_start, None)
                    braking_positions[arm_slice] = (
                        self._last_safe_positions[arm_slice] + braking_velocity[arm_slice] * dt
                    )
                if (
                    np.all(braking_positions >= self.model.lowerPositionLimit - 1e-7)
                    and np.all(braking_positions <= self.model.upperPositionLimit + 1e-7)
                ):
                    self._last_safe_positions = braking_positions
                    self._last_velocity = braking_velocity
                    self._last_acceleration = braking_acceleration
                    self._last_jerk = (braking_acceleration - previous_acceleration) / dt
                else:
                    self._last_velocity = np.zeros(self.model.nv)
                    self._last_acceleration = np.zeros(self.model.nv)
                    self._last_jerk = np.zeros(self.model.nv)
            else:
                self._last_velocity = np.zeros(self.model.nv)
                self._last_acceleration = np.zeros(self.model.nv)
                self._last_jerk = np.zeros(self.model.nv)
            self.low_acceleration_task.set_last_integration(self._last_velocity[self._base_dofs :], dt)
            self.acceleration_limit.set_last_integration(self._last_velocity, dt)
            self.jerk_limit.reset(self._last_velocity, self._last_acceleration)
            self._recovering = True
            fault = f"{type(error).__name__}: {error}"
            timings = IKPhaseTimings(
                state_update_ms=state_update_ms,
                collision_distance_ms=collision_distance_ms,
                collision_reference_ms=collision_reference_ms,
                collision_measured_ms=collision_measured_ms,
                collision_jacobian_ms=collision_jacobian_ms,
                collision_row_pack_ms=collision_row_pack_ms,
                task_construction_ms=task_construction_ms,
                qp_build_ms=qp_build_ms,
                qp_solve_ms=qp_solve_ms,
            )
            diagnostics_start = time.perf_counter()
            diagnostics = self._diagnostics(
                "holding last safe command",
                solve_time_ms,
                target_limited,
                raw_targets=(targets if validated_measured_q is not None else None),
                command_positions=self._last_safe_positions,
                timings=timings,
                fault=fault,
            )
            diagnostics_ms = 1000.0 * (time.perf_counter() - diagnostics_start)
            positions = self._last_safe_positions.copy()
            velocity = self._last_velocity.copy()
            mit_command = self._mit_command(positions, velocity, self._last_state_timestamp)
            base_command = self._base_command(positions, velocity)
            timings = replace(
                timings,
                diagnostics_ms=diagnostics_ms,
                total_ms=1000.0 * (time.perf_counter() - solve_start),
            )
            diagnostics = replace(diagnostics, timings=timings)
            return IKCommand(positions, velocity, diagnostics, base_command, mit_command)

    def _mit_fault_command(
        self, fault: str, *, started: float | None = None,
        timings: IKPhaseTimings | None = None,
    ) -> IKCommand:
        started = time.perf_counter() if started is None else started
        self.control_state = "FAULT"
        diagnostics = self._diagnostics(
            "FAULT: non-executable MIT command", 0.0, False,
            command_positions=self._last_safe_positions, fault=fault,
        )
        diagnostics = replace(diagnostics, timings=(
            IKPhaseTimings(total_ms=1000.0 * (time.perf_counter() - started))
            if timings is None else timings
        ))
        return IKCommand(
            self._last_safe_positions.copy(), self._last_velocity.copy(),
            diagnostics, executable=False,
        )

    def _mit_command(self, positions, velocity, timestamp, *, acceleration=None, measured_q=None,
                     measured_velocity=None) -> MitCommand | None:
        servo = self.config.mit_servo
        if servo is None:
            return None
        arm_slice = slice(self._base_dofs + 1, None)
        torque_ff = self._gravity_effort[arm_slice].copy()
        self.last_dynamic_feedforward = np.zeros(len(ARM_JOINTS))
        feedforward = self.config.mit_dynamic_feedforward
        if (feedforward is not None and acceleration is not None and measured_q is not None
                and measured_velocity is not None):
            # Solved-reference friction/inertia feedforward, clipped to a fraction of the
            # remaining peak-torque headroom after PD + gravity. Hold commands never use it.
            extra = feedforward.evaluate(self.gravity_model, measured_q, velocity[arm_slice],
                                         acceleration[arm_slice])
            servo_limits = np.asarray(self.effective_mit_servo.torque_limit_nm)
            pd = (np.asarray(servo.kp) * (positions[arm_slice] - measured_q[arm_slice])
                  + np.asarray(servo.kd) * (velocity[arm_slice] - measured_velocity[arm_slice]))
            headroom = feedforward.headroom_fraction * np.maximum(0.0, servo_limits - np.abs(pd + torque_ff))
            extra = np.clip(extra, -headroom, headroom)
            torque_ff = torque_ff + extra
            self.last_dynamic_feedforward = extra
        return MitCommand(
            ARM_JOINTS, positions[arm_slice], velocity[arm_slice], torque_ff,
            servo.kp, servo.kd, 0.0 if timestamp is None else float(timestamp),
        )
