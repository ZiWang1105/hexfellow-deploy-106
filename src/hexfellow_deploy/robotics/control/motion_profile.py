"""Versioned, hash-bound motion limits shared by simulation and hardware."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import yaml

from .allocator import DEFAULT_WHOLE_BODY_ALLOCATOR_CONFIG, WholeBodyAllocatorConfig
from .posture_policy import HumanPosturePolicy
from .robot_model import RobotModel, load_robot_model

if TYPE_CHECKING:
    from .mit import DynamicFeedforward, GravityCorrection, GravityPayload, MitServoConfig

from ..paths import CONFIG_DIR, PROJECT_ROOT

SAFE_MOTION_PROFILE_ID = "x7-safe-v1"
DEFAULT_MOTION_PROFILE_ID = SAFE_MOTION_PROFILE_ID
MOTION_PROFILE_DIR = CONFIG_DIR / "control" / "motion_profiles"

# Simulation opts in independently; generic and hardware callers retain SAFE.
SIMULATION_MOTION_PROFILE_ID = SAFE_MOTION_PROFILE_ID
CONTROLLER_TUNING_KEYS = (
    "position_cost",
    "orientation_cost",
    "frame_gain",
    "frame_lm_damping",
    "low_acceleration_cost",
    "simulation_arm_servo_kv",
    "lift_velocity_damping_scale",
    "target_feedforward_gain",
    "target_velocity_feedback_gain",
    "qp_primal_tolerance",
)
SCHEMA_7_CONTROLLER_TUNING_KEYS = (*CONTROLLER_TUNING_KEYS, "qp_regularization")
SCHEMA_8_CONTROLLER_TUNING_KEYS = (
    *SCHEMA_7_CONTROLLER_TUNING_KEYS,
    "settled_low_acceleration_cost",
    "settled_arm_damping_cost",
    "settling_blend_s",
    "settling_linear_velocity_m_s",
    "settling_angular_velocity_rad_s",
    "settling_position_error_m",
    "settling_orientation_error_rad",
)
SCHEMA_13_CONTROLLER_TUNING_KEYS = (
    *SCHEMA_8_CONTROLLER_TUNING_KEYS,
    "arm_posture_cost",
    "arm_posture_min_fraction",
    "arm_manipulability_cost",
    "arm_manipulability_rate",
    "arm_manipulability_floor_fraction",
    "arm_manipulability_exit_fraction",
    "arm_manipulability_activation_ramp_s",
)
SCHEMA_8_ALLOCATOR_KEYS = {
    "mode",
    "arm_limit_cost",
    "lift_active_cost",
    "base_active_cost",
    "pressure_midpoint_fraction",
    "pressure_width_fraction",
    "base_activation_distances_m",
    "lift_activation_distances_m",
    "activation_steepness",
    "dominance_epsilon_m",
    "activation_attack_s",
    "activation_release_s",
    "filter_direction_width",
    "lift_reference_error_gain",
    "base_enter_distance_m",
    "base_exit_distance_m",
    "lift_enter_distance_m",
    "lift_exit_distance_m",
    "pressure_enter",
    "pressure_exit",
    "engage_dwell_s",
    "release_dwell_s",
    "base_inactive_cost",
    "lift_inactive_cost",
    "base_anchor_follow_linear_m_s",
    "base_anchor_follow_angular_rad_s",
    "lift_anchor_follow_m_s",
    "rebase_position_error_m",
}
SCHEMA_13_ALLOCATOR_KEYS = SCHEMA_8_ALLOCATOR_KEYS | {
    "posture_assist_enter_rad",
    "posture_assist_exit_rad",
    "common_yaw_enter_rad",
    "common_yaw_exit_rad",
    "soft_assist_activation",
}
SCHEMA_14_ALLOCATOR_KEYS = SCHEMA_13_ALLOCATOR_KEYS | {
    "lift_common_translation_share",
    "base_common_translation_share",
    "base_common_yaw_share",
}
SCHEMA_15_BODY_LIMIT_KEYS = {
    "base_linear_velocity_m_s",
    "base_angular_velocity_rad_s",
    "base_linear_acceleration_m_s2",
    "base_angular_acceleration_rad_s2",
    "base_linear_jerk_m_s3",
    "base_angular_jerk_rad_s3",
    "lift_velocity_m_s",
    "lift_acceleration_m_s2",
    "lift_jerk_m_s3",
}


def _canonical_sha256(value: Mapping[str, object]) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _positive_vector(value: object, *, label: str, length: int) -> tuple[float, ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ValueError(f"{label} must contain {length} values")
    result = tuple(float(item) for item in value)
    if len(result) != length or any(not np.isfinite(item) or item <= 0.0 for item in result):
        raise ValueError(f"{label} must contain {length} finite positive values")
    return result


def _positive(value: object, *, label: str) -> float:
    result = float(value)
    if not np.isfinite(result) or result <= 0.0:
        raise ValueError(f"{label} must be finite and positive")
    return result


def _project_file(value: object, *, label: str) -> Path:
    path = (PROJECT_ROOT / str(value)).resolve()
    try:
        path.relative_to(PROJECT_ROOT)
    except ValueError as error:
        raise ValueError(f"{label} leaves the Policy_Sim repository: {path}") from error
    if not path.is_file():
        raise FileNotFoundError(f"{label} does not exist: {path}")
    return path


@dataclass(frozen=True)
class MotionProfile:
    """Validated motion contract and its canonical content hash."""

    path: Path
    schema_version: int
    profile_id: str
    sha256: str
    robot: str
    arm_model: str
    control_dt_s: float
    simulation_status: str
    hardware_status: str
    target_linear_velocity_m_s: float
    target_angular_velocity_rad_s: float
    target_linear_acceleration_m_s2: float | None
    target_linear_jerk_m_s3: float | None
    target_angular_acceleration_rad_s2: float | None
    target_angular_jerk_rad_s3: float | None
    target_planner: str
    target_translation_deadband_m: float
    target_orientation_deadband_rad: float
    target_stationary_dwell_s: float
    target_limit_check_runtime: str
    arm_velocity_rad_s: tuple[float, ...]
    arm_acceleration_rad_s2: tuple[float, ...]
    arm_jerk_rad_s3: tuple[float, ...]
    arm_position_margin_rad: float
    base_linear_velocity_m_s: float
    base_angular_velocity_rad_s: float
    base_linear_acceleration_m_s2: float
    base_angular_acceleration_rad_s2: float
    base_linear_jerk_m_s3: float
    base_angular_jerk_rad_s3: float
    lift_velocity_m_s: float
    lift_acceleration_m_s2: float
    lift_jerk_m_s3: float
    urdf_velocity_rad_s: tuple[float, ...]
    sdk_velocity_rad_s: tuple[float, ...]
    velocity_limit_authority: str
    urdf_position_lower_rad: tuple[float, ...]
    urdf_position_upper_rad: tuple[float, ...]
    sdk_position_lower_rad: tuple[float, ...] | None
    sdk_position_upper_rad: tuple[float, ...] | None
    collision_geometry_config: Path | None
    collision_geometry_sha256: str
    collision_envelope_config: Path | None
    collision_pair_certificate_config: Path | None
    collision_pair_certificate_sha256: str | None
    collision_runtime: str
    native_collision_kernel_abi: int | None
    controller_tuning: Mapping[str, float]
    allocator: WholeBodyAllocatorConfig
    raw: Mapping[str, object]
    mit_servo: MitServoConfig | None = None
    gravity_payloads: tuple[GravityPayload, ...] = ()
    maximum_feedback_age_s: float = 0.04
    gravity_urdf_sha256: str | None = None
    measured_collision_method: str = "reference_velocity_v1"
    boundary_policy: str = "legacy"
    boundary_gain_s_inv: float = 1.0
    maximum_command_delay_s: float = 0.02
    collision_sweep_max_joint_step_rad: float | None = None
    collision_sweep_max_intervals: int = 0
    collision_activation_policy: str = "static_worst_case_v2"
    motion_envelope: str = "none"  # Runtime envelope is separate from the base content hash.
    trajectory_kernel_version: int | None = None
    posture_policy: HumanPosturePolicy | None = None
    reference_velocity_fraction: float = 1.
    lift_reference_velocity_fraction: float = 1.
    tracking_objective: str = "legacy"
    stationary_arrival: str = "legacy"
    mit_effort_preview: str = "none"
    mit_gravity_correction: GravityCorrection | None = None
    mit_dynamic_feedforward: DynamicFeedforward | None = None
    mit_planning_effort_fraction: float = 1.0
    lift_reference_max_error_m: float | None = None
    base_workspace_mode: str = "bounded"
    base_boundary_governor: str = "legacy"
    robot_model: RobotModel | None = None
    self_collision_enabled: bool = True
    preparation_lift_velocity_m_s: float = .02
    base_control_enabled: bool = True
    stationary_redundancy_hold: bool = False

    @property
    def hardware_commissioned(self) -> bool:
        return self.hardware_status == "commissioned"

    def artifact_metadata(self) -> dict[str, object]:
        return {
            "robot_model": load_robot_model(self.robot_model).report(),
            "self_collision_enabled": self.self_collision_enabled,
            "stationary_redundancy_hold": self.stationary_redundancy_hold,
            "base_control_enabled": self.base_control_enabled,
            "ik_variables": 18 if self.base_control_enabled else 15,
            "preparation_lift_velocity_m_s": self.preparation_lift_velocity_m_s,
            "motion_profile_schema_version": self.schema_version,
            "motion_profile_id": self.profile_id,
            "motion_profile_sha256": self.sha256,
            "motion_profile_path": str(self.path),
            "lift_reference_max_error_m": self.lift_reference_max_error_m,
            "mit_planning_effort_fraction": self.mit_planning_effort_fraction,
            "mit_effort_preview": self.mit_effort_preview,
            "mit_gravity_correction": (None if self.mit_gravity_correction is None
                                       else self.mit_gravity_correction.source),
            "mit_dynamic_feedforward": (None if self.mit_dynamic_feedforward is None
                                        else self.mit_dynamic_feedforward.source),
            "motion_profile_simulation_status": self.simulation_status,
            "motion_profile_hardware_status": self.hardware_status,
            "motion_profile_target_planner": self.target_planner,
            "motion_profile_target_limit_check_runtime": self.target_limit_check_runtime,
            "motion_profile_collision_runtime": self.collision_runtime,
            "motion_profile_native_collision_kernel_abi": (
                self.native_collision_kernel_abi
            ),
            "motion_profile_collision_pair_certificate_sha256": (
                self.collision_pair_certificate_sha256
            ),
        }


def available_motion_profiles() -> tuple[str, ...]:
    """List supported profiles shipped with this checkout."""
    return tuple(sorted(path.stem for path in MOTION_PROFILE_DIR.glob("*.yaml")))


def motion_profile_path(reference: str | Path | None = None) -> Path:
    if reference is None:
        reference = DEFAULT_MOTION_PROFILE_ID
    candidate = Path(reference)
    if candidate.suffix or candidate.parent != Path("."):
        return candidate.resolve()
    active = (MOTION_PROFILE_DIR / f"{candidate}.yaml").resolve()
    return active


def load_motion_profile(reference: str | Path | None = None) -> MotionProfile:
    """Load and fully validate a named or explicit motion-profile YAML."""

    path = motion_profile_path(reference)
    if not path.is_file():
        names = ", ".join(available_motion_profiles()) or "none"
        raise FileNotFoundError(f"motion profile does not exist: {path}; available: {names}")
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, Mapping):
        raise ValueError("motion profile must be a YAML mapping")
    document = dict(raw)
    from .hardware_capabilities import load_hardware_capabilities

    capabilities = load_hardware_capabilities(document.get("hardware_capabilities"))
    reference_velocity_fraction = _positive(
        document.get("reference_velocity_fraction", 1.), label="reference velocity fraction")
    lift_reference_velocity_fraction = _positive(
        document.get("lift_reference_velocity_fraction", reference_velocity_fraction),
        label="lift reference velocity fraction")
    if lift_reference_velocity_fraction > 1:
        raise ValueError("lift reference velocity fraction cannot increase the speed limit")
    if reference_velocity_fraction > 1:
        raise ValueError("reference velocity fraction cannot increase the speed limit")
    schema_version = int(document.get("schema_version", 0))
    if schema_version in {18, 19, 20}:
        raise ValueError("retired experimental motion-profile schema; use baseline schema 17")
    if schema_version not in {*range(1, 18), 21, 22, 23}:
        raise ValueError("unsupported motion-profile schema")
    tracking_objective = document.get("tracking_objective", "legacy")
    stationary_arrival = document.get("stationary_arrival", "legacy")
    if tracking_objective not in {"legacy", "normalized_v1"} or stationary_arrival not in {"legacy", "continuous_v1"}:
        raise ValueError("invalid balanced tracking mode")
    if schema_version != 23 and (tracking_objective != "legacy" or stationary_arrival != "legacy"):
        raise ValueError("balanced tracking options require schema 23")
    workspace = document.get("base_workspace", {"mode": "bounded", "governor": "legacy"})
    if (not isinstance(workspace, Mapping) or set(workspace) != {"mode", "governor"}
            or workspace["mode"] not in {"bounded", "unbounded"}
            or workspace["governor"] not in {"legacy", "finite_base_v1"}):
        raise ValueError("invalid base workspace contract")
    profile_id = str(document.get("profile_id", ""))
    if not profile_id or path.parent == MOTION_PROFILE_DIR.resolve() and path.stem != profile_id:
        raise ValueError("motion profile ID must match its file name")
    if document.get("robot") != "maver_x4" or document.get("arm_model") != "archer_x7":
        raise ValueError("motion profile is not for the Maver X4 dual Archer X7")

    qualification = document.get("qualification")
    target = document.get("target_limits")
    arm = document.get("arm_limits")
    vendor = document.get("vendor_hard_limits")
    collision = document.get("self_collision")
    if not all(
        isinstance(value, Mapping) for value in (qualification, target, arm, vendor, collision)
    ):
        raise ValueError(
            "motion profile is missing qualification, target, arm, vendor, or collision fields"
        )
    assert isinstance(qualification, Mapping)
    assert isinstance(target, Mapping)
    assert isinstance(arm, Mapping)
    assert isinstance(vendor, Mapping)
    assert isinstance(collision, Mapping)

    if any(key in target for key in ("stable", "coupled", "c3")):
        raise ValueError("retired experimental target settings are not supported")

    if schema_version == 1:
        target_planner = "legacy_rate_clip_v1"
        target_linear_acceleration = None
        target_linear_jerk = None
        target_angular_acceleration = None
        target_angular_jerk = None
        target_translation_deadband = 0.0
        target_orientation_deadband = 0.0
        target_stationary_dwell = 0.0
        target_limit_check_runtime = "auto_compatible_v1"
    else:
        target_planner = str(target.get("planner", ""))
        expected_planner = (
            "c2_tracking_v4"
            if schema_version >= 17
            else "c2_tracking_v3"
            if schema_version >= 10
            else "c2_tracking_v2"
            if schema_version >= 3
            else "c2_stop_aware_v1"
        )
        if target_planner != expected_planner and not (schema_version >= 17 and target_planner in {"c2_tracking_v5", "c2_tracking_v6"}):
            raise ValueError(f"schema-v{schema_version} motion profiles require {expected_planner}")
        target_linear_acceleration = _positive(
            target.get("linear_acceleration_m_s2"), label="target linear acceleration"
        )
        target_linear_jerk = _positive(target.get("linear_jerk_m_s3"), label="target linear jerk")
        target_angular_acceleration = _positive(
            target.get("angular_acceleration_rad_s2"), label="target angular acceleration"
        )
        target_angular_jerk = _positive(
            target.get("angular_jerk_rad_s3"), label="target angular jerk"
        )
        target_limit_check_runtime = (
            str(target.get("limit_check_runtime", ""))
            if schema_version >= 12
            else "auto_compatible_v1"
        )
        if schema_version >= 12 and target_limit_check_runtime != "native_full_so3_v1":
            raise ValueError("schema-v12 requires native_full_so3_v1 target limit checks")
        if schema_version >= 8:
            target_translation_deadband = float(target.get("translation_deadband_m"))
            target_orientation_deadband = float(target.get("orientation_deadband_rad"))
            target_stationary_dwell = float(target.get("stationary_dwell_s"))
            if (
                not np.isfinite(target_translation_deadband)
                or target_translation_deadband < 0.0
                or not np.isfinite(target_orientation_deadband)
                or target_orientation_deadband < 0.0
                or not np.isfinite(target_stationary_dwell)
                or target_stationary_dwell < 0.0
            ):
                raise ValueError("target deadbands must be finite and non-negative")
        else:
            target_translation_deadband = 0.0
            target_orientation_deadband = 0.0
            target_stationary_dwell = 0.0

    simulation_status = str(qualification.get("simulation", ""))
    hardware_status = str(qualification.get("hardware", ""))
    if simulation_status not in {"candidate", "qualified"}:
        raise ValueError("simulation qualification must be candidate or qualified")
    if hardware_status not in {"uncommissioned", "commissioned"}:
        raise ValueError("hardware qualification must be uncommissioned or commissioned")

    if schema_version >= 15:
        raw_body_limits = document.get("body_limits")
        if (
            not isinstance(raw_body_limits, Mapping)
            or set(raw_body_limits) != SCHEMA_15_BODY_LIMIT_KEYS
        ):
            raise ValueError("schema-v15 requires the complete body_limits contract")
        body_limits = {
            key: _positive(raw_body_limits[key], label=key)
            for key in SCHEMA_15_BODY_LIMIT_KEYS
        }
    else:
        body_limits = {
            "base_linear_velocity_m_s": 2.0,
            "base_angular_velocity_rad_s": 2.0,
            "base_linear_acceleration_m_s2": 1.0,
            "base_angular_acceleration_rad_s2": 2.0,
            "base_linear_jerk_m_s3": 200.0,
            "base_angular_jerk_rad_s3": 400.0,
            "lift_velocity_m_s": 0.05,
            "lift_acceleration_m_s2": 0.2,
            "lift_jerk_m_s3": 40.0,
        }

    tuning = {}
    if schema_version >= 3:
        raw_tuning = document.get("controller_tuning")
        tuning_keys = (
            SCHEMA_13_CONTROLLER_TUNING_KEYS
            if schema_version >= 13
            else SCHEMA_8_CONTROLLER_TUNING_KEYS
            if schema_version >= 8
            else SCHEMA_7_CONTROLLER_TUNING_KEYS
            if schema_version >= 7
            else CONTROLLER_TUNING_KEYS
        )
        if (schema_version >= 17 and isinstance(raw_tuning, Mapping)
                and "target_tracking_bandwidth_s_inv" in raw_tuning):
            tuning_keys = (*tuning_keys, "target_tracking_bandwidth_s_inv")
        if schema_version >= 21 and isinstance(raw_tuning, Mapping) and "target_source_velocity_gain" in raw_tuning:
            tuning_keys = (*tuning_keys, "target_source_velocity_gain")
        if (schema_version >= 17 and isinstance(raw_tuning, Mapping)
                and "base_anchor_gain" in raw_tuning):
            tuning_keys = (*tuning_keys, "base_anchor_gain")
        if schema_version == 22:
            tuning_keys = (*tuning_keys, "orientation_gain", "effort_regularization_cost")
            if target_planner != "c2_tracking_v6":
                raise ValueError("schema 22 requires the v6 precision planner")
        elif target_planner == "c2_tracking_v6":
            raise ValueError("v6 precision planner requires schema 22")
        if schema_version == 23:
            for optional in ("effort_regularization_cost", "stationary_cost_blend_s", "orientation_gain",
                             "target_source_acceleration_gain", "target_source_acceleration_tau_s",
                             "measured_error_derivative_gain"):
                if isinstance(raw_tuning, Mapping) and optional in raw_tuning:
                    tuning_keys = (*tuning_keys, optional)
        if not isinstance(raw_tuning, Mapping) or set(raw_tuning) != set(tuning_keys):
            raise ValueError("schema-v3 requires the complete controller_tuning contract")
        tuning = {key: float(raw_tuning[key]) for key in tuning_keys}
        if not np.isfinite(tuning.get('measured_error_derivative_gain', 0.)) or not 0 <= tuning.get('measured_error_derivative_gain', 0.) <= 1:
            raise ValueError('measured_error_derivative_gain must be in [0, 1]')
        if "orientation_gain" in tuning and not 0 < tuning["orientation_gain"] <= 1:
            raise ValueError("orientation gain must be in (0, 1]")
        if "base_anchor_gain" in tuning and not 0 < tuning["base_anchor_gain"] <= 1:
            raise ValueError("base_anchor_gain must be in (0, 1]")
        if any(not np.isfinite(value) or value < 0.0 for value in tuning.values()):
            raise ValueError("controller tuning must be finite and non-negative")
        if tuning["position_cost"] <= 0.0 or tuning["orientation_cost"] <= 0.0:
            raise ValueError("TCP task costs must be positive")
        if tuning["qp_primal_tolerance"] <= 0.0:
            raise ValueError("QP primal tolerance must be positive")
        if (
            not 0.0 < tuning["frame_gain"] <= 1.0
            or not 0.0 <= tuning["target_feedforward_gain"] <= 1.0
        ):
            raise ValueError("controller frame/feedforward gains are outside their bounds")
        if not 0.0 <= tuning["target_velocity_feedback_gain"] <= 2.0:
            raise ValueError("target velocity feedback gain must be between zero and two")
        if not 0.0 <= tuning.get("target_source_velocity_gain", 1.) <= 1.0:
            raise ValueError("target source velocity gain must be between zero and one")
        if not 0.0 <= tuning.get("target_source_acceleration_gain", 0.) <= 1.0:
            raise ValueError("target source acceleration gain must be between zero and one")
        if not 0.01 <= tuning.get("target_source_acceleration_tau_s", 0.06) <= 0.5:
            raise ValueError("target source acceleration filter must be in [0.01, 0.5] s")
        if tuning.get("target_source_acceleration_gain", 0.) and target_planner != "c2_tracking_v4":
            raise ValueError("source acceleration feedforward requires the v4 planner")
        if "target_tracking_bandwidth_s_inv" in tuning and (
                target_planner not in {"c2_tracking_v4", "c2_tracking_v5", "c2_tracking_v6"}
                or not (2 if schema_version >= 21 else 8) <= tuning["target_tracking_bandwidth_s_inv"] <= 20):
            raise ValueError("tracking bandwidth requires v4/v5 and [8, 20] s^-1")
        if schema_version >= 8:
            for key in (
                "settling_blend_s",
                "settling_linear_velocity_m_s",
                "settling_angular_velocity_rad_s",
                "settling_position_error_m",
                "settling_orientation_error_rad",
            ):
                if tuning[key] <= 0.0:
                    raise ValueError(f"{key} must be positive")
            if tuning["settled_arm_damping_cost"] < 0.0:
                raise ValueError("settled arm damping cost must be non-negative")
        if schema_version >= 13 and (
            tuning["arm_posture_cost"] <= 0.0
            or tuning["arm_manipulability_cost"] <= 0.0
            or tuning["arm_manipulability_rate"] <= 0.0
            or tuning["arm_manipulability_activation_ramp_s"] <= 0.0
            or not 0.0 <= tuning["arm_posture_min_fraction"] <= 1.0
            or not 0.0 < tuning["arm_manipulability_floor_fraction"]
            < tuning["arm_manipulability_exit_fraction"] < 1.0
        ):
            raise ValueError("schema-v13 posture/manipulability tuning is invalid")
        if hardware_status != "uncommissioned":
            raise ValueError("responsive simulation profiles are not hardware commissioned")

    mit_servo = None
    gravity_payloads = ()
    maximum_feedback_age = 0.04
    gravity_urdf_sha256 = None
    measured_collision_method = "reference_velocity_v1"
    boundary_policy = "legacy"
    boundary_gain = 1.0
    command_delay = 0.02
    collision_sweep_step = None
    collision_sweep_intervals = 0
    collision_activation_policy = "static_worst_case_v2"
    allocator = DEFAULT_WHOLE_BODY_ALLOCATOR_CONFIG
    if schema_version >= 8:
        raw_allocator = document.get("whole_body_allocator")
        allocator_keys = (
            SCHEMA_14_ALLOCATOR_KEYS
            if schema_version >= 14
            else SCHEMA_13_ALLOCATOR_KEYS
            if schema_version >= 13
            else SCHEMA_8_ALLOCATOR_KEYS
        )
        if not isinstance(raw_allocator, Mapping) or set(raw_allocator) != allocator_keys:
            raise ValueError("schema-v8 requires the complete whole_body_allocator contract")
        allocator_values = dict(raw_allocator)
        allocator_values["base_activation_distances_m"] = tuple(
            float(value) for value in allocator_values["base_activation_distances_m"]
        )
        allocator_values["lift_activation_distances_m"] = tuple(
            float(value) for value in allocator_values["lift_activation_distances_m"]
        )
        allocator = WholeBodyAllocatorConfig(**allocator_values)
        expected_allocator = (
            "predictive_body_share_v4"
            if schema_version >= 14
            else "posture_assist_hysteresis_v3"
            if schema_version >= 13
            else "arm_first_hysteresis_v2"
        )
        if allocator.mode != expected_allocator and not (
                schema_version >= 17 and allocator.mode in {"adaptive_body_share_v5", "directional_body_share_v6"}):
            raise ValueError(
                f"schema-v{schema_version} requires the {expected_allocator} whole-body allocator"
            )
    if schema_version >= 4:
        from .mit import GravityPayload, MitServoConfig

        mit = document.get("mit")
        required_mit_fields = {
            "kp", "kd", "torque_limit_nm", "maximum_feedback_age_s",
            "gravity", "payloads", "asset_conversion", "urdf_sha256",
        }
        if schema_version >= 5:
            required_mit_fields.add("measured_collision_method")
        if schema_version >= 6:
            required_mit_fields.update(("boundary_policy", "boundary_gain_s_inv", "maximum_command_delay_s"))
        if schema_version >= 7:
            required_mit_fields.update(
                ("collision_sweep_max_joint_step_rad", "collision_sweep_max_intervals")
            )
        if schema_version >= 9:
            required_mit_fields.add("collision_activation_policy")
        if not isinstance(mit, Mapping) or set(mit) != required_mit_fields:
            raise ValueError(f"schema-v{schema_version} requires a complete MIT contract")
        if schema_version >= 5:
            measured_collision_method = str(mit["measured_collision_method"])
            if measured_collision_method != "reference_endpoint_v2":
                raise ValueError("schema-v5 requires the reference endpoint collision method")
        if schema_version >= 6:
            boundary_policy = str(mit["boundary_policy"])
            boundary_gain = _positive(mit["boundary_gain_s_inv"], label="boundary gain")
            command_delay = _positive(mit["maximum_command_delay_s"], label="command delay")
        if schema_version == 6:
            if (boundary_policy != "directional_velocity_damper_v1"
                    or boundary_gain != 1.0 or command_delay != 0.02):
                raise ValueError("schema-v6 requires the frozen lightweight boundary policy")
        elif schema_version >= 7:
            if boundary_policy != "jerk_stoppable_directional_v2":
                raise ValueError("schema-v7 requires the jerk-stoppable boundary policy")
            collision_sweep_step = _positive(
                mit["collision_sweep_max_joint_step_rad"], label="collision sweep step"
            )
            collision_sweep_intervals = int(mit["collision_sweep_max_intervals"])
            if collision_sweep_intervals < 1 or collision_sweep_intervals > 32:
                raise ValueError("collision sweep intervals must be in [1, 32]")
            if schema_version >= 9:
                collision_activation_policy = str(mit["collision_activation_policy"])
                if collision_activation_policy != "causal_reachable_v3":
                    raise ValueError("schema-v9 requires causal reachable collision activation")
        if mit["gravity"] != "nominal_pinocchio" or mit["asset_conversion"] != "unfused_urdf_v1":
            raise ValueError("MIT requires nominal gravity and unfused URDF assets")
        gravity_urdf_sha256 = str(mit["urdf_sha256"])
        if len(gravity_urdf_sha256) != 64 or any(c not in "0123456789abcdef" for c in gravity_urdf_sha256):
            raise ValueError("MIT requires a lowercase URDF SHA256")
        if not isinstance(mit["payloads"], list):
            raise ValueError("MIT payloads must be explicit declarations")
        mit_servo = MitServoConfig(
            kp=mit["kp"], kd=mit["kd"], torque_limit_nm=mit["torque_limit_nm"],
        )
        if target_planner in {"c2_tracking_v5", "c2_tracking_v6"}:
            from .hardware_capabilities import load_hardware_capabilities
            manufacturer = load_hardware_capabilities()
            for name in ("kp", "kd"):
                ratio = np.asarray(getattr(mit_servo, name)) / np.asarray(getattr(manufacturer, name))
                if np.any((ratio < .7 - 1e-12) | (ratio > 1.3 + 1e-12)):
                    raise ValueError("v5 MIT gains must remain within manufacturer +/-30%")
        gravity_payloads = tuple(GravityPayload(**payload) for payload in mit["payloads"])
        if any(payload.frame not in {"left_tcp", "right_tcp"} for payload in gravity_payloads):
            raise ValueError("MIT payloads must use a TCP frame")
        if len({payload.frame for payload in gravity_payloads}) != len(gravity_payloads):
            raise ValueError("duplicate MIT payload frame")
        maximum_feedback_age = _positive(mit["maximum_feedback_age_s"], label="feedback age")
        if tuning["simulation_arm_servo_kv"] != 0:
            raise ValueError("MIT profiles cannot enable simulation-only servo tuning")

    velocity = _positive_vector(arm.get("velocity_rad_s"), label="arm velocity", length=14)
    acceleration = _positive_vector(
        arm.get("acceleration_rad_s2"), label="arm acceleration", length=14
    )
    jerk = _positive_vector(arm.get("jerk_rad_s3"), label="arm jerk", length=14)
    urdf_velocity = _positive_vector(
        vendor.get("urdf_velocity_rad_s"), label="URDF velocity", length=14
    )
    sdk_velocity = _positive_vector(
        vendor.get("sdk_velocity_rad_s"), label="SDK velocity", length=14
    )
    velocity_authority = str(vendor.get("velocity_limit_authority", "urdf_and_sdk"))
    if velocity_authority not in {"urdf_and_sdk", "sdk", "manufacturer"}:
        raise ValueError("velocity limit authority must be urdf_and_sdk, sdk or manufacturer")
    if schema_version < 7 and velocity_authority != "urdf_and_sdk":
        raise ValueError("SDK velocity authority requires schema-v7")
    if schema_version >= 7 and velocity_authority not in {"sdk", "manufacturer"}:
        raise ValueError("schema-v7 requires explicit SDK velocity authority")
    lower = tuple(float(item) for item in vendor.get("urdf_position_lower_rad", ()))
    upper = tuple(float(item) for item in vendor.get("urdf_position_upper_rad", ()))
    if (
        len(lower) != 14
        or len(upper) != 14
        or not np.all(np.isfinite((lower, upper)))
        or np.any(np.asarray(lower) >= np.asarray(upper))
    ):
        raise ValueError("URDF position bounds must contain fourteen ordered finite intervals")
    sdk_lower = sdk_upper = None
    if schema_version >= 7:
        sdk_lower = tuple(float(item) for item in vendor.get("sdk_position_lower_rad", ()))
        sdk_upper = tuple(float(item) for item in vendor.get("sdk_position_upper_rad", ()))
        source = vendor.get("sdk_source")
        if (
            len(sdk_lower) != 14
            or len(sdk_upper) != 14
            or not np.all(np.isfinite((sdk_lower, sdk_upper)))
            or np.any(np.asarray(sdk_lower) >= np.asarray(sdk_upper))
            or not isinstance(source, Mapping)
            or set(source) != {"package_version", "git_commit", "arm_series", "config_sha256"}
            or source.get("package_version") != "1.4.10"
            or source.get("git_commit") != "2d672c7161464ce855ac23a7e2e993b688aa5db6"
            or int(source.get("arm_series", -1)) != 29
            or source.get("config_sha256")
            != "767a47dd56d1b64e9563018d9eb3764b03eec3f74d34c067d7e16efe56766e9c"
        ):
            raise ValueError("schema-v7 requires the reviewed Archer X7 SDK position/source contract")
    if velocity_authority == "manufacturer" and document.get("hardware_capabilities") is None:
        raise ValueError("manufacturer authority requires hash-bound hardware capabilities")
    hard_velocity = (
        np.asarray(capabilities.velocity_rad_s) if velocity_authority == "manufacturer" else
        np.asarray(sdk_velocity)
        if velocity_authority == "sdk"
        else np.minimum(urdf_velocity, sdk_velocity)
    )
    if np.any(np.asarray(velocity) > hard_velocity + 1e-12):
        raise ValueError("motion profile arm velocity exceeds a URDF or SDK hard limit")
    if document.get("hardware_capabilities") is not None:
        if np.any(np.asarray(velocity) > capabilities.velocity_rad_s):
            raise ValueError("motion profile exceeds manufacturer arm velocity")
        if (body_limits["lift_velocity_m_s"] > capabilities.lift_velocity_m_s
                or (1. if velocity_authority == "manufacturer" else np.sqrt(2)) * body_limits["base_linear_velocity_m_s"] > capabilities.base_linear_speed_m_s):
            raise ValueError("motion profile exceeds manufacturer lift/base linear speed")
        if mit_servo is not None and np.any(
            np.asarray(mit_servo.torque_limit_nm) > capabilities.peak_torque_nm
        ):
            raise ValueError("MIT instantaneous torque exceeds manufacturer peak rating")

    control_dt = _positive(document.get("control_dt_s"), label="control period")
    allowed_periods = (0.01, 0.02) if schema_version >= 7 else (0.02,)
    if not any(np.isclose(control_dt, period, atol=1e-12) for period in allowed_periods):
        raise ValueError("motion profile control period is unsupported by its schema")
    robot_model = None
    preparation_lift_velocity = .02
    collision_enabled = True
    if schema_version >= 21:
        if not isinstance(document.get("robot_model"), Mapping):
            raise ValueError("schema-v21 requires an explicit robot model")
        robot_model = load_robot_model(document["robot_model"])
        if gravity_urdf_sha256 != dict(robot_model.assets)[str(robot_model.urdf.relative_to(PROJECT_ROOT))]:
            raise ValueError("gravity and robot model URDF hashes differ")
        preparation = document.get("preparation_limits")
        if not isinstance(preparation, Mapping) or set(preparation) != {"lift_velocity_m_s"}:
            raise ValueError("schema-v21 requires preparation lift speed")
        preparation_lift_velocity = _positive(preparation["lift_velocity_m_s"], label="preparation lift speed")
        if preparation_lift_velocity > body_limits["lift_velocity_m_s"]:
            raise ValueError("preparation lift speed exceeds tracking limit")
        if set(collision) != {"enabled"} or collision["enabled"] is not False:
            raise ValueError("schema-v21 currently requires explicit collision disabled; no GP80 geometry qualified")
        collision_enabled = False
    profile_sha256 = _canonical_sha256(document)
    if not collision_enabled:
        geometry_path = envelope_path = pair_certificate_path = None
        pair_certificate_sha256 = None
        geometry_hash = ""
        collision_runtime = "disabled"
        native_collision_kernel_abi = 3  # SO(3) reference checks still use the native library.
    else:
        geometry_path = _project_file(collision.get("geometry_config"), label="collision geometry")
        envelope_path = _project_file(collision.get("envelope_config"), label="collision envelope")
        pair_certificate_path = None
        pair_certificate_sha256 = None
        if schema_version >= 10:
            pair_certificate_path = _project_file(
                collision.get("pair_certificate_config"), label="collision pair certificate"
            )
            pair_certificate_sha256 = str(collision.get("pair_certificate_sha256", ""))
            if (
                len(pair_certificate_sha256) != 64
                or hashlib.sha256(pair_certificate_path.read_bytes()).hexdigest()
                != pair_certificate_sha256
            ):
                raise ValueError("motion profile collision pair certificate has a stale SHA-256")
        collision_runtime = "auto_compatible_v1"
        native_collision_kernel_abi = None
        if schema_version >= 11:
            collision_runtime = str(collision.get("runtime", ""))
            native_collision_kernel_abi = int(collision.get("native_kernel_abi", 0))
            required_abi = 3 if schema_version >= 12 else 2
            if collision_runtime != "native_fused_v2" or native_collision_kernel_abi != required_abi:
                raise ValueError(
                    f"schema-v{schema_version} requires native_fused_v2 collision kernel ABI {required_abi}"
                )
        geometry_hash = str(collision.get("geometry_payload_sha256", ""))
        if len(geometry_hash) != 64:
            raise ValueError("motion profile collision geometry hash must be SHA-256")
        geometry_document = yaml.safe_load(geometry_path.read_text(encoding="utf-8"))
        from .self_collision import geometry_payload_sha256

        if (
            not isinstance(geometry_document, Mapping)
            or geometry_document.get("geometry_payload_sha256") != geometry_hash
            or geometry_payload_sha256(geometry_document) != geometry_hash
        ):
            raise ValueError("motion profile is stale for its self-collision geometry")
        profile_sha256 = _canonical_sha256(document)
        envelope_document = yaml.safe_load(envelope_path.read_text(encoding="utf-8"))
        if (
            not isinstance(envelope_document, Mapping)
            or envelope_document.get("profile_id") != profile_id
            or envelope_document.get("profile_sha256") != profile_sha256
            or envelope_document.get("geometry_payload_sha256") != geometry_hash
        ):
            raise ValueError("motion profile is stale for its self-collision envelope")
        for field in ("clearance_basis", "reviewed_proxy_clearance_m"):
            if collision.get(field) != envelope_document.get(field):
                raise ValueError("motion profile clearance policy differs from its collision envelope")

    lift_reference_max_error = None
    lift_reference = document.get("lift_reference")
    if lift_reference is not None:
        if (not isinstance(lift_reference, Mapping)
                or set(lift_reference) != {"mode", "maximum_tracking_error_m"}
                or lift_reference["mode"] != "persistent_v1"
                or schema_version < 17):
            raise ValueError("invalid persistent lift reference contract")
        lift_reference_max_error = _positive(
            lift_reference["maximum_tracking_error_m"], label="lift reference tracking bound")
    effort_fraction = float(document.get("mit_planning_effort_fraction", 1.0))
    if (not np.isfinite(effort_fraction) or not 0 < effort_fraction <= 1
            or (effort_fraction != 1 and (schema_version < 17 or mit_servo is None))):
        raise ValueError("invalid MIT planning effort fraction")
    effort_preview = document.get("mit_effort_preview", "none")
    if (effort_preview not in {"none", "jerk_prefix_v1"}
            or (effort_preview != "none" and (schema_version != 23 or mit_servo is None))):
        raise ValueError("invalid MIT effort preview")
    gravity_correction = document.get("mit_gravity_correction")
    if gravity_correction is not None:
        from .mit import GravityCorrection
        if (not isinstance(gravity_correction, Mapping)
                or set(gravity_correction) != {"scale", "offset_nm", "source"}
                or schema_version != 23 or mit_servo is None):
            raise ValueError("MIT gravity correction requires schema 23, MIT and scale/offset_nm/source")
        gravity_correction = GravityCorrection(**gravity_correction)
    dynamic_feedforward = document.get("mit_dynamic_feedforward")
    if dynamic_feedforward is not None:
        from .mit import DynamicFeedforward
        expected = {"coulomb_nm", "viscous_nm_s", "armature_kg_m2", "inertia_scale",
                    "coulomb_velocity_epsilon_rad_s", "headroom_fraction", "source"}
        if (not isinstance(dynamic_feedforward, Mapping) or set(dynamic_feedforward) != expected
                or schema_version != 23 or mit_servo is None):
            raise ValueError("MIT dynamic feedforward requires schema 23, MIT and the complete field set")
        dynamic_feedforward = DynamicFeedforward(**dynamic_feedforward)
    stationary_hold = document.get("stationary_redundancy_hold", False)
    if not isinstance(stationary_hold, bool) or (stationary_hold and schema_version < 21):
        raise ValueError("stationary redundancy hold requires schema 21")
    base_control_enabled = document.get("base_control_enabled", True)
    if not isinstance(base_control_enabled, bool) or (not base_control_enabled and schema_version != 23):
        raise ValueError("removing base IK coordinates requires schema 23")
    if schema_version == 23 and base_control_enabled:
        raise ValueError("schema 23 requires the 15-variable fixed-base controller")
    return MotionProfile(
        base_control_enabled=base_control_enabled,
        stationary_redundancy_hold=stationary_hold,
        robot_model=robot_model, self_collision_enabled=collision_enabled,
        preparation_lift_velocity_m_s=preparation_lift_velocity,
        mit_planning_effort_fraction=effort_fraction,
        mit_effort_preview=effort_preview,
        mit_gravity_correction=gravity_correction,
        mit_dynamic_feedforward=dynamic_feedforward,
        tracking_objective=tracking_objective, stationary_arrival=stationary_arrival,
        lift_reference_max_error_m=lift_reference_max_error,
        trajectory_kernel_version=1 if schema_version >= 16 else None,
        posture_policy=HumanPosturePolicy.parse(document.get("posture_policy")),
        reference_velocity_fraction=reference_velocity_fraction,
        lift_reference_velocity_fraction=lift_reference_velocity_fraction,
        path=path,
        schema_version=schema_version,
        profile_id=profile_id,
        sha256=profile_sha256,
        robot="maver_x4",
        arm_model="archer_x7",
        control_dt_s=control_dt,
        simulation_status=simulation_status,
        hardware_status=hardware_status,
        target_linear_velocity_m_s=_positive(
            target.get("linear_velocity_m_s"), label="target linear velocity"
        ),
        target_angular_velocity_rad_s=_positive(
            target.get("angular_velocity_rad_s"), label="target angular velocity"
        ),
        target_linear_acceleration_m_s2=target_linear_acceleration,
        target_linear_jerk_m_s3=target_linear_jerk,
        target_angular_acceleration_rad_s2=target_angular_acceleration,
        target_angular_jerk_rad_s3=target_angular_jerk,
        target_planner=target_planner,
        target_translation_deadband_m=target_translation_deadband,
        target_orientation_deadband_rad=target_orientation_deadband,
        target_stationary_dwell_s=target_stationary_dwell,
        target_limit_check_runtime=target_limit_check_runtime,
        arm_velocity_rad_s=velocity,
        arm_acceleration_rad_s2=acceleration,
        arm_jerk_rad_s3=jerk,
        arm_position_margin_rad=_positive(
            arm.get("position_margin_rad"), label="arm position margin"
        ),
        base_linear_velocity_m_s=body_limits["base_linear_velocity_m_s"],
        base_angular_velocity_rad_s=body_limits["base_angular_velocity_rad_s"],
        base_linear_acceleration_m_s2=body_limits["base_linear_acceleration_m_s2"],
        base_angular_acceleration_rad_s2=body_limits["base_angular_acceleration_rad_s2"],
        base_linear_jerk_m_s3=body_limits["base_linear_jerk_m_s3"],
        base_angular_jerk_rad_s3=body_limits["base_angular_jerk_rad_s3"],
        lift_velocity_m_s=body_limits["lift_velocity_m_s"],
        lift_acceleration_m_s2=body_limits["lift_acceleration_m_s2"],
        lift_jerk_m_s3=body_limits["lift_jerk_m_s3"],
        urdf_velocity_rad_s=urdf_velocity,
        sdk_velocity_rad_s=sdk_velocity,
        velocity_limit_authority=velocity_authority,
        urdf_position_lower_rad=lower,
        urdf_position_upper_rad=upper,
        sdk_position_lower_rad=sdk_lower,
        sdk_position_upper_rad=sdk_upper,
        collision_geometry_config=geometry_path,
        collision_geometry_sha256=geometry_hash,
        collision_envelope_config=envelope_path,
        collision_pair_certificate_config=pair_certificate_path,
        collision_pair_certificate_sha256=pair_certificate_sha256,
        collision_runtime=collision_runtime,
        native_collision_kernel_abi=native_collision_kernel_abi,
        controller_tuning=tuning,
        allocator=allocator,
        raw=document,
        base_workspace_mode=workspace["mode"],
        base_boundary_governor=workspace["governor"],
        mit_servo=mit_servo,
        gravity_payloads=gravity_payloads,
        maximum_feedback_age_s=maximum_feedback_age,
        gravity_urdf_sha256=gravity_urdf_sha256,
        measured_collision_method=measured_collision_method,
        boundary_policy=boundary_policy,
        boundary_gain_s_inv=boundary_gain,
        maximum_command_delay_s=command_delay,
        collision_sweep_max_joint_step_rad=collision_sweep_step,
        collision_sweep_max_intervals=collision_sweep_intervals,
        collision_activation_policy=collision_activation_policy,
    )


def controller_config_for_profile(
    reference: str | Path | MotionProfile | None = None,
    **overrides: object,
):
    """Construct a Pink config without making the profile layer import Pink eagerly."""

    from .ik_core import PinkControllerConfig
    from .robot import DEFAULT_PLANAR_BASE_CONFIG

    profile = reference if isinstance(reference, MotionProfile) else load_motion_profile(reference)
    values: dict[str, object] = {
        "control_dt": profile.control_dt_s,
        "motion_envelope": profile.motion_envelope,
        "arm_velocity_limits": profile.arm_velocity_rad_s,
        "arm_acceleration_limits": profile.arm_acceleration_rad_s2,
        "arm_jerk_limits": profile.arm_jerk_rad_s3,
        "arm_position_margin": profile.arm_position_margin_rad,
        "base": replace(
            DEFAULT_PLANAR_BASE_CONFIG,
            workspace_mode=profile.base_workspace_mode,
            boundary_governor=profile.base_boundary_governor,
            linear_velocity_limit=profile.base_linear_velocity_m_s,
            angular_velocity_limit=profile.base_angular_velocity_rad_s,
            linear_acceleration_limit=profile.base_linear_acceleration_m_s2,
            angular_acceleration_limit=profile.base_angular_acceleration_rad_s2,
            linear_jerk_limit=profile.base_linear_jerk_m_s3,
            angular_jerk_limit=profile.base_angular_jerk_rad_s3,
        ),
        "robot_model": profile.robot_model,
        "stationary_redundancy_hold": profile.stationary_redundancy_hold,
        "self_collision_enabled": profile.self_collision_enabled,
        "preparation_lift_velocity_m_s": profile.preparation_lift_velocity_m_s,
        "lift_velocity_limit": profile.lift_velocity_m_s,
        "lift_acceleration_limit": profile.lift_acceleration_m_s2,
        "lift_jerk_limit": profile.lift_jerk_m_s3,
        "target_linear_velocity_limit": profile.target_linear_velocity_m_s,
        "target_angular_velocity_limit": profile.target_angular_velocity_rad_s,
        "target_planner": profile.target_planner,
        "target_translation_deadband": profile.target_translation_deadband_m,
        "target_orientation_deadband": profile.target_orientation_deadband_rad,
        "target_stationary_dwell": profile.target_stationary_dwell_s,
        "target_limit_check_runtime": profile.target_limit_check_runtime,
        "simulation_arm_servo_kv": (3.0 if profile.target_planner == "c2_stop_aware_v1" else 0.0),
        "motion_limit_source": "profile",
        "motion_profile_id": profile.profile_id,
        "motion_profile_sha256": profile.sha256,
        "motion_profile_path": profile.path,
        "trajectory_kernel_version": profile.trajectory_kernel_version,
        "posture_policy": profile.posture_policy,
        "lift_reference_max_error_m": profile.lift_reference_max_error_m,
        "mit_planning_effort_fraction": profile.mit_planning_effort_fraction,
        "mit_effort_preview": profile.mit_effort_preview,
        "mit_gravity_correction": profile.mit_gravity_correction,
        "mit_dynamic_feedforward": profile.mit_dynamic_feedforward,
        "base_control_enabled": profile.base_control_enabled,
        "tracking_objective": profile.tracking_objective,
        "stationary_arrival": profile.stationary_arrival,
        "reference_velocity_fraction": profile.reference_velocity_fraction,
        "lift_reference_velocity_fraction": profile.lift_reference_velocity_fraction,
        "self_collision_config": profile.collision_geometry_config,
        "self_collision_envelope_config": profile.collision_envelope_config,
        "self_collision_pair_certificate_config": profile.collision_pair_certificate_config,
        "self_collision_pair_certificate_sha256": profile.collision_pair_certificate_sha256,
        "self_collision_runtime": profile.collision_runtime,
        "native_collision_kernel_abi": profile.native_collision_kernel_abi,
        "arm_position_lower_limits": (
            None
            if profile.sdk_position_lower_rad is None
            else tuple(
                np.maximum(profile.urdf_position_lower_rad, profile.sdk_position_lower_rad)
            )
        ),
        "arm_position_upper_limits": (
            None
            if profile.sdk_position_upper_rad is None
            else tuple(
                np.minimum(profile.urdf_position_upper_rad, profile.sdk_position_upper_rad)
            )
        ),
        "allocator": profile.allocator,
    }
    if profile.target_linear_acceleration_m_s2 is not None:
        values.update(
            {
                "target_linear_acceleration_limit": profile.target_linear_acceleration_m_s2,
                "target_linear_jerk_limit": profile.target_linear_jerk_m_s3,
                "target_angular_acceleration_limit": profile.target_angular_acceleration_rad_s2,
                "target_angular_jerk_limit": profile.target_angular_jerk_rad_s3,
            }
        )
    values.update(profile.controller_tuning)
    if profile.mit_servo is not None:
        values.update(
            mit_servo=profile.mit_servo,
            gravity_payloads=profile.gravity_payloads,
            maximum_feedback_age_s=profile.maximum_feedback_age_s,
            gravity_urdf_sha256=profile.gravity_urdf_sha256,
            measured_collision_method=profile.measured_collision_method,
            boundary_policy=profile.boundary_policy,
            boundary_gain_s_inv=profile.boundary_gain_s_inv,
            maximum_command_delay_s=profile.maximum_command_delay_s,
            collision_sweep_max_joint_step_rad=profile.collision_sweep_max_joint_step_rad,
            collision_sweep_max_intervals=profile.collision_sweep_max_intervals,
            collision_activation_policy=profile.collision_activation_policy,
        )
    values.update(overrides)
    return PinkControllerConfig(**values)
