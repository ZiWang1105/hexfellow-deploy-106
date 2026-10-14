"""Pure smoothing primitives and state for continuous whole-body allocation."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pinocchio as pin

ADAPTIVE_SHARE_MODES = frozenset({"adaptive_body_share_v5", "directional_body_share_v6"})
PREDICTIVE_SHARE_MODES = ADAPTIVE_SHARE_MODES | {"predictive_body_share_v4"}
POSTURE_SHARE_MODES = PREDICTIVE_SHARE_MODES | {"posture_assist_hysteresis_v3"}


def sigmoid(value: float) -> float:
    return 1.0 / (1.0 + float(np.exp(-np.clip(value, -60.0, 60.0))))


def zero_based_activation(distance: float, activation_distance: float, steepness: float) -> float:
    """Smooth sigmoid activation normalized to be exactly zero at no demand."""
    floor = sigmoid(-steepness)
    value = sigmoid(steepness * (distance / activation_distance - 1.0))
    return float(np.clip((value - floor) / (1.0 - floor), 0.0, 1.0))


def smoothstep(value: float) -> float:
    value = float(np.clip(value, 0.0, 1.0))
    return value * value * (3.0 - 2.0 * value)


def arm_limit_pressures(
    positions: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    config: WholeBodyAllocatorConfig,
) -> np.ndarray:
    """Return smooth per-joint pressure from normalized planning-limit distance."""
    normalized_margin = np.clip(np.minimum(positions - lower, upper - positions) / (upper - lower), 0.0, 0.5)
    raw = 1.0 / (
        1.0
        + np.exp(
            np.clip(
                (normalized_margin - config.pressure_midpoint_fraction) / config.pressure_width_fraction,
                -60.0,
                60.0,
            )
        )
    )
    center = config.pressure_midpoint_fraction
    width = config.pressure_width_fraction
    at_limit = sigmoid(center / width)
    at_center = sigmoid((center - 0.5) / width)
    return np.clip((raw - at_center) / (at_limit - at_center), 0.0, 1.0)


def filtered_activation(
    current: float,
    target: float,
    dt: float,
    config: WholeBodyAllocatorConfig,
) -> float:
    rising = sigmoid((target - current) / config.filter_direction_width)
    time_constant = config.activation_release_s + rising * (config.activation_attack_s - config.activation_release_s)
    blend = 1.0 - float(np.exp(-dt / time_constant))
    return float(np.clip(current + blend * (target - current), 0.0, 1.0))


@dataclass(frozen=True)
class WholeBodyAllocatorConfig:
    """Soft whole-body allocation settings; none of these are hard limits."""

    arm_limit_cost: float = 2.4
    lift_active_cost: float = 0.5
    base_active_cost: float = 3.0
    pressure_midpoint_fraction: float = 0.05
    pressure_width_fraction: float = 0.025
    base_activation_distances_m: tuple[float, float] = (0.08, 0.01)
    lift_activation_distances_m: tuple[float, float] = (0.04, 0.005)
    activation_steepness: float = 6.0
    dominance_epsilon_m: float = 0.001
    activation_attack_s: float = 0.04
    activation_release_s: float = 0.5
    filter_direction_width: float = 0.02
    lift_reference_error_gain: float = 18.8
    mode: str = "continuous_v1"
    base_enter_distance_m: float = 0.10
    base_exit_distance_m: float = 0.04
    lift_enter_distance_m: float = 0.06
    lift_exit_distance_m: float = 0.02
    pressure_enter: float = 0.75
    pressure_exit: float = 0.55
    engage_dwell_s: float = 0.12
    release_dwell_s: float = 0.20
    base_inactive_cost: float = 10.0
    lift_inactive_cost: float = 5.0
    base_anchor_follow_linear_m_s: float = 0.25
    base_anchor_follow_angular_rad_s: float = 0.5
    lift_anchor_follow_m_s: float = 0.04
    rebase_position_error_m: float = 0.005
    posture_assist_enter_rad: float = 0.35
    posture_assist_exit_rad: float = 0.20
    common_yaw_enter_rad: float = 0.20
    common_yaw_exit_rad: float = 0.08
    soft_assist_activation: float = 0.55
    lift_common_translation_share: float = 0.0
    base_common_translation_share: float = 0.0
    base_common_yaw_share: float = 0.0

    def __post_init__(self) -> None:
        if len(self.base_activation_distances_m) != 2 or len(self.lift_activation_distances_m) != 2:
            raise ValueError("whole-body activation distances must contain comfortable and near-limit values")
        positive = (
            self.arm_limit_cost,
            self.lift_active_cost,
            self.base_active_cost,
            self.pressure_width_fraction,
            self.activation_steepness,
            self.dominance_epsilon_m,
            self.activation_attack_s,
            self.activation_release_s,
            self.filter_direction_width,
            self.lift_reference_error_gain,
            self.base_enter_distance_m,
            self.base_exit_distance_m,
            self.lift_enter_distance_m,
            self.lift_exit_distance_m,
            self.engage_dwell_s,
            self.release_dwell_s,
            self.base_inactive_cost,
            self.lift_inactive_cost,
            self.base_anchor_follow_linear_m_s,
            self.base_anchor_follow_angular_rad_s,
            self.lift_anchor_follow_m_s,
            self.rebase_position_error_m,
            self.posture_assist_enter_rad,
            self.posture_assist_exit_rad,
            self.common_yaw_enter_rad,
            self.common_yaw_exit_rad,
            *self.base_activation_distances_m,
            *self.lift_activation_distances_m,
        )
        if any(not np.isfinite(value) or value <= 0.0 for value in positive):
            raise ValueError("whole-body allocator costs, distances, and smoothing values must be positive")
        if not 0.0 < self.pressure_midpoint_fraction < 0.5:
            raise ValueError("pressure midpoint must lie inside the normalized joint range")
        if self.pressure_width_fraction >= 0.5:
            raise ValueError("pressure width must be smaller than half the normalized joint range")
        if self.activation_release_s < self.activation_attack_s:
            raise ValueError("allocator release time must not be shorter than attack time")
        if self.mode not in {
            "continuous_v1",
            "arm_first_hysteresis_v2",
            "posture_assist_hysteresis_v3",
            "predictive_body_share_v4", "adaptive_body_share_v5", "directional_body_share_v6",
        }:
            raise ValueError("unknown whole-body allocator mode")
        if not 0.0 <= self.pressure_exit < self.pressure_enter <= 1.0:
            raise ValueError("allocator pressure thresholds must be ordered inside [0, 1]")
        if not self.base_exit_distance_m < self.base_enter_distance_m:
            raise ValueError("base exit distance must be below its enter distance")
        if not self.lift_exit_distance_m < self.lift_enter_distance_m:
            raise ValueError("lift exit distance must be below its enter distance")
        if self.base_active_cost > self.base_inactive_cost:
            raise ValueError("active base cost must not exceed inactive cost")
        if self.lift_active_cost > self.lift_inactive_cost:
            raise ValueError("active lift cost must not exceed inactive cost")
        if not self.posture_assist_exit_rad < self.posture_assist_enter_rad:
            raise ValueError("posture-assist exit threshold must be below its enter threshold")
        if not self.common_yaw_exit_rad < self.common_yaw_enter_rad:
            raise ValueError("common-yaw exit threshold must be below its enter threshold")
        if not 0.0 < self.soft_assist_activation <= 1.0:
            raise ValueError("soft-assist activation must lie in (0, 1]")
        shares = (
            self.lift_common_translation_share,
            self.base_common_translation_share,
            self.base_common_yaw_share,
        )
        if any(not np.isfinite(value) or not 0.0 <= value <= 1.0 for value in shares):
            raise ValueError("whole-body target shares must be finite and lie in [0, 1]")
        if not self.base_activation_distances_m[0] > self.base_activation_distances_m[1]:
            raise ValueError("comfortable base activation distance must exceed near-limit distance")
        if not self.lift_activation_distances_m[0] > self.lift_activation_distances_m[1]:
            raise ValueError("comfortable lift activation distance must exceed near-limit distance")


DEFAULT_WHOLE_BODY_ALLOCATOR_CONFIG = WholeBodyAllocatorConfig()


@dataclass(frozen=True)
class WholeBodyAllocationDiagnostics:
    common_translation_error: tuple[float, float, float]
    differential_translation_error: tuple[float, float, float]
    base_activation: float
    lift_activation: float
    arm_pressure: float
    max_arm_joint_pressure: float
    arm_joint_pressures: dict[str, float]
    base_weights: tuple[float, float, float] | None
    lift_weight: float
    arm_weights: dict[str, float]
    mode: str = "continuous_v1"
    base_engaged: bool = False
    lift_engaged: bool = False
    base_activation_reason: str = "none"
    lift_activation_reason: str = "none"
    base_enter_elapsed_s: float = 0.0
    lift_enter_elapsed_s: float = 0.0
    base_release_elapsed_s: float = 0.0
    lift_release_elapsed_s: float = 0.0
    base_anchor_position: tuple[float, float, float] | None = None
    lift_anchor_position: float = 0.0
    allocation_reference_translation: dict[str, tuple[float, float, float]] = field(
        default_factory=dict
    )
    common_yaw_error_rad: float = 0.0
    posture_error_rad: dict[str, float] = field(default_factory=dict)
    manipulability_activation: float = 0.0
    ergonomic_strain: float = 0.0
    common_target_delta: tuple[float, float, float] = (0.0, 0.0, 0.0)
    requested_base_share_delta: tuple[float, float, float] = (0.0, 0.0, 0.0)
    base_share_delta: tuple[float, float, float] = (0.0, 0.0, 0.0)
    requested_lift_share_delta_m: float = 0.0
    lift_share_delta_m: float = 0.0
    adaptive_shares: tuple[float, ...] | None = None
    body_goal: tuple[float, ...] | None = None


@dataclass(frozen=True)
class WholeBodyAllocatorState:
    """Soft allocator state committed only after a successful QP solve."""

    base_activation: float
    lift_activation: float
    base_anchor_target: pin.SE3 | None
    lift_anchor_position: float
    diagnostics: WholeBodyAllocationDiagnostics
    base_engaged: bool = False
    lift_engaged: bool = False
    base_enter_elapsed_s: float = 0.0
    lift_enter_elapsed_s: float = 0.0
    base_release_elapsed_s: float = 0.0
    lift_release_elapsed_s: float = 0.0
    base_activation_reason: str = "none"
    lift_activation_reason: str = "none"
    settling_blend: float = 0.0
    stationary_cost_blend: float = 0.0
    allocation_reference_targets: dict[str, pin.SE3] = field(default_factory=dict)
    previous_allocation_targets: dict[str, pin.SE3] = field(default_factory=dict)
    adaptive_share: object = None
