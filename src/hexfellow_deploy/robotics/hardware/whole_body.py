"""Hardware contracts and commissioning gates; no SDK/network/simulator imports."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..control.hardware_capabilities import load_hardware_capabilities
from ..control.mit_command import MitCommand
from ..control.robot import BASE_JOINTS, LEFT_ARM_JOINTS, LIFT_JOINT, RIGHT_ARM_JOINTS
from .x7 import HardwareSafetyError

STAGES = {
    "right-arm": ("right",),
    "left-arm": ("left",),
    "dual-arm": ("left", "right"),
    "arms-lift": ("left", "right", "theta"),
    "whole-body": ("left", "right", "theta", "base"),
}
JOINTS = (*BASE_JOINTS, LIFT_JOINT, *LEFT_ARM_JOINTS, *RIGHT_ARM_JOINTS)
SLICES = {"base": slice(0, 3), "theta": slice(3, 4), "left": slice(4, 11), "right": slice(11, 18)}
STATE_MAX_AGE = 0.040
STATE_MAX_SKEW = 0.020
COMMAND_TTL = 0.020
COMMAND_WATCHDOG = 0.040  # one 20 ms reference period plus at most 20 ms dispatch delay
QUEST_EXPERIMENT = "left-hold-5cm-quest-whole-body-v1"
DIRECT_QUEST_EXPERIMENT = "direct-quest-mit-v10-v1"


def vector(value, size, label):
    array = np.array(value, dtype=float, copy=True)
    if array.shape != (size,) or not np.all(np.isfinite(array)):
        raise HardwareSafetyError(f"{label}: expected {size} finite values")
    return array


def locked_names(stage):
    active = STAGES[stage]
    return tuple(name for group, part in SLICES.items() if group not in active for name in JOINTS[part])


@dataclass(frozen=True)
class RobotMapping:
    data: dict
    sha256: str
    motion_envelope: str = "none"
    feedback_policy: str = "strict"
    allow_unverified_hardware_limits: bool = False
    left_quest_demo: bool = False
    left_replay_excursion_rad: float | None = None
    commissioning_stage: str | None = None
    verification: dict | None = None
    initialization_abort_after: str | None = None
    latency_policy: str = "strict"
    demo_authorized: bool = False
    source_path: Path | None = None
    research_release: Path | None = None

    @property
    def effective_review(self):
        review = dict(self.data.get("review", {}))
        if self.verification is not None:
            self.validate_verification()
            review.update(self.verification["checks"])
        return review

    def validate_verification(self):
        if self.verification is not None and (
            not self.runtime_session or self.verification.get("mapping_sha256") != self.sha256
            or self.verification.get("profile_sha256") != self.data["commissioning"]["profile_sha256"]
            or self.verification.get("scope") not in {"control", "poses", "replay", "quest"}
        ):
            raise HardwareSafetyError("verification binding/scope mismatch")

    def validate_experiment_options(self):
        from ..control.feedback_policy import FEEDBACK_POLICIES, THETA_BOUNDED, THETA_POLICY, THETA_RECORD_ONLY

        self.validate_verification()
        if self.latency_policy not in {"strict", "hold-resume"}:
            raise HardwareSafetyError("invalid latency policy")
        if (self.latency_policy == "hold-resume" or self.demo_authorized) and not self.runtime_session:
            raise HardwareSafetyError("latency recovery/demo requires runtime session mapping")
        if (self.verification is not None and self.verification["scope"] == "poses"
                and self.motion_envelope != "first-trial"):
            raise HardwareSafetyError("verification sessions require first-trial limits")
        if self.initialization_abort_after is not None and self.commissioning_stage is None:
            raise HardwareSafetyError("initialization fault injection is commissioning-only")

        if self.commissioning_stage is not None and (
            not self.direct_quest or self.commissioning_stage not in STAGES
            or self.motion_envelope != "first-trial"
        ):
            raise HardwareSafetyError("commissioning requires a bounded direct mapping and explicit stage")

        if self.left_quest_demo and (not self.direct_quest or self.motion_envelope != "first-trial"):
            raise HardwareSafetyError("left Quest demo requires direct mapping and first-trial limits")
        if self.left_replay_excursion_rad is not None and (
            not self.left_quest_demo or not np.isfinite(self.left_replay_excursion_rad)
            or not 0 < self.left_replay_excursion_rad <= np.deg2rad(50)
        ):
            raise HardwareSafetyError("replay excursion requires bounded left demo and (0, 50] degrees")

        if self.motion_envelope not in {"none", "first-trial"} or self.feedback_policy not in FEEDBACK_POLICIES:
            raise HardwareSafetyError("invalid experimental options")
        if (self.motion_envelope != "none" or self.feedback_policy != "strict"
                or self.allow_unverified_hardware_limits) and not self.direct_quest:
            raise HardwareSafetyError("experimental options require direct Quest mapping")
        if self.feedback_policy in {THETA_RECORD_ONLY, THETA_BOUNDED} and not self.runtime_session:
            raise HardwareSafetyError("Theta runtime policy requires the runtime session mapping")
        if ((self.feedback_policy == THETA_POLICY or self.allow_unverified_hardware_limits)
                and self.motion_envelope != "first-trial"):
            raise HardwareSafetyError("Theta tolerance/unverified limits require first-trial envelope")

    @classmethod
    def load(cls, path):
        raw = Path(path).read_bytes()
        data = json.loads(raw)
        if data.get("schema_version") != 1:
            raise ValueError("robot mapping schema_version must be 1")
        for side in ("left", "right"):
            entry = data["arms"][side]
            order = entry["sdk_indices"]
            if len(order) != 7 or any(type(i) is not int for i in order) or sorted(order) != list(range(7)):
                raise ValueError(f"{side}: sdk_indices must be a permutation of 0..6")
            signs = vector(entry["signs"], 7, f"{side} signs")
            if not np.all(np.abs(signs) == 1):
                raise ValueError("joint signs must be +/-1")
            vector(entry["offsets_rad"], 7, f"{side} offsets")
        vector([data["lift_zero_offset_m"], data["base_sdk_to_model_yaw_rad"]], 2, "body mapping")
        if data.get("lift_position_mode", "sdk") not in {"sdk", "range_max_minus_sdk"}:
            raise ValueError("unsupported lift_position_mode")
        from ..control.robot_model import load_robot_model
        model = load_robot_model(data.get("robot_model"))
        vector(data["fixed_gripper_positions"], len(model.auxiliary_joints), "fixed gripper positions")
        load_hardware_capabilities(data.get("hardware_capabilities"))
        from ..control.feedback_policy import FEEDBACK_POLICIES
        policy = data.get("feedback_policy", "strict")
        latency = data.get("latency_policy", "strict")
        if policy not in FEEDBACK_POLICIES or latency not in {"strict", "hold-resume"}:
            raise ValueError("invalid configured feedback/latency policy")
        result = cls(data, hashlib.sha256(raw).hexdigest(), feedback_policy=policy, latency_policy=latency,
                     source_path=Path(path).resolve())
        if result.quest_experiment and not result.direct_quest:
            result.arm_excursion_limit("left-arm")  # Validate, without granting hardware approval.
        return result

    @property
    def direct_quest(self):
        return self.runtime_session or self.data.get("commissioning", {}).get("sequence") == DIRECT_QUEST_EXPERIMENT

    @property
    def runtime_session(self):
        return self.data.get("commissioning", {}).get("sequence") == "hardware-session-v1"

    @property
    def quest_experiment(self):
        commissioning = self.data.get("commissioning", {})
        sequence = commissioning.get("sequence")
        if sequence is None:
            return False
        if self.direct_quest:
            from .research_profiles import HIGHSPEED_PROFILES
            expected_profiles = ({"x7-gp80-highspeed-fixed-base-v1", "x7-gp80-highspeed-fixed-base-v2", "x7-gp80-highspeed-fixed-base-v3", "x7-gp80-highspeed-fixed-base-v1", "x7-gp80-highspeed-fixed-base-v2", "x7-mit-manufacturer-v1", "x7-mit-highspeed-v10-base-v1",
                                  "x7-mit-highspeed-v10-base-drag-v1", "x7-mit-highspeed-v10-base-lift-v2",
                                  "x7-mit-highspeed-v10-base-lift-v3", "x7-mit-highspeed-v10-base-lift-v4"}
                                 if self.runtime_session else {"x7-mit-highspeed-v10"})
            if self.runtime_session:
                expected_profiles |= HIGHSPEED_PROFILES
            if (commissioning.get("allowed_live_stages") != ["whole-body"]
                    or commissioning.get("motion_profile") not in expected_profiles
                    or not isinstance(commissioning.get("profile_sha256"), str)
                    or len(commissioning["profile_sha256"]) != 64):
                raise HardwareSafetyError("invalid direct Quest experiment")
            return True
        if (sequence != QUEST_EXPERIMENT or commissioning.get("first_arm") != "left"
                or commissioning.get("allowed_live_stages") != ["left-arm", "whole-body"]
                or commissioning.get("manual_distance_m") != 0.05
                or commissioning.get("motion_profile") != "x7-mit-highspeed-v10"
                or not isinstance(commissioning.get("profile_sha256"), str)
                or len(commissioning["profile_sha256"]) != 64):
            raise HardwareSafetyError("invalid experimental commissioning sequence")
        return True

    def arm_excursion_limit(self, stage):
        if self.commissioning_stage is not None:
            self.validate_experiment_options()
            if stage != self.commissioning_stage:
                raise HardwareSafetyError("commissioning stage mismatch")
            return float(np.deg2rad(12))  # 10-degree onsite targets plus tracking margin.
        if self.left_quest_demo:
            if stage != "left-arm":
                raise HardwareSafetyError("left Quest demo cannot command other stages")
            self.validate_experiment_options()
            return float(np.deg2rad(5) if self.left_replay_excursion_rad is None else self.left_replay_excursion_rad)
        if self.direct_quest:
            if stage != "whole-body":
                raise HardwareSafetyError("direct Quest is whole-body only")
            return None
        commissioning = self.data.get("commissioning")
        if not commissioning:
            return None
        limit = float(commissioning["max_arm_excursion_rad"])
        if not np.isfinite(limit) or not 0 < limit <= np.deg2rad(5):
            raise HardwareSafetyError("invalid commissioning excursion limit")
        if self.quest_experiment and stage == "whole-body":
            return None  # Full profile/hardware limits still apply in the publisher.
        return limit

    def arm_state(self, side, pos, vel):
        entry = self.data["arms"][side]
        order, signs = entry["sdk_indices"], np.asarray(entry["signs"])
        return (
            vector(pos, 7, side)[order] * signs + entry["offsets_rad"],
            vector(vel, 7, side)[order] * signs,
        )

    def arm_command(self, side, command):
        names = LEFT_ARM_JOINTS if side == "left" else RIGHT_ARM_JOINTS
        selected = command.select(names)
        entry = self.data["arms"][side]
        order, signs = entry["sdk_indices"], np.asarray(entry["signs"])
        fields = {}
        for key in ("position", "velocity", "torque_ff", "kp", "kd"):
            values = np.asarray(getattr(selected, key)).copy()
            if key == "position":
                values -= entry["offsets_rad"]
            if key in ("position", "velocity", "torque_ff"):
                values *= signs
            output = np.empty(7)
            output[order] = values
            fields[key] = output
        return fields

    def arm_effort(self, side, torque):
        """SDK joint Nm to model joint Nm; sign/order only, no gear scaling."""
        entry = self.data["arms"][side]
        return vector(torque, 7, f"{side} torque Nm")[entry["sdk_indices"]] * entry["signs"]

    def rotation(self):
        angle = self.data["base_sdk_to_model_yaw_rad"]
        c, s = np.cos(angle), np.sin(angle)
        return np.array([[c, -s], [s, c]])

    def lift_position(self, sdk_position, reported_range):
        low, high = vector(reported_range, 2, "lift reported range")
        if high <= low:
            raise HardwareSafetyError("invalid lift reported range")
        position = float(vector([sdk_position], 1, "lift SDK position")[0])
        if self.data.get("lift_position_mode", "sdk") == "range_max_minus_sdk":
            position = high - position
        return position + self.data["lift_zero_offset_m"]

    def physical_bounds(self, state):
        lift_range = state.health["theta"]["range"]
        bounds = {LIFT_JOINT: tuple(sorted(self.lift_position(v, lift_range) for v in lift_range))}
        for side, names in (("left", LEFT_ARM_JOINTS), ("right", RIGHT_ARM_JOINTS)):
            entry = self.data["arms"][side]
            limits = np.asarray(state.health[side]["joint_limits"], dtype=float)
            if limits.ndim != 2 or limits.shape[0] != 7 or limits.shape[1] < 2:
                raise HardwareSafetyError(f"invalid {side} device position bounds")
            for index, name in enumerate(names):
                ends = limits[entry["sdk_indices"][index], :2] * entry["signs"][index] + entry["offsets_rad"][index]
                bounds[name] = (float(np.min(ends)), float(np.max(ends)))
        return bounds

    def lift_command(self, model_position, reported_range):
        position = float(vector([model_position], 1, "model lift position")[0])
        position -= self.data["lift_zero_offset_m"]
        low, high = vector(reported_range, 2, "lift reported range")
        if high <= low:
            raise HardwareSafetyError("invalid lift reported range")
        return high - position if self.data.get("lift_position_mode", "sdk") == "range_max_minus_sdk" else position

    def lift_velocity(self, velocity):
        value = float(vector([velocity], 1, "lift velocity")[0])
        return -value if self.data.get("lift_position_mode", "sdk") == "range_max_minus_sdk" else value

    def runtime_limits(self, profile, state=None, *, _capabilities=None):
        """One intersection for IK and dispatch, without altering the profile hash."""
        caps = _capabilities or load_hardware_capabilities(self.data.get("hardware_capabilities"))
        effort = np.minimum(profile.mit_servo.torque_limit_nm, caps.peak_torque_nm)
        supplied = self.data.get("hardware_effort_limits_nm")
        if supplied is not None:
            supplied = vector(supplied, 14, "hardware operating effort bounds")
            if np.any(supplied <= 0):
                raise HardwareSafetyError("hardware operating effort bounds must be positive")
            effort = np.minimum(effort, supplied)
        arm_v = np.minimum(profile.arm_velocity_rad_s, caps.velocity_rad_s)
        manufacturer = profile.velocity_limit_authority == "manufacturer"
        if not manufacturer:
            arm_v = np.minimum(arm_v, profile.sdk_velocity_rad_s)
        lift_v = min(profile.lift_velocity_m_s, caps.lift_velocity_m_s,
                     self.data.get("hardware_lift_velocity_limit_m_s", caps.lift_velocity_m_s))
        base_v = np.array([profile.base_linear_velocity_m_s] * 2 + [profile.base_angular_velocity_rad_s])
        base_v[:2] = np.minimum(base_v[:2], self.base_linear_limit(_capabilities=caps) / (1. if manufacturer else np.sqrt(2)))
        supplied = self.data.get("hardware_base_velocity_limits")
        if supplied is not None:
            supplied = vector(supplied, 3, "hardware base velocity bounds")
            if np.any(supplied <= 0):
                raise HardwareSafetyError("hardware base velocity bounds must be positive")
            # SDK per-axis limits are rotated relative to the model. A common
            # inscribed bound remains valid for every yaw and both frames.
            base_v[:2] = np.minimum(base_v[:2], min(supplied[:2]) / (1. if manufacturer else np.sqrt(2)))
            base_v[2] = min(base_v[2], supplied[2])
        if state is not None:
            for side, start in (("left", 0), ("right", 7)):
                row = state.health.get(side, {})
                raw = row.get("joint_limits")
                if manufacturer and row.get("joint_limits_source") == "sdk_static":
                    raw = row.get("device_joint_limits")
                if raw is None:
                    continue
                limits = np.asarray(raw, dtype=float)
                if limits.ndim != 2 or limits.shape[0] != 7 or limits.shape[1] < 2 or limits.shape[1] == 3:
                    raise HardwareSafetyError(f"{side}: device velocity bounds missing")
                if limits.shape[1] == 2:
                    continue  # Position-only feedback: retain hash-bound SDK velocity limits.
                vel = limits[:, 2:4]
                if not np.all(np.isfinite(vel)) or np.any(vel[:, 0] >= 0) or np.any(vel[:, 1] <= 0):
                    raise HardwareSafetyError(f"{side}: invalid device velocity bounds")
                bounds = np.minimum(-vel[:, 0], vel[:, 1])[self.data["arms"][side]["sdk_indices"]]
                arm_v[start:start + 7] = np.minimum(arm_v[start:start + 7], bounds)
            theta = state.health.get("theta", {})
            if "maximum_speed" in theta:
                speed, ppm = vector([theta["maximum_speed"], theta["pulse_per_meter"]], 2, "lift speed units")
                if speed <= 0 or ppm <= 0:
                    raise HardwareSafetyError("invalid device lift speed")
                lift_v = np.floor(min(lift_v, speed / ppm) * ppm) / ppm
                if lift_v <= 0:
                    raise HardwareSafetyError("Theta effective speed below one pulse per second")
        velocity = np.r_[base_v, lift_v, arm_v]
        if self.verification is not None and self.verification["scope"] != "control":
            # Tracking acceptance keeps baseline feedback/acceleration/jerk dynamics.
            # A separate speed ceiling is enforced in both IK and dispatch.
            velocity = np.minimum(velocity, [.05, .05, .1, .02, *([.3] * 14)])
        return {"runtime_velocity_bounds": dict(zip(JOINTS, velocity, strict=True)),
                "runtime_effort_limits_nm": tuple(effort)}

    def velocity_audit(self, profile, state=None):
        caps = load_hardware_capabilities(self.data.get("hardware_capabilities"))
        effective = self.runtime_limits(profile, state)["runtime_velocity_bounds"]
        return {
            "authority": profile.velocity_limit_authority,
            "manufacturer": dict(arms=list(caps.velocity_rad_s), lift=caps.lift_velocity_m_s,
                                 base_resultant=caps.base_linear_speed_m_s),
            "sdk_static_arm_velocity_rad_s": list(profile.sdk_velocity_rad_s),
            "reported_arm_limits": {} if state is None else {
                side: {k: state.health.get(side, {}).get(k) for k in
                       ("joint_limits", "joint_limits_source", "device_joint_limits")}
                for side in ("left", "right")},
            "reported_theta_limits": {} if state is None else {k: state.health.get("theta", {}).get(k)
                for k in ("maximum_speed", "configured_speed", "pulse_per_meter")},
            "base_source_discrepancy": self.data.get("base_documented_limits"),
            "software": dict(arms=list(profile.arm_velocity_rad_s), lift=profile.lift_velocity_m_s,
                             base_resultant=profile.base_linear_velocity_m_s),
            "effective": effective,
            "reference_fraction": profile.reference_velocity_fraction,
            "lift_reference_fraction": profile.lift_reference_velocity_fraction,
            "reference": {k: v * (profile.lift_reference_velocity_fraction if k == LIFT_JOINT
                                   else profile.reference_velocity_fraction) for k, v in effective.items()},
            "base_qp": ("inscribed regular 16-gon; radial loss at most 1-cos(pi/16)"
                        if profile.velocity_limit_authority == "manufacturer" else "legacy per-axis box"),
            "units": "joint output rad/s, lift and base m/s; yaw rad/s",
            "grippers": "fixed posture model; active control requires position interface",
        }

    def authorize(self, profile, stage, supplied_hash):
        from .research_profiles import HIGHSPEED_PROFILES, require_offline_validation
        if profile is not None and profile.profile_id in HIGHSPEED_PROFILES:
            require_offline_validation(profile, mapping=self)
        from .x7 import authorize_live_profile

        self.validate_experiment_options()
        if self.commissioning_stage is not None:
            raise HardwareSafetyError("commissioning scope cannot authorize normal live control")

        if self.data.get("read_only_candidate", False):
            raise HardwareSafetyError("read-only candidate mapping cannot authorize live control")
        commissioning = self.data.get("commissioning")
        if commissioning and stage not in commissioning["allowed_live_stages"]:
            raise HardwareSafetyError("stage not authorized by commissioning configuration")
        if profile is None or profile.motion_envelope != self.motion_envelope:
            raise HardwareSafetyError("mapping/profile motion envelope mismatch")
        if self.quest_experiment and (
            profile.profile_id != commissioning["motion_profile"]
            or profile.sha256 != commissioning["profile_sha256"]
        ):
            raise HardwareSafetyError("experiment requires its exact reviewed profile hash")
        if self.data.get("lift_position_mode", "sdk") != "sdk" and not commissioning:
            raise HardwareSafetyError("reverse lift mapping is mirror-only; live control is not commissioned")
        authorize_live_profile(profile, supplied_hash)
        review = self.effective_review
        required = ("mapping_checked", "same_tool_empty", "mit_units_gains_checked",
                    "drive_protection_checked", "disconnect_stop_checked")
        missing = [name for name in required if review.get(name) is not True]
        if self.allow_unverified_hardware_limits and "drive_protection_checked" in missing:
            missing.remove("drive_protection_checked")  # Unknown, NOT verified.
        if self.verification is not None and "drive_protection_checked" in missing:
            missing.remove("drive_protection_checked")  # Explicit research acceptance, never a verified clamp.
        if commissioning:
            missing.extend(name for name in ("sdk_startup_checked", "initialization_stop_checked")
                           if review.get(name) is not True)
        if self.demo_authorized and self.runtime_session:
            missing = [name for name in missing if name not in {
                "drive_protection_checked", "disconnect_stop_checked",
                "sdk_startup_checked", "initialization_stop_checked"}]
        if "theta" in STAGES[stage] and review.get("theta_stop_checked") is not True:
            missing.append("theta_stop_checked")
        if "base" in STAGES[stage] and review.get("base_axes_checked") is not True:
            missing.append("base_axes_checked")
        if missing or not str(review.get("evidence", "")).strip():
            raise HardwareSafetyError("hardware review required: " + ", ".join(missing or ["evidence"]))
        effort_value = self.data["hardware_effort_limits_nm"]
        if effort_value is None and self.allow_unverified_hardware_limits:
            effort_value = profile.mit_servo.torque_limit_nm  # Software estimate only.
        effort = vector(effort_value, 14, "reviewed hardware effort limits or experimental software bound")
        if np.any(effort <= 0):
            raise HardwareSafetyError("hardware effort limits must be positive, not inferred from simulation")
        if "base" in STAGES[stage]:
            value = self.data.get("hardware_base_velocity_limits")
            if value is None and self.allow_unverified_hardware_limits:
                value = [profile.base_linear_velocity_m_s] * 2 + [profile.base_angular_velocity_rad_s]
            bounds = vector(value, 3, "reviewed base vx/vy/wz limits or experimental software bound")
            if np.any(bounds <= 0):
                raise HardwareSafetyError("reviewed base limits must be positive")
        # These limits are an independent reject gate, never an SDK torque-cap field.
        return np.minimum(effort, self.runtime_limits(profile)["runtime_effort_limits_nm"])

    def base_linear_limit(self, *, _capabilities=None):
        caps = _capabilities or load_hardware_capabilities(self.data.get("hardware_capabilities"))
        cap = caps.base_linear_speed_m_s
        configured = self.data.get("hardware_base_linear_speed_limit_m_s", cap)
        value = float(vector([configured], 1, "base resultant speed limit")[0])
        if value <= 0:
            raise HardwareSafetyError("base resultant speed limit must be positive")
        return min(cap, value)


@dataclass(frozen=True)
class WholeBodyState:
    q: np.ndarray
    v: np.ndarray
    received: dict[str, float]
    sequences: dict[str, int]
    health: dict
    device_timestamps: dict
    dispatch: dict | None = None
    timing: dict | None = None
    feedback_policy: str = "strict"
    latency_policy: str = "strict"

    def validate(self, now, *, check_skew=True, check_age=True):
        vector(self.q, 18, "whole-body position")
        vector(self.v, 18, "whole-body velocity")
        from ..control.feedback_policy import FeedbackAgeError, FeedbackSkewError, validate_received
        from .x7 import ControlDelay, FeedbackSkewSafetyError

        try:
            validate_received(self.received, now, self.feedback_policy, check_skew=check_skew, check_age=check_age)
        except FeedbackAgeError as error:
            raise ControlDelay(str(error)) from error
        except FeedbackSkewError as error:
            raise FeedbackSkewSafetyError(str(error)) from error
        except ValueError as error:
            raise HardwareSafetyError(str(error)) from error

    @property
    def timestamp(self):
        # MIT arm feedback epoch. Theta keeps its own timestamp, passed to IK
        # separately; this is NOT a claim that the entire body was sampled now.
        from ..control.feedback_policy import coupled_groups
        names = coupled_groups(self.feedback_policy)
        return min(self.received[n] for n in names)

    def ik_feedback(self):
        return {"feedback_received": self.received, "feedback_policy": self.feedback_policy}


@dataclass(frozen=True)
class WholeBodyCommand:
    sequence: int
    created: float
    expires: float
    state_received: dict[str, float]
    measured_q: np.ndarray
    positions: np.ndarray
    velocity: np.ndarray
    mit: MitCommand
    base_body: np.ndarray
    control_phase: str = "tracking"
    enqueued_at: float | None = None
    cycle_started: float | None = None
    recovery_epoch: int = 0


@dataclass(frozen=True)
class PreparedCommandGuard:
    """Process-local validated constants, prepared before device acquisition."""

    profile: object
    mapping: RobotMapping
    stage: str
    effort: tuple
    config: object
    capabilities: object


class CommandGuard:
    """Validate the complete group before any write; latch faults in the gateway."""

    @staticmethod
    def prepare(profile, mapping, stage, effort_limits):
        mapping.validate_experiment_options()
        if mapping.data.get("base_control_enabled", True) != profile.base_control_enabled:
            raise HardwareSafetyError("mapping/profile base control mismatch")
        if profile.motion_envelope != mapping.motion_envelope:
            raise HardwareSafetyError("command guard motion envelope mismatch")
        if stage not in STAGES:
            raise HardwareSafetyError("invalid command guard stage")
        from ..control.motion_profile import controller_config_for_profile
        from ..control.preparation import validate_preparation_command  # noqa: F401

        config = controller_config_for_profile(profile)
        caps = load_hardware_capabilities(mapping.data.get("hardware_capabilities"))
        return PreparedCommandGuard(profile, mapping, stage, tuple(vector(effort_limits, 14, "effort limits")),
                                    config, caps)

    def __init__(self, profile, mapping, stage, initial, effort_limits, *, prepared=None, stationary_reference=False):
        prepared = prepared or self.prepare(profile, mapping, stage, effort_limits)
        if (prepared.profile is not profile or prepared.mapping is not mapping or prepared.stage != stage
                or prepared.effort != tuple(effort_limits)):
            raise HardwareSafetyError("prepared command guard binding mismatch")
        self.capabilities = prepared.capabilities
        self.profile, self.mapping, self.stage = profile, mapping, stage
        self.initial = initial
        self.effort_limits = vector(effort_limits, 14, "effort limits")
        self.last_sequence = -1
        self.last_command = None
        self.last_velocity = initial.v.copy()
        if stationary_reference:
            from ..control.reference_initialization import stationary_reference_velocity

            self.last_velocity = stationary_reference_velocity(initial.v)
        self.last_acceleration = np.zeros(18)
        from ..control.following_error import FollowingErrorMonitor
        self.following_monitor = FollowingErrorMonitor()
        self.phase_anchor = initial.q.copy()
        self.last_phase = "tracking"
        config = prepared.config
        self.base_config = config.base
        self.velocity_limits = np.r_[config.base.linear_velocity_limit, config.base.linear_velocity_limit,
                                     config.base.angular_velocity_limit, config.lift_velocity_limit,
                                     config.resolved_arm_limits("velocity")]
        self.acceleration_limits = np.r_[config.base.linear_acceleration_limit, config.base.linear_acceleration_limit,
                                         config.base.angular_acceleration_limit, config.lift_acceleration_limit,
                                         config.resolved_arm_limits("acceleration")]
        self.jerk_limits = np.r_[config.base.linear_jerk_limit, config.base.linear_jerk_limit,
                                 config.base.angular_jerk_limit, config.lift_jerk_limit,
                                 config.resolved_arm_limits("jerk")]
        resolved = mapping.runtime_limits(profile, initial, _capabilities=self.capabilities)
        self._limits_cache = (self._limits_key(initial), resolved)
        self.velocity_limits = np.minimum(self.velocity_limits, list(resolved["runtime_velocity_bounds"].values()))
        self.effort_limits = np.minimum(self.effort_limits, resolved["runtime_effort_limits_nm"])

    def _limits_key(self, state):
        # Include every live/config input to the cached intersection. A firmware
        # metadata change invalidates it; health is deliberately never cached.
        return (
            tuple((side, np.asarray(state.health.get(side, {}).get("joint_limits", []), dtype=float).shape,
                   np.asarray(state.health.get(side, {}).get("joint_limits", []), dtype=float).tobytes())
                  for side in ("left", "right")),
            tuple((side, state.health.get(side, {}).get("joint_limits_source"),
                   repr(state.health.get(side, {}).get("device_joint_limits"))) for side in ("left", "right")),
            state.health.get("theta", {}).get("maximum_speed"),
            state.health.get("theta", {}).get("pulse_per_meter"),
            repr(self.mapping.data.get("hardware_effort_limits_nm")),
            repr(self.mapping.data.get("hardware_base_velocity_limits")),
        )

    def validate(self, cmd, state, now):
        if not self.profile.base_control_enabled and (
                np.any(np.abs(cmd.velocity[:3]) > 1e-7) or np.any(np.abs(cmd.base_body) > 1e-7)):
            raise HardwareSafetyError("base control is disabled: nonzero base command")
        if state.feedback_policy != self.mapping.feedback_policy:
            raise HardwareSafetyError("feedback policy mismatch")
        state.validate(now)
        physical = np.r_[self.capabilities.base_linear_speed_m_s, self.capabilities.base_linear_speed_m_s,
                         self.profile.base_angular_velocity_rad_s, self.capabilities.lift_velocity_m_s,
                         self.capabilities.velocity_rad_s]
        bad = np.flatnonzero(np.abs(state.v) > physical + 1e-7)
        if len(bad):
            i = int(bad[0])
            raise HardwareSafetyError(f"measured overspeed axis={JOINTS[i]} value={state.v[i]} "
                                      f"manufacturer={physical[i]} effective={self.velocity_limits[i]}")
        if np.linalg.norm(state.v[:2]) > self.mapping.base_linear_limit(_capabilities=self.capabilities) + 1e-7:
            raise HardwareSafetyError("measured base resultant speed exceeds manufacturer/software limit")
        times = vector([cmd.created, cmd.expires, now], 3, "command times")
        if times[0] > now or times[1] < times[0] or cmd.expires - cmd.created > COMMAND_TTL + 1e-9:
            raise HardwareSafetyError("expired/future command")
        if now >= times[1]:
            from .x7 import ControlDelay
            raise ControlDelay("expired/future command")
        if type(cmd.sequence) is not int or cmd.sequence != self.last_sequence + 1:
            raise HardwareSafetyError("replayed or skipped command sequence")
        from ..control.feedback_policy import FeedbackAgeError, FeedbackSkewError, validate_received

        try:
            validate_received(cmd.state_received, now, self.mapping.feedback_policy)
        except (FeedbackAgeError, FeedbackSkewError) as error:
            from .x7 import ControlDelay
            raise ControlDelay(f"command was calculated from stale feedback: {error}") from error
        except ValueError as error:
            raise HardwareSafetyError(f"command was calculated from stale feedback: {error}") from error
        q, v = vector(cmd.positions, 18, "command positions"), vector(cmd.velocity, 18, "command velocity")
        limit = self.mapping.arm_excursion_limit(self.stage)
        if limit is not None:
            if (np.any(np.abs(q[4:] - self.initial.q[4:]) > limit)
                    or np.any(np.abs(state.q[4:] - self.initial.q[4:]) > limit)):
                raise HardwareSafetyError(f"commissioning arm excursion exceeds {np.rad2deg(limit):g} degrees")
        body = vector(cmd.base_body, 3, "base body velocity")
        if np.linalg.norm(body[:2]) > self.mapping.base_linear_limit(_capabilities=self.capabilities) + 1e-9:
            raise HardwareSafetyError("base resultant linear speed exceeds manufacturer rating")
        key = self._limits_key(state)
        if key != self._limits_cache[0]:
            self._limits_cache = (key, self.mapping.runtime_limits(self.profile, state, _capabilities=self.capabilities))
        resolved = self._limits_cache[1]
        if self.profile.velocity_limit_authority == "manufacturer":
            resultant = min(list(resolved["runtime_velocity_bounds"].values())[:2])
            if np.linalg.norm(body[:2]) > resultant + 1e-7:
                raise HardwareSafetyError("effective base resultant velocity limit exceeded")
        if np.any(np.abs(v) > np.asarray(list(resolved["runtime_velocity_bounds"].values())) + 1e-7):
            raise HardwareSafetyError("effective hardware velocity limit exceeded")
        if cmd.mit is None or tuple(cmd.mit.joint_names) != (*LEFT_ARM_JOINTS, *RIGHT_ARM_JOINTS):
            raise HardwareSafetyError("complete, ordered dual-arm MIT command required")
        if abs(cmd.mit.timestamp_s - cmd.created) > 1e-9:
            raise HardwareSafetyError("MIT snapshot timestamp does not match command generation")
        if not np.allclose(cmd.mit.position, q[4:], atol=1e-9, rtol=0) or not np.allclose(
            cmd.mit.velocity, v[4:], atol=1e-9, rtol=0
        ):
            raise HardwareSafetyError("MIT/reference mismatch")
        servo = self.profile.mit_servo
        if not np.array_equal(cmd.mit.kp, servo.kp) or not np.array_equal(cmd.mit.kd, servo.kd):
            raise HardwareSafetyError("MIT gains differ from immutable profile")
        source_q = vector(cmd.measured_q, 18, "command source position")
        c, s = np.cos(source_q[2]), np.sin(source_q[2])
        expected_body = np.r_[np.array([[c, s], [-s, c]]) @ v[:2], v[2]]
        if not np.allclose(body, expected_body, atol=1e-7, rtol=0):
            raise HardwareSafetyError("base body/world velocity mismatch")
        if np.any(np.abs(v) > self.velocity_limits + 1e-7):
            raise HardwareSafetyError("profile velocity limit exceeded")
        if "base" in STAGES[self.stage]:
            value = self.mapping.data["hardware_base_velocity_limits"]
            if value is None and self.mapping.allow_unverified_hardware_limits:
                value = [self.profile.base_linear_velocity_m_s] * 2 + [self.profile.base_angular_velocity_rad_s]
            limits = vector(value, 3, "base hardware/software limits")
            sdk_body = np.r_[self.mapping.rotation().T @ body[:2], body[2]]
            if np.any(np.abs(sdk_body) > limits):
                raise HardwareSafetyError("base command exceeds reviewed hardware limits")
        base = self.base_config
        if base.workspace_mode == "bounded":
            low = np.array([base.x_limits_m[0] + base.position_margin_m, base.y_limits_m[0] + base.position_margin_m,
                            base.yaw_limits_rad[0] + base.yaw_margin_rad])
            high = np.array([base.x_limits_m[1] - base.position_margin_m, base.y_limits_m[1] - base.position_margin_m,
                             base.yaw_limits_rad[1] - base.yaw_margin_rad])
            if np.any(q[:3] < low) or np.any(q[:3] > high):
                raise HardwareSafetyError("base command outside startup-centered workspace")
        dt = self.profile.control_dt_s
        previous = self.initial.q if self.last_command is None else self.last_command.positions
        if not np.allclose((q[4:] - previous[4:]) / dt, v[4:], atol=1e-6, rtol=0):
            raise HardwareSafetyError("discontinuous MIT position reference")
        if self.profile.lift_reference_max_error_m is not None:
            if not np.isclose((q[3] - previous[3]) / dt, v[3], atol=1e-6, rtol=0):
                raise HardwareSafetyError("discontinuous lift position reference")
            if abs(q[3] - state.q[3]) > self.profile.lift_reference_max_error_m + 1e-7:
                raise HardwareSafetyError("hard lift following error")
        acceleration = (v - self.last_velocity) / dt
        jerk = (acceleration - self.last_acceleration) / dt
        if np.any(np.abs(acceleration) > self.acceleration_limits + 1e-5) or np.any(
            np.abs(jerk) > self.jerk_limits + 1e-3
        ):
            raise HardwareSafetyError("profile acceleration/jerk limit exceeded")
        from ..control.preparation import validate_preparation_command

        if cmd.control_phase != "tracking" and not self.mapping.runtime_session:
            raise HardwareSafetyError("preparation commands require runtime session mapping")
        phase_anchor = previous.copy() if cmd.control_phase != self.last_phase else self.phase_anchor
        try:
            validate_preparation_command(cmd.control_phase, q, v, previous, self.last_velocity,
                                         self.last_acceleration, dt, phase_anchor, measured=state.q,
                                         lift_velocity=self.profile.preparation_lift_velocity_m_s)
        except ValueError as error:
            raise HardwareSafetyError(str(error)) from error
        for name, part in SLICES.items():
            if name not in STAGES[self.stage]:
                drift_limit = 0.001 if name == "theta" else 0.002
                if np.max(np.abs(state.q[part] - self.initial.q[part])) > drift_limit:
                    raise HardwareSafetyError(f"locked {name} moved")
                if np.max(np.abs(v[part])) > 1e-7 or np.max(np.abs(q[part] - self.initial.q[part])) > drift_limit:
                    raise HardwareSafetyError(f"command moves locked {name}")
        effort = cmd.mit.kp * (q[4:] - state.q[4:]) + cmd.mit.kd * (v[4:] - state.v[4:]) + cmd.mit.torque_ff
        bounds = np.minimum(self.effort_limits, servo.torque_limit_nm)
        if np.any(np.abs(effort) > bounds + 1e-7) or np.any(np.abs(cmd.mit.torque_ff) > bounds + 1e-7):
            violation = int(np.flatnonzero((np.abs(effort) > bounds + 1e-7)
                                           | (np.abs(cmd.mit.torque_ff) > bounds + 1e-7))[0])
            raise HardwareSafetyError(
                "predicted MIT effort exceeds reviewed bound (not a physical torque cap): "
                f"joint={cmd.mit.joint_names[violation]}, predicted_nm={effort[violation]:.6f}, "
                f"feedforward_nm={cmd.mit.torque_ff[violation]:.6f}, bound_nm={bounds[violation]:.6f}, remaining_nm={bounds[violation]-abs(effort[violation]):.6f}")
        error = np.max(np.abs(q[4:] - state.q[4:]))
        from ..control.following_error import governed_profile
        following_fault = self.following_monitor.check(error, now, governed=(
            governed_profile(self.profile.profile_id) and cmd.control_phase == 'tracking'))
        if following_fault:
            raise HardwareSafetyError(following_fault)
        self.last_phase, self.phase_anchor = cmd.control_phase, phase_anchor
        self.last_sequence, self.last_command = cmd.sequence, cmd
        self.last_velocity, self.last_acceleration = v.copy(), acceleration
