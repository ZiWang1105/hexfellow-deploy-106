"""Measured commissioning acceptance, independent of control and transport."""

import math

import numpy as np

POSITION_TOLERANCE_M = 0.004
ANGLE_RELATIVE_TOLERANCE = 0.05
ANGLE_STABLE_SECONDS = 0.5
ARM_SETTLED_SPEED = 0.01
MAX_OBSERVATION_GAP_S = 0.04


class ArmMotionAccuracy:
    """Qualify the final continuous interval of the outbound endpoint dwell."""

    def __init__(self, target_angle_deg, leg_seconds, dwell_seconds):
        if not math.isfinite(target_angle_deg) or target_angle_deg == 0:
            raise ValueError("a finite nonzero target angle is required")
        self.target = float(target_angle_deg)
        self.start = leg_seconds
        self.end = leg_seconds + dwell_seconds
        self.previous = None
        self.stable_since = None
        self.duration = 0.0
        self.measured = None
        self.speed = None
        self.error = None
        self.relative_error = None
        self.finalized = False
        self.passed = False

    def observe(self, elapsed, received, measured_angle_deg, velocity_rad_s, *, now):
        if self.finalized or elapsed < self.start:
            return
        if elapsed >= self.end:
            fresh = self.previous is not None and 0 <= now - self.previous <= MAX_OBSERVATION_GAP_S
            self.finalized = True
            self.passed = bool(fresh and self.duration + 1e-12 >= ANGLE_STABLE_SECONDS)
            return
        if self.previous is not None and received <= self.previous:
            if received < self.previous or now - received > MAX_OBSERVATION_GAP_S:
                self.stable_since = None
                self.duration = 0.0
            return
        gap = None if self.previous is None else received - self.previous
        self.previous = received
        self.measured = float(measured_angle_deg)
        self.speed = float(velocity_rad_s)
        self.error = abs(self.measured - self.target)
        self.relative_error = self.error / abs(self.target)
        good = (
            math.isfinite(self.relative_error) and math.isfinite(self.speed)
            and self.relative_error <= ANGLE_RELATIVE_TOLERANCE + 1e-12
            and abs(self.speed) <= ARM_SETTLED_SPEED
            and 0 <= now - received <= MAX_OBSERVATION_GAP_S
        )
        if not good:
            self.stable_since = None
            self.duration = 0.0
        else:
            if self.stable_since is None or gap is None or gap > MAX_OBSERVATION_GAP_S:
                self.stable_since = received
            self.duration = received - self.stable_since

    def report(self):
        return {
            "target_angle_deg": self.target,
            "measured_angle_deg": self.measured,
            "absolute_error_deg": self.error,
            "relative_error": self.relative_error,
            "relative_error_tolerance": ANGLE_RELATIVE_TOLERANCE,
            "joint_velocity_rad_s": self.speed,
            "velocity_tolerance_rad_s": ARM_SETTLED_SPEED,
            "stable_seconds": self.duration,
            "required_stable_seconds": ANGLE_STABLE_SECONDS,
            "finalized": self.finalized,
            "passed": self.passed,
        }

    def failure_message(self):
        error = "no endpoint feedback" if self.relative_error is None else f"error={100 * self.relative_error:.3f}%"
        return (f"target angle accuracy failed: {error}, tolerance=5%, "
                f"stationary fresh interval={self.duration:.3f}s (required 0.5s)")


def arm_tracking_errors(initial_q, target_q, command_q, measured_q, index):
    target, command, measured = (
        float(np.rad2deg(q[index] - initial_q[index])) for q in (target_q, command_q, measured_q)
    )
    return {
        "target_angle_deg": target, "command_angle_deg": command, "measured_angle_deg": measured,
        "command_minus_target_deg": command - target,
        "measured_minus_command_deg": measured - command,
        "measured_minus_target_deg": measured - target,
    }


def settling_report(errors, q, initial_q, velocity, *, posture_test, control_state):
    reasons = []
    for side, error in errors.items():
        if error["position_m"] > POSITION_TOLERANCE_M:
            reasons.append(f"{side} position {1000 * error['position_m']:.3f}mm > 4mm")
        if error["rotation_rad"] > np.deg2rad(2):
            reasons.append(f"{side} rotation {np.rad2deg(error['rotation_rad']):.3f}deg > 2deg")
    arm_speed = float(np.max(np.abs(velocity[4:])))
    body_speed = float(np.max(np.abs(velocity[:4])))
    if arm_speed > ARM_SETTLED_SPEED:
        reasons.append(f"arm speed {arm_speed:.6f}rad/s > 0.01rad/s")
    if body_speed > 0.002:
        reasons.append(f"base/lift generalized speed {body_speed:.6f} > 0.002")
    body_delta = np.asarray(q[:4]) - np.asarray(initial_q[:4])
    if posture_test and np.any(abs(body_delta) > [0.003, 0.003, 0.01, 0.003]):
        reasons.append("base/lift return displacement exceeds existing limits")
    # 'settled' remains usable during initial BRAKING; completion also requires HOLD.
    return {
        "settled": not reasons, "tcp_errors": errors,
        "position_tolerance_m": POSITION_TOLERANCE_M, "rotation_tolerance_rad": float(np.deg2rad(2)),
        "max_arm_speed_rad_s": arm_speed, "arm_speed_tolerance_rad_s": ARM_SETTLED_SPEED,
        "max_body_generalized_speed": body_speed, "body_speed_tolerance": 0.002,
        "body_return_delta": body_delta.tolist(), "control_state": control_state,
        "unmet_conditions": reasons + ([] if control_state == "HOLD" else [f"control state is {control_state}"]),
    }
