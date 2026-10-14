"""Explicit experimental motion envelope; never a hardware rating."""

import hashlib
import json
from dataclasses import replace

CAPS = {
    "arm_velocity_rad_s": 0.3, "arm_acceleration_rad_s2": 0.6, "arm_jerk_rad_s3": 3.0,
    "base_linear_velocity_m_s": 0.05, "base_angular_velocity_rad_s": 0.1,
    "base_linear_acceleration_m_s2": 0.1, "base_angular_acceleration_rad_s2": 0.2,
    "base_linear_jerk_m_s3": 0.5, "base_angular_jerk_rad_s3": 1.0,
    "lift_velocity_m_s": 0.02, "lift_acceleration_m_s2": 0.04, "lift_jerk_m_s3": 0.2,
}


def apply_envelope(profile, envelope):
    if envelope == "none":
        if profile.motion_envelope != "none":
            raise ValueError("cannot remove an applied motion envelope")
        return profile
    if envelope != "first-trial" or profile.profile_id not in {
            "x7-mit-highspeed-v10", "x7-mit-highspeed-v10-base-v1", "x7-mit-highspeed-v10-base-lift-v2", "x7-mit-highspeed-v10-base-lift-v3", "x7-mit-highspeed-v10-base-lift-v4"}:
        raise ValueError("first-trial envelope requires v10")
    values = {}
    for key, cap in CAPS.items():
        old = getattr(profile, key)
        values[key] = tuple(min(v, cap) for v in old) if isinstance(old, tuple) else min(old, cap)
    # sha256/raw identify the immutable BASE profile, not the effective limits.
    return replace(profile, **values, motion_envelope=envelope)


def envelope_report(profile):
    limits = {key: getattr(profile, key) for key in CAPS}
    report = {"base_profile_sha256": profile.sha256, "motion_envelope": profile.motion_envelope,
              "effective_limits": limits, "posture_policy_preserved": True,
              "source": "experimental software limits, not hardware ratings"}
    report["effective_limits_sha256"] = hashlib.sha256(
        json.dumps(report, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    return report
