"""Feedback-aware reference slowdown and independent following-error limits.

The governor changes the QP objective, never the solved command or its hard
constraints. These research-policy values are not hardware qualification.
"""
from dataclasses import dataclass

import numpy as np


def governed_profile(profile_id):
    return profile_id in {
        'x7-gp80-highspeed-fixed-base-v1',
        'x7-gp80-highspeed-fixed-base-v2',
        'x7-gp80-highspeed-fixed-base-v3',
    }


@dataclass(frozen=True)
class FollowingErrorPolicy:
    slow_start_rad: float = .02
    brake_rad: float = .05
    prediction_s: float = .04
    release_per_s: float = 2.
    sustained_fault_rad: float = .08
    sustained_fault_s: float = .1
    hard_fault_rad: float = .10
    recovery_timeout_s: float = 2.


POLICY = FollowingErrorPolicy()


class FollowingErrorGovernor:
    def __init__(self):
        self.reset()

    def reset(self):
        self.scale = 1.
        self.error_rad = self.predicted_error_rad = 0.
        self.joint_index = 0

    def update(self, reference_q, measured_q, reference_v, measured_v, dt):
        error = np.asarray(reference_q) - np.asarray(measured_q)
        relative_v = np.asarray(reference_v) - np.asarray(measured_v)
        projected = error + POLICY.prediction_s * relative_v
        envelope = np.maximum(np.abs(error), np.abs(projected))
        self.joint_index = int(np.argmax(envelope))
        self.error_rad = float(np.max(np.abs(error)))
        self.predicted_error_rad = float(np.max(envelope))
        demand = float(np.clip((POLICY.brake_rad - self.predicted_error_rad)
                              / (POLICY.brake_rad - POLICY.slow_start_rad), 0., 1.))
        # Attack immediately; release gradually. The QP's acceleration and jerk
        # inequalities determine the actual continuous braking trajectory.
        self.scale = min(demand, self.scale + POLICY.release_per_s * dt)
        return self.scale

    def observation(self):
        return dict(scale=self.scale, error_rad=self.error_rad,
                    predicted_error_rad=self.predicted_error_rad,
                    joint_index=self.joint_index,
                    active=self.scale < 1. - 1e-9)


class FollowingErrorMonitor:
    """Publisher-side backstop, independent of the IK governor's state."""
    def __init__(self):
        self.severe_since = self.moderate_since = None
        self.governed = None

    def check(self, error, now, *, governed):
        if error > POLICY.hard_fault_rad:
            return 'hard arm following error'
        if governed != self.governed:
            self.severe_since = self.moderate_since = None
            self.governed = governed
        threshold = POLICY.sustained_fault_rad if governed else POLICY.brake_rad
        self.severe_since = (now if self.severe_since is None else self.severe_since) if error > threshold else None
        self.moderate_since = (now if self.moderate_since is None else self.moderate_since) if error > POLICY.brake_rad else None
        if self.severe_since is not None and now - self.severe_since >= POLICY.sustained_fault_s:
            return 'sustained arm following error'
        if governed and self.moderate_since is not None and now - self.moderate_since >= POLICY.recovery_timeout_s:
            return 'arm following error did not recover within timeout'
        return None
