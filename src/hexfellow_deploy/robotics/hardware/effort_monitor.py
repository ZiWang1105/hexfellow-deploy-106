"""Load statistics only: no inferred thermal model or unverified thermal trip."""
from __future__ import annotations

import numpy as np


class EffortLoadMonitor:
    """Integrate valid consecutive samples; do not bridge gaps or stale packets."""

    def __init__(self, continuous):
        self.continuous = np.asarray(continuous, dtype=float)
        self.integral = np.zeros_like(self.continuous)
        self.above_s = np.zeros_like(self.continuous)
        self.duration_s = 0.
        self.previous = None

    def observe(self, timestamp, effort):
        if timestamp is None or effort is None:
            self.previous = None
            return
        value = np.asarray(effort, dtype=float)
        if value.shape != self.continuous.shape or not np.isfinite(value).all() or not np.isfinite(timestamp):
            self.previous = None
            return
        if self.previous is not None:
            old_time, old = self.previous
            dt = float(timestamp) - old_time
            if dt <= 0:
                return
            if dt <= .1:
                self.integral += .5 * (old**2 + value**2) * dt
                self.above_s += .5 * ((abs(old) > self.continuous).astype(float)
                                     + (abs(value) > self.continuous).astype(float)) * dt
                self.duration_s += dt
        self.previous = (float(timestamp), value.copy())

    def report(self):
        return dict(observed_duration_s=self.duration_s,
                    rms_nm=None if self.duration_s == 0 else np.sqrt(self.integral/self.duration_s).tolist(),
                    above_continuous_s=self.above_s.tolist(), continuous_nm=self.continuous.tolist(),
                    scope='sampled load indicator; not a thermal model or overload-duration authorization')
