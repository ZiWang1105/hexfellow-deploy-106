"""Bound a persistent lift reference without resetting it to falling feedback."""
from types import SimpleNamespace

import numpy as np

from .command_limits import BoundaryStateUnrecoverable, FiniteBaseBoundaryGovernor


class LiftReferenceGovernor:
    def __init__(self, model, index, acceleration, jerk, maximum_error_m):
        if not np.isfinite(maximum_error_m) or maximum_error_m <= 0:
            raise ValueError("lift reference tracking bound must be positive")
        self.index = index
        self.maximum_error_m = maximum_error_m
        self.lower = float(model.lowerPositionLimit[index])
        self.upper = float(model.upperPositionLimit[index])
        self.model = SimpleNamespace(
            nv=model.nv, lowerPositionLimit=model.lowerPositionLimit.copy(),
            upperPositionLimit=model.upperPositionLimit.copy(), velocityLimit=model.velocityLimit.copy(),
        )
        self.governor = FiniteBaseBoundaryGovernor(self.model, [index], acceleration, jerk)
        self.governor.labels = ("lift_reference_lower", "lift_reference_upper")

    def limits(self, measured, reference, measured_velocity, velocity, acceleration,
               reaction_s, uncertainty, dt, *, upper_retreat=False):
        i = self.index
        gap = float(reference[i] - measured[i])
        if abs(gap) > self.maximum_error_m + 1e-7:
            raise BoundaryStateUnrecoverable("lift reference tracking bound exceeded", {
                "lift_tracking_error_m": gap, "maximum_error_m": self.maximum_error_m})
        # The physical bounds and the local tracking window are hard constraints.
        # Existing jerk-prefix braking reserves prevent a stalled plant from
        # winding the reference up to the window edge at nonzero speed.
        self.model.lowerPositionLimit[i] = max(self.lower, measured[i] - self.maximum_error_m)
        # At the physical upper stop, the existing retreat phase permits only
        # inward motion. The physical interval truncates uncertainty above that
        # stop; cancel ONLY this upper physical reserve, not the tracking window.
        upper = self.upper + (uncertainty[i] if upper_retreat else 0.)
        self.model.upperPositionLimit[i] = min(upper, measured[i] + self.maximum_error_m)
        return self.governor.limits(measured, reference, measured_velocity, velocity,
                                    acceleration, reaction_s, uncertainty, dt)
