"""Additional preparation limits; never replace the baseline or its histories."""

from types import SimpleNamespace

import numpy as np
from pink.limits import Limit

from .command_limits import JerkSafeVelocityLimit

PHASES = ("tracking", "preparation", "lift-retreat")
VELOCITY = np.r_[0.05, 0.05, 0.1, 0.02, np.full(14, 0.3)]
ACCELERATION = np.r_[0.1, 0.1, 0.2, 0.04, np.full(14, 0.6)]
JERK = np.r_[0.5, 0.5, 1.0, 0.2, np.full(14, 3.0)]


def velocity_limits(lift_velocity=.02):
    if not np.isfinite(lift_velocity) or not 0 < lift_velocity <= .24:
        raise ValueError("invalid preparation lift speed")
    values = VELOCITY.copy()
    values[3] = lift_velocity
    return values


class PreparationLimit(Limit):
    def __init__(self, model, history, dt, *, lift_velocity=.02):
        if model.nv != 18:
            raise ValueError("preparation requires the planar whole-body model")
        self.inward = False
        self.history = history
        self.velocity = np.minimum(model.velocityLimit, velocity_limits(lift_velocity))
        self.matrix = np.vstack((np.eye(18), -np.eye(18)))
        self.proxy = SimpleNamespace(previous_velocity=history.previous_velocity, jerk_limit=JERK)
        self.viability = JerkSafeVelocityLimit(
            SimpleNamespace(nv=18, velocityLimit=self.velocity), self.proxy, ACCELERATION, dt
        )

    def compute_qp_inequalities(self, configuration, dt):
        v, a = self.history.previous_velocity, self.history.previous_acceleration
        upper = np.minimum(v + ACCELERATION * dt, v + a * dt + JERK * dt**2)
        lower = np.maximum(v - ACCELERATION * dt, v + a * dt - JERK * dt**2)
        self.proxy.previous_velocity = v
        _, viable = self.viability.compute_qp_inequalities(configuration, dt)
        upper = np.minimum(upper, viable[:18] / dt)
        lower = np.maximum(lower, -viable[18:] / dt)
        if self.inward:
            upper[3] = min(upper[3], 0.0)
        return self.matrix, dt * np.r_[upper, -lower]


def validate_preparation_command(
    phase, positions, velocity, previous, previous_velocity, previous_acceleration, dt, anchor, *,
    measured=None, lift_velocity=.02
):
    if phase not in PHASES:
        raise ValueError("unknown control phase")
    if phase == "tracking":
        return
    a = (velocity - previous_velocity) / dt
    j = (a - previous_acceleration) / dt
    if (np.any(abs(velocity) > velocity_limits(lift_velocity) + 1e-7)
            or np.any(abs(a) > ACCELERATION + 1e-5) or np.any(abs(j) > JERK + 1e-3)):
        raise ValueError("preparation velocity/acceleration/jerk exceeded")
    if np.any(abs(velocity[:3]) > 1e-7) or np.any(abs(positions[:3] - anchor[:3]) > np.array([0.003, 0.003, 0.01])):
        raise ValueError("preparation moves the base")
    if phase == "lift-retreat":
        if np.any(abs(velocity[4:]) > 1e-7) or np.any(abs(positions[4:] - anchor[4:]) > 1e-6):
            raise ValueError("lift retreat moves arms")
        measured = previous if measured is None else measured
        if velocity[3] > 1e-9 or positions[3] > measured[3] + 1e-8:
            raise ValueError("lift retreat must move inward from the upper endpoint")
