"""Initial command history for an explicitly stationary, newly acquired robot.

Measured velocity remains independent and must still reach every physical check.
This policy is never a way to reset a moving command history.
"""
import numpy as np


def stationary_reference_velocity(measured_velocity, *, base_dofs=3):
    velocity = np.asarray(measured_velocity, dtype=float)
    if velocity.shape != (base_dofs + 15,) or not np.all(np.isfinite(velocity)):
        raise ValueError("invalid measured velocity for stationary reference initialization")
    if (np.any(abs(velocity[:base_dofs + 1]) > .002)
            or np.any(abs(velocity[base_dofs + 1:]) > .01)):
        raise ValueError("stationary reference initialization requires a stationary robot")
    return np.zeros_like(velocity)
