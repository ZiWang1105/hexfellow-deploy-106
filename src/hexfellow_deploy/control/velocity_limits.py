"""Resolve arm speed settings against the selected, immutable UMI profile."""
import numpy as np

from ..robot.joints import LEFT, RIGHT


def arm_velocity_bounds(setting, profile):
    """'profile' retains per-joint limits; a number can only narrow them."""
    values = np.asarray(profile.arm_velocity_rad_s, dtype=float)
    if values.shape != (14,) or not np.isfinite(values).all() or np.any(values <= 0):
        raise ValueError('Invalid UMI arm velocity limits')
    if setting != 'profile':
        if (isinstance(setting, bool) or not isinstance(setting, (int, float))
                or not np.isfinite(setting) or setting <= 0):
            raise ValueError("Arm velocity limit must be 'profile' or a finite positive number")
        values = np.minimum(values, setting)
    return dict(zip((*LEFT, *RIGHT), map(float, values)))
