"""Calibrated DAS Controller V3 aperture/actuator conversion."""

from __future__ import annotations

import numpy as np

# Bench calibration copied from the DAS V3 hardware descriptor.  Aperture is
# the physical distance between opposing jaw faces, not a linear motor angle.
CALIBRATED_ANGLES_RAD = np.array(
    [0.0, 0.11563, 0.23125, 0.34688, 0.46250, 0.57812, 0.69375, 0.80938, 0.925],
    dtype=np.float64,
)
CALIBRATED_WIDTHS_M = np.array(
    [0.104905, 0.094466, 0.082927, 0.070443, 0.057181, 0.043319, 0.029040, 0.014536, 0.0],
    dtype=np.float64,
)
WIRE_MAX_WIDTH_M = 0.105
PHYSICAL_MAX_WIDTH_M = float(CALIBRATED_WIDTHS_M[0])


def aperture_to_angle(aperture_m: float) -> float:
    """Convert a policy aperture to the calibrated primary-joint angle."""
    aperture = float(aperture_m)
    if not np.isfinite(aperture) or not 0.0 <= aperture <= WIRE_MAX_WIDTH_M:
        raise ValueError(f"gripper aperture must be in [0, {WIRE_MAX_WIDTH_M:g}] m")
    physical = min(aperture, PHYSICAL_MAX_WIDTH_M)
    return float(
        np.interp(
            physical,
            CALIBRATED_WIDTHS_M[::-1],
            CALIBRATED_ANGLES_RAD[::-1],
        )
    )


def angle_to_aperture(angle_rad: float) -> float:
    """Convert the measured primary-joint angle to physical jaw aperture."""
    angle = float(angle_rad)
    if not np.isfinite(angle):
        raise ValueError("gripper angle must be finite")
    clipped = float(np.clip(angle, CALIBRATED_ANGLES_RAD[0], CALIBRATED_ANGLES_RAD[-1]))
    return float(np.interp(clipped, CALIBRATED_ANGLES_RAD, CALIBRATED_WIDTHS_M))
