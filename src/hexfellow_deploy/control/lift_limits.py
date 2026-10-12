"""Shared application lift ceiling in robot-model meters, not SDK coordinates."""
import math

LIFT_MAX_M = 0.400
# Agent sessions hold the lift braked rather than commanding lift motion. The
# installed Theta reports 422 mm travel; do not confuse that physical domain
# with the historical 400 mm keyboard/moving-reference ceiling.
FIXED_LIFT_MAX_M = 0.422
# The nominal travel is rounded to millimeters; SDK endpoints are integer
# encoder counts. Allow at most half a count (capped at one micrometer) when
# comparing measured feedback against that nominal model bound. This does not
# enlarge the SDK range or the allowed command/reference range.
FIXED_LIFT_RANGE_ROUNDING_MAX_M = 1e-6


def validate_lift_target(height, context='Lift target'):
    if not math.isfinite(height) or height > LIFT_MAX_M:
        raise ValueError(f'{context} exceeds the {LIFT_MAX_M:.3f} m model lift ceiling: {height}')


def cap_keyboard_lift_target(mapping, target_sdk, reported_range):
    """Cap a position request after converting the reversed SDK coordinate."""
    height = mapping.lift_position(target_sdk, reported_range)
    return mapping.lift_command(min(height, LIFT_MAX_M), reported_range)
