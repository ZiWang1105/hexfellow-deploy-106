"""Binary gripper commands and measured endpoint states, without aperture curves."""
import math


from hexfellow_deploy.gripper_contract import validate_gripper


def gripper_state(measured, closed, opened):
    if not all(math.isfinite(x) for x in (measured, closed, opened)) or closed == opened:
        raise ValueError('Invalid measured gripper position/endpoints')
    tolerance = abs(opened - closed) * .025
    if abs(measured - opened) <= tolerance:
        return 'open'
    if abs(measured - closed) <= tolerance:
        return 'closed'
    return 'between'


class BinaryGripperServo:
    """Device motion stays smooth; the requested state is always binary or HOLD."""
    def __init__(self, closed, opened, measured, travel_time_s=1.):
        if not all(math.isfinite(x) for x in (closed, opened, measured, travel_time_s)) or closed == opened or travel_time_s <= 0:
            raise ValueError('Invalid gripper endpoints/travel time')
        self.closed, self.opened, self.reference = closed, opened, measured
        self.speed = abs(opened - closed) / travel_time_s

    def step(self, command, dt):
        command = validate_gripper(command, allow_hold=True)
        if not math.isfinite(dt) or dt <= 0:
            raise ValueError('Invalid gripper servo period')
        if command is not None:
            target = self.opened if command else self.closed
            delta = max(-self.speed * dt, min(self.speed * dt, target - self.reference))
            self.reference += delta
        return self.reference
