"""Visible, bounded onsite motions; shared by trajectories, prompts and reports."""

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class Motion:
    amplitude: float
    unit: str
    explanation: str
    leg_seconds: float = 4.0
    dwell_seconds: float = 2.0

    @property
    def duration(self):
        return 2 * (self.leg_seconds + self.dwell_seconds)

    def prompt(self):
        return (self.explanation.format(amplitude=abs(self.amplitude))
                + f"; move out over {self.leg_seconds:g} s, hold for {self.dwell_seconds:g} s, "
                + f"return over {self.leg_seconds:g} s, hold for {self.dwell_seconds:g} s.")

    def report(self):
        return {**asdict(self), "explanation": self.explanation.format(amplitude=abs(self.amplitude)),
                "duration_seconds": self.duration, "initial_hold_seconds": 3.0}


MOTIONS = {
    "left-nudge": Motion(-10.0, "deg", "Rotate left-arm joint 2 by {amplitude:g} degrees in the negative direction; hold the other arm"),
    "right-nudge": Motion(-10.0, "deg", "Rotate right-arm joint 2 by {amplitude:g} degrees in the negative direction; hold the other arm"),
    "lift-move": Motion(3.0, "cm", "Move the lift upward by {amplitude:g} cm; hold both arms and base"),
    "base-x": Motion(10.0, "cm", "Move the base {amplitude:g} cm along its initial X direction; hold remaining joints"),
    "base-y": Motion(10.0, "cm", "Move the base {amplitude:g} cm along its initial Y direction; hold remaining joints"),
    "base-yaw": Motion(10.0, "deg", "Rotate the base {amplitude:g} degrees in positive yaw; hold remaining joints"),
}
