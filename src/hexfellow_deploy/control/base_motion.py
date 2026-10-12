"""Static, task-independent limits for one body-frame base pulse."""
from dataclasses import asdict, dataclass, fields
import math
from pathlib import Path

import yaml

from ..robot.upstream import PROJECT_ROOT


DEFAULT_BASE_CONFIG = PROJECT_ROOT / "configs/controller/base_motion.yaml"


@dataclass(frozen=True)
class BaseMotionConfig:
    max_speed_m_s: float
    pulse_s: float
    ramp_s: float
    settle_s: float
    approach_distance_m: float = .45
    distance_tolerance_m: float = .03
    max_yaw_rate_rad_s: float = .25
    yaw_tolerance_rad: float = .02

    def __post_init__(self):
        for name, value in asdict(self).items():
            if (isinstance(value, bool) or not isinstance(value, (int, float))
                    or not math.isfinite(value) or value <= 0):
                raise ValueError(f"{name} must be finite and positive")
        if self.yaw_tolerance_rad >= math.pi:
            raise ValueError("Yaw tolerance must be smaller than pi radians")
        if 2 * self.ramp_s > self.pulse_s:
            raise ValueError("Base pulse must accommodate both acceleration and deceleration ramps")
        if self.approach_distance_m + self.distance_tolerance_m >= 4:
            raise ValueError("Approach distance must be inside the navigation command envelope")

    @classmethod
    def load(cls, path=DEFAULT_BASE_CONFIG):
        settings = yaml.safe_load(Path(path).read_text())
        if (not isinstance(settings, dict) or set(settings) - {f.name for f in fields(cls)}
                or not {'max_speed_m_s', 'pulse_s', 'ramp_s', 'settle_s'} <= set(settings)):
            raise ValueError("Unexpected base motion configuration")
        return cls(**settings)

    def to_dict(self):
        return asdict(self)

    def prompt_context(self):
        return ("\nFixed base motion settings: "
                f"maximum translation speed {self.max_speed_m_s:g} m/s; "
                f"maximum yaw rate {self.max_yaw_rate_rad_s:g} rad/s; "
                f"yaw request tolerance {self.yaw_tolerance_rad:g} rad; "
                f"pulse duration {self.pulse_s:g} s including "
                f"{self.ramp_s:g} s acceleration and deceleration ramps each; "
                f"then {self.settle_s:g} s of zero base command before reobservation. "
                f"Object approach distance {self.approach_distance_m:g} m horizontally from base origin; "
                f"distance tolerance {self.distance_tolerance_m:g} m. "
                "These are command limits, not measurements of chassis travel.\n")
