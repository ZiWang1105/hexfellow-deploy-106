"""Hash-bound posture preference, independent of measured-state initialization."""

import hashlib
from dataclasses import dataclass

import numpy as np
import yaml

from ..paths import PROJECT_ROOT


@dataclass(frozen=True)
class HumanPosturePolicy:
    arm_position: tuple[float, ...]
    blend_s: float
    reference_sha256: str

    def __post_init__(self):
        if (len(self.arm_position) != 14 or not np.isfinite(self.arm_position).all()
                or not np.isfinite(self.blend_s) or self.blend_s <= 0):
            raise ValueError("invalid posture reference or blend duration")
        object.__setattr__(self, "arm_position", tuple(float(x) for x in self.arm_position))

    @classmethod
    def parse(cls, document):
        if document is None:
            return None
        if document.get("mode") != "human_blend":
            raise ValueError("unsupported posture policy")
        reference = document["reference"]
        path = (PROJECT_ROOT / reference["path"]).resolve()
        path.relative_to(PROJECT_ROOT)
        raw = path.read_bytes()
        digest = hashlib.sha256(raw).hexdigest()
        if reference["sha256"] != digest:
            raise ValueError("posture reference hash mismatch")
        pose = yaml.safe_load(raw)
        arms = tuple(pose["left_arm_rad"] + pose["right_arm_rad"])
        blend = float(document["blend_s"])
        return cls(arms, blend, digest)


class PostureTransition:
    """Advance preference only while tracking, with zero endpoint derivatives."""

    def __init__(self, initial, target, duration):
        self.initial, self.target = initial.copy(), target.copy()
        self.duration, self.elapsed = duration, 0.0

    def advance(self, dt, *, active):
        if active:
            self.elapsed = min(self.duration, self.elapsed + dt)
        u = self.elapsed / self.duration
        blend = u**3 * (10.0 + u * (-15.0 + 6.0 * u))
        return self.initial + blend * (self.target - self.initial)
