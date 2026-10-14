from dataclasses import dataclass
from typing import List

import numpy as np


@dataclass
class Vec3:
    """A 3D vector."""

    x: float
    y: float
    z: float

    def to_numpy(self) -> np.ndarray:
        """Convert to numpy array."""
        return np.array([self.x, self.y, self.z])

    @classmethod
    def from_numpy(cls, arr: np.ndarray) -> "Vec3":
        """Create from numpy array."""
        return cls(arr[0], arr[1], arr[2])


@dataclass
class Quat:
    """A quaternion."""

    w: float
    x: float
    y: float
    z: float

    def to_numpy(self) -> np.ndarray:
        """Convert to numpy array."""
        return np.array([self.w, self.x, self.y, self.z])

    @classmethod
    def from_numpy(cls, arr: np.ndarray) -> "Quat":
        """Create from numpy array."""
        return cls(arr[0], arr[1], arr[2], arr[3])


@dataclass
class Vec2:
    """A 2D vector."""

    x: float
    y: float

    def to_numpy(self) -> np.ndarray:
        """Convert to numpy array."""
        return np.array([self.x, self.y])

    @classmethod
    def from_numpy(cls, arr: np.ndarray) -> "Vec2":
        """Create from numpy array."""
        return cls(arr[0], arr[1])


@dataclass
class Transform:
    """A transform."""

    position: Vec3
    orientation: Quat

    def to_numpy(self) -> np.ndarray:
        """Convert to numpy array."""
        return np.concatenate([self.position.to_numpy(), self.orientation.to_numpy()])

    @classmethod
    def from_numpy(cls, arr: np.ndarray) -> "Transform":
        """Create from numpy array."""
        return cls(Vec3.from_numpy(arr[:3]), Quat.from_numpy(arr[3:]))


@dataclass
class JointPDConfig:
    """Joint PD configuration."""

    joint_names: List[str]
    kp: List[float]
    kd: List[float]

    def validate(self) -> None:
        """Validate the configuration."""
        assert (
            len(self.joint_names) == len(self.kp) == len(self.kd)
        ), "Joint names, kp, and kd must have the same length"
