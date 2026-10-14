"""Portable MIT command semantics and model-based gravity feedforward.

No SDK, simulator state, network, or hidden plant parameters are used here.
Torque saturation models a finite actuator; it is not a claim that Hexfellow
exposes a programmable total-torque limit in its MIT packet.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np

from .hardware_capabilities import load_hardware_capabilities


def _vector(values: Sequence[float], size: int, label: str) -> np.ndarray:
    result = np.array(values, dtype=float, copy=True)
    if result.shape != (size,) or not np.all(np.isfinite(result)):
        raise ValueError(f"{label} must contain {size} finite values")
    result.setflags(write=False)
    return result


@dataclass(frozen=True)
class MitServoConfig:
    """Joint-side gains; torque bounds are explicit actuator-model assumptions."""

    kp: tuple[float, ...] = field(default_factory=lambda: load_hardware_capabilities().kp)
    kd: tuple[float, ...] = field(default_factory=lambda: load_hardware_capabilities().kd)
    # Conservative legacy default, not a manufacturer instantaneous ceiling.
    # Reviewed profiles supply their own bound up to the joint-output peak rating.
    torque_limit_nm: tuple[float, ...] = field(
        default_factory=lambda: load_hardware_capabilities().continuous_torque_nm)

    def __post_init__(self) -> None:
        for label in ("kp", "kd", "torque_limit_nm"):
            values = _vector(getattr(self, label), 14, label)
            if np.any(values < 0.0) or (label != "kd" and np.any(values == 0.0)):
                raise ValueError(f"invalid MIT {label}")
            object.__setattr__(self, label, tuple(float(value) for value in values))


@dataclass(frozen=True)
class MitCommand:
    """A transport-independent, immutable joint-side MIT command snapshot."""

    joint_names: tuple[str, ...]
    position: np.ndarray
    velocity: np.ndarray
    torque_ff: np.ndarray
    kp: np.ndarray
    kd: np.ndarray
    timestamp_s: float

    def __post_init__(self) -> None:
        names = tuple(self.joint_names)
        if (not names or any(not isinstance(name, str) or not name for name in names)
                or len(set(names)) != len(names)):
            raise ValueError("MIT joint names must be nonempty and unique")
        object.__setattr__(self, "joint_names", names)
        for label in ("position", "velocity", "torque_ff", "kp", "kd"):
            object.__setattr__(self, label, _vector(getattr(self, label), len(names), label))
        if np.any(self.kp < 0.0) or np.any(self.kd < 0.0):
            raise ValueError("MIT gains must be nonnegative")
        if not np.isfinite(self.timestamp_s) or self.timestamp_s < 0.0:
            raise ValueError("MIT timestamp must be finite and nonnegative")

    def select(self, joint_names: Sequence[str]) -> MitCommand:
        """Select an explicitly ordered arm without sharing mutable history."""
        names = tuple(joint_names)
        indices = [self.joint_names.index(name) for name in names]
        return MitCommand(
            names, self.position[indices], self.velocity[indices], self.torque_ff[indices],
            self.kp[indices], self.kd[indices], self.timestamp_s,
        )

    def motor_input(self) -> np.ndarray:
        """Affine input for a local PD motor, not a joint-position command."""
        with np.errstate(over="raise", invalid="raise"):
            try:
                result = self.kp * self.position + self.kd * self.velocity + self.torque_ff
            except FloatingPointError as error:
                raise ValueError("MIT motor input overflowed") from error
        if not np.all(np.isfinite(result)):
            raise ValueError("MIT motor input must be finite")
        return result


def mit_torque(
    command: MitCommand,
    measured_position: Sequence[float],
    measured_velocity: Sequence[float],
    torque_limit_nm: Sequence[float],
) -> tuple[np.ndarray, np.ndarray]:
    """Evaluate joint PD plus feedforward, saturating the *combined* effort.

    This pure function is also the reference for the simulator's motor model.
    The second result is the unsaturated request, for saturation diagnostics.
    """
    size = len(command.joint_names)
    q = _vector(measured_position, size, "measured position")
    v = _vector(measured_velocity, size, "measured velocity")
    bounds = _vector(torque_limit_nm, size, "torque limit")
    if np.any(bounds <= 0.0):
        raise ValueError("torque limits must be positive")
    requested = command.kp * (command.position - q) + command.kd * (command.velocity - v)
    requested += command.torque_ff
    if not np.all(np.isfinite(requested)):
        raise ValueError("MIT effort calculation overflowed")
    return np.clip(requested, -bounds, bounds), requested
