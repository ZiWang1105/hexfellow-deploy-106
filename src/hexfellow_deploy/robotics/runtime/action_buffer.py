"""Timestamped world-target trajectories with explicit chunk handoff blending."""

from __future__ import annotations

import dataclasses
from collections.abc import Mapping, Sequence
from typing import Protocol

import numpy as np

from ..contract import (
    ARM_ORDER,
    ArmTarget,
    ContractError,
    matrix_to_quaternion_wxyz,
    quaternion_wxyz_to_matrix,
)

HANDOFF_GUARD_NS = 20_000_000
HANDOFF_BLEND_NS = 200_000_000
MIN_BLEND_NS = 40_000_000


@dataclasses.dataclass(frozen=True)
class WorldTarget:
    transform: np.ndarray
    width_m: float
    body_twist: np.ndarray | None = None
    body_acceleration: np.ndarray | None = None


@dataclasses.dataclass(frozen=True)
class TimedWorldTarget:
    timestamp_ns: int
    arms: Mapping[str, WorldTarget]


@dataclasses.dataclass(frozen=True)
class InstallResult:
    accepted: bool
    stale_prefix_rows: int
    mode: str
    handoff_timestamp_ns: int | None
    old_new_overlap_ms: float
    actual_blend_ms: float
    raw_translation_jump_m: float
    raw_rotation_jump_rad: float


class Trajectory(Protocol):
    @property
    def start_ns(self) -> int: ...

    @property
    def end_ns(self) -> int: ...

    def sample(self, timestamp_ns: int) -> dict[str, WorldTarget]: ...


def _validated_target(target: WorldTarget, side: str) -> WorldTarget:
    transform = np.asarray(target.transform, dtype=np.float64)
    if transform.shape != (4, 4) or not np.all(np.isfinite(transform)):
        raise ContractError(f"{side} world target must be a finite 4x4 transform")
    if not np.allclose(transform[3], (0.0, 0.0, 0.0, 1.0), atol=1e-8):
        raise ContractError(f"{side} world target has invalid homogeneous row")
    rotation = transform[:3, :3]
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-6) or not np.isclose(
        np.linalg.det(rotation), 1.0, atol=1e-6
    ):
        raise ContractError(f"{side} world target rotation is invalid")
    if not np.isfinite(target.width_m):
        raise ContractError(f"{side} gripper target is non-finite")
    twist = np.zeros(6) if target.body_twist is None else np.asarray(target.body_twist, dtype=np.float64)
    acceleration = (
        np.zeros(6)
        if target.body_acceleration is None
        else np.asarray(target.body_acceleration, dtype=np.float64)
    )
    if twist.shape != (6,) or not np.all(np.isfinite(twist)):
        raise ContractError(f"{side} target body twist must be a finite 6-vector")
    if acceleration.shape != (6,) or not np.all(np.isfinite(acceleration)):
        raise ContractError(f"{side} target body acceleration must be a finite 6-vector")
    return WorldTarget(transform.copy(), float(target.width_m), twist.copy(), acceleration.copy())


def quintic_blend(fraction: float, duration_s: float) -> tuple[float, float, float]:
    """Return C2 quintic position, velocity, and acceleration blend factors."""

    if not np.isfinite(duration_s) or duration_s <= 0.0:
        raise ContractError("quintic blend duration must be finite and positive")
    u = float(np.clip(fraction, 0.0, 1.0))
    position = u**3 * (10.0 + u * (-15.0 + 6.0 * u))
    velocity = 30.0 * u**2 * (1.0 - u) ** 2 / duration_s
    acceleration = 60.0 * u * (1.0 - u) * (1.0 - 2.0 * u) / duration_s**2
    if u <= 0.0 or u >= 1.0:
        velocity = 0.0
        acceleration = 0.0
    return position, velocity, acceleration


def _slerp_rotation(left: np.ndarray, right: np.ndarray, fraction: float) -> np.ndarray:
    first = matrix_to_quaternion_wxyz(left)
    second = matrix_to_quaternion_wxyz(right)
    dot = float(np.dot(first, second))
    if dot < 0.0:
        second = -second
        dot = -dot
    dot = float(np.clip(dot, 0.0, 1.0))
    if dot > 1.0 - 1e-8:
        quaternion = first + fraction * (second - first)
    else:
        angle = float(np.arccos(dot))
        sine = float(np.sin(angle))
        quaternion = (
            np.sin((1.0 - fraction) * angle) / sine * first
            + np.sin(fraction * angle) / sine * second
        )
    quaternion /= np.linalg.norm(quaternion)
    return quaternion_wxyz_to_matrix(quaternion)


def interpolate_targets(
    left: Mapping[str, WorldTarget],
    right: Mapping[str, WorldTarget],
    fraction: float,
    *,
    fraction_rate: float = 0.0,
    fraction_acceleration: float = 0.0,
) -> dict[str, WorldTarget]:
    fraction = float(np.clip(fraction, 0.0, 1.0))
    result: dict[str, WorldTarget] = {}
    for side in ARM_ORDER:
        first = _validated_target(left[side], side)
        second = _validated_target(right[side], side)
        first_twist = np.asarray(first.body_twist)
        second_twist = np.asarray(second.body_twist)
        first_acceleration = np.asarray(first.body_acceleration)
        second_acceleration = np.asarray(second.body_acceleration)
        transform = np.eye(4)
        translation_delta_world = second.transform[:3, 3] - first.transform[:3, 3]
        transform[:3, 3] = first.transform[:3, 3] + fraction * translation_delta_world
        transform[:3, :3] = _slerp_rotation(first.transform[:3, :3], second.transform[:3, :3], fraction)
        delta_rotation = first.transform[:3, :3].T @ second.transform[:3, :3]
        delta_quaternion = matrix_to_quaternion_wxyz(delta_rotation)
        if delta_quaternion[0] < 0.0:
            delta_quaternion = -delta_quaternion
        sine_half = float(np.linalg.norm(delta_quaternion[1:]))
        if sine_half < 1e-12:
            rotation_vector = np.zeros(3)
        else:
            angle = 2.0 * float(np.arctan2(sine_half, delta_quaternion[0]))
            rotation_vector = angle * delta_quaternion[1:] / sine_half
        body_delta = np.concatenate(
            (first.transform[:3, :3].T @ translation_delta_world, rotation_vector)
        )
        twist = (
            (1.0 - fraction) * first_twist
            + fraction * second_twist
            + fraction_rate * body_delta
        )
        acceleration = (
            (1.0 - fraction) * first_acceleration
            + fraction * second_acceleration
            + 2.0 * fraction_rate * (second_twist - first_twist)
            + fraction_acceleration * body_delta
        )
        result[side] = WorldTarget(
            transform=transform,
            width_m=(1.0 - fraction) * first.width_m + fraction * second.width_m,
            body_twist=twist,
            body_acceleration=acceleration,
        )
    return result


class SampledTrajectory:
    def __init__(self, samples: Sequence[TimedWorldTarget]) -> None:
        if not samples:
            raise ContractError("a trajectory requires at least one sample")
        timestamps = [int(sample.timestamp_ns) for sample in samples]
        if any(right <= left for left, right in zip(timestamps, timestamps[1:], strict=False)):
            raise ContractError("trajectory timestamps must be strictly increasing")
        self._samples = tuple(
            TimedWorldTarget(
                timestamp_ns=timestamp,
                arms={side: _validated_target(sample.arms[side], side) for side in ARM_ORDER},
            )
            for timestamp, sample in zip(timestamps, samples, strict=True)
        )
        self._timestamps = np.asarray(timestamps, dtype=np.int64)

    @property
    def start_ns(self) -> int:
        return int(self._timestamps[0])

    @property
    def end_ns(self) -> int:
        return int(self._timestamps[-1])

    def sample(self, timestamp_ns: int) -> dict[str, WorldTarget]:
        timestamp_ns = int(timestamp_ns)
        if timestamp_ns <= self.start_ns:
            return dict(self._samples[0].arms)
        if timestamp_ns >= self.end_ns:
            return dict(self._samples[-1].arms)
        right_index = int(np.searchsorted(self._timestamps, timestamp_ns, side="right"))
        left = self._samples[right_index - 1]
        right = self._samples[right_index]
        duration_ns = right.timestamp_ns - left.timestamp_ns
        fraction = (timestamp_ns - left.timestamp_ns) / duration_ns
        blend, blend_rate, blend_acceleration = quintic_blend(fraction, duration_ns / 1e9)
        return interpolate_targets(
            left.arms,
            right.arms,
            blend,
            fraction_rate=blend_rate,
            fraction_acceleration=blend_acceleration,
        )


class HoldTrajectory:
    def __init__(self, targets: Mapping[str, WorldTarget], timestamp_ns: int) -> None:
        self._targets = {side: _validated_target(targets[side], side) for side in ARM_ORDER}
        self._timestamp_ns = int(timestamp_ns)

    @property
    def start_ns(self) -> int:
        return self._timestamp_ns

    @property
    def end_ns(self) -> int:
        return np.iinfo(np.int64).max

    def sample(self, timestamp_ns: int) -> dict[str, WorldTarget]:
        del timestamp_ns
        return dict(self._targets)


class BlendedTrajectory:
    def __init__(
        self,
        old: Trajectory,
        new: Trajectory,
        *,
        start_ns: int,
        duration_ns: int,
        hold_old: bool,
    ) -> None:
        self.old = old
        self.new = new
        self.blend_start_ns = int(start_ns)
        self.blend_end_ns = int(start_ns + duration_ns)
        self.hold_old = bool(hold_old)
        self._held = old.sample(min(self.blend_start_ns, old.end_ns))

    @property
    def start_ns(self) -> int:
        return min(self.old.start_ns, self.new.start_ns)

    @property
    def end_ns(self) -> int:
        return self.new.end_ns

    def sample(self, timestamp_ns: int) -> dict[str, WorldTarget]:
        timestamp_ns = int(timestamp_ns)
        if timestamp_ns < self.blend_start_ns:
            return self.old.sample(timestamp_ns)
        if timestamp_ns >= self.blend_end_ns:
            return self.new.sample(timestamp_ns)
        old = self._held if self.hold_old else self.old.sample(timestamp_ns)
        new = self.new.sample(timestamp_ns)
        u = (timestamp_ns - self.blend_start_ns) / (self.blend_end_ns - self.blend_start_ns)
        alpha, alpha_rate, alpha_acceleration = quintic_blend(
            u, (self.blend_end_ns - self.blend_start_ns) / 1e9
        )
        return interpolate_targets(
            old,
            new,
            alpha,
            fraction_rate=alpha_rate,
            fraction_acceleration=alpha_acceleration,
        )


def _largest_jump(
    old: Mapping[str, WorldTarget], new: Mapping[str, WorldTarget]
) -> tuple[float, float]:
    translation = 0.0
    rotation = 0.0
    for side in ARM_ORDER:
        translation = max(
            translation,
            float(np.linalg.norm(old[side].transform[:3, 3] - new[side].transform[:3, 3])),
        )
        delta = old[side].transform[:3, :3].T @ new[side].transform[:3, :3]
        cosine = float(np.clip((np.trace(delta) - 1.0) / 2.0, -1.0, 1.0))
        rotation = max(rotation, float(np.arccos(cosine)))
    return translation, rotation


class OverlapActionBuffer:
    """Own the active trajectory and blend every accepted replacement."""

    def __init__(self) -> None:
        self._active: Trajectory | None = None

    @property
    def active(self) -> bool:
        return self._active is not None

    @property
    def end_ns(self) -> int | None:
        return None if self._active is None else self._active.end_ns

    def set_hold(self, targets: Mapping[str, WorldTarget], timestamp_ns: int) -> None:
        self._active = HoldTrajectory(targets, timestamp_ns)

    def sample(self, timestamp_ns: int) -> dict[str, WorldTarget]:
        if self._active is None:
            raise ContractError("action buffer has no active or hold trajectory")
        return self._active.sample(timestamp_ns)

    def install(
        self,
        samples: Sequence[TimedWorldTarget],
        *,
        current_timestamp_ns: int,
        guard_ns: int = HANDOFF_GUARD_NS,
        blend_ns: int = HANDOFF_BLEND_NS,
        minimum_blend_ns: int = MIN_BLEND_NS,
    ) -> InstallResult:
        cutoff = int(current_timestamp_ns + guard_ns)
        stale_rows = sum(int(sample.timestamp_ns) <= cutoff for sample in samples)
        fresh = tuple(sample for sample in samples if int(sample.timestamp_ns) > cutoff)
        if not fresh:
            return InstallResult(False, stale_rows, "all_stale", None, 0.0, 0.0, 0.0, 0.0)
        new = SampledTrajectory(fresh)
        if self._active is None:
            self._active = new
            return InstallResult(True, stale_rows, "initial", new.start_ns, 0.0, 0.0, 0.0, 0.0)
        old = self._active
        handoff = new.start_ns
        new_coverage = new.end_ns - handoff
        overlap = max(0, min(old.end_ns, new.end_ns) - handoff)
        old_at_handoff = old.sample(handoff)
        new_at_handoff = new.sample(handoff)
        translation_jump, rotation_jump = _largest_jump(old_at_handoff, new_at_handoff)
        if overlap >= minimum_blend_ns:
            duration = min(blend_ns, overlap)
            mode = "overlap_blend"
            hold_old = False
        elif new_coverage >= minimum_blend_ns:
            duration = min(blend_ns, new_coverage)
            mode = "hold_bridge"
            hold_old = True
        else:
            return InstallResult(
                False,
                stale_rows,
                "insufficient_coverage",
                handoff,
                overlap / 1e6,
                0.0,
                translation_jump,
                rotation_jump,
            )
        self._active = BlendedTrajectory(
            old,
            new,
            start_ns=handoff,
            duration_ns=duration,
            hold_old=hold_old,
        )
        return InstallResult(
            True,
            stale_rows,
            mode,
            handoff,
            overlap / 1e6,
            duration / 1e6,
            translation_jump,
            rotation_jump,
        )


def decoded_commands_to_samples(
    timestamps_ns: Sequence[int],
    commands: Sequence[Mapping[str, ArmTarget]],
) -> tuple[TimedWorldTarget, ...]:
    if len(timestamps_ns) != len(commands):
        raise ContractError("target timestamps and decoded commands have different lengths")
    return tuple(
        TimedWorldTarget(
            int(timestamp),
            {
                side: WorldTarget(command[side].transform.copy(), command[side].width_m)
                for side in ARM_ORDER
            },
        )
        for timestamp, command in zip(timestamps_ns, commands, strict=True)
    )
