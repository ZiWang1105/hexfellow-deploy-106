"""Clutch-relative mapping from Quest controller grip poses to robot TCP targets."""

from __future__ import annotations

import dataclasses
from collections.abc import Mapping

import numpy as np
import pinocchio as pin

from ..contract import ARM_ORDER
from ..control.robot import GRIPPER_APERTURE_MAX
from ..runtime.action_buffer import WorldTarget
from .protocol import SESSION_FLAG_VISIBLE, ControllerPose, QuestPoseSample

QUEST_TO_ROBOT = np.asarray(
    (
        (0.0, 0.0, -1.0),
        (-1.0, 0.0, 0.0),
        (0.0, 1.0, 0.0),
    ),
    dtype=np.float64,
)


def quaternion_xyzw_to_matrix(quaternion: np.ndarray) -> np.ndarray:
    value = np.asarray(quaternion, dtype=np.float64)
    if value.shape != (4,) or not np.all(np.isfinite(value)):
        raise ValueError("quaternion must be a finite xyzw vector")
    norm = float(np.linalg.norm(value))
    if norm <= 1e-12:
        raise ValueError("quaternion norm must be positive")
    x, y, z, w = value / norm
    return np.asarray(
        (
            (1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)),
            (2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)),
            (2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)),
        ),
        dtype=np.float64,
    )


def rotation_distance(left: np.ndarray, right: np.ndarray) -> float:
    relative = np.asarray(left, dtype=np.float64).T @ np.asarray(right, dtype=np.float64)
    cosine = float(np.clip((np.trace(relative) - 1.0) / 2.0, -1.0, 1.0))
    return float(np.arccos(cosine))


@dataclasses.dataclass
class _SideState:
    active: bool = False
    require_release: bool = False
    controller_anchor_position: np.ndarray | None = None
    controller_anchor_rotation: np.ndarray | None = None
    robot_anchor: np.ndarray | None = None
    held_transform: np.ndarray | None = None
    held_width_m: float = GRIPPER_APERTURE_MAX
    previous_position: np.ndarray | None = None
    previous_rotation: np.ndarray | None = None
    previous_xr_time_ms: float | None = None
    previous_sequence: int | None = None


@dataclasses.dataclass(frozen=True)
class QuestMappingResult:
    targets: Mapping[str, WorldTarget]
    active: Mapping[str, bool]
    faults: Mapping[str, str | None]


class QuestTargetMapper:
    """Map independent squeeze-clutched controllers into continuous TCP goals."""

    def __init__(
        self,
        *,
        translation_scale: float = 1.0,
        rotation_scale: float = 1.0,
        engage_threshold: float = 0.8,
        release_threshold: float = 0.2,
        stale_after_ms: float = 60.0,
        maximum_linear_velocity_m_s: float = 5.0,
        maximum_angular_velocity_rad_s: float = 20.0,
    ) -> None:
        if not np.isfinite(translation_scale) or translation_scale <= 0.0:
            raise ValueError("translation scale must be finite and positive")
        if not np.isfinite(rotation_scale) or rotation_scale <= 0.0:
            raise ValueError("rotation scale must be finite and positive")
        if not 0.0 <= release_threshold < engage_threshold <= 1.0:
            raise ValueError("clutch thresholds must satisfy 0 <= release < engage <= 1")
        if stale_after_ms <= 0.0:
            raise ValueError("stale threshold must be positive")
        self.translation_scale = float(translation_scale)
        self.rotation_scale = float(rotation_scale)
        self.engage_threshold = float(engage_threshold)
        self.release_threshold = float(release_threshold)
        self.stale_after_ms = float(stale_after_ms)
        self.maximum_linear_velocity_m_s = float(maximum_linear_velocity_m_s)
        self.maximum_angular_velocity_rad_s = float(maximum_angular_velocity_rad_s)
        self._states = {side: _SideState() for side in ARM_ORDER}
        self._reference_epoch: int | None = None

    def reset(
        self,
        measured_tcp: Mapping[str, np.ndarray],
        gripper_widths: Mapping[str, float],
        *,
        require_release: bool = True,
    ) -> None:
        self._reference_epoch = None
        for side in ARM_ORDER:
            transform = np.asarray(measured_tcp[side], dtype=np.float64)
            self._states[side] = _SideState(
                require_release=require_release,
                held_transform=transform.copy(),
                held_width_m=float(gripper_widths[side]),
            )

    def force_release(
        self,
        measured_tcp: Mapping[str, np.ndarray] | None = None,
        *,
        require_release: bool = True,
    ) -> None:
        for side, state in self._states.items():
            state.active = False
            state.require_release = require_release
            state.controller_anchor_position = None
            state.controller_anchor_rotation = None
            state.robot_anchor = None
            if measured_tcp is not None:
                state.held_transform = np.asarray(measured_tcp[side], dtype=np.float64).copy()

    @staticmethod
    def _tracking_fault(pose: ControllerPose) -> str | None:
        if not pose.present:
            return "pose_missing"
        if pose.emulated_position:
            return "emulated_position"
        values = np.concatenate((pose.position_m, pose.orientation_xyzw))
        if not np.all(np.isfinite(values)):
            return "nonfinite_pose"
        return None

    def _motion_fault(self, state: _SideState, sample: QuestPoseSample, pose: ControllerPose) -> str | None:
        if state.previous_sequence == sample.sequence:
            return None
        rotation = quaternion_xyzw_to_matrix(pose.orientation_xyzw)
        fault = None
        if state.previous_xr_time_ms is not None:
            dt = (sample.xr_time_ms - state.previous_xr_time_ms) / 1000.0
            if dt <= 0.0:
                fault = "nonmonotonic_timestamp"
            else:
                assert state.previous_position is not None and state.previous_rotation is not None
                linear = float(np.linalg.norm(pose.position_m - state.previous_position) / dt)
                angular = rotation_distance(state.previous_rotation, rotation) / dt
                if linear > self.maximum_linear_velocity_m_s:
                    fault = "translation_discontinuity"
                elif angular > self.maximum_angular_velocity_rad_s:
                    fault = "rotation_discontinuity"
        state.previous_position = pose.position_m.copy()
        state.previous_rotation = rotation
        state.previous_xr_time_ms = sample.xr_time_ms
        state.previous_sequence = sample.sequence
        return fault

    def update(
        self,
        sample: QuestPoseSample | None,
        *,
        now_ns: int,
        measured_tcp: Mapping[str, np.ndarray],
        gripper_widths: Mapping[str, float],
    ) -> QuestMappingResult:
        if any(self._states[side].held_transform is None for side in ARM_ORDER):
            self.reset(measured_tcp, gripper_widths)
        faults: dict[str, str | None] = {side: None for side in ARM_ORDER}
        if sample is None:
            for side in ARM_ORDER:
                self._states[side].active = False
                self._states[side].require_release = True
                faults[side] = "no_sample"
            return self._result(faults)
        if sample.age_ms(now_ns) > self.stale_after_ms:
            for side in ARM_ORDER:
                faults[side] = "stale_sample"
            # Match the established G1 gateway behavior: a short same-session
            # sample gap freezes the last target but retains the logical
            # clutch and its anchor. The next fresh pose catches up through the
            # IK target limiter. Explicit disconnects are handled separately
            # by force_release(), while tracking/reference loss still releases
            # below.
            return self._result(faults)
        if not sample.session_flags & SESSION_FLAG_VISIBLE:
            for side in ARM_ORDER:
                self._states[side].active = False
                self._states[side].require_release = True
                faults[side] = "session_not_visible"
            return self._result(faults)
        if self._reference_epoch is None:
            self._reference_epoch = sample.reference_epoch
        elif sample.reference_epoch != self._reference_epoch:
            self.reset(measured_tcp, gripper_widths, require_release=True)
            self._reference_epoch = sample.reference_epoch
            return self._result({side: "reference_space_reset" for side in ARM_ORDER})

        for side in ARM_ORDER:
            state = self._states[side]
            pose = sample.controllers.get(side, ControllerPose.missing())
            fault = self._tracking_fault(pose)
            if fault is None:
                fault = self._motion_fault(state, sample, pose)
            if fault is not None:
                state.active = False
                state.require_release = True
                state.controller_anchor_position = None
                state.controller_anchor_rotation = None
                state.robot_anchor = None
                faults[side] = fault
                continue
            if pose.squeeze <= self.release_threshold:
                state.active = False
                state.require_release = False
            elif not state.active and not state.require_release and pose.squeeze >= self.engage_threshold:
                state.active = True
                state.controller_anchor_position = pose.position_m.copy()
                state.controller_anchor_rotation = quaternion_xyzw_to_matrix(pose.orientation_xyzw)
                state.robot_anchor = np.asarray(measured_tcp[side], dtype=np.float64).copy()
                state.held_transform = state.robot_anchor.copy()
                state.held_width_m = float(gripper_widths[side])
            if not state.active:
                continue
            assert state.controller_anchor_position is not None
            assert state.controller_anchor_rotation is not None
            assert state.robot_anchor is not None
            controller_rotation = quaternion_xyzw_to_matrix(pose.orientation_xyzw)
            delta_position = (
                self.translation_scale * QUEST_TO_ROBOT @ (pose.position_m - state.controller_anchor_position)
            )
            mapped_delta_rotation = (
                QUEST_TO_ROBOT @ (controller_rotation @ state.controller_anchor_rotation.T) @ QUEST_TO_ROBOT.T
            )
            delta_rotation = pin.exp3(
                self.rotation_scale * pin.log3(mapped_delta_rotation)
            )
            target = state.robot_anchor.copy()
            target[:3, 3] += delta_position
            target[:3, :3] = delta_rotation @ state.robot_anchor[:3, :3]
            state.held_transform = target
            state.held_width_m = GRIPPER_APERTURE_MAX * (1.0 - pose.trigger)
        return self._result(faults)

    def _result(self, faults: Mapping[str, str | None]) -> QuestMappingResult:
        return QuestMappingResult(
            targets={
                side: WorldTarget(
                    np.asarray(self._states[side].held_transform, dtype=np.float64).copy(),
                    self._states[side].held_width_m,
                )
                for side in ARM_ORDER
            },
            active={side: self._states[side].active for side in ARM_ORDER},
            faults=dict(faults),
        )
