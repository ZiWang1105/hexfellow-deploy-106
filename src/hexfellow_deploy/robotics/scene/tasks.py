"""Privileged task evaluators for tabletop rigid-object rollouts."""

from __future__ import annotations

import dataclasses
import itertools
from collections.abc import Mapping, Sequence

import mujoco
import numpy as np

from .config import LiftSuccessConfig, SuccessConfig
from .constants import (
    ORGANIZE_GOAL_SITE,
    TRASHCAN_GOAL_SITE,
    TRASHCAN_POSITION,
)

CONTAINMENT_NUMERICAL_TOLERANCE_M = 1e-4
ARM_ORDER = ("left", "right")


@dataclasses.dataclass(frozen=True)
class ObjectProgress:
    object_id: str
    fully_inside: bool
    released: bool
    stable: bool
    stable_seconds: float
    linear_velocity_m_s: float
    angular_velocity_rad_s: float
    containment_margin_m: float


@dataclasses.dataclass(frozen=True)
class TaskStatus:
    success: bool
    terminated: bool
    truncated: bool
    reason: str | None
    spawned_count: int
    required_count: int
    inside_count: int
    stable_count: int
    elapsed_seconds: float
    objects: tuple[ObjectProgress, ...]

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)


@dataclasses.dataclass(frozen=True)
class LiftHandProgress:
    side: str
    object_id: str | None
    height_delta_m: float
    held: bool
    lifted: bool


@dataclasses.dataclass(frozen=True)
class LiftObjectProgress:
    object_id: str
    held_by: str | None
    baseline_height_m: float
    current_height_m: float
    height_delta_m: float
    lifted: bool


@dataclasses.dataclass(frozen=True)
class LiftTaskStatus:
    success: bool
    terminated: bool
    truncated: bool
    reason: str | None
    spawned_count: int
    required_count: int
    held_count: int
    lifted_count: int
    required_lift_height_m: float
    required_hold_seconds: float
    simultaneous_hold_seconds: float
    elapsed_seconds: float
    hands: dict[str, LiftHandProgress]
    objects: tuple[LiftObjectProgress, ...]

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)


def _descendants(model: mujoco.MjModel, root_body_id: int) -> set[int]:
    result = {root_body_id}
    changed = True
    while changed:
        changed = False
        for body_id in range(1, model.nbody):
            if body_id not in result and int(model.body_parentid[body_id]) in result:
                result.add(body_id)
                changed = True
    return result


class RigidObjectTaskBase:
    """Shared scene indexing and workspace checks for rigid-object tasks."""

    def __init__(
        self,
        model: mujoco.MjModel,
        object_ids: Sequence[str],
        timeout_seconds: float,
    ) -> None:
        self.model = model
        self.object_ids = tuple(object_ids)
        self.timeout_seconds = float(timeout_seconds)
        if not self.object_ids:
            raise ValueError("task requires at least one object ID")
        if len(set(self.object_ids)) != len(self.object_ids):
            raise ValueError("task object IDs must be unique")
        if not np.isfinite(self.timeout_seconds) or self.timeout_seconds <= 0:
            raise ValueError("task timeout must be finite and positive")
        self.object_body_ids = {name: self._named_id(mujoco.mjtObj.mjOBJ_BODY, name) for name in self.object_ids}
        self.object_geom_ids = {
            name: tuple(
                geom_id
                for geom_id in range(model.ngeom)
                if int(model.geom_bodyid[geom_id]) in _descendants(model, body_id)
                and int(model.geom_group[geom_id]) == 3
            )
            for name, body_id in self.object_body_ids.items()
        }
        for name, geom_ids in self.object_geom_ids.items():
            if not geom_ids:
                raise RuntimeError(f"object {name!r} has no collision geometry in group 3")
        gripper_roots = {
            "left": self._named_id(
                mujoco.mjtObj.mjOBJ_BODY,
                "maver_x4_asm_7_urdf_SLDASM_link_7",
            ),
            "right": self._named_id(
                mujoco.mjtObj.mjOBJ_BODY,
                "maver_x4_asm_7_urdf_SLDASM2_link_7",
            ),
        }
        self.gripper_geom_ids_by_side = {
            side: {
                geom_id
                for geom_id in range(model.ngeom)
                if int(model.geom_bodyid[geom_id]) in _descendants(model, root)
            }
            for side, root in gripper_roots.items()
        }
        jaw_names = {
            "left": ("robot_link3", "robot_link4"),
            "right": ("robot2_link3", "robot2_link4"),
        }
        self.gripper_jaw_body_ids_by_side = {
            side: {self._named_id(mujoco.mjtObj.mjOBJ_BODY, name) for name in names}
            for side, names in jaw_names.items()
        }
        self.gripper_geom_ids = set().union(*self.gripper_geom_ids_by_side.values())
        self._started_at = 0.0

    def _named_id(self, object_type: mujoco.mjtObj, name: str) -> int:
        object_id = mujoco.mj_name2id(self.model, object_type, name)
        if object_id < 0:
            raise RuntimeError(f"scene is missing {name!r}")
        return object_id

    def reset(self, data: mujoco.MjData) -> None:
        self._started_at = float(data.time)

    def _lost_objects(self, data: mujoco.MjData) -> tuple[str, ...]:
        lost = []
        for name, body_id in self.object_body_ids.items():
            position = data.xpos[body_id]
            radial = float(np.linalg.norm(position[:2] - np.array((0.75, 0.0))))
            if position[2] < 0.55 or radial > 0.70:
                lost.append(name)
        return tuple(lost)


class StableContainmentTask(RigidObjectTaskBase):
    """Shared released-and-stable containment evaluator for open containers."""

    def __init__(
        self,
        model: mujoco.MjModel,
        object_ids: Sequence[str],
        success: SuccessConfig,
        timeout_seconds: float,
        *,
        goal_site_name: str,
        container_body_name: str,
    ) -> None:
        super().__init__(model, object_ids, timeout_seconds)
        self.config = success
        self.goal_site_id = self._named_id(mujoco.mjtObj.mjOBJ_SITE, goal_site_name)
        self.container_body_name = container_body_name
        self.container_body_id = self._named_id(mujoco.mjtObj.mjOBJ_BODY, container_body_name)
        site_size = self.model.site_size[self.goal_site_id]
        site_type = self.model.site_type[self.goal_site_id]
        if site_type == mujoco.mjtGeom.mjGEOM_BOX:
            horizontal_clearance = float(np.min(site_size[:2]))
        elif site_type == mujoco.mjtGeom.mjGEOM_CYLINDER:
            horizontal_clearance = float(site_size[0])
        else:
            raise RuntimeError(f"unsupported containment site type {site_type}")
        if self.config.inset_m >= horizontal_clearance:
            raise ValueError("containment inset eliminates the container opening")
        self._stable_since: dict[str, float | None] = {name: None for name in self.object_ids}

    def reset(self, data: mujoco.MjData) -> None:
        super().reset(data)
        self._stable_since = {name: None for name in self.object_ids}

    def _containment_margin(self, data: mujoco.MjData, object_id: str) -> float:
        goal_position = data.site_xpos[self.goal_site_id]
        goal_rotation = data.site_xmat[self.goal_site_id].reshape(3, 3)
        half_size = np.asarray(self.model.site_size[self.goal_site_id], dtype=np.float64).copy()
        minimum_margin = np.inf
        lowest_object_point = np.inf
        for geom_id in self.object_geom_ids[object_id]:
            geom_rotation = data.geom_xmat[geom_id].reshape(3, 3)
            if self.model.geom_type[geom_id] == mujoco.mjtGeom.mjGEOM_MESH:
                mesh_id = int(self.model.geom_dataid[geom_id])
                start = int(self.model.mesh_vertadr[mesh_id])
                stop = start + int(self.model.mesh_vertnum[mesh_id])
                support_points = np.asarray(self.model.mesh_vert[start:stop])
            else:
                center = self.model.geom_aabb[geom_id, :3]
                extent = self.model.geom_aabb[geom_id, 3:]
                support_points = center + np.asarray(tuple(itertools.product((-1.0, 1.0), repeat=3))) * extent
            world_points = data.geom_xpos[geom_id] + support_points @ geom_rotation.T
            goal_points = (world_points - goal_position) @ goal_rotation
            site_type = self.model.site_type[self.goal_site_id]
            if site_type == mujoco.mjtGeom.mjGEOM_BOX:
                horizontal_margin = float(np.min(half_size[:2] - self.config.inset_m - np.abs(goal_points[:, :2])))
                vertical_half_size = half_size[2]
            elif site_type == mujoco.mjtGeom.mjGEOM_CYLINDER:
                horizontal_margin = float(
                    half_size[0] - self.config.inset_m - np.max(np.linalg.norm(goal_points[:, :2], axis=1))
                )
                vertical_half_size = half_size[1]
            else:
                raise RuntimeError(f"unsupported containment site type {site_type}")
            minimum_margin = min(minimum_margin, horizontal_margin)
            lowest_object_point = min(lowest_object_point, float(np.min(goal_points[:, 2])))
        floor_margin = lowest_object_point + vertical_half_size
        entered_below_rim_margin = vertical_half_size - lowest_object_point
        return float(min(minimum_margin, floor_margin, entered_below_rim_margin))

    def _released(self, data: mujoco.MjData, object_id: str) -> bool:
        object_geoms = set(self.object_geom_ids[object_id])
        for contact_index in range(data.ncon):
            contact = data.contact[contact_index]
            pair = {int(contact.geom1), int(contact.geom2)}
            if pair & object_geoms and pair & self.gripper_geom_ids:
                return False
        return True

    def _irrecoverable(self, data: mujoco.MjData, required_count: int) -> str | None:
        lost = set(self._lost_objects(data))
        container_position = data.xpos[self.container_body_id]
        if self.container_body_name == "open_box":
            if container_position[2] < 0.55 or np.linalg.norm(container_position[:2] - np.array((0.75, 0.0))) > 0.65:
                return "box_out_of_workspace"
        elif self.container_body_name == "trashcan":
            upright = float(data.xmat[self.container_body_id].reshape(3, 3)[2, 2])
            if (
                container_position[2] < -0.05
                or np.linalg.norm(container_position[:2] - TRASHCAN_POSITION[:2]) > 0.35
                or upright < np.cos(np.deg2rad(45.0))
            ):
                return "trashcan_out_of_workspace"
            # The generic table-workspace check labels a correctly discarded
            # object as lost because the bin is below and beside the tabletop.
            lost = {name for name in lost if self._containment_margin(data, name) < -CONTAINMENT_NUMERICAL_TOLERANCE_M}
        else:
            raise RuntimeError(f"unsupported containment body {self.container_body_name!r}")
        if len(self.object_ids) - len(lost) < required_count:
            return f"insufficient_recoverable_objects:{','.join(sorted(lost))}"
        return None

    def update(
        self,
        data: mujoco.MjData,
        active_grasps: Mapping[str, str | None] | Sequence[str] = (),
    ) -> TaskStatus:
        now = float(data.time)
        held = (
            {name for name in active_grasps.values() if name is not None}
            if isinstance(active_grasps, Mapping)
            else set(active_grasps)
        )
        progress: list[ObjectProgress] = []
        for name in self.object_ids:
            body_id = self.object_body_ids[name]
            velocity = np.asarray(data.cvel[body_id])
            angular = float(np.linalg.norm(velocity[:3]))
            linear = float(np.linalg.norm(velocity[3:]))
            containment_margin = self._containment_margin(data, name)
            inside = containment_margin >= -CONTAINMENT_NUMERICAL_TOLERANCE_M
            released = name not in held and self._released(data, name)
            stable_now = (
                inside
                and released
                and linear <= self.config.max_linear_velocity_m_s
                and angular <= self.config.max_angular_velocity_rad_s
            )
            if stable_now:
                if self._stable_since[name] is None:
                    self._stable_since[name] = now
            else:
                self._stable_since[name] = None
            stable_seconds = 0.0 if self._stable_since[name] is None else now - self._stable_since[name]
            progress.append(
                ObjectProgress(
                    object_id=name,
                    fully_inside=inside,
                    released=released,
                    stable=stable_seconds >= self.config.stable_seconds,
                    stable_seconds=stable_seconds,
                    linear_velocity_m_s=linear,
                    angular_velocity_rad_s=angular,
                    containment_margin_m=containment_margin,
                )
            )
        elapsed = now - self._started_at
        required_count = min(len(progress), self.config.required_objects)
        stable_count = sum(item.stable for item in progress)
        success = stable_count >= required_count
        failure = None if success else self._irrecoverable(data, required_count)
        truncated = not success and failure is None and elapsed >= self.timeout_seconds
        reason = "success" if success else failure or ("timeout" if truncated else None)
        return TaskStatus(
            success=success,
            terminated=success or failure is not None,
            truncated=truncated,
            reason=reason,
            spawned_count=len(progress),
            required_count=required_count,
            inside_count=sum(item.fully_inside for item in progress),
            stable_count=stable_count,
            elapsed_seconds=elapsed,
            objects=tuple(progress),
        )


class OrganizeRigidObjectsTask(StableContainmentTask):
    """Evaluate stable containment in the shallow organizer."""

    def __init__(
        self,
        model: mujoco.MjModel,
        object_ids: Sequence[str],
        success: SuccessConfig,
        timeout_seconds: float,
    ) -> None:
        super().__init__(
            model,
            object_ids,
            success,
            timeout_seconds,
            goal_site_name=ORGANIZE_GOAL_SITE,
            container_body_name="open_box",
        )


class ThrowAwayObjectsTask(StableContainmentTask):
    """Require at least one released object to settle inside the trash can."""

    def __init__(
        self,
        model: mujoco.MjModel,
        object_ids: Sequence[str],
        success: SuccessConfig,
        timeout_seconds: float,
    ) -> None:
        super().__init__(
            model,
            object_ids,
            success,
            timeout_seconds,
            goal_site_name=TRASHCAN_GOAL_SITE,
            container_body_name="trashcan",
        )


class LiftTwoObjectsTask(RigidObjectTaskBase):
    """Require both hands to hold and lift distinct objects simultaneously."""

    def __init__(
        self,
        model: mujoco.MjModel,
        object_ids: Sequence[str],
        success: LiftSuccessConfig,
        timeout_seconds: float,
    ) -> None:
        if len(object_ids) != 2:
            raise ValueError("lift_two_objects requires exactly two object IDs")
        super().__init__(model, object_ids, timeout_seconds)
        self.config = success
        self._baseline_heights = {name: 0.0 for name in self.object_ids}
        self._simultaneous_since: float | None = None
        self._hold_signature: tuple[str, str] | None = None

    def reset(self, data: mujoco.MjData) -> None:
        super().reset(data)
        self._baseline_heights = {name: float(data.xpos[body_id][2]) for name, body_id in self.object_body_ids.items()}
        self._simultaneous_since = None
        self._hold_signature = None

    def update(
        self,
        data: mujoco.MjData,
        active_grasps: Mapping[str, str | None] | Sequence[str] = (),
    ) -> LiftTaskStatus:
        now = float(data.time)
        grasp_map = (
            {side: active_grasps.get(side) for side in ARM_ORDER}
            if isinstance(active_grasps, Mapping)
            else {side: None for side in ARM_ORDER}
        )
        height_deltas = {
            name: float(data.xpos[body_id][2] - self._baseline_heights[name])
            for name, body_id in self.object_body_ids.items()
        }
        left_object = grasp_map["left"]
        right_object = grasp_map["right"]
        distinct_pair = (
            left_object in self.object_body_ids and right_object in self.object_body_ids and left_object != right_object
        )
        signature = (left_object, right_object) if distinct_pair else None
        both_lifted = bool(
            distinct_pair
            and height_deltas[left_object] >= self.config.lift_height_delta_m
            and height_deltas[right_object] >= self.config.lift_height_delta_m
        )
        if both_lifted:
            if self._simultaneous_since is None or signature != self._hold_signature:
                self._simultaneous_since = now
                self._hold_signature = signature
        else:
            self._simultaneous_since = None
            self._hold_signature = None
        simultaneous_seconds = 0.0 if self._simultaneous_since is None else now - self._simultaneous_since
        success = both_lifted and simultaneous_seconds >= self.config.simultaneous_hold_seconds
        hands = {
            side: LiftHandProgress(
                side=side,
                object_id=object_id,
                height_delta_m=(height_deltas[object_id] if object_id in height_deltas else 0.0),
                held=object_id in self.object_body_ids,
                lifted=(object_id in height_deltas and height_deltas[object_id] >= self.config.lift_height_delta_m),
            )
            for side, object_id in grasp_map.items()
        }
        held_by = {object_id: side for side, object_id in grasp_map.items() if object_id in self.object_body_ids}
        objects = tuple(
            LiftObjectProgress(
                object_id=name,
                held_by=held_by.get(name),
                baseline_height_m=self._baseline_heights[name],
                current_height_m=float(data.xpos[self.object_body_ids[name]][2]),
                height_delta_m=height_deltas[name],
                lifted=height_deltas[name] >= self.config.lift_height_delta_m,
            )
            for name in self.object_ids
        )
        lost = () if success else self._lost_objects(data)
        failure = f"object_out_of_workspace:{','.join(lost)}" if lost else None
        elapsed = now - self._started_at
        truncated = not success and failure is None and elapsed >= self.timeout_seconds
        reason = "success" if success else failure or ("timeout" if truncated else None)
        return LiftTaskStatus(
            success=success,
            terminated=success or failure is not None,
            truncated=truncated,
            reason=reason,
            spawned_count=len(self.object_ids),
            required_count=2,
            held_count=sum(progress.held for progress in hands.values()),
            lifted_count=sum(progress.lifted for progress in hands.values()),
            required_lift_height_m=self.config.lift_height_delta_m,
            required_hold_seconds=self.config.simultaneous_hold_seconds,
            simultaneous_hold_seconds=simultaneous_seconds,
            elapsed_seconds=elapsed,
            hands=hands,
            objects=objects,
        )


RuntimeTask = OrganizeRigidObjectsTask | ThrowAwayObjectsTask | LiftTwoObjectsTask


def create_task_evaluator(
    task_name: str,
    model: mujoco.MjModel,
    object_ids: Sequence[str],
    success: SuccessConfig | LiftSuccessConfig,
    timeout_seconds: float,
    *,
    required_objects: int | None = None,
) -> RuntimeTask:
    """Construct the privileged evaluator registered for a task manifest."""
    if task_name == "lift_two_objects":
        if not isinstance(success, LiftSuccessConfig):
            raise TypeError("lift_two_objects requires LiftSuccessConfig")
        return LiftTwoObjectsTask(model, object_ids, success, timeout_seconds)
    if not isinstance(success, SuccessConfig):
        raise TypeError(f"{task_name} requires SuccessConfig")
    if required_objects is not None:
        success = dataclasses.replace(success, required_objects=int(required_objects))
    if task_name == "organize_rigid_objects":
        return OrganizeRigidObjectsTask(model, object_ids, success, timeout_seconds)
    if task_name == "throw_away_objects":
        return ThrowAwayObjectsTask(model, object_ids, success, timeout_seconds)
    raise ValueError(f"no runtime task registered for {task_name!r}")
