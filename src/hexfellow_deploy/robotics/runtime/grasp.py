"""Contact-validated grasp stabilization for parallel-jaw manipulation."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import mujoco
import numpy as np
import pinocchio as pin

from ..contract import ARM_ORDER
from ..control.ik import MaverX4PinkController, set_mocap_pose, site_pose
from ..control.robot import GRASP_CAPTURE_HALF_EXTENTS, GRASP_CENTER_IN_TCP


def has_opposing_jaw_contacts(touching_body_ids: set[int], jaw_body_ids: set[int]) -> bool:
    """Return true only when both calibrated opposing jaw heads touch an object."""
    return len(jaw_body_ids) == 2 and jaw_body_ids.issubset(touching_body_ids)


def _opposing_contact(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    task: Any,
    side: str,
    object_id: str,
) -> bool:
    side_geoms = task.gripper_geom_ids_by_side[side]
    object_geoms = set(task.object_geom_ids[object_id])
    touching_bodies: set[int] = set()
    for contact_index in range(data.ncon):
        contact = data.contact[contact_index]
        geom1, geom2 = int(contact.geom1), int(contact.geom2)
        if geom1 in object_geoms and geom2 in side_geoms:
            touching_bodies.add(int(model.geom_bodyid[geom2]))
        elif geom2 in object_geoms and geom1 in side_geoms:
            touching_bodies.add(int(model.geom_bodyid[geom1]))
    return has_opposing_jaw_contacts(touching_bodies, task.gripper_jaw_body_ids_by_side[side])


def object_inside_grasp_capture(
    data: mujoco.MjData,
    task: Any,
    side: str,
    object_id: str,
    tool_transform: np.ndarray,
) -> bool:
    """Reject contacts on the outside/back of both fingers.

    Opposing contacts remain the authoritative physical signal.  This guard
    only checks that the object's inertial centre is near the calibrated pinch
    volume, preventing a carrier from activating on a large object wedged
    against the backs of both jaws.
    """

    object_body_id = task.object_body_ids[object_id]
    object_world = np.asarray(data.xipos[object_body_id], dtype=np.float64)
    local = tool_transform[:3, :3].T @ (object_world - tool_transform[:3, 3])
    offset = local - np.asarray(GRASP_CENTER_IN_TCP, dtype=np.float64)
    return bool(np.all(np.abs(offset) <= np.asarray(GRASP_CAPTURE_HALF_EXTENTS, dtype=np.float64)))


class GraspContactTracker:
    """Detect physically held objects independently of carrier assistance."""

    def __init__(self, *, activation_steps: int = 3, release_steps: int = 3) -> None:
        if activation_steps < 1 or release_steps < 1:
            raise ValueError("grasp contact activation/release steps must be positive")
        self.activation_steps = int(activation_steps)
        self.release_steps = int(release_steps)
        self.reset()

    def reset(self) -> None:
        self.active: dict[str, str | None] = {side: None for side in ARM_ORDER}
        self._contact_streak: dict[tuple[str, str], int] = {}
        self._loss_streak: dict[str, int] = {side: 0 for side in ARM_ORDER}

    def _clear_side(self, side: str) -> None:
        self.active[side] = None
        self._loss_streak[side] = 0
        for key in tuple(self._contact_streak):
            if key[0] == side:
                del self._contact_streak[key]

    def update(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        task: Any,
        gripper_targets: Mapping[str, float],
        tool_transforms: Mapping[str, np.ndarray],
    ) -> dict[str, str | None]:
        held_elsewhere = {value for value in self.active.values() if value is not None}
        for side in ARM_ORDER:
            if gripper_targets[side] >= 0.09:
                self._clear_side(side)
                continue
            current = self.active[side]
            if current is not None:
                valid = _opposing_contact(model, data, task, side, current) and object_inside_grasp_capture(
                    data, task, side, current, tool_transforms[side]
                )
                self._loss_streak[side] = 0 if valid else self._loss_streak[side] + 1
                if self._loss_streak[side] >= self.release_steps:
                    held_elsewhere.discard(current)
                    self._clear_side(side)
                continue

            candidates = []
            for object_id in task.object_body_ids:
                key = (side, object_id)
                valid = (
                    object_id not in held_elsewhere
                    and _opposing_contact(model, data, task, side, object_id)
                    and object_inside_grasp_capture(data, task, side, object_id, tool_transforms[side])
                )
                self._contact_streak[key] = self._contact_streak.get(key, 0) + 1 if valid else 0
                if self._contact_streak[key] >= self.activation_steps:
                    candidates.append(object_id)
            if candidates:
                selected = sorted(candidates)[0]
                self.active[side] = selected
                held_elsewhere.add(selected)
                self._loss_streak[side] = 0
                for key in tuple(self._contact_streak):
                    if key[0] == side:
                        del self._contact_streak[key]
        return dict(self.active)


class GraspAssist:
    """Own the one-way carrier state used to stabilize sustained grasps."""

    def __init__(self, *, enabled: bool = True, kinematic: bool = False) -> None:
        self.enabled = bool(enabled)
        self.kinematic = bool(kinematic)
        self.reset()

    def reset(self) -> None:
        self.activations = 0
        self.active: dict[str, str | None] = {side: None for side in ARM_ORDER}
        self._relatives: dict[str, np.ndarray | None] = {side: None for side in ARM_ORDER}
        self._contact_streak: dict[tuple[str, str], int] = {}
        self._collision_state: dict[str, dict[int, tuple[int, int]]] = {side: {} for side in ARM_ORDER}

    def _clear_contact_streaks(self, side: str) -> None:
        """Forget contact history whenever a hand cannot continue a grasp."""
        for key in tuple(self._contact_streak):
            if key[0] == side:
                del self._contact_streak[key]

    def _set_kinematic_collision_mask(
        self,
        model: mujoco.MjModel,
        task: Any,
        side: str,
        *,
        enabled: bool,
    ) -> None:
        if not self.kinematic:
            return
        if enabled:
            state = self._collision_state[side]
            for geom_id in task.gripper_geom_ids_by_side[side]:
                state.setdefault(
                    geom_id,
                    (int(model.geom_contype[geom_id]), int(model.geom_conaffinity[geom_id])),
                )
                model.geom_contype[geom_id] = 0
                model.geom_conaffinity[geom_id] = 0
        else:
            for geom_id, (contype, conaffinity) in self._collision_state[side].items():
                model.geom_contype[geom_id] = contype
                model.geom_conaffinity[geom_id] = conaffinity
            self._collision_state[side].clear()

    @staticmethod
    def _equality_id(model: mujoco.MjModel, side: str, object_id: str) -> int:
        name = f"grasp_assist_{side}_{object_id}"
        equality_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_EQUALITY, name)
        if equality_id < 0:
            raise RuntimeError(f"scene is missing grasp-assist equality {name!r}")
        return equality_id

    @staticmethod
    def _as_pin(transform: np.ndarray) -> pin.SE3:
        return pin.SE3(transform[:3, :3], transform[:3, 3])

    def update_carriers(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        tool_transforms: Mapping[str, np.ndarray],
    ) -> None:
        if not self.enabled:
            return
        for side, object_id in self.active.items():
            relative = self._relatives[side]
            if object_id is None or relative is None:
                continue
            target = tool_transforms[side] @ relative
            set_mocap_pose(model, data, f"{side}_grasp_carrier", self._as_pin(target))
            if self.kinematic:
                body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, object_id)
                joint_id = int(model.body_jntadr[body_id])
                if joint_id < 0 or model.jnt_type[joint_id] != mujoco.mjtJoint.mjJNT_FREE:
                    raise RuntimeError(f"assisted object {object_id!r} does not have a free joint")
                qpos_address = int(model.jnt_qposadr[joint_id])
                dof_address = int(model.jnt_dofadr[joint_id])
                quaternion_xyzw = pin.Quaternion(target[:3, :3]).coeffs()
                data.qpos[qpos_address : qpos_address + 3] = target[:3, 3]
                data.qpos[qpos_address + 3 : qpos_address + 7] = (
                    quaternion_xyzw[3],
                    quaternion_xyzw[0],
                    quaternion_xyzw[1],
                    quaternion_xyzw[2],
                )
                data.qvel[dof_address : dof_address + 6] = 0.0

    def update(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        controller: MaverX4PinkController,
        task: Any,
        gripper_targets: Mapping[str, float],
        tool_transforms: Mapping[str, np.ndarray],
        validated_grasps: Mapping[str, str | None] | None = None,
    ) -> bool:
        """Release opened hands and activate carriers after three valid contacts."""
        if not self.enabled:
            return False
        for side in ARM_ORDER:
            active_object = self.active[side]
            if active_object is not None and gripper_targets[side] >= 0.09:
                equality_id = self._equality_id(model, side, active_object)
                data.eq_active[equality_id] = 0
                self._set_kinematic_collision_mask(model, task, side, enabled=False)
                self.active[side] = None
                self._relatives[side] = None
                self._clear_contact_streaks(side)

        held = {name for name in self.active.values() if name is not None}
        for side in ARM_ORDER:
            if self.active[side] is not None or gripper_targets[side] > 0.085:
                self._clear_contact_streaks(side)
                continue
            for object_id in task.object_geom_ids:
                key = (side, object_id)
                if object_id in held:
                    self._contact_streak[key] = 0
                    continue
                if validated_grasps is None:
                    valid = _opposing_contact(model, data, task, side, object_id)
                    self._contact_streak[key] = self._contact_streak.get(key, 0) + 1 if valid else 0
                    if self._contact_streak[key] < 3:
                        continue
                elif validated_grasps.get(side) != object_id:
                    continue

                object_body_id = task.object_body_ids[object_id]
                tcp = tool_transforms[side]
                object_transform = np.eye(4)
                object_transform[:3, :3] = data.xmat[object_body_id].reshape(3, 3)
                object_transform[:3, 3] = data.xpos[object_body_id]
                relative = np.eye(4)
                relative[:3, :3] = tcp[:3, :3].T @ object_transform[:3, :3]
                relative[:3, 3] = tcp[:3, :3].T @ (object_transform[:3, 3] - tcp[:3, 3])
                set_mocap_pose(model, data, f"{side}_grasp_carrier", self._as_pin(object_transform))
                mujoco.mj_forward(model, data)
                equality_id = self._equality_id(model, side, object_id)
                data.eq_active[equality_id] = 0 if self.kinematic else 1
                self._set_kinematic_collision_mask(model, task, side, enabled=True)
                self.active[side] = object_id
                self._relatives[side] = relative
                self._clear_contact_streaks(side)
                held.add(object_id)
                self.activations += 1
                current_targets = {arm: site_pose(model, data, f"{arm}_tcp") for arm in ARM_ORDER}
                controller.reset(controller.measured_configuration(data), current_targets)
                return True
        return False
