"""Identified simulation plant corrections; simulation only, never a hardware setting.

The calibration adds an equivalent wrist load (a point mass in the TCP frame,
representing identified model mass/COM error, not a declared attachment) and
per-joint friction, damping and reflected rotor inertia to the MuJoCo plant.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

SIDES = ("left", "right")


@dataclass(frozen=True)
class PlantCalibration:
    wrist_load_kg: dict
    wrist_load_com_tcp_m: dict
    frictionloss_nm: np.ndarray
    damping_nm_s: np.ndarray
    armature_kg_m2: np.ndarray
    velocity_feedback_tau_s: float | None = None
    gravity_scale: np.ndarray | None = None
    gravity_offset_nm: np.ndarray | None = None
    source: str = ""
    sha256: str = ""
    command_delay_s: float | None = None

    @classmethod
    def load(cls, path):
        payload = Path(path).read_bytes()
        document = json.loads(payload)
        if document.get("schema_version") != 1:
            raise ValueError("unsupported plant calibration schema")
        load = document["wrist_load"]
        joints = document["arm_joints"]
        vectors = {}
        for key in ("frictionloss_nm", "damping_nm_s", "armature_kg_m2"):
            value = np.asarray(joints[key], dtype=float)
            if value.shape != (7,) or not np.all(np.isfinite(value)) or np.any(value < 0):
                raise ValueError(f"{key} must contain seven finite nonnegative values")
            vectors[key] = value
        masses, coms = {}, {}
        for side in SIDES:
            mass = float(load[side]["mass_kg"])
            com = np.asarray(load[side]["com_tcp_m"], dtype=float)
            if not np.isfinite(mass) or mass < 0 or com.shape != (3,) or not np.all(np.isfinite(com)):
                raise ValueError(f"invalid {side} wrist load")
            masses[side], coms[side] = mass, com
        tau = document.get("velocity_feedback_tau_s")
        if tau is not None and (not np.isfinite(tau) or tau < 0):
            raise ValueError("velocity feedback time constant must be nonnegative")
        delay = document.get("command_delay_s")
        if delay is not None and (not np.isfinite(delay) or delay < 0):
            raise ValueError("command delay must be finite and nonnegative")
        gravity = document.get("arm_gravity")
        scale = offset = None
        if gravity is not None:
            scale = np.asarray(gravity["scale"], dtype=float)
            offset = np.asarray(gravity["offset_nm"], dtype=float)
            if scale.shape != (14,) or offset.shape != (14,) or not np.all(np.isfinite([scale, offset])):
                raise ValueError("arm gravity calibration requires fourteen finite scales and offsets")
        return cls(masses, coms, vectors["frictionloss_nm"], vectors["damping_nm_s"],
                   vectors["armature_kg_m2"], None if tau is None else float(tau), scale, offset,
                   str(Path(path)), hashlib.sha256(payload).hexdigest(),
                   None if delay is None else float(delay))

    def endpoint_options(self, overrides=None):
        """Identified total command delay, split once; explicit options win.

        The identification clock measures a total effective delay, not an extra
        delay on top of the endpoint's default 6 ms dispatch. This is simulation
        metadata only and never changes hardware transport or safety deadlines.
        """
        options = dict(overrides or {})
        if self.command_delay_s is not None:
            total = self.command_delay_s
            if 'dispatch_delay_s' not in options:
                options['dispatch_delay_s'] = (max(0., total - options['actuator_delay_s'])
                                               if 'actuator_delay_s' in options else min(.006, total))
            options.setdefault('actuator_delay_s', max(0., total - options['dispatch_delay_s']))
        if self.velocity_feedback_tau_s is not None:
            options.setdefault('velocity_feedback_tau_s', self.velocity_feedback_tau_s)
        return options

    def apply(self, plant):
        """Modify the plant's MuJoCo model in place and refresh derived constants."""
        mujoco, model = plant.mujoco, plant.model
        dofs = np.asarray(plant.core._mj_qvel_indices)[-14:]
        model.dof_frictionloss[dofs] = np.tile(self.frictionloss_nm, 2)
        model.dof_damping[dofs] = np.tile(self.damping_nm_s, 2)
        model.dof_armature[dofs] = np.tile(self.armature_kg_m2, 2)
        for side in SIDES:
            mass = self.wrist_load_kg[side]
            if mass <= 0:
                continue
            site = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, f"{side}_tcp")
            if site < 0:
                raise ValueError(f"plant has no {side}_tcp site")
            body = int(model.site_bodyid[site])
            rotation = np.empty(9)
            mujoco.mju_quat2Mat(rotation, model.site_quat[site])
            point = model.site_pos[site] + rotation.reshape(3, 3) @ self.wrist_load_com_tcp_m[side]
            _add_point_mass(mujoco, model, body, mass, point)
        # mj_setConst uses its data argument as scratch at qpos0; keep the plant state.
        mujoco.mj_setConst(model, mujoco.MjData(model))
        mujoco.mj_forward(model, plant.data)
        if self.gravity_scale is not None:
            self._install_gravity_load(plant)

    def _install_gravity_load(self, plant):
        """Make the plant's arm gravity equal scale * nominal + offset.

        MuJoCo already applies the nominal (URDF) gravity; the difference acts
        as a configuration-dependent external joint load.
        """
        from ..control.ik_core import ARM_JOINTS
        from ..control.mit import JointStateGravity
        core = plant.core
        nominal = JointStateGravity(core._gravity_source_model, core.model, core.auxiliary_joints,
                                    core._gravity_nominal_q, (), arm_joint_names=ARM_JOINTS)
        dofs = np.asarray(core._mj_qvel_indices)[-14:]
        extra_scale, offset = self.gravity_scale - 1.0, self.gravity_offset_nm

        def load(target):
            q = target.core.measured_configuration(target.data)
            arm = nominal.evaluate(q)[nominal.arm_v]
            target.data.qfrc_applied[dofs] = -(extra_scale * arm + offset)
        plant.substep_hook = load

    def describe(self):
        return dict(source=self.source, sha256=self.sha256,
                    wrist_load_kg=self.wrist_load_kg,
                    wrist_load_com_tcp_m={k: v.tolist() for k, v in self.wrist_load_com_tcp_m.items()},
                    frictionloss_nm=self.frictionloss_nm.tolist(), damping_nm_s=self.damping_nm_s.tolist(),
                    armature_kg_m2=self.armature_kg_m2.tolist(),
                    velocity_feedback_tau_s=self.velocity_feedback_tau_s,
                    command_delay_s=self.command_delay_s,
                    gravity_scale=None if self.gravity_scale is None else self.gravity_scale.tolist(),
                    gravity_offset_nm=None if self.gravity_offset_nm is None else self.gravity_offset_nm.tolist(),
                    interpretation="identified simulation correction; not a hardware attachment or firmware model")


def _add_point_mass(mujoco, model, body, mass, point):
    """Combine a point mass (body-frame position) into a body's inertial properties."""
    m0 = float(model.body_mass[body])
    c0 = np.array(model.body_ipos[body], dtype=float)
    frame = np.empty(9)
    mujoco.mju_quat2Mat(frame, model.body_iquat[body])
    frame = frame.reshape(3, 3)
    inertia = frame @ np.diag(model.body_inertia[body]) @ frame.T  # about c0, body axes
    total = m0 + mass
    com = (m0 * c0 + mass * point) / total

    def shift(m, offset):
        return m * (offset @ offset * np.eye(3) - np.outer(offset, offset))

    combined = inertia + shift(m0, c0 - com) + shift(mass, point - com)
    values, vectors = np.linalg.eigh(combined)
    if np.linalg.det(vectors) < 0:
        vectors[:, 0] *= -1
    quaternion = np.empty(4)
    mujoco.mju_mat2Quat(quaternion, vectors.reshape(-1))
    model.body_mass[body] = total
    model.body_ipos[body] = com
    model.body_iquat[body] = quaternion
    model.body_inertia[body] = np.maximum(values, 1e-9)
