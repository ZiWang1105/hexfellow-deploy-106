"""Nominal-model gravity, independent of simulator and hardware transport."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import pinocchio as pin

from .mit_command import (
    MitCommand as MitCommand,
)
from .mit_command import (
    MitServoConfig as MitServoConfig,
)
from .mit_command import (
    _vector,
)
from .mit_command import (
    mit_torque as mit_torque,
)


class ModelGravity:
    """Gravity from a fixed nominal Pinocchio model and measured joint state.

    The model includes configured tool/payload inertia, never a simulator's
    hidden current load. Gravity defaults to physical world -Z; rotated robot
    mounting is represented by the model's kinematic transforms.
    """

    def __init__(self, model: pin.Model) -> None:
        self.model = model.copy()
        self.data = self.model.createData()

    def evaluate(self, measured_q: Sequence[float]) -> np.ndarray:
        q = _vector(measured_q, self.model.nq, "gravity configuration")
        result = np.asarray(pin.computeGeneralizedGravity(self.model, self.data, q)).copy()
        if not np.all(np.isfinite(result)):
            raise ValueError("nonfinite model gravity")
        return result

@dataclass(frozen=True)
class GravityPayload:
    """Declared load COM in a named model frame; never a sensed scene object."""

    frame: str
    mass_kg: float = 0.0
    com_m: tuple[float, float, float] = (0.0, 0.0, 0.0)

    def __post_init__(self):
        if not isinstance(self.frame, str) or not self.frame:
            raise ValueError("payload requires a named frame")
        if not np.isfinite(self.mass_kg) or self.mass_kg < 0:
            raise ValueError("payload mass must be finite and nonnegative")
        object.__setattr__(self, "com_m", tuple(_vector(self.com_m, 3, "payload COM")))


@dataclass(frozen=True)
class GravityCorrection:
    """Identified per-arm-joint gravity correction: scale * nominal + offset.

    A data-driven model correction (link masses/COM differ from the URDF),
    fitted to measured static MIT equilibria; not a declared payload.
    """

    scale: tuple[float, ...]
    offset_nm: tuple[float, ...]
    source: str = ""

    def __post_init__(self):
        scale = _vector(self.scale, 14, "gravity correction scale")
        offset = _vector(self.offset_nm, 14, "gravity correction offset")
        if np.any(np.abs(scale) > 3.0) or np.any(np.abs(offset) > 3.0):
            raise ValueError("gravity correction exceeds its plausibility bounds")
        object.__setattr__(self, "scale", tuple(map(float, scale)))
        object.__setattr__(self, "offset_nm", tuple(map(float, offset)))
        object.__setattr__(self, "source", str(self.source))

    def apply(self, arm_gravity):
        return np.asarray(self.scale) * np.asarray(arm_gravity) + np.asarray(self.offset_nm)


@dataclass(frozen=True)
class DynamicFeedforward:
    """Optional MIT friction and inertia feedforward from the solved reference.

    Per-joint values (J1..J7) apply to both arms. The caller clips the result
    to a fraction of the remaining peak-torque headroom; it never relaxes the
    planning bound or the publisher's torque check. Simulation-identified.
    """

    coulomb_nm: tuple[float, ...]
    viscous_nm_s: tuple[float, ...]
    armature_kg_m2: tuple[float, ...]
    inertia_scale: float = 0.0
    coulomb_velocity_epsilon_rad_s: float = 0.02
    headroom_fraction: float = 0.5
    source: str = ""

    def __post_init__(self):
        for name in ("coulomb_nm", "viscous_nm_s", "armature_kg_m2"):
            value = _vector(getattr(self, name), 7, name)
            if np.any(value < 0) or np.any(value > 10):
                raise ValueError(f"{name} must be in [0, 10]")
            object.__setattr__(self, name, tuple(map(float, value)))
        for name, lower, upper in (("inertia_scale", 0.0, 3.0), ("coulomb_velocity_epsilon_rad_s", 1e-3, 1.0),
                                   ("headroom_fraction", 0.0, 1.0)):
            value = float(getattr(self, name))
            if not np.isfinite(value) or not lower <= value <= upper:
                raise ValueError(f"{name} must be in [{lower}, {upper}]")
            object.__setattr__(self, name, value)
        object.__setattr__(self, "source", str(self.source))

    def evaluate(self, gravity_model, measured_q, reference_velocity, reference_acceleration):
        velocity = _vector(reference_velocity, 14, "reference velocity")
        acceleration = _vector(reference_acceleration, 14, "reference acceleration")
        result = (np.tile(self.coulomb_nm, 2) * np.tanh(velocity / self.coulomb_velocity_epsilon_rad_s)
                  + np.tile(self.viscous_nm_s, 2) * velocity
                  + np.tile(self.armature_kg_m2, 2) * acceleration)
        if self.inertia_scale:
            result += self.inertia_scale * (gravity_model.arm_mass_matrix(measured_q) @ acceleration)
        return result


class JointStateGravity(ModelGravity):
    """Full nominal gravity with explicit auxiliary-joint measurements.

    The controlled model can lock gripper joints for IK; gravity retains their
    actual configuration. Omitted auxiliary feedback uses the declared nominal
    configuration, intended only for fixed-gripper/offline callers.
    """

    def __init__(self, full_model, controlled_model, auxiliary_names, nominal_q,
                 payloads=(), correction=None, arm_joint_names=()):
        super().__init__(full_model)
        self.correction = correction
        self.arm_v = [controlled_model.joints[controlled_model.getJointId(name)].idx_v
                      for name in arm_joint_names]
        if correction is not None and len(self.arm_v) != 14:
            raise ValueError("gravity correction requires the fourteen arm joint names")
        self.nominal_q = _vector(nominal_q, full_model.nq, "nominal gravity state")
        self.controlled_nq = controlled_model.nq
        self.control_q = []
        self.full_q = []
        self.full_v = []
        for joint_id in range(1, controlled_model.njoints):
            source = controlled_model.joints[joint_id]
            name = controlled_model.names[joint_id]
            target_id = self.model.getJointId(name)
            if target_id >= self.model.njoints:
                raise ValueError(f"gravity model missing joint {name}")
            target = self.model.joints[target_id]
            if (source.nq, source.nv) != (target.nq, target.nv):
                raise ValueError(f"gravity joint layout mismatch: {name}")
            self.control_q.extend(range(source.idx_q, source.idx_q + source.nq))
            self.full_q.extend(range(target.idx_q, target.idx_q + target.nq))
            self.full_v.extend(range(target.idx_v, target.idx_v + target.nv))
        self.auxiliary_names = tuple(auxiliary_names)
        self.auxiliary_q = []
        for name in self.auxiliary_names:
            joint_id = self.model.getJointId(name)
            if joint_id >= self.model.njoints or self.model.joints[joint_id].nq != 1:
                raise ValueError(f"invalid auxiliary gravity joint: {name}")
            self.auxiliary_q.append(self.model.joints[joint_id].idx_q)
        for payload in payloads:
            frame_id = self.model.getFrameId(payload.frame)
            if frame_id >= self.model.nframes:
                raise ValueError(f"unknown payload frame: {payload.frame}")
            frame = self.model.frames[frame_id]
            joint_id = frame.parentJoint if hasattr(frame, "parentJoint") else frame.parent
            placement = frame.placement * pin.SE3(np.eye(3), np.asarray(payload.com_m))
            self.model.appendBodyToJoint(
                joint_id, pin.Inertia(payload.mass_kg, np.zeros(3), np.zeros((3, 3))), placement
            )
        self.data = self.model.createData()

    def evaluate(self, measured_q, auxiliary_positions=None):
        controlled = _vector(measured_q, self.controlled_nq, "controlled gravity state")
        q = self.nominal_q.copy()
        q[self.full_q] = controlled[self.control_q]
        if auxiliary_positions is not None:
            q[self.auxiliary_q] = _vector(
                auxiliary_positions, len(self.auxiliary_names), "auxiliary gravity state"
            )
        result = super().evaluate(q)[self.full_v]
        if self.correction is not None:
            result[self.arm_v] = self.correction.apply(result[self.arm_v])
        return result

    def arm_mass_matrix(self, measured_q):
        """Rigid-body joint-space inertia of the fourteen arm joints (nominal model)."""
        if len(self.arm_v) != 14:
            raise ValueError("arm mass matrix requires the fourteen arm joint names")
        controlled = _vector(measured_q, self.controlled_nq, "controlled inertia state")
        q = self.nominal_q.copy()
        q[self.full_q] = controlled[self.control_q]
        matrix = pin.crba(self.model, self.data, q)
        matrix = np.triu(matrix) + np.triu(matrix, 1).T
        arm = [self.full_v[i] for i in self.arm_v]
        return matrix[np.ix_(arm, arm)]
