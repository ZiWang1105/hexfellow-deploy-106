"""Human-like nominal poses and strict arm null-space posture tasks."""

from __future__ import annotations

import os
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path

import numpy as np
import pinocchio as pin
import yaml
from pink import Configuration
from pink.tasks import Task

from ..paths import CONFIG_DIR

DEFAULT_NULL_POSE_PATH = CONFIG_DIR / "control" / "human_null_pose.yaml"


@dataclass(frozen=True)
class NullPose:
    """Versioned lift and dual-arm nominal posture in controller units."""

    lift_m: float
    left_arm_rad: tuple[float, ...]
    right_arm_rad: tuple[float, ...]
    metadata: dict[str, object] = field(default_factory=dict, compare=False)

    def __post_init__(self) -> None:
        if len(self.left_arm_rad) != 7 or len(self.right_arm_rad) != 7:
            raise ValueError("null pose must contain exactly seven joints per arm")
        values = np.asarray((self.lift_m, *self.left_arm_rad, *self.right_arm_rad), dtype=float)
        if not np.all(np.isfinite(values)):
            raise ValueError("null pose values must all be finite")

    def joint_positions(
        self,
        lift_joint: str,
        left_arm_joints: Sequence[str],
        right_arm_joints: Sequence[str],
    ) -> dict[str, float]:
        if len(left_arm_joints) != 7 or len(right_arm_joints) != 7:
            raise ValueError("Maver X4 null-pose mapping requires seven joints per arm")
        return {
            lift_joint: float(self.lift_m),
            **dict(zip(left_arm_joints, self.left_arm_rad, strict=True)),
            **dict(zip(right_arm_joints, self.right_arm_rad, strict=True)),
        }

    def to_dict(self, *, include_metadata: bool = True) -> dict[str, object]:
        result: dict[str, object] = {
            "schema_version": 1,
            "robot": "maver_x4",
            "lift_m": float(self.lift_m),
            "left_arm_rad": [float(value) for value in self.left_arm_rad],
            "right_arm_rad": [float(value) for value in self.right_arm_rad],
        }
        if include_metadata and self.metadata:
            result["metadata"] = self.metadata
        return result


def _numeric_tuple(value: object, label: str) -> tuple[float, ...]:
    if not isinstance(value, (list, tuple)):
        raise ValueError(f"{label} must be a sequence")
    try:
        return tuple(float(item) for item in value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{label} must contain only numbers") from error


def load_null_pose(path: str | Path = DEFAULT_NULL_POSE_PATH) -> NullPose:
    """Load and structurally validate a null-pose YAML file."""
    path = Path(path).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"null-pose configuration does not exist: {path}")
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, Mapping):
        raise ValueError(f"null-pose configuration must be a mapping: {path}")
    if raw.get("schema_version") != 1:
        raise ValueError(f"unsupported null-pose schema_version in {path}")
    if raw.get("robot") != "maver_x4":
        raise ValueError(f"null-pose robot must be 'maver_x4' in {path}")
    allowed = {"schema_version", "robot", "lift_m", "left_arm_rad", "right_arm_rad", "metadata"}
    extra = set(raw) - allowed
    if extra:
        raise ValueError(f"unexpected null-pose fields in {path}: {sorted(extra)}")
    missing = {"lift_m", "left_arm_rad", "right_arm_rad"} - set(raw)
    if missing:
        raise ValueError(f"missing null-pose fields in {path}: {sorted(missing)}")
    metadata = raw.get("metadata", {})
    if not isinstance(metadata, Mapping):
        raise ValueError(f"null-pose metadata must be a mapping in {path}")
    return NullPose(
        lift_m=float(raw["lift_m"]),
        left_arm_rad=_numeric_tuple(raw["left_arm_rad"], "left_arm_rad"),
        right_arm_rad=_numeric_tuple(raw["right_arm_rad"], "right_arm_rad"),
        metadata=dict(metadata),
    )


def save_null_pose(
    pose: NullPose,
    path: str | Path,
    *,
    include_metadata: bool = True,
) -> Path:
    """Atomically save a candidate or canonical null-pose YAML file."""
    path = Path(path).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    document = yaml.safe_dump(
        pose.to_dict(include_metadata=include_metadata),
        sort_keys=False,
        default_flow_style=False,
    )
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            stream.write(document)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_path, path)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()
    return path


def validate_null_pose_limits(
    pose: NullPose,
    model: pin.Model,
    *,
    lift_joint: str,
    left_arm_joints: Sequence[str],
    right_arm_joints: Sequence[str],
    lift_margin: float,
    arm_margin: float,
) -> tuple[float, dict[str, float]]:
    """Validate planning limits and return normalized per-arm-joint margins."""
    values = pose.joint_positions(lift_joint, left_arm_joints, right_arm_joints)
    normalized: dict[str, float] = {}
    for name, value in values.items():
        joint_id = model.getJointId(name)
        if joint_id == 0:
            raise ValueError(f"robot model is missing null-pose joint {name!r}")
        joint = model.joints[joint_id]
        if joint.nq != 1:
            raise ValueError(f"null-pose joint {name!r} is not one-dimensional")
        lower = float(model.lowerPositionLimit[joint.idx_q])
        upper = float(model.upperPositionLimit[joint.idx_q])
        margin = lift_margin if name == lift_joint else arm_margin
        planning_lower = lower + margin
        planning_upper = upper - margin
        if not planning_lower <= value <= planning_upper:
            raise ValueError(
                f"null-pose joint {name!r}={value:.9g} is outside planning limits "
                f"[{planning_lower:.9g}, {planning_upper:.9g}]"
            )
        if name != lift_joint:
            normalized[name] = min(value - planning_lower, planning_upper - value) / (
                planning_upper - planning_lower
            )
    minimum = min(normalized.values())
    if minimum < 0.05:
        name = min(normalized, key=normalized.__getitem__)
        raise ValueError(
            f"null pose starts too close to an arm limit: {name!r} has only "
            f"{100.0 * minimum:.2f}% normalized planning margin (minimum 5%)"
        )
    return float(minimum), normalized


class ArmNullspacePostureTask(Task):
    """Bias both 7-DoF arms only along their instantaneous TCP null spaces."""

    def __init__(
        self,
        model: pin.Model,
        *,
        frame_names: Mapping[str, str],
        arm_joint_names: Mapping[str, Sequence[str]],
        target_q: np.ndarray,
        cost: float | Sequence[float] | np.ndarray,
        gain: float = 1.0,
        kinematics_cache: ArmKinematicsCache | None = None,
    ) -> None:
        super().__init__(cost=cost, gain=gain, lm_damping=0.0)
        if tuple(frame_names) != ("left", "right") or tuple(arm_joint_names) != ("left", "right"):
            raise ValueError("arm null-space task requires ordered left and right mappings")
        self.model = model
        self.frame_names = dict(frame_names)
        self.velocity_indices: dict[str, np.ndarray] = {}
        for side, names in arm_joint_names.items():
            if len(names) != 7:
                raise ValueError(f"{side} arm null-space task requires seven joints")
            indices: list[int] = []
            for name in names:
                joint_id = model.getJointId(name)
                if joint_id == 0 or model.joints[joint_id].nv != 1:
                    raise ValueError(f"invalid {side} arm joint {name!r}")
                indices.append(int(model.joints[joint_id].idx_v))
            self.velocity_indices[side] = np.asarray(indices, dtype=int)
        self.kinematics_cache = kinematics_cache or ArmKinematicsCache(
            model,
            frame_names=self.frame_names,
            velocity_indices=self.velocity_indices,
            angular_length_scale_m=1.0,
        )
        self.set_target(target_q)

    def set_target(self, target_q: np.ndarray) -> None:
        target_q = np.asarray(target_q, dtype=float)
        if target_q.shape != (self.model.nq,) or not np.all(np.isfinite(target_q)):
            raise ValueError(f"null-space posture target must have shape ({self.model.nq},)")
        self.target_q = target_q.copy()

    def _projectors(self, configuration: Configuration) -> dict[str, np.ndarray]:
        return {
            side: self.kinematics_cache.basic_state(side, configuration).null_projector
            for side in self.frame_names
        }

    def compute_error(self, configuration: Configuration) -> np.ndarray:
        difference = pin.difference(self.model, self.target_q, configuration.q)
        projectors = self._projectors(configuration)
        return np.concatenate(
            tuple(projectors[side] @ difference[self.velocity_indices[side]] for side in ("left", "right"))
        )

    def compute_jacobian(self, configuration: Configuration) -> np.ndarray:
        jacobian = np.zeros((14, self.model.nv), dtype=float)
        projectors = self._projectors(configuration)
        for row, side in ((0, "left"), (7, "right")):
            jacobian[row : row + 7, self.velocity_indices[side]] = projectors[side]
        return jacobian

    def projected_error_norms(self, configuration: Configuration) -> dict[str, float]:
        error = self.compute_error(configuration)
        return {"left": float(np.linalg.norm(error[:7])), "right": float(np.linalg.norm(error[7:]))}

    def nullspace_residuals(self, configuration: Configuration) -> dict[str, float]:
        """Return ||J_arm N|| for diagnostics and regression tests."""
        projectors = self._projectors(configuration)
        return {
            side: float(
                np.linalg.norm(
                    configuration.get_frame_jacobian(self.frame_names[side])[:, self.velocity_indices[side]]
                    @ projectors[side]
                )
            )
            for side in ("left", "right")
        }

    def __repr__(self) -> str:
        return (
            "ArmNullspacePostureTask("
            f"cost={self.cost}, gain={self.gain}, lm_damping={self.lm_damping})"
        )


@dataclass(frozen=True)
class ArmKinematicsState:
    """Shared conditioning state for one arm at one controller configuration."""

    geometric_jacobian: np.ndarray
    scaled_jacobian: np.ndarray
    singular_values: np.ndarray
    rank: int
    null_projector: np.ndarray
    log_manipulability: float
    minimum_singular_value: float
    gradient: np.ndarray | None = None
    projected_gradient: np.ndarray | None = None


class ArmKinematicsCache:
    """Compute arm Jacobian derivatives and SVD exactly once per configuration."""

    def __init__(
        self,
        model: pin.Model,
        *,
        frame_names: Mapping[str, str],
        velocity_indices: Mapping[str, Sequence[int] | np.ndarray],
        angular_length_scale_m: float,
        rank_rtol: float = 1e-6,
        gram_regularization: float = 1e-8,
    ) -> None:
        if angular_length_scale_m <= 0.0 or rank_rtol <= 0.0 or gram_regularization <= 0.0:
            raise ValueError("arm kinematics cache scales and tolerances must be positive")
        if set(frame_names) != set(velocity_indices):
            raise ValueError("arm kinematics cache frame and velocity-index keys must match")
        self.model = model
        self.frame_names = dict(frame_names)
        self.velocity_indices = {
            side: np.asarray(indices, dtype=int).copy() for side, indices in velocity_indices.items()
        }
        if any(indices.shape != (7,) for indices in self.velocity_indices.values()):
            raise ValueError("each cached arm must contain seven velocity indices")
        self.angular_length_scale_m = float(angular_length_scale_m)
        self.rank_rtol = float(rank_rtol)
        self.gram_regularization = float(gram_regularization)
        self._q: np.ndarray | None = None
        self._states: dict[str, ArmKinematicsState] = {}
        self.refresh_count = 0

    @staticmethod
    def _kinematic_hessian(geometric_jacobian: np.ndarray) -> np.ndarray:
        """Return dJ/dq for a serial revolute chain using unscaled angular axes."""

        joint_count = geometric_jacobian.shape[1]
        linear = geometric_jacobian[:3, :]
        angular = geometric_jacobian[3:, :]
        angular_j = angular.T[:, np.newaxis, :]
        linear_i = linear.T[np.newaxis, :, :]
        angular_i = angular.T[np.newaxis, :, :]
        linear_hessian = np.cross(angular_j, linear_i)
        angular_hessian = np.cross(angular_j, angular_i)
        upper = np.triu(np.ones((joint_count, joint_count), dtype=bool))
        linear_hessian *= upper[:, :, np.newaxis]
        angular_hessian *= upper[:, :, np.newaxis]
        lower_i, lower_j = np.tril_indices(joint_count, k=-1)
        linear_hessian[lower_i, lower_j, :] = linear_hessian[lower_j, lower_i, :]
        result = np.zeros((joint_count, 6, joint_count), dtype=float)
        result[:, :3, :] = linear_hessian.transpose(0, 2, 1)
        result[:, 3:, :] = angular_hessian.transpose(0, 2, 1)
        return result

    def refresh(self, configuration: Configuration) -> None:
        q = np.asarray(configuration.q, dtype=float)
        if self._q is not None and np.array_equal(q, self._q):
            return
        states: dict[str, ArmKinematicsState] = {}
        for side, frame_name in self.frame_names.items():
            indices = self.velocity_indices[side]
            geometric = configuration.get_frame_jacobian(frame_name)[:, indices].copy()
            scaled = geometric.copy()
            scaled[3:, :] *= self.angular_length_scale_m

            _u, singular_values, vh = np.linalg.svd(scaled, full_matrices=True)
            cutoff = self.rank_rtol * float(singular_values[0])
            rank = int(np.count_nonzero(singular_values > cutoff))
            null_basis = vh[rank:, :].T
            null_projector = null_basis @ null_basis.T

            gram = scaled @ scaled.T + self.gram_regularization * np.eye(6)
            sign, log_determinant = np.linalg.slogdet(gram)
            if sign <= 0.0 or not np.isfinite(log_determinant):
                raise RuntimeError("arm manipulability Gram matrix must be positive definite")
            states[side] = ArmKinematicsState(
                geometric_jacobian=geometric,
                scaled_jacobian=scaled,
                singular_values=singular_values,
                rank=rank,
                null_projector=null_projector,
                log_manipulability=0.5 * float(log_determinant),
                minimum_singular_value=float(np.min(singular_values)),
            )
        self._q = q.copy()
        self._states = states
        self.refresh_count += 1

    def basic_state(self, side: str, configuration: Configuration) -> ArmKinematicsState:
        """Return Jacobian conditioning without computing its joint gradient."""

        self.refresh(configuration)
        try:
            return self._states[side]
        except KeyError as error:
            raise KeyError(f"unknown cached arm {side!r}") from error

    def state(self, side: str, configuration: Configuration) -> ArmKinematicsState:
        """Return full conditioning state, computing derivatives only on demand."""

        state = self.basic_state(side, configuration)
        if state.gradient is not None:
            return state
        hessian = self._kinematic_hessian(state.geometric_jacobian)
        hessian[:, 3:, :] *= self.angular_length_scale_m
        gram = (
            state.scaled_jacobian @ state.scaled_jacobian.T
            + self.gram_regularization * np.eye(6)
        )
        inverse_gram = np.linalg.inv(gram)
        gradient = np.empty(7, dtype=float)
        for index in range(7):
            change = state.scaled_jacobian @ hessian[index].T
            gradient[index] = float(np.einsum("ij,ji", change, inverse_gram))
        resolved = replace(
            state,
            gradient=gradient,
            projected_gradient=state.null_projector @ gradient,
        )
        self._states[side] = resolved
        return resolved


@dataclass(frozen=True)
class ArmManipulabilityMetrics:
    """Conditioning facts for one arm's TCP Jacobian."""

    log_manipulability: float
    floor_log_manipulability: float
    exit_log_manipulability: float
    minimum_singular_value: float
    critical_minimum_singular_value: float
    activation: float
    active: bool
    critical: bool


class ArmNullspaceManipulabilityTask(Task):
    """Gently improve per-arm Jacobian conditioning without moving its TCP.

    The mobile base contains prismatic joints, so Pink's generic
    ``ManipulabilityTask`` cannot be applied to the root-to-TCP path.  This
    task intentionally uses only each arm's seven revolute joints and projects
    the manipulability gradient into that arm's instantaneous TCP null space.
    It is therefore a global redundancy regularizer rather than a motion-phase
    heuristic or an elbow-direction rule.
    """

    def __init__(
        self,
        model: pin.Model,
        *,
        frame_name: str,
        arm_joint_names: Sequence[str],
        reference_q: np.ndarray,
        cost: float,
        rate: float,
        floor_fraction: float,
        angular_length_scale_m: float,
        exit_fraction: float = 0.72,
        critical_sigma_fraction: float = 0.3,
        activation_ramp_s: float = 0.1,
        gain: float = 1.0,
        side: str | None = None,
        kinematics_cache: ArmKinematicsCache | None = None,
    ) -> None:
        super().__init__(cost=cost, gain=gain, lm_damping=0.0)
        if len(arm_joint_names) != 7:
            raise ValueError("arm manipulability task requires seven joints")
        if (
            rate <= 0.0
            or not 0.0 < floor_fraction < exit_fraction < 1.0
            or not 0.0 < critical_sigma_fraction < 1.0
            or activation_ramp_s <= 0.0
            or angular_length_scale_m <= 0.0
        ):
            raise ValueError("invalid arm manipulability task parameters")
        self.model = model
        self.frame_name = frame_name
        self.side = frame_name if side is None else side
        self.rate = float(rate)
        self.floor_fraction = float(floor_fraction)
        self.exit_fraction = float(exit_fraction)
        self.critical_sigma_fraction = float(critical_sigma_fraction)
        self.activation_ramp_s = float(activation_ramp_s)
        self.angular_length_scale_m = float(angular_length_scale_m)
        self.nominal_cost = float(cost)
        indices: list[int] = []
        for name in arm_joint_names:
            joint_id = model.getJointId(name)
            if joint_id == 0 or model.joints[joint_id].nv != 1:
                raise ValueError(f"invalid arm joint {name!r}")
            indices.append(int(model.joints[joint_id].idx_v))
        self.velocity_indices = np.asarray(indices, dtype=int)
        self.kinematics_cache = kinematics_cache or ArmKinematicsCache(
            model,
            frame_names={self.side: frame_name},
            velocity_indices={self.side: self.velocity_indices},
            angular_length_scale_m=angular_length_scale_m,
        )
        self._active = False
        self._activation_phase = 0.0
        self._activation = 0.0
        self._cycle_q: np.ndarray | None = None
        self.set_reference_configuration(reference_q)

    def set_reference_configuration(self, reference_q: np.ndarray) -> None:
        reference_q = np.asarray(reference_q, dtype=float)
        if reference_q.shape != (self.model.nq,) or not np.all(np.isfinite(reference_q)):
            raise ValueError(f"reference configuration must have shape ({self.model.nq},)")
        reference = Configuration(self.model, self.model.createData(), reference_q)
        state = self.kinematics_cache.basic_state(self.side, reference)
        self.nominal_log_manipulability = state.log_manipulability
        self.nominal_minimum_singular_value = state.minimum_singular_value
        self.floor_log_manipulability = float(
            self.nominal_log_manipulability + np.log(self.floor_fraction)
        )
        self.exit_log_manipulability = float(
            self.nominal_log_manipulability + np.log(self.exit_fraction)
        )
        self.critical_minimum_singular_value = float(
            self.critical_sigma_fraction * self.nominal_minimum_singular_value
        )

    @staticmethod
    def _smootherstep(value: float) -> float:
        value = float(np.clip(value, 0.0, 1.0))
        return value**3 * (10.0 + value * (-15.0 + 6.0 * value))

    def begin_cycle(self, configuration: Configuration, dt: float) -> None:
        """Advance hysteresis and its C2 activation ramp once per control tick."""

        if not np.isfinite(dt) or dt <= 0.0:
            raise ValueError("manipulability cycle timestep must be finite and positive")
        state = self.kinematics_cache.basic_state(self.side, configuration)
        if self._active:
            if state.log_manipulability >= self.exit_log_manipulability:
                self._active = False
        elif state.log_manipulability <= self.floor_log_manipulability:
            self._active = True
        direction = 1.0 if self._active else -1.0
        self._activation_phase = float(
            np.clip(self._activation_phase + direction * dt / self.activation_ramp_s, 0.0, 1.0)
        )
        self._activation = self._smootherstep(self._activation_phase)
        # Pink squares task costs internally, so sqrt(alpha) makes the QP
        # objective weight itself vary linearly with activation.
        self.cost = self.nominal_cost * np.sqrt(self._activation)
        self._cycle_q = np.asarray(configuration.q, dtype=float).copy()

    @property
    def activation(self) -> float:
        return self._activation

    def reset_activation(self) -> None:
        self._active = False
        self._activation_phase = 0.0
        self._activation = 0.0
        self.cost = 0.0
        self._cycle_q = None

    def metrics(self, configuration: Configuration) -> ArmManipulabilityMetrics:
        state = self.kinematics_cache.basic_state(self.side, configuration)
        active = self._active if self._cycle_q is not None and np.array_equal(
            configuration.q, self._cycle_q
        ) else state.log_manipulability <= self.floor_log_manipulability
        return ArmManipulabilityMetrics(
            log_manipulability=state.log_manipulability,
            floor_log_manipulability=self.floor_log_manipulability,
            exit_log_manipulability=self.exit_log_manipulability,
            minimum_singular_value=state.minimum_singular_value,
            critical_minimum_singular_value=self.critical_minimum_singular_value,
            activation=self._activation if active else 0.0,
            active=active,
            critical=state.minimum_singular_value < self.critical_minimum_singular_value,
        )

    def compute_error(self, configuration: Configuration) -> np.ndarray:
        state = self.kinematics_cache.basic_state(self.side, configuration)
        activation = self._activation
        if self._cycle_q is None or not np.array_equal(configuration.q, self._cycle_q):
            activation = float(state.log_manipulability <= self.floor_log_manipulability)
        return np.array((-self.rate * activation,), dtype=float)

    def compute_jacobian(self, configuration: Configuration) -> np.ndarray:
        jacobian = np.zeros((1, self.model.nv), dtype=float)
        state = self.kinematics_cache.basic_state(self.side, configuration)
        activation = self._activation
        if self._cycle_q is None or not np.array_equal(configuration.q, self._cycle_q):
            activation = float(state.log_manipulability <= self.floor_log_manipulability)
        if activation <= 0.0:
            return jacobian
        state = self.kinematics_cache.state(self.side, configuration)
        if state.projected_gradient is None:
            raise RuntimeError("manipulability gradient was not resolved")
        jacobian[0, self.velocity_indices] = state.projected_gradient
        return jacobian

    def nullspace_residual(self, configuration: Configuration) -> float:
        """Return ||J_arm * projected_gradient|| for regression tests."""

        state = self.kinematics_cache.state(self.side, configuration)
        if state.projected_gradient is None:
            raise RuntimeError("manipulability gradient was not resolved")
        return float(np.linalg.norm(state.geometric_jacobian @ state.projected_gradient))

    def __repr__(self) -> str:
        return (
            "ArmNullspaceManipulabilityTask("
            f"frame={self.frame_name!r}, cost={self.cost}, rate={self.rate})"
        )
