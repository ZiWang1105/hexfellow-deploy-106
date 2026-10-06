# Adapted from PoolBot's tested native adapter. Control laws remain in the
# pinned transitive policy-sim dependency; application geometry stays external.
"""Use the checked-out UMI controller without reimplementing its control laws.

The controller consumes measured coordinates in upstream order: base x/y/yaw,
lift, left arm J1..J7, right arm J1..J7. Targets are world-frame TCP transforms.
Its returned IKCommand and nested MitCommand are the actual upstream types.
The caller owns the 50 Hz clock, command guard, transport and plant substeps.
"""

from __future__ import annotations

import copy
import hashlib
import json
import os
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import yaml

from .source import UPSTREAM_ROOT, upstream as _upstream, verify_native_extension

if TYPE_CHECKING:
    from policy_sim.control.ik_core import IKCommand

DEFAULT_PROFILE = "x7-gp80-highspeed-fixed-base-v1"


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _reference(path: Path) -> str:
    # Upstream resource loaders accept relative identifiers and keep them lexical.
    # This also preserves their .relative_to(PROJECT_ROOT) reporting contract.
    return Path(os.path.relpath(path, UPSTREAM_ROOT)).as_posix()


def bind_robot_profile(
    urdf_path: str | Path,
    binding_dir: str | Path,
    *,
    profile: str = DEFAULT_PROFILE,
) -> Path:
    """Hash-bind an application tool asset while retaining every upstream setting.

    Only model identity, model/gravity hashes and provenance differ. The new
    profile is loaded and checked by UMI's ordinary immutable profile validator.
    No file inside the submodule is written. GP80 V1's native self-collision
    setting is retained; an application scene collision guard is a separate layer.
    """
    if Path(profile).name != profile or Path(profile).suffix:
        raise ValueError("profile must name a shipped upstream motion profile")
    source = UPSTREAM_ROOT / f"config/control/motion_profiles/{profile}.yaml"
    data = yaml.safe_load(source.read_text())
    if not isinstance(data.get("robot_model"), dict):
        raise ValueError("Tool binding requires an explicit upstream robot model")
    source_model = UPSTREAM_ROOT / data["robot_model"]["path"]
    if _sha(source_model) != data["robot_model"]["sha256"]:
        raise ValueError("upstream model descriptor hash mismatch")
    descriptor = json.loads(source_model.read_text())
    urdf = Path(urdf_path).resolve()
    root = ET.parse(urdf).getroot()
    assets = {urdf}
    for mesh in root.findall(".//mesh"):
        name = mesh.attrib["filename"]
        if "://" in name:
            raise ValueError("Tool URDF mesh paths must be filesystem paths")
        assets.add((urdf.parent / name).resolve())
    output = Path(binding_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    descriptor = copy.deepcopy(descriptor)
    descriptor.update(
        model_id="hexfellow-deploy-tool",
        urdf=_reference(urdf),
        assets={_reference(path): _sha(path) for path in sorted(assets)},
        provenance={
            "upstream_descriptor": _reference(source_model),
            "upstream_descriptor_sha256": _sha(source_model),
            "tool_binding": "Application tool URDF; upstream TCP definition retained",
        },
    )
    descriptor_path = output / "robot_model.json"
    descriptor_path.write_text(json.dumps(descriptor, indent=2) + "\n")
    data = copy.deepcopy(data)
    # Upstream uses the profile ID to select the matching GP80 runtime limits
    # and following-error governor. The changed SHA identifies this tool binding.
    data["robot_model"] = dict(path=_reference(descriptor_path), sha256=_sha(descriptor_path))
    data["mit"]["urdf_sha256"] = _sha(urdf)
    data.setdefault("provenance", {})["hexfellow_deploy_binding"] = {
        "source_profile": _reference(source),
        "source_file_sha256": _sha(source),
        "changes": "robot model, gravity URDF hash and identity only; all tuning/limits retained",
    }
    profile_path = output / "motion_profile.yaml"
    profile_path.write_text(yaml.safe_dump(data, sort_keys=False))
    return profile_path


def _vector(values, size: int, label: str) -> np.ndarray:
    value = np.array(values, dtype=float, copy=True)
    if value.shape != (size,) or not np.isfinite(value).all():
        raise ValueError(f"{label} requires {size} finite coordinates")
    return value


def base_drive_preview(velocity_body, duration_s: float):
    """The upstream bounded chassis-only packet, separate from arm IK.

    Both transports must execute it at BASE_COMMAND_DT_S, hold parked arms and
    use the upstream zero-hold/release semantics. The native preview owns the
    bounds; fixed-base WholeBodyCommand gates are never bypassed or altered.
    """
    body = _upstream("hardware.hexfellow_control_console")
    velocity = _vector(velocity_body, 3, "body velocity")
    preview = body.BaseDrivePreview(*map(float, velocity), float(duration_s))
    preview.validate()
    return preview


def navigation_preview(vx: float, vy: float, wz: float, duration_s: float):
    """Scalar convenience form of the same upstream chassis preview."""
    return base_drive_preview((vx, vy, wz), duration_s)


def navigation_timing() -> dict[str, float]:
    """Return the source-owned chassis cadence, freshness and stop intervals."""
    body = _upstream("hardware.hexfellow_control_console")
    return dict(control_dt_s=body.BASE_COMMAND_DT_S,
                maximum_feedback_age_s=body.BASE_STATE_STALE_S,
                zero_hold_s=body.BASE_ZERO_HOLD_S)


def _transform(value):
    import pinocchio as pin

    if isinstance(value, pin.SE3):
        return value.copy()
    matrix = np.asarray(value, dtype=float)
    if matrix.shape != (4, 4) or not np.isfinite(matrix).all():
        raise ValueError("target must be a finite 4x4 rigid transform")
    rotation = matrix[:3, :3]
    if (not np.allclose(matrix[3], [0, 0, 0, 1], atol=1e-9, rtol=0)
            or not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-7, rtol=0)
            or not np.isclose(np.linalg.det(rotation), 1, atol=1e-7, rtol=0)):
        raise ValueError("target must be a rigid transform with a proper rotation")
    return pin.SE3(rotation.copy(), matrix[:3, 3].copy())


class UmiController:
    """A thin measured-feedback adapter for the upstream planar PinkIKCore."""

    def __init__(
        self,
        urdf_path: str | Path,
        *,
        binding_dir: str | Path,
        profile: str = DEFAULT_PROFILE,
        runtime_position_bounds=None,
        runtime_velocity_bounds=None,
        runtime_effort_limits_nm=None,
    ):
        self.profile_path = bind_robot_profile(urdf_path, binding_dir, profile=profile)
        motion = _upstream("control.motion_profile")
        self.profile = motion.load_motion_profile(self.profile_path)
        self.config = motion.controller_config_for_profile(self.profile)
        self.core = _upstream("control.ik_core").PinkIKCore(
            config=self.config,
            base_mode="planar",
            runtime_position_bounds=runtime_position_bounds,
            runtime_velocity_bounds=runtime_velocity_bounds,
            runtime_effort_limits_nm=runtime_effort_limits_nm,
        )
        verify_native_extension()
        self.coordinate_names = _upstream("control.robot").PLANAR_CONTROLLED_COORDINATES
        from .contracts import JOINTS
        if (tuple(self.coordinate_names[3:]) != JOINTS
                or len(self.coordinate_names) != 18
                or self.core.model.nq != 18 or self.core.model.nv != 18):
            raise ValueError("Portable planar model must contain base x/y/yaw and 15 physical encoders")
        root_joint = self.core.model.joints[self.core.model.getJointId('root_joint')]
        if root_joint.nq != 3 or root_joint.nv != 3 or root_joint.idx_q != 0 or root_joint.idx_v != 0:
            raise ValueError("Native planar root must expose x/y/yaw as its first three coordinates")
        for index, name in enumerate(self.coordinate_names[3:], start=3):
            joint = self.core.model.joints[self.core.model.getJointId(name)]
            if joint.idx_q != index or joint.idx_v != index:
                raise ValueError("Native model coordinate order differs from explicit planar mapping")
        self.control_dt = self.config.control_dt
        self._fk_data = self.core.model.createData()
        self._simulation_guard_binding = None

    def frame_pose(self, frame: str, q) -> np.ndarray:
        """World frame pose from public measured q; no simulator access."""
        import pinocchio as pin

        q = _vector(q, 18, "measured position")
        model = self.core.model
        frame_id = model.getFrameId(frame)
        if frame_id >= model.nframes:
            raise ValueError(f"unknown robot frame: {frame}")
        pin.framesForwardKinematics(model, self._fk_data, q)
        return self._fk_data.oMf[frame_id].homogeneous.copy()

    def tcp_poses(self, q) -> dict[str, np.ndarray]:
        return {side: self.frame_pose(f"{side}_tcp", q) for side in ("left", "right")}

    def target_for_frame(self, frame: str, target, q, *, side: str = "right") -> np.ndarray:
        """Convert a rigid cue/camera target to the unchanged upstream TCP."""
        if side not in ("left", "right"):
            raise ValueError("side must be left or right")
        model = self.core.model
        frame_id, tcp_id = model.getFrameId(frame), model.getFrameId(f"{side}_tcp")
        if frame_id >= model.nframes:
            raise ValueError(f"unknown robot frame: {frame}")
        frame_joint = model.frames[frame_id].parent
        tcp_joint = model.frames[tcp_id].parent
        if frame_joint != tcp_joint:
            raise ValueError("tool frame must be rigidly attached to the selected wrist")
        world_tool = self.frame_pose(frame, q)
        world_tcp = self.frame_pose(f"{side}_tcp", q)
        return _transform(target).homogeneous @ np.linalg.inv(world_tool) @ world_tcp

    def reset(self, q, dq, timestamp: float, *, targets=None, stationary_reference=False) -> None:
        if type(stationary_reference) is not bool:
            raise ValueError("stationary_reference must be an explicit boolean")
        targets = self.tcp_poses(q) if targets is None else targets
        self.core.reset(
            _vector(q, 18, "measured position"),
            {side: _transform(targets[side]) for side in ("left", "right")},
            measured_velocity=_vector(dq, 18, "measured velocity"),
            state_timestamp=timestamp,
            stationary_reference=stationary_reference,
        )

    def step(
        self, q, dq, targets, *, state_timestamp: float, control_timestamp: float,
        feedback_received=None, feedback_policy: str = "strict", target_sample=None,
        target_twists=None, target_accelerations=None, posture_target=None,
        posture_velocity=None, control_phase: str = "tracking",
    ) -> IKCommand:
        """Run exactly one upstream profile tick; faults are returned unchanged."""
        received = (dict.fromkeys(("base", "theta", "left", "right"), state_timestamp)
                    if feedback_received is None else dict(feedback_received))
        return self.core.step(
            _vector(q, 18, "measured position"),
            {side: _transform(targets[side]) for side in ("left", "right")},
            measured_velocity=_vector(dq, 18, "measured velocity"),
            state_timestamp=state_timestamp, control_timestamp=control_timestamp,
            feedback_received=received, feedback_policy=feedback_policy,
            target_sample=target_sample, target_twists=target_twists,
            target_accelerations=target_accelerations, posture_target=posture_target,
            posture_velocity=posture_velocity, control_phase=control_phase,
        )

    def state(self, q, dq, timestamp: float, *, received=None, sequences=None):
        """Create the same public feedback snapshot consumed by the hardware guard."""
        return _upstream("hardware.whole_body").WholeBodyState(
            _vector(q, 18, "measured position"), _vector(dq, 18, "measured velocity"),
            dict.fromkeys(("base", "theta", "left", "right"), timestamp)
            if received is None else dict(received),
            {} if sequences is None else dict(sequences), {}, {},
        )

    def packet(self, command, state, sequence: int, *, created: float, control_phase="tracking"):
        """Package an executable result for UMI CommandGuard and either transport."""
        if not command.executable or command.mit is None or command.diagnostics.fault:
            raise ValueError(f"UMI command is not executable: {command.diagnostics.fault}")
        body = _upstream("hardware.whole_body")
        return body.WholeBodyCommand(
            sequence, created, created + body.COMMAND_TTL, dict(state.received), state.q.copy(),
            command.positions.copy(), command.velocity.copy(), command.mit,
            command.base.velocity_body.copy(), control_phase,
        )

    def simulation_guard(self, initial, *, stationary_reference=False):
        """Bind UMI's stock GP80 guard to this profile, without hardware acquisition."""
        if type(stationary_reference) is not bool:
            raise ValueError("stationary_reference must be an explicit boolean")
        body = _upstream("hardware.whole_body")
        effort = self.core.effective_mit_servo.torque_limit_nm
        if self._simulation_guard_binding is None:
            source = UPSTREAM_ROOT / "config/hardware/hexfellow_gp80_highspeed_j357.json"
            data = json.loads(source.read_text())
            profile_document = yaml.safe_load(self.profile_path.read_text())
            data["robot_model"] = profile_document["robot_model"]
            data["commissioning"]["motion_profile"] = self.profile.profile_id
            data["commissioning"]["profile_sha256"] = self.profile.sha256
            data["base_control_enabled"] = self.profile.base_control_enabled
            # Preserve the ownership of the original default-pose reference.
            data["default_pose"]["path"] = str(source.parent / data["default_pose"]["path"])
            mapping_path = self.profile_path.parent / "simulation_mapping.json"
            mapping_path.write_text(json.dumps(data, indent=2) + "\n")
            mapping = body.RobotMapping.load(mapping_path)
            # Match the upstream gateway: prepare immutable model/profile data
            # before acquisition, then bind actual observations at activation.
            prepared = body.CommandGuard.prepare(self.profile, mapping, "whole-body", effort)
            self._simulation_guard_binding = mapping, prepared
        mapping, prepared = self._simulation_guard_binding
        return body.CommandGuard(
            self.profile, mapping, "whole-body", initial, effort, prepared=prepared,
            stationary_reference=stationary_reference,
        )
