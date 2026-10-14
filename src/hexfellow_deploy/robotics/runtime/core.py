"""MuJoCo/Pink runtime that executes the versioned VLA contract."""

from __future__ import annotations

import dataclasses
import json
import tempfile
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Literal

import mujoco
import numpy as np
import pinocchio as pin

from ..camera.calibration import load_calibration
from ..camera.renderer import CameraBackend, DasWristCameraRenderer
from ..camera.worker import (
    CameraRenderWorker,
    CameraWorkerConfig,
    CameraWorkerError,
    RenderDevice,
    start_camera_render_worker,
)
from ..contract import (
    ANCHOR_TIMESTAMP_KEY,
    ARM_ORDER,
    GRIPPER_KEYS,
    HISTORY_DT_MAX_RATIO,
    HISTORY_DT_MIN_RATIO,
    HISTORY_DT_NS,
    HISTORY_VALID_KEY,
    IMAGE_KEYS,
    POLICY_FRAME_POSE_KEYS,
    PROMPT_KEY,
    REPLAN_STRIDE_STEPS,
    REQUEST_ID_KEY,
    TIMESTAMP_KEY,
    ArmTarget,
    ContractError,
    decode_anchor_relative_chunk,
    transform_to_pose_wxyz,
    validate_observation,
)
from ..control.gripper import angle_to_aperture
from ..control.ik import (
    IKDiagnostics,
    MaverX4PinkController,
    PinkControllerConfig,
    apply_gripper_apertures,
    set_mocap_pose,
    site_pose,
    snap_targets_to_tcp,
)
from ..control.motion_profile import SIMULATION_MOTION_PROFILE_ID, controller_config_for_profile
from ..control.robot import CONTROLLED_JOINTS, GRIPPER_APERTURE_MAX, PLANAR_CONTROLLED_COORDINATES
from ..control.self_collision import MuJoCoSelfCollisionOracle, OraclePairDiagnostic
from ..scene.config import SceneFactory, SceneManifest
from ..scene.tasks import (
    LiftTaskStatus,
    RuntimeTask,
    TaskStatus,
    create_task_evaluator,
)
from .action_buffer import WorldTarget, interpolate_targets, quintic_blend
from .grasp import GraspAssist, GraspContactTracker

SensorMode = Literal["benchmark", "fast", "disabled"]
PHYSICS_HZ = 600
IK_HZ = 50
ACTION_HZ = 10
VIEWER_HZ = 60
PHYSICS_STEPS_PER_IK = PHYSICS_HZ // IK_HZ
PHYSICS_STEPS_PER_ACTION = PHYSICS_HZ // ACTION_HZ
PHYSICS_STEPS_PER_VIEWER_SYNC = PHYSICS_HZ // VIEWER_HZ
GRIPPER_TARGET_VELOCITY_LIMIT_M_S = 0.14

PhysicsStepObserver = Callable[["PolicySimCore"], bool]
CameraWaitObserver = Callable[[], object]


def _as_pin(transform: np.ndarray) -> pin.SE3:
    return pin.SE3(transform[:3, :3], transform[:3, 3])


@dataclasses.dataclass(frozen=True)
class RuntimeCounters:
    physics_steps: int
    ik_solves: int
    target_rate_limits: int
    gripper_clamps: int
    ik_faults: int
    grasp_assists: int


@dataclasses.dataclass(frozen=True)
class StepResult:
    observation: dict[str, object]
    reward: float
    terminated: bool
    truncated: bool
    info: dict[str, object]


@dataclasses.dataclass(frozen=True)
class ControlTraceSample:
    """One viewer-rate snapshot separating policy, safety, and measured motion."""

    sim_time: float
    action_row: int | None
    policy_targets: dict[str, np.ndarray]
    limited_targets: dict[str, np.ndarray]
    measured_tcp: dict[str, np.ndarray]
    ik_diagnostics: IKDiagnostics | None
    base_pose: np.ndarray | None
    base_velocity: np.ndarray | None
    measured_q: np.ndarray
    measured_v: np.ndarray
    arm_joint_margin_fraction: np.ndarray


@dataclasses.dataclass(frozen=True)
class ObservationSample:
    timestamp_ns: int
    left_rgb: np.ndarray
    right_rgb: np.ndarray
    policy_frame_world: dict[str, np.ndarray]
    gripper_widths_m: dict[str, float]
    overview_rgb: np.ndarray | None = None


class PolicySimCore:
    """Own one episode at a time and keep policy, IK, and physics clocks separate."""

    def __init__(
        self,
        *,
        run_dir: Path | None = None,
        scene_factory: SceneFactory | None = None,
        camera_backend: CameraBackend = "auto",
        render_device: RenderDevice = "auto",
        camera_timeout_s: float = 30.0,
        camera_face_size: int | None = None,
        grasp_assist: bool = True,
        grasp_contact_activation_s: float = 3.0 / PHYSICS_HZ,
        grasp_contact_release_s: float = 3.0 / PHYSICS_HZ,
        capture_overview: bool = False,
        grasp_assist_min_lift_m: float = 0.0,
        grasp_assist_kinematic: bool = False,
        controller_config: PinkControllerConfig | None = None,
        self_collision_oracle_enabled: bool = False,
    ) -> None:
        self.scene_factory = scene_factory or SceneFactory()
        if camera_backend not in {"auto", "reference", "optimized"}:
            raise ValueError(f"unknown camera backend {camera_backend!r}")
        if render_device not in {"auto", "nvidia", "cpu"}:
            raise ValueError(f"unknown render device {render_device!r}")
        if not np.isfinite(camera_timeout_s) or camera_timeout_s <= 0:
            raise ValueError("camera timeout must be finite and positive")
        if camera_face_size is not None and (
            not isinstance(camera_face_size, int) or isinstance(camera_face_size, bool) or camera_face_size <= 0
        ):
            raise ValueError("camera face size must be a positive integer")
        if not np.isfinite(grasp_contact_activation_s) or grasp_contact_activation_s <= 0:
            raise ValueError("grasp contact activation time must be finite and positive")
        if not np.isfinite(grasp_contact_release_s) or grasp_contact_release_s <= 0:
            raise ValueError("grasp contact release time must be finite and positive")
        if not np.isfinite(grasp_assist_min_lift_m) or grasp_assist_min_lift_m < 0:
            raise ValueError("grasp-assist minimum lift must be finite and non-negative")
        self.camera_backend = camera_backend
        self.render_device = render_device
        self.camera_timeout_s = camera_timeout_s
        self.camera_face_size = camera_face_size
        self.capture_overview = bool(capture_overview)
        self.grasp_assist_min_lift_m = float(grasp_assist_min_lift_m)
        self.controller_config = controller_config or controller_config_for_profile(SIMULATION_MOTION_PROFILE_ID)
        if hasattr(self.scene_factory, "base_config"):
            if self.scene_factory.base_config is None:
                self.scene_factory.base_config = self.controller_config.base
            elif self.scene_factory.base_config != self.controller_config.base:
                raise ValueError("scene and controller base configurations differ")
        self.self_collision_oracle_enabled = bool(self_collision_oracle_enabled)
        self._temporary_run = None
        if run_dir is None:
            self._temporary_run = tempfile.TemporaryDirectory(prefix="policy-sim-")
            run_dir = Path(self._temporary_run.name)
        self.run_dir = Path(run_dir).resolve()
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.model: mujoco.MjModel | None = None
        self.data: mujoco.MjData | None = None
        self.controller: MaverX4PinkController | None = None
        self.self_collision_oracle: MuJoCoSelfCollisionOracle | None = None
        self.camera: DasWristCameraRenderer | CameraRenderWorker | None = None
        self.task: RuntimeTask | None = None
        self.manifest: SceneManifest | None = None
        self.sensor_mode: SensorMode = "disabled"
        self._profile_name: str | None = None
        self._prompt = ""
        self._physics_steps = 0
        self._ik_solves = 0
        self._target_rate_limits = 0
        self._gripper_clamps = 0
        self._ik_faults = 0
        self._last_ik_fault: str | None = None
        self.grasp = GraspAssist(enabled=grasp_assist, kinematic=grasp_assist_kinematic)
        self.grasp_contacts = GraspContactTracker(
            activation_steps=max(1, round(grasp_contact_activation_s * PHYSICS_HZ)),
            release_steps=max(1, round(grasp_contact_release_s * PHYSICS_HZ)),
        )
        self._consecutive_ik_faults = 0
        self._gripper_targets = {"left": GRIPPER_APERTURE_MAX, "right": GRIPPER_APERTURE_MAX}
        self._last_status: TaskStatus | LiftTaskStatus | None = None
        self._last_camera_metadata: dict[str, object] = {}
        self._last_overview_rgb: np.ndarray | None = None
        self._grasp_baseline_heights: dict[str, float] = {}
        self._active_action_row: int | None = None
        self._active_policy_targets: dict[str, np.ndarray] = {}
        self._last_ik_diagnostics: IKDiagnostics | None = None
        self._last_self_collision_oracle: tuple[OraclePairDiagnostic, ...] = ()
        self._minimum_self_collision_oracle_m: dict[str, float] = {}
        self._episode_generation = 0
        self._episode_id = ""
        self._request_id = 0
        self._history_sample: ObservationSample | None = None

    def _require_ready(self) -> tuple[mujoco.MjModel, mujoco.MjData, MaverX4PinkController]:
        if self.model is None or self.data is None or self.controller is None:
            raise RuntimeError("reset() must be called before using the environment")
        return self.model, self.data, self.controller

    def _close_camera(self) -> None:
        if self.camera is not None:
            self.camera.close()
            self.camera = None

    def _invalidate_episode(self) -> None:
        """Release episode resources and prevent stale state after a failed reset."""
        self._close_camera()
        self.model = None
        self.data = None
        self.controller = None
        self.self_collision_oracle = None
        self.task = None
        self.manifest = None
        self.sensor_mode = "disabled"
        self._profile_name = None
        self._prompt = ""
        self._last_status = None
        self._last_camera_metadata = {}
        self._last_overview_rgb = None
        self._grasp_baseline_heights = {}
        self._active_action_row = None
        self._active_policy_targets = {}
        self._last_ik_diagnostics = None
        self._last_self_collision_oracle = ()
        self._minimum_self_collision_oracle_m = {}
        self._episode_id = ""
        self._request_id = 0
        self._history_sample = None
        self._physics_steps = 0
        self._ik_solves = 0
        self._target_rate_limits = 0
        self._gripper_clamps = 0
        self._ik_faults = 0
        self._last_ik_fault = None
        self._consecutive_ik_faults = 0
        self._gripper_targets = {"left": GRIPPER_APERTURE_MAX, "right": GRIPPER_APERTURE_MAX}
        self.grasp.reset()
        self.grasp_contacts.reset()

    def reset(
        self,
        *,
        seed: int,
        profile: str = "smoke",
        sensor_mode: SensorMode | None = None,
        object_ids_override: tuple[str, ...] | None = None,
        object_positions_override: Mapping[str, tuple[float, float, float]] | None = None,
        required_objects_override: int | None = None,
        appearance_seed: int | None = None,
        robot_initial_qpos: Mapping[str, float] | None = None,
    ) -> tuple[dict[str, object], dict[str, object]]:
        if not isinstance(seed, int) or isinstance(seed, bool) or seed < 0:
            raise ValueError("reset seed must be a non-negative integer")
        if profile not in self.scene_factory.config.profiles:
            raise ValueError(f"unknown profile {profile!r}; choose {sorted(self.scene_factory.config.profiles)}")
        if sensor_mode is not None and sensor_mode not in {"benchmark", "fast", "disabled"}:
            raise ValueError(f"unknown sensor mode {sensor_mode!r}")
        self._invalidate_episode()
        try:
            return self._reset_episode(
                seed=seed,
                profile=profile,
                sensor_mode=sensor_mode,
                object_ids_override=object_ids_override,
                object_positions_override=object_positions_override,
                required_objects_override=required_objects_override,
                appearance_seed=appearance_seed,
                robot_initial_qpos=robot_initial_qpos,
            )
        except Exception:
            self._invalidate_episode()
            raise

    def _reset_episode(
        self,
        *,
        seed: int,
        profile: str = "smoke",
        sensor_mode: SensorMode | None = None,
        object_ids_override: tuple[str, ...] | None = None,
        object_positions_override: Mapping[str, tuple[float, float, float]] | None = None,
        required_objects_override: int | None = None,
        appearance_seed: int | None = None,
        robot_initial_qpos: Mapping[str, float] | None = None,
    ) -> tuple[dict[str, object], dict[str, object]]:
        episode_dir = self.run_dir / "episodes" / f"{profile}_{seed:06d}"
        manifest = self.scene_factory.build(
            episode_dir,
            profile_name=profile,
            seed=seed,
            object_ids_override=object_ids_override,
            object_positions_override=object_positions_override,
            required_objects_override=required_objects_override,
            appearance_seed=appearance_seed,
            robot_initial_qpos=robot_initial_qpos,
        )
        model = mujoco.MjModel.from_xml_path(manifest.scene_path)
        return self._initialize_loaded_scene(
            model=model,
            manifest=manifest,
            profile=profile,
            sensor_mode=sensor_mode,
            episode_dir=episode_dir,
            write_reset_info=True,
        )

    def reset_current_scene(
        self,
        *,
        sensor_mode: SensorMode | None = None,
    ) -> tuple[dict[str, object], dict[str, object]]:
        """Reset physics and control state without regenerating the current MJCF.

        Candidate search uses this to trial several grasps in an identical
        settled scene.  It deliberately keeps the immutable model/manifest and
        only recreates MuJoCo data, controller, and task state.
        """

        if self.model is None or self.manifest is None or self._profile_name is None:
            raise RuntimeError("reset() must be called before reset_current_scene()")
        self._close_camera()
        # MIT config mutates actuator gains/biases in-place. Reload the immutable
        # scene before the position-servo reset settling phase on subsequent runs.
        model = self.model
        if self.controller_config.mit_servo is not None:
            model = mujoco.MjModel.from_xml_path(self.manifest.scene_path)
        return self._initialize_loaded_scene(
            model=model,
            manifest=self.manifest,
            profile=self._profile_name,
            sensor_mode=sensor_mode,
            episode_dir=None,
            write_reset_info=False,
        )

    def _initialize_loaded_scene(
        self,
        *,
        model: mujoco.MjModel,
        manifest: SceneManifest,
        profile: str,
        sensor_mode: SensorMode | None,
        episode_dir: Path | None,
        write_reset_info: bool,
    ) -> tuple[dict[str, object], dict[str, object]]:
        selected_mode = manifest.sensor_mode if sensor_mode is None else sensor_mode
        if selected_mode not in {"benchmark", "fast", "disabled"}:
            raise ValueError(f"unknown sensor mode {selected_mode!r}")
        if not np.isclose(model.opt.timestep, 1.0 / PHYSICS_HZ, atol=1e-12):
            raise RuntimeError(f"runtime requires {PHYSICS_HZ} Hz physics, got dt={model.opt.timestep}")
        data = mujoco.MjData(model)
        home_key = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "home")
        if home_key < 0:
            raise RuntimeError("task scene has no home keyframe")
        mujoco.mj_resetDataKeyframe(model, data, home_key)
        apply_gripper_apertures(model, data, GRIPPER_APERTURE_MAX, GRIPPER_APERTURE_MAX)
        # Settle free objects while the position actuators hold the home pose.
        for _ in range(round(0.25 / model.opt.timestep)):
            mujoco.mj_step(model, data)
        mujoco.mj_forward(model, data)

        controller = MaverX4PinkController(model, config=self.controller_config)
        collision_oracle = (
            None
            if not self.self_collision_oracle_enabled
            or controller.self_collision_system is None
            else MuJoCoSelfCollisionOracle(model, controller.self_collision_system.document)
        )
        controller.check_fk_parity(data)
        targets = snap_targets_to_tcp(model, data)
        controller.reset(
            controller.measured_configuration(data),
            targets,
            measured_velocity=controller.measured_velocity(data),
            measured_gripper_positions=controller.measured_gripper_positions(data),
            state_timestamp=float(data.time),
        )

        profile_config = self.scene_factory.config.profiles[profile]
        task = create_task_evaluator(
            manifest.task,
            model,
            manifest.object_ids,
            self.scene_factory.config.success,
            profile_config.timeout_seconds,
            required_objects=manifest.required_objects,
        )
        task.reset(data)

        self.model = model
        self.data = data
        self.controller = controller
        self.self_collision_oracle = collision_oracle
        self.task = task
        self.manifest = manifest
        self.sensor_mode = selected_mode
        self._profile_name = profile
        self._prompt = manifest.prompt
        self._physics_steps = 0
        self._ik_solves = 0
        self._target_rate_limits = 0
        self._gripper_clamps = 0
        self._ik_faults = 0
        self._last_ik_fault = None
        self.grasp.reset()
        self.grasp_contacts.reset()
        self._consecutive_ik_faults = 0
        self._gripper_targets = {"left": GRIPPER_APERTURE_MAX, "right": GRIPPER_APERTURE_MAX}
        tool_world = self.tool_transforms()
        self._active_action_row = None
        self._active_policy_targets = {side: transform.copy() for side, transform in tool_world.items()}
        self._last_ik_diagnostics = None
        self._last_self_collision_oracle = ()
        self._minimum_self_collision_oracle_m = {}
        self._episode_generation += 1
        self._episode_id = f"{manifest.task}:{profile}:{manifest.seed}:{self._episode_generation}"
        self._request_id = 0
        self._history_sample = None
        self._last_status = task.update(data)
        self._grasp_baseline_heights = {
            object_id: float(data.xpos[body_id][2]) for object_id, body_id in task.object_body_ids.items()
        }
        observation = self.observe()
        info = self.info()
        if write_reset_info:
            if episode_dir is None:
                raise RuntimeError("reset info requires an episode directory")
            (episode_dir / "reset_info.json").write_text(
                json.dumps(info, indent=2, sort_keys=True) + "\n", encoding="utf-8"
            )
        return observation, info

    def _sample_self_collision_oracle(self, data: mujoco.MjData) -> None:
        """Record the independent CAD oracle without feeding it back to IK."""

        if self.self_collision_oracle is None:
            return
        diagnostics = self.self_collision_oracle.sample(data)
        self._last_self_collision_oracle = diagnostics
        for diagnostic in diagnostics:
            key = f"{diagnostic.first}::{diagnostic.second}"
            self._minimum_self_collision_oracle_m[key] = min(
                self._minimum_self_collision_oracle_m.get(key, np.inf),
                diagnostic.distance_m,
            )

    def _set_gripper_target(self, side: str, desired_width_m: float, dt: float) -> None:
        """Rate-limit gripper commands independently of the C2 SE(3) spline."""

        maximum_change = GRIPPER_TARGET_VELOCITY_LIMIT_M_S * dt
        previous = self._gripper_targets[side]
        self._gripper_targets[side] = float(
            np.clip(desired_width_m, max(0.0, previous - maximum_change),
                    min(GRIPPER_APERTURE_MAX, previous + maximum_change))
        )

    def tool_transforms(self) -> dict[str, np.ndarray]:
        """Measured fingertip TCP transforms used by Pink and task geometry."""
        model, data, _ = self._require_ready()
        result: dict[str, np.ndarray] = {}
        for side in ARM_ORDER:
            pose = site_pose(model, data, f"{side}_tcp")
            transform = np.eye(4)
            transform[:3, :3] = pose.rotation
            transform[:3, 3] = pose.translation
            result[side] = transform
        return result

    def policy_frame_transforms(self) -> dict[str, np.ndarray]:
        """Measured DAS base_link transforms used as the policy frame."""
        model, data, _ = self._require_ready()
        result: dict[str, np.ndarray] = {}
        for side in ARM_ORDER:
            pose = site_pose(model, data, f"{side}_policy_eef")
            transform = np.eye(4)
            transform[:3, :3] = pose.rotation
            transform[:3, 3] = pose.translation
            result[side] = transform
        return result

    def policy_frame_to_tcp(self) -> dict[str, np.ndarray]:
        """Return the explicit fixed transform from policy frame to TCP."""
        policy = self.policy_frame_transforms()
        tcp = self.tool_transforms()
        return {side: np.linalg.inv(policy[side]) @ tcp[side] for side in ARM_ORDER}

    def decode_policy_chunk(
        self,
        chunk: np.ndarray,
        *,
        rows: int = REPLAN_STRIDE_STEPS,
        anchor_world: Mapping[str, np.ndarray] | None = None,
    ) -> tuple[dict[str, ArmTarget], ...]:
        """Decode fixed-anchor policy-frame actions and convert them to TCP."""
        if rows < 1 or rows > REPLAN_STRIDE_STEPS:
            raise ContractError(f"runtime executes 1..{REPLAN_STRIDE_STEPS} replan-stride rows per chunk, got {rows}")
        anchors = self.policy_frame_transforms() if anchor_world is None else anchor_world
        policy_commands = decode_anchor_relative_chunk(chunk, anchors, rows=rows)
        offsets = self.policy_frame_to_tcp()
        return tuple(
            {
                side: dataclasses.replace(targets[side], transform=targets[side].transform @ offsets[side])
                for side in ARM_ORDER
            }
            for targets in policy_commands
        )

    def prepare_tcp_pose(
        self,
        targets_world: Mapping[str, np.ndarray],
        *,
        timeout_s: float = 4.0,
        position_tolerance_m: float = 0.003,
        rotation_tolerance_deg: float = 2.0,
    ) -> dict[str, object]:
        """Move to a fixed, reachable TCP pose before capturing diagnostics.

        This is reset preparation, not policy execution: it advances MuJoCo so
        the position servos can settle but deliberately leaves rollout counters
        at zero.  Production evaluation never calls it.
        """
        model, data, controller = self._require_ready()
        missing = [side for side in ARM_ORDER if side not in targets_world]
        if missing:
            raise ValueError(f"missing diagnostic TCP targets for {missing}")
        targets = {side: _as_pin(np.asarray(targets_world[side], dtype=np.float64)) for side in ARM_ORDER}
        controller.reset(
            controller.measured_configuration(data),
            targets,
            measured_velocity=controller.measured_velocity(data),
            measured_gripper_positions=controller.measured_gripper_positions(data),
            state_timestamp=float(data.time),
        )
        for side in ARM_ORDER:
            set_mocap_pose(model, data, f"{side}_ik_target", targets[side])
        required_stable_steps = round(0.2 * PHYSICS_HZ)
        stable_steps = 0
        faults = 0
        solves = 0
        last_errors: dict[str, dict[str, float]] = {}
        for physics_step in range(round(timeout_s * PHYSICS_HZ)):
            if physics_step % PHYSICS_STEPS_PER_IK == 0:
                command = controller.step(
                    controller.measured_configuration(data),
                    targets,
                    measured_velocity=controller.measured_velocity(data),
                    measured_gripper_positions=controller.measured_gripper_positions(data),
                    state_timestamp=float(data.time),
                )
                controller.apply_command(data, command)
                solves += 1
                faults += int(command.diagnostics.fault is not None)
            mujoco.mj_step(model, data)
            self._sample_self_collision_oracle(data)
            measured = self.tool_transforms()
            last_errors = {}
            within = True
            for side in ARM_ORDER:
                relative = measured[side][:3, :3].T @ targets_world[side][:3, :3]
                cosine = float(np.clip((np.trace(relative) - 1.0) / 2.0, -1.0, 1.0))
                position_error = float(np.linalg.norm(measured[side][:3, 3] - targets_world[side][:3, 3]))
                rotation_error = float(np.degrees(np.arccos(cosine)))
                last_errors[side] = {
                    "position_m": position_error,
                    "rotation_deg": rotation_error,
                }
                within &= position_error <= position_tolerance_m and rotation_error <= rotation_tolerance_deg
            stable_steps = stable_steps + 1 if within else 0
            if stable_steps >= required_stable_steps:
                break
        return {
            "settled": stable_steps >= required_stable_steps,
            "sim_time_s": float(data.time),
            "solves": solves,
            "faults": faults,
            "errors": last_errors,
        }

    def gripper_widths(self) -> dict[str, float]:
        model, data, _ = self._require_ready()
        result: dict[str, float] = {}
        for side, joint_name in (("left", "robot_joint1"), ("right", "robot2_joint1")):
            joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
            angle = float(data.qpos[model.jnt_qposadr[joint_id]])
            result[side] = angle_to_aperture(angle)
        return result

    @staticmethod
    def _pin_transform(pose: pin.SE3) -> np.ndarray:
        transform = np.eye(4)
        transform[:3, :3] = pose.rotation
        transform[:3, 3] = pose.translation
        return transform

    def control_trace_sample(self) -> ControlTraceSample:
        """Expose read-only controller state for local diagnostics and visualization."""
        _, data, controller = self._require_ready()
        limited = {side: self._pin_transform(target) for side, target in controller.limited_targets().items()}
        measured_q = controller.measured_configuration(data).copy()
        arm_start = controller._base_dofs + 1
        arm_margin = np.minimum(
            measured_q[arm_start:] - controller.model.lowerPositionLimit[arm_start:],
            controller.model.upperPositionLimit[arm_start:] - measured_q[arm_start:],
        ) / (
            controller.model.upperPositionLimit[arm_start:]
            - controller.model.lowerPositionLimit[arm_start:]
        )
        return ControlTraceSample(
            sim_time=float(data.time),
            action_row=self._active_action_row,
            policy_targets={side: transform.copy() for side, transform in self._active_policy_targets.items()},
            limited_targets=limited,
            measured_tcp=self.tool_transforms(),
            ik_diagnostics=self._last_ik_diagnostics,
            base_pose=controller.base_pose(data),
            base_velocity=controller.base_velocity(data),
            measured_q=measured_q,
            measured_v=controller.measured_velocity(data).copy(),
            arm_joint_margin_fraction=np.asarray(arm_margin, dtype=float),
        )

    def _create_camera(
        self,
        model: mujoco.MjModel,
        *,
        quality: Literal["preview", "policy"],
        camera_mode: Literal["pinhole", "fisheye"],
    ) -> DasWristCameraRenderer | CameraRenderWorker:
        if self.manifest is None:
            raise RuntimeError("scene manifest is unavailable")
        config = CameraWorkerConfig(
            scene_path=str(self.manifest.scene_path),
            calibration_path=str(self.scene_factory.camera_calibration_path),
            quality=quality,
            mode=camera_mode,
            backend=self.camera_backend,
            render_device=self.render_device,
            face_size=self.camera_face_size,
            include_overview=self.capture_overview,
        )
        try:
            return start_camera_render_worker(config)
        except CameraWorkerError as error:
            if self.render_device == "nvidia":
                raise
            self._last_camera_metadata = {
                "worker_fallback": f"{type(error).__name__}: {error}",
            }
            return DasWristCameraRenderer(
                model,
                self.scene_factory.camera_calibration_path,
                quality=quality,
                backend=self.camera_backend,
                face_size=self.camera_face_size,
            )

    def _render_images(
        self,
        wait_observer: CameraWaitObserver | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        model, data, _ = self._require_ready()
        if self.sensor_mode == "disabled":
            self._last_camera_metadata = {
                "mode": "disabled",
                "score_valid": False,
                "calibration_status": "provisional",
            }
            blank = np.zeros((224, 224, 3), dtype=np.uint8)
            self._last_overview_rgb = blank.copy() if self.capture_overview else None
            return blank.copy(), blank.copy()
        quality = "policy" if self.sensor_mode == "benchmark" else "preview"
        camera_mode = "fisheye" if self.sensor_mode == "benchmark" else "pinhole"
        if self.camera is None:
            self.camera = self._create_camera(
                model,
                quality=quality,
                camera_mode=camera_mode,
            )
        calibration = load_calibration(self.scene_factory.camera_calibration_path)
        fallback = dict(self._last_camera_metadata)
        if isinstance(self.camera, CameraRenderWorker):
            try:
                result = self.camera.render(
                    model,
                    data,
                    timeout_s=self.camera_timeout_s,
                    idle_callback=wait_observer,
                )
            except CameraWorkerError as error:
                self.camera.close()
                self.camera = None
                if self.render_device != "auto":
                    raise
                self._last_camera_metadata = {
                    **fallback,
                    "worker_fallback": f"{type(error).__name__}: {error}",
                }
                self.camera = DasWristCameraRenderer(
                    model,
                    self.scene_factory.camera_calibration_path,
                    quality=quality,
                    backend=self.camera_backend,
                    face_size=self.camera_face_size,
                )
                return self._render_images(wait_observer)
            by_side = result.policy_images
            self._last_overview_rgb = result.images["overview"].copy() if self.capture_overview else None
            renderer_metadata = result.metadata
            resize_backends = ["swscale_bilinear"]
        else:
            frames = self.camera.render(data, mode=camera_mode)
            frame_by_side = {frame.side: frame for frame in frames.values()}
            by_side = {side: frame.policy_rgb for side, frame in frame_by_side.items()}
            renderer_metadata = {
                "process": "in_process",
                "requested_device": self.render_device,
                "actual_device": "unknown",
                "gl": self.camera.gl_info,
                "backend": self.camera.backend,
                **self.camera.last_timings,
            }
            resize_backends = sorted({frame.resize_backend for frame in frame_by_side.values()})
            self._last_overview_rgb = None
        if self.sensor_mode == "benchmark" and "swscale_bilinear" not in resize_backends:
            raise RuntimeError("benchmark camera requires PyAV/swscale bilinear resizing")
        self._last_camera_metadata = {
            **fallback,
            "mode": camera_mode,
            "quality": quality,
            "calibration_status": (
                "provisional" if any(frame.provisional for frame in calibration.streams.values()) else "calibrated"
            ),
            "resize_backend": resize_backends,
            **renderer_metadata,
        }
        self._last_camera_metadata["score_valid"] = (
            self.sensor_mode == "benchmark" and self._last_camera_metadata["calibration_status"] == "calibrated"
        )
        return by_side["left"].copy(), by_side["right"].copy()

    def _capture_observation_sample(
        self,
        wait_observer: CameraWaitObserver | None = None,
    ) -> ObservationSample:
        _, data, _ = self._require_ready()
        left, right = self._render_images(wait_observer)
        return ObservationSample(
            timestamp_ns=int(round(float(data.time) * 1e9)),
            left_rgb=left,
            right_rgb=right,
            policy_frame_world=self.policy_frame_transforms(),
            gripper_widths_m=self.gripper_widths(),
            overview_rgb=(None if self._last_overview_rgb is None else self._last_overview_rgb.copy()),
        )

    def capture_observation_sample(
        self,
        wait_observer: CameraWaitObserver | None = None,
    ) -> ObservationSample:
        """Capture one synchronized raw sample without mutating policy history."""

        return self._capture_observation_sample(wait_observer)

    def held_grasps(self) -> dict[str, str | None]:
        """Return objects physically held or stabilized by a validated carrier."""

        return {side: self.grasp.active[side] or self.grasp_contacts.active[side] for side in ARM_ORDER}

    def _update_grasp_state(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        controller: MaverX4PinkController,
    ) -> bool:
        if self.task is None:
            raise RuntimeError("task is not initialized")
        tools = self.tool_transforms()
        validated = self.grasp_contacts.update(
            model,
            data,
            self.task,
            self._gripper_targets,
            tools,
        )
        assist_eligible = dict(validated)
        if self.grasp_assist_min_lift_m > 0:
            for side, object_id in validated.items():
                if object_id is None:
                    continue
                body_id = self.task.object_body_ids[object_id]
                lift = float(data.xpos[body_id][2]) - self._grasp_baseline_heights[object_id]
                if lift < self.grasp_assist_min_lift_m:
                    assist_eligible[side] = None
        return self.grasp.update(
            model,
            data,
            controller,
            self.task,
            self._gripper_targets,
            tools,
            validated_grasps=assist_eligible,
        )

    def observe(self, wait_observer: CameraWaitObserver | None = None) -> dict[str, object]:
        current = self._capture_observation_sample(wait_observer)
        previous = self._history_sample
        pair_valid = False
        if previous is not None:
            dt_ratio = (current.timestamp_ns - previous.timestamp_ns) / HISTORY_DT_NS
            pair_valid = HISTORY_DT_MIN_RATIO <= dt_ratio <= HISTORY_DT_MAX_RATIO
        if not pair_valid:
            previous = current
        assert previous is not None
        observation: dict[str, object] = {
            "episode_id": self._episode_id,
            REQUEST_ID_KEY: self._request_id,
            ANCHOR_TIMESTAMP_KEY: current.timestamp_ns,
            IMAGE_KEYS[0]: np.stack((previous.left_rgb, current.left_rgb)).astype(np.uint8, copy=False),
            IMAGE_KEYS[1]: np.stack((previous.right_rgb, current.right_rgb)).astype(np.uint8, copy=False),
            POLICY_FRAME_POSE_KEYS["left"]: np.stack(
                (
                    transform_to_pose_wxyz(previous.policy_frame_world["left"]),
                    transform_to_pose_wxyz(current.policy_frame_world["left"]),
                )
            ).astype(np.float32),
            POLICY_FRAME_POSE_KEYS["right"]: np.stack(
                (
                    transform_to_pose_wxyz(previous.policy_frame_world["right"]),
                    transform_to_pose_wxyz(current.policy_frame_world["right"]),
                )
            ).astype(np.float32),
            GRIPPER_KEYS["left"]: np.asarray(
                (previous.gripper_widths_m["left"], current.gripper_widths_m["left"]),
                dtype=np.float32,
            ),
            GRIPPER_KEYS["right"]: np.asarray(
                (previous.gripper_widths_m["right"], current.gripper_widths_m["right"]),
                dtype=np.float32,
            ),
            TIMESTAMP_KEY: np.asarray((previous.timestamp_ns, current.timestamp_ns), dtype=np.uint64),
            HISTORY_VALID_KEY: np.asarray((pair_valid, True), dtype=bool),
            PROMPT_KEY: self._prompt,
        }
        validate_observation(observation)
        self._history_sample = current
        self._request_id += 1
        return observation

    def capture_history_sample(self, wait_observer: CameraWaitObserver | None = None) -> None:
        """Capture the t-100 ms raw facts used by the next async request."""
        self._history_sample = self._capture_observation_sample(wait_observer)

    def truncate(self, reason: str) -> None:
        """End the current episode without turning a user cancellation into a fault."""
        if self._last_status is None:
            raise RuntimeError("task is not initialized")
        if not self._last_status.terminated and not self._last_status.truncated:
            self._last_status = dataclasses.replace(
                self._last_status,
                terminated=False,
                truncated=True,
                reason=reason,
            )

    def advance_world_targets(
        self,
        targets: Mapping[str, WorldTarget],
        *,
        physics_steps: int = PHYSICS_STEPS_PER_IK,
        step_observer: PhysicsStepObserver | None = None,
        physics_step_observer: PhysicsStepObserver | None = None,
        continue_after_task_terminal: bool = False,
        continue_after_task_success: bool = False,
        target_sample=None,
    ) -> bool:
        """Advance one short real-time segment without invoking policy inference.

        ``continue_after_task_terminal`` is reserved for open-ended interactive
        tools. It ignores task success/timeout while preserving sustained IK
        fault and observer cancellation as hard stops.
        """
        model, data, controller = self._require_ready()
        if self.task is None or self._last_status is None:
            raise RuntimeError("task is not initialized")
        if physics_steps < 1:
            raise ValueError("physics_steps must be positive")
        pin_targets = {side: _as_pin(targets[side].transform) for side in ARM_ORDER}
        self._active_policy_targets = {side: targets[side].transform.copy() for side in ARM_ORDER}
        for side in ARM_ORDER:
            set_mocap_pose(model, data, f"{side}_ik_target", pin_targets[side])
            self._set_gripper_target(side, float(targets[side].width_m), controller.config.control_dt)
        apply_gripper_apertures(model, data, self._gripper_targets["left"], self._gripper_targets["right"])
        command = controller.step(
            controller.measured_configuration(data),
            pin_targets,
            target_sample=target_sample,
            measured_velocity=controller.measured_velocity(data),
            measured_gripper_positions=controller.measured_gripper_positions(data),
            state_timestamp=float(data.time),
            target_twists=None
            if all(targets[side].body_twist is None for side in ARM_ORDER)
            else {
                side: np.zeros(6)
                if targets[side].body_twist is None
                else np.asarray(targets[side].body_twist, dtype=float)
                for side in ARM_ORDER
            },
            target_accelerations=None
            if all(targets[side].body_acceleration is None for side in ARM_ORDER)
            else {
                side: np.zeros(6)
                if targets[side].body_acceleration is None
                else np.asarray(targets[side].body_acceleration, dtype=float)
                for side in ARM_ORDER
            },
        )
        self._last_ik_diagnostics = command.diagnostics
        self._ik_solves += 1
        self._target_rate_limits += int(command.diagnostics.target_limited)
        if command.diagnostics.fault:
            self._ik_faults += 1
            self._last_ik_fault = command.diagnostics.fault
            self._consecutive_ik_faults += 1
        else:
            self._consecutive_ik_faults = 0
        if not command.executable:
            self._last_status = dataclasses.replace(
                self._last_status, terminated=True, reason="ik_fault"
            )
            return False
        controller.apply_command(data, command)
        for _ in range(physics_steps):
            self.grasp.update_carriers(model, data, self.tool_transforms())
            mujoco.mj_step(model, data)
            self._sample_self_collision_oracle(data)
            if self.grasp.kinematic and any(self.grasp.active.values()):
                self.grasp.update_carriers(model, data, self.tool_transforms())
                mujoco.mj_forward(model, data)
            self._physics_steps += 1
            if self._update_grasp_state(model, data, controller):
                self._consecutive_ik_faults = 0
            self._last_status = self.task.update(data, self.held_grasps())
            # Demonstrations may finish their requested lift after the task's
            # minimum success dwell. Never bypass failures or the time budget.
            if continue_after_task_success and self._last_status.elapsed_seconds >= self.task.timeout_seconds:
                self._last_status = dataclasses.replace(
                    self._last_status, success=False, terminated=False, truncated=True, reason="timeout"
                )
            if self._consecutive_ik_faults >= round(0.2 * IK_HZ):
                self._last_status = dataclasses.replace(self._last_status, terminated=True, reason="ik_fault")
                return False
            if (
                not continue_after_task_terminal
                and (self._last_status.terminated or self._last_status.truncated)
                and not (continue_after_task_success and self._last_status.success)
            ):
                return False
            if physics_step_observer is not None and not physics_step_observer(self):
                self.truncate("physics_observer")
                return False
            if (
                step_observer is not None
                and self._physics_steps % PHYSICS_STEPS_PER_VIEWER_SYNC == 0
                and not step_observer(self)
            ):
                self.truncate("viewer_closed")
                return False
        return True

    def execute_chunk(
        self,
        chunk: np.ndarray,
        *,
        rows: int = REPLAN_STRIDE_STEPS,
        step_observer: PhysicsStepObserver | None = None,
        camera_wait_observer: CameraWaitObserver | None = None,
        control_observer: Callable | None = None,
    ) -> StepResult:
        model, data, controller = self._require_ready()
        if self.task is None or self._last_status is None:
            raise RuntimeError("task is not initialized")
        if self._last_status.terminated or self._last_status.truncated:
            raise RuntimeError("episode has already ended; call reset()")
        commands = self.decode_policy_chunk(chunk, rows=rows)
        self._gripper_clamps += sum(target.gripper_clamped for command in commands for target in command.values())
        chunk_start = self._physics_steps
        chunk_end = chunk_start + rows * PHYSICS_STEPS_PER_ACTION
        starting = {side: WorldTarget(self.tool_transforms()[side], self.gripper_widths()[side]) for side in ARM_ORDER}
        while self._physics_steps < chunk_end:
            if self._physics_steps == chunk_end - PHYSICS_STEPS_PER_ACTION:
                self._history_sample = self._capture_observation_sample(camera_wait_observer)
            if self._physics_steps % PHYSICS_STEPS_PER_ACTION == 0:
                row = (self._physics_steps - chunk_start) // PHYSICS_STEPS_PER_ACTION
                active = commands[row]
                self._active_action_row = row
                self._active_policy_targets = {side: active[side].transform.copy() for side in ARM_ORDER}
            if self._physics_steps % PHYSICS_STEPS_PER_IK == 0:
                elapsed = self._physics_steps - chunk_start
                command_index = min(elapsed // PHYSICS_STEPS_PER_ACTION, rows - 1)
                # The command produced now is held for one 50 Hz control tick,
                # so target the end of that tick. This reaches every 10 Hz
                # waypoint exactly, including the final row of the chunk.
                fraction = min(
                    1.0,
                    ((elapsed % PHYSICS_STEPS_PER_ACTION) + PHYSICS_STEPS_PER_IK) / PHYSICS_STEPS_PER_ACTION,
                )
                left_targets = (
                    starting
                    if command_index == 0
                    else {
                        side: WorldTarget(
                            commands[command_index - 1][side].transform,
                            commands[command_index - 1][side].width_m,
                        )
                        for side in ARM_ORDER
                    }
                )
                right_targets = {
                    side: WorldTarget(
                        commands[command_index][side].transform,
                        commands[command_index][side].width_m,
                    )
                    for side in ARM_ORDER
                }
                blend, blend_rate, blend_acceleration = quintic_blend(
                    fraction, 1.0 / ACTION_HZ
                )
                interpolated = interpolate_targets(
                    left_targets,
                    right_targets,
                    blend,
                    fraction_rate=blend_rate,
                    fraction_acceleration=blend_acceleration,
                )
                targets = {side: _as_pin(interpolated[side].transform) for side in ARM_ORDER}
                for side in ARM_ORDER:
                    set_mocap_pose(model, data, f"{side}_ik_target", targets[side])
                    self._set_gripper_target(
                        side,
                        interpolated[side].width_m,
                        controller.config.control_dt,
                    )
                apply_gripper_apertures(
                    model,
                    data,
                    self._gripper_targets["left"],
                    self._gripper_targets["right"],
                )
                command = controller.step(
                    controller.measured_configuration(data),
                    targets,
                    measured_velocity=controller.measured_velocity(data),
                    measured_gripper_positions=controller.measured_gripper_positions(data),
                    state_timestamp=float(data.time),
                    target_twists={
                        side: np.asarray(interpolated[side].body_twist, dtype=float)
                        for side in ARM_ORDER
                    },
                    target_accelerations={
                        side: np.asarray(interpolated[side].body_acceleration, dtype=float)
                        for side in ARM_ORDER
                    },
                )
                if not command.executable:
                    self._last_ik_diagnostics = command.diagnostics
                    self._ik_solves += 1
                    self._ik_faults += 1
                    self._last_ik_fault = command.diagnostics.fault
                    self._last_status = dataclasses.replace(self._last_status, terminated=True, reason="ik_fault")
                    if control_observer is not None:
                        control_observer(self)
                    break
                controller.apply_command(data, command)
                self._last_ik_diagnostics = command.diagnostics
                self._ik_solves += 1
                self._target_rate_limits += int(command.diagnostics.target_limited)
                if command.diagnostics.fault:
                    self._ik_faults += 1
                    self._last_ik_fault = command.diagnostics.fault
                    self._consecutive_ik_faults += 1
                else:
                    self._consecutive_ik_faults = 0
                if self._consecutive_ik_faults >= round(0.2 * IK_HZ):
                    self._last_status = dataclasses.replace(
                        self._last_status,
                        terminated=True,
                        reason="ik_fault",
                    )
                    break
            self.grasp.update_carriers(model, data, self.tool_transforms())
            mujoco.mj_step(model, data)
            self._sample_self_collision_oracle(data)
            if self.grasp.kinematic and any(self.grasp.active.values()):
                self.grasp.update_carriers(model, data, self.tool_transforms())
                mujoco.mj_forward(model, data)
            self._physics_steps += 1
            if self._update_grasp_state(model, data, controller):
                self._consecutive_ik_faults = 0
            self._last_status = self.task.update(data, self.held_grasps())
            if control_observer is not None and self._physics_steps % PHYSICS_STEPS_PER_IK == 0:
                control_observer(self)
            if self._last_status.terminated or self._last_status.truncated:
                break
            if (
                step_observer is not None
                and self._physics_steps % PHYSICS_STEPS_PER_VIEWER_SYNC == 0
                and not step_observer(self)
            ):
                self.truncate("viewer_closed")
                break
        observation = self.observe(camera_wait_observer)
        status = self._last_status
        return StepResult(
            observation=observation,
            reward=1.0 if status.success else 0.0,
            terminated=status.terminated,
            truncated=status.truncated,
            info=self.info(),
        )

    def counters(self) -> RuntimeCounters:
        return RuntimeCounters(
            physics_steps=self._physics_steps,
            ik_solves=self._ik_solves,
            target_rate_limits=self._target_rate_limits,
            gripper_clamps=self._gripper_clamps,
            ik_faults=self._ik_faults,
            grasp_assists=self.grasp.activations,
        )

    def info(self) -> dict[str, object]:
        if self.manifest is None or self._last_status is None:
            raise RuntimeError("reset() must be called first")
        tool_positions = {side: transform[:3, 3].tolist() for side, transform in self.tool_transforms().items()}
        object_positions = (
            {name: self.data.xpos[body_id].tolist() for name, body_id in self.task.object_body_ids.items()}
            if self.data is not None and self.task is not None
            else {}
        )
        joint_margins = None
        minimum_lift_margin_m = None
        minimum_arm_margin_rad = None
        minimum_base_translation_margin_m = None
        minimum_base_yaw_margin_rad = None
        base_pose = None
        base_velocity = None
        if self.controller is not None and self.data is not None:
            measured = self.controller.measured_configuration(self.data)
            margins = np.minimum(
                measured - self.controller.model.lowerPositionLimit,
                self.controller.model.upperPositionLimit - measured,
            )
            if self.controller.base_mode == "planar":
                coordinate_names = PLANAR_CONTROLLED_COORDINATES
                units = ("m", "m", "rad", "m", *(["rad"] * 14))
                minimum_base_translation_margin_m = (float(np.min(margins[:2]))
                    if np.all(np.isfinite(margins[:2])) else None)
                minimum_base_yaw_margin_rad = float(margins[2]) if np.isfinite(margins[2]) else None
                minimum_lift_margin_m = float(margins[3])
                minimum_arm_margin_rad = float(np.min(margins[4:]))
                base_pose = self.controller.base_pose(self.data).tolist()
                base_velocity = self.controller.base_velocity(self.data).tolist()
            else:
                coordinate_names = CONTROLLED_JOINTS
                units = ("m", *(["rad"] * 14))
                minimum_lift_margin_m = float(margins[0])
                minimum_arm_margin_rad = float(np.min(margins[1:]))
            joint_margins = {
                name: {"value": float(value) if np.isfinite(value) else None, "unit": unit}
                for name, value, unit in zip(coordinate_names, margins, units, strict=True)
            }
        return {
            "task": self.manifest.task,
            "profile": self.manifest.profile,
            "motion_profile_id": self.controller_config.motion_profile_id,
            "motion_profile_sha256": self.controller_config.motion_profile_sha256,
            "lift_velocity_damping_scale": self.controller_config.lift_velocity_damping_scale,
            "seed": self.manifest.seed,
            "base_mode": self.manifest.base_mode,
            "base_workspace": dataclasses.asdict(self.controller_config.base),
            "scene_sha256": self.manifest.scene_sha256,
            "sensor": dict(self._last_camera_metadata),
            "task_status": self._last_status.to_dict(),
            "counters": dataclasses.asdict(self.counters()),
            "last_ik_fault": self._last_ik_fault,
            "ik_diagnostics": (
                dataclasses.asdict(self._last_ik_diagnostics) if self._last_ik_diagnostics is not None else None
            ),
            "self_collision_oracle": [
                dataclasses.asdict(diagnostic) for diagnostic in self._last_self_collision_oracle
            ],
            "minimum_self_collision_oracle_m": dict(self._minimum_self_collision_oracle_m),
            "self_collision_oracle_enabled": self.self_collision_oracle_enabled,
            "active_grasps": dict(self.grasp.active),
            "physical_grasps": dict(self.grasp_contacts.active),
            "held_grasps": self.held_grasps(),
            "grasp_assist_enabled": self.grasp.enabled,
            "grasp_assist_min_lift_m": self.grasp_assist_min_lift_m,
            "grasp_assist_kinematic": self.grasp.kinematic,
            "tool_positions": tool_positions,
            "object_positions": object_positions,
            "minimum_lift_planning_margin_m": minimum_lift_margin_m,
            "minimum_arm_planning_margin_rad": minimum_arm_margin_rad,
            "minimum_base_translation_planning_margin_m": minimum_base_translation_margin_m,
            "minimum_base_yaw_planning_margin_rad": minimum_base_yaw_margin_rad,
            "base_pose_world_xyyaw": base_pose,
            "base_velocity_world_xyyaw": base_velocity,
            "planning_joint_margins": joint_margins,
        }

    def close(self) -> None:
        self._invalidate_episode()
        if self._temporary_run is not None:
            self._temporary_run.cleanup()
            self._temporary_run = None

    def __enter__(self) -> PolicySimCore:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()
