"""Deterministic task-scene generation and manifests."""

from __future__ import annotations

import dataclasses
import hashlib
import json
import xml.etree.ElementTree as ET
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Literal

import numpy as np
import yaml

from ..control.robot import BaseMode, PlanarBaseConfig
from ..paths import CONFIG_DIR
from .builder import build_tabletop_scene
from .catalog import OBJECT_CATALOG
from .layouts import LAYOUT_SELECTORS
from .layouts import select_lift_object_positions as select_lift_object_positions
from .layouts import select_open_table_object_positions as select_open_table_object_positions
from .layouts import select_organize_object_positions as select_organize_object_positions

DEFAULT_CAMERA_CALIBRATION = CONFIG_DIR / "das_wrist_cameras.json"
PHYSICS_TIMESTEP = 1.0 / 600.0


@dataclasses.dataclass(frozen=True)
class TaskDefinition:
    config_path: Path
    schema_version: int
    layout: str
    minimum_pool_size: int
    success_kind: Literal["lift", "containment"]
    include_box: bool = False
    include_trashcan: bool = False


DEFAULT_TASK_NAME = "lift_two_objects"
TASK_DEFINITIONS = {
    "lift_two_objects": TaskDefinition(
        CONFIG_DIR / "tasks" / "lift_two_objects.yaml",
        2,
        "bimanual_lateral",
        4,
        "lift",
    ),
    "organize_rigid_objects": TaskDefinition(
        CONFIG_DIR / "tasks" / "organize_rigid_objects.yaml",
        3,
        "organizer_lateral",
        8,
        "containment",
        include_box=True,
    ),
    "throw_away_objects": TaskDefinition(
        CONFIG_DIR / "tasks" / "throw_away_objects.yaml",
        2,
        "open_table_cross",
        4,
        "containment",
        include_trashcan=True,
    ),
}
TASK_CONFIG_PATHS = {name: definition.config_path for name, definition in TASK_DEFINITIONS.items()}
DEFAULT_TASK_CONFIG = TASK_CONFIG_PATHS[DEFAULT_TASK_NAME]


@dataclasses.dataclass(frozen=True)
class SuccessConfig:
    required_objects: int
    inset_m: float
    stable_seconds: float
    max_linear_velocity_m_s: float
    max_angular_velocity_rad_s: float


@dataclasses.dataclass(frozen=True)
class LiftSuccessConfig:
    lift_height_delta_m: float
    simultaneous_hold_seconds: float


@dataclasses.dataclass(frozen=True)
class TaskProfile:
    name: str
    seeds: tuple[int, ...]
    min_objects: int
    max_objects: int
    timeout_seconds: float
    sensor_mode: str
    object_pool: tuple[str, ...]


@dataclasses.dataclass(frozen=True)
class TaskConfig:
    schema_version: int
    name: str
    prompt: str
    layout: str
    include_box: bool
    include_trashcan: bool
    object_pool: tuple[str, ...]
    profiles: dict[str, TaskProfile]
    success: SuccessConfig | LiftSuccessConfig


@dataclasses.dataclass(frozen=True)
class SceneManifest:
    schema_version: int
    task: str
    profile: str
    seed: int
    prompt: str
    layout: str
    base_mode: BaseMode
    object_ids: tuple[str, ...]
    object_poses: dict[str, tuple[float, float, float]]
    timestep: float
    dynamic_table: bool
    dynamic_box: bool
    include_box: bool
    include_trashcan: bool
    scene_path: str
    scene_sha256: str
    task_config_sha256: str
    camera_calibration_sha256: str
    sensor_mode: str
    timeout_seconds: float
    required_objects: int
    appearance_seed: int | None
    base_workspace: dict | None = None
    robot_initial_qpos: dict[str, float] | None = None

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def available_tasks() -> tuple[str, ...]:
    return tuple(TASK_CONFIG_PATHS)


def load_task_config(path: Path = DEFAULT_TASK_CONFIG) -> TaskConfig:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"task config must be a mapping: {path}")
    task_name = str(raw.get("name"))
    definition = TASK_DEFINITIONS.get(task_name)
    if definition is None or raw.get("schema_version") != definition.schema_version:
        raise ValueError(f"unsupported task config in {path}")
    pool = tuple(str(item) for item in raw["object_pool"])
    minimum_pool_size = definition.minimum_pool_size
    if len(pool) != len(set(pool)) or len(pool) < minimum_pool_size:
        raise ValueError(f"{task_name} object_pool must contain at least {minimum_pool_size} unique IDs")
    unknown_assets = sorted(set(pool) - set(OBJECT_CATALOG))
    if unknown_assets:
        raise ValueError(f"unknown object assets in {path}: {unknown_assets}")
    profiles_raw = raw.get("profiles")
    if not isinstance(profiles_raw, dict) or not profiles_raw:
        raise ValueError(f"{task_name} must define at least one profile")
    profiles: dict[str, TaskProfile] = {}
    for name, value in profiles_raw.items():
        if not isinstance(value, dict):
            raise ValueError(f"profile {name} must be a mapping")
        profile_pool = tuple(str(item) for item in value.get("object_pool", pool))
        unknown = sorted(set(profile_pool) - set(pool))
        if not profile_pool or len(profile_pool) != len(set(profile_pool)) or unknown:
            raise ValueError(f"invalid object pool for profile {name}: unknown={unknown}")
        minimum, maximum = map(int, value["object_count"])
        if not 1 <= minimum <= maximum <= len(profile_pool):
            raise ValueError(f"invalid object count for profile {name}")
        sensor_mode = str(value["sensor_mode"])
        if sensor_mode not in {"fast", "benchmark", "disabled"}:
            raise ValueError(f"invalid sensor mode {sensor_mode!r}")
        seeds_raw = value.get("seeds")
        if not isinstance(seeds_raw, list) or any(
            not isinstance(seed, int) or isinstance(seed, bool) for seed in seeds_raw
        ):
            raise ValueError(f"profile {name} seeds must be integers")
        seeds = tuple(seeds_raw)
        timeout_seconds = float(value["timeout_seconds"])
        if not seeds or any(seed < 0 for seed in seeds) or len(set(seeds)) != len(seeds):
            raise ValueError(f"profile {name} seeds must be non-empty, unique, and non-negative")
        if not np.isfinite(timeout_seconds) or timeout_seconds <= 0:
            raise ValueError(f"profile {name} timeout must be finite and positive")
        profiles[name] = TaskProfile(
            name=name,
            seeds=seeds,
            min_objects=minimum,
            max_objects=maximum,
            timeout_seconds=timeout_seconds,
            sensor_mode=sensor_mode,
            object_pool=profile_pool,
        )
    success_raw = dict(raw["success"])
    if definition.success_kind == "containment":
        required_objects = int(success_raw.pop("required_objects"))
        if required_objects < 1:
            raise ValueError("success.required_objects must be positive")
        success: SuccessConfig | LiftSuccessConfig = SuccessConfig(
            required_objects=required_objects,
            **{key: float(value) for key, value in success_raw.items()},
        )
        thresholds = (
            success.stable_seconds,
            success.max_linear_velocity_m_s,
            success.max_angular_velocity_rad_s,
        )
        if (
            not np.isfinite(success.inset_m)
            or success.inset_m < 0.0
            or any(not np.isfinite(value) or value <= 0.0 for value in thresholds)
        ):
            raise ValueError("containment inset must be non-negative and stability thresholds positive")
    else:
        success = LiftSuccessConfig(
            lift_height_delta_m=float(success_raw["lift_height_delta_m"]),
            simultaneous_hold_seconds=float(success_raw["simultaneous_hold_seconds"]),
        )
        if any(
            not np.isfinite(value) or value <= 0
            for value in (success.lift_height_delta_m, success.simultaneous_hold_seconds)
        ):
            raise ValueError("lift success thresholds must be positive")
    scene_raw = dict(raw.get("scene", {}))
    layout = str(scene_raw.get("layout", ""))
    if layout != definition.layout:
        raise ValueError(f"{task_name} scene.layout must be {definition.layout!r}, got {layout!r}")
    prompt = raw.get("prompt")
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError(f"{task_name} prompt must be a non-empty string")

    def scene_flag(name: str, default: bool) -> bool:
        value = scene_raw.get(name, default)
        if not isinstance(value, bool):
            raise ValueError(f"scene.{name} must be a boolean")
        return value

    return TaskConfig(
        schema_version=int(raw["schema_version"]),
        name=task_name,
        prompt=prompt,
        layout=layout,
        include_box=scene_flag("include_box", definition.include_box),
        include_trashcan=scene_flag("include_trashcan", definition.include_trashcan),
        object_pool=pool,
        profiles=profiles,
        success=success,
    )


def select_objects(config: TaskConfig, profile: TaskProfile, seed: int) -> tuple[str, ...]:
    """Select identities independently from the legacy 3--6 object sampler."""
    if seed < 0:
        raise ValueError("seed must be non-negative")
    rng = np.random.default_rng(np.random.SeedSequence([seed, 0x4F424A]))
    count = int(rng.integers(profile.min_objects, profile.max_objects + 1))
    indices = rng.choice(len(profile.object_pool), size=count, replace=False)
    return tuple(profile.object_pool[int(index)] for index in indices)


def _read_object_poses(scene_path: Path) -> dict[str, tuple[float, float, float]]:
    root = ET.parse(scene_path).getroot()
    text = root.find("./custom/text[@name='object_poses']")
    if text is None or text.get("data") is None:
        raise RuntimeError(f"generated scene {scene_path} has no object pose metadata")
    raw = json.loads(text.get("data"))
    return {name: tuple(map(float, pose)) for name, pose in raw.items()}


class SceneFactory:
    """Build one immutable MJCF and manifest per deterministic episode."""

    def __init__(
        self,
        task_config_path: Path | None = None,
        camera_calibration_path: Path = DEFAULT_CAMERA_CALIBRATION,
        *,
        task_name: str = DEFAULT_TASK_NAME,
        base_mode: BaseMode = "planar",
        preserve_fixed_inertias: bool = False,
        base_config: PlanarBaseConfig | None = None,
    ) -> None:
        if task_config_path is None:
            if task_name not in TASK_CONFIG_PATHS:
                raise ValueError(f"unknown task {task_name!r}; choose {list(available_tasks())}")
            task_config_path = TASK_CONFIG_PATHS[task_name]
        self.task_config_path = Path(task_config_path).resolve()
        self.camera_calibration_path = Path(camera_calibration_path).resolve()
        self.config = load_task_config(self.task_config_path)
        if base_mode not in {"fixed", "planar"}:
            raise ValueError(f"unknown base mode {base_mode!r}")
        self.base_mode = base_mode
        self.preserve_fixed_inertias = preserve_fixed_inertias
        self.base_config = base_config

    def build(
        self,
        output_dir: Path,
        *,
        profile_name: str,
        seed: int,
        object_ids_override: Sequence[str] | None = None,
        object_positions_override: Mapping[str, tuple[float, float, float]] | None = None,
        required_objects_override: int | None = None,
        appearance_seed: int | None = None,
        robot_initial_qpos: Mapping[str, float] | None = None,
    ) -> SceneManifest:
        """Build a deterministic scene, optionally pinning its objects and poses.

        The overrides are intentionally narrow: batch generation can balance a
        known object pair and sample a pre-validated layout without duplicating
        scene construction.  Ordinary task resets keep the seeded profile
        behaviour unchanged.
        """
        if profile_name not in self.config.profiles:
            raise ValueError(f"unknown profile {profile_name!r}; choose {sorted(self.config.profiles)}")
        profile = self.config.profiles[profile_name]
        object_ids = (
            select_objects(self.config, profile, seed)
            if object_ids_override is None
            else tuple(str(value) for value in object_ids_override)
        )
        if len(object_ids) != len(set(object_ids)):
            raise ValueError("scene object override must not contain duplicates")
        if not profile.min_objects <= len(object_ids) <= profile.max_objects:
            raise ValueError(
                f"scene object override count {len(object_ids)} is outside profile range "
                f"[{profile.min_objects}, {profile.max_objects}]"
            )
        unknown = sorted(set(object_ids) - set(profile.object_pool))
        if unknown:
            raise ValueError(f"scene object override is outside profile pool: {unknown}")
        if appearance_seed is not None and (not isinstance(appearance_seed, int) or appearance_seed < 0):
            raise ValueError("appearance seed must be a non-negative integer")
        if object_positions_override is None:
            object_positions = LAYOUT_SELECTORS[self.config.layout](object_ids, seed)
        else:
            object_positions = {
                str(name): tuple(float(value) for value in pose) for name, pose in object_positions_override.items()
            }
            if set(object_positions) != set(object_ids):
                raise ValueError("scene object position override must contain exactly the selected objects")
            if any(len(pose) != 3 for pose in object_positions.values()):
                raise ValueError("scene object position override entries must be (x, y, yaw)")
        output_dir = Path(output_dir).resolve()
        output_dir.mkdir(parents=True, exist_ok=True)
        scene_path = output_dir / "scene.xml"
        build_tabletop_scene(
            output_path=scene_path,
            object_ids=object_ids,
            object_positions=object_positions,
            layout_seed=seed,
            include_box=self.config.include_box,
            include_trashcan=self.config.include_trashcan,
            camera_calibration=self.camera_calibration_path,
            timestep=PHYSICS_TIMESTEP,
            dynamic_table=False,
            dynamic_box=False,
            base_mode=self.base_mode,
            appearance_seed=appearance_seed,
            preserve_fixed_inertias=self.preserve_fixed_inertias,
            robot_initial_qpos=robot_initial_qpos,
            **({"base_config": self.base_config} if self.base_config is not None else {}),
        )
        if isinstance(self.config.success, SuccessConfig):
            if required_objects_override is None:
                required_objects = min(len(object_ids), self.config.success.required_objects)
            else:
                required_objects = int(required_objects_override)
                if not 1 <= required_objects <= len(object_ids):
                    raise ValueError("required object override must be within the selected object count")
        elif required_objects_override is not None:
            raise ValueError("required object override is only valid for containment tasks")
        else:
            required_objects = 2
        manifest = SceneManifest(
            schema_version=self.config.schema_version,
            task=self.config.name,
            profile=profile_name,
            seed=seed,
            prompt=self.config.prompt,
            layout=self.config.layout,
            base_mode=self.base_mode,
            base_workspace=dataclasses.asdict(self.base_config) if self.base_config is not None else None,
            object_ids=object_ids,
            object_poses=_read_object_poses(scene_path),
            timestep=PHYSICS_TIMESTEP,
            dynamic_table=False,
            dynamic_box=False,
            include_box=self.config.include_box,
            include_trashcan=self.config.include_trashcan,
            scene_path=str(scene_path),
            scene_sha256=_file_sha256(scene_path),
            task_config_sha256=_file_sha256(self.task_config_path),
            camera_calibration_sha256=_file_sha256(self.camera_calibration_path),
            sensor_mode=profile.sensor_mode,
            timeout_seconds=profile.timeout_seconds,
            required_objects=required_objects,
            appearance_seed=appearance_seed,
            robot_initial_qpos=None if robot_initial_qpos is None else dict(robot_initial_qpos),
        )
        (output_dir / "scene_manifest.json").write_text(
            json.dumps(manifest.to_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        return manifest
