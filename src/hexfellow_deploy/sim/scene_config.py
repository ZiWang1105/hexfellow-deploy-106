"""Load geometry and initial conditions without accepting task or prompt fields."""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
import re

import yaml


@dataclass(frozen=True)
class SceneConfig:
    path: Path
    data: dict

    def resolve(self, reference: str) -> Path:
        path = Path(reference).expanduser()
        return (self.path.parent / path).resolve()

    @property
    def base_actuation(self):
        # Preserve the plant limits of older scene YAMLs when replaying them.
        return self.data["robot"].get("base_actuation", {
            "linear_velocity_limit_m_s": .1, "angular_velocity_limit_rad_s": .25})


def _keys(value, required, optional=(), label="configuration"):
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a mapping")
    missing = set(required) - value.keys()
    unknown = value.keys() - set(required) - set(optional)
    if missing or unknown:
        raise ValueError(f"{label}: missing keys {sorted(missing)}, unknown keys {sorted(unknown)}")


def _number(value, label):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{label} must be a finite number")
    return float(value)


def _vector(value, length, label, *, positive=False):
    if not isinstance(value, list) or len(value) != length:
        raise ValueError(f"{label} must contain {length} numbers")
    numbers = [_number(v, label) for v in value]
    if positive and min(numbers) <= 0:
        raise ValueError(f"{label} must be positive")
    return numbers


def _name(value):
    if not isinstance(value, str) or not re.fullmatch(r"[a-zA-Z][a-zA-Z0-9_]*", value):
        raise ValueError("Entity and geometry names must be simple identifiers")


def load_scene(path: str | Path) -> SceneConfig:
    path = Path(path).resolve()
    data = yaml.safe_load(path.read_text())
    _keys(data, ["schema_version", "physics", "robot", "camera", "floor", "lights", "entities"],
          ["assemblies"])
    if type(data["schema_version"]) is not int or data["schema_version"] != 1:
        raise ValueError("Unsupported scene schema_version")
    physics = data["physics"]
    _keys(physics, ["timestep_s", "gravity_m_s2"], label="physics")
    if _number(physics["timestep_s"], "timestep_s") <= 0:
        raise ValueError("timestep_s must be positive")
    _vector(physics["gravity_m_s2"], 3, "gravity")
    robot = data["robot"]
    _keys(robot, ["source_checkout", "source_commit", "urdf", "urdf_sha256",
                  "initial_base_xyyaw", "base_bounds", "initial_joints"],
          ["assembly", "base_actuation"], label="robot")
    if "base_actuation" in robot:
        _keys(robot["base_actuation"], ["linear_velocity_limit_m_s", "angular_velocity_limit_rad_s"],
              label="base actuation")
        for name, value in robot["base_actuation"].items():
            if _number(value, name) <= 0:
                raise ValueError(f"{name} must be positive")
    pose = _vector(robot["initial_base_xyyaw"], 3, "base pose")
    _keys(robot["base_bounds"], ["x_m", "y_m", "yaw_rad"], label="base bounds")
    for value, axis in zip(pose, ["x_m", "y_m", "yaw_rad"]):
        bounds = _vector(robot["base_bounds"][axis], 2, axis)
        if not bounds[0] < 0 < bounds[1] or not bounds[0] <= value <= bounds[1]:
            raise ValueError(f"{axis} must straddle zero and contain the initial position")
    if not isinstance(robot["initial_joints"], dict):
        raise ValueError("initial_joints must be a mapping")
    for name, value in robot["initial_joints"].items():
        _name(name)
        _number(value, name)
    camera = data["camera"]
    _keys(camera, ["calibration", "calibration_sha256", "calibration_resolution_wh",
                   "image_resolution_wh"], ["pose_source", "depth_range_m"], label="camera")
    if camera.get("pose_source", "calibration") not in {"calibration", "robot_optical_frame"}:
        raise ValueError("Unknown camera pose_source")
    depth_range = _vector(camera.get("depth_range_m", [.2, 8.]), 2, "depth range", positive=True)
    if not depth_range[0] < depth_range[1] <= 65.535:
        raise ValueError("Depth range must increase and fit uint16 millimeters")
    for key in ["calibration_resolution_wh", "image_resolution_wh"]:
        _vector(camera[key], 2, key, positive=True)
        if any(type(value) is not int for value in camera[key]):
            raise ValueError(f"{key} must contain integers")
    _keys(data["floor"], ["half_size_m", "rgba"], label="floor")
    _vector(data["floor"]["half_size_m"], 3, "floor size", positive=True)
    _vector(data["floor"]["rgba"], 4, "floor rgba")
    if not isinstance(data["lights"], list):
        raise ValueError("lights must be a list")
    for light in data["lights"]:
        _keys(light, ["position_m", "direction", "diffuse", "directional"], label="light")
        _vector(light["position_m"], 3, "light position")
        direction = _vector(light["direction"], 3, "light direction")
        if sum(v*v for v in direction) < 1e-12:
            raise ValueError("Light direction must be nonzero")
        _vector(light["diffuse"], 3, "light diffuse")
        if type(light["directional"]) is not bool:
            raise ValueError("directional must be a boolean")
    if not isinstance(data["entities"], list):
        raise ValueError("entities must be a list")
    names = set()
    for entity in data["entities"]:
        _keys(entity, ["name", "static", "position_m", "quaternion_xyzw"],
              ["boxes", "mesh_asset"], label="entity")
        _name(entity["name"])
        if entity["name"] in names:
            raise ValueError("Duplicate entity name")
        names.add(entity["name"])
        if type(entity["static"]) is not bool:
            raise ValueError("static must be a boolean")
        _vector(entity["position_m"], 3, "entity position")
        quat = _vector(entity["quaternion_xyzw"], 4, "entity quaternion")
        if abs(sum(v*v for v in quat) - 1) > 1e-6:
            raise ValueError("Entity quaternion must have unit norm")
        if ("boxes" in entity) == ("mesh_asset" in entity):
            raise ValueError("Entity needs exactly one of boxes or mesh_asset")
        if "boxes" in entity:
            if not entity["static"]:
                raise ValueError("Dynamic primitive bodies are not supported in scene schema v1")
            if not isinstance(entity["boxes"], list) or not entity["boxes"]:
                raise ValueError("boxes must be a nonempty list")
            box_names = set()
            for box in entity["boxes"]:
                _keys(box, ["name", "position_m", "size_m", "rgba"], label="box")
                _name(box["name"])
                if box["name"] in box_names:
                    raise ValueError("Duplicate box name within entity")
                box_names.add(box["name"])
                _vector(box["position_m"], 3, "box position")
                _vector(box["size_m"], 3, "box full size", positive=True)
                _vector(box["rgba"], 4, "box rgba")
        else:
            asset = entity["mesh_asset"]
            _keys(asset, ["mjcf", "scale", "mass_kg"], label="mesh asset")
            for key in ["scale", "mass_kg"]:
                if _number(asset[key], key) <= 0:
                    raise ValueError(f"{key} must be positive")
    if not isinstance(data.get("assemblies", []), list):
        raise ValueError("assemblies must be a list")
    for assembly in data.get("assemblies", []):
        _keys(assembly, ["name", "mjcf", "sha256", "position_m", "quaternion_xyzw"],
              ["initial_joints"], label="passive assembly")
        _name(assembly["name"])
        if assembly["name"] in names:
            raise ValueError("Duplicate entity/assembly name")
        names.add(assembly["name"])
        if not isinstance(assembly["mjcf"], str) or not re.fullmatch(r"[0-9a-f]{64}", assembly["sha256"]):
            raise ValueError("Assembly requires an MJCF path and SHA256")
        _vector(assembly["position_m"], 3, "assembly position")
        quaternion = _vector(assembly["quaternion_xyzw"], 4, "assembly quaternion")
        if abs(sum(v*v for v in quaternion)-1) > 1e-6:
            raise ValueError("Assembly quaternion must have unit norm")
        if not isinstance(assembly.get("initial_joints", {}), dict):
            raise ValueError("Assembly initial_joints must be a mapping")
        for name, value in assembly.get("initial_joints", {}).items():
            _name(name)
            _number(value, name)
    return SceneConfig(path, data)
