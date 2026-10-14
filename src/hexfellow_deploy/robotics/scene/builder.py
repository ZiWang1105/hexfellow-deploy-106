#!/usr/bin/env python3
"""Build a configurable interactive Maver X4 tabletop scene.

Objects and their tabletop poses may be selected explicitly or randomized.
Every selected object, the table, the central box, and the floor trash can are
dynamic and can be manipulated in MuJoCo's interactive viewer.
"""

from __future__ import annotations

import copy
import json
import math
import os
import secrets
import tempfile
import xml.etree.ElementTree as ET
from collections.abc import Mapping, Sequence
from pathlib import Path

import mujoco
import numpy as np

from ..camera.calibration import (
    DEFAULT_CALIBRATION as DEFAULT_CAMERA_CALIBRATION,
)
from ..camera.calibration import (
    axis_fov_degrees,
    link7_from_das_base,
    link7_from_mujoco_camera,
    link7_from_optical,
    load_calibration,
    matrix_to_quaternion_wxyz,
)
from ..camera.renderer import FACE_BASES
from ..control.robot import (
    BASE_JOINTS,
    CONTROLLED_JOINTS,
    DEFAULT_PLANAR_BASE_CONFIG,
    LEFT_IK_TARGET,
    LEFT_POLICY_EEF_SITE,
    LEFT_TCP_SITE,
    READY_QPOS,
    RIGHT_IK_TARGET,
    RIGHT_POLICY_EEF_SITE,
    RIGHT_TCP_SITE,
    BaseMode,
    PlanarBaseConfig,
)
from ..control.robot import (
    build_scene as build_robot_scene,
)
from ..control.robot_model import load_robot_model
from ..paths import ASSET_DIR, SCENE_DIR
from .catalog import OBJECT_CATALOG, ObjectSpec, normalize_object_tokens, resolve_object_ids
from .constants import OBJECT_GAP as OBJECT_GAP
from .constants import (
    ORGANIZE_BOX_INNER_HALF_EXTENTS,
    ORGANIZE_BOX_OUTER_HALF_EXTENTS,
    ORGANIZE_BOX_WALL_HEIGHT,
    ORGANIZE_GOAL_SITE,
    TABLE_CENTER,
    TABLE_TOP_Z,
    TRASHCAN_GOAL_SITE,
    TRASHCAN_HEIGHT,
    TRASHCAN_INNER_RADIUS,
    TRASHCAN_OUTER_RADIUS,
    TRASHCAN_POSITION,
)
from .constants import TABLE_USABLE_RADIUS as TABLE_USABLE_RADIUS
from .layouts import layout_objects

DEFAULT_OUTPUT = SCENE_DIR / "maver_x4_tabletop_scene.xml"


def _values(values: np.ndarray | list[float] | tuple[float, ...]) -> str:
    return " ".join(f"{float(value):.10g}" for value in values)


def _quat_z(yaw: float) -> str:
    return _values((math.cos(yaw / 2), 0, 0, math.sin(yaw / 2)))


def _asset_path(path: Path, output_path: Path) -> str:
    """Return a portable asset path relative to the generated scene."""
    return Path(os.path.relpath(path.resolve(), output_path.parent)).as_posix()


def _rebase_asset_files(asset: ET.Element, source_scene: Path, output_path: Path) -> None:
    """Keep inherited robot assets valid when writing to another directory."""
    for element in asset:
        file_name = element.get("file")
        if file_name:
            element.set("file", _asset_path(source_scene.parent / file_name, output_path))


def _box_inertia(mass: float, size: tuple[float, float, float]) -> tuple[float, float, float]:
    x, y, z = size
    return (
        mass * (y * y + z * z) / 12,
        mass * (x * x + z * z) / 12,
        mass * (x * x + y * y) / 12,
    )


def _look_at_axes(position: np.ndarray, target: np.ndarray, up: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return camera x/y axes for a camera looking at target."""
    forward = target - position
    forward /= np.linalg.norm(forward)
    z_axis = -forward
    x_axis = np.cross(up, z_axis)
    x_axis /= np.linalg.norm(x_axis)
    y_axis = np.cross(z_axis, x_axis)
    return x_axis, y_axis


def _look_at_xyaxes(position: np.ndarray, target: np.ndarray, up: np.ndarray) -> str:
    x_axis, y_axis = _look_at_axes(position, target, up)
    return _values(np.concatenate((x_axis, y_axis)))


def _mesh_bounds(path: Path) -> tuple[np.ndarray, np.ndarray]:
    minimum = np.full(3, math.inf)
    maximum = np.full(3, -math.inf)
    with path.open(encoding="utf-8", errors="ignore") as obj_file:
        for line in obj_file:
            if line.startswith("v "):
                vertex = np.array([float(value) for value in line.split()[1:4]])
                minimum = np.minimum(minimum, vertex)
                maximum = np.maximum(maximum, vertex)
    if not np.all(np.isfinite(minimum)):
        raise RuntimeError(f"No vertices found in {path}")
    return minimum, maximum


def _ensure_section(root: ET.Element, tag: str, after: str | None = None) -> ET.Element:
    section = root.find(tag)
    if section is not None:
        return section
    section = ET.Element(tag)
    if after is None or root.find(after) is None:
        root.append(section)
    else:
        root.insert(list(root).index(root.find(after)) + 1, section)
    return section


def _configure_global_scene(root: ET.Element, timestep: float = 0.002) -> tuple[ET.Element, ET.Element]:
    if timestep <= 0 or not math.isfinite(timestep):
        raise ValueError("MuJoCo timestep must be finite and positive")
    root.set("model", "maver_x4_tabletop_scene")

    option = _ensure_section(root, "option", "compiler")
    option.attrib.clear()
    option.attrib.update(
        {
            "timestep": f"{timestep:.12g}",
            "gravity": "0 0 -9.81",
            "integrator": "implicitfast",
            "solver": "Newton",
            "cone": "elliptic",
            "iterations": "100",
            "noslip_iterations": "5",
        }
    )

    visual = _ensure_section(root, "visual", "option")
    visual.clear()
    # Native GenRobot frames are 1600x1300.  MuJoCo rejects a Renderer larger
    # than this offscreen framebuffer even when named cameras declare a larger
    # resolution.
    ET.SubElement(visual, "global", {"offwidth": "1600", "offheight": "1300"})
    ET.SubElement(
        visual,
        "headlight",
        {"ambient": "0.16 0.16 0.16", "diffuse": "0 0 0", "specular": "0 0 0"},
    )
    ET.SubElement(visual, "rgba", {"haze": "0.7 0.75 0.8 1"})
    ET.SubElement(visual, "quality", {"shadowsize": "4096", "offsamples": "4"})
    ET.SubElement(visual, "map", {"zfar": "8", "shadowclip": "3", "shadowscale": "0.8"})

    default = _ensure_section(root, "default", "visual")
    default.clear()
    ET.SubElement(default, "joint", {"damping": "0.5", "armature": "0.002"})
    ET.SubElement(
        default,
        "geom",
        {
            "friction": "0.8 0.005 0.0001",
            "solref": "0.006 1.5",
            "solimp": "0.9 0.95 0.001",
            "condim": "4",
        },
    )

    asset = _ensure_section(root, "asset", "default")
    worldbody = root.find("worldbody")
    if worldbody is None:
        raise RuntimeError("Robot scene has no worldbody")
    return asset, worldbody


def _replace_scene_assets(asset: ET.Element, output_path: Path) -> None:
    for element in list(asset):
        if element.get("name") in {"ground_checker", "ground_material"}:
            asset.remove(element)

    additions = [
        ET.Element(
            "texture",
            {
                "name": "sky_gradient",
                "type": "skybox",
                "builtin": "gradient",
                "rgb1": "0.72 0.80 0.88",
                "rgb2": "0.12 0.16 0.22",
                "width": "512",
                "height": "3072",
            },
        ),
        ET.Element(
            "texture",
            {
                "name": "floor_tile",
                "type": "2d",
                "builtin": "checker",
                "rgb1": "0.56 0.58 0.60",
                "rgb2": "0.48 0.50 0.52",
                "mark": "edge",
                "markrgb": "0.34 0.36 0.38",
                "width": "512",
                "height": "512",
            },
        ),
        ET.Element(
            "material",
            {
                "name": "floor_material",
                "texture": "floor_tile",
                "texrepeat": "10 10",
                "texuniform": "true",
                "reflectance": "0.05",
                "roughness": "0.9",
            },
        ),
        ET.Element(
            "texture",
            {
                "name": "wood_texture",
                "type": "2d",
                # MuJoCo's texture loader expects PNG (or its custom binary
                # format), so the downloaded Poly Haven JPG is vendored both
                # in its original form and as a lossless PNG conversion.
                "file": _asset_path(ASSET_DIR / "textures" / "wood_table_diff_1k.png", output_path),
            },
        ),
        ET.Element(
            "material",
            {
                "name": "wood_material",
                "texture": "wood_texture",
                "texrepeat": "2 2",
                "texuniform": "true",
                "reflectance": "0.08",
                "roughness": "0.55",
            },
        ),
        ET.Element(
            "texture",
            {
                "name": "cardboard_texture",
                "type": "2d",
                "builtin": "checker",
                "rgb1": "0.58 0.39 0.22",
                "rgb2": "0.52 0.34 0.19",
                "width": "512",
                "height": "512",
            },
        ),
        ET.Element(
            "material",
            {
                "name": "cardboard_material",
                "texture": "cardboard_texture",
                "texrepeat": "4 4",
                "texuniform": "true",
                "roughness": "0.95",
            },
        ),
        ET.Element("material", {"name": "remote_shell", "rgba": "0.88 0.89 0.87 1", "roughness": "0.65"}),
        ET.Element("material", {"name": "remote_screen", "rgba": "0.12 0.22 0.24 1", "reflectance": "0.18"}),
        ET.Element("material", {"name": "button_dark", "rgba": "0.16 0.17 0.18 1", "roughness": "0.7"}),
        ET.Element("material", {"name": "button_blue", "rgba": "0.18 0.42 0.72 1", "roughness": "0.65"}),
        ET.Element("material", {"name": "button_red", "rgba": "0.75 0.12 0.10 1", "roughness": "0.65"}),
        ET.Element(
            "material",
            {"name": "trashcan_material", "rgba": "0.10 0.17 0.22 1", "roughness": "0.72"},
        ),
    ]
    for element in reversed(additions):
        asset.insert(0, element)


def _configure_robot(worldbody: ET.Element, root: ET.Element) -> None:
    ground = next((g for g in worldbody.findall("geom") if g.get("name") == "ground"), None)
    if ground is None:
        ground = ET.Element("geom", {"name": "ground"})
        worldbody.insert(0, ground)
    ground.attrib.update(
        {
            "type": "plane",
            "pos": "0 0 0",
            "size": "0 0 0.05",
            "material": "floor_material",
            "group": "0",
            "contype": "1",
            "conaffinity": "3",
        }
    )

    for light in list(worldbody.findall("light")):
        worldbody.remove(light)
    worldbody.insert(
        1,
        ET.Element(
            "light",
            {
                "name": "key_light",
                "pos": "0.1 -1.7 2.8",
                "dir": "0.2 0.35 -1",
                "diffuse": "0.72 0.69 0.64",
                "specular": "0.25 0.25 0.25",
                "castshadow": "true",
            },
        ),
    )
    worldbody.insert(
        2,
        ET.Element(
            "light",
            {
                "name": "fill_light",
                "pos": "1.8 1.1 2.2",
                "dir": "-0.45 -0.25 -1",
                "diffuse": "0.38 0.43 0.52",
                "specular": "0.08 0.08 0.08",
                "castshadow": "false",
            },
        ),
    )
    worldbody.insert(
        3,
        ET.Element(
            "light",
            {
                "name": "rim_light",
                "pos": "-1.0 1.5 1.8",
                "dir": "0.35 -0.45 -1",
                "diffuse": "0.22 0.25 0.30",
                "castshadow": "false",
            },
        ),
    )

    lift = worldbody.find(".//body[@name='maver_x4_asm_URDF_link_1']")
    if lift is None:
        raise RuntimeError("Could not locate the robot lift body")
    for body in lift.iter("body"):
        body.set("gravcomp", "1")
    lift.set("gravcomp", "1")
    for geom in lift.iter("geom"):
        geom.set("group", "1")
        geom.set("contype", "2")
        geom.set("conaffinity", "1")
        geom.set("friction", "0.9 0.005 0.0001")
    for geom in worldbody.findall("geom"):
        if geom is not ground:
            geom.set("group", "1")
            geom.set("contype", "0")
            geom.set("conaffinity", "0")

    from ..control.hardware_capabilities import load_hardware_capabilities
    from ..control.robot import ARM_JOINTS

    peak_efforts = dict(zip(ARM_JOINTS, load_hardware_capabilities().peak_torque_nm, strict=True))
    for joint in lift.iter("joint"):
        name = joint.get("name", "")
        if name == "maver_x4_asm_URDF_joint_1":
            joint.set("actuatorfrcrange", "-2000 2000")
            joint.set("damping", "45")
        elif "robot" in name:
            joint.set("actuatorfrcrange", "-40 40")
        elif name in peak_efforts:
            bound = peak_efforts[name]
            joint.set("actuatorfrcrange", f"{-bound:g} {bound:g}")

    actuator = root.find("actuator")
    if actuator is not None:
        for position in actuator.findall("position"):
            joint = position.get("joint", "")
            if joint == "maver_x4_asm_URDF_joint_1":
                position.set("kp", "1200")
            elif "robot" in joint:
                position.set("kp", "300")
            else:
                position.set("kp", "80")


def _apply_appearance_randomization(root: ET.Element, seed: int | None) -> None:
    """Apply small deterministic episode-level lighting/material changes."""

    if seed is None:
        return
    if seed < 0:
        raise ValueError("appearance seed must be non-negative")
    rng = np.random.default_rng(np.random.SeedSequence([seed, 0x41505052]))
    worldbody = root.find("worldbody")
    asset = root.find("asset")
    if worldbody is None or asset is None:
        raise RuntimeError("scene is missing worldbody or asset")
    for light in worldbody.findall("light"):
        diffuse = np.fromstring(light.get("diffuse", "0 0 0"), sep=" ")
        if diffuse.shape != (3,):
            continue
        intensity = float(rng.uniform(0.88, 1.12))
        tint = np.asarray((rng.uniform(0.96, 1.04), rng.uniform(0.96, 1.04), rng.uniform(0.96, 1.04)))
        light.set("diffuse", _values(np.clip(diffuse * intensity * tint, 0.0, 1.0)))
    for material in asset.findall("material"):
        if material.get("name") not in {"wood_material", "cardboard_material"}:
            continue
        roughness = float(material.get("roughness", "0.7"))
        varied = float(np.clip(roughness + rng.uniform(-0.06, 0.06), 0.1, 1.0))
        material.set("roughness", f"{varied:.6g}")


def _add_ik_frames(worldbody: ET.Element, robot_model=None) -> None:
    """Add policy base-link frames, practical TCPs, and draggable IK targets."""
    spec = load_robot_model(robot_model)
    tcp_specs = (
        ("maver_x4_asm_7_urdf_SLDASM_link_7", LEFT_TCP_SITE, "1 0.32 0.08 0.75"),
        ("maver_x4_asm_7_urdf_SLDASM2_link_7", RIGHT_TCP_SITE, "0.05 0.72 1 0.75"),
    )
    for body_name, site_name, rgba in tcp_specs:
        parent = worldbody.find(f".//body[@name='{body_name}']")
        if parent is None:
            raise RuntimeError(f"Could not attach TCP site to {body_name}")
        ET.SubElement(
            parent,
            "site",
            {
                "name": site_name,
                "type": "sphere",
                "pos": _values(spec.tcp_offset_m),
                "size": "0.009",
                "rgba": rgba,
                "group": "4",
            },
        )

    policy_transform = link7_from_das_base() if spec.auxiliary_joints else np.eye(4)
    if not spec.auxiliary_joints:
        policy_transform[:3, 3] = spec.tcp_offset_m
    policy_specs = (
        ("maver_x4_asm_7_urdf_SLDASM_link_7", LEFT_POLICY_EEF_SITE, "1 0.75 0.08 0.55"),
        ("maver_x4_asm_7_urdf_SLDASM2_link_7", RIGHT_POLICY_EEF_SITE, "0.05 0.9 1 0.55"),
    )
    for body_name, site_name, rgba in policy_specs:
        parent = worldbody.find(f".//body[@name='{body_name}']")
        if parent is None:
            raise RuntimeError(f"Could not attach policy EEF site to {body_name}")
        ET.SubElement(
            parent,
            "site",
            {
                "name": site_name,
                "type": "sphere",
                "pos": _values(policy_transform[:3, 3]),
                "quat": _values(matrix_to_quaternion_wxyz(policy_transform[:3, :3])),
                "size": "0.006",
                "rgba": rgba,
                "group": "4",
            },
        )

    target_specs = (
        (LEFT_IK_TARGET, (0.68, 0.24, 1.05), "1 0.32 0.08 0.65"),
        (RIGHT_IK_TARGET, (0.68, -0.24, 1.05), "0.05 0.72 1 0.65"),
    )
    for target_name, position, handle_rgba in target_specs:
        body = ET.SubElement(
            worldbody,
            "body",
            {"name": target_name, "mocap": "true", "pos": _values(position)},
        )
        common = {"contype": "0", "conaffinity": "0", "group": "4"}
        ET.SubElement(
            body,
            "geom",
            {
                "name": f"{target_name}_handle",
                "type": "sphere",
                "size": "0.018",
                "rgba": handle_rgba,
                **common,
            },
        )
        for axis, endpoint, rgba in (
            ("x", "0.075 0 0", "0.95 0.12 0.10 0.85"),
            ("y", "0 0.075 0", "0.12 0.85 0.20 0.85"),
            ("z", "0 0 0.075", "0.10 0.35 1 0.85"),
        ):
            ET.SubElement(
                body,
                "geom",
                {
                    "name": f"{target_name}_{axis}_axis",
                    "type": "capsule",
                    "fromto": f"0 0 0 {endpoint}",
                    "size": "0.004",
                    "rgba": rgba,
                    **common,
                },
            )
    for side in ("left", "right"):
        carrier = ET.SubElement(
            worldbody,
            "body",
            {"name": f"{side}_grasp_carrier", "mocap": "true", "pos": "0 0 -1"},
        )
        ET.SubElement(
            carrier,
            "site",
            {"name": f"{side}_grasp_carrier_site", "size": "0.002", "rgba": "0 0 0 0", "group": "4"},
        )


def _add_table(worldbody: ET.Element, *, dynamic: bool = True) -> None:
    body = ET.SubElement(worldbody, "body", {"name": "table", "pos": _values(TABLE_CENTER)})
    if dynamic:
        ET.SubElement(body, "freejoint", {"name": "table_freejoint"})
    ET.SubElement(body, "inertial", {"pos": "0 0 -0.2261", "mass": "18", "diaginertia": "2.086 2.086 1.171"})
    common = {
        "material": "wood_material",
        "contype": "1",
        "conaffinity": "3",
        "group": "0",
        "friction": "0.85 0.01 0.0002",
    }
    ET.SubElement(
        body, "geom", {"name": "tabletop", "type": "cylinder", "pos": "0 0 0", "size": "0.45 0.025", **common}
    )
    ET.SubElement(
        body,
        "geom",
        {"name": "table_pedestal", "type": "cylinder", "pos": "0 0 -0.3475", "size": "0.08 0.3225", **common},
    )
    ET.SubElement(
        body, "geom", {"name": "table_foot", "type": "cylinder", "pos": "0 0 -0.67", "size": "0.27 0.025", **common}
    )


def _add_open_box(worldbody: ET.Element, *, dynamic: bool = True) -> None:
    mass = 0.35
    outer_x, outer_y = ORGANIZE_BOX_OUTER_HALF_EXTENTS
    inner_x, inner_y = ORGANIZE_BOX_INNER_HALF_EXTENTS
    base_half_height = 0.006
    wall_half_thickness = 0.006
    wall_half_height = (ORGANIZE_BOX_WALL_HEIGHT - 2 * base_half_height) / 2
    wall_center_z = 2 * base_half_height + wall_half_height
    body = ET.SubElement(
        worldbody,
        "body",
        {"name": "open_box", "pos": f"0.75 0 {TABLE_TOP_Z + 0.001:g}"},
    )
    if dynamic:
        ET.SubElement(body, "freejoint", {"name": "open_box_freejoint"})
    ET.SubElement(
        body,
        "inertial",
        {
            "pos": f"0 0 {ORGANIZE_BOX_WALL_HEIGHT / 2:g}",
            "mass": f"{mass:g}",
            "diaginertia": _values(
                _box_inertia(
                    mass,
                    (2 * outer_x, 2 * outer_y, ORGANIZE_BOX_WALL_HEIGHT),
                )
            ),
        },
    )
    # Category 4 collides with category-1 table/objects through conaffinity=1,
    # but not with category-2 robot links. The wide organizer surrounds the
    # policy-ready wrists, so robot contact here would preload the arm servos.
    common = {
        "type": "box",
        "material": "cardboard_material",
        "contype": "4",
        "conaffinity": "1",
        "group": "0",
        "friction": "0.75 0.006 0.0001",
    }
    for name, position, size in (
        ("box_base", _values((0, 0, base_half_height)), _values((outer_x, outer_y, base_half_height))),
        (
            "box_wall_left",
            _values((-inner_x - wall_half_thickness, 0, wall_center_z)),
            _values((wall_half_thickness, outer_y, wall_half_height)),
        ),
        (
            "box_wall_right",
            _values((inner_x + wall_half_thickness, 0, wall_center_z)),
            _values((wall_half_thickness, outer_y, wall_half_height)),
        ),
        (
            "box_wall_front",
            _values((0, -inner_y - wall_half_thickness, wall_center_z)),
            _values((inner_x, wall_half_thickness, wall_half_height)),
        ),
        (
            "box_wall_back",
            _values((0, inner_y + wall_half_thickness, wall_center_z)),
            _values((inner_x, wall_half_thickness, wall_half_height)),
        ),
    ):
        ET.SubElement(
            body,
            "geom",
            {"name": name, "pos": position, "size": size, **common},
        )
    ET.SubElement(
        body,
        "site",
        {
            "name": ORGANIZE_GOAL_SITE,
            "type": "box",
            # Exact physical interior. The task checker applies its configured
            # inset while preserving resting contact with the z=0.012 base.
            # Extend 3 mm into the base to tolerate normal contact solver
            # penetration while keeping the upper semantic volume at the rim.
            "pos": _values((0, 0, (ORGANIZE_BOX_WALL_HEIGHT + 0.009) / 2)),
            "size": _values((inner_x, inner_y, (ORGANIZE_BOX_WALL_HEIGHT - 0.009) / 2)),
            "rgba": "0 0.8 0.2 0",
            "group": "4",
        },
    )


def _add_trashcan(worldbody: ET.Element) -> None:
    """Add a movable, open-top trash can on the robot's right side."""
    mass = 1.5
    body = ET.SubElement(worldbody, "body", {"name": "trashcan", "pos": _values(TRASHCAN_POSITION)})
    ET.SubElement(body, "freejoint", {"name": "trashcan_freejoint"})
    ET.SubElement(
        body,
        "inertial",
        {"pos": f"0 0 {TRASHCAN_HEIGHT * 0.44:g}", "mass": f"{mass:g}", "diaginertia": "0.055 0.055 0.060"},
    )
    common = {
        "material": "trashcan_material",
        "contype": "1",
        "conaffinity": "3",
        "group": "0",
        "friction": "0.8 0.006 0.0001",
    }
    ET.SubElement(
        body,
        "geom",
        {
            "name": "trashcan_base",
            "type": "cylinder",
            "pos": "0 0 0.012",
            "size": f"{TRASHCAN_OUTER_RADIUS:g} 0.012",
            **common,
        },
    )
    wall_count = 24
    wall_radius = (TRASHCAN_OUTER_RADIUS + TRASHCAN_INNER_RADIUS) / 2
    radial_half_width = (TRASHCAN_OUTER_RADIUS - TRASHCAN_INNER_RADIUS) / 2
    tangent_half_width = wall_radius * math.tan(math.pi / wall_count) * 1.05
    for index in range(wall_count):
        angle = index * 2 * math.pi / wall_count
        ET.SubElement(
            body,
            "geom",
            {
                "name": f"trashcan_wall_{index:02d}",
                "type": "box",
                "pos": _values(
                    (
                        wall_radius * math.cos(angle),
                        wall_radius * math.sin(angle),
                        TRASHCAN_HEIGHT / 2,
                    )
                ),
                "quat": _quat_z(angle),
                "size": _values((radial_half_width, tangent_half_width, TRASHCAN_HEIGHT / 2)),
                **common,
            },
        )
    semantic_floor_z = 0.009
    semantic_half_height = (TRASHCAN_HEIGHT - semantic_floor_z) / 2
    ET.SubElement(
        body,
        "site",
        {
            "name": TRASHCAN_GOAL_SITE,
            "type": "cylinder",
            "pos": _values((0.0, 0.0, semantic_floor_z + semantic_half_height)),
            "size": _values((TRASHCAN_INNER_RADIUS, semantic_half_height)),
            "rgba": "0 0.8 0.2 0",
            "group": "4",
        },
    )


def _import_scanned_object(
    asset: ET.Element,
    worldbody: ET.Element,
    spec: ObjectSpec,
    position: tuple[float, float],
    yaw: float,
    output_path: Path,
) -> None:
    prefix = spec.object_id
    if spec.directory is None:
        raise ValueError(f"{prefix} is not a scanned object")
    source_dir = spec.asset_directory
    source_xml = source_dir / "model.xml"
    if not source_xml.exists():
        detail = (
            " Run tools/simulation/import_sketchfab_assets.py after exporting SKETCHFAB_TOKEN."
            if spec.asset_group == "sketchfab"
            else ""
        )
        raise FileNotFoundError(f"Missing scanned object asset: {source_xml}.{detail}")

    source_root = ET.parse(source_xml).getroot()
    source_asset = source_root.find("asset")
    source_body = source_root.find("worldbody/body")
    if source_asset is None or source_body is None:
        raise RuntimeError(f"Malformed scanned object MJCF: {source_xml}")

    name_map: dict[str, str] = {}
    for element in source_asset:
        original_name = element.get("name")
        if original_name:
            name_map[original_name] = f"{prefix}_{original_name}"
    for element in source_asset:
        imported = copy.deepcopy(element)
        if imported.get("name"):
            imported.set("name", name_map[imported.get("name")])
        for reference in ("texture", "material", "mesh"):
            if imported.get(reference) in name_map:
                imported.set(reference, name_map[imported.get(reference)])
        if imported.get("file"):
            imported.set("file", _asset_path(source_dir / imported.get("file"), output_path))
        asset.append(imported)

    source_model = mujoco.MjModel.from_xml_path(str(source_xml))
    source_body_id = mujoco.mj_name2id(source_model, mujoco.mjtObj.mjOBJ_BODY, "model")
    source_mass = float(source_model.body_mass[source_body_id])
    target_mass = spec.mass
    scaled_inertia = source_model.body_inertia[source_body_id] * target_mass / source_mass
    minimum, _maximum = _mesh_bounds(source_dir / "model.obj")
    x, y = position
    body = ET.SubElement(
        worldbody,
        "body",
        {
            "name": prefix,
            "pos": f"{float(x):g} {float(y):g} {TABLE_TOP_Z - minimum[2] + 0.001:g}",
            "quat": _quat_z(yaw),
        },
    )
    ET.SubElement(body, "freejoint", {"name": f"{prefix}_freejoint"})
    ET.SubElement(
        body,
        "inertial",
        {
            "pos": _values(source_model.body_ipos[source_body_id]),
            "quat": _values(source_model.body_iquat[source_body_id]),
            "mass": f"{target_mass:g}",
            "diaginertia": _values(scaled_inertia),
        },
    )
    for index, source_geom in enumerate(source_body.findall("geom")):
        geom = copy.deepcopy(source_geom)
        geom.set("name", f"{prefix}_visual" if index == 0 else f"{prefix}_collision_{index - 1}")
        for reference in ("material", "mesh"):
            if geom.get(reference) in name_map:
                geom.set(reference, name_map[geom.get(reference)])
        if geom.get("group") == "2" or index == 0:
            geom.set("group", "2")
            geom.set("contype", "0")
            geom.set("conaffinity", "0")
        else:
            geom.set("group", "3")
            geom.set("contype", "1")
            geom.set("conaffinity", "3")
            geom.set("friction", "0.8 0.005 0.0001")
            geom.set("rgba", "0.15 0.8 0.25 0.38")
        body.append(geom)
    ET.SubElement(
        body,
        "site",
        {"name": f"{prefix}_grasp_site", "pos": "0 0 0.001", "size": "0.003", "rgba": "0 0 0 0", "group": "4"},
    )


def _add_remote(worldbody: ET.Element, position: tuple[float, float], yaw: float) -> None:
    spec = OBJECT_CATALOG["remote"]
    mass = spec.mass
    x, y = position
    body = ET.SubElement(
        worldbody,
        "body",
        {"name": "remote", "pos": f"{x:g} {y:g} {TABLE_TOP_Z + 0.010:g}", "quat": _quat_z(yaw)},
    )
    ET.SubElement(body, "freejoint", {"name": "remote_freejoint"})
    ET.SubElement(
        body,
        "site",
        {"name": "remote_grasp_site", "pos": "0 0 0.001", "size": "0.003", "rgba": "0 0 0 0", "group": "4"},
    )
    ET.SubElement(
        body,
        "inertial",
        {"mass": f"{mass:g}", "pos": "0 0 0", "diaginertia": _values(_box_inertia(mass, (0.17, 0.052, 0.018)))},
    )
    ET.SubElement(
        body,
        "geom",
        {
            "name": "remote_collision_shell",
            "type": "box",
            "size": "0.085 0.026 0.009",
            "material": "remote_shell",
            "contype": "1",
            "conaffinity": "3",
            "group": "3",
            "friction": "0.72 0.004 0.0001",
        },
    )
    visual = {"contype": "0", "conaffinity": "0", "group": "2"}
    ET.SubElement(
        body,
        "geom",
        {
            "name": "remote_lcd",
            "type": "box",
            "pos": "-0.046 0 0.0094",
            "size": "0.025 0.018 0.0008",
            "material": "remote_screen",
            **visual,
        },
    )
    ET.SubElement(
        body,
        "geom",
        {
            "name": "remote_power",
            "type": "cylinder",
            "pos": "-0.012 0 0.0101",
            "size": "0.006 0.0011",
            "material": "button_red",
            **visual,
        },
    )
    ET.SubElement(
        body,
        "geom",
        {
            "name": "remote_mode",
            "type": "box",
            "pos": "0.010 0 0.0100",
            "size": "0.008 0.011 0.001",
            "material": "button_blue",
            **visual,
        },
    )
    for row in range(4):
        for column in range(3):
            x = 0.030 + row * 0.015
            y = (column - 1) * 0.014
            ET.SubElement(
                body,
                "geom",
                {
                    "name": f"remote_button_{row}_{column}",
                    "type": "cylinder",
                    "pos": f"{x:g} {y:g} 0.0101",
                    "size": "0.0045 0.0011",
                    "material": "button_dark" if row else "button_blue",
                    **visual,
                },
            )


def _add_grasp_equalities(root: ET.Element, selected_ids: Sequence[str]) -> None:
    equality = root.find("equality")
    if equality is None:
        equality = ET.SubElement(root, "equality")
    else:
        equality.clear()
    for side in ("left", "right"):
        for object_id in selected_ids:
            ET.SubElement(
                equality,
                "weld",
                {
                    "name": f"grasp_assist_{side}_{object_id}",
                    "body1": f"{side}_grasp_carrier",
                    "body2": object_id,
                    "relpose": "0 0 0 1 0 0 0",
                    "active": "false",
                    "solref": "0.05 1",
                    "solimp": "0.9 0.95 0.001",
                    "torquescale": "0.03",
                },
            )


def _add_cameras(worldbody: ET.Element, calibration_path: Path = DEFAULT_CAMERA_CALIBRATION,
                 *, wrist_cameras=True) -> None:
    overview_pos = np.array([1.85, -1.75, 1.62])
    overview_target = np.array([0.66, -0.16, 0.62])
    ET.SubElement(
        worldbody,
        "camera",
        {
            "name": "overview",
            "pos": _values(overview_pos),
            "xyaxes": _look_at_xyaxes(overview_pos, overview_target, np.array([0.0, 0.0, 1.0])),
            "fovy": "50",
        },
    )
    trashcan_pos = np.array([1.48, -1.52, 1.08])
    trashcan_target = np.array([0.75, -0.42, 0.48])
    ET.SubElement(
        worldbody,
        "camera",
        {
            "name": "trashcan",
            "pos": _values(trashcan_pos),
            "xyaxes": _look_at_xyaxes(trashcan_pos, trashcan_target, np.array([0.0, 0.0, 1.0])),
            "fovy": "52",
        },
    )

    if not wrist_cameras:
        return

    # Both optical centers are fixed to the actual DAS base transform.  The
    # pinhole cameras are an interactive approximation; the six hidden 90-deg
    # cameras are resampled by das_wrist_camera.py into the GenRobot fisheye.
    calibration = load_calibration(calibration_path)
    wrist_specs = (
        ("left", "maver_x4_asm_7_urdf_SLDASM_link_7"),
        ("right", "maver_x4_asm_7_urdf_SLDASM2_link_7"),
    )
    for side, body_name in wrist_specs:
        parent = worldbody.find(f".//body[@name='{body_name}']")
        if parent is None:
            raise RuntimeError(f"Could not attach camera to {body_name}")
        camera = calibration.for_side(side)
        camera_transform = link7_from_mujoco_camera(camera)
        _, vertical_fov = axis_fov_degrees(camera)
        ET.SubElement(
            parent,
            "camera",
            {
                "name": f"{side}_wrist",
                "pos": _values(camera_transform[:3, 3]),
                "quat": _values(matrix_to_quaternion_wxyz(camera_transform[:3, :3])),
                "fovy": f"{vertical_fov:.10g}",
                "resolution": f"{camera.native_size[0]} {camera.native_size[1]}",
            },
        )
        optical_transform = link7_from_optical(camera)
        for face, (right, up, forward) in FACE_BASES.items():
            face_camera_from_optical = np.column_stack((right, up, -forward))
            rotation_link7_from_face = optical_transform[:3, :3] @ face_camera_from_optical
            ET.SubElement(
                parent,
                "camera",
                {
                    "name": f"{side}_wrist_cube_{face}",
                    "pos": _values(optical_transform[:3, 3]),
                    "quat": _values(matrix_to_quaternion_wxyz(rotation_link7_from_face)),
                    "fovy": "90",
                    "resolution": "1024 1024",
                },
            )


def _add_scene_metadata(
    root: ET.Element,
    selected_ids: Sequence[str],
    layout_seed: int,
    include_box: bool,
    include_trashcan: bool,
    layout: Mapping[str, tuple[tuple[float, float], float]],
    dynamic_table: bool,
    include_table: bool,
) -> None:
    custom = root.find("custom")
    if custom is None:
        custom = ET.Element("custom")
        asset = root.find("asset")
        root.insert(list(root).index(asset) if asset is not None else 0, custom)
    else:
        asset_contract = custom.find("text[@name='mit_asset_contract']")
        custom.clear()
        if asset_contract is not None:
            custom.append(asset_contract)
    ET.SubElement(
        custom,
        "text",
        {"name": "selected_objects", "data": ",".join(selected_ids) if selected_ids else "none"},
    )
    ET.SubElement(custom, "numeric", {"name": "layout_seed", "data": str(layout_seed)})
    ET.SubElement(custom, "numeric", {"name": "include_box", "data": "1" if include_box else "0"})
    ET.SubElement(custom, "numeric", {"name": "include_table", "data": "1" if include_table else "0"})
    poses = {
        object_id: [
            round(layout[object_id][0][0], 8),
            round(layout[object_id][0][1], 8),
            round(layout[object_id][1], 8),
        ]
        for object_id in selected_ids
    }
    ET.SubElement(custom, "text", {"name": "object_poses", "data": json.dumps(poses, separators=(",", ":"))})
    ET.SubElement(
        custom,
        "numeric",
        {"name": "include_trashcan", "data": "1" if include_trashcan else "0"},
    )
    ET.SubElement(
        custom,
        "numeric",
        {"name": "dynamic_table", "data": "1" if include_table and dynamic_table else "0"},
    )


def _write_tree(root: ET.Element, output_path: Path) -> None:
    ET.indent(root, space="  ")
    output_path.write_text(ET.tostring(root, encoding="unicode") + "\n", encoding="utf-8")


def _add_home_keyframe(
    root: ET.Element, output_path: Path,
    robot_initial_qpos: Mapping[str, float] | None = None,
    robot_model=None,
) -> None:
    for existing in list(root.findall("keyframe")):
        root.remove(existing)
    _write_tree(root, output_path)
    model = mujoco.MjModel.from_xml_path(str(output_path))

    qpos = model.qpos0.copy()
    spec = load_robot_model(robot_model)
    initial = {n: READY_QPOS[n] for n in CONTROLLED_JOINTS}
    initial.update(zip(spec.auxiliary_joints, spec.fixed_auxiliary_positions, strict=True))
    if robot_initial_qpos is not None:
        if set(robot_initial_qpos) - set((*BASE_JOINTS, *CONTROLLED_JOINTS)):
            raise ValueError("robot initial configuration contains unsupported joints")
        initial.update(robot_initial_qpos)
    for joint_name, value in initial.items():
        joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
        if joint_id < 0:
            raise RuntimeError(f"Ready-pose joint not found: {joint_name}")
        if not np.isfinite(value) or (
            model.jnt_limited[joint_id]
            and not model.jnt_range[joint_id, 0] <= value <= model.jnt_range[joint_id, 1]
        ):
            raise ValueError(f"invalid initial joint position for {joint_name}: {value}")
        qpos[model.jnt_qposadr[joint_id]] = value

    ctrl = np.zeros(model.nu)
    for actuator_id in range(model.nu):
        joint_id = int(model.actuator_trnid[actuator_id, 0])
        joint_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, joint_id)
        ctrl[actuator_id] = (
            0.0 if joint_name in BASE_JOINTS
            else initial.get(joint_name, qpos[model.jnt_qposadr[joint_id]])
        )

    keyframe = ET.SubElement(root, "keyframe")
    ET.SubElement(keyframe, "key", {"name": "home", "qpos": _values(qpos), "ctrl": _values(ctrl)})
    _write_tree(root, output_path)


def _build_tabletop_scene(
    output_path: Path = DEFAULT_OUTPUT,
    refresh_robot: bool = False,
    *,
    object_ids: Sequence[str] | str | None = None,
    object_positions: Mapping[str, tuple[float, float, float | None]] | None = None,
    layout_seed: int | None = None,
    include_box: bool = True,
    include_trashcan: bool = True,
    camera_calibration: Path = DEFAULT_CAMERA_CALIBRATION,
    timestep: float = 0.002,
    dynamic_table: bool = True,
    dynamic_box: bool = True,
    robot_scene_path: Path | None = None,
    base_mode: BaseMode = "planar",
    include_table: bool = True,
    appearance_seed: int | None = None,
    preserve_fixed_inertias: bool = False,
    base_config: PlanarBaseConfig = DEFAULT_PLANAR_BASE_CONFIG,
    robot_initial_qpos: Mapping[str, float] | None = None,
    robot_model=None,
) -> Path:
    """Build and validate a selected tabletop scene.

    Omitted object selection, positions, and seed produce a fresh random scene.
    Supplying a seed makes both semantic-category selection and layout repeatable.
    """
    model_spec = load_robot_model(robot_model)
    output_path = output_path.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if layout_seed is None:
        layout_seed = secrets.randbits(32)
    if object_positions and (normalize_object_tokens(object_ids) in (None, ["random"])):
        raise ValueError("Explicit --positions require an explicit --objects selection")
    selected_ids = resolve_object_ids(object_ids, layout_seed)
    layout = layout_objects(selected_ids, layout_seed, include_box, object_positions)
    temporary_robot = robot_scene_path is None
    if temporary_robot:
        temporary_directory = tempfile.TemporaryDirectory(prefix="policy-sim-robot-", dir=output_path.parent)
        robot_scene = Path(temporary_directory.name) / "robot.xml"
        build_robot_scene(urdf_path=model_spec.urdf, output_path=robot_scene, base_mode=base_mode,
                          base_config=base_config,
                          preserve_fixed_inertias=preserve_fixed_inertias)
    else:
        temporary_directory = None
        robot_scene = Path(robot_scene_path).resolve()
        if refresh_robot or not robot_scene.exists():
            build_robot_scene(urdf_path=model_spec.urdf, output_path=robot_scene, base_mode=base_mode,
                          base_config=base_config,
                              preserve_fixed_inertias=preserve_fixed_inertias)

    root = ET.parse(robot_scene).getroot()
    if preserve_fixed_inertias and root.find(".//body[@name='robot_base']") is None:
        raise ValueError("MIT requires a fresh unfused robot scene; use a separate cache or rebuild it")
    has_planar_base = all(
        root.find(f".//joint[@name='{name}']") is not None
        for name in ("mobile_base_x", "mobile_base_y", "mobile_base_yaw")
    )
    if has_planar_base != (base_mode == "planar"):
        raise ValueError(f"robot scene {robot_scene} does not match requested base mode {base_mode!r}")
    if has_planar_base:
        for name, limits in zip(
            ("mobile_base_x", "mobile_base_y", "mobile_base_yaw"),
            (base_config.x_limits_m, base_config.y_limits_m, base_config.yaw_limits_rad), strict=True,
        ):
            joint = root.find(f".//joint[@name='{name}']")
            bounded = joint.get("limited") != "false"
            if bounded != (base_config.workspace_mode == "bounded"):
                raise ValueError("cached robot base workspace differs; rebuild the robot scene")
            if bounded and not np.allclose(np.fromstring(joint.get("range", ""), sep=" "), limits):
                raise ValueError("cached robot base range differs; rebuild the robot scene")
    for keyframe in list(root.findall("keyframe")):
        root.remove(keyframe)
    asset, worldbody = _configure_global_scene(root, timestep=timestep)
    _rebase_asset_files(asset, robot_scene, output_path)
    if temporary_directory is not None:
        temporary_directory.cleanup()
    _replace_scene_assets(asset, output_path)
    _configure_robot(worldbody, root)
    _apply_appearance_randomization(root, appearance_seed)
    _add_ik_frames(worldbody, model_spec)
    if include_table:
        _add_table(worldbody, dynamic=dynamic_table)
    if include_trashcan:
        _add_trashcan(worldbody)
    if include_box:
        _add_open_box(worldbody, dynamic=dynamic_box)
    for object_id in selected_ids:
        spec = OBJECT_CATALOG[object_id]
        position, yaw = layout[object_id]
        if spec.is_scanned:
            _import_scanned_object(asset, worldbody, spec, position, yaw, output_path)
        elif object_id == "remote":
            _add_remote(worldbody, position, yaw)
        else:
            raise RuntimeError(f"No object builder registered for {object_id}")
    _add_grasp_equalities(root, selected_ids)
    _add_cameras(worldbody, camera_calibration, wrist_cameras=bool(model_spec.auxiliary_joints))
    _add_scene_metadata(
        root,
        selected_ids,
        layout_seed,
        include_box,
        include_trashcan,
        layout,
        dynamic_table,
        include_table,
    )
    _add_home_keyframe(root, output_path, robot_initial_qpos, model_spec)

    model = mujoco.MjModel.from_xml_path(str(output_path))
    free_joint_count = sum(model.jnt_type == mujoco.mjtJoint.mjJNT_FREE)
    print(
        f"Built {output_path}\n"
        f"  bodies={model.nbody}, joints={model.njnt}, nq={model.nq}, nv={model.nv}\n"
        f"  geoms={model.ngeom}, actuators={model.nu}, free_objects={free_joint_count}, cameras={model.ncam}\n"
        f"  objects={','.join(selected_ids) if selected_ids else 'none'}, table={'yes' if include_table else 'no'}, "
        f"box={'yes' if include_box else 'no'}, trashcan={'yes' if include_trashcan else 'no'}, "
        f"base={base_mode}, seed={layout_seed}"
    )
    return output_path


def build_tabletop_scene(
    output_path: Path = DEFAULT_OUTPUT,
    refresh_robot: bool = False,
    *,
    object_ids: Sequence[str] | str | None = None,
    object_positions: Mapping[str, tuple[float, float, float | None]] | None = None,
    layout_seed: int | None = None,
    include_box: bool = True,
    include_trashcan: bool = True,
    camera_calibration: Path = DEFAULT_CAMERA_CALIBRATION,
    timestep: float = 0.002,
    dynamic_table: bool = True,
    dynamic_box: bool = True,
    robot_scene_path: Path | None = None,
    base_mode: BaseMode = "planar",
    base_config: PlanarBaseConfig = DEFAULT_PLANAR_BASE_CONFIG,
    include_table: bool = True,
    appearance_seed: int | None = None,
    preserve_fixed_inertias: bool = False,
    robot_initial_qpos: Mapping[str, float] | None = None,
    robot_model=None,
) -> Path:
    """Atomically build a scene without an implicit shared robot XML cache."""

    final = Path(output_path).resolve()
    final.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{final.name}.tmp-", suffix=".xml", dir=final.parent)
    os.close(descriptor)
    temporary = Path(temporary_name)
    temporary.unlink()
    try:
        _build_tabletop_scene(
            temporary,
            refresh_robot,
            object_ids=object_ids,
            object_positions=object_positions,
            layout_seed=layout_seed,
            include_box=include_box,
            include_trashcan=include_trashcan,
            camera_calibration=camera_calibration,
            timestep=timestep,
            dynamic_table=dynamic_table,
            dynamic_box=dynamic_box,
            robot_scene_path=robot_scene_path,
            base_mode=base_mode,
            base_config=base_config,
            include_table=include_table,
            appearance_seed=appearance_seed,
            preserve_fixed_inertias=preserve_fixed_inertias,
            robot_initial_qpos=robot_initial_qpos,
            robot_model=robot_model,
        )
        os.replace(temporary, final)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    return final


def parse_object_positions(values: Sequence[str] | None) -> dict[str, tuple[float, float, float | None]] | None:
    if values is None or list(values) == ["random"]:
        return None
    if "random" in values:
        raise ValueError("'random' cannot be combined with explicit positions")
    positions: dict[str, tuple[float, float, float | None]] = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"Invalid position '{value}'; expected ID=x,y[,yaw]")
        object_id, raw_pose = value.split("=", 1)
        if object_id not in OBJECT_CATALOG:
            raise ValueError(f"Unknown object position ID: {object_id}")
        if object_id in positions:
            raise ValueError(f"Position for {object_id} was supplied more than once")
        parts = raw_pose.split(",")
        if len(parts) not in (2, 3):
            raise ValueError(f"Invalid position '{value}'; expected ID=x,y[,yaw]")
        try:
            numeric = [float(part) for part in parts]
        except ValueError as error:
            raise ValueError(f"Invalid numeric position '{value}'") from error
        positions[object_id] = (numeric[0], numeric[1], numeric[2] if len(numeric) == 3 else None)
    return positions
