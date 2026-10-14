#!/usr/bin/env python3
"""Maver X4 names, planar-base configuration, and MuJoCo conversion."""

from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import pinocchio as pin

from ..paths import SCENE_DIR, URDF_DIR
from .posture import DEFAULT_NULL_POSE_PATH, load_null_pose

BaseMode = Literal["fixed", "planar"]

BASE_BODY = "mobile_base"
BASE_X_JOINT = "mobile_base_x"
BASE_Y_JOINT = "mobile_base_y"
BASE_YAW_JOINT = "mobile_base_yaw"
BASE_JOINTS = (BASE_X_JOINT, BASE_Y_JOINT, BASE_YAW_JOINT)
BASE_ACTUATORS = tuple(f"velocity_{name}" for name in BASE_JOINTS)
BASE_FRAME = "maver_x4_asm_maver_x4_base_link"

LIFT_JOINT = "maver_x4_asm_URDF_joint_1"
LEFT_ARM_JOINTS = tuple(f"maver_x4_asm_7_urdf_SLDASM_joint_{index}" for index in range(1, 8))
RIGHT_ARM_JOINTS = tuple(f"maver_x4_asm_7_urdf_SLDASM2_joint_{index}" for index in range(1, 8))
ARM_JOINTS = (*LEFT_ARM_JOINTS, *RIGHT_ARM_JOINTS)
CONTROLLED_JOINTS = (LIFT_JOINT, *LEFT_ARM_JOINTS, *RIGHT_ARM_JOINTS)
PLANAR_CONTROLLED_COORDINATES = (*BASE_JOINTS, *CONTROLLED_JOINTS)
GRIPPER_JOINTS = tuple(f"robot{side}_joint{index}" for side in ("", "2") for index in range(1, 7))

TCP_OFFSET = (0.00948191, 0.0, 0.262)
GRIPPER_APERTURE_MAX = 0.105
GRIPPER_ANGLE_MAX = 0.925
# The fingertip TCP is intentionally placed ahead of the physical pinch
# region.  This point was calibrated against the enabled per-link gripper
# collision meshes: it is centred in the open jaw channel rather than inside
# the palm linkage.  Expert grasp planning targets this pinch centre instead
# of placing the TCP origin on an object's centre of mass.
GRASP_CENTER_IN_TCP = (0.010, 0.0, -0.035)
# A conservative object-centre guard used in addition to opposing-jaw contact.
# Coordinates are expressed in the TCP frame around GRASP_CENTER_IN_TCP.
GRASP_CAPTURE_HALF_EXTENTS = (0.050, 0.080, 0.075)
LEFT_TCP_SITE = "left_tcp"
RIGHT_TCP_SITE = "right_tcp"
LEFT_POLICY_EEF_SITE = "left_policy_eef"
RIGHT_POLICY_EEF_SITE = "right_policy_eef"
LEFT_IK_TARGET = "left_ik_target"
RIGHT_IK_TARGET = "right_ik_target"

DEFAULT_NULL_POSE = load_null_pose(DEFAULT_NULL_POSE_PATH)
READY_QPOS = {
    **DEFAULT_NULL_POSE.joint_positions(LIFT_JOINT, LEFT_ARM_JOINTS, RIGHT_ARM_JOINTS),
    **{f"robot_joint{index}": 0.0 for index in range(1, 7)},
    **{f"robot2_joint{index}": 0.0 for index in range(1, 7)},
}


@dataclass(frozen=True)
class PlanarBaseConfig:
    """Physical, planning, and soft-priority settings for the mobile base."""

    x_limits_m: tuple[float, float] = (-0.20, 0.20)
    y_limits_m: tuple[float, float] = (-0.20, 0.20)
    yaw_limits_rad: tuple[float, float] = (-math.pi / 6.0, math.pi / 6.0)
    position_margin_m: float = 0.01
    yaw_margin_rad: float = 0.02
    linear_velocity_limit: float = 2.0
    angular_velocity_limit: float = 2.0
    linear_acceleration_limit: float = 1.0
    angular_acceleration_limit: float = 2.0
    linear_jerk_limit: float = 200.0
    angular_jerk_limit: float = 400.0
    anchor_position_cost: float = 10.0
    anchor_yaw_cost: float = 10.0
    linear_servo_kv: float = 5000.0
    angular_servo_kv: float = 200.0
    linear_force_limit: float = 400.0
    angular_force_limit: float = 80.0
    linear_joint_damping: float = 50.0
    angular_joint_damping: float = 5.0
    workspace_mode: Literal["bounded", "unbounded"] = "bounded"
    boundary_governor: Literal["legacy", "finite_base_v1"] = "legacy"

    def __post_init__(self) -> None:
        if self.workspace_mode not in {"bounded", "unbounded"}:
            raise ValueError("unknown base workspace mode")
        if self.boundary_governor not in {"legacy", "finite_base_v1"}:
            raise ValueError("unknown base boundary governor")
        for label, limits in (("x", self.x_limits_m), ("y", self.y_limits_m), ("yaw", self.yaw_limits_rad)):
            if (len(limits) != 2 or not all(math.isfinite(value) for value in limits)
                    or not limits[0] < 0.0 < limits[1]):
                raise ValueError(f"{label} limits must straddle zero, got {limits}")
        if self.position_margin_m <= 0.0 or self.yaw_margin_rad <= 0.0:
            raise ValueError("planar-base planning margins must be positive")
        if 2.0 * self.position_margin_m >= min(
            self.x_limits_m[1] - self.x_limits_m[0], self.y_limits_m[1] - self.y_limits_m[0]
        ):
            raise ValueError("planar-base position margin eliminates the workspace")
        if 2.0 * self.yaw_margin_rad >= self.yaw_limits_rad[1] - self.yaw_limits_rad[0]:
            raise ValueError("planar-base yaw margin eliminates the workspace")
        positive = (
            self.linear_velocity_limit,
            self.angular_velocity_limit,
            self.linear_acceleration_limit,
            self.angular_acceleration_limit,
            self.linear_jerk_limit,
            self.angular_jerk_limit,
            self.anchor_position_cost,
            self.anchor_yaw_cost,
            self.linear_servo_kv,
            self.angular_servo_kv,
            self.linear_force_limit,
            self.angular_force_limit,
            self.linear_joint_damping,
            self.angular_joint_damping,
        )
        if any(not math.isfinite(value) or value <= 0.0 for value in positive):
            raise ValueError("planar-base limits, costs, gains, damping, and force caps must be positive")


DEFAULT_PLANAR_BASE_CONFIG = PlanarBaseConfig()

DEFAULT_URDF = URDF_DIR / "maver_x4_asm_asm.urdf"
DEFAULT_OUTPUT = SCENE_DIR / "maver_x4_scene.xml"


def _values(values) -> str:
    return " ".join(f"{float(value):g}" for value in values)


def _planar_root_inertia(urdf_path: Path) -> pin.Inertia:
    root_joint = pin.JointModelComposite()
    for joint in (pin.JointModelPX(), pin.JointModelPY(), pin.JointModelRZ()):
        root_joint.addJoint(joint)
    model = pin.buildModelFromUrdf(str(urdf_path), root_joint)
    return model.inertias[model.getJointId("root_joint")]


def _wrap_planar_base(
    worldbody: ET.Element,
    urdf_path: Path,
    config: PlanarBaseConfig,
) -> ET.Element:
    """Put the converted fixed robot below a three-DoF planar root body."""
    mobile = ET.Element("body", {"name": BASE_BODY})
    inertia = _planar_root_inertia(urdf_path)
    matrix = inertia.inertia
    ET.SubElement(
        mobile,
        "inertial",
        {
            "pos": _values(inertia.lever),
            "mass": f"{inertia.mass:g}",
            "fullinertia": _values(
                (
                    matrix[0, 0],
                    matrix[1, 1],
                    matrix[2, 2],
                    matrix[0, 1],
                    matrix[0, 2],
                    matrix[1, 2],
                )
            ),
        },
    )
    joint_specs = (
        (BASE_X_JOINT, "slide", "1 0 0", config.x_limits_m, config.linear_joint_damping),
        (BASE_Y_JOINT, "slide", "0 1 0", config.y_limits_m, config.linear_joint_damping),
        (BASE_YAW_JOINT, "hinge", "0 0 1", config.yaw_limits_rad, config.angular_joint_damping),
    )
    for name, joint_type, axis, limits, damping in joint_specs:
        ET.SubElement(
            mobile,
            "joint",
            {
                "name": name,
                "type": joint_type,
                "axis": axis,
                **({"range": _values(limits)} if config.workspace_mode == "bounded"
                   else {"limited": "false"}),
                "damping": f"{damping:g}",
            },
        )
    for child in list(worldbody):
        worldbody.remove(child)
        mobile.append(child)
    worldbody.append(mobile)
    return mobile


def load_nominal_mujoco_model(urdf_path: Path = DEFAULT_URDF):
    """Compile nominal inertias without fixed-body fusion.

    MuJoCo 3.10 fixed-body fusion changes the wrist assembly centre of mass
    for this URDF. Keeping fixed bodies makes gravity agree with the unfused
    nominal model; no masses, inertias or source assets are changed.
    """
    import mujoco

    urdf_path = Path(urdf_path).resolve()
    root = ET.parse(urdf_path).getroot()
    extension = root.find("mujoco")
    if extension is None:
        extension = ET.SubElement(root, "mujoco")
    compiler = extension.find("compiler")
    if compiler is None:
        compiler = ET.SubElement(extension, "compiler")
    compiler.set("fusestatic", "false")
    compiler.set("strippath", "false")
    for mesh in root.findall(".//mesh"):
        mesh.set("filename", str((urdf_path.parent / mesh.attrib["filename"]).resolve()))
    return mujoco.MjModel.from_xml_string(ET.tostring(root, encoding="unicode"))


def build_scene(
    urdf_path: Path = DEFAULT_URDF,
    output_path: Path = DEFAULT_OUTPUT,
    *,
    base_mode: BaseMode = "planar",
    base_config: PlanarBaseConfig = DEFAULT_PLANAR_BASE_CONFIG,
    preserve_fixed_inertias: bool = False,
) -> Path:
    """Build a standalone MJCF scene while keeping mesh paths relative to it."""
    import mujoco

    urdf_path = urdf_path.resolve()
    output_path = output_path.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Let MuJoCo do the URDF kinematic/inertial conversion. Saving the compiled
    # model gives us normal MJCF that can then be extended with a scene.
    model = (
        load_nominal_mujoco_model(urdf_path)
        if preserve_fixed_inertias
        else mujoco.MjModel.from_xml_path(str(urdf_path))
    )
    with tempfile.NamedTemporaryFile(suffix=".xml") as temporary_xml:
        mujoco.mj_saveLastXML(temporary_xml.name, model)
        root = ET.parse(temporary_xml.name).getroot()

    root.set("model", "maver_x4_scene")
    if preserve_fixed_inertias:
        custom = root.find("custom")
        if custom is None:
            custom = ET.SubElement(root, "custom")
        ET.SubElement(custom, "text", {
            "name": "mit_asset_contract",
            "data": json.dumps({
                "conversion": "unfused_urdf_v1",
                "urdf_sha256": hashlib.sha256(urdf_path.read_bytes()).hexdigest(),
                "mujoco_version": mujoco.__version__,
            }, sort_keys=True),
        })
    compiler = root.find("compiler")
    if compiler is None:
        compiler = ET.Element("compiler")
        root.insert(0, compiler)
    compiler.set("angle", "radian")
    compiler.set("autolimits", "true")

    compiler_index = list(root).index(compiler)
    option = ET.Element(
        "option",
        {
            "timestep": "0.002",
            "integrator": "implicitfast",
            # This scene is intended for inspecting/posing the CAD model. Zero
            # gravity keeps the long arms in place without controller tuning.
            "gravity": "0 0 0",
        },
    )
    root.insert(compiler_index + 1, option)

    visual = ET.Element("visual")
    ET.SubElement(visual, "global", {"offwidth": "1280", "offheight": "960"})
    ET.SubElement(visual, "headlight", {"ambient": "0.2 0.2 0.2", "diffuse": "0.5 0.5 0.5", "specular": "0.1 0.1 0.1"})
    ET.SubElement(visual, "rgba", {"haze": "0.15 0.2 0.25 1"})
    ET.SubElement(visual, "quality", {"shadowsize": "4096"})
    root.insert(compiler_index + 2, visual)

    default = ET.Element("default")
    ET.SubElement(default, "joint", {"damping": "0.5", "armature": "0.002"})
    root.insert(compiler_index + 3, default)

    asset = root.find("asset")
    if asset is None:
        asset = ET.Element("asset")
        root.insert(compiler_index + 4, asset)

    # MuJoCo keeps URDF mesh paths relative to the URDF directory.  The
    # generated MJCF lives under scenes/, so rewrite them relative to that
    # output file instead.
    for mesh in asset.findall("mesh"):
        mesh_file = mesh.get("file")
        if mesh_file:
            source_mesh = (urdf_path.parent / mesh_file).resolve()
            mesh.set("file", Path(os.path.relpath(source_mesh, output_path.parent)).as_posix())
    asset.insert(
        0,
        ET.Element(
            "texture",
            {
                "name": "ground_checker",
                "type": "2d",
                "builtin": "checker",
                "rgb1": "0.22 0.26 0.30",
                "rgb2": "0.12 0.15 0.18",
                "width": "512",
                "height": "512",
                "mark": "edge",
                "markrgb": "0.8 0.8 0.8",
            },
        ),
    )
    asset.insert(
        1,
        ET.Element(
            "material",
            {
                "name": "ground_material",
                "texture": "ground_checker",
                "texrepeat": "8 8",
                "texuniform": "true",
                "reflectance": "0.08",
                "roughness": "0.8",
            },
        ),
    )

    worldbody = root.find("worldbody")
    if worldbody is None:
        raise RuntimeError("MuJoCo's URDF conversion did not produce a worldbody")

    if base_mode not in {"fixed", "planar"}:
        raise ValueError(f"unknown base mode {base_mode!r}")
    if base_mode == "planar":
        _wrap_planar_base(worldbody, urdf_path, base_config)

    # The exported URDF uses detailed STL files as both visual and collision
    # meshes. For a fast, stable pose-inspection scene, render them but exclude
    # them from collision detection.
    for geom in worldbody.iter("geom"):
        geom.set("group", "1")
        geom.set("contype", "0")
        geom.set("conaffinity", "0")

    worldbody.insert(
        0,
        ET.Element(
            "geom",
            {
                "name": "ground",
                "type": "plane",
                "pos": "0 0 0",
                "size": "0 0 0.05",
                "material": "ground_material",
                "group": "0",
                "contype": "0",
                "conaffinity": "0",
            },
        ),
    )
    worldbody.insert(
        1,
        ET.Element(
            "light",
            {
                "name": "key_light",
                "pos": "-1.5 -1.5 3",
                "dir": "0.35 0.35 -1",
                "diffuse": "0.65 0.65 0.65",
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
                "pos": "1.5 1 2",
                "dir": "-0.4 -0.2 -1",
                "diffuse": "0.25 0.3 0.4",
                "castshadow": "false",
            },
        ),
    )

    actuator = ET.Element("actuator")
    home_qpos: list[float] = [0.0, 0.0, 0.0] if base_mode == "planar" else []
    home_ctrl: list[float] = []
    if base_mode == "planar":
        for name, kv, velocity, force in (
            (
                BASE_X_JOINT,
                base_config.linear_servo_kv,
                base_config.linear_velocity_limit,
                base_config.linear_force_limit,
            ),
            (
                BASE_Y_JOINT,
                base_config.linear_servo_kv,
                base_config.linear_velocity_limit,
                base_config.linear_force_limit,
            ),
            (
                BASE_YAW_JOINT,
                base_config.angular_servo_kv,
                base_config.angular_velocity_limit,
                base_config.angular_force_limit,
            ),
        ):
            ET.SubElement(
                actuator,
                "velocity",
                {
                    "name": f"velocity_{name}",
                    "joint": name,
                    "kv": f"{kv:g}",
                    "ctrllimited": "true",
                    "ctrlrange": f"{-velocity:g} {velocity:g}",
                    "forcelimited": "true",
                    "forcerange": f"{-force:g} {force:g}",
                },
            )
            home_ctrl.append(0.0)
    joint_elements = {joint.get("name"): joint for joint in worldbody.iter("joint") if joint.get("name") is not None}
    for joint_id in range(model.njnt):
        joint_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, joint_id)
        if joint_name is None:
            continue
        lower, upper = (float(value) for value in model.jnt_range[joint_id])
        joint_type = model.jnt_type[joint_id]
        if joint_type == mujoco.mjtJoint.mjJNT_SLIDE:
            kp = "500"
            # The 43 kg lift assembly needs substantially more damping than
            # the lightweight gripper slides when both retain URDF effort=10.
            damping = "30" if upper - lower > 0.1 else "1"
        else:
            kp = "50"
            damping = "1"
        joint_elements[joint_name].set("damping", damping)
        ET.SubElement(
            actuator,
            "position",
            {
                "name": f"position_{joint_name}",
                "joint": joint_name,
                "kp": kp,
                "ctrllimited": "true",
                "ctrlrange": f"{lower:g} {upper:g}",
            },
        )
        home_qpos.append(0.0)
        home_ctrl.append(0.0)
    root.append(actuator)

    keyframe = ET.Element("keyframe")
    ET.SubElement(
        keyframe,
        "key",
        {
            "name": "home",
            "qpos": _values(home_qpos),
            "ctrl": _values(home_ctrl),
        },
    )
    root.append(keyframe)

    ET.indent(root, space="  ")
    output_path.write_text(ET.tostring(root, encoding="unicode") + "\n", encoding="utf-8")

    # Compile the final scene now so generation fails early on invalid MJCF or
    # missing assets instead of later when the viewer starts.
    scene_model = mujoco.MjModel.from_xml_path(str(output_path))
    print(
        f"Built {output_path}\n"
        f"  bodies={scene_model.nbody}, joints={scene_model.njnt}, "
        f"geoms={scene_model.ngeom}, actuators={scene_model.nu}"
    )
    return output_path
