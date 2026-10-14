"""Compile task-independent YAML scenes with a derived physical robot assembly."""

from __future__ import annotations

from dataclasses import dataclass
import copy
import hashlib
import json
import math
from pathlib import Path
import subprocess
import tempfile
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

from .scene_config import SceneConfig
from .assemblies import add_assembly
from ..robot.assembly import apply_assembly, load_assembly
from ..robot.upstream import UMI_ROOT, gitlink_revision, load_umi


def _values(values) -> str:
    return " ".join(f"{float(value):.15g}" for value in values)


def _quat_wxyz(xyzw):
    return [xyzw[3], *xyzw[:3]]


def _check_hash(path: Path, expected: str):
    if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
        raise ValueError(f"Source hash changed: {path}")


def _asset_body(entity, config, target_assets):
    spec = entity["mesh_asset"]
    path = config.resolve(spec["mjcf"])
    source = ET.parse(path).getroot()
    bodies = source.findall("./worldbody/body")
    if len(bodies) != 1 or bodies[0].find("body") is not None:
        raise ValueError("Mesh assets must contain one rigid body")
    if bodies[0].find("joint") is not None or bodies[0].find("freejoint") is not None:
        raise ValueError("Scene YAML owns the rigid body's dynamic/static setting")
    prefix = f"entity_{entity['name']}_"
    scale = spec["scale"]
    for asset in source.findall("./asset/*"):
        asset = copy.deepcopy(asset)
        for attr in ["name", "texture", "material"]:
            if attr in asset.attrib:
                asset.set(attr, prefix + asset.get(attr))
        if asset.get("file"):
            asset.set("file", str((path.parent / asset.get("file")).resolve()))
        if asset.tag == "mesh":
            old_scale = np.fromstring(asset.get("scale", "1 1 1"), sep=" ")
            asset.set("scale", _values(old_scale * scale))
        target_assets.append(asset)
    body = copy.deepcopy(bodies[0])
    body.set("name", prefix + "body")
    for element in body.iter():
        if element is not body:
            for attr in ["name", "mesh", "material"]:
                if attr in element.attrib:
                    element.set(attr, prefix + element.get(attr))
            if element.get("pos"):
                element.set("pos", _values(np.fromstring(element.get("pos"), sep=" ") * scale))
    inertial = body.find("inertial")
    if inertial is None or not inertial.get("diaginertia"):
        raise ValueError("Mesh asset requires an explicit principal inertia")
    ratio = spec["mass_kg"] / float(inertial.get("mass"))
    inertial.set("mass", str(spec["mass_kg"]))
    inertia = np.fromstring(inertial.get("diaginertia"), sep=" ")
    inertial.set("diaginertia", _values(inertia * ratio * scale**2))
    if not entity["static"]:
        ET.SubElement(body, "freejoint", name=prefix + "free")
    for geom in body.findall("geom"):
        if geom.get("contype", "1") != "0":
            geom.set("friction", "0.8 0.005 0.0001")
    return body


@dataclass
class BuiltScene:
    xml_path: Path
    config: SceneConfig
    model: mujoco.MjModel
    data: mujoco.MjData
    camera_k: np.ndarray
    camera_size_wh: tuple[int, int]
    camera_raw_size_wh: tuple[int, int]
    camera_accepted: bool


def build_scene(config: SceneConfig, output: str | Path) -> BuiltScene:
    """Build geometry only. No instruction, reward or task selector is accepted."""
    output = Path(output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    spec = config.data
    robot = spec["robot"]
    source = config.resolve(robot["source_checkout"])
    commit = subprocess.check_output(["git", "-C", str(source), "rev-parse", "HEAD"], text=True).strip()
    expected = gitlink_revision() if robot["source_commit"] == "gitlink" else robot["source_commit"]
    if commit != expected:
        raise ValueError("Robot source commit differs from scene's pinned source")
    urdf = source / robot["urdf"]
    _check_hash(urdf, robot["urdf_sha256"])
    if source != UMI_ROOT:
        raise ValueError("Robot source must be the deployment-owned resource directory")
    load_umi()
    from hexfellow_deploy.robotics.control import robot as robot_source
    if Path(robot_source.__file__).resolve() != Path(__file__).resolve().parents[1] / "robotics/control/robot.py":
        raise ValueError("A different hexfellow_deploy.robotics checkout is already imported")

    bounds = robot["base_bounds"]
    base_config = robot_source.PlanarBaseConfig(
        x_limits_m=tuple(bounds["x_m"]), y_limits_m=tuple(bounds["y_m"]),
        yaw_limits_rad=tuple(bounds["yaw_rad"]),
        linear_velocity_limit=config.base_actuation["linear_velocity_limit_m_s"],
        angular_velocity_limit=config.base_actuation["angular_velocity_limit_rad_s"],
    )
    with tempfile.TemporaryDirectory(prefix="scene-robot-") as temp:
        robot_xml = Path(temp) / "robot.xml"
        robot_source.build_scene(urdf, robot_xml, base_mode="planar", base_config=base_config,
                                 preserve_fixed_inertias=True)
        root = ET.parse(robot_xml).getroot()
        for asset in root.findall("./asset/*"):
            if asset.get("file"):
                asset.set("file", str((robot_xml.parent / asset.get("file")).resolve()))
    root.set("model", "configured_environment")
    for keyframe in root.findall("keyframe"):
        root.remove(keyframe)
    root.find("option").set("gravity", _values(spec["physics"]["gravity_m_s2"]))
    root.find("option").set("timestep", str(spec["physics"]["timestep_s"]))
    world = root.find("worldbody")
    for body in world.iter("body"):
        body.set("gravcomp", "1")  # Explicit ideal robot gravity compensation.
    assembly = None
    if "assembly" in robot:
        assembly = load_assembly(config.resolve(robot["assembly"]))
        apply_assembly(root, source, output.parent / "robot_assets", assembly)
        for body in world.iter("body"):
            body.set("gravcomp", "1" if assembly["actuation"]["gravity_compensation"] else "0")
    ground = world.find("geom[@name='ground']")
    ground.attrib.pop("material", None)
    ground.set("size", _values(spec["floor"]["half_size_m"]))
    ground.set("rgba", _values(spec["floor"]["rgba"]))
    ground.set("contype", "1")
    ground.set("conaffinity", "3")
    for light in list(world.findall("light")):
        world.remove(light)
    for index, light in enumerate(spec["lights"]):
        ET.SubElement(world, "light", name=f"scene_light_{index}",
                      pos=_values(light["position_m"]), dir=_values(light["direction"]),
                      diffuse=_values(light["diffuse"]),
                      directional=str(light["directional"]).lower())
    assets = root.find("asset")
    for entity in spec["entities"]:
        if "mesh_asset" in entity:
            body = _asset_body(entity, config, assets)
        else:
            body = ET.Element("body", name=f"entity_{entity['name']}_body")
            for box in entity["boxes"]:
                ET.SubElement(body, "geom", name=f"entity_{entity['name']}_{box['name']}",
                              type="box", pos=_values(box["position_m"]),
                              size=_values(np.asarray(box["size_m"]) / 2),
                              rgba=_values(box["rgba"]), group="0",
                              friction="0.8 0.005 0.0001")
        body.set("pos", _values(entity["position_m"]))
        body.set("quat", _values(_quat_wxyz(entity["quaternion_xyzw"])))
        world.append(body)

    passive_initial = {}
    for assembly_spec in spec.get("assemblies", []):
        passive_initial.update(add_assembly(root, config, assembly_spec))

    camera = spec["camera"]
    calibration_path = config.resolve(camera["calibration"])
    _check_hash(calibration_path, camera["calibration_sha256"])
    calibration = json.loads(calibration_path.read_text())
    transform = np.asarray(calibration["T_mount_from_camera"])
    k = np.asarray(calibration["K"], dtype=float)
    width, height = camera["image_resolution_wh"]
    old_width, old_height = camera["calibration_resolution_wh"]
    sx, sy = width / old_width, height / old_height
    k[0, 0] *= sx
    k[1, 1] *= sy
    k[0, 2] = (k[0, 2] + 0.5) * sx - 0.5
    k[1, 2] = (k[1, 2] + 0.5) * sy - 0.5
    raw_width, raw_height = width + 160, height + 160
    parent = world.find(f".//body[@name='{calibration['mount_frame']}']")
    if parent is None:
        raise ValueError("Robot does not contain the calibrated camera mount")
    if camera.get("pose_source") == "robot_optical_frame":
        if assembly is None:
            raise ValueError("Robot optical camera requires an assembly configuration")
        parent = world.find(f".//body[@name='{assembly['camera']['optical_frame']}']")
        transform = np.eye(4)
        transform[2, 3] = assembly["camera"]["lens_forward_offset_m"]
    rotation = np.ascontiguousarray(transform[:3, :3] @ np.diag([1., -1., -1.]))
    quaternion = np.empty(4)
    mujoco.mju_mat2Quat(quaternion, rotation.reshape(9))
    ET.SubElement(parent, "camera", name="policy_rgb", pos=_values(transform[:3, 3]),
                  quat=_values(quaternion), fovy=str(math.degrees(2*math.atan(raw_height/(2*k[1, 1])))))
    visual = root.find("visual/global")
    visual.set("offwidth", str(max(1280, raw_width)))
    visual.set("offheight", str(max(960, raw_height)))
    ET.indent(root, space="  ")
    output.write_text(ET.tostring(root, encoding="unicode") + "\n")
    model = mujoco.MjModel.from_xml_path(str(output))
    data = mujoco.MjData(model)
    values = dict(robot["initial_joints"])
    if set(values) != set(robot_source.CONTROLLED_JOINTS):
        raise ValueError("initial_joints must specify all 15 lift/arm coordinates exactly")
    values.update(dict(zip(robot_source.BASE_JOINTS, robot["initial_base_xyyaw"])))
    for name, value in values.items():
        joint = model.joint(name)
        if model.jnt_limited[joint.id] and not model.jnt_range[joint.id, 0] <= value <= model.jnt_range[joint.id, 1]:
            raise ValueError(f"Initial joint outside model limits: {name}")
        data.qpos[joint.qposadr[0]] = value
    for name in robot["initial_joints"]:
        data.ctrl[model.actuator(f"position_{name}").id] = values[name]
    for name, value in passive_initial.items():
        joint = model.joint(name)
        if joint.type[0] not in (mujoco.mjtJoint.mjJNT_HINGE, mujoco.mjtJoint.mjJNT_SLIDE):
            raise ValueError("Assembly initial_joints supports only passive scalar joints")
        if model.jnt_limited[joint.id] and not model.jnt_range[joint.id, 0] <= value <= model.jnt_range[joint.id, 1]:
            raise ValueError(f"Initial passive joint outside limits: {name}")
        data.qpos[joint.qposadr[0]] = value
    if assembly:
        for side in ["left", "right"]:
            for label in ["positive", "negative"]:
                name = f"{side}_jaw_{label}"
                data.qpos[model.joint(name).qposadr[0]] = assembly["gripper"]["opening_m"] / 2
                data.ctrl[model.actuator("position_"+name).id] = assembly["gripper"]["opening_m"] / 2
    mujoco.mj_forward(model, data)
    keys = ET.SubElement(root, "keyframe")
    ET.SubElement(keys, "key", name="initial", qpos=_values(data.qpos), ctrl=_values(data.ctrl))
    ET.indent(root, space="  ")
    output.write_text(ET.tostring(root, encoding="unicode") + "\n")
    model = mujoco.MjModel.from_xml_path(str(output))
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, model.key("initial").id)
    mujoco.mj_forward(model, data)
    return BuiltScene(output, config, model, data, k, (width, height), (raw_width, raw_height),
                      bool(calibration["calibration_accepted"]))


def policy_camera_maps(scene: BuiltScene):
    """Crop calibrated pixels; only unequal focal lengths require X resampling."""
    width, height = scene.camera_size_wh
    raw_width, raw_height = scene.camera_raw_size_wh
    k = scene.camera_k
    crop_x, crop_y = (raw_width-width)//2, (raw_height-height)//2
    x = (np.arange(width, dtype=np.float32) - k[0, 2]) * k[1, 1] / k[0, 0] + k[0, 2] + crop_x
    y = np.arange(height, dtype=np.float32) + crop_y
    if min(x.min(), y.min()) < 0 or x.max() > raw_width-1 or y.max() > raw_height-1:
        raise ValueError("Configured intrinsics exceed raw render margin")
    return np.meshgrid(x, y)


def update_policy_camera(renderer, scene: BuiltScene, option):
    """Use the calibrated principal point in the OpenGL frustum, not a shifted Z map."""
    renderer.update_scene(scene.data, camera="policy_rgb", scene_option=option)
    width, height = scene.camera_size_wh
    raw_width, raw_height = scene.camera_raw_size_wh
    k = scene.camera_k
    cx = k[0, 2] + (raw_width-width)//2
    cy = k[1, 2] + (raw_height-height)//2
    # Pixel (0, 0) is its center; the top/left image edges are at -0.5.
    # MuJoCo averages the two GL cameras for non-stereo rendering.
    for camera in renderer.scene.camera:
        unit = camera.frustum_near / k[1, 1]
        camera.frustum_top = (cy + .5) * unit
        camera.frustum_bottom = -(raw_height - cy - .5) * unit
        camera.frustum_center = ((raw_width - 1)/2 - cx) * unit


def policy_renderer(scene: BuiltScene):
    """Sample optical pixel centers; MSAA depth resolves can bias tilted surfaces."""
    width, height = scene.camera_raw_size_wh
    samples = scene.model.vis.quality.offsamples
    try:
        scene.model.vis.quality.offsamples = 0
        return mujoco.Renderer(scene.model, height=height, width=width)
    finally:
        scene.model.vis.quality.offsamples = samples


def render_policy_rgb(scene: BuiltScene) -> np.ndarray:
    """Render the mounted camera, preserving its calibrated principal point."""
    import cv2
    option = mujoco.MjvOption()
    option.geomgroup[3] = 0  # Collision hulls are not visual image content.
    option.sitegroup[:] = 0
    with policy_renderer(scene) as renderer:
        update_policy_camera(renderer, scene, option)
        raw = renderer.render().copy()
    return cv2.remap(raw, *policy_camera_maps(scene), interpolation=cv2.INTER_LINEAR)


def read_policy_depth(renderer, scene: BuiltScene, maps):
    """Read the already-updated RGB camera's optical Z with no edge blending."""
    import cv2
    renderer.enable_depth_rendering()
    try:
        depth = cv2.remap(renderer.render(), *maps, interpolation=cv2.INTER_NEAREST,
                          borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    finally:
        renderer.disable_depth_rendering()
    near, far = scene.config.data["camera"].get("depth_range_m", [.2, 8.])
    depth[~np.isfinite(depth) | (depth < near) | (depth > far)] = 0
    return depth.astype(np.float32, copy=False)


def render_policy_depth(scene: BuiltScene) -> np.ndarray:
    """Offline preview with the same registration and units as live observations."""
    option = mujoco.MjvOption()
    option.geomgroup[3] = 0
    option.sitegroup[:] = 0
    with policy_renderer(scene) as renderer:
        update_policy_camera(renderer, scene, option)
        return read_policy_depth(renderer, scene, policy_camera_maps(scene))


def render_overview(scene: BuiltScene) -> np.ndarray:
    """Operator-only overview. This image must never enter policy observations."""
    positions = [scene.config.data["robot"]["initial_base_xyyaw"][:2]]
    positions += [e["position_m"][:2] for e in scene.config.data["entities"]]
    positions = np.asarray(positions)
    camera = mujoco.MjvCamera()
    camera.lookat[:] = [*np.mean([positions.min(0), positions.max(0)], axis=0), 0.65]
    camera.distance = max(3., float(np.linalg.norm(np.ptp(positions, axis=0))) * 1.6)
    camera.azimuth = 130
    camera.elevation = -25
    option = mujoco.MjvOption()
    option.geomgroup[3] = 0
    with mujoco.Renderer(scene.model, height=720, width=1280) as renderer:
        renderer.update_scene(scene.data, camera=camera, scene_option=option)
        return renderer.render().copy()
