"""Derived robot assembly: aligned camera mount and articulated GP80 CAD.

No scene entities or task names enter this module. Vendor geometry is retained;
the three disconnected GP80 STL components become a palm and two moving jaws.
Contact pads are an explicit approximate rubber-pad model.
"""
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
import trimesh
import yaml

from .joints import BASE, LIFT, LEFT, RIGHT, JOINTS, SIDES

# Plant/actuator coordinates only. These are NOT encoder channels.
BASE_JOINTS = ("mobile_base_x", "mobile_base_y", "mobile_base_yaw")


def values(x):
    return " ".join(f"{float(v):.12g}" for v in x)


def load_assembly(path):
    return yaml.safe_load(Path(path).read_text())


def apply_assembly(root, source: Path, output_dir: Path, config):
    world = root.find("worldbody")
    # Elliptic friction and a no-slip refinement resolve slow contact creep
    # under sustained grasp loads; no equality or object attachment is added.
    root.find("option").set("cone", "elliptic")
    root.find("option").set("impratio", "10")
    root.find("option").set("noslip_iterations", "5")
    mount = config["camera_mount"]
    world.find(f".//body[@name='{mount['body']}']").set("pos", values(mount["position_m"]))
    asset = root.find("asset")
    actuator = root.find("actuator")
    contact = ET.SubElement(root, "contact")
    # Full robot/world contacts; robot self contacts use non-adjacent bodies.
    # Fixed/adjacent assemblies are mechanically connected, not obstacles.
    for geom in world.findall(".//geom"):
        geom.set("contype", "2")
        geom.set("conaffinity", "3")
        geom.set("friction", "0.8 0.005 0.0001")
    moving_ancestors = {}
    def walk(body, moving):
        name = body.get("name")
        if body.find("joint") is not None:
            moving = moving + [name]
        moving_ancestors[name] = moving
        for child in body.findall("body"):
            walk(child, moving)
    for body in world.findall("body"):
        walk(body, [])
    bodies = list(moving_ancestors)
    for i, first in enumerate(bodies):
        for second in bodies[i+1:]:
            a, b = moving_ancestors[first], moving_ancestors[second]
            common = 0
            for x, y in zip(a, b):
                if x != y:
                    break
                common += 1
            if len(a) + len(b) - 2*common <= 2:
                ET.SubElement(contact, "exclude", body1=first, body2=second)

    output_dir.mkdir(parents=True, exist_ok=True)
    parts = trimesh.load(source / "urdf/meshes/hexfellow_gp80/base_link.STL",
                         force="mesh").split(only_watertight=False)
    if len(parts) != 3:
        raise ValueError("Expected exactly palm and two GP80 CAD jaw components")
    components = {}
    for part in parts:
        name = "palm" if part.bounds[1, 2] < .08 else (
            "positive" if part.center_mass[1] > 0 else "negative")
        components[name] = part
        path = output_dir / f"gp80_{name}.stl"
        part.export(path)
        ET.SubElement(asset, "mesh", name=f"gp80_{name}", file=str(path.resolve()))

    grip = config["gripper"]
    for side, tool in [("left", "robot_base"), ("right", "robot2_base")]:
        parent = world.find(f".//body[@name='{tool}']")
        for geom in list(parent.findall("geom")):
            parent.remove(geom)
        parent.find("inertial").set("mass", str(grip["mass_kg"] - 2*grip["jaw_mass_kg"]))
        ET.SubElement(parent, "geom", name=f"{side}_palm", type="mesh", mesh="gp80_palm",
                      rgba="0.8 0.8 0.82 1", contype="2", conaffinity="3", group="1")
        ET.SubElement(parent, "site", name=f"{side}_tcp", pos=values(grip["tcp_in_tool_m"]),
                      size="0.003", rgba="0 0 0 0", group="4")
        for label, sign in [("positive", 1), ("negative", -1)]:
            name = f"{side}_jaw_{label}"
            body = ET.SubElement(parent, "body", name=name)
            body.set("gravcomp", "1" if config["actuation"]["gravity_compensation"] else "0")
            ET.SubElement(body, "inertial", pos=values(components[label].center_mass),
                          mass=str(grip["jaw_mass_kg"]), diaginertia="0.00003 0.00003 0.000015")
            ET.SubElement(body, "joint", name=name, type="slide", axis=f"0 {sign} 0",
                          range=f"0 {grip['opening_m']/2}", damping="2", armature="0.001")
            ET.SubElement(body, "geom", name=name+"_cad", type="mesh", mesh=f"gp80_{label}",
                          rgba="0.22 0.24 0.26 1", contype="2", conaffinity="3", group="1")
            ET.SubElement(body, "geom", name=name+"_pad", type="box",
                          pos=values([0, sign*grip["pad_half_size_m"][1], .130]),
                          size=values(grip["pad_half_size_m"]), rgba="0.08 0.09 0.10 1",
                          friction=f"{grip['pad_friction']} 0.01 0.001", condim="6",
                          solref="0.008 1", solimp="0.95 0.99 0.001", contype="2", conaffinity="3")
            ET.SubElement(actuator, "position", name="position_"+name, joint=name,
                          kp="1800", kv="16", ctrlrange=f"0 {grip['opening_m']/2}",
                          forcerange=f"-{grip['jaw_force_limit_n']} {grip['jaw_force_limit_n']}")
            ET.SubElement(contact, "exclude", body1=tool, body2=name)
        ET.SubElement(contact, "exclude", body1=f"{side}_jaw_positive", body2=f"{side}_jaw_negative")
    tuning = config["actuation"]
    for element in actuator:
        joint = element.get("joint", "")
        if joint in JOINTS:
            kind = "lift" if joint == LIFT else "arm"
            element.set("kp", str(tuning[kind+"_kp"]))
            element.set("kv", str(tuning[kind+"_kv"]))
            force = tuning[kind+"_force_limit_"+("n" if kind == "lift" else "nm")]
            element.set("forcerange", f"{-force} {force}")
    sensors = root.find("sensor")
    if sensors is None:
        sensors = ET.SubElement(root, "sensor")
    # Ideal encoder emulation for physically instrumentable joints only.
    # No root pose, root velocity, object transform, or contact sensor is added.
    for name in JOINTS:
        ET.SubElement(sensors, "jointpos", name="encoder_position_"+name, joint=name)
        ET.SubElement(sensors, "jointvel", name="encoder_velocity_"+name, joint=name)
    for side in SIDES:
        for jaw in ("positive", "negative"):
            name = f"{side}_jaw_{jaw}"
            ET.SubElement(sensors, "jointpos", name="encoder_position_"+name, joint=name)
    return config
