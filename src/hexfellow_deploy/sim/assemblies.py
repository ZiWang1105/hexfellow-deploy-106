"""Task-independent, namespaced passive MJCF assemblies configured from YAML."""
import copy
import hashlib
from pathlib import Path
import xml.etree.ElementTree as ET

import mujoco
import numpy as np


def numbers(value):
    return " ".join(f"{float(x):.15g}" for x in value)


def _orientation(element):
    q = np.array([1., 0., 0., 0.])
    if element.get("quat"):
        q = np.fromstring(element.attrib.pop("quat"), sep=" ")
    elif element.get("euler"):
        mujoco.mju_euler2Quat(q, np.fromstring(element.attrib.pop("euler"), sep=" "), "xyz")
    elif element.get("zaxis"):
        mujoco.mju_quatZ2Vec(q, np.fromstring(element.attrib.pop("zaxis"), sep=" "))
    elif any(element.get(a) for a in ("axisangle", "xyaxes")):
        raise ValueError("Assembly root orientations must use quat, euler, or zaxis")
    return q / np.linalg.norm(q)


def add_assembly(root, config, spec):
    """Import geometry/passive joints only; no actuators, sensors or attachments."""
    path = config.resolve(spec["mjcf"])
    if hashlib.sha256(path.read_bytes()).hexdigest() != spec["sha256"]:
        raise ValueError(f"Assembly source hash changed: {path}")
    source = ET.parse(path).getroot()
    if source.tag != "mujoco" or any(c.tag not in {"asset", "worldbody"} for c in source):
        raise ValueError("Passive assembly supports only asset and worldbody sections")
    if source.find("worldbody") is None:
        raise ValueError("Passive assembly needs worldbody geometry")
    prefix = f"entity_{spec['name']}_"
    world, assets = root.find("worldbody"), root.find("asset")
    references = {"name", "mesh", "material", "texture"}
    allowed = {"body", "geom", "inertial", "joint", "freejoint", "site", "flexcomp",
               "edge", "contact", "elasticity"}
    for element in source.find("worldbody").iter():
        if element.tag != "worldbody" and element.tag not in allowed:
            raise ValueError(f"Unsupported passive assembly element {element.tag}")
        if element.get("class") or element.get("childclass"):
            raise ValueError("Assembly defaults must be expanded before import")
    for element in source.findall("./asset/*"):
        if element.tag not in {"mesh", "texture", "material"}:
            raise ValueError(f"Unsupported assembly asset {element.tag}")
    offset = np.array(spec["position_m"])
    xyzw = spec["quaternion_xyzw"]
    rotation = np.array([xyzw[3], *xyzw[:3]])
    matrix = np.empty(9)
    mujoco.mju_quat2Mat(matrix, rotation)
    for section, destination in ((source.find("asset"), assets), (source.find("worldbody"), world)):
        if section is None:
            continue
        for original in section:
            element = copy.deepcopy(original)
            for node in element.iter():
                for attribute in references:
                    if node.get(attribute):
                        node.set(attribute, prefix + node.get(attribute))
                if node.get("file"):
                    node.set("file", str((path.parent / node.get("file")).resolve()))
            if destination is world:
                if element.tag not in {"body", "geom", "site", "flexcomp"}:
                    raise ValueError("Assembly worldbody contains a non-geometric root")
                position = np.fromstring(element.get("pos", "0 0 0"), sep=" ")
                element.set("pos", numbers(offset + matrix.reshape(3, 3) @ position))
                result = np.empty(4)
                mujoco.mju_mulQuat(result, rotation, _orientation(element))
                element.set("quat", numbers(result))
            destination.append(element)
    # Only passive scalar joint initial positions; free bodies use their XML poses.
    return {prefix + name: value for name, value in spec.get("initial_joints", {}).items()}
