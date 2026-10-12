"""Robot-only FK and Jacobians. This model has no scene objects."""
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
import pinocchio as pin
from scipy.spatial.transform import Rotation

from .assembly import JOINTS, LEFT, RIGHT, values
from .joints import SIDES


class Kinematics:
    def __init__(self, urdf: Path, assembly):
        self.gripper_geometry = dict(assembly["gripper"])
        xml = ET.parse(urdf).getroot()
        mount = xml.find("joint[@name='zed_bracket_mount']/origin")
        mount.set("xyz", values(assembly["camera_mount"]["position_m"]))
        self.model = pin.buildModelFromXML(ET.tostring(xml, encoding="unicode"))
        self.tcp_frames = {}
        for side, tool in (("left", "robot_base"), ("right", "robot2_base")):
            parent = self.model.frames[self.model.getFrameId(tool)]
            self.tcp_frames[side] = self.model.addFrame(pin.Frame(
                side + "_tcp", parent.parent, self.model.getFrameId(tool),
                parent.placement * pin.SE3(np.eye(3), np.array(assembly["gripper"]["tcp_in_tool_m"])),
                pin.FrameType.OP_FRAME))
        self.data = self.model.createData()
        self.indices = {name: self.model.joints[self.model.getJointId(name)].idx_q for name in JOINTS}
        self.arm_indices = {s: np.array([self.indices[n] for n in names])
                            for s, names in (("left", LEFT), ("right", RIGHT))}
        self.camera_frame = self.model.getFrameId(assembly["camera"]["optical_frame"])
        self.camera_offset = pin.SE3(np.eye(3), np.array([0., 0., assembly["camera"]["lens_forward_offset_m"]]))

    def update(self, joints):
        self.q = np.zeros(self.model.nq)
        for name, value in joints.items():
            if name in self.indices:
                self.q[self.indices[name]] = value
        pin.forwardKinematics(self.model, self.data, self.q)
        pin.updateFramePlacements(self.model, self.data)

    def pose(self, side):
        return self.data.oMf[self.tcp_frames[side]].copy()

    def camera_pose(self):
        return self.data.oMf[self.camera_frame] * self.camera_offset

    def jacobian(self, side):
        return pin.computeFrameJacobian(self.model, self.data, self.q, self.tcp_frames[side],
                                        pin.ReferenceFrame.LOCAL_WORLD_ALIGNED)[:, self.arm_indices[side]]

    def packet(self):
        poses = {side: self.pose(side) for side in SIDES}
        return {"tcp_poses": {side: {
                    "position_m": pose.translation.tolist(),
                    "quaternion_xyzw": Rotation.from_matrix(pose.rotation).as_quat().tolist()}
                    for side, pose in poses.items()},
                "camera_T_base_from_optical": self.camera_pose().homogeneous.tolist()}
