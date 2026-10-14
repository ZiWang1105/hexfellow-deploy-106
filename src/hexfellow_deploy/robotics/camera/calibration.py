"""Calibration and coordinate helpers for the DAS Controller V3 wrist cameras.

The public transform convention is deliberately explicit:
``T_das_base_from_optical`` maps OpenCV optical-frame points (+X right, +Y
down, +Z forward) into the DAS ``robot_base`` frame.  MuJoCo cameras use +X
right, +Y up and look along -Z, so only the camera-frame basis changes; the
optical center does not.
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from ..paths import CONFIG_DIR

DEFAULT_CALIBRATION = CONFIG_DIR / "das_wrist_cameras.json"

# Fixed joints from link-7 to the DAS base in urdf/maver_x4_asm_asm.urdf.
T_LINK7_MOUNT = np.array([0.02348191, 0.0, 0.042], dtype=np.float64)
T_MOUNT_DAS_BASE = np.array([-0.0767216914704, 0.000038794, 0.0908414417539], dtype=np.float64)
R_MOUNT_DAS_BASE_Y = -1.30899633

# OpenCV optical coordinates -> the equivalent MuJoCo camera frame.
R_OPTICAL_FROM_MUJOCO_CAMERA = np.diag([1.0, -1.0, -1.0])


@dataclass(frozen=True)
class CameraCalibration:
    stream_name: str
    side: str
    frame_id: str
    distortion_model: str
    calibration_size: tuple[int, int]
    native_size: tuple[int, int]
    policy_size: tuple[int, int]
    K: np.ndarray
    D: np.ndarray
    translation: np.ndarray
    quaternion_xyzw: np.ndarray
    native_scale: np.ndarray
    native_offset: np.ndarray
    provisional: bool
    inherited_from: str | None = None

    @property
    def native_K(self) -> np.ndarray:
        """Return K mapped from calibration pixels into native video pixels."""
        matrix = self.K.copy()
        sx, sy = self.native_scale
        ox, oy = self.native_offset
        matrix[0, 0] *= sx
        matrix[1, 1] *= sy
        matrix[0, 2] = matrix[0, 2] * sx + ox
        matrix[1, 2] = matrix[1, 2] * sy + oy
        return matrix

    @property
    def rotation_das_from_optical(self) -> np.ndarray:
        return quaternion_xyzw_to_matrix(self.quaternion_xyzw)

    @property
    def transform_das_from_optical(self) -> np.ndarray:
        transform = np.eye(4)
        transform[:3, :3] = self.rotation_das_from_optical
        transform[:3, 3] = self.translation
        return transform


@dataclass(frozen=True)
class DasCameraRigCalibration:
    source: str
    native_size: tuple[int, int]
    policy_size: tuple[int, int]
    resize_filter: str
    streams: Mapping[str, CameraCalibration]

    def for_side(self, side: str) -> CameraCalibration:
        matches = [camera for camera in self.streams.values() if camera.side == side]
        if len(matches) != 1:
            raise ValueError(f"Expected exactly one {side} camera, found {len(matches)}")
        return matches[0]


def rotation_y(angle: float) -> np.ndarray:
    cosine, sine = math.cos(angle), math.sin(angle)
    return np.array(
        [[cosine, 0.0, sine], [0.0, 1.0, 0.0], [-sine, 0.0, cosine]],
        dtype=np.float64,
    )


def quaternion_xyzw_to_matrix(quaternion: np.ndarray) -> np.ndarray:
    x, y, z, w = np.asarray(quaternion, dtype=np.float64)
    norm = math.sqrt(float(x * x + y * y + z * z + w * w))
    if norm < 1e-12:
        raise ValueError("Camera quaternion has zero norm")
    x, y, z, w = x / norm, y / norm, z / norm, w / norm
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def matrix_to_quaternion_wxyz(matrix: np.ndarray) -> np.ndarray:
    """Convert a rotation matrix into MuJoCo's wxyz quaternion order."""
    matrix = np.asarray(matrix, dtype=np.float64)
    trace = float(np.trace(matrix))
    if trace > 0.0:
        scale = math.sqrt(trace + 1.0) * 2.0
        w = 0.25 * scale
        x = (matrix[2, 1] - matrix[1, 2]) / scale
        y = (matrix[0, 2] - matrix[2, 0]) / scale
        z = (matrix[1, 0] - matrix[0, 1]) / scale
    else:
        axis = int(np.argmax(np.diag(matrix)))
        if axis == 0:
            scale = math.sqrt(1.0 + matrix[0, 0] - matrix[1, 1] - matrix[2, 2]) * 2.0
            w = (matrix[2, 1] - matrix[1, 2]) / scale
            x = 0.25 * scale
            y = (matrix[0, 1] + matrix[1, 0]) / scale
            z = (matrix[0, 2] + matrix[2, 0]) / scale
        elif axis == 1:
            scale = math.sqrt(1.0 + matrix[1, 1] - matrix[0, 0] - matrix[2, 2]) * 2.0
            w = (matrix[0, 2] - matrix[2, 0]) / scale
            x = (matrix[0, 1] + matrix[1, 0]) / scale
            y = 0.25 * scale
            z = (matrix[1, 2] + matrix[2, 1]) / scale
        else:
            scale = math.sqrt(1.0 + matrix[2, 2] - matrix[0, 0] - matrix[1, 1]) * 2.0
            w = (matrix[1, 0] - matrix[0, 1]) / scale
            x = (matrix[0, 2] + matrix[2, 0]) / scale
            y = (matrix[1, 2] + matrix[2, 1]) / scale
            z = 0.25 * scale
    quaternion = np.array([w, x, y, z], dtype=np.float64)
    return quaternion / np.linalg.norm(quaternion)


def link7_from_das_base() -> np.ndarray:
    transform = np.eye(4)
    transform[:3, :3] = rotation_y(R_MOUNT_DAS_BASE_Y)
    transform[:3, 3] = T_LINK7_MOUNT + T_MOUNT_DAS_BASE
    return transform


def link7_from_optical(camera: CameraCalibration) -> np.ndarray:
    return link7_from_das_base() @ camera.transform_das_from_optical


def link7_from_mujoco_camera(camera: CameraCalibration) -> np.ndarray:
    transform = link7_from_optical(camera)
    result = transform.copy()
    result[:3, :3] = transform[:3, :3] @ R_OPTICAL_FROM_MUJOCO_CAMERA
    return result


def _resolved_stream_data(raw_streams: Mapping[str, Any], name: str) -> tuple[dict[str, Any], str | None]:
    raw = dict(raw_streams[name])
    inherited_from = raw.pop("inherits", None)
    if inherited_from is None:
        return raw, None
    if inherited_from == name or inherited_from not in raw_streams:
        raise ValueError(f"Invalid camera inheritance for {name}: {inherited_from}")
    parent, _ = _resolved_stream_data(raw_streams, inherited_from)
    parent.update(raw)
    return parent, inherited_from


def load_calibration(path: Path = DEFAULT_CALIBRATION) -> DasCameraRigCalibration:
    path = Path(path).resolve()
    with path.open("r", encoding="utf-8") as handle:
        raw = json.load(handle)
    if raw.get("schema_version") != 1:
        raise ValueError(f"Unsupported DAS camera schema: {raw.get('schema_version')}")
    native_size = tuple(int(value) for value in raw["native_resolution"])
    policy_size = tuple(int(value) for value in raw["policy_resolution"])
    streams: dict[str, CameraCalibration] = {}
    for name in raw["streams"]:
        values, inherited_from = _resolved_stream_data(raw["streams"], name)
        transform = values["T_das_base_from_optical"]
        mapping = values.get("native_mapping", {})
        calibration_size = tuple(int(value) for value in values["calibration_resolution"])
        default_scale = np.asarray(native_size, dtype=np.float64) / np.asarray(calibration_size, dtype=np.float64)
        streams[name] = CameraCalibration(
            stream_name=name,
            side=str(values["side"]),
            frame_id=str(values["frame_id"]),
            distortion_model=str(values["distortion_model"]),
            calibration_size=calibration_size,
            native_size=native_size,
            policy_size=policy_size,
            K=np.asarray(values["K"], dtype=np.float64).reshape(3, 3),
            D=np.asarray(values["D"], dtype=np.float64).reshape(4),
            translation=np.asarray(transform["translation_m"], dtype=np.float64).reshape(3),
            quaternion_xyzw=np.asarray(transform["quaternion_xyzw"], dtype=np.float64).reshape(4),
            native_scale=np.asarray(mapping.get("scale_xy", default_scale), dtype=np.float64),
            native_offset=np.asarray(mapping.get("offset_xy", (0.0, 0.0)), dtype=np.float64),
            provisional=bool(values.get("provisional", False)),
            inherited_from=inherited_from,
        )
    sides = [camera.side for camera in streams.values()]
    if sorted(sides) != ["left", "right"]:
        raise ValueError(f"Expected one left and one right camera, got {sides}")
    return DasCameraRigCalibration(
        source=str(raw.get("source", "")),
        native_size=native_size,
        policy_size=policy_size,
        resize_filter=str(raw.get("resize_filter", "swscale_bilinear")),
        streams=streams,
    )


def equidistant_rays(camera: CameraCalibration) -> tuple[np.ndarray, np.ndarray]:
    """Return native-resolution optical rays and the monotonic-model mask."""
    width, height = camera.native_size
    intrinsic = camera.native_K
    columns, rows = np.meshgrid(np.arange(width, dtype=np.float64), np.arange(height, dtype=np.float64))
    xd = (columns - intrinsic[0, 2]) / intrinsic[0, 0]
    yd = (rows - intrinsic[1, 2]) / intrinsic[1, 1]
    theta_distorted = np.hypot(xd, yd)
    k1, k2, k3, k4 = camera.D

    # Locate the end of the monotonic branch instead of Newton-iterating into
    # a second, non-physical polynomial solution near the image corners.
    theta_grid = np.linspace(0.0, math.pi, 20001)
    theta2 = theta_grid * theta_grid
    derivative = 1 + 3 * k1 * theta2 + 5 * k2 * theta2**2 + 7 * k3 * theta2**3 + 9 * k4 * theta2**4
    nonpositive = np.flatnonzero(derivative <= 0.0)
    theta_limit = theta_grid[nonpositive[0]] if len(nonpositive) else math.pi
    limit2 = theta_limit * theta_limit
    distorted_limit = theta_limit * (1 + k1 * limit2 + k2 * limit2**2 + k3 * limit2**3 + k4 * limit2**4)
    valid = theta_distorted <= distorted_limit

    theta = np.minimum(theta_distorted, distorted_limit).copy()
    for _ in range(8):
        theta2 = theta * theta
        polynomial = 1 + k1 * theta2 + k2 * theta2**2 + k3 * theta2**3 + k4 * theta2**4
        derivative = 1 + 3 * k1 * theta2 + 5 * k2 * theta2**2 + 7 * k3 * theta2**3 + 9 * k4 * theta2**4
        theta -= (theta * polynomial - theta_distorted) / np.maximum(derivative, 1e-9)
        theta = np.clip(theta, 0.0, theta_limit)

    radial_scale = np.divide(np.sin(theta), theta_distorted, out=np.ones_like(theta), where=theta_distorted > 1e-12)
    rays = np.stack((xd * radial_scale, yd * radial_scale, np.cos(theta)), axis=-1)
    rays[~valid] = 0.0
    return rays.astype(np.float32), valid


def axis_fov_degrees(camera: CameraCalibration) -> tuple[float, float]:
    """Compute full left+right and top+bottom FOV on the central image axes."""
    rays, valid = equidistant_rays(camera)
    width, height = camera.native_size
    center_x = int(round(float(camera.native_K[0, 2])))
    center_y = int(round(float(camera.native_K[1, 2])))
    samples = (
        rays[center_y, 0],
        rays[center_y, width - 1],
        rays[0, center_x],
        rays[height - 1, center_x],
    )
    sample_valid = (valid[center_y, 0], valid[center_y, width - 1], valid[0, center_x], valid[height - 1, center_x])
    if not all(sample_valid):
        raise ValueError("Central image axes extend outside the valid fisheye model")
    angles = [math.degrees(math.acos(float(np.clip(ray[2], -1.0, 1.0)))) for ray in samples]
    return angles[0] + angles[1], angles[2] + angles[3]
