"""Dual-mode DAS V3 wrist-camera renderer for MuJoCo.

``pinhole`` renders the named wrist cameras directly. ``fisheye`` renders six
90-degree cube faces at the same optical center and resamples those faces with
the OpenCV equidistant model stored in the rig calibration.
"""

from __future__ import annotations

import time
from collections.abc import Mapping
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Literal

import mujoco
import numpy as np
from PIL import Image

from .calibration import (
    DEFAULT_CALIBRATION,
    CameraCalibration,
    DasCameraRigCalibration,
    axis_fov_degrees,
    equidistant_rays,
    link7_from_mujoco_camera,
    link7_from_optical,
    load_calibration,
    matrix_to_quaternion_wxyz,
)

CameraMode = Literal["pinhole", "fisheye"]
CameraBackend = Literal["auto", "reference", "optimized"]
FACE_ORDER = ("front", "right", "left", "down", "up", "back")

# Each tuple contains right, up, and forward axes in OpenCV optical coordinates.
FACE_BASES: Mapping[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {
    "front": (
        np.array([1.0, 0.0, 0.0]),
        np.array([0.0, -1.0, 0.0]),
        np.array([0.0, 0.0, 1.0]),
    ),
    "right": (
        np.array([0.0, 0.0, -1.0]),
        np.array([0.0, -1.0, 0.0]),
        np.array([1.0, 0.0, 0.0]),
    ),
    "left": (
        np.array([0.0, 0.0, 1.0]),
        np.array([0.0, -1.0, 0.0]),
        np.array([-1.0, 0.0, 0.0]),
    ),
    "down": (
        np.array([1.0, 0.0, 0.0]),
        np.array([0.0, 0.0, 1.0]),
        np.array([0.0, 1.0, 0.0]),
    ),
    "up": (
        np.array([1.0, 0.0, 0.0]),
        np.array([0.0, 0.0, -1.0]),
        np.array([0.0, -1.0, 0.0]),
    ),
    "back": (
        np.array([-1.0, 0.0, 0.0]),
        np.array([0.0, -1.0, 0.0]),
        np.array([0.0, 0.0, -1.0]),
    ),
}


@dataclass(frozen=True)
class WristCameraFrame:
    native_rgb: np.ndarray
    policy_rgb: np.ndarray
    timestamp: float
    stream_name: str
    side: str
    mode: CameraMode
    provisional: bool
    resize_backend: str


@dataclass(frozen=True)
class _FaceMap:
    face: str
    map_x: np.ndarray
    map_y: np.ndarray
    mask: np.ndarray


@dataclass(frozen=True)
class _AtlasMap:
    faces: tuple[_FaceMap, ...]
    map_x: np.ndarray
    map_y: np.ndarray


def apply_calibration_to_model(model: mujoco.MjModel, calibration: DasCameraRigCalibration) -> None:
    """Update named pinhole and cube cameras in an already compiled model."""
    for camera in calibration.streams.values():
        main_transform = link7_from_mujoco_camera(camera)
        main_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, f"{camera.side}_wrist")
        if main_id < 0:
            raise ValueError(f"Model has no {camera.side}_wrist camera")
        model.cam_pos[main_id] = main_transform[:3, 3]
        model.cam_quat[main_id] = matrix_to_quaternion_wxyz(main_transform[:3, :3])
        model.cam_fovy[main_id] = axis_fov_degrees(camera)[1]
        optical_transform = link7_from_optical(camera)
        for face, (right, up, forward) in FACE_BASES.items():
            camera_id = mujoco.mj_name2id(
                model,
                mujoco.mjtObj.mjOBJ_CAMERA,
                f"{camera.side}_wrist_cube_{face}",
            )
            if camera_id < 0:
                raise ValueError(f"Model has no {camera.side} cube camera for {face}")
            rotation = optical_transform[:3, :3] @ np.column_stack((right, up, -forward))
            model.cam_pos[camera_id] = optical_transform[:3, 3]
            model.cam_quat[camera_id] = matrix_to_quaternion_wxyz(rotation)


class DasWristCameraRenderer:
    """Render the two calibrated wrist streams from a MuJoCo state."""

    def __init__(
        self,
        model: mujoco.MjModel,
        calibration: DasCameraRigCalibration | Path | None = None,
        *,
        quality: Literal["preview", "policy"] = "policy",
        output_size: tuple[int, int] | None = None,
        backend: CameraBackend = "auto",
        face_size: int | None = None,
        reflections: bool | None = None,
        offsamples: int | None = None,
    ) -> None:
        if calibration is None or isinstance(calibration, (str, Path)):
            calibration = load_calibration(DEFAULT_CALIBRATION if calibration is None else Path(calibration))
        model_calibration = calibration
        if quality not in ("preview", "policy"):
            raise ValueError(f"Unknown wrist-camera quality: {quality}")
        if output_size is not None:
            output_width, output_height = output_size
            native_width, native_height = calibration.native_size
            if output_width <= 0 or output_height <= 0:
                raise ValueError("Camera output dimensions must be positive")
            ratio = np.array(
                [output_width / native_width, output_height / native_height],
                dtype=np.float64,
            )
            streams = {
                name: replace(
                    camera,
                    native_size=output_size,
                    native_scale=camera.native_scale * ratio,
                    native_offset=camera.native_offset * ratio,
                )
                for name, camera in calibration.streams.items()
            }
            calibration = replace(calibration, native_size=output_size, streams=streams)
        self.model = model
        self.calibration = calibration
        self.quality = quality
        if reflections is not None and not isinstance(reflections, bool):
            raise ValueError("Camera reflections must be a boolean")
        self.reflections = quality == "policy" if reflections is None else reflections
        if offsamples is not None and (
            not isinstance(offsamples, int) or isinstance(offsamples, bool) or offsamples < 0
        ):
            raise ValueError("Camera offscreen samples must be a non-negative integer")
        if offsamples is not None:
            model.vis.quality.offsamples = offsamples
        self.offsamples = int(model.vis.quality.offsamples)
        if backend not in ("auto", "reference", "optimized"):
            raise ValueError(f"Unknown camera backend: {backend}")
        self.backend = self._resolve_backend(backend)
        apply_calibration_to_model(model, model_calibration)
        self.face_size = (256 if quality == "preview" else 1024) if face_size is None else face_size
        if not isinstance(self.face_size, int) or isinstance(self.face_size, bool) or self.face_size <= 0:
            raise ValueError("Cube-face size must be positive")
        native_width, native_height = calibration.native_size
        self._native_renderer = mujoco.Renderer(model, height=native_height, width=native_width)
        self._face_renderer = mujoco.Renderer(model, height=self.face_size, width=self.face_size)
        self._option = mujoco.MjvOption()
        self._option.geomgroup[3] = 0
        self._option.geomgroup[4] = 0
        self._face_maps = {camera.stream_name: self._build_face_maps(camera) for camera in calibration.streams.values()}
        self._atlas_maps = {
            stream_name: self._build_atlas_map(face_maps) for stream_name, face_maps in self._face_maps.items()
        }
        self._atlas = np.empty((self.face_size, self.face_size * 5, 3), dtype=np.uint8)
        self._face_pixels = np.empty((self.face_size, self.face_size, 3), dtype=np.uint8)
        self._resize_backend = self._detect_resize_backend()
        self.last_timings: dict[str, float | int] = {}
        self.gl_info = self._query_gl_info()

    @staticmethod
    def _resolve_backend(backend: CameraBackend) -> Literal["reference", "optimized"]:
        if backend == "reference":
            return "reference"
        try:
            import cv2  # noqa: F401
        except ImportError:
            if backend == "optimized":
                raise RuntimeError("optimized camera backend requires OpenCV") from None
            return "reference"
        return "optimized"

    def _query_gl_info(self) -> dict[str, str]:
        result = {"vendor": "unknown", "renderer": "unknown", "version": "unknown"}
        context = getattr(self._face_renderer, "_gl_context", None)
        if type(context).__module__ == "mujoco.egl":
            # Query the actual EGL display, not PyOpenGL's GL dispatch table.
            # Mixed system/Conda GL libraries can segfault in glGetString;
            # Python exception handling cannot contain that native crash.
            # EGL_VERSION is not GL_VERSION, and the GPU model is unknown.
            try:
                from mujoco import egl

                for key, enum in (
                    ("vendor", egl.EGL.EGL_VENDOR),
                    ("egl_version", egl.EGL.EGL_VERSION),
                ):
                    value = egl.EGL.eglQueryString(egl.EGL_DISPLAY, enum)
                    if value:
                        result[key] = value.decode("utf-8", errors="replace")
                result["api"] = "egl"
            except Exception:
                pass
            return result
        try:
            from OpenGL import GL

            if context is not None:
                context.make_current()
            for key, enum in (
                ("vendor", GL.GL_VENDOR),
                ("renderer", GL.GL_RENDERER),
                ("version", GL.GL_VERSION),
            ):
                value = GL.glGetString(enum)
                if value:
                    result[key] = value.decode("utf-8", errors="replace")
        except Exception:
            pass
        return result

    @staticmethod
    def _detect_resize_backend() -> str:
        try:
            import av  # noqa: F401
        except ImportError:
            return "pillow_bilinear"
        return "swscale_bilinear"

    def close(self) -> None:
        self._native_renderer.close()
        self._face_renderer.close()

    def __enter__(self) -> DasWristCameraRenderer:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def _build_face_maps(self, camera: CameraCalibration) -> tuple[_FaceMap, ...]:
        rays, valid = equidistant_rays(camera)
        components = np.abs(rays)
        dominant_axis = np.argmax(components, axis=-1)
        x, y, z = rays[..., 0], rays[..., 1], rays[..., 2]
        selectors = {
            "front": (dominant_axis == 2) & (z >= 0),
            "back": (dominant_axis == 2) & (z < 0),
            "right": (dominant_axis == 0) & (x >= 0),
            "left": (dominant_axis == 0) & (x < 0),
            "down": (dominant_axis == 1) & (y >= 0),
            "up": (dominant_axis == 1) & (y < 0),
        }
        maps: list[_FaceMap] = []
        for face in FACE_ORDER:
            right, up, forward = FACE_BASES[face]
            depth = rays @ forward
            horizontal = np.divide(rays @ right, depth, out=np.zeros_like(depth), where=depth > 1e-8)
            vertical_up = np.divide(rays @ up, depth, out=np.zeros_like(depth), where=depth > 1e-8)
            map_x = (horizontal + 1.0) * 0.5 * (self.face_size - 1)
            map_y = (1.0 - vertical_up) * 0.5 * (self.face_size - 1)
            mask = valid & selectors[face] & (depth > 0)
            map_x[~mask] = -1.0
            map_y[~mask] = -1.0
            maps.append(
                _FaceMap(
                    face=face,
                    map_x=map_x.astype(np.float32),
                    map_y=map_y.astype(np.float32),
                    mask=mask,
                )
            )
        return tuple(maps)

    def _build_atlas_map(self, face_maps: tuple[_FaceMap, ...]) -> _AtlasMap:
        active = tuple(face_map for face_map in face_maps if np.any(face_map.mask))
        if len(active) > 5:
            raise RuntimeError("The calibrated fisheye unexpectedly needs more than five cube faces")
        height, width = face_maps[0].mask.shape
        map_x = np.full((height, width), -1.0, dtype=np.float32)
        map_y = np.full((height, width), -1.0, dtype=np.float32)
        for index, face_map in enumerate(active):
            map_x[face_map.mask] = face_map.map_x[face_map.mask] + index * self.face_size
            map_y[face_map.mask] = face_map.map_y[face_map.mask]
        return _AtlasMap(faces=active, map_x=map_x, map_y=map_y)

    def _resize_policy(self, image: np.ndarray) -> np.ndarray:
        width, height = self.calibration.policy_size
        if self._resize_backend == "swscale_bilinear":
            import av

            frame = av.VideoFrame.from_ndarray(image, format="rgb24")
            return frame.reformat(width=width, height=height, format="rgb24").to_ndarray()
        return np.asarray(Image.fromarray(image).resize((width, height), resample=Image.Resampling.BILINEAR))

    @staticmethod
    def _sample_face(face_pixels: np.ndarray, face_map: _FaceMap) -> np.ndarray:
        """Bilinearly sample one cube face without relying on OpenCV bindings."""
        height, width = face_map.mask.shape
        sampled = np.zeros((height, width, 3), dtype=np.uint8)
        rows, columns = np.nonzero(face_map.mask)
        source_x = np.clip(face_map.map_x[rows, columns], 0.0, face_pixels.shape[1] - 1.0)
        source_y = np.clip(face_map.map_y[rows, columns], 0.0, face_pixels.shape[0] - 1.0)
        x0 = np.floor(source_x).astype(np.int32)
        y0 = np.floor(source_y).astype(np.int32)
        x1 = np.minimum(x0 + 1, face_pixels.shape[1] - 1)
        y1 = np.minimum(y0 + 1, face_pixels.shape[0] - 1)
        wx = (source_x - x0)[:, None]
        wy = (source_y - y0)[:, None]
        top = face_pixels[y0, x0].astype(np.float32) * (1.0 - wx) + face_pixels[y0, x1].astype(np.float32) * wx
        bottom = face_pixels[y1, x0].astype(np.float32) * (1.0 - wx) + face_pixels[y1, x1].astype(np.float32) * wx
        values = top * (1.0 - wy) + bottom * wy
        sampled[rows, columns] = np.clip(np.rint(values), 0, 255).astype(np.uint8)
        return sampled

    def _render_pinhole(self, data: mujoco.MjData, side: str) -> np.ndarray:
        camera_name = f"{side}_wrist"
        self._native_renderer.update_scene(data, camera=camera_name, scene_option=self._option)
        self._apply_render_flags(self._native_renderer.scene)
        return np.asarray(self._native_renderer.render()).copy()

    def _apply_render_flags(self, scene: mujoco.MjvScene) -> None:
        scene.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = False
        scene.flags[mujoco.mjtRndFlag.mjRND_REFLECTION] = self.reflections

    def _render_fisheye_reference(self, data: mujoco.MjData, camera: CameraCalibration) -> np.ndarray:
        width, height = camera.native_size
        output = np.zeros((height, width, 3), dtype=np.uint8)
        for face_map in self._face_maps[camera.stream_name]:
            camera_name = f"{camera.side}_wrist_cube_{face_map.face}"
            self._face_renderer.update_scene(data, camera=camera_name, scene_option=self._option)
            # Each face has a separate shadow frustum; shadows would create
            # discontinuous illumination at cube boundaries after resampling.
            self._apply_render_flags(self._face_renderer.scene)
            face_pixels = self._face_renderer.render(out=self._face_pixels)
            sampled = self._sample_face(face_pixels, face_map)
            output[face_map.mask] = sampled[face_map.mask]
        return output

    def _render_fisheye_optimized(self, data: mujoco.MjData, camera: CameraCalibration) -> np.ndarray:
        import cv2

        atlas_map = self._atlas_maps[camera.stream_name]
        atlas_width = len(atlas_map.faces) * self.face_size
        atlas = self._atlas[:, :atlas_width]
        for index, face_map in enumerate(atlas_map.faces):
            camera_name = f"{camera.side}_wrist_cube_{face_map.face}"
            self._face_renderer.update_scene(data, camera=camera_name, scene_option=self._option)
            self._apply_render_flags(self._face_renderer.scene)
            self._face_renderer.render(out=self._face_pixels)
            atlas[:, index * self.face_size : (index + 1) * self.face_size] = self._face_pixels
        return cv2.remap(
            atlas,
            atlas_map.map_x,
            atlas_map.map_y,
            interpolation=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(0, 0, 0),
        )

    def render(self, data: mujoco.MjData, *, mode: CameraMode = "fisheye") -> dict[str, WristCameraFrame]:
        if mode not in ("pinhole", "fisheye"):
            raise ValueError(f"Unknown camera mode: {mode}")
        started = time.perf_counter()
        render_seconds = 0.0
        resize_seconds = 0.0
        result: dict[str, WristCameraFrame] = {}
        for stream_name, camera in self.calibration.streams.items():
            render_started = time.perf_counter()
            if mode == "pinhole":
                native = self._render_pinhole(data, camera.side)
            elif self.backend == "optimized":
                native = self._render_fisheye_optimized(data, camera)
            else:
                native = self._render_fisheye_reference(data, camera)
            render_seconds += time.perf_counter() - render_started
            resize_started = time.perf_counter()
            policy_rgb = self._resize_policy(native)
            resize_seconds += time.perf_counter() - resize_started
            result[stream_name] = WristCameraFrame(
                native_rgb=native,
                policy_rgb=policy_rgb,
                timestamp=float(data.time),
                stream_name=stream_name,
                side=camera.side,
                mode=mode,
                provisional=camera.provisional,
                resize_backend=self._resize_backend,
            )
        self.last_timings = {
            "render_s": render_seconds,
            "resize_s": resize_seconds,
            "total_s": time.perf_counter() - started,
            "face_size": self.face_size,
            "offsamples": self.offsamples,
            "reflections": self.reflections,
            "mode": mode,
            "faces_per_wrist": (
                len(next(iter(self._atlas_maps.values())).faces)
                if mode == "fisheye" and self.backend == "optimized"
                else len(FACE_ORDER)
                if mode == "fisheye"
                else 1
            ),
        }
        return result
