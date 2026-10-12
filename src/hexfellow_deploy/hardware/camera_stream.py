"""One camera owner per stream; consumers copy the latest RGB-D frame.

No robot SDK, controller, or policy imports. Device creation/open/read/close all
happen on the capture thread. Slow consumers never queue old camera frames.
"""
from dataclasses import dataclass
import threading
import time

import numpy as np

from ..interfaces.depth import MAX_DEPTH_M, validate_depth


@dataclass
class CameraFrame:
    rgb: np.ndarray
    depth_m: np.ndarray
    received_s: float  # host monotonic receipt time, NOT synchronized robot time
    device_timestamp_s: float
    intrinsics: dict


def clean_depth(depth):
    depth = np.array(depth, dtype=np.float32, copy=True)
    depth[~np.isfinite(depth) | (depth < 0) | (depth > MAX_DEPTH_M)] = 0
    return depth


class CameraStream:
    def __init__(self, factory, *, fps=15, stale_s=2):
        if not 0 < fps <= 120 or not 0 < stale_s <= 60:
            raise ValueError("Invalid capture rate or stale timeout")
        self.factory, self.fps, self.stale_s = factory, fps, stale_s
        self.lock = threading.Lock()
        self.stop_event = threading.Event()
        self.thread = None
        self.frame = None
        self.sequence = 0
        self.error = None

    def start(self):
        if self.thread is not None:
            raise RuntimeError("Camera stream already started")
        self.thread = threading.Thread(target=self._capture, name="rgbd-capture", daemon=True)
        self.thread.start()
        return self

    def _capture(self):
        source = None
        try:
            source = self.factory()
            source.open()
            while not self.stop_event.is_set():
                started = time.monotonic()
                frame = source.read()
                if frame is not None:
                    if (frame.rgb.dtype != np.uint8 or frame.rgb.ndim != 3
                            or frame.rgb.shape[2] != 3):
                        raise ValueError("Camera must supply uint8 RGB")
                    validate_depth(frame.depth_m, frame.rgb.shape)
                    if not np.isfinite([frame.received_s, frame.device_timestamp_s]).all():
                        raise ValueError("Invalid camera timestamp")
                    # SDK buffers are reused; publication owns an independent copy.
                    owned = CameraFrame(frame.rgb.copy(), frame.depth_m.copy(),
                                        frame.received_s, frame.device_timestamp_s,
                                        dict(frame.intrinsics))
                    with self.lock:
                        self.frame = owned
                        self.sequence += 1
                self.stop_event.wait(max(0, 1 / self.fps - (time.monotonic() - started)))
        except Exception as error:
            with self.lock:
                self.error = f"{type(error).__name__}: {error}"
        finally:
            if source is not None:
                try:
                    source.close()
                except Exception as error:
                    with self.lock:
                        self.error = f"Camera close failed: {error}"

    def health(self):
        """Small metadata only: watchdogs must never copy or encode pixels."""
        with self.lock:
            frame = self.frame
            age = None if frame is None else time.monotonic() - frame.received_s
            available = (frame is not None and self.error is None and age <= self.stale_s
                         and not self.stop_event.is_set())
            return dict(sequence=self.sequence, age_s=age, available=available,
                        received_s=None if frame is None else frame.received_s,
                        error=self.error, running=self.thread is not None and self.thread.is_alive())

    def snapshot(self):
        # Published arrays are owned by this stream and never modified. Retain
        # their reference under the lock, copy outside it for mutable consumers.
        with self.lock:
            frame = self.frame
            age = None if frame is None else time.monotonic() - frame.received_s
            available = (frame is not None and self.error is None and age <= self.stale_s
                         and not self.stop_event.is_set())
            status = dict(sequence=self.sequence, age_s=age, available=available,
                          error=self.error, running=self.thread is not None and self.thread.is_alive())
            if not available:
                return status, None
        return status, CameraFrame(frame.rgb.copy(), frame.depth_m.copy(), frame.received_s,
                                   frame.device_timestamp_s, dict(frame.intrinsics))

    def close(self):
        self.stop_event.set()
        if self.thread is not None:
            self.thread.join(timeout=5)
            if self.thread.is_alive():
                raise RuntimeError("Camera SDK did not finish capture within shutdown timeout")


class ZedSource:
    def __init__(self, config):
        self.config, self.camera = config, None

    def open(self):
        import pyzed.sl as sl
        self.sl = sl
        init = sl.InitParameters()
        init.camera_resolution = getattr(sl.RESOLUTION, self.config.get("resolution", "HD720"))
        init.camera_fps = int(self.config.get("fps", 15))
        init.coordinate_units = sl.UNIT.METER
        init.depth_mode = sl.DEPTH_MODE.NEURAL
        if self.config.get('sdk_root'):
            from pathlib import Path
            init.optional_settings_path = str(Path(self.config['sdk_root']) / 'settings')
        if self.config.get("serial"):
            init.set_from_serial_number(int(self.config["serial"]))
        self.camera = sl.Camera()
        result = self.camera.open(init)
        if result != sl.ERROR_CODE.SUCCESS:
            raise RuntimeError(f"ZED open failed: {result}")
        self.image, self.depth = sl.Mat(), sl.Mat()
        self.runtime = sl.RuntimeParameters()
        cfg = self.camera.get_camera_information().camera_configuration
        k = cfg.calibration_parameters.left_cam
        self.intrinsics = dict(fx=k.fx, fy=k.fy, cx=k.cx, cy=k.cy,
                               width=cfg.resolution.width, height=cfg.resolution.height,
                               frame="rectified_left_optical")

    def read(self):
        sl = self.sl
        result = self.camera.grab(self.runtime)
        if result != sl.ERROR_CODE.SUCCESS:
            raise RuntimeError(f"ZED capture failed: {result}")
        for result in (self.camera.retrieve_image(self.image, sl.VIEW.LEFT),
                       self.camera.retrieve_measure(self.depth, sl.MEASURE.DEPTH)):
            if result != sl.ERROR_CODE.SUCCESS:
                raise RuntimeError(f"ZED retrieve failed: {result}")
        # SDK image is BGRA. Depth is registered optical-axis Z in meters.
        rgb = self.image.get_data()[:, :, :3][:, :, ::-1].copy()
        timestamp = self.camera.get_timestamp(sl.TIME_REFERENCE.IMAGE).get_nanoseconds() / 1e9
        return CameraFrame(rgb, clean_depth(self.depth.get_data()), time.monotonic(),
                           timestamp, self.intrinsics)

    def close(self):
        if self.camera is not None:
            self.camera.close()


class RealSenseSource:
    def __init__(self, config):
        self.config, self.pipeline, self.started = config, None, False

    def open(self):
        import pyrealsense2 as rs
        self.pipeline = rs.pipeline()
        cfg = rs.config()
        if self.config.get("serial"):
            cfg.enable_device(str(self.config["serial"]))
        w, h = self.config.get("size_wh", [640, 480])
        fps = int(self.config.get("fps", 15))
        cfg.enable_stream(rs.stream.color, w, h, rs.format.rgb8, fps)
        cfg.enable_stream(rs.stream.depth, w, h, rs.format.z16, fps)
        profile = self.pipeline.start(cfg)
        self.started = True
        self.scale = profile.get_device().first_depth_sensor().get_depth_scale()
        self.align = rs.align(rs.stream.color)

    def read(self):
        frames = self.align.process(self.pipeline.wait_for_frames(timeout_ms=1000))
        color, depth = frames.get_color_frame(), frames.get_depth_frame()
        if not color or not depth:
            raise RuntimeError("RealSense missing registered RGB/depth pair")
        k = color.profile.as_video_stream_profile().intrinsics
        import cv2
        rgb = np.asanyarray(color.get_data()).copy()
        depth_m = clean_depth(np.asanyarray(depth.get_data()) * self.scale)
        # Publish a pinhole image pair, matching the policy/simulator contract.
        if any(abs(x) > 1e-12 for x in k.coeffs):
            if str(k.model) not in ('distortion.brown_conrady', 'distortion.modified_brown_conrady'):
                raise RuntimeError('Unsupported RealSense color distortion; require rectified pinhole RGB-D')
            if not hasattr(self, 'rectification'):
                matrix = np.array([[k.fx, 0, k.ppx], [0, k.fy, k.ppy], [0, 0, 1.]])
                self.rectification = cv2.initUndistortRectifyMap(matrix, np.array(k.coeffs),
                    None, matrix, (k.width, k.height), cv2.CV_32FC1)
            rgb = cv2.remap(rgb, *self.rectification, interpolation=cv2.INTER_LINEAR)
            depth_m = cv2.remap(depth_m, *self.rectification, interpolation=cv2.INTER_NEAREST)
        intrinsics = dict(fx=k.fx, fy=k.fy, cx=k.ppx, cy=k.ppy, width=k.width, height=k.height,
                          distortion_model='rectified_pinhole', coefficients=[0.] * 5,
                          frame="color_optical")
        return CameraFrame(rgb, depth_m,
                           time.monotonic(), color.get_timestamp() / 1000, intrinsics)

    def close(self):
        if self.started:
            self.pipeline.stop()
            self.started = False


class SyntheticSource:
    """Explicit software-only preview fixture; never presented as hardware."""
    def open(self):
        self.counter = 0

    def read(self):
        self.counter += 1
        rgb = np.zeros((120, 160, 3), dtype=np.uint8)
        rgb[:, :, 1] = self.counter % 256
        depth = np.full((120, 160), 1.25, dtype=np.float32)
        now = time.monotonic()
        return CameraFrame(rgb, depth, now, now, dict(frame="synthetic", width=160, height=120))

    def close(self):
        pass


def make_source(config):
    sources = {"zed": ZedSource, "realsense": RealSenseSource}
    if config.get("type") not in sources:
        raise ValueError("Camera type must be zed or realsense")
    if config.get('python') or config.get('sdk_root'):
        from .camera_process import ExternalCameraSource
        return ExternalCameraSource(config)
    return sources[config["type"]](config)
