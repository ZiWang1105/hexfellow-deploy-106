"""Isolated off-screen camera process for mixed GLFW/EGL systems.

MuJoCo selects its OpenGL platform when it is imported.  A desktop viewer needs
GLFW, while the expensive policy cameras are best rendered through headless EGL
on the discrete GPU.  A spawned process gives each side its own GL platform and
also lets the desktop event loop stay responsive while an observation renders.
"""

from __future__ import annotations

import multiprocessing
import os
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Literal

import numpy as np

RenderDevice = Literal["auto", "nvidia", "cpu"]
CameraImageKind = Literal["policy", "native"]
_SPAWN_ENV_LOCK = threading.Lock()


class CameraWorkerError(RuntimeError):
    """Raised when the isolated renderer cannot initialize or answer."""


@dataclass(frozen=True)
class CameraWorkerConfig:
    scene_path: str
    calibration_path: str
    quality: Literal["preview", "policy"]
    mode: Literal["pinhole", "fisheye"]
    backend: Literal["auto", "reference", "optimized"]
    render_device: RenderDevice
    face_size: int | None = None
    egl_device_id: int | None = None
    output_size: tuple[int, int] | None = None
    image_kind: CameraImageKind = "policy"
    reflections: bool | None = None
    offsamples: int | None = None
    include_overview: bool = False


@dataclass(frozen=True)
class CameraWorkerResult:
    images: dict[str, np.ndarray]
    timestamp: float
    metadata: dict[str, object]
    image_kind: CameraImageKind

    @property
    def policy_images(self) -> dict[str, np.ndarray]:
        if self.image_kind != "policy":
            raise CameraWorkerError("camera worker result contains native images, not policy images")
        return self.images


def _configure_gl(config: CameraWorkerConfig) -> None:
    if config.render_device == "cpu":
        os.environ["MUJOCO_GL"] = "osmesa"
        os.environ["PYOPENGL_PLATFORM"] = "osmesa"
        os.environ.pop("MUJOCO_EGL_DEVICE_ID", None)
        return
    os.environ["MUJOCO_GL"] = "egl"
    os.environ["PYOPENGL_PLATFORM"] = "egl"
    if config.egl_device_id is not None:
        os.environ["MUJOCO_EGL_DEVICE_ID"] = str(config.egl_device_id)
    else:
        os.environ.pop("MUJOCO_EGL_DEVICE_ID", None)


def _worker_main(connection: Any, config: CameraWorkerConfig) -> None:
    renderer = None
    overview_renderer = None
    stage = "imports"
    try:
        _configure_gl(config)
        import mujoco

        from .renderer import DasWristCameraRenderer

        stage = "load_scene"
        model = mujoco.MjModel.from_xml_path(config.scene_path)
        data = mujoco.MjData(model)
        state_signature = int(mujoco.mjtState.mjSTATE_INTEGRATION)
        state_size = mujoco.mj_stateSize(model, state_signature)
        stage = "renderer_initialization"
        renderer = DasWristCameraRenderer(
            model,
            config.calibration_path,
            quality=config.quality,
            output_size=config.output_size,
            backend=config.backend,
            face_size=config.face_size,
            reflections=config.reflections,
            offsamples=config.offsamples,
        )
        if config.include_overview:
            overview_renderer = mujoco.Renderer(model, height=224, width=224)
        if config.image_kind not in ("policy", "native"):
            raise ValueError(f"Unknown camera image kind {config.image_kind!r}")
        image_width, image_height = (
            renderer.calibration.policy_size if config.image_kind == "policy" else renderer.calibration.native_size
        )
        renderer_name = " ".join(renderer.gl_info.values()).lower()
        uses_nvidia = "nvidia" in renderer_name
        if config.render_device == "nvidia" and not uses_nvidia:
            raise RuntimeError(f"EGL initialized but did not select NVIDIA; reported renderer is {renderer.gl_info!r}")
        # Context creation alone does not prove that the worker can render.
        # This scratch state is replaced by every submitted state snapshot.
        stage = "startup_frame"
        mujoco.mj_forward(model, data)
        startup_frames = renderer.render(data, mode=config.mode)
        if {frame.side for frame in startup_frames.values()} != {"left", "right"}:
            raise RuntimeError("camera startup frame is missing a wrist stream")
        for frame in startup_frames.values():
            pixels = frame.policy_rgb if config.image_kind == "policy" else frame.native_rgb
            if pixels.shape != (image_height, image_width, 3) or pixels.dtype != np.uint8:
                raise RuntimeError("camera startup frame has invalid shape or dtype")
        connection.send(
            {
                "kind": "ready",
                "state_size": state_size,
                "metadata": {
                    "process": "spawned",
                    "startup_frame_verified": True,
                    "requested_device": config.render_device,
                    "actual_device": "nvidia" if uses_nvidia else "cpu_or_unknown",
                    "gl": renderer.gl_info,
                    "backend": renderer.backend,
                    "face_size": renderer.face_size,
                    "image_kind": config.image_kind,
                    "image_shape": [image_height, image_width, 3],
                    "reflections": renderer.reflections,
                    "offsamples": renderer.offsamples,
                },
            }
        )
        stage = "render_requests"
        while True:
            request = connection.recv()
            if request["kind"] == "close":
                break
            if request["kind"] != "render":
                raise ValueError(f"Unknown camera-worker request {request['kind']!r}")
            state = np.asarray(request["state"], dtype=np.float64)
            if state.shape != (state_size,):
                raise ValueError(f"Camera state has shape {state.shape}, expected {(state_size,)}")
            state_started = time.perf_counter()
            mujoco.mj_setState(model, data, state, state_signature)
            mujoco.mj_forward(model, data)
            state_seconds = time.perf_counter() - state_started
            mode = request.get("mode", config.mode)
            if mode not in ("pinhole", "fisheye"):
                raise ValueError(f"Unknown camera mode {mode!r}")
            frames = renderer.render(data, mode=mode)
            images = {
                frame.side: frame.policy_rgb if config.image_kind == "policy" else frame.native_rgb
                for frame in frames.values()
            }
            if overview_renderer is not None:
                overview_renderer.update_scene(data, camera="overview")
                images["overview"] = np.asarray(overview_renderer.render(), dtype=np.uint8).copy()
            connection.send(
                {
                    "kind": "result",
                    "timestamp": float(data.time),
                    "images": images,
                    "metadata": {
                        "state_sync_s": state_seconds,
                        **renderer.last_timings,
                    },
                }
            )
    except BaseException as error:
        try:
            connection.send({"kind": "error", "error": f"{stage}: {type(error).__name__}: {error}"})
        except Exception:
            pass
    finally:
        if overview_renderer is not None:
            try:
                overview_renderer.close()
            except Exception:
                pass
        if renderer is not None:
            try:
                renderer.close()
            except Exception as error:
                try:
                    connection.send({"kind": "error", "error": f"camera cleanup failed: {error}"})
                except (BrokenPipeError, EOFError, OSError):
                    pass
        connection.close()


class CameraRenderWorker:
    """Own a persistent per-episode renderer in a spawned process."""

    def __init__(self, config: CameraWorkerConfig, *, startup_timeout_s: float = 45.0) -> None:
        if not np.isfinite(startup_timeout_s) or startup_timeout_s <= 0:
            raise ValueError("camera worker startup timeout must be finite and positive")
        context = multiprocessing.get_context("spawn")
        self._connection, child_connection = context.Pipe()
        self._process = context.Process(
            target=_worker_main,
            args=(child_connection, config),
            name="policy-camera-renderer",
            daemon=True,
        )
        # ``spawn`` imports the parent's main module before calling the target.
        # MuJoCo may therefore be imported before ``_worker_main`` gets a chance
        # to choose EGL. Set the environment for the process snapshot, then
        # restore the desktop process immediately after it starts.
        keys = ("MUJOCO_GL", "PYOPENGL_PLATFORM", "MUJOCO_EGL_DEVICE_ID")
        with _SPAWN_ENV_LOCK:
            previous = {key: os.environ.get(key) for key in keys}
            _configure_gl(config)
            try:
                self._process.start()
            finally:
                for key, value in previous.items():
                    if value is None:
                        os.environ.pop(key, None)
                    else:
                        os.environ[key] = value
        child_connection.close()
        try:
            ready = self._receive(startup_timeout_s)
        except Exception:
            self.close()
            raise
        if ready.get("kind") != "ready":
            self.close()
            raise CameraWorkerError(str(ready.get("error", "camera worker failed during startup")))
        self.state_size = int(ready["state_size"])
        self.metadata = dict(ready["metadata"])
        self.image_kind = str(self.metadata["image_kind"])
        if self.image_kind not in ("policy", "native"):
            self.close()
            raise CameraWorkerError(f"invalid worker image kind {self.image_kind!r}")
        self.image_shape = tuple(int(value) for value in self.metadata["image_shape"])
        self.include_overview = bool(config.include_overview)
        self._requested_time: float | None = None

    def _receive(
        self,
        timeout_s: float,
        idle_callback: Callable[[], object] | None = None,
    ) -> dict[str, object]:
        if not np.isfinite(timeout_s) or timeout_s <= 0:
            raise ValueError("camera worker timeout must be finite and positive")
        deadline = time.monotonic() + timeout_s
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise CameraWorkerError(f"camera worker did not answer within {timeout_s:g}s")
            if self._connection.poll(min(1.0 / 60.0, remaining)):
                try:
                    response = self._connection.recv()
                except EOFError as error:
                    self._process.join(timeout=0.05)
                    raise CameraWorkerError(
                        f"camera worker exited without a response (exitcode={self._process.exitcode})"
                    ) from error
                if not isinstance(response, dict):
                    raise CameraWorkerError(f"invalid camera-worker response {type(response).__name__}")
                if response.get("kind") == "error":
                    raise CameraWorkerError(str(response.get("error")))
                return response
            if idle_callback is not None:
                idle_callback()
            if not self._process.is_alive():
                raise CameraWorkerError(f"camera worker exited with code {self._process.exitcode}")

    def render(
        self,
        model: Any,
        data: Any,
        *,
        timeout_s: float = 30.0,
        idle_callback: Callable[[], object] | None = None,
    ) -> CameraWorkerResult:
        self.submit(model, data)
        requested_time = self._requested_time
        assert requested_time is not None
        try:
            response = self._receive(timeout_s, idle_callback)
        except Exception:
            self._requested_time = None
            raise
        return self._decode_result(response, requested_time)

    @property
    def is_busy(self) -> bool:
        return self._requested_time is not None

    def submit(self, model: Any, data: Any, *, mode: Literal["pinhole", "fisheye"] | None = None) -> None:
        """Submit one state without blocking; call :meth:`poll` for its result."""
        import mujoco

        if self.is_busy:
            raise CameraWorkerError("camera worker already has a render request in flight")
        if mode is not None and mode not in ("pinhole", "fisheye"):
            raise ValueError(f"unknown camera mode {mode!r}")
        signature = int(mujoco.mjtState.mjSTATE_INTEGRATION)
        state = np.empty(self.state_size, dtype=np.float64)
        mujoco.mj_getState(model, data, state, signature)
        self._requested_time = float(data.time)
        request: dict[str, object] = {"kind": "render", "state": state}
        if mode is not None:
            request["mode"] = mode
        try:
            self._connection.send(request)
        except Exception:
            self._requested_time = None
            raise

    def poll(self) -> CameraWorkerResult | None:
        """Return the in-flight result when ready without blocking."""
        if not self.is_busy:
            return None
        if not self._connection.poll(0):
            if not self._process.is_alive():
                self._requested_time = None
                raise CameraWorkerError(f"camera worker exited with code {self._process.exitcode}")
            return None
        try:
            response = self._connection.recv()
        except EOFError as error:
            self._requested_time = None
            self._process.join(timeout=0.05)
            raise CameraWorkerError(
                f"camera worker exited without a response (exitcode={self._process.exitcode})"
            ) from error
        if not isinstance(response, dict):
            self._requested_time = None
            raise CameraWorkerError(f"invalid camera-worker response {type(response).__name__}")
        if response.get("kind") == "error":
            self._requested_time = None
            raise CameraWorkerError(str(response.get("error")))
        requested_time = self._requested_time
        assert requested_time is not None
        return self._decode_result(response, requested_time)

    def _decode_result(self, response: dict[str, object], requested_time: float) -> CameraWorkerResult:
        self._requested_time = None
        if response.get("kind") != "result":
            raise CameraWorkerError(f"unexpected camera-worker response {response.get('kind')!r}")
        timestamp = float(response["timestamp"])
        if not np.isclose(timestamp, requested_time, atol=1e-12):
            raise CameraWorkerError(
                f"camera state timestamp mismatch: requested {requested_time}, rendered {timestamp}"
            )
        raw_images = response.get("images")
        expected_sides = {
            "left",
            "right",
            *(("overview",) if self.__dict__.get("include_overview", False) else ()),
        }
        if not isinstance(raw_images, dict) or set(raw_images) != expected_sides:
            raise CameraWorkerError(f"camera worker must return exactly {sorted(expected_sides)} images")
        images = {side: np.asarray(image) for side, image in raw_images.items()}
        for side, image in images.items():
            expected_shape = (224, 224, 3) if side == "overview" else self.image_shape
            if image.dtype != np.uint8 or image.shape != expected_shape:
                raise CameraWorkerError(
                    f"camera worker {side} image must be uint8{expected_shape}, got {image.dtype}{image.shape}"
                )
        metadata = response.get("metadata")
        if not isinstance(metadata, dict):
            raise CameraWorkerError("camera worker result metadata must be a mapping")
        return CameraWorkerResult(
            images=images,
            timestamp=timestamp,
            metadata={**self.metadata, **metadata},
            image_kind=self.image_kind,
        )

    def close(self) -> None:
        connection = getattr(self, "_connection", None)
        process = getattr(self, "_process", None)
        if connection is not None:
            try:
                if process is not None and process.is_alive():
                    connection.send({"kind": "close"})
            except (BrokenPipeError, EOFError, OSError):
                pass
        if process is not None:
            deadline = time.monotonic() + 2.0
            while (
                connection is not None
                and getattr(self, "_requested_time", None) is not None
                and process.is_alive()
                and time.monotonic() < deadline
            ):
                try:
                    if connection.poll(0.05):
                        connection.recv()
                        self._requested_time = None
                except (BrokenPipeError, EOFError, OSError):
                    break
            process.join(timeout=max(0.0, deadline - time.monotonic()))
            if process.is_alive():
                process.terminate()
                process.join(timeout=2.0)
        if connection is not None:
            connection.close()
        self._connection = None
        self._process = None
        self._requested_time = None

    def __enter__(self) -> CameraRenderWorker:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()


def start_camera_render_worker(config: CameraWorkerConfig) -> CameraRenderWorker:
    """Start the best isolated renderer while making explicit NVIDIA requests strict."""
    candidates: list[tuple[CameraWorkerConfig, float]] = []
    nvidia_visible = Path("/dev/nvidia0").exists()
    if config.render_device in ("auto", "nvidia") and nvidia_visible:
        candidates.extend(
            (replace(config, render_device="nvidia", egl_device_id=device_id), 12.0)
            for device_id in (None, 0, 1, 2, 3)
        )
        if config.render_device == "auto":
            candidates.append((replace(config, render_device="cpu", egl_device_id=None), 30.0))
    else:
        candidates.append((config, 30.0))

    errors: list[str] = []
    for candidate, timeout_s in candidates:
        try:
            worker = CameraRenderWorker(candidate, startup_timeout_s=timeout_s)
            worker.metadata["requested_device"] = config.render_device
            worker.metadata["selected_candidate_device"] = candidate.render_device
            if errors:
                worker.metadata["device_selection_errors"] = tuple(errors)
            return worker
        except CameraWorkerError as error:
            device = candidate.render_device
            if candidate.egl_device_id is not None:
                device = f"{device}:{candidate.egl_device_id}"
            errors.append(f"{device}: {error}")
    raise CameraWorkerError("; ".join(errors) if errors else "camera worker has no render-device candidate")
