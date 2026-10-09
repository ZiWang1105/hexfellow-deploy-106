#!/usr/bin/env python3
"""Owned RealSense wrist RGB process for calibration; never controls the robot.

IPC matches calibration_camera_worker.py: ready.json, status.json, preview.jpg,
request.json {id, after, q}, <id>/left.png and result.json, and a stop file.
The left.png name is historical IPC compatibility, not a stereo-left image.

Live captures use the SDK colour optical frame. GLOBAL_TIME is a host-corrected
device clock, not hardware-trigger synchronization. Sensor metadata corrects
the SDK readout timestamp to the exposure midpoint. The uncertainty reported
below is an engineering estimate with a declared SDK clock-mapping allowance,
not a certified error bound. Unsupported/unreliable timing makes a capture
ineligible, while keeping its raw image available for inspection.
"""
import argparse
from collections import deque
from datetime import datetime, timezone
import json
from pathlib import Path
import signal
import sys
import threading
import time

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src'))
from policy_sim.calibration.charuco import detect

WIDTH, HEIGHT, FPS = 1280, 720, 30
DEFAULT_SERIAL = '243222074562'
DEFAULT_K = np.array([[909.8963623046875, 0., 643.9840698242188],
                      [0., 909.7846069335938, 373.11004638671875],
                      [0., 0., 1.]])
TIMING_ALLOWANCE_S = .002
MAX_UNCERTAINTY_S = .005
MAX_FRAME_AGE_S = .150


def atomic(path, value):
    """Only the owning worker writes these files."""
    path = Path(path)
    tmp = path.with_name(path.name + '.tmp')
    tmp.write_text(json.dumps(value, allow_nan=False, indent=2) + '\n')
    tmp.replace(path)


def save_image(path, bgr):
    path = Path(path)
    tmp = path.with_name(path.stem + '.tmp' + path.suffix)
    if not cv2.imwrite(str(tmp), bgr):
        raise RuntimeError(f'Image save failed: {path}')
    tmp.replace(path)


def clock_bridge():
    """Bracket realtime with monotonic; never equate receipt with exposure."""
    before = time.monotonic()
    real = time.time()
    after = time.monotonic()
    return dict(realtime_s=real, monotonic_s=(before + after) / 2,
                uncertainty_s=(after - before) / 2)


def signed_counter_delta_us(a, b):
    """D400 metadata uses a 32-bit microsecond counter (~71.6 min wrap)."""
    return ((int(a) - int(b) + 2**31) % 2**32) - 2**31


class ExposureClock:
    """Validate the SDK exposure-to-host clock mapping without a receipt fit.

    A receipt-time linear fit cannot identify USB transport delay, so it is
    deliberately not used as a hardware-clock fallback. A host SYSTEM_TIME
    frame may be the SDK's UVC fallback timestamp and is also ineligible.
    """
    def __init__(self, warmup_s=2., allowance_s=TIMING_ALLOWANCE_S):
        self.warmup_s = warmup_s
        self.allowance_s = allowance_s
        self.samples = deque(maxlen=90)
        self.bridges = deque(maxlen=90)
        self.start = None
        self.previous_frame = None

    def map(self, timestamp_ms, domain, metadata, received, bridge, frame_number):
        reason = None
        base = dict(image_time=received, image_timestamp_ns=int(round(timestamp_ms * 1e6)),
                    timestamp_domain=domain, host_receipt_monotonic_s=received,
                    host_clock_bridge=bridge, synchronization='stationary host-time pairing; no hardware exposure trigger',
                    timestamp_semantics='host receipt only; exposure mapping unavailable',
                    clock_mapping_uncertainty_s=1., timing_accepted=False,
                    sdk_clock_mapping_allowance_s=self.allowance_s,
                    uncertainty_is_estimate=True)
        if 'global_time' not in domain.lower():
            reason = 'Exposure pairing needs SDK GLOBAL_TIME and sensor metadata; receipt/system/hardware fallback is not accepted'
        elif not all(name in metadata for name in ('frame_timestamp', 'sensor_timestamp')):
            reason = 'Exposure-center sensor_timestamp/frame_timestamp metadata unavailable'
        if reason:
            base['timing_reason'] = reason
            return base

        readout_s = timestamp_ms / 1000.
        delta = signed_counter_delta_us(metadata['sensor_timestamp'], metadata['frame_timestamp']) / 1e6
        frame_time = readout_s + delta + bridge['monotonic_s'] - bridge['realtime_s']
        base.update(image_time=frame_time, exposure_midpoint_timestamp_ns=int(round((readout_s + delta) * 1e9)),
                    timestamp_semantics='SDK GLOBAL_TIME readout plus device exposure-midpoint metadata correction',
                    exposure_midpoint_from_readout_s=delta,
                    frame_age_at_receipt_s=received-frame_time)
        if not np.isfinite(frame_time) or abs(delta) > .150:
            base['timing_reason'] = 'Invalid exposure metadata correction'
            return base
        if self.previous_frame is not None and frame_number <= self.previous_frame:
            base['timing_reason'] = 'Repeated/backward RGB frame number'
            return base
        self.previous_frame = frame_number
        if self.start is None:
            self.start = received
        # Unwrap the metadata counter on the SDK epoch before comparing offsets.
        raw_s = metadata['frame_timestamp'] / 1e6
        raw_s += round((readout_s - raw_s) / (2**32 / 1e6)) * (2**32 / 1e6)
        sdk_offset_s = readout_s - raw_s
        self.samples.append((received, sdk_offset_s))
        self.bridges.append(bridge['monotonic_s'] - bridge['realtime_s'])
        sample_array = np.asarray(self.samples)
        sdk_jitter = 1.
        drift = None
        if len(self.samples) >= 15 and sample_array[-1, 0]-sample_array[0, 0] >= .4:
            x = sample_array[:, 0] - sample_array[:, 0].mean()
            y = sample_array[:, 1] - sample_array[:, 1].mean()
            drift = float(np.dot(x, y) / np.dot(x, x))
            sdk_jitter = float(np.max(np.abs(y - drift*x)))
        host_variation = float(np.ptp(self.bridges))
        uncertainty = self.allowance_s + sdk_jitter + host_variation + bridge['uncertainty_s'] + 1e-6
        base.update(clock_mapping_uncertainty_s=uncertainty,
                    sdk_mapping_window_residual_s=sdk_jitter,
                    sdk_mapping_drift_s_per_s=drift,
                    host_bridge_window_variation_s=host_variation,
                    clock_mapping_samples=len(self.samples))
        if received-self.start < self.warmup_s:
            reason = 'Warming up the SDK exposure clock mapping'
        elif frame_time > received + uncertainty:
            reason = 'Mapped exposure timestamp is later than frame receipt'
        elif received-frame_time > MAX_FRAME_AGE_S:
            reason = 'RGB exposure is more than 150 ms old at receipt'
        elif uncertainty > MAX_UNCERTAINTY_S:
            reason = 'Estimated camera clock mapping uncertainty exceeds 5 ms'
        elif drift is None or abs(drift) > .002:
            reason = 'SDK global clock mapping has excessive drift'
        base.update(timing_accepted=reason is None,
                    timing_reason=reason or 'Exposure clock mapping passed the stationary host-time checks')
        return base


def frame_metadata(rs, frame):
    values = {}
    for name in ('frame_counter', 'frame_timestamp', 'sensor_timestamp',
                 'actual_exposure', 'gain_level', 'auto_exposure',
                 'time_of_arrival', 'backend_timestamp', 'actual_fps'):
        enum = getattr(rs.frame_metadata_value, name, None)
        if enum is not None and frame.supports_frame_metadata(enum):
            values[name] = int(frame.get_frame_metadata(enum))
    return values


class SDKCornerRectifier:
    """Convert native RGB pixels to pinhole pixels with the SDK's own model.

    This transforms corner rays only. Saved RGB images remain native, and raw
    corner coordinates are retained separately from the PnP coordinates.
    """
    def __init__(self, rs, intrinsics):
        self.rs, self.intrinsics = rs, intrinsics
        self.k = np.array([[intrinsics.fx, 0., intrinsics.ppx],
                           [0., intrinsics.fy, intrinsics.ppy], [0., 0., 1.]])
        if intrinsics.model not in (rs.distortion.none, rs.distortion.brown_conrady,
                                    rs.distortion.inverse_brown_conrady):
            raise ValueError(f'Unsupported RGB distortion model: {intrinsics.model}')

    def __call__(self, pixels):
        rays = np.asarray([self.rs.rs2_deproject_pixel_to_point(self.intrinsics,
                          list(map(float, pixel)), 1.) for pixel in np.asarray(pixels).reshape(-1, 2)])
        if not np.isfinite(rays).all() or np.any(rays[:, 2] <= 0):
            raise ValueError('Invalid SDK RGB corner rays')
        projected = rays @ self.k.T
        return projected[:, :2]/projected[:, 2, None]


class RealSenseRGB:
    def __init__(self, serial, width=WIDTH, height=HEIGHT, fps=FPS):
        import pyrealsense2 as rs
        self.rs = rs
        self.pipeline = rs.pipeline()
        self.started = False
        self.condition = threading.Condition()
        self.stop = threading.Event()
        self.latest = None
        self.error = None
        self.clock = ExposureClock()
        self.sequence = 0
        context = rs.context()
        device = next((d for d in context.query_devices()
                       if d.get_info(rs.camera_info.serial_number) == str(serial)), None)
        if device is None:
            raise RuntimeError(f'RealSense serial {serial} is not connected')
        sensor_options = []
        for sensor in device.query_sensors():
            if not any(profile.stream_type() == rs.stream.color for profile in sensor.get_stream_profiles()):
                continue
            options = dict(sensor=sensor.get_info(rs.camera_info.name))
            for name, value in [('global_time_enabled', 1.), ('frames_queue_size', 1.)]:
                option = getattr(rs.option, name)
                if sensor.supports(option):
                    try:
                        sensor.set_option(option, value)
                        options[name] = sensor.get_option(option)
                    except RuntimeError as error:
                        options[name + '_error'] = str(error)
            sensor_options.append(options)
        config = rs.config()
        config.enable_device(str(serial))
        config.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)
        try:
            self.profile = self.pipeline.start(config)
            self.started = True
            profile = self.profile.get_stream(rs.stream.color).as_video_stream_profile()
            intrinsics = profile.get_intrinsics()
            self.k = np.array([[intrinsics.fx, 0., intrinsics.ppx],
                               [0., intrinsics.fy, intrinsics.ppy], [0., 0., 1.]])
            native_distortion = np.asarray(intrinsics.coeffs, dtype=float)
            self.pixel_to_pinhole = (SDKCornerRectifier(rs, intrinsics)
                if np.any(native_distortion != 0.) else None)
            # PnP and its saved image_points always use zero-distortion pinhole
            # coordinates. Native coefficients and pixels remain explicit.
            self.distortion = np.zeros(5)
            device = self.profile.get_device()
            self.metadata = dict(backend='realsense', serial=device.get_info(rs.camera_info.serial_number),
                device=device.get_info(rs.camera_info.name), firmware=device.get_info(rs.camera_info.firmware_version),
                usb=device.get_info(rs.camera_info.usb_type_descriptor), width=intrinsics.width, height=intrinsics.height,
                fps=fps, K=self.k.tolist(), distortion=self.distortion.tolist(),
                distortion_model='none', native_distortion=native_distortion.tolist(),
                native_distortion_model=str(intrinsics.model),
                corner_processing='SDK-deprojected pinhole pixels' if self.pixel_to_pinhole else 'native zero-distortion pixels',
                frame=f'realsense_{serial}_color_optical',
                view='native RGB colour optical', pixel_format='BGR8', optical_axes='x right, y down, z forward',
                processing='native SDK RGB; no image smoothing, depth filters, or alignment',
                sensor_options=sensor_options, started_utc=datetime.now(timezone.utc).isoformat())
            self.thread = threading.Thread(target=self._capture, name='RealSense calibration RGB', daemon=True)
            self.thread.start()
        except Exception:
            self.close()
            raise

    def _capture(self):
        try:
            while not self.stop.is_set():
                frames = self.pipeline.wait_for_frames(3000)
                color = frames.get_color_frame()
                if not color:
                    continue
                received = time.monotonic()
                bridge = clock_bridge()
                data = np.asanyarray(color.get_data()).copy()
                raw_metadata = frame_metadata(self.rs, color)
                stamp = float(color.get_timestamp())
                number = int(color.get_frame_number())
                timing = self.clock.map(stamp, str(color.get_frame_timestamp_domain()),
                                        raw_metadata, received, bridge, number)
                info = dict(timing, rgb_timestamp_ms=stamp, rgb_frame_number=number,
                            rgb_timestamp_domain=str(color.get_frame_timestamp_domain()),
                            sensor_metadata=raw_metadata)
                with self.condition:
                    self.sequence += 1
                    # One latest frame, never an unbounded processing queue.
                    self.latest = (self.sequence, data, info)
                    self.condition.notify_all()
        except Exception as error:
            if not self.stop.is_set():
                self.error = f'{type(error).__name__}: {error}'
            self.stop.set()
            with self.condition:
                self.condition.notify_all()

    def next_frame(self, previous):
        with self.condition:
            self.condition.wait_for(lambda: self.error or self.stop.is_set()
                                    or (self.latest is not None and self.latest[0] != previous), timeout=.25)
            if self.error:
                raise RuntimeError(self.error)
            if self.latest is None or self.latest[0] == previous:
                return None
            sequence, image, metadata = self.latest
            return sequence, image, dict(metadata)

    def close(self):
        self.stop.set()
        # Stop may raise after physical disconnect; preserve the original fault.
        if self.started:
            try:
                self.pipeline.stop()
            except RuntimeError:
                pass
            self.started = False
        if hasattr(self, 'thread'):
            self.thread.join(timeout=4.)


class SimulationRGB:
    def __init__(self, scene, serial, k=DEFAULT_K):
        import mujoco
        from policy_sim.control.robot import CONTROLLED_JOINTS
        self.mujoco = mujoco
        self.model = mujoco.MjModel.from_xml_path(str(scene))
        self.data = mujoco.MjData(self.model)
        self.gripper_positions={}
        for side in ('positive','negative'):
            name=f'calibration_left_jaw_{side}'
            jid=mujoco.mj_name2id(self.model,mujoco.mjtObj.mjOBJ_JOINT,name)
            if jid>=0:
                self.data.qpos[self.model.jnt_qposadr[jid]]=.04
                self.gripper_positions[name]=.04
        self.indices = [self.model.joint(name).qposadr[0] for name in CONTROLLED_JOINTS]
        self.k = np.asarray(k, dtype=float)
        self.distortion = np.zeros(5)
        self.camera_name = 'wrist_calibration_view'
        camera = self.model.camera(self.camera_name)
        # OpenGL cameras have centred principal points and square pixels. Render
        # a larger image, then map exact native fx/fy/cx/cy to the render. The
        # pixel centre is (width-1)/2, not width/2.
        offwidth, offheight = 1400, 800
        self.model.vis.global_.offwidth = max(self.model.vis.global_.offwidth, offwidth)
        self.model.vis.global_.offheight = max(self.model.vis.global_.offheight, offheight)
        camera.fovy[0] = np.rad2deg(2*np.arctan(offheight/(2*self.k[1, 1])))
        if hasattr(self.model, 'cam_sensorsize'):
            self.model.cam_sensorsize[camera.id] = 0.
        self.renderer = mujoco.Renderer(self.model, height=offheight, width=offwidth)
        self.option = mujoco.MjvOption()
        self.option.geomgroup[3:5] = 1
        # Group 5 is the enclosing camera housing: keep collision geometry in
        # the model, but the camera must see through its own lens cover.
        self.option.geomgroup[5] = 0
        u, v = np.meshgrid(np.arange(WIDTH, dtype=np.float32), np.arange(HEIGHT, dtype=np.float32))
        self.mapx = ((offwidth-1)/2 + self.k[1, 1]/self.k[0, 0]*(u-self.k[0, 2])).astype(np.float32)
        self.mapy = ((offheight-1)/2 + v-self.k[1, 2]).astype(np.float32)
        if (self.mapx.min() < 0 or self.mapx.max() >= offwidth
                or self.mapy.min() < 0 or self.mapy.max() >= offheight):
            raise ValueError('Expanded simulation render does not cover the specified intrinsics')
        self.metadata = dict(backend='mujoco', serial=str(serial), width=WIDTH, height=HEIGHT,
                             fps=FPS, simulated_gripper_positions_m=self.gripper_positions, K=self.k.tolist(), distortion=[0.]*5,
                             distortion_model='none', frame=f'realsense_{serial}_color_optical',
                             view='simulated RGB colour optical', optical_axes='x right, y down, z forward',
                             simulation_only=True, camera_name=self.camera_name,
                             expanded_render_size=[offwidth, offheight], principal_point_mapping='pixel-centre remap')

    def render(self, q):
        q = np.asarray(q, dtype=float)
        if q.shape != (18,) or not np.all(np.isfinite(q)):
            raise ValueError('Simulation camera request needs finite q[18]')
        self.data.qpos[self.indices] = q[3:]
        self.mujoco.mj_forward(self.model, self.data)
        self.renderer.update_scene(self.data, camera=self.camera_name, scene_option=self.option)
        self.renderer.scene.flags[self.mujoco.mjtRndFlag.mjRND_SHADOW] = False
        rgb = self.renderer.render()
        bgr = cv2.cvtColor(cv2.remap(rgb, self.mapx, self.mapy, cv2.INTER_LINEAR), cv2.COLOR_RGB2BGR)
        at = time.monotonic()
        return bgr, dict(image_time=at, image_timestamp_ns=None, timing_accepted=True,
                         timing_reason='Simulation frame with explicitly supplied q',
                         clock_mapping_uncertainty_s=0., simulation_only=True,
                         host_receipt_monotonic_s=at, timestamp_semantics='simulation render completion monotonic time',
                         synchronization='simulation state supplied by request; not real hardware synchronization')

    def close(self):
        self.renderer.close()


def read_request(directory):
    path = directory / 'request.json'
    if not path.exists():
        return None
    request = json.loads(path.read_text())
    if (not isinstance(request.get('id'), int) or request['id'] < 0
            or not np.isfinite(request.get('after', float('nan')))):
        raise ValueError('Camera request requires a nonnegative integer id and finite monotonic after')
    return request


def simulation_intrinsics(path, serial):
    """Use this camera's recorded SDK pinhole profile, never another serial's K."""
    if path is None:
        if str(serial) != DEFAULT_SERIAL:
            raise ValueError('Non-default simulated camera requires --simulation-intrinsics')
        return DEFAULT_K.copy()
    profile=json.loads(Path(path).read_text())
    if str(profile['serial']) != str(serial):
        raise ValueError('Simulation intrinsics serial differs from requested camera')
    if (profile['width'],profile['height'],profile.get('fps',FPS)) != (WIDTH,HEIGHT,FPS):
        raise ValueError('Simulation profile must be 1280x720 at 30 fps')
    k=np.asarray(profile['K'],dtype=float)
    if (k.shape != (3,3) or not np.isfinite(k).all() or k[0,0]<=0 or k[1,1]<=0
            or not np.allclose(k[2],[0,0,1])):
        raise ValueError('Invalid simulation pinhole intrinsics')
    return k


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=['live', 'sim'], required=True)
    parser.add_argument('--directory', type=Path, required=True)
    parser.add_argument('--scene', type=Path)
    parser.add_argument('--serial', default=DEFAULT_SERIAL)
    parser.add_argument('--simulation-intrinsics', type=Path, help='Recorded SDK profile or validated pose plan; simulation only')
    parser.add_argument('--width', type=int, default=WIDTH)
    parser.add_argument('--height', type=int, default=HEIGHT)
    parser.add_argument('--fps', type=int, default=FPS)
    parser.add_argument('--board-config', type=Path, default=ROOT/'config/camera/charuco_87mm.json')
    args = parser.parse_args()
    if args.mode=='live' and args.simulation_intrinsics:
        parser.error('--simulation-intrinsics is simulation only; live intrinsics come from the SDK')
    args.directory.mkdir(parents=True, exist_ok=True)
    cfg = json.loads(args.board_config.read_text())
    stopping = threading.Event()
    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, lambda *_: stopping.set())
    camera = None
    try:
        if args.mode == 'live':
            camera = RealSenseRGB(args.serial, args.width, args.height, args.fps)
        else:
            if (args.width, args.height, args.fps) != (WIDTH, HEIGHT, FPS):
                parser.error('Simulated wrist plan uses 1280x720 at 30 fps')
            if args.scene is None:
                raise ValueError('--scene is required for simulation')
            camera = SimulationRGB(args.scene, args.serial, simulation_intrinsics(args.simulation_intrinsics,args.serial))
        atomic(args.directory/'ready.json', dict(camera.metadata, mode=args.mode, board=cfg,
             board_config=str(args.board_config), ipc_image_name='left.png (native colour, historical IPC name)',
             timing_policy=dict(max_uncertainty_s=MAX_UNCERTAINTY_S, max_frame_age_s=MAX_FRAME_AGE_S,
                                sdk_clock_mapping_allowance_s=TIMING_ALLOWANCE_S,
                                hardware_trigger_synchronized=False)))
        last_preview, sequence, done = 0., 0, None
        while not stopping.is_set() and not (args.directory/'stop').exists():
            request = read_request(args.directory)
            if request and (args.directory/str(request['id'])/'result.json').exists():
                done = request['id']
            pending = request and request['id'] != done
            if args.mode == 'live':
                frame = camera.next_frame(sequence)
                if frame is None:
                    continue
                sequence, bgr, timing = frame
            else:
                if not pending or time.monotonic() <= request['after']+.2:
                    stopping.wait(.02)
                    continue
                bgr, timing = camera.render(request['q'])
            new_capture = (pending and timing['host_receipt_monotonic_s'] > request['after']+.2
                           and (not timing['timing_accepted'] or timing['image_time'] > request['after']+.2))
            if time.monotonic()-last_preview < 1/6 and not new_capture:
                continue
            detection, annotated = detect(bgr, cfg, camera.k, camera.distortion,
                pixel_to_pinhole=getattr(camera, 'pixel_to_pinhole', None))
            result = dict(detection, **timing)
            result.update(simulation_only=args.mode == 'sim',
                          camera_serial=str(args.serial), camera_frame=camera.metadata['frame'],
                          image_size=[camera.metadata['width'], camera.metadata['height']])
            if not timing['timing_accepted']:
                result['visual_accepted'] = bool(detection['accepted'])
                result['accepted'] = False
                result['reason'] = timing['timing_reason']
            save_image(args.directory/'preview.jpg', cv2.resize(annotated, (960, 540)))
            atomic(args.directory/'status.json', dict(result, at=time.monotonic(), sequence=sequence))
            last_preview = time.monotonic()
            if new_capture:
                folder = args.directory/str(request['id'])
                folder.mkdir(exist_ok=True)
                save_image(folder/'left.png', bgr)
                atomic(folder/'metadata.json', dict(camera.metadata, **timing,
                       request_id=request['id'], requested_after_monotonic_s=request['after'],
                       capture_processing_finished_monotonic_s=time.monotonic()))
                atomic(folder/'result.json', result)
                done = request['id']
    except Exception as error:
        atomic(args.directory/'error.json', dict(error=f'{type(error).__name__}: {error}',
                                                at=time.monotonic()))
        raise
    finally:
        if camera is not None:
            camera.close()


if __name__ == '__main__':
    main()
