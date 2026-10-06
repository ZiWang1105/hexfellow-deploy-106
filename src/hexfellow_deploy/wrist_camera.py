"""Independent wrist RGB capture with optional time-matched robot calibration."""
import json
from pathlib import Path
import time

from deploy_anything.components.interface.base_interface import DABaseInterface


class NativeWristCamera(DABaseInterface):
    def __init__(self, name, images, output, serial=None, fps=30, time_budget=.04,
                 feedback=None, workspace=None, deployment_config=None, calibration=None):
        super().__init__(name, [images]+([] if feedback is None else [feedback]), time_budget)
        self.images, self.output, self.serial, self.fps = images, Path(output), serial, fps
        self.feedback, self.workspace = feedback, workspace
        self.deployment_config, self.calibration = deployment_config, calibration

    def initialize(self):
        if type(self.fps) is not int or not 0 < self.fps <= 60:
            raise ValueError('Wrist camera FPS must be an integer in (0,60]')
        self._initialized.value = True
        return True

    def clean_up(self):
        self._cleaned_up.value = True
        return True

    def run(self):
        import numpy as np
        import pyrealsense2 as rs
        pipeline = rs.pipeline()
        started = False
        sampler = geometry = None
        health = dict(modality='rgb_only', extrinsics_available=False, frames=0,
                      publications=0, publication_misses=0, error=None)
        def report():
            path = self.output/'wrist_camera_health.json'
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp = path.with_suffix('.tmp')
            tmp.write_text(json.dumps(dict(health, observed_s=time.monotonic()), indent=2)+'\n')
            tmp.replace(path)
        try:
            serial = self.serial
            if self.calibration:
                if self.feedback is None or not serial:
                    raise ValueError('Calibrated wrist requires an explicit serial and encoder feedback')
                from .workspace import activate
                from .camera_sync import EncoderHistory, FeedbackSampler
                root = activate(self.workspace)
                from mobile_manipulator.hardware.config import load_deployment
                from mobile_manipulator.hardware.wrist_observation import WristGeometry, color_intrinsics
                cfg = load_deployment(root/self.deployment_config, execute=True, mock=False, research=True)
                geometry = WristGeometry(root/self.calibration, serial, cfg['robot']['urdf'],
                                         cfg['robot']['assembly'], cfg['robot']['camera_encoder_max_skew_s'])
                self.output.mkdir(parents=True, exist_ok=True)
                (self.output/'wrist_calibration.json').write_bytes(geometry.calibration_bytes)
                history = EncoderHistory(cfg['robot']['camera_max_age_s']+geometry.max_skew_s)
                sampler = FeedbackSampler(self.feedback, history).start()
                health.update(extrinsics_available=True, calibration=geometry.metadata,
                              unmatched_frames=0)
            if not serial:
                devices = list(rs.context().query_devices())
                if len(devices) != 1:
                    raise RuntimeError('Select wrist_camera.serial when zero or multiple RealSense devices are connected')
                serial = devices[0].get_info(rs.camera_info.serial_number)
            config = rs.config()
            config.enable_device(str(serial))
            config.enable_stream(rs.stream.color, self.images.width, self.images.height, rs.format.rgb8, self.fps)
            profile = pipeline.start(config)
            started = True
            if profile.get_device().get_info(rs.camera_info.serial_number) != str(serial):
                raise ValueError('Active wrist device does not match requested serial')
            if geometry:
                intrinsics = color_intrinsics(profile.get_stream(rs.stream.color))
                health['fixed_camera_intrinsics'] = intrinsics
            # Keep the latest frame instead of accumulating a capture backlog.
            for sensor in profile.get_device().query_sensors():
                if sensor.supports(rs.option.frames_queue_size):
                    sensor.set_option(rs.option.frames_queue_size, 1.)
            health.update(serial=str(serial), device=profile.get_device().get_info(rs.camera_info.name),
                          width=self.images.width, height=self.images.height, fps=self.fps)
            next_report = 0.
            while not self.stopping:
                frames = pipeline.wait_for_frames(timeout_ms=5000 if not health['frames'] else 1000)
                color = frames.get_color_frame()
                if not color:
                    continue
                received = time.monotonic()
                value = dict(rgb=np.asanyarray(color.get_data()).copy(), received_s=received)
                health.update(frames=health['frames']+1, received_s=received)
                if geometry:
                    if sampler.error:
                        raise RuntimeError(sampler.error)
                    row = history.closest(received, time.monotonic(), geometry.max_skew_s)
                    if row is None:
                        health['unmatched_frames'] += 1
                        if received >= next_report:
                            report()
                            next_report = received+1.
                        continue
                    value = geometry.observation(value['rgb'], received, row['measured'], intrinsics)
                if self.images.put(value):
                    health['publications'] += 1
                    health['published_s'] = time.monotonic()
                    self._ready.value = True
                else:
                    health['publication_misses'] += 1
                if received >= next_report:
                    report()
                    next_report = received+1.
        except Exception as error:
            health['error'] = f'{type(error).__name__}: {error}'
            raise
        finally:
            try:
                if started:
                    pipeline.stop()
            finally:
                try:
                    if sampler is not None:
                        sampler.close()
                finally:
                    health['closed'] = True
                    report()
