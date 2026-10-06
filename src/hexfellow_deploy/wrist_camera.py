"""Independent RGB-only RealSense context; no robot or calibration access."""
import json
from pathlib import Path
import time

from deploy_anything.components.interface.base_interface import DABaseInterface


class NativeWristCamera(DABaseInterface):
    def __init__(self, name, images, output, serial=None, fps=30, time_budget=.04):
        super().__init__(name, [images], time_budget)
        self.images, self.output, self.serial, self.fps = images, Path(output), serial, fps

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
                health['closed'] = True
                report()
