"""Bounded JSON on DA numeric shared memory; no custom mailbox or pickle."""
import json
import time
import numpy as np
from deploy_anything.components.comms.wrappers import ArrayCommWrapper


class JsonTopic(ArrayCommWrapper):
    def __init__(self, name, capacity=65536, prefix="", *, create=True):
        if capacity < 256:
            raise ValueError("JSON topic is too small")
        super().__init__(prefix + name, shape=(capacity,), dtype="uint8", create=create)

    def put(self, value, *, timeout=0.):
        encoded = json.dumps(value, allow_nan=False, separators=(",", ":")).encode()
        if len(encoded) >= self.shape[0]:
            raise ValueError("JSON message exceeds topic capacity")
        payload = np.zeros(self.shape, dtype=np.uint8)
        payload[:len(encoded)] = np.frombuffer(encoded, dtype=np.uint8)
        return self.send(self.ArrayData(time.monotonic(), payload), timeout=timeout)

    def get(self, *, timeout=0.):
        row = self.receive(timeout=timeout)
        if row.timestamp <= 0:
            return None
        return json.loads(row.array.tobytes().split(b"\0", 1)[0])


class RGBTopic(ArrayCommWrapper):
    """Atomic RGB pixels, capture time and optional matched robot geometry."""
    def __init__(self, name, width=640, height=480, prefix='', *, create=True):
        self.width, self.height = width, height
        super().__init__(prefix+name, shape=(8192+height*width*3,), dtype='uint8', create=create)

    def put(self, value, *, timeout=.001):
        from hexfellow_deploy.interfaces.contract import validate_rgb_camera
        validate_rgb_camera(value)
        if value['rgb'].shape != (self.height, self.width, 3):
            raise ValueError('RGB camera resolution differs from shared-memory schema')
        meta = json.dumps({k:v for k,v in value.items() if k not in ('rgb', 'received_s')},
                          allow_nan=False).encode()
        if len(meta) >= 8192:
            raise ValueError('Wrist metadata exceeds shared-memory capacity')
        buf = np.zeros(self.shape, dtype=np.uint8)
        buf[:len(meta)] = np.frombuffer(meta, dtype=np.uint8)
        buf[8192:] = value['rgb'].reshape(-1)
        return self.send(self.ArrayData(value['received_s'], buf), timeout=timeout)

    def get(self, *, timeout=.001):
        row = self.receive(timeout=timeout)
        if row.timestamp <= 0:
            return None
        meta = json.loads(row.array[:8192].tobytes().split(b'\0', 1)[0])
        return dict(meta, rgb=row.array[8192:].reshape(self.height, self.width, 3), received_s=row.timestamp)


class RGBDTopic(ArrayCommWrapper):
    """One atomic DA shared-memory publication of RGB, depth and encoder metadata."""
    def __init__(self, name, width=640, height=360, prefix="", *, create=True):
        self.width, self.height = width, height
        super().__init__(prefix+name, shape=(8192+width*height*7,), dtype="uint8", create=create)

    def put(self, measured, rgb, depth_m, intrinsics, robot_fk=None):
        from hexfellow_deploy.interfaces.contract import validate_sensor_packet
        validate_sensor_packet(dict(measured, rgb=rgb, depth_m=depth_m), image=True)
        if rgb.shape != (self.height, self.width, 3):
            raise ValueError("Camera resolution differs from shared-memory schema")
        meta = json.dumps(dict(measured=measured, intrinsics=intrinsics, robot_fk=robot_fk), allow_nan=False).encode()
        if len(meta) >= 8192:
            raise ValueError("Camera metadata exceeds capacity")
        buf = np.zeros(self.shape, dtype=np.uint8)
        buf[:len(meta)] = np.frombuffer(meta, dtype=np.uint8)
        size = self.width*self.height
        buf[8192:8192+size*3] = rgb.reshape(-1)
        buf[8192+size*3:] = np.asarray(depth_m, dtype=np.float32).view(np.uint8).reshape(-1)
        return self.send(self.ArrayData(time.monotonic(), buf), timeout=.001)

    def get(self):
        row = self.receive(timeout=.01)
        if row.timestamp <= 0:
            return None
        value = json.loads(row.array[:8192].tobytes().split(b"\0", 1)[0])
        size = self.width*self.height
        value.update(published_s=row.timestamp,
            rgb=row.array[8192:8192+size*3].reshape(self.height, self.width, 3),
            depth_m=row.array[8192+size*3:].view(np.float32).reshape(self.height, self.width))
        return value
