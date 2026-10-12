"""Bounded trusted-process mailboxes; locks protect copies, never consumer work."""
import pickle
import queue
import numpy as np


class LatestMessage:
    """Metadata only. Pixels and plant state use fixed-format ArrayMailbox."""
    def __init__(self, context, capacity=131072):
        self.buffer = context.RawArray('B', capacity)
        self.length = context.RawValue('i', 0)
        self.sequence = context.RawValue('q', 0)
        self.lock = context.Lock()

    def put(self, message):
        payload = pickle.dumps(message, protocol=5)
        if len(payload) > len(self.buffer):
            raise ValueError('Metadata exceeds mailbox capacity')
        if not self.lock.acquire(timeout=.0005):
            return False
        try:
            np.frombuffer(self.buffer, dtype=np.uint8)[:len(payload)] = np.frombuffer(payload, dtype=np.uint8)
            self.length.value = len(payload)
            self.sequence.value += 1
        finally:
            self.lock.release()
        return True

    def get(self):
        if not self.lock.acquire(timeout=.0005):
            raise queue.Empty
        try:
            if not self.length.value:
                raise queue.Empty
            payload = bytes(memoryview(self.buffer).cast('B')[:self.length.value])
        finally:
            self.lock.release()
        return pickle.loads(payload)


class ArrayMailbox:
    """Single-writer, latest-only numeric snapshots. Never for MIT references."""
    def __init__(self, context, size):
        self.buffer = context.RawArray('d', size)
        self.sequence = context.RawValue('q', 0)
        self.lock = context.Lock()

    def put(self, values):
        values = np.asarray(values, dtype=np.float64)
        if values.shape != (len(self.buffer),):
            raise ValueError('Snapshot shape mismatch')
        if not self.lock.acquire(timeout=.0005):
            return False
        try:
            np.frombuffer(self.buffer)[:] = values
            self.sequence.value += 1
        finally:
            self.lock.release()
        return True

    def get(self):
        if not self.lock.acquire(timeout=.0005):
            raise queue.Empty
        try:
            if not self.sequence.value:
                raise queue.Empty
            return self.sequence.value, np.frombuffer(self.buffer).copy()
        finally:
            self.lock.release()


class RGBDMailbox:
    """Registered images and their encoder packet committed as one capture."""
    def __init__(self, context, width, height):
        self.width, self.height = width, height
        self.pixels = context.RawArray('B', width * height * 7)
        self.metadata = context.RawArray('d', 34)
        self.sequence = context.RawValue('q', 0)
        self.lock = context.Lock()

    def put(self, packet, captured_wall_s):
        from ..robot.joints import JOINTS, SIDES
        from .contract import validate_sensor_packet
        validate_sensor_packet(packet, image=True)
        if packet['rgb'].shape != (self.height, self.width, 3):
            raise ValueError('RGB-D mailbox resolution mismatch')
        if not self.lock.acquire(timeout=.001):
            return False
        try:
            size = self.width * self.height
            raw = np.frombuffer(self.pixels, dtype=np.uint8)
            raw[:3*size] = packet['rgb'].reshape(-1)
            raw[3*size:].view(np.float32)[:] = packet['depth_m'].reshape(-1)
            np.frombuffer(self.metadata)[:] = [packet['timestamp_s'], captured_wall_s,
                *[packet['joint_positions'][n] for n in JOINTS],
                *[packet['joint_velocities'][n] for n in JOINTS],
                *[['open','closed','between'].index(packet['gripper_state'][n]) for n in SIDES]]
            self.sequence.value += 1
        finally:
            self.lock.release()
        return True

    def get(self):
        from ..robot.joints import JOINTS, SIDES
        if not self.lock.acquire(timeout=.01):
            raise queue.Empty
        try:
            if not self.sequence.value:
                raise queue.Empty
            sequence = self.sequence.value
            raw = np.frombuffer(self.pixels, dtype=np.uint8).copy()
            meta = np.frombuffer(self.metadata).copy()
        finally:
            self.lock.release()
        size = self.width*self.height
        return sequence, float(meta[1]), dict(timestamp_s=float(meta[0]),
            joint_positions=dict(zip(JOINTS, map(float, meta[2:17]))),
            joint_velocities=dict(zip(JOINTS, map(float, meta[17:32]))),
            gripper_state={s: ['open','closed','between'][int(meta[32+i])] for i,s in enumerate(SIDES)},
            rgb=raw[:3*size].reshape(self.height,self.width,3),
            depth_m=raw[3*size:].view(np.float32).reshape(self.height,self.width))
