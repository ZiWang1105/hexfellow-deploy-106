"""Optional bounded, instance-local telemetry instrumentation; no control writes."""

import hashlib
import json
import os
import queue
import threading
import time
from collections import deque
from pathlib import Path


class FeedbackGapState:
    """Constant-size descriptive classifier; never changes feedback safety gates."""
    def __init__(self):
        self.previous = self.first = self.onset = None
        self.count = 0
        self.over = {10: 0, 20: 0, 40: 0}
        self.recent = deque(maxlen=3)
        self.last_abnormal = None

    def observe(self, received):
        self.count += 1
        if self.first is None:
            self.first = received
        if self.previous is not None:
            gap = (received - self.previous) * 1000
            for threshold in self.over:
                self.over[threshold] += int(gap > threshold)
            if gap > 20:
                self.last_abnormal = received
                self.recent.append(received)
            if self.onset is None and (gap > 40 or (
                    len(self.recent) == 3 and self.recent[-1] - self.recent[0] <= 10)):
                self.onset = received
        self.previous = received

    def report(self):
        return {'frames': self.count, 'gaps_over_ms': self.over,
                'classification': 'Degraded' if self.onset is not None else (
                    'Isolated spikes' if self.over[20] else 'Healthy'),
                'first_received': self.first, 'last_received': self.previous,
                'degradation_detected_at': self.onset,
                'last_abnormal_at': self.last_abnormal,
                'scope': 'all accepted callback frames including startup; descriptive only'}


def lifecycle_api_type(base_type, recorder):
    """Observe SDK lifecycle before its background thread starts; no SDK edits."""
    class LifecycleApi(base_type):
        def __init__(self, *args, **kwargs):
            self.lifecycle_reconnects = 0
            self.lifecycle_id = f'{os.getpid()}:{id(self)}'
            self.lifecycle_recorder = recorder
            recorder.record('sdk_create', pid=os.getpid(), instance=self.lifecycle_id,
                            url=kwargs.get('ws_url'), realtime_ns=time.time_ns())
            super().__init__(*args, **kwargs)
            recorder.record('sdk_constructed', instance=self.lifecycle_id)

        def close(self):
            recorder.record('sdk_close_start', instance=self.lifecycle_id)
            try:
                result = super().close()
                thread = getattr(self, '_HexDeviceApi__loop_thread', None)
                if thread is not None and thread is not threading.current_thread():
                    thread.join(timeout=3)
                closed = thread is None or not thread.is_alive()
                recorder.record('sdk_close_end', instance=self.lifecycle_id, loop_closed=closed)
                if not closed:
                    raise RuntimeError('SDK background loop did not close')
                return result
            except BaseException as error:
                recorder.record('sdk_close_error', instance=self.lifecycle_id, error=str(error))
                raise

    async def reconnect(self):
        self.lifecycle_reconnects += 1
        recorder.record('sdk_reconnect', instance=self.lifecycle_id)
        return await super(LifecycleApi, self)._HexDeviceApi__reconnect_ws()

    async def capture(self, *args, **kwargs):
        message = await super(LifecycleApi, self)._HexDeviceApi__capture_data_frame_from_websocket(*args, **kwargs)
        if message is not None and message.HasField('kcp_server_status'):
            recorder.record('kcp_handshake', instance=self.lifecycle_id,
                            session_id=int(message.session_id),
                            server_port=int(message.kcp_server_status.server_port))
        return message

    # Explicit names are necessary: Python's class-name mangling would miss SDK hooks.
    LifecycleApi._HexDeviceApi__reconnect_ws = reconnect
    LifecycleApi._HexDeviceApi__capture_data_frame_from_websocket = capture
    return LifecycleApi


class TransportRecorder:
    def __init__(self, path, capacity=8192):
        self.stream = path.open("x")
        self.queue = queue.Queue(maxsize=capacity)
        self.dropped = 0
        self.written = 0
        self.error = None
        self.thread = threading.Thread(target=self._write, name="telemetry-diagnostics", daemon=True)
        self.thread.start()
        self.record("clock_anchor", realtime_ns=time.time_ns())

    def record(self, kind, **fields):
        try:
            self.queue.put_nowait({"kind": kind, "at": time.monotonic(), **fields})
        except queue.Full:
            self.dropped += 1

    def _write(self):
        try:
            with self.stream:
                while True:
                    row = self.queue.get()
                    if row is None:
                        return
                    self.stream.write(json.dumps(row, allow_nan=False) + "\n")
                    self.written += 1
        except Exception as error:
            self.error = str(error)

    def close(self):
        try:
            self.queue.put(None, timeout=2)
        except queue.Full:
            self.error = self.error or "diagnostic queue did not drain"
        self.thread.join(timeout=3)
        return {"written": self.written, "dropped": self.dropped,
                "error": self.error, "complete": not self.thread.is_alive()
                and self.error is None and self.dropped == 0}


class DeviceRecorder:
    """Attach endpoint identity without changing legacy event names."""
    def __init__(self, recorder, device):
        self.recorder, self.device = recorder, device

    def record(self, kind, **fields):
        self.recorder.record(kind, device=self.device, **fields)


class TimedLock:
    """Preserve the SDK RLock, including reentrancy, while timing each context."""
    def __init__(self, lock, recorder):
        self.lock, self.recorder = lock, recorder
        self.local = threading.local()

    def __enter__(self):
        start = time.monotonic()
        self.lock.acquire()
        acquired = time.monotonic()
        stack = getattr(self.local, "stack", None)
        if stack is None:
            stack = self.local.stack = []
        stack.append((start, acquired))
        return self

    def __exit__(self, *args):
        start, acquired = self.local.stack.pop()
        ended = time.monotonic()
        self.lock.release()
        self.recorder.record("kcp_lock", started=start, acquired=acquired, ended=ended,
                             thread=threading.current_thread().name)


def instrument_kcp(api, recorder):
    client = getattr(api, "_HexDeviceApi__kcp_client", None)
    if client is None:
        raise RuntimeError("Theta KCP instrumentation unavailable for this SDK instance")
    client._kcp_lock = TimedLock(client._kcp_lock, recorder)
    receive = client._receive_from_socket
    process = client._process_received_data

    def recv():
        data = receive()
        if data:
            recorder.record("udp_userspace_receive", bytes=len(data))
        return data

    def dispatch(data):
        started = time.monotonic()
        try:
            return process(data)
        finally:
            recorder.record("kcp_dispatch", started=started, bytes=len(data))

    client._receive_from_socket = recv
    client._process_received_data = dispatch


def record_sdk_identity(recorder):
    import hex_device
    import hex_device.hex_device_api as api_module
    import hex_device.kcp_client_core as kcp_module

    recorder.record("sdk_identity", version=hex_device.__version__, files={
        str(module.__file__): hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest()
        for module in (api_module, kcp_module)
    })


def raw_timestamp(message):
    if (not hasattr(message, "HasField") or not message.HasField("time_stamp")
            or not message.time_stamp.HasField("monotonic_time_stamp")):
        return None
    stamp = message.time_stamp.monotonic_time_stamp
    return {"seconds": int(stamp.seconds), "nanoseconds": int(stamp.nanoseconds)}
