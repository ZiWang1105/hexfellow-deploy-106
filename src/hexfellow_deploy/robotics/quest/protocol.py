"""Versioned binary protocol used by the minimal Quest WebXR page."""

from __future__ import annotations

import dataclasses
import struct
import threading
import time
from collections import deque
from collections.abc import Mapping

import numpy as np

PROTOCOL_MAGIC = b"QXR1"
PROTOCOL_VERSION = 1
PROTOCOL_MAGIC_V2 = b"QXR2"
PROTOCOL_VERSION_V2 = 2
POSE_FLAG_PRESENT = 1 << 0
POSE_FLAG_EMULATED_POSITION = 1 << 1
SESSION_FLAG_VISIBLE = 1 << 0
SESSION_FLAG_LOCAL_FLOOR = 1 << 1

_HEADER = struct.Struct("<4sHHIIddf")
_HEADER_V2 = struct.Struct("<4sHHIIdddf")
_HAND = struct.Struct("<B3x3d4dff")
POSE_PACKET_SIZE = _HEADER.size + 2 * _HAND.size
POSE_PACKET_SIZE_V2 = _HEADER_V2.size + 2 * _HAND.size


@dataclasses.dataclass(frozen=True)
class ControllerPose:
    """One controller grip pose in the current WebXR reference space."""

    present: bool
    emulated_position: bool
    position_m: np.ndarray
    orientation_xyzw: np.ndarray
    trigger: float
    squeeze: float

    @classmethod
    def missing(cls) -> ControllerPose:
        return cls(
            present=False,
            emulated_position=False,
            position_m=_readonly(np.zeros(3, dtype=np.float64)),
            orientation_xyzw=_readonly(np.array((0.0, 0.0, 0.0, 1.0), dtype=np.float64)),
            trigger=0.0,
            squeeze=0.0,
        )


@dataclasses.dataclass(frozen=True)
class QuestPoseSample:
    """One WebXR-frame sample plus host-side timing metadata."""

    sequence: int
    reference_epoch: int
    session_flags: int
    xr_time_ms: float
    sent_time_ms: float
    frame_rate_hz: float
    controllers: Mapping[str, ControllerPose]
    host_received_ns: int
    clock_offset_ms: float | None = None
    clock_uncertainty_ms: float | None = None
    sampled_time_ms: float | None = None
    transport: str = "websocket"

    @property
    def callback_to_host_ms(self) -> float | None:
        if self.clock_offset_ms is None:
            return None
        # The XR callback timestamp may be a predicted display time. The
        # packet's performance.now() send timestamp shares the clock-sync time
        # origin and therefore measures transport latency without prediction
        # bias.
        return self.host_received_ns / 1e6 - (self.sent_time_ms + self.clock_offset_ms)

    def latency_breakdown(self, now_ns: int) -> dict:
        """Do not label callback/queue delay as isolated network latency."""
        synced = (self.clock_offset_ms is not None and self.clock_uncertainty_ms is not None
                  and np.isfinite(self.clock_offset_ms) and np.isfinite(self.clock_uncertainty_ms))
        sampled = self.sampled_time_ms
        return {
            "clock_quality": "synchronized" if synced else "unsynchronized",
            "clock_uncertainty_ms": self.clock_uncertainty_ms,
            "sample_to_send_ms": None if sampled is None else self.sent_time_ms - sampled,
            "send_to_host_callback_ms": self.callback_to_host_ms if synced else None,
            "host_callback_to_consumer_ms": (int(now_ns) - self.host_received_ns) / 1e6,
            "sample_to_consumer_ms": (int(now_ns) / 1e6 - sampled - self.clock_offset_ms)
            if synced and sampled is not None else None,
        }

    def age_ms(self, now_ns: int | None = None) -> float:
        now = time.monotonic_ns() if now_ns is None else int(now_ns)
        if (
            self.transport == "webrtc"
            and self.sampled_time_ms is not None
            and self.clock_offset_ms is not None
        ):
            return now / 1e6 - (self.sampled_time_ms + self.clock_offset_ms)
        return (now - self.host_received_ns) / 1e6


def _readonly(value: np.ndarray) -> np.ndarray:
    result = np.asarray(value, dtype=np.float64).copy()
    result.setflags(write=False)
    return result


def _validate_hand(side: str, pose: ControllerPose) -> ControllerPose:
    position = np.asarray(pose.position_m, dtype=np.float64)
    orientation = np.asarray(pose.orientation_xyzw, dtype=np.float64)
    values = np.concatenate((position.reshape(-1), orientation.reshape(-1)))
    if position.shape != (3,) or orientation.shape != (4,) or not np.all(np.isfinite(values)):
        raise ValueError(f"{side} controller pose must contain finite position[3] and quaternion[4]")
    norm = float(np.linalg.norm(orientation))
    if pose.present and not 0.8 <= norm <= 1.2:
        raise ValueError(f"{side} controller quaternion norm is invalid: {norm:.6f}")
    if not np.isfinite(pose.trigger) or not np.isfinite(pose.squeeze):
        raise ValueError(f"{side} controller buttons must be finite")
    if pose.present:
        orientation = orientation / norm
    else:
        orientation = np.array((0.0, 0.0, 0.0, 1.0), dtype=np.float64)
    return ControllerPose(
        present=bool(pose.present),
        emulated_position=bool(pose.emulated_position),
        position_m=_readonly(position),
        orientation_xyzw=_readonly(orientation),
        trigger=float(np.clip(pose.trigger, 0.0, 1.0)),
        squeeze=float(np.clip(pose.squeeze, 0.0, 1.0)),
    )


def decode_pose_packet(payload: bytes, *, host_received_ns: int | None = None) -> QuestPoseSample:
    """Decode and validate one fixed-size little-endian WebXR packet."""

    sampled_ms: float | None = None
    if len(payload) == POSE_PACKET_SIZE_V2:
        magic, version, session_flags, sequence, epoch, xr_ms, sampled_ms, sent_ms, frame_rate = (
            _HEADER_V2.unpack_from(payload)
        )
        header_size = _HEADER_V2.size
    elif len(payload) == POSE_PACKET_SIZE:
        magic, version, session_flags, sequence, epoch, xr_ms, sent_ms, frame_rate = (
            _HEADER.unpack_from(payload)
        )
        header_size = _HEADER.size
    else:
        raise ValueError(
            f"Quest pose packet must be {POSE_PACKET_SIZE} bytes (QXR1) or "
            f"{POSE_PACKET_SIZE_V2} bytes (QXR2), got {len(payload)}"
        )
    if (magic, version) not in {
        (PROTOCOL_MAGIC, PROTOCOL_VERSION),
        (PROTOCOL_MAGIC_V2, PROTOCOL_VERSION_V2),
    }:
        raise ValueError(f"unsupported Quest pose protocol magic/version: {magic!r}/{version}")
    timing_values = (xr_ms, sent_ms, frame_rate) if sampled_ms is None else (
        xr_ms, sampled_ms, sent_ms, frame_rate
    )
    if not all(np.isfinite(value) for value in timing_values):
        raise ValueError("Quest timing fields must be finite")
    if frame_rate <= 0.0 or frame_rate > 240.0:
        raise ValueError(f"Quest frame rate is invalid: {frame_rate}")
    controllers: dict[str, ControllerPose] = {}
    offset = header_size
    for side in ("left", "right"):
        unpacked = _HAND.unpack_from(payload, offset)
        flags = unpacked[0]
        controllers[side] = _validate_hand(
            side,
            ControllerPose(
                present=bool(flags & POSE_FLAG_PRESENT),
                emulated_position=bool(flags & POSE_FLAG_EMULATED_POSITION),
                position_m=np.asarray(unpacked[1:4], dtype=np.float64),
                orientation_xyzw=np.asarray(unpacked[4:8], dtype=np.float64),
                trigger=float(unpacked[8]),
                squeeze=float(unpacked[9]),
            ),
        )
        offset += _HAND.size
    return QuestPoseSample(
        sequence=int(sequence),
        reference_epoch=int(epoch),
        session_flags=int(session_flags),
        xr_time_ms=float(xr_ms),
        sent_time_ms=float(sent_ms),
        frame_rate_hz=float(frame_rate),
        controllers=controllers,
        host_received_ns=time.monotonic_ns() if host_received_ns is None else int(host_received_ns),
        sampled_time_ms=None if sampled_ms is None else float(sampled_ms),
    )


def encode_pose_packet(sample: QuestPoseSample) -> bytes:
    """Reference encoder used by synthetic publishers and protocol tests."""

    if sample.sampled_time_ms is None:
        payload = bytearray(POSE_PACKET_SIZE)
        _HEADER.pack_into(
            payload, 0, PROTOCOL_MAGIC, PROTOCOL_VERSION, int(sample.session_flags),
            int(sample.sequence), int(sample.reference_epoch), float(sample.xr_time_ms),
            float(sample.sent_time_ms), float(sample.frame_rate_hz),
        )
        offset = _HEADER.size
    else:
        payload = bytearray(POSE_PACKET_SIZE_V2)
        _HEADER_V2.pack_into(
            payload, 0, PROTOCOL_MAGIC_V2, PROTOCOL_VERSION_V2,
            int(sample.session_flags), int(sample.sequence), int(sample.reference_epoch),
            float(sample.xr_time_ms), float(sample.sampled_time_ms),
            float(sample.sent_time_ms), float(sample.frame_rate_hz),
        )
        offset = _HEADER_V2.size
    for side in ("left", "right"):
        pose = _validate_hand(side, sample.controllers.get(side, ControllerPose.missing()))
        flags = int(pose.present) * POSE_FLAG_PRESENT | int(pose.emulated_position) * POSE_FLAG_EMULATED_POSITION
        _HAND.pack_into(
            payload,
            offset,
            flags,
            *pose.position_m,
            *pose.orientation_xyzw,
            pose.trigger,
            pose.squeeze,
        )
        offset += _HAND.size
    return bytes(payload)


class LatestPoseMailbox:
    """Thread-safe, single-slot handoff that can never accumulate stale frames."""

    def __init__(self, *, capacity=256, history_ms=2000.) -> None:
        self._condition = threading.Condition()
        self._latest: QuestPoseSample | None = None
        self._history = deque(maxlen=capacity)
        self._cursor = 0
        self._history_ns = round(history_ms * 1e6)

    def clear(self) -> None:
        with self._condition:
            self._latest = None
            self._history.clear()

    def put(self, sample: QuestPoseSample) -> None:
        with self._condition:
            self._latest = sample
            self._cursor += 1
            self._history.append((self._cursor, sample))
            while self._history and sample.host_received_ns - self._history[0][1].host_received_ns > self._history_ns:
                self._history.popleft()
            self._condition.notify_all()

    def snapshot_since(self, cursor, *, max_batch=32):
        """Never wait for a packet. A lagging reader explicitly sees overflow."""
        with self._condition:
            rows = [sample for index, sample in self._history if index > cursor]
            overflow = (bool(self._history) and cursor < self._history[0][0] - 1) or len(rows) > max_batch
            return self._cursor, tuple(rows[-max_batch:]), overflow

    @property
    def cursor(self):
        with self._condition:
            return self._cursor

    def get(self) -> QuestPoseSample | None:
        with self._condition:
            return self._latest

    def wait(self, timeout_s: float | None = None) -> QuestPoseSample | None:
        with self._condition:
            if self._latest is None:
                self._condition.wait(timeout_s)
            return self._latest


@dataclasses.dataclass(frozen=True)
class ClockEstimate:
    host_minus_client_ms: float
    uncertainty_ms: float
    exchanges: int


class ClockSynchronizer:
    """NTP-style monotonic clock alignment for browser performance.now()."""

    def __init__(self, *, capacity: int = 64) -> None:
        self._samples: deque[tuple[float, float]] = deque(maxlen=capacity)
        self._lock = threading.Lock()

    def reset(self) -> None:
        with self._lock:
            self._samples.clear()

    def add_exchange(
        self,
        *,
        host_send_ns: int,
        client_receive_ms: float,
        client_send_ms: float,
        host_receive_ns: int,
    ) -> ClockEstimate:
        host_send_ms = host_send_ns / 1e6
        host_receive_ms = host_receive_ns / 1e6
        client_processing_ms = max(0.0, client_send_ms - client_receive_ms)
        network_rtt_ms = max(0.0, host_receive_ms - host_send_ms - client_processing_ms)
        uncertainty_ms = 0.5 * network_rtt_ms
        host_minus_client_ms = 0.5 * (host_send_ms - client_receive_ms + host_receive_ms - client_send_ms)
        if not np.isfinite(host_minus_client_ms) or not np.isfinite(uncertainty_ms):
            raise ValueError("clock exchange must be finite")
        with self._lock:
            self._samples.append((uncertainty_ms, host_minus_client_ms))
            return self._estimate_locked()

    def estimate(self) -> ClockEstimate | None:
        with self._lock:
            return None if not self._samples else self._estimate_locked()

    def _estimate_locked(self) -> ClockEstimate:
        selected = sorted(self._samples)[: min(8, len(self._samples))]
        offsets = np.asarray([offset for _, offset in selected], dtype=np.float64)
        uncertainties = np.asarray([uncertainty for uncertainty, _ in selected], dtype=np.float64)
        return ClockEstimate(
            host_minus_client_ms=float(np.median(offsets)),
            uncertainty_ms=float(np.percentile(uncertainties, 95)),
            exchanges=len(self._samples),
        )
