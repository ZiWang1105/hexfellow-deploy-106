"""Bounded timestamped EEF stream. No robot, network, or IK dependencies.

Execution time and producer freshness are distinct. Future execution timestamps
never refresh a source heartbeat. The consumer owns its playback/blending state.
"""
import threading
from collections import deque
from dataclasses import dataclass
from numbers import Integral

from ..contract import ARM_ORDER
from .action_buffer import WorldTarget, _validated_target, interpolate_targets, quintic_blend


@dataclass(frozen=True)
class EEFFrame:
    timestamp_ns: int                 # host-monotonic execution time
    source_ns: int                    # host-aligned capture/generation time
    received_ns: int
    sequence: int
    epoch: int
    arms: dict[str, WorldTarget]
    anchor_epochs: dict[str, int]
    active: dict[str, bool]


class TargetStreamBuffer:
    """Immutable frame snapshots under a short lock; interpolation outside it."""
    def __init__(self, capacity=256, *, history_ms=None):
        self._frames = deque(maxlen=capacity)
        self._lock = threading.Lock()
        self._history_ns = None if history_ms is None else round(history_ms * 1e6)

    def clear(self):
        with self._lock:
            self._frames.clear()

    def append(self, frame):
        if any(isinstance(v, bool) or not isinstance(v, Integral) or v < 0 for v in (
                frame.timestamp_ns, frame.source_ns, frame.received_ns, frame.sequence, frame.epoch)):
            raise ValueError('frame clocks, sequence and epoch must be nonnegative integers')
        arms = {s: _validated_target(frame.arms[s], s) for s in ARM_ORDER}
        copy = EEFFrame(frame.timestamp_ns, frame.source_ns, frame.received_ns,
                        frame.sequence, frame.epoch, arms, dict(frame.anchor_epochs), dict(frame.active))
        with self._lock:
            if self._frames and (frame.epoch != self._frames[-1].epoch
                                 or frame.timestamp_ns <= self._frames[-1].timestamp_ns):
                raise ValueError('target epoch/time changed without clearing stream')
            self._frames.append(copy)
            if self._history_ns is not None:
                while len(self._frames) > 2 and frame.timestamp_ns - self._frames[1].timestamp_ns > self._history_ns:
                    self._frames.popleft()

    def sample(self, now_ns, anchor_epochs, fallback):
        with self._lock:
            frames = tuple(self._frames)
        result, modes = {}, []
        for side in ARM_ORDER:
            rows = [r for r in frames if r.anchor_epochs[side] == anchor_epochs[side]]
            if not rows:
                result[side] = _validated_target(fallback[side], side)
                modes.append('empty')
                continue
            if now_ns <= rows[0].timestamp_ns:
                result[side] = _validated_target(rows[0].arms[side], side)
                modes.append('priming')
                continue
            if now_ns >= rows[-1].timestamp_ns:
                result[side] = _validated_target(rows[-1].arms[side], side)
                modes.append('endpoint_hold')
                continue
            right = next(i for i, row in enumerate(rows) if row.timestamp_ns > now_ns)
            a, b = rows[right - 1], rows[right]
            duration = (b.timestamp_ns - a.timestamp_ns) / 1e9
            fraction = (now_ns - a.timestamp_ns) / (b.timestamp_ns - a.timestamp_ns)
            result[side] = interpolate_targets(a.arms, b.arms, fraction,
                                               fraction_rate=1 / duration)[side]
            modes.append('interpolating')
        return result, dict(depth=len(frames), modes=modes,
                            coverage_ms=0. if not frames else (frames[-1].timestamp_ns - now_ns) / 1e6)


class ReferencePlayback:
    """Rejoin from the emitted reference, never jump to a newly arrived endpoint."""
    def __init__(self, buffer, blend_ms):
        self.buffer, self.blend_ns = buffer, round(blend_ms * 1e6)
        self.reset()

    def reset(self):
        self.last = self.anchor_epochs = self.blend_from = None
        self.starved = False
        self.blend_start = None
        self.last_sequence = None

    def sample(self, now_ns, anchors, fallback, sequence, *, stale=False):
        targets, diagnostics = self.buffer.sample(now_ns, anchors, fallback)
        changed = self.anchor_epochs is not None and anchors != self.anchor_epochs
        if changed and self.last is not None:
            for side in ARM_ORDER:
                if anchors[side] != self.anchor_epochs[side]:
                    self.last[side] = _validated_target(fallback[side], side)
                    if self.blend_from is not None:
                        self.blend_from[side] = self.last[side]
        if stale and self.last is not None:
            targets = self.last
            diagnostics['modes'] = ['stale_hold'] * 2
            self.blend_start = self.blend_from = None
        elif (self.starved and self.last is not None and sequence != self.last_sequence
              and self.blend_start is None and not changed):
            self.blend_start, self.blend_from = now_ns, self.last
        if self.blend_start is not None:
            u = (now_ns - self.blend_start) / self.blend_ns
            blend, rate, acceleration = quintic_blend(u, self.blend_ns / 1e9)
            targets = interpolate_targets(self.blend_from, targets, blend,
                                          fraction_rate=rate, fraction_acceleration=acceleration)
            diagnostics['rejoining'] = True
            if u >= 1:
                self.blend_start = self.blend_from = None
        self.starved = stale or any(m in ('empty', 'endpoint_hold') for m in diagnostics['modes'])
        self.last = targets
        self.anchor_epochs, self.last_sequence = dict(anchors), sequence
        return targets, diagnostics
