"""Session-preserving delay recovery. Never grants device ownership.

The SDK gateway owns the recovery epoch. A parent can request resumption only
after new stationary feedback and drainage of old KCP motion have qualified it.
"""
from dataclasses import dataclass

import numpy as np

from .x7 import ControlDelay, HardwareSafetyError


@dataclass(frozen=True)
class ResumeRequest:
    epoch: int


class FreshInputWindow:
    """Only advancing input sequences qualify automatic clutch recovery."""
    def __init__(self, *, stable_s=.5, max_gap_s=.1):
        self.stable_s, self.max_gap_s = stable_s, max_gap_s
        self.reset()

    def reset(self):
        self.started = self.last_at = self.sequence = None

    def observe(self, sequence, now):
        if self.last_at is None or now - self.last_at > self.max_gap_s:
            self.started = now
        if sequence == self.sequence:
            return False
        self.sequence, self.last_at = sequence, now
        return now - self.started >= self.stable_s


class DelayRecovery:
    def __init__(self, clock):
        self.clock = clock
        self.epoch = 0
        self.phase = "RUNNING"
        self.reason = None
        self.started = None
        self.total_hold_s = 0.
        self.holds = 0
        self.resumes = 0
        self.previous = {}
        self.stable = {}
        self.seed = None
        self.drained = False

    def enter(self, reason):
        if self.phase != "RUNNING":
            return False
        self.epoch += 1
        self.holds += 1
        self.phase = "DELAY_HOLD"
        self.reason, self.started = str(reason), self.clock()
        self.previous, self.stable, self.seed = {}, {}, None
        return True

    def observe(self, state, *, drained):
        from .whole_body import SLICES
        self.drained = drained
        try:
            state.validate(self.clock())
        except ControlDelay:
            self.previous, self.stable = {}, {}
            self.phase = "DELAY_HOLD"
            return False
        if not drained:
            self.previous, self.stable = {}, {}
            self.phase = "DELAY_HOLD"
            return False
        for name, part in SLICES.items():
            stamp = state.received[name]
            old = self.previous.get(name)
            if old is not None and stamp < old:
                raise HardwareSafetyError("feedback timestamp regressed during recovery")
            if stamp == old:
                continue
            self.previous[name] = stamp
            if old is None or stamp - old > .04 + 1e-9:
                self.stable[name] = stamp
            if np.max(np.abs(state.v[part])) > (.01 if name in {"left", "right"} else .002):
                self.stable[name] = stamp
        ready = all(self.previous.get(n, 0) - self.stable.get(n, self.clock()) >= .5
                    for n in SLICES)
        self.phase = "RECOVERING" if ready else "DELAY_HOLD"
        return ready

    def resume(self, seed):
        if self.phase != "RECOVERING":
            raise HardwareSafetyError("recovery has no stationary feedback qualification")
        self.total_hold_s += self.clock() - self.started
        self.seed = seed
        self.phase = "RUNNING"
        self.resumes += 1

    def report(self):
        return dict(phase=self.phase, epoch=self.epoch, reason=self.reason,
                    holds=self.holds, resumes=self.resumes,
                    transport_drained=self.drained,
                    stable_seconds={n: self.previous[n] - self.stable.get(n, self.previous[n])
                                    for n in self.previous},
                    hold_seconds=self.total_hold_s + (
                        self.clock() - self.started if self.phase != "RUNNING" else 0.))
