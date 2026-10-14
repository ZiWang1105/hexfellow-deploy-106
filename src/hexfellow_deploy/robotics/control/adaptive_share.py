"""Bounded body participation, with transactionally restorable demand history."""
from dataclasses import dataclass

import numpy as np


def directional_lift_anchor(anchor, measured, delta, coherence, deadband=1e-5):
    """Drop stale soft lift demand opposed to coherent vertical motion.

    This changes an allocation preference, not the integrated reference or
    measured state. The QP still enforces every velocity/acceleration/jerk bound.
    """
    if coherence > .9 and abs(delta) > deadband and (anchor - measured) * delta < 0:
        return float(measured), True
    return float(anchor), False


@dataclass
class AdaptiveShare:
    goal: np.ndarray | None = None
    shares: np.ndarray | None = None

    def copy(self):
        return AdaptiveShare(None if self.goal is None else self.goal.copy(),
                             None if self.shares is None else self.shares.copy())

    def step(self, anchor, delta, coherence, strain, gap, lower, upper, rates, nominal, dt, stopped=False):
        anchor, delta, rates = map(np.asarray, (anchor, delta, rates))
        if self.goal is None:
            self.goal = anchor.copy()
            self.shares = np.asarray(nominal).copy()
        # Feedback lag and proximity to a finite bound reduce participation.
        health = np.clip(1. - np.abs(gap) / np.array([.04, .04, .08, .02]), 0., 1.)
        direction = np.sign(delta)
        room = np.where(direction >= 0, upper - anchor, anchor - lower)
        availability = np.clip(room / np.maximum(rates * .5, 1e-9), 0., 1.)
        desired = np.minimum(np.asarray(nominal) * (1. + strain), [.35, .35, .25, .35])
        desired *= np.asarray(coherence) * health * availability
        self.shares += (1. - np.exp(-dt / .22)) * (desired - self.shares)
        # Base yaw is represented by SO(3) anchors. Preserve the nearest goal
        # across +/-pi instead of interpreting a wrap as 2*pi of deferred work.
        yaw_gap = self.goal[2] - anchor[2]
        if abs(yaw_gap) > np.pi:
            self.goal[2] = anchor[2] + np.arctan2(np.sin(yaw_gap), np.cos(yaw_gap))
        pending = self.goal - anchor
        # A reversal, stop, or tracking exhaustion must not release old debt later.
        clear = (pending * delta < 0) | (health < .1) | stopped
        self.goal = np.where(clear, anchor, self.goal)
        if not stopped:
            self.goal += self.shares * delta
        self.goal = np.clip(self.goal, lower, upper)
        self.goal = anchor + np.clip(self.goal - anchor, -rates * .2, rates * .2)
        return np.clip(self.goal - anchor, -rates * dt, rates * dt)
