"""Mapped TCP recordings and a feedback-driven MIT preparation/replay sequence.

No SDK or simulator imports. Recorded joints specify the initial posture only.
"""

from __future__ import annotations

import hashlib
import io
import json
import zipfile
from pathlib import Path

import numpy as np
import pinocchio as pin

from .ik_core import TCP_FRAMES
from .target_timing import RecordingTimeline, TargetSample

COLLISION_MODES = ("enabled", "disabled")


def collision_config(profile, mode):
    from .motion_profile import controller_config_for_profile

    if mode not in COLLISION_MODES:
        raise ValueError("collision mode must be enabled or disabled")
    return controller_config_for_profile(profile, self_collision_enabled=mode == "enabled")


def tcp_poses(core, q):
    return core._command_frame_poses(q)


def planar_transform(q):
    return pin.SE3(pin.exp3(np.array([0., 0., q[2]])), np.array([q[0], q[1], 0.]))


class RecordedTargets:
    def __init__(self, path, start_tick=None, end_tick=None, *, clock_mode="recorded"):
        path = Path(path)
        path = path / "quest_targets.npz" if path.is_dir() else path
        if path.stat().st_size > 128 * 1024**2:
            raise ValueError("recording exceeds 128 MiB")
        payload = path.read_bytes()
        with zipfile.ZipFile(io.BytesIO(payload)) as archive:
            if sum(row.file_size for row in archive.infolist()) > 256 * 1024**2:
                raise ValueError("expanded recording exceeds 256 MiB")
        self.summary = json.loads(path.with_name("summary.json").read_text())
        if not isinstance(self.summary, dict):
            raise ValueError("recording summary must be an object")
        with np.load(io.BytesIO(payload), allow_pickle=False) as stored:
            if int(stored["trace_schema_version"]) not in (4, 5, 6):
                raise ValueError("unsupported target recording schema")
            ticks = stored["original_tick"]
            if (ticks.ndim != 1 or len(ticks) < 2 or not np.issubdtype(ticks.dtype, np.integer)
                    or ticks[0] < 0 or np.any(np.diff(ticks) != 1)):
                raise ValueError("recording ticks must be consecutive nonnegative integers")
            start_tick = int(ticks[0]) if start_tick is None else start_tick
            end_tick = int(ticks[-1]) if end_tick is None else end_tick
            if not ticks[0] <= start_tick < end_tick <= ticks[-1]:
                raise ValueError("invalid replay tick interval")
            selected = np.flatnonzero((ticks >= start_tick) & (ticks <= end_tick))
            self.timeline = RecordingTimeline.load(stored, selected, mode=clock_mode)
            transforms = np.asarray(stored["transforms"], dtype=float)
            if transforms.shape != (len(ticks), 2, 4, 4):
                raise ValueError("expected dual-arm transforms of shape (ticks, 2, 4, 4)")
            self.transforms = transforms[selected].copy()
            rotations = self.transforms[:, :, :3, :3]
            if (not np.isfinite(self.transforms).all()
                    or not np.allclose(self.transforms[:, :, 3], [0., 0., 0., 1.], atol=1e-8, rtol=0)
                    or not np.allclose(rotations.swapaxes(-1, -2) @ rotations, np.eye(3), atol=1e-6, rtol=0)
                    or not np.allclose(np.linalg.det(rotations), 1., atol=1e-6, rtol=0)):
                raise ValueError("recorded targets must be finite rigid transforms")
            q = np.asarray(stored["measured_q"], dtype=float)
            if q.shape != (len(ticks), 18) or not np.isfinite(q[selected[0]]).all():
                raise ValueError("recording requires a finite 18D starting posture")
            self.initial_q = q[selected[0]].copy()
            epoch = stored["raw_reference_epoch"]
            if epoch.shape != ticks.shape:
                raise ValueError("invalid reference epoch metadata")
            self.epochs = np.asarray(epoch[selected], dtype=np.int64).copy()
            self.source_times = self.source_sequences = None
            if (clock_mode == "recorded" and "raw_sampled_time_ms" in stored
                    and "sample_sequence" in stored):
                source_times = np.asarray(stored["raw_sampled_time_ms"], dtype=float)
                source_sequences = np.asarray(stored["sample_sequence"], dtype=np.int64)
                if source_times.shape != ticks.shape or source_sequences.shape != ticks.shape:
                    raise ValueError("invalid source timestamp/sequence shape")
                self.source_times = source_times[selected].copy() / 1000.
                self.source_sequences = source_sequences[selected].copy()
            changes = np.r_[0, np.flatnonzero(np.diff(epoch[selected])) + 1]
            transitions = [(int(ticks[selected[i]]), int(epoch[selected[i]])) for i in changes]
        self.duration = self.timeline.duration
        self.metadata = dict(path=str(path.resolve()), sha256=hashlib.sha256(payload).hexdigest(),
                             start_tick=int(start_tick), end_tick=int(end_tick), source_duration_s=self.duration,
                             clock_quality=self.timeline.quality, interpolation="causal_zero_order_hold",
                             input="mapped_dual_tcp_targets", spatial_scale=1., reference_epoch_transitions=transitions)

    def sample(self, elapsed, speed=1.):
        i = self.timeline.index(elapsed, speed)
        if (self.source_times is not None and np.isfinite(self.source_times[i])
                and self.source_sequences[i] >= 0):
            return TargetSample(float(self.source_times[i]) / speed, int(self.source_sequences[i]),
                                int(self.epochs[i]), quality="recorded_source_monotonic")
        return TargetSample(float(self.timeline.times[i]) / speed, i, int(self.epochs[i]),
                            quality=self.timeline.quality)

    def targets(self, elapsed, speed, transform):
        i = self.timeline.index(elapsed, speed)
        result = {}
        for arm, side in enumerate(TCP_FRAMES):
            matrix = self.transforms[i, arm]
            result[side] = transform * pin.SE3(matrix[:3, :3], matrix[:3, 3])
        return result


class PostureSegment:
    def __init__(self, origin, target, core):
        self.origin, self.target = origin.copy(), target.copy()
        self.delta = pin.difference(core.model, origin, target)
        distance = np.abs(self.delta)
        self.duration = float(max(8., np.max(1.875 * distance / core.model.velocityLimit),
                                  np.max(np.sqrt((10. / np.sqrt(3.)) * distance / core._acceleration_limits)),
                                  np.max(np.cbrt(60. * distance / core._jerk_limits))))

    def sample(self, elapsed):
        u = float(np.clip(elapsed / self.duration, 0., 1.))
        blend = 10 * u**3 - 15 * u**4 + 6 * u**5
        derivative = (30 * u**2 - 60 * u**3 + 30 * u**4) / self.duration
        return self.origin + blend * self.delta, derivative * self.delta


class ReplaySequence:
    """Preparation waits for arrival; recording time never waits for TCP error."""

    def __init__(self, core, recording, initial_q, speed=1.):
        if not np.isfinite(speed) or not 0 < speed <= 1:
            raise ValueError("replay speed must be in (0, 1]")
        self.core, self.recording, self.speed = core, recording, float(speed)
        self.transform = planar_transform(initial_q) * planar_transform(recording.initial_q).inverse()
        self.initial = initial_q.copy()
        self.aligned = recording.initial_q.copy()
        self.aligned[:3] = initial_q[:3]
        self.down = initial_q.copy()
        self.down[4:] = np.deg2rad([90, 90, 0, 0, 0, 0, 0, -90, 90, 0, 0, 0, 0, 0])
        for q in (self.initial, self.down, self.aligned):
            core._validate_configuration_vector(q)
        self.segments = {"arms_down": PostureSegment(self.initial, self.down, core),
                         "alignment": PostureSegment(self.down, self.aligned, core)}
        self.phase, self.started, self.stable = "hold", 0., 0.
        self.last_elapsed = 0.
        self.target_sample = None
        self.events = [{"phase": self.phase, "elapsed_s": 0.}]
        self.last_targets = tcp_poses(core, initial_q)

    def _enter(self, phase, elapsed):
        self.phase, self.started, self.stable = phase, elapsed, 0.
        self.events.append({"phase": phase, "elapsed_s": elapsed})
        if phase == "stop":
            self.core.request_stop()

    def sample(self, elapsed, q, v, *, fresh=True, observe=True):
        if not np.isfinite(elapsed) or elapsed < self.last_elapsed:
            raise ValueError("replay clock must be finite and monotonic")
        dt = elapsed - self.last_elapsed
        self.last_elapsed = elapsed
        phase_time = max(0., elapsed - self.started)
        self.target_sample = None
        posture, velocity = None, None
        if self.phase in {"hold", "arms_down", "alignment"}:
            if self.phase == "hold":
                posture, velocity, duration = self.initial, np.zeros(18), 3.
                goal = self.initial
            else:
                segment = self.segments[self.phase]
                posture, velocity = segment.sample(phase_time)
                duration, goal = segment.duration, segment.target
            tolerance = np.r_[.003, .003, .01, .003, np.full(14, .02)]
            speed_limit = np.r_[.003, .003, .02, .003, np.full(14, .02)]
            arrived = bool(np.all(abs(q - goal) <= tolerance) and np.all(abs(v) <= speed_limit))
            if fresh and (arrived or not observe):
                self.stable += dt
            elif not arrived:
                self.stable = 0.
            dwell = 3. if self.phase == "hold" else .5
            if phase_time >= duration and self.stable >= dwell:
                next_phase = {"hold": "arms_down", "arms_down": "alignment", "alignment": "replay"}[self.phase]
                self._enter(next_phase, elapsed + self.core.config.control_dt)
            elif observe and phase_time > duration + 10.:
                raise RuntimeError(f"{self.phase} did not reach its posture within 10 seconds")
            self.last_targets = tcp_poses(self.core, posture)
        elif self.phase in {"replay", "settle"}:
            replay_time = phase_time if self.phase == "replay" else self.recording.duration / self.speed
            self.last_targets = self.recording.targets(replay_time, self.speed, self.transform)
            if self.phase == "replay":
                self.target_sample = self.recording.sample(replay_time, self.speed)
            if self.phase == "replay" and phase_time >= self.recording.duration / self.speed:
                self._enter("settle", elapsed)
            elif self.phase == "settle":
                actual = tcp_poses(self.core, q)
                arrived = all(np.linalg.norm(actual[s].translation - t.translation) <= .003
                              and np.linalg.norm(pin.log3(actual[s].rotation.T @ t.rotation)) <= np.deg2rad(2)
                              for s, t in self.last_targets.items())
                if fresh and (arrived or not observe):
                    self.stable += dt
                elif not arrived:
                    self.stable = 0.
                if self.stable >= .5:
                    self._enter("stop", elapsed)
                elif observe and phase_time > 10.:
                    raise RuntimeError("final TCP targets did not settle within 10 seconds")
        elif self.phase == "stop":
            if self.core.control_state == "HOLD" or (not observe and phase_time >= 2.):
                self._enter("complete", elapsed)
            elif phase_time > 2.:
                raise RuntimeError("controlled stop did not reach HOLD within 2 seconds")
        return self.last_targets, posture, velocity
