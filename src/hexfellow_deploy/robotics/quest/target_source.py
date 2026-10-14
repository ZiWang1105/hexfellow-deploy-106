"""Quest adapter for a controller-clock EEF stream; no SDK or robot ownership."""
import math
from collections import Counter

import numpy as np
import pinocchio as pin

from ..contract import ARM_ORDER
from ..control.target_timing import TargetSample
from ..runtime.action_buffer import WorldTarget
from ..runtime.target_stream import EEFFrame, ReferencePlayback, TargetStreamBuffer
from .protocol import SESSION_FLAG_VISIBLE


class SourceTimeout(RuntimeError):
    """Recoverable source failure: use the existing controlled brake."""


class SourceFault(RuntimeError):
    """Tracking or reference-frame change requires explicit requalification."""


class QuestTargetSource:
    def __init__(self, mailbox, mapper, config):
        self.mailbox, self.mapper, self.config = mailbox, mapper, config
        self.buffer = TargetStreamBuffer(config.history_capacity, history_ms=config.history_ms)
        self.playback = ReferencePlayback(self.buffer, config.rejoin_blend_ms)
        self.counts = Counter()
        self.cursor = mailbox.cursor
        self.pending = []
        self.latest = None
        self.last_source_ms = self.last_aligned_ns = self.last_received_ns = None
        self.last_sequence = self.reference_epoch = None
        self.clock_error = None
        self.mapping_enabled = True
        self.tick = self.epoch = 0
        self.anchors = dict.fromkeys(ARM_ORDER, 0)
        self.active = dict.fromkeys(ARM_ORDER, False)
        self.fallback = None
        self.diagnostics = {}

    def reset(self, poses, widths, *, require_release=True):
        self.mapper.reset(poses, widths, require_release=require_release)
        self.buffer.clear()
        self.playback.reset()
        self.cursor = self.mailbox.cursor
        self.pending.clear()
        self.mapping_enabled = True
        self.epoch += 1
        self.anchors = dict.fromkeys(ARM_ORDER, self.epoch)
        self.active = dict.fromkeys(ARM_ORDER, False)
        self.fallback = {s: WorldTarget(np.asarray(poses[s]).copy(), widths[s]) for s in ARM_ORDER}
        self._previous_raw = None
        # Reset playback, NOT the producer heartbeat. Re-arming must not turn an
        # old packet into a fresh sample or clear timestamp regression evidence.
        if self.latest is None:
            sample = self.mailbox.get()
            if sample is not None:
                self._ingest(sample, sample.host_received_ns)
                self.pending.clear()

    def suspend(self):
        self.mapping_enabled = False
        self.pending.clear()

    def _ingest(self, sample, now_ns):
        if not sample.session_flags & SESSION_FLAG_VISIBLE:
            raise SourceFault('Quest session is not visible')
        if self.reference_epoch is not None and sample.reference_epoch != self.reference_epoch:
            raise SourceFault('Quest reference space changed')
        for side in ARM_ORDER:
            pose = sample.controllers.get(side)
            fault = 'pose_missing' if pose is None else self.mapper._tracking_fault(pose)
            if fault:
                raise SourceFault(f'Quest tracking fault: {side}:{fault}')
        if self.last_sequence is not None:
            delta = (sample.sequence - self.last_sequence) & 0xffffffff
            if delta == 0 or delta >= 0x80000000:
                self.counts['duplicate' if delta == 0 else 'out_of_order'] += 1
                return
        values = (sample.sampled_time_ms, sample.clock_offset_ms, sample.clock_uncertainty_ms)
        if any(v is None or not math.isfinite(v) for v in values):
            self.clock_error = 'clock_invalid: missing/nonfinite source timing'
            self.counts['timestamp_rejected'] += 1
            return
        uncertainty = sample.clock_uncertainty_ms
        aligned = round((sample.sampled_time_ms + sample.clock_offset_ms) * 1e6)
        if (not 0 <= uncertainty <= self.config.clock_uncertainty_ms
                or aligned > sample.host_received_ns + round(uncertainty * 1e6)
                or self.last_source_ms is not None and sample.sampled_time_ms <= self.last_source_ms
                or self.last_aligned_ns is not None and aligned <= self.last_aligned_ns):
            self.clock_error = 'clock_invalid: uncertainty/future/regressed timestamp'
            self.counts['timestamp_rejected'] += 1
            return
        self.clock_error = None
        self.last_source_ms, self.last_aligned_ns = sample.sampled_time_ms, aligned
        self.last_sequence, self.reference_epoch = sample.sequence, sample.reference_epoch
        self.latest = sample
        if self.age_ms(now_ns) >= self.config.age_stale_ms:
            self.counts['stale_filtered'] += 1
            # Stale motion is unusable; a release must still reach the consumer.
            if self.mapping_enabled and any(
                    p.squeeze <= self.mapper.release_threshold for p in sample.controllers.values()):
                self.pending.append((sample, aligned))
            return
        self.last_received_ns = sample.host_received_ns
        self.counts['accepted'] += 1
        if self.mapping_enabled:
            self.pending.append((sample, aligned))

    def age_ms(self, now_ns):
        if self.latest is None:
            return math.inf
        return (now_ns - self.last_aligned_ns) / 1e6 + self.latest.clock_uncertainty_ms

    def poll(self, now_ns):
        self.cursor, rows, overflow = self.mailbox.snapshot_since(self.cursor, max_batch=self.config.max_batch)
        clock_incident = None
        for sample in rows:
            self._ingest(sample, now_ns)
            clock_incident = self.clock_error or clock_incident
        if overflow or len(self.pending) > self.config.max_batch:
            self.pending.clear()
            self.counts['overflow'] += 1
            self.suspend()
            raise SourceTimeout('source_overflow: unread target/clutch history lost')
        age = self.age_ms(now_ns)
        gap = math.inf if self.last_received_ns is None else (now_ns - self.last_received_ns) / 1e6
        state = 'normal'
        for level in ('warning', 'stale', 'timeout'):
            if (age >= getattr(self.config, 'age_' + level + '_ms')
                    or gap >= getattr(self.config, 'gap_' + level + '_ms')):
                state = level
        self.diagnostics.update(source_state=state, source_age_ms=age if math.isfinite(age) else None,
                                no_update_ms=gap if math.isfinite(gap) else None,
                                source_counts=dict(self.counts), raw_sequence=self.last_sequence,
                                source_ns=self.last_aligned_ns)
        if self.clock_error or clock_incident:
            self.diagnostics['hold_reason'] = self.clock_error or clock_incident
            raise SourceTimeout(self.clock_error or clock_incident)
        if state == 'timeout':
            reason = 'source_timeout: ' + ('no_new_target' if gap >= self.config.gap_timeout_ms else 'source_age')
            self.diagnostics['hold_reason'] = reason
            raise SourceTimeout(reason)
        self.diagnostics.pop('hold_reason', None)
        return self.latest

    def targets(self, poses, widths, now_ns):
        sample = self.poll(now_ns)
        if self.fallback is None:
            raise RuntimeError('reset source before playback')
        previous_raw = getattr(self, '_previous_raw', None)
        jumps = dict(position_m=0., rotation_rad=0.)
        for row, aligned in self.pending:
            if (now_ns - aligned) / 1e6 + row.clock_uncertainty_ms >= self.config.age_stale_ms:
                for side in ARM_ORDER:
                    if row.controllers[side].squeeze <= self.mapper.release_threshold:
                        state = self.mapper._states[side]
                        if state.active:
                            state.active = False
                            state.require_release = False
                            self.active[side] = False
                            self.anchors[side] += 1
                            held = (state.held_transform if self.playback.last is None
                                    else self.playback.last[side].transform)
                            state.held_transform = held.copy()
                            self.fallback[side] = WorldTarget(held.copy(), widths[side])
                continue
            # Health owns source age. Mapper still checks tracking and physical
            # controller discontinuities using the same source sampling clock.
            from dataclasses import replace
            mapped = self.mapper.update(replace(row, xr_time_ms=row.sampled_time_ms), now_ns=now_ns,
                                        measured_tcp=poses, gripper_widths=widths)
            if any(mapped.faults.values()):
                raise SourceFault(f'Quest mapping fault: {mapped.faults}')
            for side in ARM_ORDER:
                if mapped.active[side] != self.active[side]:
                    self.anchors[side] += 1
                    if not mapped.active[side] and self.playback.last is not None:
                        # Release freezes the reference actually emitted, not
                        # a later raw endpoint that is still in the buffer.
                        held = self.playback.last[side]
                        self.mapper._states[side].held_transform = held.transform.copy()
                        mapped.targets[side] = WorldTarget(held.transform.copy(), widths[side])
                    self.fallback[side] = mapped.targets[side]
                if previous_raw is not None:
                    a, b = previous_raw[side].transform, mapped.targets[side].transform
                    jumps['position_m'] = max(jumps['position_m'], float(np.linalg.norm(b[:3, 3] - a[:3, 3])))
                    jumps['rotation_rad'] = max(jumps['rotation_rad'],
                        float(np.linalg.norm(pin.log3(a[:3, :3].T @ b[:3, :3]))))
            self.active = dict(mapped.active)
            self.buffer.append(EEFFrame(aligned + round(self.config.playout_delay_ms * 1e6), aligned,
                                        row.host_received_ns, row.sequence, self.epoch, mapped.targets,
                                        self.anchors, self.active))
            previous_raw = mapped.targets
        self.pending.clear()
        self._previous_raw = previous_raw
        previous_output = self.playback.last
        targets, playback = self.playback.sample(now_ns, self.anchors, self.fallback, self.last_sequence,
                                               stale=self.diagnostics['source_state'] == 'stale')
        speeds = {}
        previous_ns = self.diagnostics.get('reference_time_ns')
        if previous_output is not None and previous_ns is not None and now_ns > previous_ns:
            dt = (now_ns - previous_ns) / 1e9
            for side in ARM_ORDER:
                a, b = previous_output[side].transform, targets[side].transform
                speeds[side] = dict(linear_m_s=float(np.linalg.norm(b[:3, 3] - a[:3, 3]) / dt),
                    angular_rad_s=float(np.linalg.norm(pin.log3(a[:3, :3].T @ b[:3, :3])) / dt))
        self.tick += 1
        self.target_sample = TargetSample(now_ns / 1e9, self.tick, self.epoch, 'resampled_reference')
        self.diagnostics.update(playback=playback, raw_target_jump=jumps,
                                reference_time_ns=now_ns, reference_sequence=self.tick,
                                interpolated_velocity=speeds)
        return targets, dict(self.active), sample
