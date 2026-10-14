"""Diagnostics aggregation runs in the existing logger process, not the IK tick."""
from collections import Counter, defaultdict, deque

import numpy as np


def hold_category(reason):
    reason = str(reason).lower()
    if 'operator_release' in reason:
        return 'operator_release'
    if 'workspace' in reason:
        return 'workspace_protection'
    if 'clock_invalid' in reason:
        return 'clock_invalid'
    if 'source_' in reason or 'quest pose older' in reason:
        return 'source_timeout'
    if 'controldeadlineexceeded' in reason or 'scheduling' in reason or 'control slot' in reason:
        return 'ik_deadline'
    if 'dispatch' in reason or 'command' in reason or 'publish' in reason:
        return 'dispatch_delay'
    if 'feedback' in reason or 'ipc' in reason:
        return 'feedback_delay'
    return 'safety_or_operator'


class StreamDiagnostics:
    def __init__(self):
        self.rows = defaultdict(lambda: deque(maxlen=3000))
        self.holds = Counter()
        self.modes = Counter()
        self.deadline_misses = 0
        self.cycles = 0
        self.latest = {}
        self.temperature_max = {}
        from ..control.hardware_capabilities import load_hardware_capabilities
        from .effort_monitor import EffortLoadMonitor
        continuous = load_hardware_capabilities().continuous_torque_nm
        self.effort_monitors = {"predicted": EffortLoadMonitor(continuous),
                                "left_sdk": EffortLoadMonitor(continuous[:7]),
                                "right_sdk": EffortLoadMonitor(continuous[7:])}

    def record(self, row):
        kind = row.get('kind')
        if kind in ('source_hold', 'delay_hold', 'schedule_miss', 'operator_release'):
            category = hold_category(row.get('reason'))
            self.holds[category] += 1
            if kind == 'schedule_miss':
                self.deadline_misses += 1
        if kind == 'terminal_command' and row.get('command') in ('stop', 'quit'):
            self.holds['operator_stop'] += 1
        if kind == 'cycle':
            self.cycles += 1
            self.effort_monitors["predicted"].observe(
                row.get("at"), (row.get("predicted_mit_effort") or {}).get("requested_nm"))
            health = (row.get("state") or {}).get("health", {})
            for side in ("left", "right"):
                feedback = health.get(side, {})
                for field in ("motor_temperature_c", "driver_temperature_c"):
                    values = feedback.get(field)
                    if values is not None:
                        valid = [float(v) for v in values if v is not None and np.isfinite(v)]
                        if valid:
                            key = side + "_" + field
                            self.temperature_max[key] = max(self.temperature_max.get(key, -np.inf), max(valid))
                self.effort_monitors[side + "_sdk"].observe(
                    feedback.get("received"), feedback.get("effort_model_nm"))
            self.latest = {key: row.get(key) for key in ('at', 'control_state', 'source', 'quest_link', 'hold_reason')}
            for key in ('loop_dt_ms', 'schedule_lateness_ms', 'control_cycle_ms', 'qp_solve_ms'):
                value = row.get(key)
                if value is not None:
                    self.rows[key].append(value)
            total = (row.get('timings') or {}).get('total_ms')
            if total is not None:
                self.rows['ik_total_ms'].append(total)
            if row.get('fault'):
                category = hold_category(row['fault'])
                self.holds[category] += 1
                self.deadline_misses += category == 'ik_deadline'
            for mode in (row.get('source') or {}).get('playback', {}).get('modes', []):
                self.modes[mode] += 1

    def report(self):
        stats = {k: dict(mean=float(np.mean(v)), p95=float(np.percentile(v, 95)),
                         p99=float(np.percentile(v, 99)), max=float(max(v)), count=len(v))
                 for k, v in self.rows.items() if v}
        dt = stats.get('loop_dt_ms', {}).get('mean', 0.)
        return dict(temperature_max_c=dict(self.temperature_max),
                    temperature_freshness="SDK cache timestamps unavailable; not thermal qualification",
                    effort_load={k: v.report() for k, v in self.effort_monitors.items()},
                    cycles=self.cycles, actual_ik_hz=1000 / dt if dt > 0 else None,
                    deadline_misses=self.deadline_misses, hold_counts=dict(self.holds),
                    playback_modes=dict(self.modes), rolling_metrics=stats, latest=self.latest,
                    metrics_window='last 3000 completed IK cycles; includes gaps between solves')
