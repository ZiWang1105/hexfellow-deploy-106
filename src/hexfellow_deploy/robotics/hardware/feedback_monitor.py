"""Read-only receive-time diagnostics; never a motion authorization."""
from __future__ import annotations

import argparse
import csv
import dataclasses
import fcntl
import json
import math
import time
from collections import deque
from datetime import datetime, timezone
from pathlib import Path

from .timing_stats import LatencyStats
from .x7 import HardwareSafetyError, FeedbackUnavailable

DEVICES = ('left', 'right', 'base', 'theta')


def feedback_metrics(state, now):
    stamps = {n: float(state.received[n]) for n in DEVICES}
    if not math.isfinite(now) or not all(math.isfinite(t) for t in stamps.values()):
        raise HardwareSafetyError('invalid feedback clock')
    ages = {n: (now - t) * 1000 for n, t in stamps.items()}
    coupled = DEVICES if state.feedback_policy == 'strict' else DEVICES[:3]
    skew = (max(stamps[n] for n in coupled) - min(stamps[n] for n in coupled)) * 1000
    return dict(ages_ms=ages, skew_ms=skew,
                all_device_skew_ms=(max(stamps.values()) - min(stamps.values())) * 1000,
                pairwise_ms={f'{a}-{b}': abs(stamps[a]-stamps[b])*1000
                             for i, a in enumerate(DEVICES) for b in DEVICES[i+1:]})


class FeedbackObservations:
    """Single gateway-thread statistics, including snapshots missed by the parent."""
    def __init__(self):
        self.samples = self.invalid_samples = self.event_count = 0
        self.last_at = None
        self.max_sample_gap_ms = 0.
        self.max_skew_ms = 0.
        self.recent = deque(maxlen=8192)
        self.distribution = LatencyStats()
        self.events = []
        self.active = None
        self.latest = {}

    @staticmethod
    def window(endpoint, start, end):
        source = endpoint.diagnostic_window()
        result = {}
        for name, values in source.items():
            result[name] = {
                'frames': [v for v in values.get('frames', []) if start <= v['received'] <= end],
                'udp': [v for v in values.get('udp', []) if start <= v['udp_received_at'] <= end],
                'kcp_process': [v for v in values.get('kcp_process', []) if start <= v[0] <= end],
                'clock_note': values.get('clock_note'),
            }
        return result

    def observe(self, state, now, endpoint=None):
        self.samples += 1
        if self.last_at is not None:
            self.max_sample_gap_ms = max(self.max_sample_gap_ms, (now-self.last_at)*1000)
        self.last_at = now
        error = None
        try:
            metrics = feedback_metrics(state, now)
            state.validate(now)
        except HardwareSafetyError as exc:
            error = str(exc)
            try:
                metrics = feedback_metrics(state, now)
            except HardwareSafetyError:
                metrics = {}
        if metrics:
            self.max_skew_ms = max(self.max_skew_ms, metrics['skew_ms'])
            while self.recent and self.recent[-1][1] <= metrics['skew_ms']:
                self.recent.pop()
            self.recent.append((now, metrics['skew_ms']))
            while self.recent and self.recent[0][0] < now-1:
                self.recent.popleft()
            for key in ('skew_ms', 'all_device_skew_ms'):
                self.distribution.add(key, metrics[key]/1000)
            for name, age in metrics['ages_ms'].items():
                self.distribution.add(name+'_age_ms', age/1000)
        if error:
            self.invalid_samples += 1
            if self.active is None:
                self.event_count += 1
                self.active = dict(at=now, error=error, received=dict(state.received),
                                   metrics=metrics, recovered_at=None)
                if len(self.events) < 3:
                    self.events.append(self.active)
                    if endpoint is not None:
                        try:
                            self.active['before'] = self.window(endpoint, now-.1, now)
                        except Exception as exc:
                            self.active['window_error'] = str(exc)
        elif self.active is not None:
            self.active['recovered_at'] = now
            self.active = None
        for event in self.events:
            if endpoint is not None and now >= event['at']+.1 and 'after' not in event and 'window_error' not in event:
                try:
                    event['after'] = self.window(endpoint, event['at'], event['at']+.1)
                    event['post_window_complete'] = True
                except Exception as exc:
                    event['window_error'] = str(exc)
        self.latest = {**metrics, 'samples': self.samples, 'invalid_samples': self.invalid_samples,
                       'event_count': self.event_count, 'error': error,
                       'max_skew_ms': self.max_skew_ms,
                       'last_second_max_skew_ms': (self.recent[0][1] if self.recent else 0.),
                       'max_sample_gap_ms': self.max_sample_gap_ms}
        return self.latest

    def report(self):
        return {**self.latest, 'scope': 'gateway snapshots on feedback wakeup and 4ms fallback; not every hardware sample',
                'gateway_observed_feedback_clean': self.samples > 0 and self.invalid_samples == 0,
                'diagnostics_complete': all(e.get('post_window_complete', False) for e in self.events),
                'events_retained': len(self.events), 'events_not_retained': self.event_count-len(self.events),
                'events': self.events, 'distribution': self.distribution.report()}


def attach_feedback_result(report, *, require_clean=False):
    """Readiness may recover; strict audit must not hide gateway observations."""
    value = report.get('gateway', {}).get('feedback_observations', {})
    report['sampled_cycles_passed'] = bool(report.get('passed'))
    report['gateway_observed_feedback_clean'] = value.get('gateway_observed_feedback_clean', False)
    report['diagnostics_complete'] = value.get('diagnostics_complete', False)
    if require_clean and not (report['gateway_observed_feedback_clean'] and report['diagnostics_complete']):
        report['passed'] = False
        report.setdefault('error', 'Gateway observed feedback anomalies or incomplete diagnostics; passing main-loop samples does not establish an anomaly-free session')


def main(argv=None):
    from ..paths import CONFIG_DIR, PROJECT_ROOT
    from ..control.first_trial import apply_envelope
    from ..control.motion_profile import load_motion_profile
    from .gateway import HardwareGateway
    from .whole_body import RobotMapping
    from .verify import save_json
    from .verification import software_binding
    from .realtime import control_runtime
    from contextlib import ExitStack

    parser = argparse.ArgumentParser(description='Read-only feedback skew monitoring; no control acquisition or IK.')
    parser.add_argument('--seconds', type=float, default=60, help='0 runs until Ctrl+C')
    parser.add_argument('--robot-config', type=Path, default=CONFIG_DIR/'hardware/hexfellow_runtime.json')
    parser.add_argument('--run-dir', type=Path, default=PROJECT_ROOT/'runs/hexfellow-feedback-monitor')
    args = parser.parse_args(argv)
    if not math.isfinite(args.seconds) or args.seconds < 0:
        parser.error('--seconds must be finite and nonnegative')
    mapping = dataclasses.replace(RobotMapping.load(args.robot_config),
                                  motion_envelope='first-trial')
    profile = apply_envelope(load_motion_profile(mapping.data['commissioning']['motion_profile']), 'first-trial')
    root = PROJECT_ROOT/'runs'
    root.mkdir(exist_ok=True)
    with (root/'.hexfellow-onsite.lock').open('a') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            parser.exit(1, 'Another verification/control process is running; exit it first.\n')
        directory = args.run_dir/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S.%fZ')
        directory.mkdir(parents=True)
        gateway = HardwareGateway(mapping, profile, 'whole-body', 'probe', profile.mit_servo.torque_limit_nm)
        report = dict(mode='feedback-monitor', diagnostic_only=True, completed=False,
                      interrupted=False, motion_commands_sent=0, hardware_execution_qualified=False,
                      software_binding=software_binding(mapping))
        print('Read-only monitoring: no control acquisition or motion commands. Continue recording after skew violations; Ctrl+C to end.', flush=True)
        print('Control skew = max minus min latest receive time across left/right arms and base; Theta is reported separately.', flush=True)
        scope = ExitStack()
        started = None
        try:
            report['cpu_scheduling'] = scope.enter_context(control_runtime())
            gateway.start()
            started = None
            next_print = 0.
            initial_deadline = time.monotonic()+15
            fields = ['at', 'left_age_ms', 'right_age_ms', 'base_age_ms', 'theta_age_ms',
                      'skew_ms', 'all_device_skew_ms', 'gateway_samples', 'gateway_events', 'status']
            with (directory/'samples.csv').open('w', newline='') as stream:
                writer = csv.DictWriter(stream, fieldnames=fields)
                writer.writeheader()
                while True:
                    now = time.monotonic()
                    if started is not None and args.seconds and now-started >= args.seconds:
                        report['completed'] = True
                        break
                    try:
                        state = gateway.state(fresh=False)
                    except FeedbackUnavailable:
                        if started is None and now >= initial_deadline:
                            raise HardwareSafetyError('No complete feedback within 15 seconds')
                        time.sleep(.004)
                        continue
                    if started is None:
                        started = now
                    metrics = feedback_metrics(state, time.monotonic())
                    observed = (state.timing or {}).get('feedback_observations', {})
                    status = 'normal'
                    try:
                        state.validate(time.monotonic())
                    except HardwareSafetyError as exc:
                        status = str(exc)
                    writer.writerow(dict(at=now, **{n+'_age_ms': metrics['ages_ms'][n] for n in DEVICES},
                        skew_ms=metrics['skew_ms'], all_device_skew_ms=metrics['all_device_skew_ms'],
                        gateway_samples=observed.get('samples'), gateway_events=observed.get('event_count'), status=status))
                    if now >= next_print:
                        print(f"Skew {metrics['skew_ms']:.2f} ms | including Theta {metrics['all_device_skew_ms']:.2f} ms | "
                              f"age_ms { {n: round(v, 2) for n, v in metrics['ages_ms'].items()} } | "
                              f"last-second max {observed.get('last_second_max_skew_ms', 0):.2f} | "
                              f"session max {observed.get('max_skew_ms', 0):.2f} | "
                              f"anomaly events {observed.get('event_count', 0)} | {status}", flush=True)
                        next_print = now+1
                    time.sleep(.004)
        except KeyboardInterrupt:
            report['interrupted'] = True
        except Exception as exc:
            report['error'] = f'{type(exc).__name__}: {exc}'
        finally:
            try:
                report['gateway'] = gateway.close()
            except Exception as exc:
                report['error'] = report.get('error') or f'close failed: {exc}'
            scope.close()
            outcome = report.get('gateway', {})
            report['elapsed_seconds'] = time.monotonic()-started if started is not None else 0.
            report['motion_commands_sent'] = outcome.get('motion_commands_sent', 0)
            report['collection_succeeded'] = bool((report['completed'] or report['interrupted'])
                and not report.get('error') and not outcome.get('fault') and not outcome.get('stop_failures')
                and report['motion_commands_sent'] == 0)
            observations = outcome.get('feedback_observations', {})
            report['gateway_observed_feedback_clean'] = observations.get('gateway_observed_feedback_clean', False)
            report['diagnostics_complete'] = observations.get('diagnostics_complete', False)
            with (directory/'events.jsonl').open('w') as stream:
                for event in observations.get('events', []):
                    stream.write(json.dumps(event, ensure_ascii=False, allow_nan=False)+'\n')
            save_json(directory/'summary.json', report)
            print('Report:', directory/'summary.json', flush=True)
            if report.get('error') or outcome.get('fault'):
                print('Capture fault:', report.get('error') or outcome.get('fault'), flush=True)
            print('Capture completion does not establish clean communication; inspect gateway_observed_feedback_clean and diagnostics_complete.', flush=True)
        return 0 if report['collection_succeeded'] else 1
