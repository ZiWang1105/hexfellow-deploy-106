"""Small shutdown evidence and bounded, isolated diagnostic export."""
import json
import multiprocessing as mp
import tempfile
import time
from pathlib import Path

SHUTDOWN_TIMEOUT = 35.0


class ThetaObservedStop:
    """Demo evidence of stationary behavior; never a firmware brake ACK."""
    def __init__(self):
        self.previous = None
        self.since = None
        self.low = self.high = None
        self.details = {}

    def reset(self):
        self.since = self.low = self.high = None

    def update(self, state, now, requested, transport, healthy):
        import math
        stamp, position = state.received['theta'], float(state.q[3])
        velocity = float(state.v[3])
        row = state.health['theta']
        self.details = dict(acceptance='unconfirmed', firmware_brake_confirmed=row.get('state') == 'LsBrake',
                            brake_received=transport.get('brake_received', False), transport=transport,
                            observed_stable_s=0., position_range_m=None)
        valid = (healthy and all(math.isfinite(v) for v in (stamp, position, velocity, now))
                 and 0 <= now-stamp <= .08 and stamp > requested
                 and transport.get('brake_received') and transport.get('no_pending_motion')
                 and stamp > transport.get('acknowledged_at', float('inf'))
                 and row.get('state') in {'LsBrake', 'LsAlgorithmControl'})
        if not valid:
            self.reset()
            return False
        if self.previous is not None and stamp <= self.previous[0]:
            if stamp < self.previous[0]:
                self.reset()
            return False
        derived_speed = None if self.previous is None else abs(position-self.previous[1])/(stamp-self.previous[0])
        if self.previous is not None and stamp-self.previous[0] > .08:
            self.reset()
            derived_speed = None
        self.previous = (stamp, position)
        if derived_speed is None or abs(velocity) > .002 or derived_speed > .002:
            self.reset()
            return False
        if self.since is None:
            self.since, self.low, self.high = stamp, position, position
        self.low, self.high = min(self.low, position), max(self.high, position)
        if self.high-self.low > .001:
            self.since, self.low, self.high = stamp, position, position
        stable = stamp-self.since
        self.details.update(observed_stable_s=stable, position_range_m=self.high-self.low)
        if stable >= .5:
            self.details['acceptance'] = 'observed_stationary_after_brake_receipt'
            return True
        return False


def compact_state(state):
    if state is None:
        return None
    from .ik_runner import jsonable
    return jsonable({name: getattr(state, name) for name in ('q', 'v', 'received', 'health')})


def physical_estop(state, armed_at):
    if state is None or armed_at is None:
        return False
    health = state['health'] if isinstance(state, dict) else state.health
    received = state['received'] if isinstance(state, dict) else state.received
    return any(((row.get('parking_stop') or {}).get('category') in {0, 8}
                or (name == 'theta' and row.get('state') == 'LsEmergencyStop'))
               and received.get(name, 0) >= armed_at for name, row in health.items())


def expected_estop_success(outcome, armed_at):
    if (outcome.get('kind') != 'closed' or outcome.get('stop_failures')
            or 'parking/emergency stop is active' not in str(outcome.get('fault', ''))
            or not physical_estop(outcome.get('fault_state'), armed_at)):
        return False
    state = outcome['fault_state']
    health = state['health'] if isinstance(state, dict) else state.health
    for name in ('left', 'right', 'theta', 'base'):
        if not outcome.get('stop_observations', {}).get(name, {}).get('continuous_stop_confirmed'):
            return False
        row = health.get(name, {})
        parking = row.get('parking_stop')
        if parking is not None and parking.get('category') not in {0, 8}:
            return False
        if name != 'theta' and (not row.get('motor_states') or any(s != 'normal' for s in row['motor_states'])):
            return False
        if name == 'theta' and not row.get('calibrated'):
            return False
        if name == 'base' and row.get('warning') is not None:
            return False
        if outcome.get('direct_downlink', {}).get(name, {}).get('fault'):
            return False
    return True


def _write_diagnostics(path, data):
    from .ik_runner import jsonable
    target = Path(path)
    temporary = target.with_suffix('.tmp')
    with temporary.open('w') as output:
        json.dump(jsonable(data), output, allow_nan=False)
    temporary.replace(target)


def export_diagnostics(report):
    """Only the diagnostic child may be terminated; never the SDK gateway."""
    keys = ('feedback_window', 'stop_window',
            'first_feedback_anomaly', 'feedback_latency_all_frames')
    data = {key: report.pop(key) for key in keys if key in report}
    if report.get('fault_state') is not None:
        data['fault_state'] = report['fault_state']
        state = report['fault_state']
        report['fault_state'] = compact_state(state) if not isinstance(state, dict) else {
            key: state[key] for key in ('q', 'v', 'received', 'health')}
    if not data:
        return
    directory = Path(tempfile.mkdtemp(prefix='hexfellow-shutdown-'))
    path = directory / 'diagnostics.json'
    child = mp.get_context('spawn').Process(target=_write_diagnostics, args=(str(path), data),
                                          name='hexfellow-shutdown-diagnostics')
    started = time.monotonic()
    try:
        child.start()
        child.join(3.)
        if child.is_alive():
            child.terminate()
            child.join(1.)
        complete = child.exitcode == 0 and path.is_file()
        report['diagnostics'] = dict(path=str(path), complete=complete,
                                     elapsed_s=time.monotonic()-started, exitcode=child.exitcode)
        if not complete:
            report['diagnostics']['error'] = 'diagnostic writer failed or exceeded 3 seconds'
    except Exception as error:
        report['diagnostics'] = dict(path=str(path), complete=False, error=str(error))
