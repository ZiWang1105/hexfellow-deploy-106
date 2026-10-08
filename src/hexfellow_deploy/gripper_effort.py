"""Position-based motor-effort regulation, not a hardware force ceiling.

The input is measured motor position/velocity/effort, never contact truth. Motor
torque is not fingertip force. Physical contact-force calibration is separate
from this feedback law and is not established by its software tests.
"""
from dataclasses import dataclass, asdict
from collections import deque
import math
from statistics import median

from .gripper_contract import validate_gripper


@dataclass(frozen=True)
class EffortGripperConfig:
    closed_rad: float
    open_rad: float
    # Optional explicit software trip, not a firmware torque/force limit.
    max_torque_nm: float | None = None
    hold_torque_nm: float = .04
    contact_torque_nm: float = .025
    effort_deadband_nm: float = .005
    max_speed_rad_s: float = .2
    regulation_speed_rad_s: float = .2
    closing_acceleration_rad_s2: float | None = .2
    max_lead_rad: float = .01
    position_ramp_approach: bool = False
    effort_gain_rad_per_nm_s: float = .5
    contact_speed_rad_s: float = .05
    contact_duration_s: float = .04
    feedback_timeout_s: float = .05
    max_step_s: float = .02
    endpoint_tolerance_rad: float = .002
    feedback_tolerance_rad: float | None = None
    stall_timeout_s: float = 1.
    opening_effort_nm: float | None = None
    progress_tolerance_rad: float | None = None
    tracking_window_s: float = .06
    tracking_position_span_rad: float = .0002
    tracking_min_lead_rad: float = .0005

    def __post_init__(self):
        for name, value in asdict(self).items():
            if name == 'position_ramp_approach':
                if type(value) is not bool:
                    raise ValueError('position_ramp_approach must be boolean')
                continue
            if name in ('max_torque_nm', 'opening_effort_nm', 'progress_tolerance_rad',
                        'closing_acceleration_rad_s2', 'feedback_tolerance_rad') and value is None:
                continue
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                raise ValueError(f'Invalid gripper setting: {name}')
            if name not in ('closed_rad', 'open_rad') and value <= 0:
                raise ValueError(f'Gripper setting must be positive: {name}')
        if self.closed_rad == self.open_rad:
            raise ValueError('Distinct gripper endpoints required')
        if not self.contact_torque_nm < self.hold_torque_nm:
            raise ValueError('Require contact < holding motor torque')
        if self.max_torque_nm is not None and self.hold_torque_nm + self.effort_deadband_nm >= self.max_torque_nm:
            raise ValueError('Holding deadband must remain below maximum motor torque')
        if self.max_lead_rad >= abs(self.closed_rad - self.open_rad):
            raise ValueError('Reference lead must be smaller than motor stroke')
        if self.tracking_window_s >= self.stall_timeout_s:
            raise ValueError('Tracking window must fit within the no-progress timeout')
        if self.tracking_min_lead_rad >= self.max_lead_rad:
            raise ValueError('Tracking lead threshold must be smaller than the reference bound')
        if (self.max_torque_nm is not None and self.opening_effort_nm is not None
                and self.opening_effort_nm + self.effort_deadband_nm >= self.max_torque_nm):
            raise ValueError('Opening effort band must remain below maximum motor torque')


@dataclass(frozen=True)
class GripperFeedback:
    position_rad: float
    velocity_rad_s: float
    effort_nm: float
    received_s: float
    healthy: bool = True


@dataclass(frozen=True)
class GripperOutput:
    state: str
    position_rad: float | None
    brake: bool
    closing_effort_nm: float | None
    fault: str | None = None
    effort_regulating: bool = False
    regulation_unsettled_s: float = 0.
    reference_velocity_rad_s: float = 0.
    effort_target_nm: float | None = None
    position_error_rad: float | None = None
    effort_error_nm: float | None = None
    requested_velocity_rad_s: float = 0.
    reference_reconciling: bool = False
    reference_limited: bool = False
    filtered_closing_effort_nm: float | None = None
    tracking_blocked: bool = False
    tracking_window_elapsed_s: float = 0.
    tracking_position_span_rad: float | None = None
    tracking_min_lead_rad: float | None = None


class EffortGripperController:
    """One stateful owner; duplicate feedback never integrates twice.

    Positive effort is assumed to produce increasing motor position. Closure
    direction comes from the endpoints. That sign assumption is unverified on
    the physical GP80 and must be checked on an instrumented fixture.
    Faults latch; a later command cannot automatically resume motion.
    """
    def __init__(self, config: EffortGripperConfig):
        self.config = config
        self.direction = 1. if config.closed_rad > config.open_rad else -1.
        self.reference = None
        self.state = 'idle'
        self.fault = None
        self.previous = None
        self.previous_now = None
        self.request = None
        self.contact_since = None
        self.stall_since = None
        self.progress_position = None
        self.stall_diagnostic = None
        self.effort_regulating = False
        self.regulation_unsettled_since = None
        self.reference_velocity = 0.
        self.reference_reconciling = self.reference_limited = False
        self.requested_velocity = 0.
        self.last_output = None
        self.filtered_effort = None
        self.effort_samples = deque(maxlen=3)
        self.closing_progress_position = self.closing_progress_since = None
        self.tracking_samples = deque()
        self.tracking_blocked = False
        self.tracking_elapsed = 0.
        self.tracking_span = self.tracking_lead = None

    def _tracking_window(self, q, stamp):
        """Detect stationary feedback despite a sustained closing-side target.

        Only distinct, fresh reports enter here. Use position range rather than
        endpoint displacement so oscillation is not mistaken for stationarity.
        The reference is the previous target, not one we are about to generate.
        """
        c = self.config
        if self.tracking_samples and stamp-self.tracking_samples[-1][0] > min(c.feedback_timeout_s, c.tracking_window_s/2):
            self.tracking_samples.clear()
        self.tracking_samples.append((stamp, q, (self.reference-q)*self.direction))
        while len(self.tracking_samples) > 2 and self.tracking_samples[1][0] <= stamp-c.tracking_window_s:
            self.tracking_samples.popleft()
        self.tracking_elapsed = stamp-self.tracking_samples[0][0]
        self.tracking_span = max(r[1] for r in self.tracking_samples)-min(r[1] for r in self.tracking_samples)
        self.tracking_lead = min(r[2] for r in self.tracking_samples)
        self.tracking_blocked = (len(self.tracking_samples) >= 3 and
            self.tracking_elapsed+1e-12 >= c.tracking_window_s and
            self.tracking_span <= c.tracking_position_span_rad and
            self.tracking_lead >= c.tracking_min_lead_rad)

    def _output(self, effort, q, now):
        self.last_output = GripperOutput(
            self.state, self.reference, False, self.previous.effort_nm*self.direction,
            effort_regulating=self.effort_regulating,
            regulation_unsettled_s=(0. if self.regulation_unsettled_since is None else
                                    max(0., now-self.regulation_unsettled_since)),
            reference_velocity_rad_s=self.reference_velocity,
            effort_target_nm=self.config.hold_torque_nm if self.request == 0 else self.config.opening_effort_nm,
            position_error_rad=self.reference - q,
            effort_error_nm=self.config.hold_torque_nm-effort if self.request == 0 else None,
            requested_velocity_rad_s=self.requested_velocity,
            reference_reconciling=self.reference_reconciling,
            reference_limited=self.reference_limited,
            filtered_closing_effort_nm=effort,
            tracking_blocked=self.tracking_blocked,
            tracking_window_elapsed_s=self.tracking_elapsed,
            tracking_position_span_rad=self.tracking_span,
            tracking_min_lead_rad=self.tracking_lead)
        return self.last_output

    def _regulate(self, q, effort, stamp, dt):
        """Projected effort integrator; holding is a diagnostic, not another law.

        Integrate only the issued reference (anti-windup). The reference envelope
        limits position error but is not a force cap. A large approach error is
        reconciled toward this envelope at the existing SDK slew limit, only
        toward opening; the effort law then uses the lower regulation rate.
        """
        c = self.config
        lo, hi = sorted((c.open_rad, c.closed_rad))
        error = c.hold_torque_nm-effort
        velocity = (0. if abs(error) <= c.effort_deadband_nm else
                    self.direction*c.effort_gain_rad_per_nm_s*error)
        speed = min(c.regulation_speed_rad_s, c.max_speed_rad_s)
        velocity = max(-speed, min(speed, velocity))
        # A raw high sample may pause tightening before the median catches up.
        # It cannot trigger a different controller or exit regulation.
        if self.previous.effort_nm*self.direction > c.hold_torque_nm+c.effort_deadband_nm:
            velocity = self.direction*min(0., velocity*self.direction)
        self.requested_velocity = velocity
        candidate = min(hi, max(lo, self.reference+velocity*dt))
        lower, upper = max(lo, q-c.max_lead_rad), min(hi, q+c.max_lead_rad)
        projected = min(upper, max(lower, candidate))
        # If external motion would require tightening against the effort law,
        # do not silently drag the reference along with measured position.
        if (projected-self.reference)*self.direction > max(0., velocity*self.direction*dt)+1e-12:
            return self._fail('Measured position moved outside the bounded reference window')
        self.reference_reconciling = (self.reference-q)*self.direction > c.max_lead_rad+1e-12
        limit = (c.max_speed_rad_s if self.reference_reconciling else speed)*dt
        delta = max(-limit, min(limit, projected-self.reference))
        self.reference_limited = abs(projected-candidate) > 1e-12 or abs(delta-(projected-self.reference)) > 1e-12
        self.reference += delta
        self.reference_velocity = delta/dt if dt else 0.
        in_band = (not self.reference_reconciling and
                   abs(error) <= c.effort_deadband_nm and
                   abs(self.previous.velocity_rad_s) <= c.contact_speed_rad_s)
        if in_band:
            if self.contact_since is None:
                self.contact_since = stamp
        else:
            self.contact_since = None
        self.state = ('holding' if self.contact_since is not None and
                      stamp-self.contact_since >= c.contact_duration_s else 'regulating')
        # Elapsed unsettled time is telemetry only, never a motion deadline.
        # The trial supervisor owns total duration and operator cancellation.
        if self.state == 'holding':
            self.regulation_unsettled_since = None
        elif self.regulation_unsettled_since is None:
            self.regulation_unsettled_since = stamp
        return self._output(effort, q, stamp)

    def _close(self, q, effort, stamp, dt):
        c = self.config
        self._tracking_window(q, stamp)
        if (not self.effort_regulating and self.tracking_blocked and
                effort >= c.contact_torque_nm and
                abs(self.previous.velocity_rad_s) <= c.contact_speed_rad_s):
            self.effort_regulating = True
            self.contact_since = None
            self.regulation_unsettled_since = stamp
        if self.effort_regulating:
            return self._regulate(q, effort, stamp, dt)
        self.state = 'closing'
        if abs(c.closed_rad-q) <= c.endpoint_tolerance_rad and abs(self.previous.velocity_rad_s) <= c.contact_speed_rad_s:
            self.state = 'closed'
        progress = (q-self.closing_progress_position)*self.direction
        tolerance = c.progress_tolerance_rad or c.endpoint_tolerance_rad
        if progress >= tolerance:
            self.closing_progress_position, self.closing_progress_since = q, stamp
        elif self.state != 'closed' and stamp-self.closing_progress_since >= c.stall_timeout_s:
            self.stall_diagnostic = dict(action='closing', window_s=stamp-self.closing_progress_since,
                measured_progress_rad=progress, required_progress_rad=tolerance,
                remaining_motor_rad=abs(c.closed_rad-q), reference_error_rad=self.reference-q,
                measured_effort_nm=self.previous.effort_nm, filtered_closing_effort_nm=effort,
                tracking_blocked=self.tracking_blocked)
            return self._fail('No measured gripper progress during approach; '
                              f'net progress {progress:.6f} rad; required {tolerance:.6f} rad')
        speed = c.max_speed_rad_s
        if c.closing_acceleration_rad_s2 is not None:
            speed = min(speed, max(0., self.reference_velocity*self.direction)+c.closing_acceleration_rad_s2*dt)
        loaded_lag = ((self.reference-q)*self.direction >= c.tracking_min_lead_rad and
                      self.previous.effort_nm*self.direction > c.hold_torque_nm+c.effort_deadband_nm)
        if self.tracking_blocked or loaded_lag:
            speed = 0.
        delta = self.direction*min(abs(c.closed_rad-self.reference), speed*dt)
        candidate = self.reference+delta
        if not c.position_ramp_approach:
            lo, hi = sorted((c.open_rad, c.closed_rad))
            lower = max(lo, self.reference-c.max_speed_rad_s*dt, q-c.max_lead_rad)
            upper = min(hi, self.reference+c.max_speed_rad_s*dt, q+c.max_lead_rad)
            if self.direction > 0:
                upper = min(upper, candidate)
            else:
                lower = max(lower, candidate)
            if lower > upper+1e-12:
                return self._fail('Measured position moved outside the bounded reference window')
            candidate = min(upper, max(lower, candidate))
        self.reference_velocity = (candidate-self.reference)/dt if dt else 0.
        self.requested_velocity = self.direction*speed
        self.reference = candidate
        return self._output(effort, q, stamp)

    def stop(self, reason):
        self.state, self.fault = 'fault', self.fault or reason
        return GripperOutput(self.state, None, True, None, self.fault)

    _fail = stop

    def step(self, request, feedback: GripperFeedback, now_s):
        if self.fault:
            return self._fail(self.fault)
        try:
            request = validate_gripper(request, allow_hold=True)
        except ValueError as error:
            return self._fail(str(error))
        c = self.config
        values = (feedback.position_rad, feedback.velocity_rad_s, feedback.effort_nm,
                  feedback.received_s, now_s)
        if any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) for v in values):
            return self._fail('Non-finite or invalid motor feedback/time')
        if feedback.healthy is not True:
            return self._fail('Motor health unavailable or abnormal')
        if not 0 <= now_s - feedback.received_s <= c.feedback_timeout_s:
            return self._fail('Motor feedback expired or timestamp is in the future')
        if self.previous_now is not None and now_s < self.previous_now:
            return self._fail('Controller clock moved backwards')
        if self.previous_now is not None and now_s - self.previous_now > c.feedback_timeout_s:
            return self._fail('Gripper control update expired')
        self.previous_now = now_s
        lo, hi = sorted((c.closed_rad, c.open_rad))
        feedback_tolerance = c.feedback_tolerance_rad or c.endpoint_tolerance_rad
        if not lo-feedback_tolerance <= feedback.position_rad <= hi+feedback_tolerance:
            return self._fail('Motor position outside configured travel')
        effort = feedback.effort_nm * self.direction
        # An explicitly selected trip remains immediate, before smoothing.
        # There is no inferred motor-force rating or arbitrary default cutoff.
        if c.max_torque_nm is not None and abs(feedback.effort_nm) >= c.max_torque_nm:
            return self._fail('Measured motor effort reached provisional trip threshold: '
                              f'{feedback.effort_nm:+.6f} Nm; limit {c.max_torque_nm:.6f} Nm')
        if self.previous is not None:
            if feedback.received_s < self.previous.received_s:
                return self._fail('Motor feedback timestamp moved backwards')
            if feedback.received_s == self.previous.received_s:
                if feedback != self.previous:
                    return self._fail('Repeated timestamp changed motor feedback')
                # Cancellation is immediate even without a new sample. A new
                # direction otherwise waits for genuinely new measured feedback.
                if request is None:
                    self.request, self.state = None, 'idle'
                    self.last_output = GripperOutput('idle', None, True, effort)
                    return self.last_output
                if request != self.request:
                    return GripperOutput('waiting_for_feedback', None, True, effort)
                return self.last_output or GripperOutput('idle', None, True, effort)
            dt = min(c.max_step_s, feedback.received_s - self.previous.received_s)
        else:
            dt = 0.
        self.previous = feedback
        # Filter only the regulation/contact signal. The raw absolute trip
        # above remains instantaneous, including on duplicate feedback.
        if not self.effort_samples:
            self.effort_samples.extend([effort]*3)
        else:
            self.effort_samples.append(effort)
        self.filtered_effort = median(self.effort_samples)
        effort = self.filtered_effort if request == 0 else effort
        q = min(hi, max(lo, feedback.position_rad))
        if request != self.request or self.reference is None:
            self.reference = q
            self.contact_since = self.stall_since = None
            self.progress_position = q
            self.effort_regulating = False
            self.regulation_unsettled_since = None
            self.reference_velocity = 0.
            self.reference_reconciling = self.reference_limited = False
            self.requested_velocity = 0.
            self.closing_progress_position, self.closing_progress_since = q, feedback.received_s
            self.state = 'idle' if request is None else ('opening' if request else 'closing')
            self.request = request
            self.tracking_samples.clear()
            self.tracking_blocked = False
            self.tracking_elapsed = 0.
            self.tracking_span = self.tracking_lead = None
        if request is None:
            self.reference = q
            self.last_output = GripperOutput('idle', None, True, effort)
            return self.last_output
        if not request:
            return self._close(q, effort, feedback.received_s, dt)
        # Opening diagnostics retain their prior position/effort behavior.
        target = c.open_rad
        delta = target-self.reference
        tolerance = c.progress_tolerance_rad or c.endpoint_tolerance_rad
        progress = (self.progress_position-q)*self.direction
        if abs(target-q) <= c.endpoint_tolerance_rad and abs(feedback.velocity_rad_s) <= c.contact_speed_rad_s:
            self.state = 'open'
        elif progress >= tolerance:
            self.progress_position, self.stall_since = q, feedback.received_s
        elif self.stall_since is None:
            self.stall_since = feedback.received_s
        elif feedback.received_s-self.stall_since >= c.stall_timeout_s:
            self.stall_diagnostic = dict(action='opening', window_s=feedback.received_s-self.stall_since,
                measured_progress_rad=progress, required_progress_rad=tolerance,
                remaining_motor_rad=abs(target-q), reference_error_rad=self.reference-q,
                max_lead_rad=c.max_lead_rad, measured_effort_nm=feedback.effort_nm)
            return self._fail(f'No measured gripper progress: {progress:.6f} rad '
                              f'over {feedback.received_s-self.stall_since:.3f} s; '
                              f'requires {tolerance:.6f} rad; remaining {abs(target-q):.6f} rad; '
                              f'reference error {self.reference-q:+.6f} rad '
                              f'(lead limit {c.max_lead_rad:.6f} rad)')
        if c.opening_effort_nm is not None and self.state != 'open':
            error = c.opening_effort_nm+effort
            delta = (0. if abs(error) <= c.effort_deadband_nm else
                     -self.direction*c.effort_gain_rad_per_nm_s*error*dt)
            self.state = 'opening_relief' if error < -c.effort_deadband_nm else 'opening'
        limit = c.max_speed_rad_s*dt
        delta = max(-limit, min(limit, delta))
        lower = max(lo, self.reference-limit, q-c.max_lead_rad)
        upper = min(hi, self.reference+limit, q+c.max_lead_rad)
        if c.opening_effort_nm is not None:
            # Opening relief cannot cross measured position toward closure.
            if self.direction > 0:
                upper = min(upper, q)
            else:
                lower = max(lower, q)
        if lower > upper+1e-12:
            return self._fail('Measured position moved outside the bounded reference window')
        previous = self.reference
        self.reference = min(upper, max(lower, previous+delta))
        self.reference_velocity = (self.reference-previous)/dt if dt else 0.
        return self._output(effort, q, feedback.received_s)


def closing_effort_settings(closed, opened, *, effort_trip_nm=None):
    """Shared operator-selected motor-effort profile for either GP80."""
    return EffortGripperConfig(closed_rad=closed, open_rad=opened,
        hold_torque_nm=.1, max_speed_rad_s=min(abs(closed-opened), .02*250),
        closing_acceleration_rad_s2=None, max_step_s=1/250,
        regulation_speed_rad_s=.05, max_lead_rad=.005, position_ramp_approach=True,
        contact_duration_s=.15, contact_speed_rad_s=.01,
        progress_tolerance_rad=.0002, max_torque_nm=effort_trip_nm)

