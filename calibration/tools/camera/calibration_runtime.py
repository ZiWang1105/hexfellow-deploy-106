"""Calibration input adapter for the existing guarded hardware session.

This module never owns a robot SDK endpoint. session.run owns acquisition,
reference generation, publishing, measured feedback and stopping.
"""
from collections import deque
import queue
import time
import numpy as np
from hexfellow_deploy.robotics.hardware import session
from hexfellow_deploy.robotics.control.target_replay import tcp_poses
from hexfellow_deploy.robotics.hardware.x7 import HardwareSafetyError
from calibration_outcomes import InvalidCalibrationPose

# Capture acceptance only: unrelated to controller/publisher torque and motion limits.
MAX_CAPTURE_OFFSET_RAD = np.deg2rad(5.)


def capture_axes(side):
    if side not in ('left','right'):raise ValueError('Unknown capture arm')
    active=slice(4,11) if side=='left' else slice(11,18)
    other=np.ones(18,dtype=bool);other[active]=False
    return active,other


def capture_position_check(q,target,side='right'):
    q,target=np.asarray(q),np.asarray(target)
    if q.shape!=(18,) or target.shape!=(18,) or not np.all(np.isfinite(np.r_[q,target])):
        raise ValueError('Capture requires finite measured and target configurations')
    error=q-target
    active,other=capture_axes(side)
    if np.any(abs(error[other])>session.TOLERANCE[other]):
        raise ValueError(f'Base/lift/{"left" if side=="right" else "right"} arm moved outside capture tolerance')
    if np.max(abs(error[active]))>MAX_CAPTURE_OFFSET_RAD:
        raise InvalidCalibrationPose(f'{side.capitalize()}-arm offset exceeds 5 deg capture envelope')
    return dict(capture_side=side,active_offset_deg=np.degrees(error[active]).tolist(),
                maximum_active_offset_deg=float(np.degrees(np.max(abs(error[active])))),
                right_offset_deg=np.degrees(error[11:]).tolist(),
                maximum_right_offset_deg=float(np.degrees(np.max(abs(error[11:])))))


class MeasuredPoseArrival:
    """Accept a bounded, stationary measured endpoint after reference completion.

    Does not change commanded positions or gains. Other preparation/return paths
    retain SessionController.Arrival. Capture independently checks geometry and
    image/feedback pairing before using measured FK.
    """
    def __init__(self,side='right'):
        self.side=side
        self.active,self.other=capture_axes(side)
        self.previous = None
        self.rows = deque()
        self.observed_s = 0.
        self.details = {}

    def update(self, state, target, reference, reference_velocity, now, dt):
        stamps = np.array([state.received[s] for s in ('left','right','theta','base')])
        q, v = np.asarray(state.q), np.asarray(state.v)
        if (q.shape != (18,) or v.shape != (18,) or not np.all(np.isfinite(np.r_[q,v,stamps]))
                or np.any(now-stamps > .04+1e-9) or np.any(stamps-now > .005)):
            raise HardwareSafetyError('Calibration settling requires fresh, finite feedback')
        if self.previous is not None:
            if np.any(stamps < self.previous):
                self.rows.clear()
                raise HardwareSafetyError('Calibration feedback clock moved backwards')
            if np.any(stamps == self.previous):
                return False  # Repeated samples never certify stationarity.
            gap = stamps-self.previous
            if np.any(gap > .04+1e-9):
                self.rows.clear()
            self.observed_s += min(dt, float(np.min(gap)))
        self.previous = stamps
        error = q-np.asarray(target)
        ref_error = np.asarray(reference)-np.asarray(target)
        # Base/lift drift and uninvolved left arm use their existing tolerances.
        body_ok = bool(np.all(abs(error[self.other]) <= session.TOLERANCE[self.other]))
        offset_ok = bool(np.max(abs(error[self.active])) <= MAX_CAPTURE_OFFSET_RAD)
        reference_done = bool(np.max(abs(ref_error[3:])) <= .001
                              and np.max(abs(np.asarray(reference_velocity)[3:])) <= .001)
        slow = bool(np.all(abs(v) <= session.SETTLED_SPEED))
        if not (body_ok and offset_ok and reference_done and slow):
            self.rows.clear()
        else:
            at = float(np.min(stamps))
            self.rows.append((at,q.copy()))
            while len(self.rows)>1 and self.rows[1][0] <= at-1.:
                self.rows.popleft()
        duration = self.rows[-1][0]-self.rows[0][0] if self.rows else 0.
        spread_ok = bool(self.rows and np.all(np.ptp([r[1] for r in self.rows],axis=0) <= session.TOLERANCE*.5))
        stable = duration >= 1.-1e-9 and spread_ok
        reason = ('reference still settling' if not reference_done else
                  f'base/lift/{"left" if self.side=="right" else "right"} arm outside arrival tolerance' if not body_ok else
                  f'{self.side}-arm offset exceeds 5 deg capture envelope' if not offset_ok else
                  'measured joints still moving' if not slow else
                  'measured position is drifting' if not spread_ok else
                  'waiting for 1 second of stationary feedback')
        self.details = dict(status='stable_measured_pose' if stable else reason,
            capture_side=self.side,active_offset_deg=np.degrees(error[self.active]).tolist(),
            maximum_active_offset_deg=float(np.degrees(np.max(abs(error[self.active])))),
            measured_q=q.tolist(),target_q=np.asarray(target).tolist(),
            right_offset_deg=np.degrees(error[11:]).tolist(),
            maximum_right_offset_deg=float(np.degrees(np.max(abs(error[11:])))),
            capture_offset_limit_deg=float(np.degrees(MAX_CAPTURE_OFFSET_RAD)),stationary_duration_s=duration,
            observed_s=self.observed_s,reference_done=reference_done,
            capture_pose_source='measured joint feedback; never the commanded target')
        if not stable and self.observed_s > 10.:
            error = InvalidCalibrationPose if body_ok else HardwareSafetyError
            raise error(f'Calibration measured-pose settling timed out: {reason}; '
                f'maximum {self.side}-arm target offset {self.details["maximum_active_offset_deg"]:.3f} deg')
        return stable


class StationaryHold:
    """Require one second of fresh, stationary HOLD after a rejected movement."""
    def __init__(self):
        self.previous = None
        self.rows = deque()
        self.started = None

    def update(self, state, now):
        stamps = np.array([state.received[s] for s in ('left','right','theta','base')])
        q, v = np.asarray(state.q), np.asarray(state.v)
        if (q.shape != (18,) or v.shape != (18,) or not np.all(np.isfinite(np.r_[q,v,stamps]))
                or np.any(now-stamps > .04+1e-9) or np.any(stamps-now > .005)):
            raise HardwareSafetyError('Invalid-pose recovery requires fresh, finite feedback')
        if self.started is None:self.started=now
        if now-self.started > 10.:
            raise HardwareSafetyError('Could not establish stationary HOLD after rejected pose')
        if self.previous is not None:
            if np.any(stamps < self.previous):
                raise HardwareSafetyError('Invalid-pose recovery feedback clock moved backwards')
            if np.any(stamps == self.previous):return False
            if np.any(stamps-self.previous > .04+1e-9):self.rows.clear()
        self.previous=stamps
        if np.any(abs(v)>session.SETTLED_SPEED):
            self.rows.clear();return False
        at=float(min(stamps));self.rows.append((at,q.copy()))
        while len(self.rows)>1 and self.rows[1][0]<=at-1.:self.rows.popleft()
        stable=bool(at-self.rows[0][0]>=1.-1e-9 and
                    np.all(np.ptp([r[1] for r in self.rows],axis=0)<=session.TOLERANCE*.5))
        if stable:self.started=now
        return stable


def describe_arm_arrival_timeout(error, arrival, state, *, phase, pose_id):
    """Explain an existing arrival rejection; do not change its acceptance rule."""
    for side, start in (('left', 4), ('right', 11)):
        if str(error) != f'{side} did not settle in 10 seconds of new observations':
            continue
        detail = arrival.details.get(side)
        if detail is None:
            return None
        target = np.asarray(detail['target'])
        actual = np.asarray(state.q[start:start+7])
        speed = np.abs(state.v[start:start+7])
        tolerance = arrival.tolerance[start:start+7]
        speed_limit = session.SETTLED_SPEED[start:start+7]
        joints = []
        for j in range(7):
            joints.append(dict(joint=f'{side} J{j+1}',
                target_deg=float(np.degrees(target[j])), actual_deg=float(np.degrees(actual[j])),
                error_deg=float(np.degrees(actual[j]-target[j])),
                tolerance_deg=float(np.degrees(tolerance[j])),
                speed_deg_s=float(np.degrees(speed[j])),
                speed_limit_deg_s=float(np.degrees(speed_limit[j])),
                position_failed=bool(abs(actual[j]-target[j]) > tolerance[j]),
                speed_failed=bool(speed[j] > speed_limit[j])))
        failures = []
        for joint in joints:
            if joint['position_failed']:
                failures.append(f"{joint['joint']} position error {joint['error_deg']:+.3f} deg "
                    f"(allowed +/-{joint['tolerance_deg']:.3f}; target {joint['target_deg']:.3f}, "
                    f"measured {joint['actual_deg']:.3f})")
            if joint['speed_failed']:
                failures.append(f"{joint['joint']} speed {joint['speed_deg_s']:.3f} deg/s "
                    f"(maximum {joint['speed_limit_deg_s']:.3f})")
        context = f'Calibration pose {pose_id}' if phase == 'calibration' and pose_id > 0 else phase
        stationary = bool(np.all(speed <= speed_limit))
        message = (f"{context}: {side} arm arrival timed out after 10 seconds. "
                   + '; '.join(failures) + '. '
                   + ('The arm is stationary, but outside the position tolerance. ' if stationary else '')
                   + 'Calibration stopped; no capture accepted for this pose.')
        return dict(kind='arm_arrival_timeout', original_error=str(error), phase=phase,
                    pose_id=pose_id, side=side, stationary=stationary, joints=joints,
                    message=message)
    return None


def stationary_pair(rows, image_time, *, uncertainty=.0):
    """Pair a host-monotonic frame inside a fresh stationary feedback window.

This is host-time stationary pairing, not hardware-trigger synchronization.
"""
    if not np.isfinite(image_time) or not 0 <= uncertainty <= .005:
        raise ValueError('Invalid camera clock mapping')
    selected = [r for r in rows if image_time-.55 <= r['at'] <= image_time+.12]
    if len(selected) < 15 or selected[0]['at'] > image_time-.45 or selected[-1]['at'] < image_time+.05:
        raise ValueError('Need stationary feedback bracketing the frame for 0.5 seconds')
    if max(np.diff([r['at'] for r in selected])) > .08:
        raise ValueError('Feedback gap in capture window')
    q, v = np.array([r['q'] for r in selected]), np.array([r['v'] for r in selected])
    if np.any(np.abs(v) > session.SETTLED_SPEED) or np.any(np.ptp(q, axis=0) > session.TOLERANCE*.5):
        raise ValueError('Robot moved during capture window')
    if any(r['age'] > .08 or r['phase'] not in ('hold', 'ready') for r in selected):
        raise ValueError('Feedback stale or motion active during capture')
    closest = min(selected, key=lambda r: abs(r['at']-image_time))
    skew = abs(closest['at']-image_time)
    if skew + uncertainty > .04:
        raise ValueError('Image/feedback host-time skew exceeds 40 ms')
    return dict(closest, image_joint_skew_s=skew, clock_mapping_uncertainty_s=uncertainty,
                stationary_window_s=[selected[0]['at'], selected[-1]['at']],
                synchronization='stationary host-time pairing; no hardware exposure trigger')


class CalibrationController(session.SessionController):
    def __init__(self, *args, source, **kwargs):
        super().__init__(*args, **kwargs)
        self.source = source
        self.todo = deque()
        self.pose_id = 0
        self.full_default_active = False
        self.invalid_hold = None
        source.bounds = [self.core.model.lowerPositionLimit.tolist(),self.core.model.upperPositionLimit.tolist()]

    def command(self, name, payload, state, now):
        try:return self._command(name,payload,state,now)
        except Exception:
            self.source.pending_command=None
            self.source.interruption_count+=1
            raise

    def _command(self, name, payload, state, now):
        if name in ('stop', 'quit'):
            self.todo.clear()
            self.full_default_active=False
            self.source.paused = name == 'stop'
            return super().command(name, payload, state, now)
        if name == 'default' and payload and payload.get('full_default'):
            if np.max(abs(state.q[3:]-np.asarray(payload['origin'])[3:]))>.005:
                raise ValueError('Robot moved since default path check; retry')
            self.todo.clear();self.full_default_active=True
            self.source.pending_invalid=None;self.invalid_hold=None
            self.source.prepared=False;self.source.target_pose=-2
            self.source.message='Moving lift and both arms to default (lift 300 mm).'
            return super().command('default',None,state,now)
        if name == 'default':
            if self.source.prepared:raise ValueError('Already prepared; use Next or Resume')
            # "Prepare" acquires an initial HOLD; never moves to the stack's D pose.
            self.hold_q = self.core._last_safe_positions.copy()
            self.enter('hold', now)
            self.source.prepared = True
            self.source.paused = False
            self.source.message = 'Ready at current position.'
            return
        if name != 'calibration-path':
            raise ValueError('Calibration accepts prepare/next/pause/resume/capture/solve/quit only')
        if not self.idle or np.any(abs(state.v) > session.SETTLED_SPEED):
            raise ValueError('Wait for measured stationary HOLD')
        if np.max(np.abs(state.q[3:]-payload['origin'][3:])) > .005:
            raise ValueError('Robot moved while checking path; request Next/Resume again')
        reference_origin = payload.get('reference_origin')
        if reference_origin is not None:
            if np.max(abs(self.core._last_safe_positions[3:]-np.asarray(reference_origin)[3:])) > .005:
                raise ValueError('Reference changed since its path check; recheck needed')
        elif np.max(np.abs(self.core._last_safe_positions[3:]-payload['origin'][3:])) > .02:
            raise ValueError('Reference differs from checked origin; recheck needed')
        for target in payload['targets']:self.core._validate_configuration_vector(np.asarray(target))
        self.todo = deque(payload['targets'])
        self.source.pending_invalid=None;self.invalid_hold=None
        self.pose_id = payload['pose_id']
        self._segment(state, now)
        self.source.target_pose = self.pose_id
        self.source.paused = False
        self.source.pose_settling = None

    def _segment(self, state, now):
        target = np.asarray(self.todo.popleft())
        self.posture(target, state, now, 'calibration')
        self.measured_arrival = MeasuredPoseArrival(self.source.capture_side)
        # Reuse preparation acceleration/jerk bounds and add the route speed limit.
        self.segment.duration = max(self.segment.duration,
                                    1.875*np.max(abs(self.segment.delta[4:]))/.15)
        self.source.message = ('Returning arms to default; lift stays fixed.' if self.pose_id==-1
                               else f'Moving to calibration pose {self.pose_id}.')

    @property
    def control_phase(self):
        return 'preparation' if self.phase == 'calibration' else super().control_phase

    def sample(self, state, now, quest=None):
        try:
            return self._sample(state, now, quest)
        except HardwareSafetyError as error:
            _,other=capture_axes(self.source.capture_side)
            if (self.phase=='calibration' and self.pose_id>0 and
                    str(error)==f'{self.source.capture_side} did not settle in 10 seconds of new observations' and
                    np.all(abs(state.q[other]-self.segment.target[other])<=session.TOLERANCE[other])):
                return self._reject_pose(error,state,now,quest)
            detail = describe_arm_arrival_timeout(error, self.arrival, state,
                phase=self.phase, pose_id=self.source.target_pose)
            if detail is None:
                raise
            self.source.failure_detail = detail
            self.source.message = detail['message']
            self.source.paused = True
            self.source.interruption_count += 1
            print('STOP REASON: ' + detail['message'], flush=True)
            # Keep the existing guarded shutdown. Never certify this as arrival.
            raise HardwareSafetyError(detail['message']) from error

    def _reject_pose(self, error, state, now, quest=None):
        self.todo.clear()
        self.source.pending_invalid=dict(pose_id=self.pose_id,reason=str(error),
            measured_q=state.q.tolist(),settling=self.source.pose_settling,
            ready=False)
        self.source.arrived_pose=0
        self.source.pending_command=None
        self.source.message=f'Pose {self.pose_id}: invalid — {error}. Stopping before skipping.'
        self.invalid_hold=StationaryHold()
        # Preserve the normal deceleration and support reference. This internal
        # stop does not count as an operator pause or a successful arrival.
        super().command('stop',None,state,now)
        return super().sample(state,now,quest)

    def _sample(self, state, now, quest=None):
        if self.phase != 'calibration':
            result=super().sample(state, now, quest)
            invalid=self.source.pending_invalid
            if invalid is not None and self.phase=='hold':
                if getattr(self,'invalid_hold_record',None) is not invalid:
                    self.invalid_hold_record=invalid;self.invalid_hold=StationaryHold()
                invalid['ready']=self.invalid_hold.update(state,now)
            if self.full_default_active and self.phase=='hold':
                self.full_default_active=False
                # Only declare completion if the configured target was reached.
                target=session.pose_vector(self.default_pose,state.q[:3])
                if np.all(abs(state.q[3:]-target[3:])<=session.TOLERANCE[3:]):
                    self.source.prepared=True;self.source.paused=False
                    self.source.arrived_pose=self.source.target_pose=0
                    self.source.bounds=[self.core.model.lowerPositionLimit.tolist(),self.core.model.upperPositionLimit.tolist()]
                    self.source.message=f'Default pose reached (300 mm target; measured lift {state.q[3]*1000:.1f} mm). Ready to start calibration.'
                self.source.pending_command=None
            return result
        posture, velocity = self.segment.sample(now-self.started)
        final_capture_pose = self.pose_id > 0 and not self.todo
        if final_capture_pose:
            arrived = False
            if now-self.started >= self.segment.duration:
                try:
                    arrived = self.measured_arrival.update(state,self.segment.target,
                        self.core._last_safe_positions,self.core._last_velocity,now,self.core.config.control_dt)
                except InvalidCalibrationPose as error:
                    self.source.pose_settling=self.measured_arrival.details
                    return self._reject_pose(error,state,now,quest)
                except HardwareSafetyError as error:
                    self.source.failure_detail = dict(kind='measured_pose_settling',pose_id=self.pose_id,
                        message=str(error),settling=self.measured_arrival.details)
                    self.source.message = str(error)
                    self.source.paused = True
                    self.source.interruption_count += 1
                    print('STOP REASON: '+str(error),flush=True)
                    raise
                self.source.pose_settling = self.measured_arrival.details
                self.source.message = (f'Pose {self.pose_id}: {self.measured_arrival.details["status"]}; '
                    f'maximum {self.source.capture_side}-arm target offset {self.measured_arrival.details["maximum_active_offset_deg"]:.2f} deg.')
        else:
            arrived = self.arrival.update(state, self.segment.target, self.core.config.control_dt,
                                          timeout_active=now-self.started > self.segment.duration)
        if now-self.started >= self.segment.duration and arrived:
            if self.todo:
                self._segment(state, now+self.core.config.control_dt)
            else:
                self.hold_q = self.core._last_safe_positions.copy()
                self.enter('hold', now)
                self.source.arrived_pose = self.pose_id
                self.source.message = ('Arms at default; lift unchanged.' if self.pose_id==-1
                                       else f'Pose {self.pose_id}: measured arm is stationary; capture uses measured angles.')
        return tcp_poses(self.core, posture), posture, velocity


class CalibrationSource:
    def __init__(self, *, live=False, capture_side='right'):
        capture_axes(capture_side)
        self.capture_side=capture_side
        self.live = live
        self.commands = queue.Queue(maxsize=1)
        self.stop = False
        self.quit = False
        self.heartbeat = time.monotonic()
        self.message = 'Read-only. Prepare acquires stationary HOLD; Next moves.'
        self.prepared = self.paused = False
        self.target_pose = self.arrived_pose = 0
        self.rows = deque(maxlen=500)
        self.last_display = None
        self.simulation_plant = None
        self.binding = None
        self.bounds = None
        self.finished = False
        self.capture_busy = False
        self.interruption_count = 0
        self._stale_latched = False
        self.pending_command = None
        self.failure_detail = None
        self.pose_settling = None
        self.pending_invalid = None

    def validate_session(self, args, mapping, profile):
        if args.mode not in ('sim', 'live') or args.commands:
            raise HardwareSafetyError('Calibration requires an interactive sim/live session')
        if mapping.feedback_policy != 'strict' or mapping.latency_policy != 'hold-resume':
            raise HardwareSafetyError('Calibration requires strict feedback with guarded hold/resume')
        if self.live:
            from hexfellow_deploy.robotics.hardware.verification import software_binding
            if self.binding is None or software_binding(mapping) != self.binding:
                raise HardwareSafetyError('Deployment changed since preflight')

    def check_viewer_ready(self):
        if self.quit or self.stop or time.monotonic()-self.heartbeat > 2:
            raise HardwareSafetyError('Operator interface not ready')

    def before_acquire(self, args, mapping, profile):
        self.validate_session(args, mapping, profile)
        self.check_viewer_ready()

    def controller_factory(self, *args, **kwargs):
        return CalibrationController(*args, source=self, **kwargs)

    def poll(self, state, engine, now, paused):
        at = time.monotonic()
        phase = engine.phase if engine else 'readonly'
        # received is the session simulation clock in sim and host monotonic in live.
        age = max(0., now-min(state.received.values()))
        row = dict(at=at, q=state.q.tolist(), v=state.v.tolist(), age=age,
                   received=dict(state.received), phase=phase)
        if engine is not None and hasattr(engine,'core'):
            row['reference_q'] = engine.core._last_safe_positions.tolist()
        self.rows.append(row)
        self.last_display = row
        if ((self.pending_command=='default' and self.prepared)
                or (self.pending_command=='calibration-path' and phase=='calibration')):
            self.pending_command=None
        if self.quit:
            return ('quit', None) if engine is None or not engine.quit_requested else None
        stale = at-self.heartbeat > 2
        if stale and not self._stale_latched:self.interruption_count+=1
        self._stale_latched=stale
        if self.stop or stale or paused:
            self.pending_command=None
            self.stop = False
            if engine and engine.phase in ('calibration','stopping'):self.paused = True
            while not self.commands.empty():
                try:self.commands.get_nowait()
                except queue.Empty:break
            if engine and engine.phase not in ('stopping', 'hold', 'ready'):
                self.message = 'Stopping: operator pause, stale browser, or feedback delay.'
                return ('stop', None)
            return None
        try:return self.commands.get_nowait()
        except queue.Empty:return None

    def suspend(self, engine):
        self.interruption_count += 1
        self.paused = True
        if engine is not None:engine.todo.clear()
        self.message = 'Feedback delay: held. Explicit Resume with a new path check required.'

    def recovered(self, engine, state, now):
        self.paused = True
        engine.todo.clear()
        engine.hold_q = engine.core._last_safe_positions.copy()
        engine.enter('hold', now)

    def diagnostics(self, engine, command, targets):
        return dict(calibration=True, pose=self.target_pose, message=self.message,
                    input_age_ms=1000*(time.monotonic()-self.heartbeat),
                    active=engine.phase=='calibration',
                    position_errors_mm=[1000*command.diagnostics.left_position_error,1000*command.diagnostics.right_position_error])

    def summary(self):
        return dict(calibration=True, arrived_pose=self.arrived_pose, operator_started=True,
                    failure_detail=self.failure_detail,pose_settling=self.pose_settling,
                    pending_invalid=self.pending_invalid)

    def check_command(self, command, *, phase):
        if phase == 'calibration' and command.diagnostics.status == 'large/unreachable target error':
            raise HardwareSafetyError('Calibration target could not be followed')
