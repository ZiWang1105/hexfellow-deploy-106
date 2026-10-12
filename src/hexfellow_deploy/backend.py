"""One DA actuator owner and IPC protocol for both physical and simulated devices."""
from dataclasses import asdict
from collections import deque
import json
import math
from pathlib import Path
import time

from deploy_anything.components.interface.base_interface import DABaseInterface
from .actuation import StreamContract, encode_mit, deployment
from .reference import Receipt
from .workspace import activate


def diagnostic_value(value):
    """SDK telemetry arrays remain numeric in the offline diagnostic artifact."""
    import numpy as np
    if isinstance(value, (np.ndarray, np.generic)):
        return value.tolist()
    raise TypeError(f'Unsupported diagnostic value: {type(value).__name__}')


class ControlCancelled(RuntimeError):
    """Explicit component shutdown interrupted a wait; no command was published."""


class BackendProxy:
    def __init__(self, frames, feedback, backend_status, timeout=30., receipts=None):
        self.frames, self.feedback, self.status = frames, feedback, backend_status
        self.receipts = receipts
        self.selected = None
        self.created_s = None
        until = time.monotonic()+timeout
        while time.monotonic() < until:
            status = self.status.get()
            if status and status.get('fault'):
                raise RuntimeError(status['fault'])
            if status and status.get('metadata'):
                self.metadata = status['metadata']
                return
            time.sleep(.01)
        raise RuntimeError('Backend metadata unavailable')

    def read(self, *, timeout_s=.001):
        started = time.monotonic()
        status = self._actuator_status()
        if status and status.get('fault'):
            raise RuntimeError(status['fault'])
        row = self.feedback.get(timeout=timeout_s)
        self.feedback_read = dict(started_s=started, completed_s=time.monotonic(),
                                  acquired=row is not None)
        if row is not None:
            self.selected = row
        if self.selected is None:
            raise RuntimeError('Backend feedback unavailable')
        row = self.selected
        self.feedback_read.update(publication_sequence=row.get('publication_sequence'),
                                  published_s=row.get('published_s'), received=row['received'])
        return row['measured']

    def _actuator_status(self):
        topic = getattr(self, 'receipts', None)
        if topic is None:
            return self.status.get()
        state = topic.get()
        if state is None or state.get('fault'):
            detail = self.status.get(timeout=.001)
            if detail and detail.get('fault'):
                return detail
            if state and state.get('fault'):
                return dict(state, fault='Actuator fault; detailed diagnostic publication pending')
        return state

    def read_for_control(self, *, timeout_s, min_validity_s, stopped=lambda: False, on_wait=None):
        """Refresh before entering stateful IK; never renew sensor/command clocks."""
        from policy_sim.control.feedback_policy import (
            validate_received, received_deadline, FeedbackSkewError)
        if (not all(math.isfinite(x) for x in (timeout_s, min_validity_s))
                or not 0 < min_validity_s < timeout_s):
            raise ValueError('Feedback wait requires a finite positive solve margin below timeout')
        started = time.monotonic()
        deadline = min(started+timeout_s, getattr(self, 'command_deadline', float('inf')))
        retries = 0
        while True:
            if stopped():
                raise ControlCancelled('Feedback refresh cancelled by stop request')
            remaining = deadline-time.monotonic()
            if remaining <= min_validity_s:
                raise RuntimeError('Fresh feedback unavailable before command health deadline; IK not advanced')
            measured = self.read(timeout_s=min(.001, remaining-min_validity_s))
            now = time.monotonic()
            received = self.received_for(measured)
            self.feedback_refresh = dict(retries=retries, elapsed_ms=(now-started)*1000,
                source_age_ms={name:1000*(now-stamp) for name,stamp in received.items()})
            try:
                # Invalid/future clocks still fail immediately. Age or temporary
                # skew is retried before the core can latch a stateful fault.
                validate_received(received, now, 'strict', check_age=False)
                valid = received_deadline(received, 'strict')-now >= min_validity_s
            except FeedbackSkewError:
                valid = False
            if valid and deadline-now > min_validity_s:
                return measured
            retries += 1
            if on_wait is not None:
                on_wait(retries)
            time.sleep(max(0., min(.001, deadline-time.monotonic()-min_validity_s)))

    def wait_ready(self, stopped):
        until = time.monotonic()+30
        while time.monotonic() < until and not stopped():
            self.read()
            status = self.status.get(timeout=.001)
            if status and status.get('phase') == 'running':
                return status['reference_seed']
            time.sleep(.004)
        raise RuntimeError('Backend acquisition did not complete')

    def received_for(self, measured):
        if measured != self.selected['measured']:
            raise RuntimeError('Feedback provenance does not match selected sample')
        return self.selected['received']

    def submit(self, sequence, command, mit, reference_velocity, *, timeout_s=.1,
               stopped=lambda: False, on_wait=None):
        frame = dict(sequence=sequence, created_s=float(mit.timestamp_s), command=command,
                     mit=encode_mit(mit), received=self.selected['received'],
                     measured=self.selected['measured'], reference_velocity=reference_velocity)
        # A failed semaphore acquisition writes nothing. Keep this exact pending
        # reference and retry; PinkIK must not solve again or refresh its clock.
        deadline = frame['created_s']+timeout_s
        started = time.monotonic()
        retries = 0
        while True:
            if stopped():
                raise ControlCancelled('Reference publication cancelled by stop request')
            status = self._actuator_status()
            if status and status.get('fault'):
                raise RuntimeError(status['fault'])
            remaining = deadline-time.monotonic()
            if remaining <= 0:
                raise RuntimeError(f'Reference publication deadline expired after {retries} retries; '
                                   'IK history remains pending')
            if self.frames.put(frame, timeout=min(.001, remaining)):
                self.command_deadline = deadline
                return dict(retries=retries, elapsed_ms=(time.monotonic()-started)*1000)
            retries += 1
            if on_wait is not None:
                on_wait(retries)
            time.sleep(max(0., min(.001, deadline-time.monotonic())))

    def acknowledgement(self, sequence):
        status = self._actuator_status()
        if status and status.get('fault'):
            raise RuntimeError(status['fault'])
        value = None if not status else status.get('receipt')
        return Receipt(**value) if value and value['sequence'] == sequence else None

    def close(self):
        pass  # The DA backend component owns all physical resources.


class BackendComponent(DABaseInterface):
    def __init__(self, name, frames, feedback, backend_status, control_status, measured,
                 plant_state, workspace, output, deployment_config, scene, backend='simulation',
                 execute=False, period_s=.004, command_timeout_s=.1, feedback_timeout_s=.1,
                 submit_delay_s=0., time_budget=.003, receipts=None, startup_lift_m=None):
        super().__init__(name, [frames,feedback,backend_status,control_status,measured,plant_state]
                         +([receipts] if receipts is not None else []), time_budget)
        self.receipts = receipts
        self.frames, self.feedback, self.status, self.control = frames, feedback, backend_status, control_status
        self.measured, self.plant_state = measured, plant_state
        self.workspace, self.output, self.config, self.scene = workspace, output, deployment_config, scene
        self.backend, self.execute, self.period = backend, execute, period_s
        self.command_timeout, self.feedback_timeout, self.submit_delay = command_timeout_s, feedback_timeout_s, submit_delay_s
        from .startup_lift import validate_target
        self.startup_lift_m = validate_target(startup_lift_m)

    def initialize(self):
        if self.backend not in ('simulation','hardware') or (self.backend == 'hardware' and not self.execute):
            raise ValueError('Hardware requires explicit --execute --confirm-onsite-ready')
        StreamContract(self.command_timeout, self.feedback_timeout)
        self._initialized.value = True
        return True

    def clean_up(self):
        self._cleaned_up.value = True
        return True

    def run(self):
        root = activate(self.workspace)
        driver = None
        native_diagnostic = None
        feedback_timing = deque(maxlen=512)
        pending = None
        stream = StreamContract(self.command_timeout, self.feedback_timeout)
        state = dict(phase='starting', fault=None, receipt=None, backend=self.backend)
        output = Path(self.output)
        try:
            config,mapping,profile,metadata = deployment(root,self.config,execute=self.execute)
            stream = StreamContract(self.command_timeout,self.feedback_timeout,metadata=metadata)
            if self.backend == 'simulation':
                from .simulation import SimulationPlant
                driver = SimulationPlant(root/self.scene, output, submit_delay_s=self.submit_delay,
                                         effort_limits=metadata['effort_limits_nm'], profile=profile,
                                         lift_speed_limit_m_s=metadata['velocity_bounds']['maver_x4_asm_URDF_joint_1'])
            else:
                from .native import NativeDriver
                driver = NativeDriver(mapping,profile,config,timeout_s=self.command_timeout,
                                      startup_lift_m=self.startup_lift_m)
            state.update(metadata=metadata, phase='prepared')
            self.status.put(state, timeout=.01)
            # Controller construction happens before native acquisition/streaming.
            deadline = time.monotonic()+30
            while not self.stopping:
                snapshot = driver.snapshot
                self.feedback.put(dict(measured=snapshot['measured'], received=snapshot['received']))
                control = self.control.get()
                if control and control.get('phase') == 'initialized':
                    break
                if time.monotonic() >= deadline or (control and control.get('fault')):
                    raise RuntimeError('Controller initialization failed')
                time.sleep(.004)
            if self.stopping:
                return
            def startup_progress(report):
                # Continue measured-time camera pairing while IK waits for its
                # final stationary seed. No policy action runs during this phase.
                snapshot = driver.snapshot
                self.feedback.put(dict(measured=snapshot['measured'], received=snapshot['received']), timeout=.001)
                self.measured.put(snapshot['measured'])
                if self.backend == 'simulation':
                    self.plant_state.put(snapshot)
                state.update(phase='startup_lift', startup_lift=report)
                self.status.put(state, timeout=.001)
            if self.startup_lift_m is not None:
                print(f'Startup lift: positioning to {self.startup_lift_m:.3f} m before arm preparation.', flush=True)
            driver.acquire(startup_lift_m=self.startup_lift_m,
                           progress=startup_progress, stopped=lambda: self.stopping)
            state['startup_lift'] = getattr(driver, 'startup_lift_report', None)
            if self.startup_lift_m is not None:
                print('Startup lift complete; locking lift for the policy session.', flush=True)
            seed = driver.snapshot
            stream.seed(seed['measured'])
            self.feedback.put(dict(measured=seed['measured'],received=seed['received']),timeout=.01)
            (output/'backend_metadata.json').write_text(json.dumps(metadata,indent=2)+'\n')
            state.pop('metadata',None)
            state.update(phase='running',reference_seed=seed['measured'])
            self.status.put(state, timeout=.01)
            self._ready.value = True
            pending = None
            first_until = time.monotonic()+1.
            next_render = 0.
            feedback_sequence = 0
            next_status = 0.
            while not self.stopping:
                started = time.monotonic()
                snapshot = driver.snapshot
                driver.check()
                measured = snapshot['measured']
                feedback_sequence += 1
                published_s = time.monotonic()
                published = self.feedback.put(dict(measured=measured, received=snapshot['received'],
                    publication_sequence=feedback_sequence, published_s=published_s), timeout=.001)
                feedback_timing.append(dict(started_s=started, completed_s=time.monotonic(),
                    publication_sequence=feedback_sequence, published_s=published_s,
                    received=snapshot['received'], snapshot_timing=snapshot.get('timing'), published=published))
                if started >= next_render:
                    self.measured.put(measured)
                    if self.backend == 'simulation':
                        self.plant_state.put(snapshot)
                    next_render = started+.02
                if stream.latest:
                    stream.check(measured)
                elif started > first_until:
                    raise RuntimeError('Controller did not produce a first reference')
                if pending is not None:
                    receipt = driver.acknowledgement(pending)
                    if receipt:
                        state['receipt'] = asdict(receipt)
                        pending = None
                frame = self.frames.get()
                if frame and (stream.latest is None or frame['sequence'] > stream.latest['sequence']):
                    if pending is not None:
                        raise RuntimeError('Controller advanced before actuator acceptance')
                    mit = stream.accept(frame,measured)
                    driver.submit(frame['sequence'],frame['command'],mit,
                                  created_s=frame['created_s'],received=frame['received'],measured=frame['measured'],reference_velocity=frame['reference_velocity'])
                    pending = frame['sequence']
                    state['latest_sequence'] = pending
                    state.pop('reference_seed',None)
                state.update(published_s=time.monotonic(), pending=pending)
                if self.receipts is not None:
                    self.receipts.put(state)
                if started >= next_status or self.receipts is None:
                    self.status.put(state, timeout=.001)
                    next_status = started+.1
                time.sleep(max(0.,started+self.period-time.monotonic()))
        except BaseException as error:
            state.update(phase='fault',fault=f'{type(error).__name__}: {error}')
            if 'frame' in locals() and frame is not None:
                state['rejected_frame'] = frame
            self.status.put(state, timeout=.01)
            if self.receipts is not None:
                self.receipts.put(state, timeout=.001)
        finally:
            if driver is not None:
                if hasattr(driver, 'diagnostics'):
                    try:
                        native_diagnostic = driver.diagnostics()
                    except Exception as error:
                        state['native_diagnostic_error'] = f'{type(error).__name__}: {error}'
                try:
                    until = time.monotonic()+self.command_timeout
                    while pending is not None and not state.get('fault') and time.monotonic() < until:
                        receipt = driver.acknowledgement(pending)
                        if receipt:
                            state['receipt'] = asdict(receipt)
                            pending = None
                            self.status.put(state,timeout=.001)
                            if self.receipts is not None:
                                self.receipts.put(state,timeout=.001)
                        else:
                            time.sleep(.001)
                    state.update(cancelled_pending=pending, pending=pending, published_s=time.monotonic())
                    driver.close()
                except Exception as error:
                    state['shutdown_error'] = f'{type(error).__name__}: {error}'
            if native_diagnostic is not None:
                try:
                    native_diagnostic['feedback_publications'] = list(feedback_timing)
                    # Large evidence belongs on disk after stop, never in the
                    # periodic shared-memory status or the policy observation.
                    path = output/'native_transport.json'
                    path.write_text(json.dumps(native_diagnostic, indent=2, allow_nan=False,
                                               default=diagnostic_value)+'\n')
                    state['native_diagnostic_path'] = str(path)
                except Exception as error:
                    state['native_diagnostic_error'] = f'{type(error).__name__}: {error}'
            if not state.get('fault'):
                state['phase'] = 'closed'
            self.status.put(state,timeout=.01)
            (output/'backend_final.json').write_text(json.dumps(state,indent=2)+'\n')
            while state.get('fault') and not self.stopping:
                time.sleep(.1)
