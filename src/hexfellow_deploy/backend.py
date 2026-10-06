"""One DA actuator owner and IPC protocol for both physical and simulated devices."""
from dataclasses import asdict
import json
from pathlib import Path
import time

from deploy_anything.components.interface.base_interface import DABaseInterface
from .actuation import StreamContract, encode_mit, deployment
from .reference import Receipt
from .workspace import activate


class BackendProxy:
    def __init__(self, frames, feedback, backend_status, timeout=30.):
        self.frames, self.feedback, self.status = frames, feedback, backend_status
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

    def read(self):
        status = self.status.get()
        if status and status.get('fault'):
            raise RuntimeError(status['fault'])
        row = self.feedback.get()
        if row is not None:
            self.selected = row
        if self.selected is None:
            raise RuntimeError('Backend feedback unavailable')
        row = self.selected
        return row['measured']

    def wait_ready(self, stopped):
        until = time.monotonic()+30
        while time.monotonic() < until and not stopped():
            self.read()
            status = self.status.get()
            if status and status.get('phase') == 'running':
                return status['reference_seed']
            time.sleep(.004)
        raise RuntimeError('Backend acquisition did not complete')

    def received_for(self, measured):
        if measured != self.selected['measured']:
            raise RuntimeError('Feedback provenance does not match selected sample')
        return self.selected['received']

    def submit(self, sequence, command, mit, reference_velocity):
        frame = dict(sequence=sequence, created_s=float(mit.timestamp_s), command=command,
                     mit=encode_mit(mit), received=self.selected['received'],
                     measured=self.selected['measured'], reference_velocity=reference_velocity)
        if not self.frames.put(frame, timeout=.001):
            raise RuntimeError('Reference publication unavailable; do not advance history')

    def acknowledgement(self, sequence):
        status = self.status.get()
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
                 submit_delay_s=0., time_budget=.003):
        super().__init__(name, [frames,feedback,backend_status,control_status,measured,plant_state], time_budget)
        self.frames, self.feedback, self.status, self.control = frames, feedback, backend_status, control_status
        self.measured, self.plant_state = measured, plant_state
        self.workspace, self.output, self.config, self.scene = workspace, output, deployment_config, scene
        self.backend, self.execute, self.period = backend, execute, period_s
        self.command_timeout, self.feedback_timeout, self.submit_delay = command_timeout_s, feedback_timeout_s, submit_delay_s

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
                driver = NativeDriver(mapping,profile,config,timeout_s=self.command_timeout)
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
            driver.acquire()
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
            while not self.stopping:
                started = time.monotonic()
                snapshot = driver.snapshot
                driver.check()
                measured = snapshot['measured']
                self.feedback.put(dict(measured=measured, received=snapshot['received']))
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
                self.status.put(state)
                time.sleep(max(0.,started+self.period-time.monotonic()))
        except BaseException as error:
            state.update(phase='fault',fault=f'{type(error).__name__}: {error}')
            if 'frame' in locals() and frame is not None:
                state['rejected_frame'] = frame
            self.status.put(state, timeout=.01)
        finally:
            if driver is not None:
                try:
                    until = time.monotonic()+self.command_timeout
                    while pending is not None and not state.get('fault') and time.monotonic() < until:
                        receipt = driver.acknowledgement(pending)
                        if receipt:
                            state['receipt'] = asdict(receipt)
                            pending = None
                            self.status.put(state,timeout=.001)
                        else:
                            time.sleep(.001)
                    state.update(cancelled_pending=pending, pending=pending, published_s=time.monotonic())
                    driver.close()
                except Exception as error:
                    state['shutdown_error'] = f'{type(error).__name__}: {error}'
            if not state.get('fault'):
                state['phase'] = 'closed'
            self.status.put(state,timeout=.01)
            (output/'backend_final.json').write_text(json.dumps(state,indent=2)+'\n')
            while state.get('fault') and not self.stopping:
                time.sleep(.1)
