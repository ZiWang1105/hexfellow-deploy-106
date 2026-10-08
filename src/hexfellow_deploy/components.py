"""DA-managed owners: latest-state IPC, one reference writer, no old gateway."""
from dataclasses import asdict
import logging
from pathlib import Path
import queue
import threading
import time
import json

from deploy_anything.components.interface.base_interface import DABaseInterface
from .reference import ReferenceOwner
from .timing import CycleBudget
from .workspace import activate, build_shared_executor


class ControlComponent(DABaseInterface):
    def __init__(self, name, request, status, frames, feedback, backend_status, workspace,
                 output, period_s, time_budget, health_timeout_s=.1, startup_carry=False,
                 inject_every=0, inject_delay_s=0., receipts=None):
        super().__init__(name, [request, status, frames, feedback, backend_status]
                         +([receipts] if receipts is not None else []), time_budget)
        self.receipts = receipts
        self.request, self.status = request, status
        self.frames, self.feedback, self.backend_status = frames, feedback, backend_status
        self.workspace, self.output = workspace, output
        self.period, self.health_timeout = period_s, health_timeout_s
        self.startup_carry = startup_carry
        self.inject_every, self.inject_delay = inject_every, inject_delay_s

    def initialize(self):
        if self.health_timeout <= self.period or self.inject_delay < 0:
            raise ValueError("Invalid timing configuration")
        self._initialized.value = True
        return True

    def clean_up(self):
        self._cleaned_up.value = True
        return True

    def run(self):
        root = activate(self.workspace)
        from .backend import BackendProxy, ControlCancelled
        from .actuation import encode_mit
        from mobile_manipulator.interfaces.contract import Action
        from mobile_manipulator.control.startup_pose import DEFAULT_STARTUP_POSE, load_ik_pose
        logger = logging.getLogger(self.name)
        output = Path(self.output)
        output.mkdir(parents=True, exist_ok=True)
        owner, plant = ReferenceOwner(), None
        executor = None
        measured = None
        budget = CycleBudget(self.period, self.time_budget)
        rows = queue.Queue(maxsize=4096)
        recording_error = []
        def write():
            try:
                with (output / "control.jsonl").open("w") as stream:
                    while True:
                        row = rows.get()
                        if row is None:
                            return
                        stream.write(json.dumps(row, allow_nan=False) + "\n")
            except Exception as error:
                recording_error.append(str(error))
        writer = threading.Thread(target=write, name="control-recording", daemon=True)
        writer.start()
        ticks = 0
        dropped_records = 0
        last_request = None
        final_hold = False
        pending_since = None
        previous_solve_started = None
        accepted_latency_ms = None
        state = dict(phase="starting", diagnostic={}, request_id=None)
        publisher_stop = threading.Event()
        def publish():
            while not publisher_stop.wait(.05):
                self.status.put(dict(state, published_s=time.monotonic(), generated=owner.generated,
                    accepted=owner.accepted, pending=owner.pending, fault=owner.fault,
                    acknowledgement=None if owner.last_receipt is None else asdict(owner.last_receipt)))
        publisher = threading.Thread(target=publish, name="telemetry-publish", daemon=True)
        publisher.start()
        try:
            plant = BackendProxy(self.frames, self.feedback, self.backend_status, receipts=self.receipts)
            executor = build_shared_executor(plant.metadata, self.period)
            executor.ik.control_clock = time.monotonic
            executor.ik.feedback_provider = plant.received_for
            executor.ik.report['actuation'] = 'Shared MIT frame -> DA actuator owner -> native SDK or MuJoCo torque servo'
            state = dict(state, phase='initialized')
            self.status.put(state, timeout=.01)
            seed = plant.wait_ready(lambda: self.stopping)
            executor.ik.initialize_reference(seed,stationary_reference=True)
            if self.startup_carry:
                executor.start_carry(load_ik_pose(DEFAULT_STARTUP_POSE), plant.read())
            self._ready.value = True
            state = dict(phase="running", diagnostic={'mode': 'startup_carry' if self.startup_carry else 'hold'},
                         request_id=None, controller=executor.ik.report)
            while not self.stopping:
                started = time.monotonic()
                measured = plant.read()
                if started-measured["timestamp_s"] > self.health_timeout:
                    raise RuntimeError("Actual plant feedback is stale")
                if owner.pending is not None:
                    receipt = plant.acknowledgement(owner.pending)
                    if receipt is not None:
                        owner.acknowledge(receipt)
                        accepted_latency_ms = (receipt.accepted_at-pending_since)*1000
                        pending_since = None
                    elif started-pending_since > self.health_timeout:
                        raise RuntimeError("Actuator acknowledgement unavailable; IK history cannot advance")
                    else:
                        state = dict(state, phase="waiting_for_acceptance")
                        time.sleep(min(.001, self.period))
                        continue
                request = self.request.get()
                if request is not None and request["id"] != last_request and not final_hold:
                    last_request = request["id"]
                    try:
                        if request['operation'] == 'final_hold':
                            executor.hold(measured)
                            final_hold = True
                            state = dict(state, final_hold=True)
                        elif request["operation"] == "hold":
                            executor.hold(measured)
                        elif request["operation"] == "target":
                            observed = request["observed"]
                            if started-observed["timestamp_s"] > 90:
                                raise ValueError("Observation expired; reobserve")
                            from mobile_manipulator.robot.joints import LIFT
                            if any(abs(measured["joint_positions"][n]-q) > (.01 if n == LIFT else .05)
                                   for n, q in observed["joint_positions"].items()):
                                raise ValueError("Posture changed during inference; reobserve")
                            executor.submit(Action.from_dict(request["action"]), observed)
                        else:
                            raise ValueError("Unknown control request")
                        state = dict(state, request_id=last_request, request_error=None)
                    except ValueError as error:
                        state = dict(state, request_id=last_request, request_error=str(error))
                if self.inject_every and ticks and ticks % self.inject_every == 0:
                    time.sleep(self.inject_delay)
                    # Read actual new feedback after the injected delay.
                    measured = plant.read()
                def feedback_wait(retries):
                    nonlocal state
                    state = dict(state, phase='waiting_for_feedback', feedback_refresh_retries=retries)
                    if retries == 1:
                        logger.warning('Waiting for fresh feedback before IK; reference history unchanged')
                measured = plant.read_for_control(timeout_s=self.health_timeout,
                    min_validity_s=self.time_budget, stopped=lambda: self.stopping, on_wait=feedback_wait)
                plant.created_s = time.monotonic()
                generated = owner.generate(lambda: executor.tick(measured))
                sequence, (command, diagnostic) = generated
                solved = time.monotonic()
                def publication_wait(retries):
                    nonlocal state
                    state = dict(state, phase='waiting_for_publication', publication_retries=retries)
                    if retries == 1:
                        logger.warning('Reference shared memory busy; retrying pending sequence %s', sequence)
                publication = plant.submit(sequence, command, executor.ik.last_result.mit,
                             {n:float(executor.ik.last_result.velocity[i]) for n,i in executor.ik.indices.items()},
                             timeout_s=self.health_timeout, stopped=lambda: self.stopping,
                             on_wait=publication_wait)
                pending_since = time.monotonic()
                ticks += 1
                timing = budget.finish(started)
                if timing["budget_overrun"]:
                    logger.warning("Control budget exceeded: %.3f ms; execution continues", timing["elapsed_ms"])
                state = dict(state, phase="running", ticks=ticks, diagnostic=diagnostic,
                             timing=timing, request_id=last_request, publication=publication,
                             publication_retries=publication['retries'])
                try:
                    rows.put_nowait(dict(sequence=sequence, started_s=started, solved_s=solved,
                        submitted_s=pending_since, timing=timing, publication=publication, measured=measured,
                        reference=command, diagnostic=diagnostic, request_id=last_request,
                        mit=encode_mit(executor.ik.last_result.mit),
                        reference_velocity={n:float(executor.ik.last_result.velocity[i]) for n,i in executor.ik.indices.items()},
                        source_received=plant.received_for(measured),
                        active_tcp_targets=None if executor.target is None or executor.requires_base else {
                            side: dict(position_m=pos.tolist(),rotation=rot.tolist())
                            for side,(pos,rot) in executor.arm_targets.items()},
                        interval_ms=None if previous_solve_started is None else (started-previous_solve_started)*1000,
                        feedback_refresh=plant.feedback_refresh,
                        feedback_read=plant.feedback_read,
                        previous_acceptance_ms=accepted_latency_ms,
                        accepted_before_solve=owner.accepted))
                except queue.Full:
                    dropped_records += 1
                    logger.warning("Control recording queue full; dropped a diagnostic row")
                previous_solve_started = started
                # Work includes validation, submission and recording enqueue.
                time.sleep(max(0., started+self.period-time.monotonic()))
        except ControlCancelled as error:
            # Shutdown can arrive between solving and publishing. Preserve the
            # unaccepted sequence in the final record; do not call it a fault or
            # invent an actuator acknowledgement.
            state = dict(state, cancellation=str(error))
        except BaseException as error:
            owner.fail(f"{type(error).__name__}: {error}")
            state = dict(state, fault_feedback=measured)
            if plant is not None:
                state = dict(state, feedback_read=getattr(plant, 'feedback_read', None),
                             feedback_refresh=getattr(plant, 'feedback_refresh', None))
            if executor is not None:
                state = dict(state, ik_fault_info=getattr(executor.ik.core,"_fault_info",None))
                capture = getattr(executor.ik.core,'last_qp_failure',None)
                if capture is not None:
                    try:
                        (output/'qp_failure.json').write_text(json.dumps(dict(
                            scalar_conflicts=capture.scalar_conflicts(),fault=owner.fault),indent=2))
                        import numpy as np
                        np.savez_compressed(output/'qp_failure.npz',**capture.arrays)
                    except Exception as diagnostic_error:
                        recording_error.append(str(diagnostic_error))
            state = dict(state, phase="fault", fault=owner.fault)
            logger.exception("Control session failed; diagnostics remain available")
        finally:
            if plant is not None:
                try:
                    # A normal stop drains only the already submitted reference.
                    # It must not report a generated command as accepted by fiat.
                    deadline = time.monotonic()+self.health_timeout
                    while owner.pending is not None and not owner.fault and time.monotonic() < deadline:
                        receipt = plant.acknowledgement(owner.pending)
                        if receipt is not None:
                            owner.acknowledge(receipt)
                        else:
                            time.sleep(.001)
                    if owner.pending is not None and not owner.fault:
                        state = dict(state, cancelled_pending=owner.pending)
                    plant.close()
                except Exception as error:
                    owner.fail(f'Cleanup failed: {error}')
            state = dict(state, phase='fault' if owner.fault else 'closed')
            publisher_stop.set()
            publisher.join(2)
            final = dict(state, generated=owner.generated, accepted=owner.accepted,
                         pending=owner.pending, fault=owner.fault, recording_error=recording_error,
                         dropped_records=dropped_records,
                         evidence_kind="shared_MIT_control_and_actuator_protocol")
            self.status.put(final, timeout=.01)
            (output / "control_final.json").write_text(json.dumps(final, indent=2)+"\n")
            rows.put(None, timeout=2)
            writer.join(3)
            # A controller fault does not erase the operator's diagnostic service.
            while owner.fault and not self.stopping:
                time.sleep(.1)
