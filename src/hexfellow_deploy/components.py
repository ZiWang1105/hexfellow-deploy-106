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
from .workspace import activate, build_executor


class ControlComponent(DABaseInterface):
    def __init__(self, name, request, status, measured, plant_state, workspace,
                 output, scene, controller, period_s, time_budget,
                 backend="simulation", health_timeout_s=.1, startup_carry=False,
                 inject_every=0, inject_delay_s=0., submit_delay_s=0.):
        super().__init__(name, [request, status, measured, plant_state], time_budget)
        self.request, self.status, self.measured_topic, self.plant_topic = request, status, measured, plant_state
        self.workspace, self.output, self.scene, self.controller = workspace, output, scene, controller
        self.period, self.health_timeout = period_s, health_timeout_s
        self.backend, self.startup_carry = backend, startup_carry
        self.inject_every, self.inject_delay, self.submit_delay = inject_every, inject_delay_s, submit_delay_s

    def initialize(self):
        if self.backend != "simulation":
            raise ValueError("Native MIT migration requires the qualified native adapter; old gateway is not a fallback")
        if self.health_timeout <= self.period or self.inject_delay < 0 or self.submit_delay < 0:
            raise ValueError("Invalid timing configuration")
        self._initialized.value = True
        return True

    def clean_up(self):
        self._cleaned_up.value = True
        return True

    def run(self):
        root = activate(self.workspace)
        from .simulation import SimulationPlant
        from mobile_manipulator.interfaces.contract import Action
        from mobile_manipulator.control.startup_pose import DEFAULT_STARTUP_POSE, load_ik_pose
        from mobile_manipulator.sim.scene_config import load_scene
        logger = logging.getLogger(self.name)
        output = Path(self.output)
        output.mkdir(parents=True, exist_ok=True)
        owner, plant = ReferenceOwner(), None
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
        last_request = None
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
                if plant is not None:
                    snapshot = plant.snapshot
                    self.measured_topic.put(snapshot["measured"])
                    self.plant_topic.put(snapshot)
        publisher = threading.Thread(target=publish, name="telemetry-publish", daemon=True)
        publisher.start()
        try:
            scene_path = root / self.scene
            executor = build_executor(load_scene(scene_path), root / self.controller, self.period)
            plant = SimulationPlant(scene_path, output, submit_delay_s=self.submit_delay)
            executor.ik.control_clock = time.monotonic
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
                if request is not None and request["id"] != last_request:
                    last_request = request["id"]
                    try:
                        if request["operation"] == "hold":
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
                generated = owner.generate(lambda: executor.tick(measured))
                sequence, (command, diagnostic) = generated
                solved = time.monotonic()
                plant.submit(sequence, command, executor.ik.last_result.mit)
                pending_since = time.monotonic()
                ticks += 1
                timing = budget.finish(started)
                if timing["budget_overrun"]:
                    logger.warning("Control budget exceeded: %.3f ms; execution continues", timing["elapsed_ms"])
                state = dict(state, phase="running", ticks=ticks, diagnostic=diagnostic,
                             timing=timing, request_id=last_request)
                try:
                    rows.put_nowait(dict(sequence=sequence, started_s=started, solved_s=solved,
                        submitted_s=pending_since, timing=timing, measured=measured,
                        reference=command, diagnostic=diagnostic,
                        interval_ms=None if previous_solve_started is None else (started-previous_solve_started)*1000,
                        previous_acceptance_ms=accepted_latency_ms,
                        accepted_before_solve=owner.accepted))
                except queue.Full:
                    logger.warning("Control recording queue full; dropped a diagnostic row")
                previous_solve_started = started
                # Work includes validation, submission and recording enqueue.
                time.sleep(max(0., started+self.period-time.monotonic()))
        except BaseException as error:
            owner.fail(f"{type(error).__name__}: {error}")
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
                        owner.fail('Final actuator acknowledgement unavailable')
                    plant.close()
                except Exception as error:
                    owner.fail(f'Cleanup failed: {error}')
            state = dict(state, phase='fault' if owner.fault else 'closed')
            publisher_stop.set()
            publisher.join(2)
            final = dict(state, generated=owner.generated, accepted=owner.accepted,
                         pending=owner.pending, fault=owner.fault, recording_error=recording_error,
                         evidence_kind="simulation_position_actuators_not_native_MIT")
            self.status.put(final, timeout=.01)
            (output / "control_final.json").write_text(json.dumps(final, indent=2)+"\n")
            rows.put(None, timeout=2)
            writer.join(3)
            # A controller fault does not erase the operator's diagnostic service.
            while owner.fault and not self.stopping:
                time.sleep(.1)
