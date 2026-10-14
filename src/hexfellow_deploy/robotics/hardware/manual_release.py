"""Single-use, single-arm zero-torque diagnostic; independent of model mapping."""

import copy
import json
import multiprocessing as mp
import queue
import signal
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from .hexfellow_endpoint import TelemetryTap, passive_api_close
from .whole_body import vector
from .x7 import HardwareSafetyError
from .x7_joint_console import ARM_PORTS, build_scoped_websocket_url
from .x7_sdk_smoke import _device_identity, _load_vendor_sdk, _wait_for_x7


class ZeroTorqueGate:
    """Checked at SDK send time, including commands snapshotted before stopping."""

    def __init__(self, heartbeat, stopped, clock=time.monotonic):
        self.clock, self.heartbeat, self.stopped = clock, heartbeat, stopped
        self.lock = threading.RLock()
        self.phase = "readonly"
        self.init_sent = False
        self.owned = False
        self.deadline = 0.
        self.fault = None
        self.counts = dict(initialize=0, zero_torque=0, brake=0, release=0, blocked=0)

    def stop(self):
        with self.lock:
            self.phase = "stopping"

    def filter(self, message):
        with self.lock:
            now = self.clock()
            if self.phase in {"initializing", "active"}:
                if self.stopped() or not 0 <= now - self.heartbeat() <= .25 or now > self.deadline:
                    self.fault = self.fault or "release stopped or watchdog expired"
                    self.phase = "stopping"
            kind = message.WhichOneof("down")
            if kind in {"set_report_frequency", "placeholder_message"}:
                return message
            if kind != "arm_command" or message.arm_command.WhichOneof("command") != "arm_exclusive_command":
                return self.reject()
            exclusive = message.arm_command.arm_exclusive_command
            kind = exclusive.WhichOneof("exclusive_command")
            if kind == "api_control_initialize":
                if exclusive.api_control_initialize and self.phase == "initializing" and not self.init_sent:
                    self.init_sent = True
                    self.counts["initialize"] += 1
                    return message
                if not exclusive.api_control_initialize and self.phase == "stopping" and self.init_sent:
                    self.counts["release"] += 1
                    return message
            if kind == "arm_api_control_command":
                control = exclusive.arm_api_control_command
                if control.WhichOneof("command") != "motor_targets":
                    return self.reject()
                targets = control.motor_targets.targets
                modes = [t.WhichOneof("target") for t in targets]
                if not modes and self.phase == "initializing":
                    return message
                if self.owned and self.phase == "stopping":
                    result = copy.deepcopy(message)
                    dest = result.arm_command.arm_exclusive_command.arm_api_control_command.motor_targets
                    dest.ClearField("targets")
                    for _ in range(7):
                        dest.targets.add(brake=True)
                    self.counts["brake"] += 1
                    return result
                if self.owned and self.phase == "active":
                    if modes == ["torque"] * 7 and all(t.torque == 0. for t in targets):
                        self.counts["zero_torque"] += 1
                        return message
                    if modes == ["brake"] * 7 and all(t.brake for t in targets):
                        self.fault = "SDK requested brake"
                        self.phase = "stopping"
                        self.counts["brake"] += 1
                        return message
            return self.reject()

    def reject(self):
        self.counts["blocked"] += 1
        if self.phase in {"initializing", "active"}:
            self.fault = "unexpected command (calibration/rearm/nonzero/other-device commands prohibited)"
            self.phase = "stopping"
        return None


def release_api_type(base, gate):
    class ReleaseApi(base):
        async def _send_down_message(self, data):
            allowed = gate.filter(data)
            if allowed is not None:
                return await super()._send_down_message(allowed)

        async def _device_periodic_runner(self, device_id):
            from hex_device.hands import Hands

            if isinstance(self._device_id_map.get(device_id), Hands):
                return  # No auxiliary gripper command tasks in a release connection.
            return await super()._device_periodic_runner(device_id)
    return ReleaseApi


def checked_row(tap, now, *, health=True):
    row = tap.read()
    if not 0 <= now - row["received"] <= .1:
        raise HardwareSafetyError("selected arm feedback older than 100 ms")
    vector(row["pos"], 7, "SDK positions")
    vector(row["vel"], 7, "SDK velocities")
    if health:
        reasons = []
        if row["parking_stop"]:
            reasons.append(f"parking_stop={row['parking_stop']}")
        if row["motor_states"] != ["normal"] * 7:
            reasons.append(f"motor_states={row['motor_states']}")
        if not row["calibrated"]:
            reasons.append("calibrated=False (automatic calibration disabled)")
        if reasons:
            error = HardwareSafetyError("release precheck: " + "; ".join(reasons))
            error.feedback_snapshot = row
            raise error
    return row


def release_preflight(device, tap, stopped, heartbeat, record, *, clock, sleep):
    """Wait for startup readiness without ever acquiring control or clearing faults."""
    deadline = clock() + 5.
    last_status = None
    while True:
        now = clock()
        if stopped() or not 0 <= now - heartbeat() <= .25:
            raise HardwareSafetyError("viewer stopped before release precheck completed")
        row = checked_row(tap, now, health=False)
        status = {key: row.get(key) for key in
                  ("calibrated", "initialized", "motor_states", "parking_stop", "session_holder", "mine")}
        status_key = repr(status)
        if status_key != last_status:
            details = {}
            for method in ("get_motor_error_codes", "get_motor_warnings"):
                getter = getattr(device, method, None)
                if getter is not None:
                    details[method.removeprefix("get_")] = getter()
            record("precheck", feedback_snapshot=row, **details)
            last_status = status_key
        # Actual reported errors are not treated as a startup delay.
        if row["parking_stop"] or (row["motor_states"] is not None and row["motor_states"] != ["normal"] * 7):
            return checked_row(tap, clock())
        if row["calibrated"] and row["motor_states"] == ["normal"] * 7:
            return row
        if now >= deadline:
            return checked_row(tap, clock())  # Detailed reason, with the final snapshot.
        sleep(.02)


def release_session(device, tap, gate, command_type, stopped, heartbeat, record,
                    *, clock=time.monotonic, sleep=time.sleep):
    """Returns only after confirmed stop, or before any control request was made."""
    attempted = False
    error = None
    try:
        names, lower, upper = _device_identity(device)
        if names != tuple(f"joint_{i}" for i in range(1, 8)):
            raise HardwareSafetyError("unexpected SDK joint order")
        row = release_preflight(device, tap, stopped, heartbeat, record, clock=clock, sleep=sleep)
        if row["session_holder"] != 0 or np.max(np.abs(row["vel"])) > .05:
            raise HardwareSafetyError("arm must be stationary and have no control owner")
        if np.any(np.asarray(row["pos"]) < lower) or np.any(np.asarray(row["pos"]) > upper):
            raise HardwareSafetyError("initial SDK positions outside limits")
        initial = np.asarray(row["pos"])
        device.command_timeout_check(True)
        with gate.lock:
            if stopped() or not 0 <= clock() - heartbeat() <= .25:
                raise HardwareSafetyError("viewer stopped before acquisition")
            gate.phase = "initializing"
            gate.deadline = clock() + .1
        attempted = True
        record("initializing")
        device.start()
        until = clock() + 5.
        while True:
            row = checked_row(tap, clock())
            if stopped() or gate.fault or clock() > gate.deadline or not 0 <= clock() - heartbeat() <= .25:
                raise HardwareSafetyError(gate.fault or "stopped during acquisition")
            if row["session_holder"] not in (0, row["mine"]):
                raise HardwareSafetyError("control owner changed during acquisition")
            if np.max(np.abs(np.asarray(row["pos"]) - initial)) > .01 or np.max(np.abs(row["vel"])) > .05:
                raise HardwareSafetyError("arm moved during acquisition")
            with gate.lock:
                gate.deadline = min(clock() + .1, row["received"] + .1)
                gate.owned = row["mine"] != 0 and row["session_holder"] == row["mine"]
            if gate.owned and row["initialized"]:
                break
            if clock() >= until:
                raise HardwareSafetyError("control acquisition exceeded 5 seconds")
            sleep(.02)
        with gate.lock:
            if gate.phase != "initializing" or gate.fault or stopped():
                raise HardwareSafetyError("release cancelled before activation")
            gate.phase = "active"
        record("active")
        while not stopped():
            now = clock()
            row = checked_row(tap, now)
            if not row["initialized"] or row["mine"] == 0 or row["session_holder"] != row["mine"]:
                with gate.lock:
                    gate.owned = False
                raise HardwareSafetyError("selected arm control ownership lost")
            with gate.lock:
                if gate.fault or gate.phase != "active" or now > gate.deadline or not 0 <= now - heartbeat() <= .25:
                    raise HardwareSafetyError(gate.fault or "viewer heartbeat expired")
                gate.deadline = min(now + .1, row["received"] + .1)
            device.motor_command(command_type.TORQUE, [0.] * 7)
            record("sample", sdk_positions=row["pos"], sdk_velocities=row["vel"], received=row["received"])
            sleep(.02)
    except Exception as exc:
        error = str(exc)
        record("fault", error=error, feedback_snapshot=getattr(exc, "feedback_snapshot", None))
    finally:
        gate.stop()
        if attempted:
            # Keep the brake sender alive if stopping cannot be confirmed.
            record("stopping")
            requested = clock()
            last_received = requested
            stable = None
            released = False
            warned = False
            while True:
                now = clock()
                try:
                    row = checked_row(tap, now, health=False)
                    own = row["mine"] != 0 and row["session_holder"] == row["mine"]
                    with gate.lock:
                        gate.owned = own
                    quiet = row["received"] > max(requested, last_received) and np.max(np.abs(row["vel"])) <= .01
                    last_received = max(last_received, row["received"])
                    if own and not released:
                        device.motor_command(command_type.BRAKE, [True] * 7)
                    stable = (now if stable is None else stable) if quiet else None
                    if stable is not None and now - stable >= .1:
                        if own and not released:
                            device.stop()
                            released = True
                            requested = now
                            stable = None
                        elif row["session_holder"] == 0 and (released or not gate.init_sent):
                            record("stopped", stop_confirmed=True, error=error)
                            break
                        elif row["session_holder"] == 0 and not released:
                            # Cancel a potentially pending initialize on this connection.
                            device.stop()
                            released = True
                            requested = now
                            stable = None
                except Exception:
                    stable = None
                    if gate.owned and not released:
                        try:
                            device.motor_command(command_type.BRAKE, [True] * 7)
                        except Exception:
                            pass
                if not warned and now - requested > 2.:
                    warned = True
                    record("unconfirmed", error="STOP NOT CONFIRMED; use physical E-stop; brake worker retained")
                sleep(.02)
    return error


def _worker(mapping, side, heartbeat, stop, statuses, output):
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    signal.signal(signal.SIGTERM, lambda *_: stop.set())
    api = None
    gate = ZeroTorqueGate(lambda: heartbeat.value, stop.is_set)
    summary = {"side": side, "mapping_sha256": mapping.sha256, "stop_confirmed": False}
    with (Path(output) / "release.jsonl").open("x") as stream:
        def record(event, **details):
            row = dict(event=event, at=time.monotonic(), **details)
            try:
                stream.write(json.dumps(row, default=lambda value: np.asarray(value).tolist()) + "\n")
                stream.flush()
            except (OSError, TypeError, ValueError) as error:
                stop.set()
                summary["logging_error"] = str(error)
            if event != "sample":
                summary.update(row)
                try:
                    (Path(output) / "release-summary.json").write_text(
                        json.dumps(summary, indent=2, default=lambda value: np.asarray(value).tolist()))
                except OSError as error:
                    stop.set()
                    summary["logging_error"] = str(error)
                try:
                    statuses.put_nowait(row)
                except queue.Full:
                    pass
        try:
            api_type, command_type, robot_type = _load_vendor_sdk()
            url = build_scoped_websocket_url(mapping.data["arm_controller_ip"], ARM_PORTS[side],
                                             mapping.data["interface"])
            # Ordered WebSocket delivery: no KCP retransmission of superseded zero commands.
            api = release_api_type(api_type, gate)(ws_url=url, control_hz=250, enable_kcp=False, local_port=0)
            device = _wait_for_x7(api, robot_type, 5, clock=time.monotonic, sleep=time.sleep)
            tap = TelemetryTap(side, device)
            deadline = time.monotonic() + 5
            while tap.value is None:
                if stop.is_set() or time.monotonic() > deadline:
                    raise HardwareSafetyError("no selected-arm feedback before release")
                time.sleep(.01)
            release_session(device, tap, gate, command_type, stop.is_set, lambda: heartbeat.value, record)
        except Exception as error:
            record("fault", error=str(error))
        finally:
            gate.stop()
            if api is not None:
                try:
                    passive_api_close(api)
                except Exception as error:
                    record("fault", error=f"close: {error}")
            summary["wire_counts"] = gate.counts
            (Path(output) / "release-summary.json").write_text(
                json.dumps(summary, indent=2, default=lambda value: np.asarray(value).tolist()))


class ManualRelease:
    def __init__(self, mapping, side):
        self.mapping, self.side = mapping, side
        self.context = mp.get_context("spawn")
        self.heartbeat = self.context.Value("d", time.monotonic())
        self.stop = self.context.Event()
        self.statuses = self.context.Queue(maxsize=32)
        self.process = None
        self.used = False
        self.failed = False
        self.exit_reported = False
        self.message = f"{side.upper()} READ ONLY | terminal u + Enter release | b + Enter brake"
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
        self.output = Path("runs") / f"manual-release-{stamp}"
        self.output.mkdir(parents=True, exist_ok=False)
        self.trace = (self.output / "mirror.jsonl").open("x")

    def update(self, state, variant):
        now = time.monotonic()
        if not 0 <= now - state.received[self.side] <= .1:
            self.request_stop()
        else:
            self.heartbeat.value = now
        self.trace.write(json.dumps(dict(at=now, variant=variant, source_model_q=state.q.tolist(),
                                       raw_sdk={s: state.health[s].get("pos") for s in ("left", "right")}),
                                   default=lambda value: np.asarray(value).tolist()) + "\n")
        for _ in range(32):
            try:
                row = self.statuses.get_nowait()
            except queue.Empty:
                break
            self.failed |= row["event"] in {"fault", "unconfirmed"}
            self.message = f"{self.side.upper()} {row['event'].upper()} | {row.get('error') or ''}"
            print(self.message, flush=True)
            if row["event"] == "precheck":
                snapshot = row["feedback_snapshot"]
                print("Release precheck:", {k: snapshot.get(k) for k in
                      ("calibrated", "initialized", "motor_states", "parking_stop", "session_holder")},
                      "motor_error_codes:", row.get("motor_error_codes"),
                      "motor_warnings:", row.get("motor_warnings"), flush=True)
        if self.process is not None and self.process.exitcode is not None:
            if self.process.exitcode != 0:
                self.failed = True
                self.message = "Release worker failed; stop unconfirmed — use physical E-stop"
                if not self.exit_reported:
                    print(self.message, "| Report:", self.output, flush=True)
                    self.exit_reported = True

    def start(self):
        if self.used:
            print(f"Release rejected: {self.message}. One release per run; restart to try again.", flush=True)
            return
        self.used = True
        self.process = self.context.Process(target=_worker, args=(self.mapping, self.side, self.heartbeat,
                                                                 self.stop, self.statuses, self.output),
                                            name=f"manual-release-{self.side}")
        self.process.start()
        self.message = f"{self.side.upper()} CONNECTING — support the arm"
        print(self.message, "| Report:", self.output, flush=True)

    def request_stop(self):
        if self.stop.is_set():
            return
        self.stop.set()
        self.used = True
        self.message = f"{self.side.upper()} STOP REQUESTED; release disabled for this run"
        print(self.message, flush=True)

    def close(self):
        self.request_stop()
        self.trace.close()
        if self.process is not None:
            self.process.join(timeout=3.)
            if self.process.is_alive():
                self.failed = True
                print(f"STOP UNCONFIRMED; brake worker PID {self.process.pid} retained."
                      " Use physical E-stop.", flush=True)
            else:
                try:
                    summary = json.loads((self.output / "release-summary.json").read_text())
                    self.failed |= bool(summary.get("error") or summary.get("logging_error"))
                    self.failed |= self.process.exitcode != 0
                    initialized = summary.get("wire_counts", {}).get("initialize", 0) > 0
                    self.failed |= initialized and not summary["stop_confirmed"]
                except (OSError, ValueError, KeyError):
                    self.failed = True
        print("Manual release report:", self.output, flush=True)
        return not self.failed
