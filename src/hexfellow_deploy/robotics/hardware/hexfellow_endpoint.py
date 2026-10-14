"""Version-pinned SDK compatibility boundary for the whole-body gateway.

Constructors are passive. Per-device receive hooks timestamp *new telemetry*,
not polling calls. This module must be owned by the gateway process only.
"""

from __future__ import annotations

import copy
import dataclasses
import os
import re
import threading
import time
from collections import deque

import numpy as np

from .hexfellow_control_console import (
    _load_vendor_sdk,
    _parking_stop_report,
    _wait_for_device,
    build_endpoint_specs,
)
from .whole_body import SLICES, STAGES, WholeBodyState, vector
from .x7 import HardwareSafetyError, ControlDelay


class TelemetryTap:
    """Instance-local receive hook, pinned alongside the SDK ingress hook."""

    def __init__(self, name, device, clock=time.monotonic, ingress=None, recorder=None):
        self.name, self.device, self.clock = name, device, clock
        self.lock = threading.Lock()
        self.value = None
        self.error = None
        self.feedback_failure = None
        self.sequence = 0
        self.duplicates = 0
        self.ingress = ingress
        self.recorder = recorder
        from .timing_stats import LatencyStats
        self.latencies = LatencyStats()
        from .transport_diagnostics import FeedbackGapState
        self.gap_state = FeedbackGapState() if os.environ.get('POLICY_SIM_LIFECYCLE_DIAGNOSTICS') == '1' else None
        self.on_update = None
        self.frames = deque(maxlen=1024)
        self.original = device._update
        device._update = self.update

    def update(self, message, timestamp):
        received = self.clock()
        accepted = self.original(message, timestamp)
        if not accepted:
            return accepted
        with self.lock:
            if self.error is not None:
                return accepted  # malformed/regressing telemetry requires reconnect
        try:
            value = self.capture()
            value["received"] = received
            value["timestamp"] = timestamp.to_dict()
            if self.name == "theta":
                from .theta_velocity import device_nanoseconds
                raw = getattr(self.ingress, "raw_timestamp", None)
                value["device_time_ns"] = device_nanoseconds(raw)
                value["raw_timestamp"] = raw
            value["capture_completed"] = self.clock()
            value["sdk_message_entered"] = getattr(self.ingress, "entered", received)
            value["udp_received_at"] = getattr(self.ingress, "udp_received_at", None)
            value["kcp_process_entered"] = getattr(self.ingress, "kcp_process_entered", None)
            value["udp_ingress"] = getattr(self.ingress, "udp_ingress", None)
            with self.lock:
                if self.value is not None:
                    previous = self.value["timestamp"]
                    now_key = (value["timestamp"]["s"], value["timestamp"]["ns"])
                    old_key = (previous["s"], previous["ns"])
                    if self.name == "theta":
                        now_key = value["device_time_ns"]
                        old_key = self.value["device_time_ns"]
                    if now_key == old_key:
                        self.duplicates += 1
                        fields = ("odom", "body_velocity") if self.name == "base" else (
                            ("pos",) if self.name == "theta" else ("pos", "vel"))
                        if any(not np.array_equal(self.value.get(k), value.get(k)) for k in fields):
                            self.error = f"{self.name} repeated device timestamp has changed raw feedback"
                            self.feedback_failure = {
                                "kind": "raw_feedback_changed", "reason": self.error, "observed_at": received,
                                "devices": {self.name: {
                                    "previous_raw": {
                                        k: self.value[k] for k in (*fields, "timestamp", "received", "sequence")
                                        if k in self.value
                                    },
                                    "current_raw": {
                                        k: value[k] for k in (*fields, "timestamp", "received") if k in value
                                    },
                                    "delta_raw": {k: (np.asarray(value[k]) - self.value[k]).tolist() for k in fields},
                                }},
                            }
                        return accepted
                    if now_key < old_key:
                        raise ValueError(f"{self.name} telemetry timestamp regressed")
                if self.value is not None:
                    self.latencies.add("new_frame_interval", received - self.value["received"])
                self.latencies.add("capture", value["capture_completed"] - received)
                if value.get("udp_received_at") is not None:
                    self.latencies.add("udp_to_tap", received - value["udp_received_at"])
                ingress = value.get("udp_ingress") or {}
                if ingress.get("kernel_monotonic_estimate") is not None:
                    self.latencies.add("kernel_to_receive_estimate", ingress["udp_received_at"] - ingress["kernel_monotonic_estimate"])
                if self.name == "theta":
                    from .theta_velocity import estimate_frame
                    value.update(estimate_frame(value, self.value))
                self.sequence += 1
                if self.gap_state is not None:
                    self.gap_state.observe(received)
                value["sequence"] = self.sequence
                self.value = value
                self.frames.append({"sequence": self.sequence, "received": received,
                    **{k: value[k] for k in ("pos", "device_time_ns", "device_dt_ns", "receive_dt_s",
                                            "velocity", "velocity_valid") if k in value},
                    "capture_completed": value["capture_completed"],
                    "raw_timestamp": getattr(self.ingress, "raw_timestamp", None),
                    "udp_received_at": value["udp_received_at"],
                    "api_entered": value["sdk_message_entered"],
                    "kcp_process_entered": value["kcp_process_entered"]})
                self.error = None
            if self.recorder is not None:
                self.recorder.record(f"{self.name}_frame", sequence=self.sequence, received=received,
                                     completed=value["capture_completed"],
                                     entered=value["sdk_message_entered"],
                                     raw_timestamp=getattr(self.ingress, "raw_timestamp", None),
                                     session_id=getattr(self.ingress, "session_id", None),
                                     sdk_timestamp=value["timestamp"],
                                     measurement={k: value[k] for k in ("pos", "device_time_ns", "device_dt_ns",
                                                  "receive_dt_s", "velocity", "velocity_valid") if k in value})
        except Exception as error:
            with self.lock:
                self.error = f"{self.name} telemetry: {error}"
                self.feedback_failure = {"kind": "invalid_telemetry", "reason": self.error,
                    "observed_at": received, "devices": {self.name: {
                        "previous_raw": copy.deepcopy(self.value),
                        "current_raw": {**locals().get("value", {}), "velocity_valid": False},
                        "raw_timestamp": getattr(self.ingress, "raw_timestamp", None)}}}
            if self.recorder is not None:
                self.recorder.record(f"{self.name}_rejected", failure=self.feedback_failure)
        if self.on_update is not None:
            self.on_update()
        return accepted

    def capture(self):
        device = self.device
        common = {"parking_stop": _parking_stop_report(device.get_parking_stop_detail())}
        if self.name == "theta":
            # get_motor_positions() clears the SDK 'new data' flag. Read exactly
            # once per update, never synthesize freshness from getter success.
            common.update(
                pos=float(device.get_motor_positions()),
                range=list(device.get_pos_range()),
                pulse_per_meter=float(device.get_pulse_per_meter()),
                configured_speed=float(device.get_move_speed()),
                maximum_speed=float(device.get_max_move_speed()),
                state=device.get_state(), calibrated=bool(device._calibrated),
            )
        else:
            status = device.get_simple_motor_status(pop=False)
            if status is None:
                raise HardwareSafetyError("missing motor status")
            common.update(
                pos=status["pos"], vel=status["vel"], motor_states=device.get_motor_states(),
                session_holder=int(device.get_session_holder()), mine=int(device.get_my_session_id()),
                initialized=bool(device._api_control_initialized),
            )
            if self.name in ("left", "right"):
                # These SDK getters expose cached values without a temperature timestamp.
                common["temperature_freshness"] = "unknown_cached_sdk_value"
                for field, getter in (("motor_temperature_c", "get_motor_temperatures"),
                                      ("driver_temperature_c", "get_motor_driver_temperatures")):
                    read = getattr(device, getter, None)
                    cached = None if read is None else read()
                    common[field] = (None if cached is None else
                                     [float(x) if np.isfinite(x) else None for x in cached])
                common["effort_device_timestamp"] = status.get("ts")
                effort = status.get("eff")
                common["effort_sdk_nm"] = (None if effort is None else
                                           vector(effort, 7, f"{self.name} effort feedback").tolist())
            if self.name == "base":
                common.update(odom=device.get_vehicle_position(pop=False),
                              body_velocity=device.get_vehicle_speed(pop=False),
                              state=device.get_base_state(), warning=device.get_warning())
                if hasattr(device, "get_battery_info"):
                    common["battery"] = device.get_battery_info()
            else:
                common.update(calibrated=bool(device._calibrated),
                              protocol_version=list(getattr(device, "_MotorBase__proto_version", (0, 0))),
                              joint_names=list(device.get_joint_names()),
                              joint_limits=device.get_joint_limits(), joint_limits_source="sdk_static",
                              series=device.get_arm_series())
        return copy.deepcopy(common)

    def read(self):
        with self.lock:
            if self.error:
                error = HardwareSafetyError(self.error)
                if self.feedback_failure is not None:
                    error.feedback_failure = self.feedback_failure
                raise error
            if self.value is None:
                raise HardwareSafetyError(f"waiting for a new {self.name} telemetry frame")
            # Capture owns a detached snapshot; updates replace the entire row.
            # Readers never mutate its nested values. Avoid copying four complete
            # metadata trees for every gateway wake.
            value = dict(self.value)
            value["duplicate_frames"] = self.duplicates
            return value


def install_ingress_clock(api, recorder=None):
    """Time parsed-message entry, NOT kernel arrival or sensor acquisition."""
    context = threading.local()
    original = api._process_api_up

    def capture_identity(message):
        if message.HasField("log"):
            match = re.search(r"Backend Git hash: ([0-9a-f]+), Build time: (\S+)", str(message.log))
            if match:
                api._policy_firmware = {"backend_git": match[1], "build_time": match[2],
                                        "source": "server_greeting"}

    # Connection startup can precede hook installation. Copy the bounded SDK
    # buffer without consuming data or inventing an identity from configuration.
    for message in list(getattr(api, "_HexDeviceApi__raw_data", ())):
        capture_identity(message)

    def process(message):
        capture_identity(message)
        context.entered = time.monotonic()
        transport = getattr(getattr(api, "_HexDeviceApi__kcp_client", None), "_policy_udp", None)
        context.udp_received_at = getattr(transport, "last_receive", None)
        context.kcp_process_entered = getattr(transport, "process_entered", None)
        context.udp_ingress = getattr(transport, "last_receive_row", None)
        from .transport_diagnostics import raw_timestamp
        context.raw_timestamp = raw_timestamp(message)
        context.session_id = int(getattr(message, "session_id", 0))
        if recorder is not None:
            recorder.record("api_up", entered=context.entered,
                            theta=message.HasField("linear_lift_status"),
                            raw_timestamp=context.raw_timestamp)
        return original(message)

    api._process_api_up = process
    return context


def passive_api_close(api):
    """SDK close() otherwise sends stop() even for sessions it never acquired.

    Owned sessions are explicitly stopped and checked *before* reaching here.
    Suppress the SDK's indiscriminate stop calls on this closing instance only.
    """
    devices = [*api._internal_device_list, *api._optional_device_list]
    saved = []
    for device in devices:
        if hasattr(device, "stop"):
            saved.append((device, device.stop))
            device.stop = lambda: None
    try:
        api.close()
    finally:
        for device, stop in saved:
            device.stop = stop


class HexWholeBodyEndpoint:
    def __init__(self, mapping, *, timeout=5.0, read_only=False, transport_options=None, recorder=None,
                 downlink_gate_factory=None):
        if downlink_gate_factory is not None and (read_only or not mapping.direct_quest):
            raise HardwareSafetyError("custom downlink requires a live direct endpoint")
        options = transport_options or {}
        if options and not read_only:
            raise HardwareSafetyError("Theta transport experiments are read-only")
        if (options.get("theta_transport", "kcp") not in {"kcp", "websocket"}
                or options.get("theta_report_hz", 250) not in {100, 250}):
            raise ValueError("unsupported Theta transport experiment")
        self.mapping, self.timeout = mapping, timeout
        mapping.validate_experiment_options()
        self.previous_feedback = None
        self.feedback_failure = None
        self.apis, self.devices, self.taps = [], {}, {}
        self.owned = set()
        self.origin = None
        self.observed_sockets = {}
        self.observed_kcp = {}
        self.unwrapped_yaw = None
        self.retreat_active = False
        self.retreat_ceiling = None
        self.write_count = 0
        self.dispatch = None
        self.direct_gates = {}
        api_type, self.command_type, robot_types = _load_vendor_sdk()
        from .sdk_transport import immediate_api_type
        api_type = immediate_api_type(api_type)
        if read_only:
            from .readonly_sdk import readonly_api_type

            api_type = readonly_api_type(api_type)
        from hex_device.motor_base import MitMotorCommand

        self.mit_type = MitMotorCommand
        data = mapping.data
        self.specs = build_endpoint_specs(data["arm_controller_ip"], data["theta_controller_ip"],
                                         data["base_controller_ip"], data["interface"])
        try:
            for spec in self.specs:
                # 250 Hz transport polling also for base/lift; this does not
                # increase the 50 Hz IK reference rate or firmware state rate.
                theta = spec.name == "theta"
                use_kcp = not theta or options.get("theta_transport", "kcp") == "kcp"
                connection_type = api_type
                if not read_only:
                    from .direct_quest import gated_api_type
                    from .mit_wire import MitWireGate

                    gate = (downlink_gate_factory or MitWireGate)(spec.name)
                    gate.checked_transport = isinstance(gate, MitWireGate)
                    gate.hold_resume = mapping.latency_policy == "hold-resume"
                    self.direct_gates[spec.name] = gate
                    connection_type = gated_api_type(api_type, gate)
                from .transport_diagnostics import DeviceRecorder
                device_recorder = DeviceRecorder(recorder, spec.name) if recorder is not None else None
                if (read_only and device_recorder is not None
                        and os.environ.get('POLICY_SIM_LIFECYCLE_DIAGNOSTICS') == '1'):
                    from .transport_diagnostics import lifecycle_api_type
                    connection_type = lifecycle_api_type(connection_type, device_recorder)
                api = connection_type(ws_url=spec.url, control_hz=250, enable_kcp=use_kcp, local_port=0)
                self.apis.append(api)
                ingress = install_ingress_clock(api, device_recorder)
                device = _wait_for_device(api, robot_types[spec.expected_robot_type], timeout,
                                          endpoint_name=spec.name)
                from .sdk_transport import install_checked_udp
                install_checked_udp(api, self.direct_gates.get(spec.name))
                self.devices[spec.name] = device
                self.taps[spec.name] = TelemetryTap(spec.name, device, ingress=ingress, recorder=device_recorder)
                if theta and options.get("theta_report_hz", 250) != 250:
                    import asyncio

                    from hex_device.generated import public_api_types_pb2
                    frequency = getattr(public_api_types_pb2, f"Rf{options['theta_report_hz']}Hz")
                    future = asyncio.run_coroutine_threadsafe(
                        api.reset_report_frequency(frequency), api._HexDeviceApi__loop)
                    future.result(timeout=timeout)
                if device_recorder is not None and use_kcp:
                    from .transport_diagnostics import instrument_kcp
                    instrument_kcp(api, device_recorder)
                    client = getattr(api, '_HexDeviceApi__kcp_client', None)
                    device_recorder.record('connection_ready',
                        instance=getattr(api, 'lifecycle_id', None),
                        local_port=client.get_local_port(), server_port=client.server_port,
                        realtime_ns=time.time_ns())
            deadline = time.monotonic() + timeout
            while True:
                try:
                    self.state()
                    break
                except HardwareSafetyError:
                    if time.monotonic() >= deadline:
                        raise
                    time.sleep(0.005)
        except BaseException as error:
            if self.feedback_failure is not None:
                error.feedback_failure = self.feedback_failure
            self.close()
            raise

    def check_transport(self):
        for gate in getattr(self, "direct_gates", {}).values():
            if gate.fault and gate.phase != "stopping":
                raise HardwareSafetyError(gate.fault)
        for api in self.apis:
            client = getattr(api, "_HexDeviceApi__kcp_client", None)
            generation = (client, getattr(client, "_kcp", None))
            if not hasattr(self, "observed_kcp"):
                self.observed_kcp = {}
            previous = self.observed_kcp.get(id(api))
            if previous is not None and generation != previous:
                raise HardwareSafetyError("controller KCP transport disconnected/reconnected")
            if client is not None:
                self.observed_kcp[id(api)] = generation
            transport = getattr(client, "_policy_udp", None)
            if transport is not None and transport.fault and not (
                    transport.gate is not None and transport.gate.phase == "stopping"):
                raise HardwareSafetyError(transport.fault)
            socket = getattr(api, "_HexDeviceApi__websocket", None)
            previous_socket = getattr(self, "observed_sockets", {}).get(id(api))
            if previous_socket is not None and socket is not previous_socket:
                raise HardwareSafetyError("controller transport disconnected/reconnected")
            if socket is not None:
                self.observed_sockets[id(api)] = socket
            if socket is not None and (getattr(socket, "closed", False) is True
                                       or getattr(getattr(socket, "state", None), "name", None) == "CLOSED"):
                raise HardwareSafetyError("controller transport disconnected")

    def diagnostic_window(self):
        """Bounded transport evidence; export outside the receive hot path."""
        result = {}
        for name, tap in self.taps.items():
            with tap.lock:
                frames = list(tap.frames)
            api = self.apis[list(self.taps).index(name)]
            client = getattr(api, '_HexDeviceApi__kcp_client', None)
            transport = getattr(client, '_policy_udp', None)
            result[name] = {"frames": frames,
                "udp": list(getattr(transport, 'receive_rows', ())),
                "udp_send": list(getattr(transport, 'rows', ())),
                "kcp_process": list(getattr(transport, 'process_rows', ())),
                "clock_note": "device deltas only; kernel-to-monotonic conversion is estimated"}
        return result

    def state(self):
        worker = getattr(self, "feedback_worker", None)
        if worker is not None:
            return worker.read()
        return self._build_state()

    def start_feedback_worker(self, seed):
        """Start only after acquisition has finished mutating runtime mapping."""
        from contextlib import nullcontext
        from .feedback_snapshot import FeedbackSnapshotWorker

        if getattr(self, "feedback_worker", None) is not None:
            raise HardwareSafetyError("feedback worker already started")
        worker = FeedbackSnapshotWorker(self._build_state, seed,
            context=getattr(self, "feedback_worker_context", nullcontext),
            notify=getattr(self, "feedback_notify", None))
        self.feedback_worker = worker
        worker.start()

    def stop_feedback_worker(self):
        worker = getattr(self, "feedback_worker", None)
        if worker is not None:
            worker.close()
            self.feedback_worker_report = worker.report()
            # No synchronous builder may run until the background owner exits.
            self.feedback_worker = None

    def _build_state(self):
        self.check_transport()
        snapshot_started = time.monotonic()
        if any(getattr(api, "blocked_control_messages", 0) for api in self.apis):
            raise HardwareSafetyError("read-only downlink blocked an unexpected SDK control message")
        rows = {name: tap.read() for name, tap in self.taps.items()}
        for spec, api in zip(getattr(self, "specs", ()), self.apis):
            if spec.name in rows:
                rows[spec.name]["firmware"] = dict(getattr(api, "_policy_firmware", {"source": "unreported"}))
        if set(rows) != set(SLICES):
            raise HardwareSafetyError("incomplete robot feedback")
        q, v = np.zeros(18), np.zeros(18)
        for side in ("left", "right"):
            row = rows[side]
            if row["series"] != 29 or row["joint_names"] != self.mapping.data["arms"][side]["sdk_joint_names"]:
                raise HardwareSafetyError(f"{side} SDK identity/joint order mismatch")
            q[SLICES[side]], v[SLICES[side]] = self.mapping.arm_state(side, row["pos"], row["vel"])
            effort = row.get("effort_sdk_nm")
            row["effort_model_nm"] = (None if effort is None else self.mapping.arm_effort(side, effort).tolist())
        lift = rows["theta"]
        q[3] = self.mapping.lift_position(lift["pos"], lift["range"])
        if not lift.get("velocity_valid", False):
            raise ControlDelay("Theta velocity awaiting two consecutive valid device reports")
        v[3] = self.mapping.lift_velocity(lift["velocity"])
        odom = vector(rows["base"]["odom"], 3, "base odometry")
        speed = vector(rows["base"]["body_velocity"], 3, "base measured body velocity")
        old = self.previous_feedback
        base = rows["base"]
        old_base = None if old is None else old.health["base"]
        identity = ("sequence", "received", "timestamp")
        repeated = old_base is not None and any(base[k] == old_base[k] for k in identity)
        if repeated:
            if (not all(base[k] == old_base[k] for k in identity)
                    or not np.array_equal(odom, old_base["odom"])
                    or not np.array_equal(speed, old_base["body_velocity"])):
                reason = "base repeated frame identity/raw feedback changed"
                self.record_feedback_failure(reason, rows, kind="raw_feedback_changed")
                raise HardwareSafetyError(reason)
            # Reapplying angular unwrap can change the last bit even for identical
            # telemetry. One frame owns one transform, including world velocity.
            q[:3], v[:3] = old.q[:3], old.v[:3]
        else:
            if self.origin is None:
                self.origin = odom.copy()
            yaw = float(np.arctan2(np.sin(odom[2] - self.origin[2]), np.cos(odom[2] - self.origin[2])))
            c, s = np.cos(self.origin[2]), np.sin(self.origin[2])
            q[:2] = self.mapping.rotation() @ np.array([[c, s], [-s, c]]) @ (odom[:2] - self.origin[:2])
            if self.mapping.runtime_session:
                previous_yaw = getattr(self, "unwrapped_yaw", None)
                if previous_yaw is not None:
                    yaw = previous_yaw + np.arctan2(np.sin(yaw - previous_yaw), np.cos(yaw - previous_yaw))
            q[2] = yaw
            c, s = np.cos(yaw), np.sin(yaw)
            v[:2] = np.array([[c, -s], [s, c]]) @ self.mapping.rotation() @ speed[:2]
            v[2] = speed[2]
        dispatch = self.dispatch
        if self.mapping.runtime_session:
            dispatch = {**(dispatch or {}), "wire": {name: getattr(gate, "last_dispatch", None)
                                                    for name, gate in self.direct_gates.items()},
                        "wire_first": {name: getattr(gate, "first_dispatch", None)
                                       for name, gate in self.direct_gates.items()}}
        result = WholeBodyState(q, v, {n: r["received"] for n, r in rows.items()},
                              {n: r["sequence"] for n, r in rows.items()}, rows,
                              {n: r["timestamp"] for n, r in rows.items()}, dispatch,
                              {"snapshot_started": snapshot_started,
                               "snapshot_completed": time.monotonic()}, self.mapping.feedback_policy,
                              self.mapping.latency_policy)
        from ..control.feedback_policy import validate_group_progress

        old = self.previous_feedback
        try:
            validate_group_progress(None if old is None else old.received, result.received,
                                    None if old is None else old.q, q,
                                    None if old is None else old.v, v,
                                    theta_dt_s=(None if old is None else
                                        (lift["device_time_ns"] - old.health["theta"]["device_time_ns"]) / 1e9),
                                    lift_speed=(getattr(self, "feedback_lift_speed",
                                                       self.mapping.data["hardware_lift_velocity_limit_m_s"])
                                                if self.mapping.feedback_policy
                                                in {"strict", "theta-record-only", "theta-bounded"} else .02))
        except ValueError as error:
            self.record_feedback_failure(str(error), rows, q=q, v=v, kind="derived_feedback_changed")
            raise HardwareSafetyError(str(error)) from error
        if self.mapping.runtime_session:
            self.unwrapped_yaw = float(q[2])
        self.previous_feedback = result
        return result

    def record_feedback_failure(self, reason, rows, *, kind, q=None, v=None):
        """Preserve the first offending pair; cleanup must not replace evidence."""
        if getattr(self, "feedback_failure", None) is not None:
            return
        old = self.previous_feedback
        devices = {}
        for name, part in SLICES.items():
            keys = ("sequence", "received", "timestamp", "pos", "vel", "odom", "body_velocity")
            before = {} if old is None else old.health[name]
            row = {"previous_raw": {k: before[k] for k in keys if k in before},
                   "current_raw": {k: rows[name][k] for k in keys if k in rows[name]}}
            if old is not None:
                row.update(previous_q=old.q[part].tolist(), previous_v=old.v[part].tolist())
                if q is not None:
                    row.update(current_q=q[part].tolist(), current_v=v[part].tolist(),
                               delta_q=(q[part] - old.q[part]).tolist(), delta_v=(v[part] - old.v[part]).tolist())
                if name == "base":
                    row["delta_odom"] = (np.asarray(rows[name]["odom"]) - before["odom"]).tolist()
                    row["delta_body_velocity"] = (np.asarray(rows[name]["body_velocity"])
                                                   - before["body_velocity"]).tolist()
            devices[name] = row
        self.feedback_failure = {"kind": kind, "reason": reason, "observed_at": time.monotonic(), "devices": devices}

    def check_health(self, state, stage, *, acquiring=False, check_timing=True):
        for name in STAGES[stage]:
            gate = getattr(self, "direct_gates", {}).get(name)
            if gate is not None and gate.fault:
                raise HardwareSafetyError(f"{name} downlink: {gate.fault}")
        state.validate(time.monotonic(), check_skew=False, check_age=check_timing)
        if getattr(self, "mapping", None) is not None and self.mapping.direct_quest:
            from .direct_quest import check_power

            check_power(state, acquiring=acquiring or not self.owned)
        for name, row in state.health.items():
            if row["parking_stop"] is not None:
                raise HardwareSafetyError(f"{name} parking/emergency stop is active")
            if name == "theta":
                if row['state'] == 'LsEmergencyStop':
                    raise HardwareSafetyError('theta parking/emergency stop is active')
                if not row["calibrated"] or row["state"] not in {"LsBrake", "LsAlgorithmControl"}:
                    raise HardwareSafetyError("Theta must be calibrated and controllable; automatic homing forbidden")
            else:
                states = row["motor_states"]
                if not states or any(item != "normal" for item in states):
                    raise HardwareSafetyError(f"{name} motor health unavailable or abnormal")
                if name in STAGES[stage] and name != "base" and not row["calibrated"] and not (
                    acquiring and (self.mapping.effective_review.get("sdk_startup_checked") is True
                                   or self.mapping.commissioning_stage == stage or self.mapping.demo_authorized)
                ):
                    raise HardwareSafetyError(
                        f"{name} is not calibrated in this API session; "
                        "this does not prove an incorrect mechanical zero. "
                        "start() may send calibrate; live startup semantics remain unreviewed, refusing")
                if name == "base" and row["warning"] is not None:
                    raise HardwareSafetyError(f"base warning: {row['warning']}")
                if name in self.owned and not acquiring:
                    if row["mine"] == 0 or row["mine"] != row["session_holder"] or not row["initialized"]:
                        raise HardwareSafetyError(f"{name} control session lost")
                elif name not in self.owned and row["session_holder"] != 0:
                    raise HardwareSafetyError(f"{name} is controlled by another session")

        # Check group coherence only after all non-retryable hardware faults.
        state.validate(time.monotonic(), check_skew=check_timing, check_age=check_timing)

    def initialize_lift_retreat(self, state):
        """Arm the same inward-only publisher guard for real and simulated devices."""
        self.retreat_active = False
        self.retreat_ceiling = None
        if self.mapping.runtime_session:
            high = max(self.mapping.lift_position(x, state.health["theta"]["range"])
                       for x in state.health["theta"]["range"])
            self.retreat_active = state.q[3] > high - .002
            self.retreat_ceiling = state.q[3]

    def acquire(self, stage, profile):
        self.initialization_checkpoints = []

        def checkpoint(name):
            self.initialization_checkpoints.append(name)
            if self.mapping.initialization_abort_after == name:
                raise HardwareSafetyError(f"initialization cancelled at {name}")

        def check_cancelled():
            if getattr(self, "cancel_requested", lambda: False)():
                raise HardwareSafetyError("initialization cancelled; stopping acquired devices")

        check_cancelled()
        state = self.state()
        self.check_health(state, stage, acquiring=True)
        for side in ("left", "right"):
            if side in STAGES[stage] and tuple(getattr(
                self.devices[side], "_MotorBase__proto_version", (0, 0)
            )) < (1, 4):
                raise HardwareSafetyError("MIT deployment requires protocol >=1.4 radians")
        if self.mapping.direct_quest:
            from .direct_quest import check_start_position

            check_start_position(state, allow_lift_retreat=self.mapping.runtime_session)
            self.initialize_lift_retreat(state)
        if self.mapping.feedback_policy != "strict" and time.monotonic() - state.received["theta"] > .04:
            raise HardwareSafetyError("acquisition requires a fresh Theta frame within 40 ms")
        initial_q = state.q.copy()
        if np.max(np.abs(state.v[4:])) > 0.01 or np.max(np.abs(state.v[:4])) > 0.002:
            raise HardwareSafetyError("robot must be stationary before acquisition")
        # Configure Theta once, BEFORE any arms/base acquire. SDK deliberately
        # sleeps 0.5s after a speed update; never do this in the control loop.
        if "theta" in STAGES[stage]:
            lift = state.health["theta"]
            from ..control.robot import LIFT_JOINT
            lift_velocity = self.mapping.runtime_limits(profile, state)["runtime_velocity_bounds"][LIFT_JOINT]
            self.feedback_lift_speed = lift_velocity
            self.owned.add("theta")
            if "theta" in getattr(self, "direct_gates", {}):
                self.direct_gates["theta"].transition("initializing")
            speed_pulses = int(round(lift_velocity * lift["pulse_per_meter"]))  # runtime_limits already floors
            if speed_pulses <= 0:
                raise HardwareSafetyError("Theta speed cannot be represented in integer pulses")
            self.devices["theta"].set_move_speed(speed_pulses)
            for _ in range(60):
                check_cancelled()
                time.sleep(0.01)
            state = self.state()
            configured = state.health["theta"]["configured_speed"]
            if configured != speed_pulses:
                raise HardwareSafetyError("Theta speed configuration was not acknowledged")
            checkpoint("theta-speed")
        for name in STAGES[stage]:
            check_cancelled()
            if name == "theta":
                continue
            self.check_health(self.state(), stage, acquiring=True)
            self.owned.add(name)  # cleanup also covers a partially acquired session
            if name in getattr(self, "direct_gates", {}):
                self.direct_gates[name].transition("initializing")
            if name == "base":
                self.devices[name].set_vehicle_speed(0.0, 0.0, 0.0)
            else:
                self.devices[name].command_timeout_check(True)
                self.devices[name].motor_command(self.command_type.BRAKE, [True] * 7)
            self.devices[name].start()
        deadline = time.monotonic() + self.timeout
        while True:
            check_cancelled()
            state = self.state()
            if np.max(np.abs(state.q - initial_q)) > 0.002:
                raise HardwareSafetyError("robot pose changed during API initialization")
            try:
                self.check_health(state, stage)
                from ..control.reference_initialization import stationary_reference_velocity

                try:
                    stationary_reference_velocity(state.v)
                except ValueError as error:
                    raise HardwareSafetyError(str(error)) from error
                # All requested devices are acknowledged before testing the cancellation boundary.
                label = "base-ready" if "base" in STAGES[stage] else f"{STAGES[stage][-1]}-ready"
                if label not in self.initialization_checkpoints:
                    checkpoint(label)
                if self.mapping.direct_quest:
                    from .direct_quest import check_power

                    check_power(state, acquiring=True)
                for name in self.owned:
                    gate = getattr(self, "direct_gates", {}).get(name)
                    if gate is not None:
                        gate.transition("active")
                return state
            except HardwareSafetyError as error:
                if "initialization cancelled" in str(error):
                    raise
                if time.monotonic() >= deadline:
                    raise
                time.sleep(0.004)

    def publish(self, command, stage):
        state = self.state()
        self.check_health(state, stage)
        return self._publish_packets(command, stage, state)

    def publish_validated(self, command, stage, state):
        # Supervisor has validated this immutable snapshot once. Recheck clocks,
        # transport identity and asynchronous faults, without recapturing it.
        self.check_transport()
        worker = getattr(self, "feedback_worker", None)
        if worker is not None:
            worker.read()  # Surface a builder fault latched after snapshot selection.
        state.validate(time.monotonic())
        health = {}
        for name, tap in self.taps.items():
            with tap.lock:
                if tap.error or tap.value is None:
                    raise HardwareSafetyError(tap.error or "missing telemetry")
                health[name] = tap.value  # Published rows are replaced, never mutated.
        # Ownership, stop flags and motor health may change after the snapshot.
        self.check_health(dataclasses.replace(state, health=health), stage)
        return self._publish_packets(command, stage, state)

    def _prepare_dispatch(self, command, stage, state):
        clock = getattr(self, "clock", time.monotonic)
        packets = {}
        if self.mapping.data.get("base_control_enabled") is False:
            if np.any(np.abs(command.base_body) > 1e-7) or np.any(np.abs(command.velocity[:3]) > 1e-7):
                raise HardwareSafetyError("base control is disabled: nonzero publisher command")
            # Copy the packet object; do not mutate the reference retained by the guard.
            command = dataclasses.replace(command, base_body=np.zeros(3))
        # Prevalidate the entire group, including real SDK joint/height bounds.
        for side in ("left", "right"):
            if side not in STAGES[stage]:
                continue
            fields = self.mapping.arm_command(side, command.mit)
            limits = np.asarray(state.health[side]["joint_limits"], dtype=float)
            if limits.shape[0] != 7 or limits.shape[1] < 2 or not np.all(np.isfinite(limits)):
                raise HardwareSafetyError("invalid SDK joint limits")
            if np.any(fields["position"] < limits[:, 0] + 0.012) or np.any(fields["position"] > limits[:, 1] - 0.012):
                raise HardwareSafetyError(f"{side} command exceeds SDK joint bounds")
            packets[side] = [self.mit_type(torque=float(fields["torque_ff"][i]), speed=float(fields["velocity"][i]),
                                          position=float(fields["position"][i]), kp=float(fields["kp"][i]), kd=float(fields["kd"][i]))
                             for i in range(7)]
        height = None
        if "theta" in STAGES[stage]:
            height = self.mapping.lift_command(command.positions[3], state.health["theta"]["range"])
            low, high = state.health["theta"]["range"]
            uncertainty = 0.0
            if self.mapping.feedback_policy != "strict":
                from ..control.feedback_policy import theta_uncertainty

                uncertainty = theta_uncertainty(
                    state.received, clock(), self.mapping.feedback_policy,
                    lift_speed=getattr(self, "feedback_lift_speed",
                                       self.mapping.data["hardware_lift_velocity_limit_m_s"]))
            if self.mapping.runtime_session and getattr(self, "retreat_active", False):
                if (command.control_phase != "lift-retreat" or not low <= height <= high - .002
                        or command.positions[3] > state.q[3] + 1e-8
                        or command.velocity[3] > 1e-9):
                    raise HardwareSafetyError("invalid upper-endpoint retreat command")
                self.retreat_ceiling = float(command.positions[3])
                if height >= low + .003 and state.health["theta"]["pos"] >= low + .003:
                    self.retreat_active = False
            elif not low + 0.002 + uncertainty <= height <= high - 0.002 - uncertainty:
                raise HardwareSafetyError("Theta command outside device bounds")
        from functools import partial
        from .mit_wire import MitWireGate

        dispatch = {}
        for name in STAGES[stage]:
            expected = None
            if isinstance(getattr(self, "direct_gates", {}).get(name), MitWireGate):
                from hex_device.generated.public_api_down_pb2 import APIDown
                expected = APIDown()
            if name in packets:
                if expected is not None:
                    targets = expected.arm_command.arm_exclusive_command.arm_api_control_command.motor_targets
                    for value in packets[name]:
                        target = targets.targets.add().mit_target
                        for key in ("torque", "speed", "position", "kp", "kd"):
                            setattr(target, key, float(getattr(value, key)))
                write = partial(self.devices[name].motor_command, self.command_type.MIT, packets[name])
            elif name == "theta":
                lift = self.devices[name]
                if expected is not None:
                    pulses = lift.convert_rad_to_positions(np.array([height]), lift._pulse_per_rotation)
                    expected.linear_lift_command.target_pos = int(np.sign(lift._max_pos) * pulses[0])
                write = partial(lift.motor_command, self.command_type.POSITION, height)
            else:
                xy = self.mapping.rotation().T @ command.base_body[:2]
                if expected is not None:
                    speed = expected.base_command.simple_move_command.xyz_speed
                    speed.speed_x, speed.speed_y = float(xy[0]), float(xy[1])
                    speed.speed_z = float(command.base_body[2])
                write = partial(self.devices[name].set_vehicle_speed,
                                float(xy[0]), float(xy[1]), float(command.base_body[2]))
            dispatch[name] = (expected, write)
        return dispatch

    def _publish_packets(self, command, stage, state):
        from .mit_wire import MitWireGate, authorize_group
        from .control_timing import feedback_deadline
        from ..control.feedback_policy import coupled_groups, received_deadline

        clock = getattr(self, "clock", time.monotonic)
        dispatch = self._prepare_dispatch(command, stage, state)
        entries = [(self.direct_gates[name], expected) for name, (expected, _) in dispatch.items()
                   if isinstance(getattr(self, "direct_gates", {}).get(name), MitWireGate)]
        authorize_group(entries, command.sequence, command.created,
                        min(command.expires, feedback_deadline(state),
                            received_deadline(command.state_received, self.mapping.feedback_policy)),
                        cycle_started=command.cycle_started,
                        source_received=min(command.state_received[n] for n in coupled_groups(self.mapping.feedback_policy)),
                        source_expires=received_deadline(command.state_received, self.mapping.feedback_policy))
        sent = {}
        for name, (_, write) in dispatch.items():
            if clock() >= command.expires:
                raise ControlDelay("command expired during group dispatch; partial group must stop")
            write()
            self.write_count += 1
            sent[name] = clock()
        for name in dispatch:
            wake = getattr(self.devices[name], "_policy_wake", None)
            if wake is not None:
                wake()
        self.dispatch = {"sequence": command.sequence, "sdk_enqueued_at": sent,
                         "group_skew_s": max(sent.values()) - min(sent.values())}

    def delay_hold(self, reason, epoch):
        """Brake without releasing ownership; no SDK start/stop or clear-stop."""
        for name in self.owned:
            gate = self.direct_gates[name]
            gate.delay_hold(reason)
            gate.recovery_epoch = epoch
        self._hold_refreshed_at = 0.
        self.maintain_delay_hold()

    def maintain_delay_hold(self):
        now = time.monotonic()
        if now - getattr(self, '_hold_refreshed_at', 0.) < .02:
            return
        self._hold_refreshed_at = now
        if "theta" in self.owned:
            lift = self.devices["theta"]
            with lift._status_lock:
                lift._target_pos = None
                lift._target_speed = None
        for name in self.owned:
            if name == "base":
                self.devices[name].set_vehicle_speed(0., 0., 0.)
            else:
                self.devices[name].motor_command(self.command_type.BRAKE, True if name == "theta" else [True] * 7)
            wake = getattr(self.devices[name], "_policy_wake", None)
            if wake is not None:
                wake()

    def delay_drained(self, epoch):
        for name in self.owned:
            api = self.apis[list(self.taps).index(name)]
            client = getattr(api, '_HexDeviceApi__kcp_client', None)
            transport = getattr(client, '_policy_udp', None)
            if transport is None or not transport.motion_drained() or not transport.hold_confirmed(epoch):
                return False
        return True

    def resume_delay(self, epoch):
        for name in self.owned:
            self.direct_gates[name].resume_delay(epoch)

    def stop(self):
        stop_requested = time.monotonic()
        progress = getattr(self, 'shutdown_progress', lambda *a, **k: None)
        progress('stop_commands', 'started')
        failures = []
        for name in self.owned:
            gate = getattr(self, "direct_gates", {}).get(name)
            if gate is not None:
                gate.transition("stopping")
        # SDK Theta can snapshot a pending target before observing BRAKE. Clear
        # unsent targets under its lock; the wire gate covers already snapshotted
        # targets by replacing them with BRAKE after this phase transition.
        if "theta" in self.owned:
            lift = self.devices["theta"]
            if hasattr(lift, "_status_lock"):
                with lift._status_lock:
                    lift._target_pos = None
                    lift._target_speed = None
                    lift._send_calibrate = None
        for name in tuple(self.owned):
            try:
                if name == "base":
                    self.devices[name].set_vehicle_speed(0.0, 0.0, 0.0)
                else:
                    self.devices[name].motor_command(self.command_type.BRAKE, True if name == "theta" else [True] * 7)
                wake = getattr(self.devices[name], "_policy_wake", None)
                if wake is not None:
                    wake()
            except Exception as error:
                failures.append(f"{name} brake: {error}")
        # Repeat base zero while SDK transport continues; then release owned sessions.
        for _ in range(13):
            if "base" in self.owned:
                try:
                    self.devices["base"].set_vehicle_speed(0.0, 0.0, 0.0)
                except Exception as error:
                    failures.append(f"base zero: {error}")
            if "theta" in self.owned:
                try:
                    self.devices["theta"].motor_command(self.command_type.BRAKE, True)
                except Exception as error:
                    failures.append(f"theta brake: {error}")
            time.sleep(0.02)
        for name in self.owned - {"theta"}:
            try:
                self.devices[name].stop()
            except Exception as error:
                failures.append(f"{name} release: {error}")
        # Brakes and releases precede joining a potentially delayed builder.
        # Stop confirmation then owns synchronous conversion history again.
        try:
            self.stop_feedback_worker()
        except Exception as error:
            failures.append(f"feedback worker: {error}")
        progress('stop_commands', 'completed')
        progress('stop_confirmation', 'started')
        deadline = time.monotonic() + 2.0
        pending = set(self.owned)
        self.stop_observations = {}
        self.stop_feedback_errors = []
        from .shutdown import ThetaObservedStop
        theta_observer = ThetaObservedStop()
        demo_stop = bool(getattr(getattr(self, 'mapping', None), 'demo_authorized', False))
        stable_since, last_received = {}, {}
        while pending and time.monotonic() < deadline:
            try:
                state = self.state()
                now = time.monotonic()
                if not demo_stop:
                    from ..control.feedback_policy import validate_received
                    validate_received(state.received, now, 'strict')
                for name in tuple(pending):
                    row = state.health[name]
                    received = state.received[name]
                    age_limit = .08 if name == 'theta' and demo_stop else .04
                    if not np.isfinite(received) or not 0 <= now-received <= age_limit:
                        stable_since.pop(name, None)
                        if name == 'theta':
                            theta_observer.reset()
                        self.stop_observations[name] = dict(continuous_stop_confirmed=False,
                            reason='stale_or_invalid_feedback', state=row.get('state'))
                        continue
                    if received <= max(stop_requested, last_received.get(name, stop_requested)):
                        if received < last_received.get(name, stop_requested):
                            stable_since.pop(name, None)
                            if name == 'theta':
                                theta_observer.reset()
                        continue
                    if received - last_received.get(name, received) > age_limit:
                        stable_since.pop(name, None)
                    last_received[name] = received
                    stopped = np.max(np.abs(state.v[SLICES[name]])) <= (0.002 if name in {"base", "theta"} else 0.01)
                    released = row["state"] == "LsBrake" if name == "theta" else row["session_holder"] == 0
                    armed_at = getattr(self, 'estop_armed_at', lambda: None)()
                    emergency = bool(armed_at is not None and received >= armed_at and (
                        (row.get('parking_stop') or {}).get('category') in {0, 8}
                        or (name == 'theta' and row.get('state') == 'LsEmergencyStop')))
                    if stopped and (released or emergency):
                        stable_since.setdefault(name, received)
                    else:
                        stable_since.pop(name, None)
                    stable_s = received - stable_since.get(name, received)
                    observed_stop = False
                    if name == 'theta' and demo_stop and not emergency:
                        transport = self.theta_stop_evidence(stop_requested)
                        healthy = (row.get('calibrated') is True
                            and set(state.health) == set(SLICES)
                            and all(h.get('parking_stop') is None and h.get('warning') is None
                                    and (n == 'theta' or bool(h.get('motor_states'))
                                         and all(s == 'normal' for s in h['motor_states']))
                                    for n, h in state.health.items())
                            and not any(g.fault for g in self.direct_gates.values()))
                        observed_stop = theta_observer.update(state, now, stop_requested, transport, healthy)
                    confirmed = observed_stop or bool(stopped and (released or emergency) and stable_s >= .1)
                    self.stop_observations[name] = {
                        "stable_duration_s": stable_s,
                        "continuous_stop_confirmed": confirmed,
                        "received_after_stop": state.received[name] > stop_requested,
                        "stationary": bool(stopped), "released_or_braked": bool(released),
                        "stopped_by_estop": emergency,
                        "state": row.get("state"), "session_holder": row.get("session_holder"),
                        "velocity": state.v[SLICES[name]].tolist(),
                    }
                    if name == 'theta':
                        self.stop_observations[name].update(firmware_brake_confirmed=bool(released),
                            acceptance='firmware_brake' if released else ('physical_estop' if emergency else 'unconfirmed'))
                        if demo_stop and not emergency:
                            self.stop_observations[name].update(theta_observer.details)
                            if released:
                                self.stop_observations[name]['acceptance'] = 'firmware_brake'
                    if confirmed:
                        pending.remove(name)
            except Exception as error:
                stable_since.clear()  # Missing/stale feedback breaks continuity.
                theta_observer.reset()
                if len(self.stop_feedback_errors) < 256:
                    self.stop_feedback_errors.append({"at": time.monotonic(),
                        "error": f"{type(error).__name__}: {error}"})
            time.sleep(0.01)
        failures.extend(f"{name}: stop not confirmed; use physical E-stop" for name in pending)
        progress('stop_confirmation', 'failed' if pending else 'completed', error='; '.join(failures) or None)
        self.owned.clear()
        return failures

    def theta_stop_evidence(self, requested):
        for api in getattr(self, 'apis', []):
            client = getattr(api, '_HexDeviceApi__kcp_client', None)
            transport = getattr(client, '_policy_udp', None)
            if transport is not None and transport.gate is self.direct_gates.get('theta'):
                return transport.stop_evidence(requested)
        return dict(brake_received=False, no_pending_motion=False)

    def close(self):
        failures = self.stop() if self.owned else []
        try:
            self.stop_feedback_worker()
        except Exception as error:
            failures.append(f"feedback worker: {error}")
        try:
            self.stop_window = self.diagnostic_window()
        except Exception as error:
            self.stop_window_error = str(error)
        for api in reversed(self.apis):
            progress = getattr(self, 'shutdown_progress', lambda *a, **k: None)
            index = self.apis.index(api)
            name = list(self.taps)[index] if index < len(self.taps) else str(index)
            progress('sdk_close', 'started', device=name)
            deadline = time.monotonic() + 6.
            try:
                # SDK 1.4.10 is_api_exit() tests loop.is_closed(), but its loop
                # thread exits run_until_complete() without calling loop.close().
                # Verify actual sender/receiver/loop threads instead.
                client = getattr(api, '_HexDeviceApi__kcp_client', None)
                threads = [getattr(api, '_HexDeviceApi__loop_thread', None)]
                threads += [getattr(client, key, None) for key in ('_update_thread', '_receive_thread')]
                passive_api_close(api)
                for thread in threads:
                    if thread is not None:
                        thread.join(timeout=max(0., deadline-time.monotonic()))
                if threads[0] is None or any(t is not None and t.is_alive() for t in threads):
                    raise HardwareSafetyError('SDK worker thread exit not confirmed within 6 seconds')
                progress('sdk_close', 'completed', device=name)
            except Exception as error:
                failures.append(f"{name} API close: {error}")
                progress('sdk_close', 'failed', device=name, error=str(error))
        self.apis.clear()
        return failures
