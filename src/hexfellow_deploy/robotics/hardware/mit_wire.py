"""Publisher-issued MIT wire leases; SDK serialization cannot renew authority.

20 ms admission TTL and 40 ms source lifetime are separate. These are host
protections, not firmware disconnect guarantees. POSITION replay is independent.
"""

import copy
import time
from collections import deque
from contextlib import ExitStack

from .direct_quest import DirectDownlinkGate
from .whole_body import COMMAND_TTL, COMMAND_WATCHDOG
from .x7 import ControlDelay, HardwareSafetyError


def motion_payload(name, data):
    if name in {"left", "right"} and data.WhichOneof("down") == "arm_command":
        arm = data.arm_command
        if arm.WhichOneof("command") == "arm_exclusive_command":
            command = arm.arm_exclusive_command
            if command.WhichOneof("exclusive_command") == "arm_api_control_command":
                control = command.arm_api_control_command
                if control.WhichOneof("command") == "motor_targets":
                    targets = control.motor_targets
                    if [t.WhichOneof("target") for t in targets.targets] == ["mit_target"] * 7:
                        return targets.SerializeToString()
    elif name == "theta" and data.WhichOneof("down") == "linear_lift_command":
        command = data.linear_lift_command
        if command.WhichOneof("command") == "target_pos":
            return command.SerializeToString()
    elif name == "base" and data.WhichOneof("down") == "base_command":
        command = data.base_command
        if command.WhichOneof("command") == "simple_move_command":
            move = command.simple_move_command
            if move.WhichOneof("command") == "xyz_speed":
                return move.xyz_speed.SerializeToString()
    return None


def brake_packet(name, data):
    result = copy.deepcopy(data)
    result.Clear()
    if name in {"left", "right"}:
        targets = result.arm_command.arm_exclusive_command.arm_api_control_command.motor_targets
        for _ in range(7):
            targets.targets.add(brake=True)
    elif name == "theta":
        result.linear_lift_command.brake = True
    else:
        result.base_command.simple_move_command.xyz_speed.SetInParent()
    return result


class MitWireGate(DirectDownlinkGate):
    def __init__(self, name, clock=time.monotonic):
        super().__init__(name)
        self.clock = clock
        self.sources = deque(maxlen=2)
        self.sequence = -1
        self.sent_sequence = -1
        self.deadline = 0.0
        self.startup_timing = {}
        self.first_dispatch_timing = deque(maxlen=512)
        self.first_dispatch = None
        self._recorded_dispatches = set()
        self.authorization_timing = deque(maxlen=2)
        from .timing_stats import LatencyStats
        self.latencies = LatencyStats()
        self.source_timing = {}
        self.hold_resume = False
        self.delay_reason = None
        self.recovery_epoch = 0
        self.stop_token = None
        self.counts = {"motion": 0, "replacement_brake": 0, "initialize": 0, "release": 0,
                       "calibrate": 0, "brake": 0}

    def delay_hold(self, reason):
        with self.lock:
            if self.phase == "stopping" or self.fault:
                return
            self.delay_reason = str(reason)
            self.phase = "holding"
            self.sources.clear()

    def resume_delay(self, epoch):
        with self.lock:
            if self.phase != "holding" or self.fault:
                raise HardwareSafetyError("wire recovery requires holding without a fault")
            self.sources.clear()
            self.sequence = self.sent_sequence = -1
            self._recorded_dispatches.clear()
            self.source_timing.clear()
            self.first_dispatch = None
            self.last_dispatch = None
            self.delay_reason = None
            self.recovery_epoch = epoch
            self.transition("active")

    def transition(self, phase):
        with self.lock:
            super().transition(phase)
            if phase == "initializing":
                self.deadline = self.clock() + 8.0
            elif phase == "active":
                now = self.clock()
                self.startup_timing["active_at"] = now
                self.deadline = now + COMMAND_WATCHDOG
            elif phase == "stopping":
                if self.stop_token is None:
                    self.stop_token = self.clock()
                self.sources.clear()

    def record_dispatch(self, data, started, completed, *, sequence):
        """SDK send-return timing, not delivery or physical execution acknowledgement."""
        payload = motion_payload(self.name, data)
        if payload is None:
            return
        with self.lock:
            source = next((s for s in reversed(self.sources) if s[0] == sequence and s[2] == payload), None)
            if source is not None:
                self.startup_timing.setdefault("first_sdk_send_started", started)
                self.startup_timing.setdefault("first_sdk_send_returned", completed)
                self.last_dispatch = {"sequence": source[0], "epoch": self.recovery_epoch,
                                      "created": source[4],
                                      "sdk_send_started": started, "sdk_send_returned": completed,
                                      "hardware_acknowledged": False,
                                      "udp": getattr(self, "last_udp_dispatch", None)}

                # This set contains only the current epoch; resume clears it.
                # Keep sequence keys compatible with the actual send path.
                if sequence not in self._recorded_dispatches:
                    if len(self.first_dispatch_timing) == self.first_dispatch_timing.maxlen:
                        old = self.first_dispatch_timing[0]
                        if old["epoch"] == self.recovery_epoch:
                            self._recorded_dispatches.discard(old["sequence"])
                    self._recorded_dispatches.add(sequence)
                    udp = getattr(self, "last_udp_dispatch", None)
                    cycle, received = self.source_timing.get(sequence, (None, None))
                    if udp is not None and udp.get("sequence") == sequence:
                        sent = udp["first_udp_send_started"]
                        if cycle is not None:
                            self.latencies.add("cycle_to_udp", sent - cycle)
                        if received is not None:
                            self.latencies.add("source_to_udp", sent - received)
                    authorized = next((r[1] for r in self.authorization_timing if r[0] == sequence), None)
                    self.first_dispatch = {**self.last_dispatch, "authorized_at": authorized}
                    self.first_dispatch_timing.append(self.first_dispatch)
                    self.latencies.add("command_to_first_sdk", started - self.first_dispatch["created"])
                    self.latencies.add("first_sdk_call", completed - started)

    def _validate_authorization(self, sequence, created, expires, payload, source_expires, now):
        import math

        if self.fault:
            raise HardwareSafetyError(f"{self.name} downlink: {self.fault}")
        if (getattr(self, "checked_transport", False) and self.sequence >= 0
                and self.sequence not in self._recorded_dispatches):
            raise ControlDelay("previous MIT reference has no confirmed first SDK dispatch")
        if self.hold_resume and self.phase == "holding":
            raise ControlDelay("wire is holding for delay recovery")
        if all(math.isfinite(v) for v in (created, expires, now)) and created <= expires <= now:
            raise ControlDelay("invalid or expired MIT publisher authorization")
        if (self.phase != "active" or type(sequence) is not int
                or sequence <= self.sequence or payload is None
                or not all(math.isfinite(v) for v in (created, expires, now))
                or not created <= now < expires <= created + COMMAND_TTL + 1e-9):
            raise HardwareSafetyError("invalid or expired MIT publisher authorization")
        if source_expires is not None and (not math.isfinite(source_expires) or source_expires < expires):
            raise HardwareSafetyError("invalid source feedback deadline")

    def _commit_authorization(self, sequence, created, expires, payload, now,
                              cycle_started, source_received, source_expires):
        self.authorization_timing.append((sequence, now))
        self.source_timing[sequence] = (cycle_started, source_received)
        for old in list(self.source_timing):
            if old < sequence - 1:
                del self.source_timing[old]
        self.sequence = sequence
        self.startup_timing.setdefault("first_authorized", now)
        self.deadline = min(created + COMMAND_WATCHDOG,
                            source_expires if source_expires is not None else float('inf'))
        self.sources.append((sequence, self.deadline, payload, expires, created))

    def authorize(self, sequence, created, expires, data, *, cycle_started=None, source_received=None,
                  source_expires=None):
        authorize_group([(self, data)], sequence, created, expires,
                        cycle_started=cycle_started, source_received=source_received,
                        source_expires=source_expires)

    def filter(self, data):
        with self.lock:
            now = self.clock()
            if self.phase in {"initializing", "active"} and now >= self.deadline:
                if self.hold_resume and self.phase == "active":
                    self.delay_hold("MIT publisher source lease expired")
                    return brake_packet(self.name, data) if motion_payload(self.name, data) is not None else super().filter(data)
                self.fault = self.fault or "MIT publisher source lease expired"
                self.startup_timing.setdefault("fault_observed_at", now)
                self.phase = "stopping"
                self.sources.clear()
            payload = motion_payload(self.name, data)
            if self.phase == 'stopping' and payload is not None:
                self.counts['replacement_brake'] += 1
                return brake_packet(self.name, data)
            if self.hold_resume and self.phase == "holding" and payload is not None:
                self.counts["replacement_brake"] += 1
                return brake_packet(self.name, data)
            if payload is not None and self.phase != "readonly":
                source = next((s for s in reversed(self.sources)
                               if now < s[1] and s[0] >= self.sent_sequence and s[2] == payload
                               and (s[0] == self.sent_sequence or now < s[3])), None)
                # An SDK loop can have copied the preceding target while the
                # publisher installs a new group. Never transmit an expired old
                # snapshot; discard it only while a newer validated reference is
                # still admissible. Unknown/regressing packets remain faults.
                if source is None and self.phase == "active" and not self.fault and self.sources:
                    latest = self.sources[-1]
                    superseded = any(s[2] == payload and s[0] == self.sent_sequence
                                     and s[0] < latest[0] and now >= s[1] for s in self.sources)
                    if superseded and now < min(latest[1], latest[3]):
                        self.counts.setdefault("superseded_snapshot", 0)
                        self.counts["superseded_snapshot"] += 1
                        return None
                # Zero base speed is also legal during initialization/stopping.
                zero_base = self.name == "base" and payload == b""
                if self.phase == "active" and not self.fault and source is not None:
                    self.sent_sequence = source[0]
                    self.counts["motion"] += 1
                elif not zero_base:
                    if self.hold_resume and self.phase == "active" and any(s[2] == payload for s in self.sources):
                        self.delay_hold("MIT wire source expired")
                        self.counts["replacement_brake"] += 1
                        return brake_packet(self.name, data)
                    self.fault = self.fault or "unauthorized, regressing or expired MIT wire packet"
                    self.startup_timing.setdefault("fault_observed_at", now)
                    self.phase = "stopping"
                    self.sources.clear()
                    self.counts["replacement_brake"] += 1
                    return brake_packet(self.name, data)
            allowed = super().filter(data)
            if self.fault:
                self.startup_timing.setdefault("fault_observed_at", now)
            if allowed is not None:
                if data.WhichOneof("down") == "arm_command":
                    command = data.arm_command.arm_exclusive_command
                    field = command.WhichOneof("exclusive_command")
                    if field == "api_control_initialize":
                        self.counts["initialize" if command.api_control_initialize else "release"] += 1
                    elif field == "calibrate":
                        self.counts["calibrate"] += 1
                if data == brake_packet(self.name, data):
                    self.counts["brake"] += 1
            return allowed


def authorize_group(entries, sequence, created, expires, *, cycle_started=None,
                    source_received=None, source_expires=None):
    """Install all wire leases before SDK setters can yield to periodic senders.

    No SDK locks, callbacks or I/O are taken under these gate locks. Validation
    failure changes no gate. Old payloads retain their original deadlines.
    """
    entries = list(entries)
    if len({id(gate) for gate, _ in entries}) != len(entries):
        raise HardwareSafetyError("duplicate gate in MIT authorization group")
    prepared = [(gate, motion_payload(gate.name, data)) for gate, data in entries]
    with ExitStack() as locks:
        for gate, _ in sorted(prepared, key=lambda entry: (entry[0].name, id(entry[0]))):
            locks.enter_context(gate.lock)
        for gate, payload in prepared:
            gate._validate_authorization(sequence, created, expires, payload, source_expires, gate.clock())
        # Include time spent validating the whole group in admission checks.
        if any(gate.clock() >= expires for gate, _ in prepared):
            raise ControlDelay("command expired during group authorization")
        for gate, payload in prepared:
            gate._commit_authorization(sequence, created, expires, payload, gate.clock(),
                                       cycle_started, source_received, source_expires)


async def send_checked(api, data):
    """Pinned SDK 1.4.10 framing, propagating its otherwise swallowed send failures.

    A successful KCP enqueue still is not a controller acknowledgement.
    """
    import inspect

    from hex_device.hex_device_api import (
        CURRENT_PROTOCOL_MAJOR_VERSION,
        CURRENT_PROTOCOL_MINOR_VERSION,
        HexSocketOpcode,
        HexSocketParser,
    )

    data.protocol_major_version = CURRENT_PROTOCOL_MAJOR_VERSION
    data.protocol_minor_version = CURRENT_PROTOCOL_MINOR_VERSION
    message = data.SerializeToString()
    if getattr(api, '_stream_mode', False) and not getattr(api, '_policy_force_kcp', False):
        result = api._send_down_callback(message)
        if inspect.isawaitable(result):
            result = await result
        if result is False:
            raise HardwareSafetyError('SDK stream send rejected')
        return
    kcp = getattr(api, '_HexDeviceApi__kcp_client', None)
    if api.enable_kcp and kcp is not None:
        from .sdk_transport import install_checked_udp
        adapter = install_checked_udp(api, getattr(api, "_policy_gate", None))
        frame = HexSocketParser.create_header(message, HexSocketOpcode.Binary)
        from contextlib import nullcontext
        gate = getattr(api, "_policy_gate", None)
        with kcp._kcp_lock, (gate.lock if gate is not None else nullcontext()):
            context = None
            if gate is not None and gate.phase == 'stopping':
                # A periodic SDK task may have serialized a target before stop.
                if motion_payload(gate.name, data) is not None:
                    data = brake_packet(gate.name, data)
                    data.protocol_major_version = CURRENT_PROTOCOL_MAJOR_VERSION
                    data.protocol_minor_version = CURRENT_PROTOCOL_MINOR_VERSION
                    frame = HexSocketParser.create_header(data.SerializeToString(), HexSocketOpcode.Binary)
                stop_packet = brake_packet(gate.name, data)
                stop_packet.protocol_major_version = CURRENT_PROTOCOL_MAJOR_VERSION
                stop_packet.protocol_minor_version = CURRENT_PROTOCOL_MINOR_VERSION
                if data == stop_packet and gate.stop_token is not None:
                    context = dict(sequence=-1, deadline=float('inf'), lease_deadline=float('inf'),
                                   recovery_epoch=gate.recovery_epoch, hold=True, stop_token=gate.stop_token)
            if gate is not None and gate.hold_resume and gate.phase == "holding":
                # Quarantine SDK snapshots taken before the gateway entered hold.
                if motion_payload(gate.name, data) is not None:
                    data = brake_packet(gate.name, data)
                    data.protocol_major_version = CURRENT_PROTOCOL_MAJOR_VERSION
                    data.protocol_minor_version = CURRENT_PROTOCOL_MINOR_VERSION
                    frame = HexSocketParser.create_header(data.SerializeToString(), HexSocketOpcode.Binary)
                hold_packet = brake_packet(gate.name, data)
                hold_packet.protocol_major_version = CURRENT_PROTOCOL_MAJOR_VERSION
                hold_packet.protocol_minor_version = CURRENT_PROTOCOL_MINOR_VERSION
                if data == hold_packet or motion_payload(gate.name, data) == b'':
                    if adapter.hold_pending(gate.recovery_epoch):
                        return  # The KCP update task retransmits; never build an unbounded hold queue.
                    context = dict(sequence=-1, deadline=float('inf'), lease_deadline=float('inf'),
                                   recovery_epoch=gate.recovery_epoch, hold=True)
            if gate is not None and motion_payload(gate.name, data) is not None and gate.phase == "active":
                with gate.lock:
                    source = next((s for s in reversed(gate.sources)
                                   if s[0] == gate.sent_sequence and s[2] == motion_payload(gate.name, data)), None)
                if source is None:
                    if gate.name != 'base' or motion_payload('base', data) != b'':
                        raise HardwareSafetyError('missing wire source')
                else:
                    first = source[0] not in gate._recorded_dispatches
                    context = dict(sequence=source[0], deadline=source[3] if first else source[1], lease_deadline=source[1],
                                   recovery_epoch=gate.recovery_epoch)
            adapter.context = context
            try:
                adapter.enqueue_context(frame, context)
                if kcp.send(frame) is False or adapter.fault:
                    raise HardwareSafetyError(adapter.fault or 'SDK KCP enqueue rejected')
                if context is not None and 'first_udp_send_started' not in context and not (
                        gate.phase == 'stopping' or gate.hold_resume and gate.phase == 'holding'):
                    if gate.hold_resume:
                        gate.delay_hold('KCP deferred first motion flush')
                        return
                    raise HardwareSafetyError('KCP did not flush motion before returning')
                if context is not None and not context.get('hold'):
                    gate.last_udp_dispatch = dict(context)
            finally:
                adapter.context = None
    else:
        websocket = getattr(api, '_HexDeviceApi__websocket', None)
        if websocket is None:
            raise HardwareSafetyError('SDK transport unavailable for send')
        await websocket.send(message)
