"""Local-only SDK/KCP sink for read-only timing audits; never addresses a robot.

This exercises serialization, four SDK KCP clients, their receive/update threads
and UDP submission. It is not a firmware emulator or a mechanical qualification.
"""
import asyncio
import time
from types import SimpleNamespace

import numpy as np

from .mit_wire import MitWireGate, send_checked
from .sdk_transport import install_checked_udp
from .x7 import HardwareSafetyError


class LoopbackSink:
    def __init__(self):
        from hex_device.kcp_client_core import KCPClient
        from hex_device.motor_base import MotorBase
        self.clients, self.apis, self.gates = [], {}, {}
        self.received = dict.fromkeys(('left', 'right', 'theta', 'base'), 0)
        self.loop = asyncio.new_event_loop()
        self.serializer = MotorBase.__new__(MotorBase)
        self.serializer._MotorBase__proto_version = (1, 6)
        try:
            for index, name in enumerate(self.received):
                sender, receiver = KCPClient(), KCPClient()
                self.clients.extend((sender, receiver))
                sender.config_kcp('::1', receiver.local_port, 700 + index)
                receiver.config_kcp('::1', sender.local_port, 700 + index)
                receiver.set_message_callback(lambda data, n=name: self._receive(n, data))
                sender.set_message_callback(lambda data: None)
                gate = MitWireGate(name)
                api = SimpleNamespace(enable_kcp=True, _HexDeviceApi__kcp_client=sender, _policy_gate=gate)
                install_checked_udp(api, gate)
                self.apis[name], self.gates[name] = api, gate
                receiver.start()
                sender.start()
        except BaseException:
            self.close()
            raise

    def _receive(self, name, data):
        self.received[name] += 1

    def send(self, candidate, sequence, created, deadline):
        from hex_device.generated.public_api_down_pb2 import APIDown
        from hex_device.motor_base import MitMotorCommand, MotorCommand
        times = {}
        for name, api in self.apis.items():
            gate = self.gates[name]
            # Each read-only cycle is an independent counterfactual. Nothing sent
            # to this sink is assumed to have changed the observed robot posture.
            gate.fault = None
            gate.transition('active')
            packet = APIDown()
            if name in ('left', 'right'):
                offset = 0 if name == 'left' else 7
                mit = candidate.mit
                values = [MitMotorCommand(position=float(mit.position[i]), speed=float(mit.velocity[i]),
                    torque=float(mit.torque_ff[i]), kp=float(mit.kp[i]), kd=float(mit.kd[i]))
                    for i in range(offset, offset + 7)]
                targets = self.serializer._construct_target_motor_msg(np.ones(7), MotorCommand.create_mit_command(values))
                packet.arm_command.arm_exclusive_command.arm_api_control_command.motor_targets.CopyFrom(targets)
            elif name == 'theta':
                # Local sink encoding only; the real endpoint retains calibrated
                # pulse direction/range conversion, covered by endpoint tests.
                packet.linear_lift_command.target_pos = int(candidate.positions[3] * 10000)
            else:
                speed = packet.base_command.simple_move_command.xyz_speed
                speed.speed_x, speed.speed_y, speed.speed_z = map(float, candidate.velocity[:3])
            gate.authorize(sequence, created, deadline, packet)
            allowed = gate.filter(packet)
            if gate.fault or allowed is None:
                raise HardwareSafetyError(gate.fault or 'local sink refused packet')
            started = time.monotonic()
            self.loop.run_until_complete(send_checked(api, allowed))
            gate.record_dispatch(allowed, started, time.monotonic(), sequence=sequence)
            times[name] = dict(gate.last_udp_dispatch)
        return times

    def close(self):
        for client in self.clients:
            client.stop()
        self.clients.clear()
        self.loop.close()
