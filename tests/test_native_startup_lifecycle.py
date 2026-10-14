"""Startup brakes use real wire gates; no robot endpoints are opened."""
from pathlib import Path
from types import SimpleNamespace
import asyncio
import struct
import threading

import pytest
from hexfellow_deploy.workspace import activate
ROOT = activate(Path(__file__).resolve().parents[1])
from hexfellow_deploy.native import ContinuousGate, StreamEndpoint
from hexfellow_deploy.robotics.hardware.mit_wire import brake_packet, send_checked
from hexfellow_deploy.robotics.hardware.sdk_transport import CheckedUDP
from hex_device.generated.public_api_down_pb2 import APIDown


def gate(name, clock):
    result = ContinuousGate(name, clock=lambda: clock[0], defer_activation=True)
    result.transition('initializing')
    result.transition('active')
    return result


def motion(name):
    p = APIDown()
    if name == 'theta':
        p.linear_lift_command.target_pos = 100
    elif name == 'base':
        p.base_command.simple_move_command.xyz_speed.speed_x = .1
    else:
        for _ in range(7):
            p.arm_command.arm_exclusive_command.arm_api_control_command.motor_targets.targets.add().mit_target.kp = 50
    return p


@pytest.mark.parametrize('name', ['left', 'right', 'base'])
def test_braked_startup_beyond_old_one_second_deadline_then_finite_motion_lease(name):
    clock = [10.]
    g = gate(name, clock)
    brake = brake_packet(name, APIDown())
    for t in (10.1, 11.1, 12.):
        clock[0] = t
        assert g.filter(brake) == brake
        assert g.fault is None and g.phase == 'ready'
    g.finish_startup()
    command = motion(name)
    g.authorize(1, 12., 12.1, command)
    assert g.filter(command) == command
    clock[0] = 12.101
    assert g.filter(command) == brake
    assert g.fault == 'MIT publisher source lease expired'


@pytest.mark.parametrize('name', ['left', 'right', 'base'])
def test_startup_has_a_finite_deadline_and_forbids_early_motion(name):
    clock = [10.]
    g = gate(name, clock)
    with pytest.raises(RuntimeError, match='authorization'):
        g.authorize(1, 10., 10.1, motion(name))
    clock[0] = 18.01
    g.filter(brake_packet(name, APIDown()))
    assert g.fault == 'Native startup deadline expired before stream activation'
    with pytest.raises(RuntimeError):
        g.finish_startup()


def test_fixed_lift_has_no_unused_motion_lease_and_cannot_resume():
    clock = [10.]
    g = gate('theta', clock)
    clock[0] = 100.
    brake = brake_packet('theta', APIDown())
    assert g.filter(brake) == brake
    assert g.fault is None and g.stop_token is None
    with pytest.raises(RuntimeError):
        g.finish_startup()
    with pytest.raises(RuntimeError):
        g.resume_delay(2)
    assert g.filter(motion('theta')) is None
    assert g.fault


def test_fixed_lift_blocks_unsolicited_calibration():
    g = gate('theta', [10.])
    packet = APIDown()
    packet.linear_lift_command.calibrate = True
    assert g.filter(packet) is None
    assert g.fault


def test_checked_transport_brake_receipt_does_not_consume_shutdown_token(monkeypatch):
    """Exercise actual framing, cumulative KCP ACK and stop attribution in memory."""
    clock = [10.]
    monkeypatch.setattr('time.monotonic', lambda: clock[0])
    g = gate('theta', clock)
    client = SimpleNamespace(_kcp_lock=threading.RLock(), server_address='::1', server_port=1,
                             _sock=object(), _kcp=object(), _receive_from_socket=lambda: None,
                             _process_received_data=lambda data: None)
    transport = CheckedUDP(client, g)
    captured = []
    # Simulate KCP assigning a sequence and a peer acknowledging it. No sockets.
    def send(frame):
        source = transport.pending_frames.pop(bytes(frame)).popleft()
        transport.pending_count -= 1
        source['conv'] = 9
        sn = len(captured)
        captured.append(source)
        transport.unacked[sn] = source
        return True
    client.send = send
    api = SimpleNamespace(enable_kcp=True, _HexDeviceApi__kcp_client=client, _policy_gate=g)
    endpoint = object.__new__(StreamEndpoint)
    endpoint.direct_gates = {'theta': g}
    endpoint.apis = [api]
    brake = brake_packet('theta', APIDown())
    asyncio.run(send_checked(api, g.filter(brake)))
    assert not endpoint.fixed_lift_brake_evidence()['brake_received']
    clock[0] = 10.01
    client._process_received_data(struct.pack('<IBBHIIII', 9, 82, 0, 128, 0, 0, 1, 0))
    assert endpoint.fixed_lift_brake_evidence()['brake_received']
    assert endpoint.fixed_lift_brake_evidence()['acknowledged_at'] == 10.01
    assert g.stop_token is None
    clock[0] = 20.
    g.transition('stopping')
    asyncio.run(send_checked(api, g.filter(brake)))
    assert captured[-1]['stop_token'] == 20.
    client._process_received_data(struct.pack('<IBBHIIII', 9, 82, 0, 128, 0, 1, 2, 0))
    assert transport.stop_evidence(20.)['brake_received']
    assert not endpoint.fixed_lift_brake_evidence()['brake_received']
