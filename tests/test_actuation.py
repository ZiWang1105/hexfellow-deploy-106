from pathlib import Path
import time
import numpy as np
import pytest
from hexfellow_deploy.workspace import activate

activate(Path(__file__).resolve().parents[1])
from hexfellow_deploy.actuation import StreamContract,encode_mit,decode_mit
from hexfellow_deploy.native import ContinuousGate
from hexfellow_deploy.backend import BackendProxy
from hexfellow_deploy.robot.joints import JOINTS,LEFT,RIGHT
from hexfellow_deploy.robotics.control.mit_command import MitCommand,mit_torque
from hex_device.generated.public_api_down_pb2 import APIDown


def example(now=1.):
    q = {n:0. for n in JOINTS}
    q[JOINTS[0]] = .3
    measured = dict(timestamp_s=now,joint_positions=q,joint_velocities={n:0. for n in JOINTS},
                    gripper_state={'left':'closed','right':'closed'})
    mit = MitCommand(LEFT+RIGHT,np.zeros(14),np.zeros(14),np.zeros(14),np.ones(14)*200,np.ones(14)*5,now)
    command = dict(joint_positions=q,base_velocity_body=[0.,0.,0.],gripper_open=dict(left=1,right=0))
    return dict(sequence=1,created_s=now,command=command,mit=encode_mit(mit)),measured


def test_wire_frame_round_trip_and_separate_stream_health_clocks():
    now=[1.]
    frame,measured=example()
    stream=StreamContract(.1,.1,clock=lambda:now[0])
    received=stream.accept(frame,measured)
    np.testing.assert_array_equal(received.motor_input(),decode_mit(frame['mit']).motor_input())
    now[0]=1.065
    stream.check(dict(measured,timestamp_s=1.060))
    now[0]=1.101
    with pytest.raises(RuntimeError,match='stopped updating'):
        stream.check(dict(measured,timestamp_s=1.100))


def test_fresh_commands_do_not_hide_stale_feedback():
    frame,measured=example()
    stream=StreamContract(.1,.1,clock=lambda:1.11)
    frame['created_s']=1.11
    with pytest.raises(RuntimeError,match='feedback is stale'):
        stream.accept(frame,measured)


def arm_packet(position=0.):
    packet=APIDown()
    targets=packet.arm_command.arm_exclusive_command.arm_api_control_command.motor_targets
    for _ in range(7):
        target=targets.targets.add().mit_target
        target.position=position
        target.kp=200.
        target.kd=5.
    return packet


def test_native_wire_accepts_ordinary_delay_but_brakes_stalled_stream():
    now=[1.]
    gate=ContinuousGate('right',clock=lambda:now[0],timeout_s=.1)
    gate.transition('active')
    packet=arm_packet()
    gate.authorize(1,1.,1.1,packet,source_received=.995,source_expires=1.1)
    now[0]=1.065
    assert gate.filter(packet)==packet and gate.fault is None
    gate.record_dispatch(packet,1.065,1.066,sequence=1)
    now[0]=1.08
    gate.authorize(2,1.075,1.175,packet)
    now[0]=1.176
    assert gate.filter(packet)!=packet
    assert gate.fault and gate.phase=='stopping'
    with pytest.raises(RuntimeError):
        gate.authorize(3,1.176,1.276,packet)


def test_unknown_native_payload_is_not_accepted():
    gate=ContinuousGate('left',clock=lambda:1.)
    gate.transition('active')
    gate.authorize(1,1.,1.1,arm_packet())
    assert gate.filter(arm_packet(1.))!=arm_packet(1.)
    assert gate.fault


def test_busy_read_preserves_original_feedback_epoch():
    frame,measured=example()
    class Topic:
        def __init__(self,rows):self.rows=iter(rows)
        def get(self, **kwargs):return next(self.rows)
    status=Topic([{'metadata':{}},{},{}])
    feedback=Topic([dict(measured=measured,received={'left':1.}),None])
    proxy=object.__new__(BackendProxy)
    proxy.status,proxy.feedback,proxy.selected=status,feedback,None
    assert proxy.read()==measured
    assert proxy.read()==measured
    assert proxy.received_for(measured)=={'left':1.}
