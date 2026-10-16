"""Operator deadman expires inside the controller even without new IPC."""
from types import SimpleNamespace
import numpy as np
import pytest
import hexfellow_deploy.control.executor as module
from hexfellow_deploy.control.executor import PoseExecutor
from hexfellow_deploy.control.base_motion import BaseMotionConfig


def make_executor(monkeypatch, now):
    monkeypatch.setattr(module.time, 'monotonic', lambda: now[0])
    monkeypatch.setattr(module, 'validate_sensor_packet', lambda packet: None)
    obj = object.__new__(PoseExecutor)
    obj.base_motion = BaseMotionConfig(.4, 1., .2, .5)
    obj.dt, obj.startup, obj.manual_base_command = .02, None, None
    obj._base_reference = np.zeros(3)
    obj.gripper = {'left':0, 'right':1}
    obj.ik = SimpleNamespace(hold=lambda: None, core=SimpleNamespace(control_state='HOLD'),
        step=lambda _: ({'unchanged':1}, SimpleNamespace(status='HOLD')))
    return obj


def test_lease_expires_without_further_input_and_brakes(monkeypatch):
    now = [10.]
    obj = make_executor(monkeypatch, now)
    obj.manual_base([.1,0,.15], 10.25, {})
    obj.tick({})
    obj.ik.core.control_state = 'BRAKING'
    for _ in range(8):
        command, diagnostic = obj.tick({})
    assert command['base_velocity_body'] == pytest.approx([.1,0,.15])
    assert command['gripper_open'] == {'left':0,'right':1}
    now[0] = 10.251
    for _ in range(8):
        command, diagnostic = obj.tick({})
    assert command['base_velocity_body'] == pytest.approx([0,0,0])
    assert diagnostic['base_stopped']
    assert diagnostic['mode'] == 'manual_input_expired'


@pytest.mark.parametrize('velocity,deadline', [([.5,0,0],10.2),([0,0,.3],10.2),
    ([float('nan'),0,0],10.2),([0,0,0],11),([0,0,0],9),([0,0,0],float('nan'))])
def test_invalid_leases_and_limits(monkeypatch, velocity, deadline):
    with pytest.raises(ValueError):
        make_executor(monkeypatch, [10.]).manual_base(velocity, deadline, {})


def test_hold_cancels_manual_and_preserves_reference(monkeypatch):
    obj = make_executor(monkeypatch, [10.])
    obj.manual_base([.1,0,0],10.2,{})
    obj.hold({})
    assert obj.manual_base_command is None
    assert obj.gripper == {'left':0, 'right':1}
