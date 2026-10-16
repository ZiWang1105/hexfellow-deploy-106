"""Scanning uses the existing constrained hold and bounded base envelope."""
from types import SimpleNamespace
import numpy as np
import pytest
from hexfellow_deploy.control.executor import PoseExecutor
from hexfellow_deploy.control.base_motion import BaseMotionConfig


def executor():
    obj=object.__new__(PoseExecutor)
    obj.base_motion=BaseMotionConfig(.4,1.,.2,.5)
    obj.startup=None
    obj.gripper={'left':0,'right':1}
    obj.hold=lambda measured: None
    return obj


def test_scan_preserves_grippers_and_uses_full_ramp_area():
    obj=executor()
    obj.scan_pulse([.16,0],0,{})
    np.testing.assert_allclose(obj.pulse_velocity,[.2,0,0])
    assert obj.gripper=={'left':0,'right':1}
    assert obj.navigation['source']=='operator_scan_plan'
    assert obj.requires_base


@pytest.mark.parametrize('xy,yaw', [([.33,0],0),([0,0],.21),([.1,0],.1),([float('nan'),0],0),([0],0)])
def test_scan_rejects_unbounded_or_mixed_motion(xy,yaw):
    with pytest.raises(ValueError):
        executor().scan_pulse(xy,yaw,{})
