"""The real-workflow variant must select left joints and retain operator guards."""
import json
from pathlib import Path
import sys
from types import SimpleNamespace
import numpy as np
import pytest

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'tools/camera'))
from calibration_runtime import MeasuredPoseArrival,capture_position_check
from calibration_outcomes import InvalidCalibrationPose
from calibration_sequence import Sequence
from left_hand_validation import LeftLocalChecker,load_plan
from policy_sim.hardware.x7 import HardwareSafetyError


def test_left_capture_uses_five_degrees_without_relaxing_other_arm_or_lift():
    target=np.zeros(18);q=target.copy();q[4]=np.deg2rad(4.9)
    result=capture_position_check(q,target,'left')
    assert result['maximum_active_offset_deg']==pytest.approx(4.9)
    assert result['maximum_right_offset_deg']==0
    q[4]=np.deg2rad(5.1)
    with pytest.raises(InvalidCalibrationPose):capture_position_check(q,target,'left')
    q[4]=0;q[11]=.021
    with pytest.raises(ValueError,match='right arm'):capture_position_check(q,target,'left')
    q[:]=0;q[3]=.004
    with pytest.raises(ValueError,match='Base/lift'):capture_position_check(q,target,'left')


def test_left_arrival_certifies_measured_not_requested_angles():
    gate=MeasuredPoseArrival('left');target=np.zeros(18)
    for i in range(51):
        t=i*.02;q=target.copy();q[7]=.05
        state=SimpleNamespace(q=q,v=target.copy(),received=dict.fromkeys(('left','right','base','theta'),t))
        accepted=gate.update(state,target,target,target,t,.02)
        assert accepted==(i==50)
    assert gate.details['capture_side']=='left' and gate.details['measured_q'][7]==.05
    state.received['left']=0
    with pytest.raises(HardwareSafetyError,match='fresh'):gate.update(state,target,target,target,1.1,.02)


def step(sequence,**changes):
    v=dict(now=0,prepared=True,phase='hold',arrived=1,target=1,captured={},solved=False,
           interrupted=0,paused=False,busy=False,fresh_operator=True)
    v.update(changes);return sequence.step(**v)


def test_three_pose_sequence_defaults_then_returns_then_compares():
    s=Sequence(pose_count=3,preparation='prepare-default',return_before_solve=True)
    assert step(s,prepared=False) is None
    s.start();assert step(s,prepared=False)=='prepare-default'
    assert step(s,prepared=False) is None
    assert step(s,phase='default') is None
    assert step(s,now=1) is None
    assert step(s,now=2.1)=='capture'
    assert step(s,captured={1:{}})=='next'
    done={1:{},2:{},3:{}}
    assert step(s,captured=done)=='return'
    assert step(s,captured=done,arrived=-1,target=-1)=='solve'
    assert step(s,captured=done,arrived=-1,target=-1,solved=True) is None
    assert s.mode=='complete'


def test_validation_browser_loss_and_invalid_pose_keep_guarded_behavior():
    s=Sequence(pose_count=3,preparation='prepare-default',return_before_solve=True);s.start()
    assert step(s,captured={1:{}},ineligible={2:{},3:{}})=='return'
    assert step(s,pending_invalid={'ready':False}) is None
    assert step(s,pending_invalid={'ready':True})=='skip-invalid'
    assert step(s,fresh_operator=False) is None and s.mode=='paused'
    assert step(s,fresh_operator=True) is None
    s.start();assert step(s,interrupted=1) is None and s.mode=='paused'


def test_left_plan_preserves_actual_other_joints_and_checks_reference(monkeypatch):
    c=object.__new__(LeftLocalChecker)
    q=np.zeros(18);q[3]=.3;q[11:]=.01
    target=q.copy();target[4:11]=.4
    c.plan_document={'poses':[{'q':target.tolist()}]}
    checked=[]
    monkeypatch.setattr(c,'check_pose',lambda *a:{})
    monkeypatch.setattr(c,'edge_full',lambda a,b,*rest:checked.append((a.copy(),b.copy())) or .02)
    reference=q.copy();reference[4]=.023
    payload=c.plan(q,1,reference_origin=reference)
    np.testing.assert_array_equal(np.array(payload['targets'][0])[11:],q[11:])
    assert payload['path_origins_checked']==['measured','commanded_reference']
    assert len(checked)==3 and checked[-1][0][4]==.023
    monkeypatch.setattr(c,'edge_full',lambda *a:(_ for _ in ()).throw(ValueError('blocked path')))
    with pytest.raises(InvalidCalibrationPose):c.plan(q,1)
    monkeypatch.setattr(c,'check_pose',lambda *a:(_ for _ in ()).throw(ValueError('unsafe current state')))
    with pytest.raises(ValueError) as error:c.plan(q,1)
    assert not isinstance(error.value,InvalidCalibrationPose)


def test_reviewed_plan_rejects_modified_targets(tmp_path):
    p=ROOT/'config/camera/left_hand_validation.json'
    plan=load_plan(p)
    plan['poses'][0]['q'][4]+=.1
    changed=tmp_path/'plan.json';changed.write_text(json.dumps(plan))
    with pytest.raises(ValueError,match='simulated targets'):load_plan(changed)
