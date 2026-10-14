from collections import deque
from types import SimpleNamespace
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'tools/camera'))
import calibration_runtime as runtime
from calibration_path import LocalChecker
from calibration_outcomes import InvalidCalibrationPose
from hexfellow_deploy.robotics.hardware.x7 import HardwareSafetyError


def state_at(t,offset=-.02231):
    q=np.zeros(18);q[11]=offset
    return SimpleNamespace(q=q,v=np.zeros(18),received=dict.fromkeys(('left','right','base','theta'),t))


def test_stationary_offset_is_accepted_without_replacing_measured_angles():
    gate=runtime.MeasuredPoseArrival();target=np.zeros(18)
    for i in range(51):
        state=state_at(i*.02)
        accepted=gate.update(state,target,target,target,i*.02,.02)
        assert accepted == (i==50)
    assert gate.details['measured_q'][11]==-.02231
    assert gate.details['target_q'][11]==0


def test_capture_rechecks_current_offset_instead_of_trusting_old_settling():
    q=state_at(0).q;target=np.zeros(18)
    assert runtime.capture_position_check(q,target)['maximum_right_offset_deg']>1.146
    q[11]=np.deg2rad(4.9)
    assert runtime.capture_position_check(q,target)['maximum_right_offset_deg']==pytest.approx(4.9)
    q[11]=np.deg2rad(5.1)
    with pytest.raises(InvalidCalibrationPose,match='5 deg'):runtime.capture_position_check(q,target)
    q[:]=0.;q[3]=.004
    with pytest.raises(ValueError,match='Base/lift'):runtime.capture_position_check(q,target)


@pytest.mark.parametrize('condition',['moving','drifting','reference','large_offset','left','lift'])
def test_unstable_or_unbounded_endpoint_times_out(condition):
    gate=runtime.MeasuredPoseArrival();target=np.zeros(18)
    error=HardwareSafetyError if condition in ('left','lift') else InvalidCalibrationPose
    with pytest.raises(error,match='timed out'):
        for i in range(520):
            t=i*.02;state=state_at(t);ref=target.copy()
            if condition=='moving':state.v[11]=.03
            if condition=='drifting':state.q[11]=.009*np.sin(2*np.pi*t);state.v[11]=0
            if condition=='reference':ref[11]=.003
            if condition=='large_offset':state.q[11]=.1
            if condition=='left':state.q[4]=.022
            if condition=='lift':state.q[3]=.004
            assert not gate.update(state,target,ref,target,t,.02)


def test_repeated_stale_and_gapped_feedback_cannot_certify_stationarity():
    gate=runtime.MeasuredPoseArrival();zero=np.zeros(18);state=state_at(0.)
    for _ in range(100):assert not gate.update(state,zero,zero,zero,0.,.02)
    assert not gate.rows or gate.details['stationary_duration_s']==0
    with pytest.raises(HardwareSafetyError,match='fresh'):
        gate.update(state,zero,zero,zero,.05,.02)
    for i in range(1,30):
        assert not gate.update(state_at(i*.02),zero,zero,zero,i*.02,.02)
    assert not gate.update(state_at(1.),zero,zero,zero,1.,.02)
    assert gate.details['stationary_duration_s']==0


def test_capture_endpoint_keeps_support_reference_and_default_uses_original_guard(monkeypatch):
    zero=np.zeros(18);state=state_at(0)
    core=SimpleNamespace(model=SimpleNamespace(lowerPositionLimit=np.full(18,-3.),upperPositionLimit=np.full(18,3.)),
        _last_safe_positions=zero.copy(),_last_velocity=zero.copy(),config=SimpleNamespace(control_dt=.02))
    source=runtime.CalibrationSource()
    c=runtime.CalibrationController(core,state,None,None,source=source)
    c.phase='calibration';c.pose_id=source.target_pose=1;c.started=0.
    c.segment=SimpleNamespace(target=zero.copy(),duration=0.,sample=lambda t:(zero.copy(),zero.copy()))
    c.measured_arrival=runtime.MeasuredPoseArrival()
    monkeypatch.setattr(runtime,'tcp_poses',lambda *a:None)
    for i in range(51):c.sample(state_at(i*.02),i*.02)
    assert c.phase=='hold' and source.arrived_pose==1
    np.testing.assert_array_equal(c.hold_q,zero)  # Never drop PD holding torque by rebasing to actual q.
    c.phase='calibration';c.pose_id=-1;c.started=0.;c.arrival=c.new_arrival()
    with pytest.raises(HardwareSafetyError,match='right J1'):
        for i in range(520):c.sample(state_at(i*.02),i*.02)


def test_next_move_requires_certificate_for_actual_reference(monkeypatch):
    state=state_at(0);zero=np.zeros(18)
    core=SimpleNamespace(model=SimpleNamespace(lowerPositionLimit=np.full(18,-3.),upperPositionLimit=np.full(18,3.)),
        _last_safe_positions=zero.copy(),_validate_configuration_vector=lambda q:None)
    source=runtime.CalibrationSource();c=runtime.CalibrationController(core,state,None,None,source=source)
    payload=dict(origin=state.q.copy(),targets=[zero.copy()],pose_id=2)
    monkeypatch.setattr(c,'_segment',lambda *args:None)
    with pytest.raises(ValueError,match='Reference differs'):c.command('calibration-path',payload,state,0.)
    payload['reference_origin']=zero.copy()
    c.command('calibration-path',payload,state,0.)
    assert source.target_pose==2
    core._last_safe_positions[11]=.006
    with pytest.raises(ValueError,match='Reference changed'):c.command('calibration-path',payload,state,0.)


def test_measured_geometry_and_reference_path_are_separately_checked(monkeypatch):
    c=object.__new__(LocalChecker)
    c.study=SimpleNamespace(q0=np.zeros(15),pi=np.arange(15),clearance=lambda a:(.02,None))
    q=np.zeros(18);bounds=[np.full(18,-1),np.full(18,1)]
    assert c.check_pose(q,bounds)['minimum_clearance_m']==.02
    c.study.clearance=lambda a:(.01,['board','arm'])
    with pytest.raises(ValueError,match='clearance'):c.check_pose(q,bounds)
    c.study.clearance=lambda a:(.02,None)
    q[11]=2.
    with pytest.raises(ValueError,match='bounds'):c.check_pose(q,bounds)
    q[:]=0.;ref=q.copy();ref[11]=.02231
    monkeypatch.setattr(c,'_plan',lambda *args:dict(origin=q.tolist(),targets=[q.tolist()],pose_id=2))
    checked=[]
    def edge(a,b,bounds):
        checked.append((np.array(a),np.array(b)))
        return .02
    monkeypatch.setattr(c,'edge_full',edge)
    payload=c.plan(q,2,bounds,reference_origin=ref)
    assert len(checked)==2 and checked[0][0][11]==.02231
    assert payload['reference_origin']==ref.tolist()
    monkeypatch.setattr(c,'edge_full',lambda *a:(_ for _ in ()).throw(ValueError('blocked reference path')))
    with pytest.raises(ValueError,match='blocked reference'):c.plan(q,2,bounds,reference_origin=ref)
