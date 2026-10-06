from io import StringIO
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'tools/camera'))
import calibration_runtime as runtime
from calibration_outcomes import InvalidCalibrationPose
from calibration_path import LocalChecker,PathChecker
from calibration_sequence import Sequence
from policy_sim.hardware.x7 import HardwareSafetyError


def state_at(t,offset=0.):
    q=np.zeros(18);q[11]=offset
    return SimpleNamespace(q=q,v=np.zeros(18),received=dict.fromkeys(('left','right','base','theta'),t))


def test_five_degrees_is_capture_limit_for_each_right_joint():
    for j in range(11,18):
        for sign in (-1,1):
            gate=runtime.MeasuredPoseArrival();zero=np.zeros(18)
            for i in range(51):
                state=state_at(i*.02);state.q[j]=sign*np.deg2rad(5.)
                accepted=gate.update(state,zero,zero,zero,i*.02,.02)
            assert accepted and gate.details['capture_offset_limit_deg']==5.
            state.q[j]=sign*np.deg2rad(5.01)
            with pytest.raises(InvalidCalibrationPose):runtime.capture_position_check(state.q,zero)


@pytest.mark.parametrize('intermediate',[False,True])
def test_unreachable_pose_stops_then_waits_for_stationary_hold(monkeypatch,intermediate):
    zero=np.zeros(18);state=state_at(0.,.1);stops=[]
    core=SimpleNamespace(model=SimpleNamespace(lowerPositionLimit=np.full(18,-3.),upperPositionLimit=np.full(18,3.)),
        _last_safe_positions=zero.copy(),_last_velocity=zero.copy(),config=SimpleNamespace(control_dt=.02),
        control_state='TRACKING',_normal_lift_upper=None,request_stop=lambda:stops.append(True),
        _command_frame_poses=lambda q:None)
    source=runtime.CalibrationSource();source.prepared=True
    c=runtime.CalibrationController(core,state,None,None,source=source)
    c.phase='calibration';c.pose_id=source.target_pose=2;c.started=0.
    c.segment=SimpleNamespace(target=zero.copy(),duration=0.,sample=lambda t:(zero.copy(),zero.copy()))
    c.measured_arrival=runtime.MeasuredPoseArrival()
    if intermediate:c.todo.append(zero.copy())
    monkeypatch.setattr(runtime,'tcp_poses',lambda *a:None)
    for i in range(520):
        now=i*.02;c.sample(state_at(now,.1),now)
        if c.phase=='stopping':break
    assert stops==[True] and c.phase=='stopping' and not c.todo
    assert source.pending_invalid['pose_id']==2 and not source.pending_invalid['ready']
    assert source.arrived_pose==0 and source.failure_detail is None
    assert source.interruption_count==0 and not source.paused
    core.control_state='HOLD'
    # A HOLD control state alone is insufficient: confirm measured stationarity.
    for k in range(51):
        t=now+(k+1)*.02;c.sample(state_at(t,.1),t)
        assert source.pending_invalid['ready']==(k==50)
    np.testing.assert_array_equal(c.hold_q,zero)
    # The accepted reference remains intact even though actual position differs.
    state=state_at(t+.02,.1);state.v[11]=.1
    c.sample(state,t+.02)
    assert not source.pending_invalid['ready']


def test_invalid_pose_does_not_override_pause_network_or_browser_interruption():
    s=Sequence();s.start()
    args=dict(now=0,prepared=True,phase='stopping',arrived=0,target=2,captured={1:{}},
        solved=False,interrupted=0,paused=False,busy=False,fresh_operator=True,
        pending_invalid=dict(pose_id=2,ready=False))
    assert s.step(**args) is None
    args['phase']='hold'
    assert s.step(**args) is None
    args['pending_invalid']['ready']=True
    assert s.step(**args)=='skip-invalid'
    s.pause();assert s.step(**args) is None
    s.start();assert s.step(**dict(args,interrupted=1)) is None
    s.start();assert s.step(**dict(args,fresh_operator=False)) is None
    s.start()
    assert s.step(**dict(args,pending_invalid=None,ineligible={2:{}}))=='next'


def test_recovery_never_accepts_stale_or_repeated_feedback():
    gate=runtime.StationaryHold();state=state_at(0)
    for _ in range(60):assert not gate.update(state,0)
    with pytest.raises(HardwareSafetyError,match='fresh'):gate.update(state,.05)


def test_only_typed_planning_rejection_crosses_process_as_invalid():
    checker=object.__new__(PathChecker)
    checker.process=SimpleNamespace(stdin=StringIO(),stdout=StringIO(
        '{"error":"No approach","error_kind":"invalid_pose","pose_id":4}\n'
        '{"error":"Current pose unsafe","error_kind":"failure"}\n'))
    with pytest.raises(InvalidCalibrationPose) as caught:checker.request({})
    assert caught.value.pose_id==4
    with pytest.raises(ValueError) as caught:checker.request({})
    assert not isinstance(caught.value,InvalidCalibrationPose)


def test_reference_origin_failure_stops_but_target_approach_failure_skips(monkeypatch):
    checker=object.__new__(LocalChecker);zero=np.zeros(18);calls=[]
    monkeypatch.setattr(checker,'_plan',lambda *a:dict(targets=[zero.tolist()]))
    def edge(*args):
        calls.append(1)
        if len(calls)==2:raise ValueError('target path clearance')
        return .02
    monkeypatch.setattr(checker,'edge_full',edge)
    with pytest.raises(InvalidCalibrationPose,match='approach'):
        checker.plan(zero,4,reference_origin=zero)
    def unsafe(*a):raise ValueError('unsafe origin')
    monkeypatch.setattr(checker,'edge_full',unsafe)
    with pytest.raises(ValueError) as caught:checker.plan(zero,4,reference_origin=zero)
    assert not isinstance(caught.value,InvalidCalibrationPose)


def test_no_approach_is_skippable_but_unsafe_current_pose_is_not():
    checker=object.__new__(LocalChecker)
    checker.study=SimpleNamespace(q0=np.zeros(15),pi=np.arange(15),
        lower=np.full(7,-2.),upper=np.full(7,2.),rng=np.random.default_rng(1),
        clearance=lambda q:(.02,None))
    checker.selected=[np.zeros(7)]
    checker.route=dict(poses=[dict(right_arm_q=[0.]*7)],segments=[])
    checker.edge=lambda *args:(False,.01)
    with pytest.raises(InvalidCalibrationPose,match='approach'):
        checker._plan(np.zeros(18),1)
    checker.study.clearance=lambda q:(.01,None)
    with pytest.raises(ValueError) as caught:checker._plan(np.zeros(18),1)
    assert not isinstance(caught.value,InvalidCalibrationPose)
