import sys
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'tools/camera'))
from calibration_runtime import stationary_pair,CalibrationSource,CalibrationController


def rows():
    return [dict(at=float(t),q=[0.]*18,v=[0.]*18,age=.004,phase='hold')
            for t in np.arange(9.45,10.13,.02)]


def test_stationary_frame_bracketing():
    r=stationary_pair(rows(),10.,uncertainty=.0001)
    assert r['image_joint_skew_s']<.02
    assert r['stationary_window_s'][0]<9.55


@pytest.mark.parametrize('kind',['moving','stale','phase','gap','not_bracketed','clock'])
def test_bad_capture_rejected(kind):
    data=rows();uncertainty=0
    if kind=='moving':data[10]['v'][11]=.2
    if kind=='stale':data[10]['age']=.2
    if kind=='phase':data[10]['phase']='stopping'
    if kind=='gap':data=data[:4]+data[10:]
    if kind=='not_bracketed':data=data[:-8]
    if kind=='clock':uncertainty=.1
    with pytest.raises(ValueError):stationary_pair(data,10.,uncertainty=uncertainty)


def test_source_pause_discards_queued_motion():
    s=CalibrationSource();s.stop=True
    s.commands.put(('calibration-path',{}))
    state=SimpleNamespace(q=np.zeros(18),v=np.zeros(18),received={'left':1,'right':1,'base':1,'theta':1})
    engine=SimpleNamespace(phase='calibration')
    assert s.poll(state,engine,1,False)==('stop',None)
    assert s.commands.empty() and s.paused


def test_stale_browser_never_acquires():
    s=CalibrationSource();s.heartbeat=-100
    s.commands.put(('default',None))
    state=SimpleNamespace(q=np.zeros(18),v=np.zeros(18),received={'left':1,'right':1,'base':1,'theta':1})
    assert s.poll(state,None,1,False) is None
    assert s.commands.empty()


def test_pause_at_arrived_hold_does_not_require_moving_again_to_capture():
    s=CalibrationSource();s.stop=True;s.arrived_pose=s.target_pose=1
    state=SimpleNamespace(q=np.zeros(18),v=np.zeros(18),received={'left':1,'right':1,'base':1,'theta':1})
    assert s.poll(state,SimpleNamespace(phase='hold'),1,False) is None
    assert not s.paused


def controller():
    state=SimpleNamespace(q=np.zeros(18),v=np.zeros(18))
    core=SimpleNamespace(model=SimpleNamespace(lowerPositionLimit=np.full(18,-2.),upperPositionLimit=np.full(18,2.)),
                         _last_safe_positions=np.zeros(18),_normal_lift_upper=None)
    def validate(q):
        if np.max(abs(q))>2:raise ValueError('outside limits')
    core._validate_configuration_vector=validate
    source=CalibrationSource()
    return CalibrationController(core,state,None,None,source=source),source,state


def test_prepare_is_hold_not_default_pose_movement():
    c,s,state=controller()
    c.command('default',None,state,0)
    assert s.prepared and c.phase=='hold' and c.segment is None
    np.testing.assert_array_equal(c.hold_q,state.q)


def test_all_path_targets_checked_before_first_motion():
    c,s,state=controller()
    bad=np.zeros(18);bad[-1]=3
    with pytest.raises(ValueError,match='limits'):
        c.command('calibration-path',dict(origin=np.zeros(18),targets=[np.zeros(18),bad],pose_id=1),state,0)
    assert c.phase=='hold' and c.segment is None and s.target_pose==0


def test_preflight_reports_charger_and_feedback_failure_together():
    from calibrate_robot import preflight_failure_detail
    report=dict(motion_readiness_error='charging: disconnect charger before whole-body control',
                gateway_observed_feedback_clean=False,performance_passed=True,
                gateway={'feedback_observations':{'invalid_samples':1,'max_skew_ms':20.85,
                    'events':[{'error':'robot feedback skew exceeds 20 ms'}]}})
    message=preflight_failure_detail(report)
    assert 'disconnect charger' in message
    assert '20.85' in message and 'skew exceeds 20 ms' in message
    assert 'timing acceptance limits exceeded' not in message


def test_pending_path_ack_waits_for_controller_motion_phase():
    s=CalibrationSource();s.pending_command='calibration-path'
    state=SimpleNamespace(q=np.zeros(18),v=np.zeros(18),received={'left':1,'right':1,'base':1,'theta':1})
    s.poll(state,SimpleNamespace(phase='hold'),1,False)
    assert s.pending_command=='calibration-path'
    s.poll(state,SimpleNamespace(phase='calibration'),1,False)
    assert s.pending_command is None


def test_rejected_robot_command_invalidates_automatic_sequence():
    c,s,state=controller();s.pending_command='calibration-path'
    bad=np.full(18,3.)
    with pytest.raises(ValueError):
        c.command('calibration-path',dict(origin=np.zeros(18),targets=[bad],pose_id=1),state,0)
    assert s.pending_command is None and s.interruption_count==1


def test_connection_failure_does_not_claim_measured_timing_violation():
    from calibrate_robot import preflight_failure_detail
    report=dict(cycles=0,performance_passed=False,gateway_observed_feedback_clean=False,
                gateway={'fault':'TimeoutError: timed out waiting for right at its configured endpoint'})
    message=preflight_failure_detail(report)
    assert 'timed out waiting for right' in message
    assert 'timing acceptance limits exceeded' not in message
    assert 'unknown invalid samples' not in message


@pytest.mark.parametrize('code',[99,101])
def test_mid_audit_network_loss_is_not_reported_as_latency_failure(code):
    from calibrate_robot import preflight_failure_detail
    report=dict(cycles=1242,performance_passed=False,
                gateway={'fault':f'HardwareSafetyError: [Errno {code}] network failure'})
    message=preflight_failure_detail(report)
    assert 'network connection failed' in message
    assert 'timing acceptance limits exceeded' not in message


def test_full_default_delegates_to_deployment_retreat_and_default(monkeypatch):
    from policy_sim.hardware import session
    c,s,state=controller();calls=[]
    monkeypatch.setattr(session.SessionController,'command',lambda self,name,payload,state,now:calls.append((name,payload)))
    c.command('default',dict(full_default=True,origin=state.q.tolist()),state,0)
    assert calls==[('default',None)]
    assert c.full_default_active and not s.prepared and s.target_pose==-2


def test_full_default_rejects_changed_origin_before_delegating(monkeypatch):
    from policy_sim.hardware import session
    c,s,state=controller();calls=[]
    monkeypatch.setattr(session.SessionController,'command',lambda *args:calls.append(args))
    origin=state.q.copy();origin[-1]=.02
    with pytest.raises(ValueError,match='moved since'):
        c.command('default',dict(full_default=True,origin=origin.tolist()),state,0)
    assert not calls and not c.full_default_active
