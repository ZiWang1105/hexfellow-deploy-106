import sys
from pathlib import Path
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'tools/camera'))
from calibrate_robot import startup_audit,retryable_feedback_audit


def transient():
    return dict(mode='audit',passed=False,motion_commands_sent=0,motion_readiness_passed=True,
        performance_passed=True,cycles=1500,failed_cycles=0,skipped_slots=0,trace_complete=True,
        diagnostics_complete=True,trace_dropped_rows=0,gateway_observed_feedback_clean=False,
        initial_health={'right':{}},gateway=dict(kind='closed',fault=None,stop_failures=[],motion_commands_sent=0,
        feedback_observations=dict(events=[dict(error='robot feedback skew exceeds 20 ms',recovered_at=1)])))


def test_retry_preserves_failure_and_requires_new_pass(tmp_path):
    bad=transient();good=dict(bad,passed=True,gateway_observed_feedback_clean=True)
    reports=iter([bad,good]);seen=[]
    def run(path):seen.append(path);return next(reports)
    final,path=startup_audit(run,lambda r:r['passed'],tmp_path)
    assert final is good and len(seen)==2 and bad['passed'] is False
    assert path.parent.name=='audit-retry-2'
    assert 'false' in (tmp_path/'startup-audits.json').read_text()


def test_repeated_failure_stops_after_three_audits(tmp_path):
    seen=[]
    def run(path):seen.append(path);return transient()
    final,_=startup_audit(run,lambda r:r['passed'],tmp_path)
    assert len(seen)==3 and final['passed'] is False


@pytest.mark.parametrize('change',[{'motion_readiness_passed':False},{'performance_passed':False},
    {'cycles':0},{'failed_cycles':1},{'skipped_slots':1},{'trace_complete':False},
    {'motion_commands_sent':1},{'diagnostics_complete':False}])
def test_nontransient_failures_do_not_retry(change):
    r=transient();r.update(change);assert not retryable_feedback_audit(r)


def test_unrecovered_or_failed_shutdown_does_not_retry():
    r=transient();r['gateway']['feedback_observations']['events'][0]['recovered_at']=None
    assert not retryable_feedback_audit(r)
    r=transient();r['gateway']['stop_failures']=['failure'];assert not retryable_feedback_audit(r)


def test_early_stale_abort_retries_new_audit_but_never_accepts_old_one(tmp_path):
    bad=transient();bad.update(cycles=454,failed_cycles=1,performance_passed=False,
        diagnostics_complete=False,error='ControlDelay: stale robot feedback, ages_ms=48')
    bad['gateway']['feedback_observations']['events'][0]['recovered_at']=None
    assert retryable_feedback_audit(bad)
    good=dict(transient(),passed=True,gateway_observed_feedback_clean=True)
    reports=iter([bad,good]);seen=[]
    def run(path):seen.append(path);return next(reports)
    final,_=startup_audit(run,lambda r:r['passed'],tmp_path)
    assert len(seen)==2 and final is good and bad['passed'] is False
    bad['gateway']['stop_failures']=['shutdown failure']
    assert not retryable_feedback_audit(bad)


def early_skew():
    report=transient()
    report.update(cycles=24,failed_cycles=1,performance_passed=False,diagnostics_complete=False,
        error='FeedbackSkewSafetyError: robot feedback skew exceeds 20 ms after bounded coherence wait')
    report['gateway']['feedback_observations'].update(max_skew_ms=25.8204059,
        ages_ms={'left':25.7429,'right':26.0555,'base':.2351,'theta':4.7246})
    report['gateway']['feedback_observations']['events'][0]['recovered_at']=None
    return report


def test_early_skew_can_only_trigger_a_new_complete_audit(tmp_path):
    bad=early_skew();good=dict(transient(),passed=True,gateway_observed_feedback_clean=True)
    reports=iter([bad,good]);seen=[]
    def run(path):seen.append(path);return next(reports)
    final,_=startup_audit(run,lambda r:r['passed'],tmp_path)
    assert len(seen)==2 and final is good and not bad['passed']
    from calibrate_robot import preflight_failure_detail
    message=preflight_failure_detail(bad)
    assert '25.82 ms' in message and 'right 26.06 ms' in message
    assert '24/1500' in message and 'qualification incomplete' in message
    assert 'timing acceptance limits exceeded' not in message


@pytest.mark.parametrize('condition',['gateway_motion','unclosed','gateway_fault','stop_failure','close_error','readiness','missing_motion_count'])
def test_skew_retry_requires_verified_readonly_clean_shutdown(condition):
    r=early_skew()
    if condition=='gateway_motion':r['gateway']['motion_commands_sent']=1
    if condition=='unclosed':r['gateway']['kind']='running'
    if condition=='gateway_fault':r['gateway']['fault']='network unreachable'
    if condition=='stop_failure':r['gateway']['stop_failures']=['failed']
    if condition=='close_error':r['close_error']='close failed'
    if condition=='readiness':r['motion_readiness_passed']=False
    if condition=='missing_motion_count':del r['gateway']['motion_commands_sent']
    assert not retryable_feedback_audit(r)
