import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'tools/camera'))
from calibration_sequence import Sequence


def step(s,**changes):
    state=dict(now=0,prepared=True,phase='hold',arrived=1,target=1,captured={},solved=False,
               interrupted=0,paused=False,busy=False,fresh_operator=True)
    state.update(changes)
    return s.step(**state)


def test_no_motion_without_operator_start():
    s=Sequence()
    assert step(s) is None
    s.start();assert step(s,prepared=False)=='prepare'


def test_capture_after_settling_then_advance_then_solve():
    s=Sequence();s.start()
    assert step(s,phase='calibration') is None
    assert step(s,now=1) is None
    assert step(s,now=1.9) is None
    assert step(s,now=2.1)=='capture'
    assert step(s,captured={1:{}})=='next'
    assert step(s,captured=dict.fromkeys(range(1,31)))=='solve'
    assert step(s,captured=dict.fromkeys(range(1,31)),solved=True) is None
    assert s.mode=='complete'


def test_feedback_or_browser_failure_needs_explicit_restart():
    for changes in ({'interrupted':1},{'fresh_operator':False}):
        s=Sequence();s.start();assert step(s,**changes) is None
        assert s.mode=='paused' and step(s,now=100) is None


def test_pause_return_and_resume_are_distinct():
    s=Sequence();s.start();s.pause();assert step(s,paused=True) is None
    s.start();assert step(s,paused=True)=='resume'
    s.start(returning=True);assert step(s,phase='stopping') is None
    assert step(s,paused=True)=='return'
    assert step(s,arrived=-1,target=-1) is None and s.mode=='paused'


def test_busy_never_queues_next_or_capture():
    s=Sequence();s.start()
    assert step(s,busy=True,now=100) is None


def test_preparation_is_not_queued_twice_while_controller_initializes():
    s=Sequence();s.start()
    assert step(s,prepared=False)=='prepare'
    assert step(s,prepared=False,now=5) is None


def test_not_eligible_advances_and_all_attempts_finish():
    s=Sequence();s.start()
    assert step(s,ineligible={1:{'reason':'zero corners'}})=='next'
    captured={i:{} for i in range(1,31) if i%4}
    ineligible={i:{} for i in range(1,31) if not i%4}
    assert step(s,arrived=30,target=30,captured=captured,ineligible=ineligible)=='solve'
    assert step(s,captured=captured,ineligible=ineligible,solved=True) is None
    assert s.mode=='complete'


def test_all_images_ineligible_still_terminates_without_recapturing():
    s=Sequence();s.start()
    assert step(s,ineligible=dict.fromkeys(range(1,31)))=='solve'
    assert step(s,ineligible=dict.fromkeys(range(1,31)),solved=True) is None
    assert s.mode=='complete'


def test_skipped_images_do_not_override_robot_interruption():
    s=Sequence();s.start()
    assert step(s,ineligible={1:{}},interrupted=1) is None
    assert s.mode=='paused'


def test_wrist_fit_is_saved_before_guarded_return():
    s=Sequence(return_after_solve=True);s.start()
    attempts=dict.fromkeys(range(1,31))
    assert step(s,captured=attempts)=='solve'
    assert step(s,captured=attempts,solved=True)=='return'
    assert step(s,captured=attempts,solved=True,phase='calibration') is None
    assert step(s,captured=attempts,solved=True,arrived=-1,target=-1) is None
    assert s.mode=='complete'


def test_head_validation_keeps_return_before_comparison():
    s=Sequence(pose_count=3,return_before_solve=True);s.start()
    attempts=dict.fromkeys(range(1,4))
    assert step(s,captured=attempts)=='return'
    assert step(s,captured=attempts,arrived=-1,target=-1)=='solve'
    assert step(s,captured=attempts,solved=True,arrived=-1,target=-1) is None
    assert s.mode=='complete'
