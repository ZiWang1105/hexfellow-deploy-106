from hexfellow_deploy.experiment import FinalHold


def test_final_hold_waits_for_ack_and_uninterrupted_actual_hold():
    hold = FinalHold(10.)
    stationary = dict(final_hold=True, diagnostic=dict(ik_status='HOLD', base_stopped=True))
    assert hold.update(1., {}) == 'request'
    # A busy request topic or competing policy write cannot start retention.
    assert hold.update(4., {}) == 'request'
    assert hold.update(5., dict(final_hold=True, diagnostic=dict(ik_status='BRAKING'))) == 'waiting'
    assert hold.update(6., stationary) == 'waiting'
    assert hold.update(15., stationary) == 'waiting'
    assert hold.update(15.1, dict(final_hold=True, diagnostic=dict(ik_status='HOLD',base_stopped=False))) == 'waiting'
    assert hold.update(15.2, stationary) == 'waiting'
    assert hold.update(25.2, stationary) == 'completed_budget'


def test_final_hold_failure_is_bounded_and_not_success():
    hold = FinalHold(10.)
    assert hold.update(0., {}) == 'request'
    assert hold.update(26., {}) == 'final_hold_failed'
