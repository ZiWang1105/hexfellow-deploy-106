"""Measured position-window detection, independent of contact truth or hardware."""
import json
from pathlib import Path

import pytest

from hexfellow_deploy.gripper_effort import (
    EffortGripperConfig, EffortGripperController, GripperFeedback,
)


def make(reverse=False, **kwargs):
    return EffortGripperController(EffortGripperConfig(
        closed_rad=0. if reverse else 1., open_rad=1. if reverse else 0., **kwargs))


def step(c, t, q=.5, effort=.06, speed=0.):
    return c.step(0, GripperFeedback(q, speed, c.direction*effort, t), t)


@pytest.mark.parametrize('reverse', [False, True])
def test_moving_follower_with_high_effort_does_not_enter_contact_or_backoff(reverse):
    c = make(reverse)
    q = .5
    for i in range(400):
        o = step(c, i*.004, q)
        assert not o.brake and o.state == 'closing'
        assert not o.tracking_blocked and not o.effort_regulating
        assert (o.position_rad-q)*c.direction >= 0
        q = o.position_rad
    assert (q-.5)*c.direction > .1


@pytest.mark.parametrize('reverse', [False, True])
def test_stationary_loaded_feedback_requires_full_window_of_actual_target_lag(reverse):
    c = make(reverse)
    history = []
    for i in range(100):
        t = i*.004
        preceding = c.reference
        o = step(c, t)
        history.append((t, preceding))
        assert not o.brake
        if o.state == 'regulating':
            assert o.tracking_blocked
            assert o.tracking_position_span_rad == 0
            assert o.tracking_min_lead_rad >= c.config.tracking_min_lead_rad
            assert o.tracking_window_elapsed_s >= c.config.tracking_window_s
            window = [(s, r) for s, r in history if t-.06 <= s <= t]
            assert all(r is not None and (r-.5)*c.direction >= .0005 for _, r in window)
            break
    else:
        pytest.fail('Stationary, loaded reference never produced obstruction evidence')


def test_oscillation_is_not_stationarity_even_when_net_displacement_is_zero():
    c = make()
    for i in range(160):
        q = .5 + (.0003 if (i//4)%2 else -.0003)
        o = step(c, i*.004, q)
        assert not o.brake and not o.tracking_blocked
        assert o.state == 'closing'
    assert o.tracking_position_span_rad == pytest.approx(.0006)


def test_duplicate_feedback_cannot_fill_window_and_a_large_report_gap_resets_it():
    c = make()
    for i in range(20):
        o = step(c, i*.004, effort=.03)
    row = c.previous
    before = list(c.tracking_samples)
    assert c.step(0, row, row.received_s+.01) == o
    assert list(c.tracking_samples) == before
    # A 40-ms feedback gap is still fresh but cannot establish continuous stop.
    o = step(c, row.received_s+.04, effort=.03)
    assert not o.brake and not o.tracking_blocked
    assert o.tracking_window_elapsed_s == 0


def test_zero_effort_stall_pauses_target_without_claiming_contact_then_faults():
    c = make()
    paused = None
    for i in range(260):
        o = step(c, i*.004, effort=0.)
        if o.brake:
            break
        assert o.state == 'closing' and not o.effort_regulating
        if o.tracking_blocked:
            paused = o.position_rad if paused is None else paused
            assert o.position_rad == paused
    assert paused is not None
    assert o.brake and 'No measured gripper progress' in o.fault


def test_new_request_resets_window_and_raw_trip_never_waits_for_window():
    c = make(max_torque_nm=.1)
    for i in range(50):
        o = step(c, i*.004, effort=0.)
    assert o.tracking_blocked
    row = GripperFeedback(.5, 0., 0., .2)
    o = c.step(1, row, .2)
    assert not o.tracking_blocked and o.tracking_window_elapsed_s == 0
    o = c.step(0, GripperFeedback(.5, 0., 0., .204), .204)
    assert not o.tracking_blocked and not o.effort_regulating
    assert step(c, .208, effort=.1).brake


@pytest.mark.parametrize('settings', [
    {'tracking_window_s': 1.}, {'tracking_window_s': 0.},
    {'tracking_min_lead_rad': .01}, {'tracking_position_span_rad': float('nan')},
])
def test_invalid_tracking_config(settings):
    with pytest.raises(ValueError):
        make(**settings)


def test_physical_183901_replay_no_longer_reverses_on_initial_effort_rise():
    data = json.loads((Path(__file__).parent/'fixtures/gripper_close_183901.json').read_text())
    c = EffortGripperController(EffortGripperConfig(**{k:v for k,v in data['config'].items() if not k.startswith('backoff_')}))
    start = data['samples'][0]['now_s']
    first_backoff = None
    references = []
    for r in data['samples']:
        row = GripperFeedback(r['pos'], r['vel'], r['effort_nm'], r['received'])
        o = c.step(0, row, r['now_s'])
        if o.brake:
            break
        references.append(o.position_rad)
        if r['now_s']-start < .1:
            assert o.state == 'closing' and not o.effort_regulating
        if o.state == 'regulating' and first_backoff is None:
            first_backoff = r['now_s']-start
            assert o.tracking_blocked
    assert first_backoff is not None and first_backoff > .1
    assert max(references) > data['samples'][0]['pos']+.003
    # This short recorded tail ends before the separate settling deadline.
    # It must not be labeled a grasp, nor fail the approach progress check.
    assert not o.brake and o.state != 'holding'


def test_recorded_retreat_is_not_failed_for_losing_closing_progress():
    data = json.loads((Path(__file__).parent/'fixtures/gripper_close_195905_contact.json').read_text())
    c = EffortGripperController(EffortGripperConfig(**{k:v for k,v in data['config'].items() if not k.startswith('backoff_')}))
    states = set()
    for r in data['samples']:
        o = c.step(0, GripperFeedback(r['pos'], r['vel'], r['effort_nm'], r['received']), r['now_s'])
        assert not o.brake, o.fault
        states.add(o.state)
    assert 'regulating' in states and o.effort_regulating
    assert c.regulation_unsettled_since is not None
    # A separate synthetic continuation supplies stable holding feedback at the
    # retreated position: no extra closing progress is required to hold.
    for i in range(1, 601):
        t = r['now_s']+i*.004
        o = c.step(0, GripperFeedback(r['pos'], 0., .1, t), t)
        assert not o.brake, o.fault
    assert o.state == 'holding' and c.regulation_unsettled_since is None
