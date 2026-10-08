"""Candidate controller checks with no device connections or physical claims."""
from dataclasses import replace

import numpy as np
import pytest

from hexfellow_deploy.gripper_effort import (
    EffortGripperConfig, EffortGripperController, GripperFeedback,
)
from hexfellow_deploy.gripper_sdk import (
    GP80EffortPacketBuilder, require_closure_enabled,
)


def feedback(t, q=.5, effort=0., speed=0.):
    return GripperFeedback(q, speed, effort, t)


def controller(**kwargs):
    return EffortGripperController(EffortGripperConfig(closed_rad=1., open_rad=0., **kwargs))


@pytest.mark.parametrize('effort', [.1, .103896, .2, -.1, -.2])
def test_no_default_absolute_effort_cutoff_when_position_is_following(effort):
    servo = controller()
    assert servo.config.max_torque_nm is None
    q = .5
    for i in range(80):
        out = servo.step(0, feedback(i*.004, q=q, effort=effort), i*.004)
        assert not out.brake and out.state == 'closing'
        q = out.position_rad
    assert q > .5


def test_unresolved_effort_continues_past_three_seconds_without_restarting_approach():
    servo = controller()
    entered = None
    for i in range(900):
        t = i*.004
        out = servo.step(0, feedback(t, effort=.2), t)
        if out.brake:
            break
        if out.effort_regulating:
            entered = t if entered is None else entered
            assert out.state == 'regulating'
    assert entered is not None
    assert not out.brake and out.state == 'regulating'
    assert t-entered > 3.
    assert out.regulation_unsettled_s > 3.



@pytest.mark.parametrize('reverse', [False, True])
def test_contact_effort_error_controls_both_directions_without_reentering_approach(reverse):
    servo = controller() if not reverse else EffortGripperController(EffortGripperConfig(0., 1.))
    sign = servo.direction
    for i in range(70):
        t = i*.004
        out = servo.step(0, feedback(t, effort=sign*.04), t)
    assert out.state == 'holding'
    # Both high and low effort use the same signed velocity law; repeated
    # binary close commands never reset to the fast position approach.
    for level, expected_sign in ((.08, -1), (.02, 1), (.04, 0)):
        for _ in range(20):
            t += .004
            before = out.position_rad
            out = servo.step(0, feedback(t, effort=sign*level), t)
            assert out.effort_regulating and out.state in ('regulating', 'holding')
        delta = (out.position_rad-before)*sign
        assert delta*expected_sign > 0 if expected_sign else abs(delta) < 1e-12
        assert abs(out.reference_velocity_rad_s) <= servo.config.regulation_speed_rad_s
    out = servo.step(1, feedback(t+.004), t+.004)
    assert out.state == 'opening' and not out.effort_regulating



def test_blocked_low_effort_does_not_wind_up_or_claim_contact():
    servo = controller()
    for i in range(260):
        out = servo.step(0, feedback(i*.004), i*.004)
        if out.brake:
            break
        assert out.position_rad <= .51
        assert out.state == 'closing'
    assert out.brake and 'No measured gripper progress' in out.fault


def test_effort_relief_waits_for_tracking_window_but_not_contact_debounce():
    servo = controller(contact_duration_s=.3)
    for i in range(100):
        t = i*.004
        out = servo.step(0, feedback(t, effort=.08), t)
        assert not out.brake
        if out.state == 'regulating':
            assert out.tracking_blocked
            assert out.tracking_window_elapsed_s >= .06
            assert t < .3
            break
        assert out.state == 'closing'
    else:
        pytest.fail('Stationary, lagging, loaded gripper did not enter regulation')


@pytest.mark.parametrize('reverse', [False, True])
def test_torque_without_sustained_lag_does_not_cancel_approach(reverse):
    servo = controller() if not reverse else EffortGripperController(EffortGripperConfig(0., 1.))
    sign = servo.direction
    before = servo.step(0, feedback(1.), 1.).position_rad
    # Measured torque levels from the failed 17:21 closing run. Velocity is
    # above the contact gate, so regulating must not falsely imply contact.
    for i, effort in enumerate((.069264, .043290, .025974, .086580, .043290, 0.), 1):
        t = 1. + .004 * i
        out = servo.step(0, feedback(t, effort=sign*effort, speed=.1), t)
        assert out.state == 'closing' and not out.brake
        actual = (out.position_rad-before)*sign
        assert 0 < actual <= .2*.004+1e-12
        assert not out.tracking_blocked
        before = out.position_rad
    # A new opening request clears the closing-regulation latch.
    out = servo.step(1, feedback(1.028), 1.028)
    assert not out.effort_regulating and out.state == 'opening'


def test_regulation_without_progress_or_confirmed_contact_still_stops():
    servo = controller()
    for i in range(270):
        t = i*.004
        out = servo.step(0, feedback(t, effort=.04, speed=.1), t)
        if out.brake:
            break
    assert out.brake and 'No measured gripper progress' in out.fault


def test_small_lead_stall_reports_measured_progress_without_inferring_contact():
    servo = controller(max_lead_rad=.001, max_speed_rad_s=.025)
    # A small initial response followed by a static mechanism, as in the
    # 2026-10-07 opening trial. Do not mask it by extending the watchdog.
    for i in range(252):
        t = i * .004
        q = .5 - min(.000153, t * .001)
        out = servo.step(1, feedback(t, q=q), t)
        if out.brake:
            break
    assert out.brake and 'No measured gripper progress:' in out.fault
    assert 'contact' not in out.fault
    detail = servo.stall_diagnostic
    assert detail['measured_progress_rad'] == pytest.approx(.000153)
    assert detail['reference_error_rad'] == pytest.approx(-.001)
    assert detail['required_progress_rad'] == .002
    assert detail['action'] == 'opening'
    assert detail['window_s'] >= 1.
    assert servo.step(1, feedback(t+.004, q=q), t+.004).brake
    assert servo.stall_diagnostic == detail


def test_external_motion_cannot_cause_unbounded_reference_jump():
    servo = controller()
    servo.step(0, feedback(1.), 1.)
    out = servo.step(0, feedback(1.004, q=.6), 1.004)
    assert out.brake and 'reference window' in out.fault


@pytest.mark.parametrize('effort', [.1, -.1, .3, -.3])
@pytest.mark.parametrize('command', [0, 1])
def test_effort_trip_is_instantaneous_bidirectional_and_latched(effort, command):
    servo = controller(max_torque_nm=.1)
    out = servo.step(command, feedback(1., effort=effort), 1.)
    assert out.brake and out.position_rad is None
    again = servo.step(1, feedback(1.004), 1.004)
    assert again.fault == out.fault


@pytest.mark.parametrize('mutation', [
    lambda f: replace(f, effort_nm=float('nan')),
    lambda f: replace(f, effort_nm=None),
    lambda f: replace(f, velocity_rad_s=float('inf')),
    lambda f: replace(f, healthy=False),
    lambda f: replace(f, received_s=.8),
    lambda f: replace(f, received_s=1.01),
    lambda f: replace(f, position_rad=1.2),
])
def test_invalid_feedback_produces_only_brake(mutation):
    out = controller().step(0, mutation(feedback(1.)), 1.)
    assert out.brake and out.position_rad is None and out.fault


def test_duplicate_feedback_never_integrates_and_cancel_is_immediate():
    servo = controller()
    row = feedback(1.)
    first = servo.step(0, row, 1.)
    for i in range(1, 10):
        assert servo.step(0, row, 1.+i*.004) == first
    assert servo.step(1, row, 1.04).brake
    assert servo.step(None, row, 1.044).brake
    assert servo.step(0, row, 1.048).brake
    assert servo.step(0, row, 1.06).fault


def test_changed_duplicate_and_clock_discontinuities_latch():
    servo = controller()
    servo.step(0, feedback(1.), 1.)
    assert 'Repeated timestamp' in servo.step(0, feedback(1., q=.501), 1.001).fault
    servo = controller()
    servo.step(0, feedback(1.), 1.)
    assert 'update expired' in servo.step(0, feedback(1.1), 1.1).fault
    servo = controller()
    servo.step(0, feedback(1.), 1.)
    assert 'backwards' in servo.step(0, feedback(.999), .999).fault


def test_large_but_valid_sample_gap_does_not_catch_up():
    servo = controller()
    servo.step(0, feedback(1.), 1.)
    out = servo.step(0, feedback(1.04), 1.04)
    # Both acceleration and integration use the capped 20 ms step, not the
    # full 40 ms gap. No full-speed restart after delayed feedback.
    assert out.position_rad - .5 == pytest.approx(.2*.02*.02)


@pytest.mark.parametrize('reverse', [False, True])
def test_closing_starts_gently_and_can_reach_full_reference_speed(reverse):
    servo = controller() if not reverse else EffortGripperController(EffortGripperConfig(0., 1.))
    sign = servo.direction
    q = .5
    previous_speed = 0.
    servo.step(0, feedback(0., q=q), 0.)
    for i in range(1, 301):
        out = servo.step(0, feedback(i*.004, q=q), i*.004)
        assert not out.brake
        speed = (out.position_rad-q)*sign/.004
        assert 0 <= speed <= .2 + 1e-10
        assert speed - previous_speed <= .2*.004 + 1e-10
        if i == 1:
            assert abs(out.position_rad-q) == pytest.approx(.0000032)
        q, previous_speed = out.position_rad, speed
    assert previous_speed == pytest.approx(.2)


def test_raw_trip_never_waits_for_tracking_window():
    servo = controller(max_torque_nm=.1)
    servo.step(0, feedback(0.), 0.)
    out = servo.step(0, feedback(.004, effort=.08), .004)
    assert out.state == 'closing' and not out.tracking_blocked
    out = servo.step(0, feedback(.008, effort=.103896), .008)
    assert out.brake and out.position_rad is None
    assert '+0.103896 Nm; limit 0.100000 Nm' in out.fault


@pytest.mark.parametrize('reverse', [False, True])
def test_effort_integrator_saturates_without_windup_and_reverses_on_error(reverse):
    servo = controller() if not reverse else EffortGripperController(EffortGripperConfig(0., 1.))
    sign = servo.direction
    for i in range(220):
        t = i*.004
        out = servo.step(0, feedback(t, effort=sign*.2), t)
        assert not out.brake
    assert out.effort_regulating and out.reference_limited
    assert out.position_rad == pytest.approx(.5-sign*servo.config.max_lead_rad)
    before = out.position_rad
    # No hidden integral continues to accumulate beyond the issued position:
    # reversal responds after the median's one-sample delay.
    for _ in range(2):
        t += .004
        out = servo.step(0, feedback(t, effort=sign*.02), t)
    assert (out.position_rad-before)*sign > 0
    assert out.state == 'regulating'



def test_one_in_band_sample_cannot_clear_unsettled_telemetry_or_claim_holding():
    servo = controller()
    entered = None
    for i in range(1000):
        t = i*.004
        level = .04 if i%20 < 2 else .08
        out = servo.step(0, feedback(t, effort=level), t)
        if out.brake:
            break
        if out.effort_regulating:
            entered = t if entered is None else entered
            assert out.state == 'regulating'
            assert servo.regulation_unsettled_since == entered
    assert not out.brake and out.state == 'regulating'
    assert t-entered > 3.
    assert out.regulation_unsettled_s > 3.



def test_backoff_does_not_mask_expired_feedback_and_cancellation():
    servo = controller()
    servo.step(0, feedback(0.), 0.)
    row = feedback(.004, effort=.08)
    servo.step(0, row, .004)
    assert servo.step(None, row, .005).brake
    assert servo.step(None, row, .006).brake
    other = controller()
    other.step(0, feedback(0.), 0.)
    other.step(0, row, .004)
    assert 'expired' in other.step(0, row, .055).fault


def test_recorded_nonclosing_run_never_claims_hold_from_small_moving_effort():
    """Old fixed feedback cannot demonstrate new physical motion."""
    import json
    from pathlib import Path
    fixture = json.loads((Path(__file__).parent/'fixtures/gripper_close_182523.json').read_text())
    servo = EffortGripperController(EffortGripperConfig(**{k:v for k,v in fixture['config'].items() if not k.startswith('backoff_')}))
    outputs = []
    for r in fixture['samples']:
        out = servo.step(0, GripperFeedback(r['pos'], r['vel'], r['effort_nm'], r['received']), r['now_s'])
        outputs.append(out)
        if out.brake:
            break
    # The revised reference produces sustained lag against this frozen old
    # input, so it enters regulation. The 1.2-s record cannot establish either
    # new physical motion; it never claims a hold.
    assert not out.brake and out.effort_regulating
    assert not any(o.state == 'holding' for o in outputs)
    assert max(o.position_rad for o in outputs) > fixture['samples'][0]['pos']+servo.config.tracking_min_lead_rad


@pytest.mark.parametrize('reverse', [False, True])
def test_effort_oscillation_does_not_trigger_a_settling_timeout(reverse):
    servo = controller() if not reverse else EffortGripperController(EffortGripperConfig(0., 1.))
    entered = None
    for i in range(1100):
        t = i*.004
        effort = .08 if i%40 < 10 else 0.
        q = .5+servo.direction*(.0001 if i%40 < 10 else 0.)
        out = servo.step(0, feedback(t, q=q, effort=effort*servo.direction), t)
        if out.brake:
            break
        if out.effort_regulating:
            entered = t if entered is None else entered
            assert out.state == 'regulating'
    assert entered is not None and not out.brake and out.state == 'regulating'
    assert t-entered > 3.
    assert out.regulation_unsettled_s > 3.



def test_duplicate_feedback_does_not_satisfy_load_persistence_or_filter():
    servo = controller(max_torque_nm=.1)
    servo.step(0, feedback(0.), 0.)
    row = feedback(.004, effort=.08)
    first = servo.step(0, row, .004)
    for t in (.008, .012, .016, .020):
        assert servo.step(0, row, t) == first
    assert first.filtered_closing_effort_nm == 0.
    # The emergency trip still uses one raw sample, independent of filtering.
    assert servo.step(0, feedback(.024, effort=.1), .024).brake


def test_moving_feedback_cannot_project_a_stopped_reference_further_closed():
    servo = controller(max_lead_rad=.005)
    servo.step(0, feedback(0.), 0.)
    # A target-band sample requests no tightening. Measured motion cannot
    # silently drag the target forward through the lead-window projection.
    out = servo.step(0, feedback(.004, q=.506, effort=.04), .004)
    assert out.brake and 'reference window' in out.fault


@pytest.mark.parametrize('command', [.5, True, 'close'])
def test_binary_contract_preserved(command):
    assert controller().step(command, feedback(1.), 1.).brake


@pytest.mark.parametrize('setting', [
    {'max_torque_nm': 0}, {'hold_torque_nm': .1, 'max_torque_nm': .1}, {'contact_torque_nm': .04},
    {'effort_gain_rad_per_nm_s': float('nan')}, {'max_lead_rad': 1.},
    {'feedback_timeout_s': True}, {'position_ramp_approach': 1},
    {'closing_acceleration_rad_s2': 0},
    {'closing_acceleration_rad_s2': float('nan')},
])
def test_invalid_controller_config_rejected(setting):
    with pytest.raises(ValueError):
        controller(**setting)


def test_real_sdk_encoder_bypasses_legacy_position_accumulator_without_sending():
    from hex_device.hands import Hands
    from hex_device.motor_base import CommandType
    from hex_device.generated.public_api_types_pb2 import SecondaryDeviceType
    hand = Hands(1, SecondaryDeviceType.SdtHandGp80G1, 1, (1, 6), None, read_only=True)
    hand.get_motor_pulse_per_rotations = lambda: np.array([10000.])
    hand.motor_command(CommandType.POSITION, [1.])
    legacy_target = hand._target_command
    builder = GP80EffortPacketBuilder(hand, EffortGripperConfig(1., 0., max_torque_nm=.1))
    previous = None
    for i in range(50):
        now = i*.004
        row = dict(pos=.5, vel=0., effort_nm=.03, received=now, states=['normal'])
        msg = builder.build(0, row, now, command_expires_s=now+.02)
        previous = builder.last_output.position_rad
        expected = int(hand.convert_rad_to_positions(np.array([previous]), np.array([10000.]))[0])
        target = msg.secondary_device_command.hand_command.motor_targets.targets[0]
        assert target.WhichOneof('target') == 'position' and target.position == expected
        assert hand._target_command is legacy_target
        assert getattr(hand, '_last_command_send', None) is None
    row.update(received=.2, effort_nm=.08)
    builder.build(0, row, .2, command_expires_s=.22)
    assert builder.last_output.position_rad == previous
    for i in range(1, 7):
        t = .2+i*.004
        row.update(received=t)
        builder.build(0, row, t, command_expires_s=t+.02)
    assert builder.last_output.position_rad < previous
    row.update(received=.228, effort_nm=.2)
    message = builder.build(0, row, .228, command_expires_s=.248)
    assert message.secondary_device_command.hand_command.motor_targets.targets[0].brake


def test_expired_command_cannot_be_renewed_by_fresh_sensor_feedback():
    from types import SimpleNamespace
    hand = SimpleNamespace(motor_count=1, device_id=1, get_joint_limits=lambda:(0., 1.))
    builder = GP80EffortPacketBuilder(hand, EffortGripperConfig(1., 0.))
    row = dict(pos=.5, vel=0., effort_nm=0., received=1., states=['normal'])
    msg = builder.build(0, row, 1., command_expires_s=.999)
    assert msg.secondary_device_command.hand_command.motor_targets.targets[0].brake
    assert 'lease' in builder.last_output.fault


@pytest.mark.parametrize('pulses', [None, np.array([0.]), np.array([float('nan')]), np.array([1., 2.])])
def test_invalid_sdk_conversion_brakes_and_preserves_first_fault(pulses):
    from types import SimpleNamespace
    hand = SimpleNamespace(motor_count=1, device_id=1, get_joint_limits=lambda:(0., 1.),
                           get_motor_pulse_per_rotations=lambda: pulses)
    builder = GP80EffortPacketBuilder(hand, EffortGripperConfig(1., 0.))
    row = dict(pos=.5, vel=0., effort_nm=0., received=1., states=['normal'])
    msg = builder.build(0, row, 1., command_expires_s=1.02)
    assert msg.secondary_device_command.hand_command.motor_targets.targets[0].brake
    fault = builder.last_output.fault
    assert 'encoding failed' in fault
    builder.build(0, row, 1.03, command_expires_s=1.02)
    assert builder.last_output.fault == fault


def test_noisy_contact_does_not_skip_persistence_check():
    servo = controller()
    for i in range(100):
        effort = .03 if i % 2 else .02
        out = servo.step(0, feedback(i*.004, effort=effort), i*.004)
        assert out.state != 'holding'  # Neither alternating level is in the hold band.
        assert abs(out.position_rad-.5) <= .01 + 1e-12








def test_live_closure_is_disabled_by_default_but_hold_and_open_remain_available():
    for cfg in ({}, {'closure_enabled':False}, {'closure_enabled':'true'}):
        with pytest.raises(RuntimeError, match='closure disabled'):
            require_closure_enabled(cfg, 0, side='right')
        for request in (None, 1):
            require_closure_enabled(cfg, request, side='right')


@pytest.mark.parametrize('stop_angle', [.2, .4, .65])
def test_mujoco_obstruction_holds_effort_without_contact_truth(stop_angle):
    """Synthetic motor/stop model, not GP80 force calibration or pickup success."""
    import mujoco
    xml = f'''<mujoco><option timestep="0.001" gravity="0 0 0" integrator="implicitfast"/>
    <worldbody><body><joint name="motor" type="hinge" axis="0 0 1"
        range="0 {stop_angle}" limited="true" damping="0.03"
        solreflimit=".005 1" solimplimit=".99 .99 .001"/>
        <geom type="box" size=".02 .02 .02" mass=".2"/></body></worldbody>
    <compiler angle="radian"/>
    <actuator><position joint="motor" kp="8" kv=".1" forcerange="-1 1"/></actuator>
    </mujoco>'''
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    servo = controller()
    trace = []
    out = None
    for i in range(6500):
        if i % 4 == 0:
            row = feedback(float(data.time), q=float(data.qpos[0]),
                           effort=float(data.actuator_force[0]), speed=float(data.qvel[0]))
            out = servo.step(0, row, float(data.time))
            assert not out.brake, out.fault
            data.ctrl[0] = out.position_rad
        mujoco.mj_step(model, data)
        trace.append((float(data.time), float(data.actuator_force[0]), float(data.qpos[0])))
    assert out.state == 'holding'
    tail = np.array(trace[-500:])
    assert np.mean(tail[:, 1]) == pytest.approx(.04, abs=.006)
    assert np.max(np.abs(np.array(trace)[:, 1])) < .1
    assert abs(float(data.qpos[0])-stop_angle) < .01


@pytest.mark.parametrize('reverse', [False, True])
def test_opening_effort_advances_below_band_and_unloads_above_it(reverse):
    cfg=EffortGripperConfig(0. if reverse else 1., 1. if reverse else 0.,
        opening_effort_nm=.04, effort_gain_rad_per_nm_s=.1,
        max_lead_rad=.002, max_speed_rad_s=.002, progress_tolerance_rad=.0002)
    servo=EffortGripperController(cfg)
    sign=servo.direction
    for i in range(30):
        out=servo.step(1,feedback(i*.004,effort=-sign*.01),i*.004)
    previous=out.position_rad
    assert (previous-.5)*sign < 0
    out=servo.step(1,feedback(.12,effort=-sign*.08),.12)
    assert out.state=='opening_relief' and not out.brake
    assert (out.position_rad-previous)*sign>0
    assert (out.position_rad-.5)*sign<=0
    assert abs(out.position_rad-previous)<=.002*.004+1e-12
    before=out.position_rad
    out=servo.step(1,feedback(.124,effort=-sign*.04),.124)
    assert out.position_rad == before
    # Persistent relief never crosses the measured position toward closure.
    for i in range(32,100):
        out=servo.step(1,feedback(i*.004,effort=-sign*.08),i*.004)
        assert not out.brake and (out.position_rad-.5)*sign<=0


def test_opening_effort_slow_progress_does_not_use_endpoint_tolerance_as_stall_threshold():
    servo=controller(opening_effort_nm=.04, max_lead_rad=.002, max_speed_rad_s=.002,
        progress_tolerance_rad=.0002, endpoint_tolerance_rad=.001)
    for i in range(1000):
        t=i*.004
        q=.5-.0004*t
        out=servo.step(1,feedback(t,q=q,effort=-.03,speed=-.0004),t)
        assert not out.brake,out.fault
    assert out.state=='opening'


def test_opening_effort_oscillation_does_not_reset_directional_progress_watchdog():
    servo=controller(opening_effort_nm=.04, max_lead_rad=.004, max_speed_rad_s=.002,
        progress_tolerance_rad=.0002)
    for i in range(260):
        t=i*.004
        # Motion in the wrong direction then back is not progress toward open.
        q=.5+(.0003 if (i//20)%2 else 0.)
        out=servo.step(1,feedback(t,q=q),t)
        if out.brake:break
    assert out.brake and 'No measured gripper progress' in out.fault


@pytest.mark.parametrize('value',[0.,.095,.1,float('nan'),True])
def test_opening_effort_invalid_band_is_rejected(value):
    with pytest.raises(ValueError):controller(opening_effort_nm=value, max_torque_nm=.1)


def test_opening_effort_duplicate_does_not_repeat_relief_and_trip_stays_latched():
    servo=controller(opening_effort_nm=.04, max_torque_nm=.1)
    servo.step(1,feedback(0.),0.)
    servo.step(1,feedback(.004),.004)
    row=feedback(.008,effort=-.08)
    first=servo.step(1,row,.008)
    assert servo.step(1,row,.009)==first
    assert servo.step(1,feedback(.012,effort=-.1),.012).brake
    assert servo.step(1,feedback(.016),.016).brake


@pytest.mark.parametrize('reverse', [False, True])
def test_trajectory_error_unwinds_without_large_step_then_stays_in_regulation(reverse):
    c = EffortGripperController(EffortGripperConfig(
        closed_rad=0. if reverse else 1., open_rad=1. if reverse else 0.,
        position_ramp_approach=True, max_speed_rad_s=5., max_step_s=.004,
        max_lead_rad=.005, regulation_speed_rad_s=.05,
        closing_acceleration_rad_s2=None))
    q = .5
    previous = q
    saw_unwind = False
    saw_recovered = False
    for i in range(200):
        t = i*.004
        # Give the approach enough low-load time to accumulate target error,
        # then provide an obstruction and, after retreat, an unloaded motor.
        effort = 0. if i < 8 else (.08 if not saw_unwind else .04)
        out = c.step(0, GripperFeedback(q, 0., c.direction*effort, t), t)
        assert not out.brake, out.fault
        assert abs(out.position_rad-previous) <= .020000001
        if out.reference_reconciling:
            saw_unwind = True
            assert (out.position_rad-previous)*c.direction < 0
        if saw_unwind and not out.reference_reconciling:
            saw_recovered = True
            assert out.effort_regulating
            assert abs(out.position_rad-q) <= .005000001
            assert abs(out.reference_velocity_rad_s) <= .050000001
        previous = out.position_rad
    assert saw_unwind and saw_recovered


@pytest.mark.parametrize('reverse', [False, True])
def test_high_effort_continues_unloading_past_old_retreat_anchor(reverse):
    # Regression for 203112: actual retreat passed the old fixed 0.002-rad
    # anchor. The new controller has no anchor or 0.5-s recovery transition.
    c = EffortGripperController(EffortGripperConfig(
        closed_rad=0. if reverse else 1., open_rad=1. if reverse else 0.,
        hold_torque_nm=.1, max_speed_rad_s=5., regulation_speed_rad_s=.05,
        max_lead_rad=.005, closing_acceleration_rad_s2=None))
    sign = c.direction
    for i in range(40):
        t = i*.004
        out = c.step(0, feedback(t, effort=sign*.33), t)
    assert out.effort_regulating
    previous = out.position_rad
    old_anchor = .5-sign*.002
    for i in range(1, 200):
        t += .004
        q = .5-sign*min(.0033, i*.00006)
        out = c.step(0, feedback(t, q=q, effort=sign*.33), t)
        assert not out.brake, out.fault
        assert out.state == 'regulating'
        assert (out.position_rad-previous)*sign <= 1e-12
        assert abs(out.position_rad-previous) <= .05*.004+1e-12
        previous = out.position_rad
    assert (out.position_rad-old_anchor)*sign < -.003
    assert (out.position_rad-q)*sign < 0



@pytest.mark.parametrize('end', ['cancel', 'stale'])
def test_continued_regulation_after_three_seconds_still_honors_cancel_and_freshness(end):
    servo = controller()
    for i in range(1000):
        t = i*.004
        row = feedback(t, effort=.2)
        out = servo.step(0, row, t)
        assert not out.brake
    assert out.effort_regulating and out.regulation_unsettled_s > 3.
    if end == 'cancel':
        out = servo.step(None, row, t+.004)
        assert out.brake and out.fault is None
    else:
        out = servo.step(0, row, t+.051)
        assert out.brake and 'expired' in out.fault
