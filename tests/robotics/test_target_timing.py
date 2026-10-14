import numpy as np
import pinocchio as pin
import pytest

from hexfellow_deploy.robotics.control.target_motion import (
    CartesianMotionLimits,
    ContinuousTargetMotionPlanner,
    TargetTrajectoryError,
)
from hexfellow_deploy.robotics.control.target_timing import RecordingTimeline, TargetSample


def test_recorded_viewer_drag_keeps_a_feasible_braking_reserve():
    import json
    from pathlib import Path
    from hexfellow_deploy.robotics.control.ik_core import PinkIKCore
    from hexfellow_deploy.robotics.control.motion_profile import controller_config_for_profile, load_motion_profile
    recording = json.loads((Path(__file__).parents[1]/'fixtures/drag_target_planner.json').read_text())
    core = PinkIKCore(config=controller_config_for_profile(
        load_motion_profile('x7-gp80-highspeed-fixed-base-v3')), base_mode='planar')
    p = core._target_planners['right']
    p.reset(pin.SE3(np.asarray(recording['initial_pose'])))
    dt = recording['dt']
    for row in recording['samples']:
        goal = pin.SE3(np.asarray(row['pose']))
        motion, _ = p.step(goal, dt, sample=TargetSample(**row['sample']))
        assert np.linalg.norm(motion.body_twist[:3]) <= p.limits.linear_velocity_m_s+1e-9
        assert np.linalg.norm(motion.body_acceleration[:3]) <= p.limits.linear_acceleration_m_s2+1e-9
        assert np.linalg.norm(motion.body_jerk[:3]) <= p.limits.linear_jerk_m_s3+1e-9
        if p._segment is not None:
            # Validate the full polynomial, not only its sampled endpoint.
            segment = p._segment
            assert p._within_limits(segment.translation_coefficients, segment.rotation_coefficients, segment.duration_s)
    sample = row['sample']
    for tick in range(1, 301):
        motion, _ = p.step(goal, dt, sample=TargetSample(sample['time_s']+tick*dt,
            sample['sequence']+tick, sample['epoch'], sample['quality']))
        assert np.linalg.norm(motion.body_twist[:3]) <= p.limits.linear_velocity_m_s+1e-9
        assert np.linalg.norm(motion.body_acceleration[:3]) <= p.limits.linear_acceleration_m_s2+1e-9
        assert np.linalg.norm(motion.body_jerk[:3]) <= p.limits.linear_jerk_m_s3+1e-9
    assert p.settled
    np.testing.assert_allclose(motion.pose.homogeneous, goal.homogeneous, atol=1e-8)


def test_causal_irregular_clock_never_reads_future_and_preserves_duration():
    data = dict(original_tick=np.arange(4), control_time_ns=np.array([100, 130, 210, 260]) * 1_000_000)
    timeline = RecordingTimeline.load(data, np.arange(4))
    assert timeline.duration == pytest.approx(.16)
    assert timeline.index(.029) == 0
    assert timeline.index(.031) == 1
    assert timeline.index(.109) == 1
    assert timeline.index(.159) == 2
    assert timeline.index(5.) == 3
    assert timeline.index(.2, .25) == 1
    assert timeline.quality == 'recorded_host_monotonic'


def test_old_clock_estimation_is_labelled_and_legacy_is_explicit():
    data = dict(original_tick=np.arange(3), raw_sampled_time_ms=np.array([10., 30., 90.]),
                sample_age_ms=np.array([4., 7., 9.]))
    estimated = RecordingTimeline.load(data, np.arange(3))
    assert estimated.duration == pytest.approx(.085)
    assert estimated.quality.startswith('estimated_')
    assert RecordingTimeline.load(data, np.arange(3), mode='legacy_fixed_20ms').duration == .04
    with pytest.raises(ValueError, match='no usable wall clock'):
        RecordingTimeline.load(dict(original_tick=np.arange(3)), np.arange(3))


@pytest.mark.parametrize('clock', [[0, 1, 1], [0, 2, 1], [0, np.nan, 2]])
def test_invalid_recorded_clocks_fail_closed(clock):
    with pytest.raises(ValueError, match='strictly increasing'):
        RecordingTimeline.load(dict(original_tick=np.arange(3), control_time_ns=clock), np.arange(3))


def planner():
    p = ContinuousTargetMotionPlanner(CartesianMotionLimits(.8, 4, 40, 3, 14, 160),
                                      continuous_tracking=True)
    p.reset(pin.SE3.Identity())
    return p


def test_source_interval_controls_velocity_and_duplicates_do_not_zero_it():
    p = planner()
    pose = pin.SE3.Identity()
    p.step(pose, .02, sample=TargetSample(1., 1))
    pose.translation[0] = .002
    p.step(pose, .02, sample=TargetSample(1.1, 2))
    assert p._previous_velocity_world[0] == pytest.approx(.02)
    saved = p.snapshot()
    p.step(pose, .02, sample=TargetSample(1.1, 2))
    assert p.input_event == 'repeated_packet'
    assert p._previous_velocity_world[0] == pytest.approx(.02)
    p.restore(saved)
    assert p._last_sample.sequence == 2
    with pytest.raises(TargetTrajectoryError, match='out_of_order'):
        p.step(pose, .02, sample=TargetSample(1.05, 3))


def test_epoch_resets_derivatives_and_static_samples_settle():
    p = planner()
    pose = pin.SE3.Identity()
    for i in range(30):
        pose.translation[0] = min(i, 10) * .001
        motion, _ = p.step(pose, .02, sample=TargetSample(i * .02, i))
        assert np.linalg.norm(motion.body_twist[:3]) <= .8 + 1e-6
        assert np.linalg.norm(motion.body_acceleration[:3]) <= 4 + 1e-6
    p.step(pose, .02, sample=TargetSample(0., 0, epoch=1))
    assert np.linalg.norm(p._previous_velocity_world) == 0
    for i in range(1, 100):
        p.step(pose, .02, sample=TargetSample(i * .02, i, epoch=1))
    assert p.settled
    assert np.linalg.norm(p.motion().pose.translation - pose.translation) < 1e-8


def test_duplicate_timeout_is_explicit_braking_and_is_rollback_safe():
    p = planner()
    pose = pin.SE3.Identity()
    p.step(pose, .02, sample=TargetSample(0., 0))
    pose.translation[0] = .001
    p.step(pose, .02, sample=TargetSample(.02, 1))
    for _ in range(6):
        p.step(pose, .02, sample=TargetSample(.02, 1))
    assert p.input_event == 'stale_packet_braking'
    saved = p.snapshot()
    first, _ = p.step(pose, .02, sample=TargetSample(.02, 1))
    p.restore(saved)
    second, _ = p.step(pose, .02, sample=TargetSample(.02, 1))
    np.testing.assert_array_equal(first.body_twist, second.body_twist)


@pytest.mark.parametrize('velocity_damping', [0., .2])
def test_shaped_stationary_target_keeps_feedforward_in_reference_frame(velocity_damping, monkeypatch):
    from hexfellow_deploy.robotics.control.ik_core import PinkControllerConfig, PinkIKCore

    config = PinkControllerConfig(target_planner='c2_tracking_v4', target_feedforward_gain=1.,
                                  target_velocity_feedback_gain=velocity_damping, self_collision_enabled=False)
    core = PinkIKCore(config=config, base_mode='planar')
    q = core.home_q.copy()
    poses = core._command_frame_poses(q)
    core.reset(q, poses, measured_velocity=np.zeros(18), state_timestamp=0.)
    targets = {s: p.copy() for s, p in poses.items()}
    targets['left'].translation += [.03, .01, 0.]
    targets['left'].rotation = targets['left'].rotation @ pin.exp3(np.array([.1, .2, 0.]))
    checked = []
    task = core.frame_tasks['left']
    original = task.set_target_motion

    def inspect(pose, twist, dt):
        motion = core._target_planners['left'].motion()
        assert not core._target_planners['left'].request_is_moving
        assert np.linalg.norm(motion.body_twist) > 0
        current = core.configuration.get_transform_frame_to_world(task.frame)
        epsilon = 1e-5
        forward = pin.log6(current.actInv(pose * pin.exp6(motion.body_twist * epsilon))).vector
        backward = pin.log6(current.actInv(pose * pin.exp6(-motion.body_twist * epsilon))).vector
        expected = (config.target_feedforward_gain + velocity_damping) * (forward - backward) / (2 * epsilon)
        np.testing.assert_allclose(twist, expected, atol=1e-10, rtol=0.)
        checked.append(True)
        original(pose, twist, dt)

    monkeypatch.setattr(task, 'set_target_motion', inspect)
    core.step(q, targets, measured_velocity=np.zeros(18), state_timestamp=.02, control_timestamp=.02)
    assert checked


def test_continuous_rotating_reference_bounds_without_input_prediction(monkeypatch):
    p = planner()

    def forbidden(*args, **kwargs):
        raise AssertionError('input prediction must not be used in v4')

    monkeypatch.setattr(p, '_predict_tracking_destination', forbidden)
    for i in range(250):
        t = .02 * i
        pose = pin.SE3(pin.exp3(np.array([.08 * np.sin(t), .05 * np.sin(1.3*t), .03 * np.cos(t)])),
                       np.array([.02 * np.sin(t), .01 * np.cos(t), .01 * np.sin(1.3*t)]))
        p.step(pose, .02, sample=TargetSample(t, i))
        assert max(p.utilization().__dict__.values()) <= 1.0001


def test_large_host_clock_does_not_delay_exact_control_ticks():
    timeline = RecordingTimeline.load(dict(original_tick=np.arange(3),
        control_time_ns=9_876_543_210_987_654 + np.arange(3, dtype=np.int64) * 20_000_000), np.arange(3))
    assert timeline.index(.02) == 1
    assert timeline.index(.04) == 2
