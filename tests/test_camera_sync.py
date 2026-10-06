"""Physical image/encoder synchronization regressions without live devices."""
from collections import deque
from pathlib import Path
import time

import numpy as np
import pytest

from hexfellow_deploy.workspace import activate

ROOT = activate(Path(__file__).resolve().parents[3])
from hexfellow_deploy.camera_sync import EncoderHistory, FeedbackSampler


def feedback(timestamp, position=1.):
    return dict(measured=dict(timestamp_s=timestamp, joint_positions={'lift': position},
                             joint_velocities={'lift': 0.},
                             gripper_state={'left': 'closed', 'right': 'open'}),
                received={'left': timestamp, 'right': timestamp, 'theta': timestamp})


def test_history_matches_delayed_camera_frame_that_latest_feedback_cannot_match():
    history = EncoderHistory(.25)
    earlier, latest = feedback(99.81, .2), feedback(99.997, .3)
    history.add(earlier, 100.)
    history.add(latest, 100.)
    camera_received = 99.80
    assert abs(latest['measured']['timestamp_s']-camera_received) > .08
    matched = history.closest(camera_received, 100., .08)
    assert matched == earlier
    assert matched['measured']['timestamp_s'] == 99.81
    assert matched['received'] == earlier['received']


def test_history_selects_nearest_sample_and_keeps_owned_feedback():
    history = EncoderHistory(.25)
    original = feedback(10., .2)
    history.add(original, 10.02)
    original['measured']['joint_positions']['lift'] = 900.
    original['received']['theta'] = 900.
    history.add(feedback(10.015, .3), 10.02)
    assert history.closest(10.013, 10.02, .08)['measured']['joint_positions']['lift'] == .3
    selected = history.closest(10., 10.02, .08)
    assert selected['measured']['joint_positions']['lift'] == .2
    assert selected['received']['theta'] == 10.
    selected['measured']['joint_positions']['lift'] = -900.
    assert history.closest(10., 10.02, .08)['measured']['joint_positions']['lift'] == .2


def test_history_rejects_stale_future_and_outside_skew_samples():
    history = EncoderHistory(.25)
    history.add(feedback(20.), 20.)
    assert history.closest(20.09, 20.10, .08) is None
    assert history.closest(20., 20.251, .08) is None
    assert history.closest(20.01, 20., .08) is None
    assert history.add(feedback(20.1), 20.) is False
    assert history.closest(20., 20., .08)['measured']['timestamp_s'] == 20.


@pytest.mark.parametrize('timestamp', [float('nan'), float('inf'), float('-inf')])
def test_history_ignores_invalid_feedback_clock(timestamp):
    history = EncoderHistory(.25)
    assert history.add(feedback(timestamp), 20.) is False
    assert history.closest(20., 20., .08) is None


def test_history_capacity_and_monotonic_order_bound_retained_samples():
    history = EncoderHistory(.25, capacity=3)
    for stamp in (9.80, 9.84, 9.88, 9.92, 9.96):
        history.add(feedback(stamp), 10.)
    assert history.closest(9.80, 10., .001) is None
    assert history.closest(9.84, 10., .001) is None
    for stamp in (9.88, 9.92, 9.96):
        assert history.closest(stamp, 10., .001)['measured']['timestamp_s'] == stamp
    assert history.add(feedback(9.82, 999.), 10.) is False
    assert history.closest(9.82, 10., .001) is None


def test_equal_timestamps_replace_content_without_consuming_history_capacity():
    history = EncoderHistory(.25, capacity=2)
    history.add(feedback(30., .1), 30.02)
    history.add(feedback(30.01, .2), 30.02)
    for position in (.3, .4, .5):
        history.add(feedback(30.01, position), 30.02)
    assert history.closest(30., 30.02, .001)['measured']['joint_positions']['lift'] == .1
    assert history.closest(30.01, 30.02, .001)['measured']['joint_positions']['lift'] == .5


def test_sampler_survives_transient_missing_reads_and_stops_its_reader():
    started = time.monotonic()
    rows = deque([None, feedback(started-.002, .1), None, feedback(started-.001, .2)])
    class Topic:
        def __init__(self):
            self.calls = 0
        def get(self, **kwargs):
            self.calls += 1
            return rows.popleft() if rows else None
    topic = Topic()
    history = EncoderHistory(.25)
    sampler = FeedbackSampler(topic, history, period_s=.001).start()
    try:
        until = time.monotonic()+1.
        matched = None
        while time.monotonic() < until:
            matched = history.closest(started-.001, time.monotonic(), .0005)
            if matched is not None:
                break
            time.sleep(.001)
        assert matched is not None
        assert matched['measured']['joint_positions']['lift'] == .2
        assert sampler.error is None
    finally:
        sampler.close()
    calls_at_close = topic.calls
    time.sleep(.01)
    assert topic.calls == calls_at_close


def test_sampler_exposes_reader_exception_and_closes():
    class Topic:
        def get(self, **kwargs):
            raise RuntimeError('test encoder read failed')
    sampler = FeedbackSampler(Topic(), EncoderHistory(.25), period_s=.001).start()
    try:
        until = time.monotonic()+1.
        while sampler.error is None and time.monotonic() < until:
            time.sleep(.001)
        assert 'test encoder read failed' in sampler.error
    finally:
        sampler.close()


@pytest.mark.parametrize('case', ['recovered_after_gap', 'publication_retry', 'unchanged_frame'])
def test_native_camera_recovers_and_retries_without_duplicate_frames(monkeypatch, case):
    from hexfellow_deploy import camera, camera_sync
    from mobile_manipulator.hardware import camera_stream, config, observations
    from mobile_manipulator.robot import assembly
    class Clock:
        now = 100.
        calls = 0
        def monotonic(self):
            return self.now
        def sleep(self, delay):
            self.calls += 1
            if self.calls > 8:
                raise AssertionError('Camera failed to publish the recovered frame')
            self.now += .260 if case == 'recovered_after_gap' and self.calls == 1 else max(delay, .004)
            if case == 'unchanged_frame' and self.calls == 3:
                component.signal_stop()
    clock = Clock()
    monkeypatch.setattr(camera, 'time', clock)
    monkeypatch.setattr(config, 'load_deployment', lambda *args, **kwargs: dict(
        robot=dict(policy_camera='head', urdf='', assembly='', camera_max_age_s=.25,
                   camera_encoder_max_skew_s=.08), camera_configs={'head': {}}, calibrations={}))
    monkeypatch.setattr(assembly, 'load_assembly', lambda path: None)
    class Kinematics:
        def __init__(self, *args):
            pass
        def update(self, joints):
            self.joints = dict(joints)
        def packet(self):
            return {'joints': self.joints}
    monkeypatch.setattr(observations, 'HardwareKinematics', Kinematics)
    class Sampler:
        instance = None
        def __init__(self, topic, history, period_s=.004):
            self.history, self.error, self.closed = history, None, False
            self.reads = self.misses = 0
            Sampler.instance = self
        def start(self):
            return self
        def close(self):
            self.closed = True
            return self
    monkeypatch.setattr(camera_sync, 'FeedbackSampler', Sampler)
    monkeypatch.setattr(camera, 'FeedbackSampler', Sampler, raising=False)
    class Stream:
        closed = False
        snapshots = 0
        def __init__(self, *args, **kwargs):
            pass
        def start(self):
            return self
        def health(self):
            received = clock.now-.010 if case == 'recovered_after_gap' else 99.990
            if Sampler.instance is not None:
                Sampler.instance.history.add(feedback(clock.now-.003), clock.now)
            return dict(sequence=1 if received == 99.990 else 2, age_s=clock.now-received,
                        received_s=received, available=True, error=None, running=True)
        def snapshot(self):
            self.snapshots += 1
            health = self.health()
            frame = camera_stream.CameraFrame(np.zeros((24, 32, 3), dtype=np.uint8),
                np.ones((24, 32), dtype=np.float32), health['received_s'], health['received_s'],
                dict(fx=30., fy=30., cx=16., cy=12., width=32, height=24))
            return health, frame
        def close(self):
            self.closed = True
    stream = Stream()
    monkeypatch.setattr(camera_stream, 'CameraStream', lambda *args, **kwargs: stream)
    class Topic:
        name = 'feedback'
        def get(self, **kwargs):
            return feedback(clock.now-.003)
    class Images:
        name, width, height = 'images', 32, 24
        def __init__(self):
            self.rows = []
        def put(self, measured, *args):
            self.rows.append(measured)
            if case != 'unchanged_frame' and len(self.rows) == 2:
                component.signal_stop()
            return not (case == 'publication_retry' and len(self.rows) == 1)
    images = Images()
    component = camera.NativeCamera('camera', Topic(), images, str(ROOT), 'unused.yaml', fps=15)
    component.initialize()
    component.run()
    assert len(images.rows) == (1 if case == 'unchanged_frame' else 2)
    assert stream.snapshots == len(images.rows)
    assert images.rows[0]['timestamp_s'] == pytest.approx(99.997)
    if case != 'unchanged_frame':
        assert images.rows[1]['timestamp_s'] == pytest.approx(100.257 if case == 'recovered_after_gap' else 99.997)
    assert component.ready and stream.closed and Sampler.instance.closed
