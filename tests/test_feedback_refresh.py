"""Feedback lock contention must not feed stale cache into stateful PinkIK."""
import threading
import time
import uuid

import pytest
from hexfellow_deploy.workspace import activate
from pathlib import Path
activate(Path(__file__).resolve().parents[3])
from policy_sim.control.feedback_policy import FeedbackAgeError, validate_received
from hexfellow_deploy.backend import BackendProxy
from hexfellow_deploy.reference import ReferenceOwner
from hexfellow_deploy.control_topics import FeedbackTopic
from mobile_manipulator.robot.joints import JOINTS


def row(stamp):
    return dict(measured=dict(timestamp_s=stamp, joint_positions=dict.fromkeys(JOINTS,0.),
                joint_velocities=dict.fromkeys(JOINTS,0.), gripper_state=dict(left='closed',right='open')),
                received=dict.fromkeys(('left', 'right', 'base', 'theta'), stamp))


def proxy_for(topic):
    proxy = object.__new__(BackendProxy)
    proxy.feedback = topic
    proxy.status = type('Status', (), {'get': lambda self: None})()
    proxy.selected = row(time.monotonic()-.055)
    proxy.command_deadline = time.monotonic()+.09
    return proxy


def test_real_shm_busy_read_reproduces_stale_cache_then_refresh_recovers():
    topic = FeedbackTopic('feedback_retry_'+uuid.uuid4().hex)
    topic.initialize()
    proxy = proxy_for(topic)
    locked, release = threading.Event(), threading.Event()
    published = []
    def writer():
        with topic._hold(None):
            locked.set()
            release.wait(1.)
        fresh = row(time.monotonic())
        published.append(fresh)
        assert topic.put(fresh, timeout=.02)
    thread = threading.Thread(target=writer)
    thread.start()
    try:
        assert locked.wait(1.)
        old = proxy.read()
        assert proxy.feedback_read['acquired'] is False
        with pytest.raises(FeedbackAgeError):
            validate_received(proxy.received_for(old), time.monotonic())
        timer = threading.Timer(.012, release.set)
        timer.start()
        owner = ReferenceOwner()
        solves = []
        measured = proxy.read_for_control(timeout_s=.1, min_validity_s=.006)
        owner.generate(lambda: solves.append(measured))
        thread.join(1.)
        timer.join(1.)
        assert proxy.feedback_refresh['retries'] > 0
        assert solves == [published[0]['measured']]
        assert measured['timestamp_s'] == published[0]['measured']['timestamp_s']
        assert proxy.received_for(measured) == published[0]['received']
        validate_received(proxy.received_for(measured), time.monotonic())
    finally:
        release.set()
        thread.join(1.)
        topic.clean_up()


def test_stalled_publication_does_not_reset_prior_command_deadline():
    class StaleTopic:
        def get(self, **kwargs):
            return row(time.monotonic()-.055)
    proxy = proxy_for(StaleTopic())
    deadline = proxy.command_deadline = time.monotonic()+.02
    owner = ReferenceOwner()
    with pytest.raises(RuntimeError, match='health deadline; IK not advanced'):
        measured = proxy.read_for_control(timeout_s=.1, min_validity_s=.006)
        owner.generate(lambda: measured)
    assert proxy.command_deadline == deadline
    assert owner.generated == 0


@pytest.mark.parametrize('condition', ['stop', 'backend_fault', 'future_clock'])
def test_refresh_does_not_hide_stop_device_failure_or_bad_clocks(condition):
    class Topic:
        def get(self, **kwargs):
            return row(time.monotonic()+1 if condition == 'future_clock' else time.monotonic())
    proxy = proxy_for(Topic())
    if condition == 'backend_fault':
        proxy.status.get = lambda: {'fault': 'device disconnected'}
    with pytest.raises((ValueError, RuntimeError)):
        proxy.read_for_control(timeout_s=.1, min_validity_s=.006, stopped=lambda: condition == 'stop')
