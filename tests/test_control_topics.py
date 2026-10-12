"""Exercise the actual DA semaphore and fixed control schemas."""
from copy import deepcopy
from pathlib import Path
import time
import uuid

import numpy as np
import pytest
from hexfellow_deploy.workspace import activate
activate(Path(__file__).resolve().parents[1])
from hexfellow_deploy.control_topics import FeedbackTopic, MeasuredTopic, ReferenceTopic, ReceiptTopic
from hexfellow_deploy.backend import BackendProxy
from hexfellow_deploy.robot.joints import JOINTS, LEFT, RIGHT


def sensor():
    return dict(timestamp_s=245619.320475107,
                joint_positions={n:i*.01 for i,n in enumerate(JOINTS)},
                joint_velocities={n:-i*.02 for i,n in enumerate(JOINTS)},
                gripper_state=dict(left='between', right='open'))


def feedback():
    return dict(measured=sensor(), received=dict(left=245619.320236632,
        right=245619.319158026, theta=245619.313158217, base=245619.318047952),
        publication_sequence=123, published_s=245619.321)


def reference():
    return dict(sequence=181, created_s=245619.320475107,
        measured=sensor(), received=feedback()['received'],
        command=dict(joint_positions=sensor()['joint_positions'],
                     base_velocity_body=[.02, -.01, .03], gripper_open=dict(left=None, right=1)),
        reference_velocity=sensor()['joint_velocities'],
        mit=dict(joint_names=list(LEFT+RIGHT), timestamp_s=245619.320475107,
                 **{k:[i*.13+j for i in range(14)]
                    for j,k in enumerate(('position','velocity','torque_ff','kp','kd'))}))


@pytest.mark.parametrize('cls,value', [(FeedbackTopic,feedback()), (MeasuredTopic,sensor()),
    (ReferenceTopic,reference()), (ReceiptTopic,dict(receipt=dict(sequence=181,
    accepted_at=245619.322,scope='sdk_accepted'), pending=None,latest_sequence=181,fault=False))])
def test_roundtrip_preserves_all_fields_and_source_clocks_without_json(cls, value, monkeypatch):
    topic = cls('numeric_'+uuid.uuid4().hex)
    topic.initialize()
    reader = cls.attach(topic.name)
    try:
        def no_json(*args, **kwargs):
            raise AssertionError('JSON entered numeric control path')
        monkeypatch.setattr('json.dumps', no_json)
        monkeypatch.setattr('json.loads', no_json)
        assert topic.put(value)
        assert reader.get() == value
        # A failed lock acquisition neither publishes a partial frame nor
        # invents a new timestamp for the previous publication.
        with topic._hold(None):
            assert reader.get(timeout=.001) is None
            assert not reader.put(value, timeout=.001)
        assert reader.get() == value
    finally:
        reader.clean_up()
        topic.clean_up()


@pytest.mark.parametrize('mutation', ['joint_order', 'fractional_grip', 'nan', 'extra_sensor', 'missing_source'])
def test_bad_reference_rejected_before_publication(mutation):
    topic = ReferenceTopic('numeric_'+uuid.uuid4().hex)
    topic.initialize()
    value = reference()
    try:
        assert topic.put(value)
        bad = deepcopy(value)
        if mutation == 'joint_order': bad['mit']['joint_names'].reverse()
        if mutation == 'fractional_grip': bad['command']['gripper_open']['left'] = .5
        if mutation == 'nan': bad['mit']['torque_ff'][3] = float('nan')
        if mutation == 'extra_sensor': bad['measured']['world_base'] = [0,0,0]
        if mutation == 'missing_source': del bad['received']['right']
        with pytest.raises(ValueError): topic.put(bad)
        assert topic.get() == value
    finally:
        topic.clean_up()


def test_unknown_schema_is_rejected():
    topic = MeasuredTopic('numeric_'+uuid.uuid4().hex)
    topic.initialize()
    try:
        assert topic.send(topic.ArrayData(time.monotonic(), np.ones(topic.shape)*2))
        with pytest.raises(ValueError, match='version'): topic.get()
    finally:
        topic.clean_up()


def test_numeric_receipt_path_does_not_read_diagnostic_json_until_fault():
    topic = ReceiptTopic('numeric_'+uuid.uuid4().hex)
    topic.initialize()
    proxy = object.__new__(BackendProxy)
    proxy.receipts = topic
    calls = []
    class Status:
        def get(self, **kwargs):
            calls.append(True)
            return {'fault':'device disconnected'}
    proxy.status = Status()
    try:
        assert topic.put(dict(receipt=dict(sequence=1, accepted_at=2.,scope='udp_sent')))
        assert proxy.acknowledgement(1).scope == 'udp_sent'
        assert not calls
        assert topic.put(dict(fault='device disconnected'))
        with pytest.raises(RuntimeError, match='device disconnected'): proxy.acknowledgement(1)
        assert calls
    finally:
        topic.clean_up()
