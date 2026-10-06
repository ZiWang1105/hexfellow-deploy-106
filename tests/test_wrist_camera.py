"""RGB-only wrist observations must not acquire invented calibration."""
import uuid
import numpy as np
import pytest
from hexfellow_deploy.topics import RGBTopic
from hexfellow_deploy.policy import wrist_context


def test_rgb_topic_roundtrip_and_contention_preserve_capture_time():
    topic = RGBTopic('wrist_'+uuid.uuid4().hex,width=8,height=6)
    topic.initialize()
    reader = RGBTopic.attach(topic.name,width=8,height=6)
    image = np.arange(144,dtype=np.uint8).reshape(6,8,3)
    try:
        assert topic.put(dict(rgb=image,received_s=12.5))
        value = reader.get()
        assert set(value) == {'rgb','received_s'} and value['received_s']==12.5
        np.testing.assert_array_equal(value['rgb'],image)
        with topic._hold(None):
            assert reader.get() is None
        assert reader.get()['received_s']==12.5
        with pytest.raises(ValueError,match='resolution'):
            topic.put(dict(rgb=image[:3],received_s=12.6))
    finally:
        reader.clean_up()
        topic.clean_up()


@pytest.mark.parametrize('stamp,head,expected',[(10.,10.,True),(9.,10.,False),(10.1,10.,False),(9.8,10.,False)])
def test_policy_waits_for_fresh_nearby_wrist_frame(stamp,head,expected):
    class Topic:
        def get(self):return dict(rgb=np.zeros((2,3,3),np.uint8),received_s=stamp)
    assert (wrist_context(Topic(),head,now=10.) is not None) == expected


def test_missing_wrist_does_not_silently_become_head_only():
    class Topic:
        def get(self):return None
    assert wrist_context(Topic(),10.,10.) is None
