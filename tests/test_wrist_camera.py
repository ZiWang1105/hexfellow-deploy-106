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


def physical_builder(side=None):
    from pathlib import Path
    from mobile_manipulator.hardware.wrist_observation import WristGeometry
    root=Path(__file__).resolve().parents[3]
    candidate, serial = {
        None: ('realsense_d435i_right_candidate_20261006.json', '243222074562'),
        'left': ('realsense_d405_left_candidate_20261008.json', '230422271628'),
        'right': ('realsense_d435i_right_candidate_20261008.json', '243222074562'),
    }[side]
    return WristGeometry(root/'third_party/hexfellow-deploy/calibration/config/camera'/candidate,
        serial,root/'third_party/UMI_Hardware/urdf/maver_x4_gp80_zed_40deg.urdf',
        root/'configs/robot/maver_gp80.yaml')


def physical_feedback(stamp):
    from mobile_manipulator.robot.joints import JOINTS
    return dict(timestamp_s=stamp,joint_positions={n:.1 for n in JOINTS},
                joint_velocities={n:0. for n in JOINTS},gripper_state=dict(left='open',right='closed'))


def test_calibrated_rgb_topic_atomically_roundtrips_image_and_geometry():
    builder=physical_builder()
    topic=RGBTopic('wrist_geometry_'+uuid.uuid4().hex,width=8,height=6)
    topic.initialize()
    reader=RGBTopic.attach(topic.name,width=8,height=6)
    try:
        for stamp in (12.,12.1):
            value=builder.observation(np.full((6,8,3),int(stamp*10),np.uint8),stamp,
                physical_feedback(stamp-.01),[[100.,0,4.],[0,100.,3.],[0,0,1.]])
            assert topic.put(value)
            received=reader.get()
            np.testing.assert_array_equal(received.pop('rgb'),value.pop('rgb'))
            assert received==value
    finally:
        reader.clean_up();topic.clean_up()


@pytest.mark.parametrize('side', ['left', 'right'])
def test_native_capture_binds_serial_matches_encoders_and_uses_active_intrinsics(monkeypatch,tmp_path,side):
    import json
    import time
    import sys
    from pathlib import Path
    from types import SimpleNamespace
    from hexfellow_deploy.wrist_camera import NativeWristCamera
    from hexfellow_deploy import camera_sync
    from mobile_manipulator.hardware import config as deployment
    root=Path(__file__).resolve().parents[3]
    builder=physical_builder(side)
    serial=builder.metadata['camera_serial']
    filename=f"realsense_{'d405' if side=='left' else 'd435i'}_{side}_candidate_20261008.json"
    monkeypatch.setattr(deployment,'load_deployment',lambda *a,**kw:dict(robot=dict(
        urdf=root/'third_party/UMI_Hardware/urdf/maver_x4_gp80_zed_40deg.urdf',
        assembly=root/'configs/robot/maver_gp80.yaml',camera_encoder_max_skew_s=.08,camera_max_age_s=.25)))
    state={};captured=[]
    class Sampler:
        error=None
        def __init__(self,topic,history):state['history']=history
        def start(self):return self
        def close(self):state['sampler_closed']=True
    monkeypatch.setattr(camera_sync,'FeedbackSampler',Sampler)
    class Images:
        width=8; height=6
        def put(self,value):captured.append(value);return True
    camera=NativeWristCamera('test_wrist',Images(),tmp_path,serial=serial,
        workspace=str(root),deployment_config='unused',feedback=object(),
        calibration='third_party/hexfellow-deploy/calibration/config/camera/'+filename)
    intr=SimpleNamespace(fx=120.,fy=121.,ppx=4.,ppy=3.,coeffs=[0.]*5)
    device=SimpleNamespace(get_info=lambda field:serial if field=='serial' else 'test device',query_sensors=lambda:[])
    profile=SimpleNamespace(get_device=lambda:device,get_stream=lambda stream:SimpleNamespace(
        as_video_stream_profile=lambda:SimpleNamespace(get_intrinsics=lambda:intr)))
    class Pipeline:
        def start(self,config):return profile
        def stop(self):state['pipeline_stopped']=True
        def wait_for_frames(self,**kw):
            count=state.get('frames',0)+1;state['frames']=count
            if count==2:
                now=time.monotonic()
                state['history'].add(dict(measured=physical_feedback(now-.005)),now)
                camera.signal_stop()
            return SimpleNamespace(get_color_frame=lambda:SimpleNamespace(get_data=lambda:np.zeros((6,8,3),np.uint8)))
    class Config:
        def enable_device(self,serial):state['serial']=serial
        def enable_stream(self,*args):state['stream']=args
    monkeypatch.setitem(sys.modules,'pyrealsense2',SimpleNamespace(pipeline=Pipeline,config=Config,
        stream=SimpleNamespace(color='color'),format=SimpleNamespace(rgb8='rgb8'),
        camera_info=SimpleNamespace(serial_number='serial',name='name'),option=SimpleNamespace(frames_queue_size='queue')))
    camera.run()
    assert state['sampler_closed'] and state['pipeline_stopped']
    assert state['serial']==serial
    assert captured[0]['geometry']['arm']==side
    assert len(captured)==1
    g=captured[0]['geometry']
    assert g['fixed_camera_intrinsics']==[[120.,0,4.],[0,121.,3.],[0,0,1.]]
    assert abs(g['encoder_timestamp_s']-captured[0]['received_s'])<.08
    assert g['calibration_sha256']==builder.metadata['calibration_sha256']
    assert (tmp_path/'wrist_calibration.json').read_bytes()==builder.calibration_bytes
    health=json.loads((tmp_path/'wrist_camera_health.json').read_text())
    assert health['unmatched_frames']==1 and health['publications']==1 and health['error'] is None
