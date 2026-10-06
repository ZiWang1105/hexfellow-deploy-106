"""Transient plant contacts must not disappear between 15 Hz replay frames."""
import json
from pathlib import Path
from types import SimpleNamespace

import mujoco

from hexfellow_deploy.physics_audit import PhysicsAudit, pickup_evaluation
from hexfellow_deploy.workspace import activate

ROOT = activate(Path(__file__).resolve().parents[3])
from mobile_manipulator.sim.recording import SceneRecorder


def fixture(tmp_path):
    model=mujoco.MjModel.from_xml_string('''<mujoco><option timestep="0.002" gravity="0 0 0"/>
      <worldbody><site name="left_tcp" pos="0 0 .25"/><site name="right_tcp" pos="0 1 .25"/>
      <body name="entity_apple_body" pos="0 0 .05"><freejoint name="object"/>
        <geom type="sphere" size=".05" mass=".1"/></body>
      <body name="entity_table_body" pos="2 0 .25"><geom name="table" type="box" size=".03 .03 .03"/></body>
      <body name="robot_probe" pos="2 1 .25"><freejoint name="probe_joint"/><geom name="probe" type="sphere" size=".03"/></body>
      <body name="left_jaw_positive" mocap="true" pos=".052 0 .25"><geom type="box" size=".005 .05 .05"/></body>
      <body name="left_jaw_negative" mocap="true" pos="-.052 0 .25"><geom type="box" size=".005 .05 .05"/></body>
      </worldbody></mujoco>''')
    data=mujoco.MjData(model)
    mujoco.mj_forward(model,data)
    audit=PhysicsAudit(model,data)
    recorder=SceneRecorder(SimpleNamespace(model=model,data=data),tmp_path,video=False,sensors=False)
    return model,data,audit,recorder


def test_single_physics_step_collision_between_recorded_frames_is_captured(tmp_path):
    model,data,audit,recorder=fixture(tmp_path)
    try:
        for step in range(40):
            data.time=step*.002
            data.qpos[int(model.joint('probe_joint').qposadr[0])+1]=0. if step==15 else 1.
            mujoco.mj_forward(model,data)
            audit.step(data)
            recorder.step()
        assert recorder.robot_static_contact_frames==0
        report=audit.report()
        assert report['robot_static_contact_steps']==1
        assert report['first_robot_static_contact']['time_s']==.030
        assert report['complete'] is True
        audit.save(tmp_path)
        assert json.loads((tmp_path/'physics_evaluation.json').read_text())['samples']==40
    finally:
        recorder.close()


def test_two_millisecond_jaw_contact_loss_resets_ten_second_retention(tmp_path):
    model,data,audit,recorder=fixture(tmp_path)
    try:
        data.qpos[2]=.25
        loss_step=5015
        for step in range(5051):
            data.time=step*.002
            data.mocap_pos[0,0]=.2 if step==loss_step else .052
            mujoco.mj_forward(model,data)
            audit.step(data)
            recorder.step()
        assert recorder.last_evaluation['objects'][0]['retained_s']>=10.
        report=audit.report()
        assert 0.<report['objects'][0]['retained_s']<.1
        assert report['robot_static_contact_steps']==0
        scored=pickup_evaluation(report,'entity_apple_body',final_hold=True,completed_budget=True)
        assert scored['accepted'] is False
        assert scored['retained_pickup'] is False
    finally:
        recorder.close()


def test_missing_physics_sample_resets_retention_and_invalidates_coverage(tmp_path):
    model,data,audit,recorder=fixture(tmp_path)
    try:
        data.qpos[2]=.25
        for timestamp in (0.,.002,10.1):
            data.time=timestamp
            mujoco.mj_forward(model,data)
            audit.step(data)
        assert audit.report()['complete'] is False
        assert audit.report()['objects'][0]['retained_s']==0.
    finally:
        recorder.close()


def test_missing_initial_physics_samples_invalidates_coverage(tmp_path):
    model,data,audit,recorder=fixture(tmp_path)
    try:
        data.time=.02
        mujoco.mj_forward(model,data)
        audit.step(data)
        assert audit.report()['complete'] is False
    finally:
        recorder.close()
