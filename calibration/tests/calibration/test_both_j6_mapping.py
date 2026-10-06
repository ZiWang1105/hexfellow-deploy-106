"""Direction changes must affect feedback, commands, efforts and bounds together."""
import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace
import numpy as np
import yaml
from policy_sim.hardware.whole_body import RobotMapping
from policy_sim.control.mit_command import MitCommand
from policy_sim.control.robot import LEFT_ARM_JOINTS,RIGHT_ARM_JOINTS

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'tools/camera'))
from replay_joint_mapping import remap_q

OLD=ROOT/'config/hardware/hexfellow_gp80_highspeed_right_j6.json'
NEW=ROOT/'config/hardware/hexfellow_gp80_highspeed_both_j6.json'


def test_only_left_j6_changed_and_default_sdk_targets_preserved():
    old,new=[RobotMapping.load(p) for p in (OLD,NEW)]
    expected=json.loads(json.dumps(old.data['arms']));expected['left']['signs'][5]=-1
    assert new.data['arms']==expected
    a,b=[yaml.safe_load((p.parent/m.data['default_pose']['path']).read_text()) for p,m in zip((OLD,NEW),(old,new))]
    for side in ('left','right'):
        oa,na=old.data['arms'][side],new.data['arms'][side]
        np.testing.assert_allclose((np.array(a[side+'_arm_rad'])-oa['offsets_rad'])*oa['signs'],
            (np.array(b[side+'_arm_rad'])-na['offsets_rad'])*na['signs'],atol=1e-15)
    np.testing.assert_array_equal(remap_q(a['metadata']['captured_model_q'],old.data,new.data),b['metadata']['captured_model_q'])
    capture=b['metadata']['capture_mapping']
    assert capture['arms']==new.data['arms']
    assert hashlib.sha256(json.dumps(capture,sort_keys=True,separators=(',',':')).encode()).hexdigest()==b['metadata']['capture_mapping_sha256']
    assert hashlib.sha256((NEW.parent/new.data['default_pose']['path']).read_bytes()).hexdigest()==new.data['default_pose']['sha256']


def test_position_velocity_effort_and_limits_use_same_left_sign():
    old,new=[RobotMapping.load(p) for p in (OLD,NEW)]
    pos=np.linspace(.1,.7,14)
    cmd=MitCommand((*LEFT_ARM_JOINTS,*RIGHT_ARM_JOINTS),pos,np.full(14,.2),np.full(14,.3),np.full(14,20),np.full(14,2),1.)
    for side in ('left','right'):
        a,b=old.arm_command(side,cmd),new.arm_command(side,cmd)
        for key in a:
            expected=a[key].copy()
            if side=='left' and key in ('position','velocity','torque_ff'):expected[5]*=-1
            np.testing.assert_allclose(b[key],expected)
        q,v=new.arm_state(side,b['position'],b['velocity'])
        np.testing.assert_allclose(q,pos[:7] if side=='left' else pos[7:])
        np.testing.assert_allclose(v,.2)
        np.testing.assert_allclose(new.arm_effort(side,b['torque_ff']),.3)
    health={s:{'joint_limits':[[-.4,1.2]]*7} for s in ('left','right')};health['theta']={'range':[0,.422]}
    limits=new.physical_bounds(SimpleNamespace(health=health))
    assert limits[LEFT_ARM_JOINTS[5]]==limits[RIGHT_ARM_JOINTS[5]]==(-1.2,.4)


def test_offline_replay_reinterprets_measurements_and_is_reversible():
    old,new=[RobotMapping.load(p) for p in (OLD,NEW)]
    q=np.linspace(-.8,.9,18);q[9]=np.deg2rad(-41.22235107421875)
    expected=q.copy();expected[9]*=-1
    corrected=remap_q(q,old.data,new.data)
    np.testing.assert_allclose(corrected,expected,atol=1e-15)
    np.testing.assert_allclose(remap_q(corrected,new.data,old.data),q,atol=1e-15)
    np.testing.assert_allclose(remap_q(q,old.data,new.data,velocity=True),expected,atol=1e-15)
