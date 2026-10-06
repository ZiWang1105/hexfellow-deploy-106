"""Offline checks for the operator-selected right-wrist hypothesis B."""
import hashlib,json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import yaml
from policy_sim.hardware.whole_body import RobotMapping
from policy_sim.control.mit_command import MitCommand
from policy_sim.control.robot import LEFT_ARM_JOINTS,RIGHT_ARM_JOINTS

ROOT=Path(__file__).resolve().parents[2]
OLD=ROOT/'config/hardware/hexfellow_gp80_highspeed_j357.json'
NEW=ROOT/'config/hardware/hexfellow_gp80_highspeed_right_j6.json'


def test_only_right_j6_sign_changes():
    old=RobotMapping.load(OLD);new=RobotMapping.load(NEW)
    assert old.data['arms']['left']==new.data['arms']['left']
    expected=json.loads(json.dumps(old.data['arms']['right']));expected['signs'][5]*=-1
    assert new.data['arms']['right']==expected
    assert new.data['arms']['right']['signs'][6]==-1


def test_saved_default_preserves_sdk_targets_and_capture_metadata():
    old,new=[RobotMapping.load(p) for p in (OLD,NEW)]
    poses=[yaml.safe_load((p.parent/m.data['default_pose']['path']).read_text()) for p,m in zip((OLD,NEW),(old,new))]
    for side,start in [('left',4),('right',11)]:
        om,nm=old.data['arms'][side],new.data['arms'][side]
        for a,b in [(poses[0][side+'_arm_rad'],poses[1][side+'_arm_rad']),
                    (poses[0]['metadata']['captured_model_q'][start:start+7],poses[1]['metadata']['captured_model_q'][start:start+7])]:
            np.testing.assert_allclose((np.array(a)-om['offsets_rad'])*om['signs'],(np.array(b)-nm['offsets_rad'])*nm['signs'],atol=1e-15)
    assert poses[1]['lift_m']==.3
    capture=poses[1]['metadata']['capture_mapping']
    assert capture['arms']==new.data['arms']
    assert hashlib.sha256(json.dumps(capture,sort_keys=True,separators=(',',':')).encode()).hexdigest()==poses[1]['metadata']['capture_mapping_sha256']
    assert hashlib.sha256((NEW.parent/new.data['default_pose']['path']).read_bytes()).hexdigest()==new.data['default_pose']['sha256']


def test_feedback_and_command_conversion_and_asymmetric_joint_limits():
    old,new=RobotMapping.load(OLD),RobotMapping.load(NEW)
    pos=np.linspace(.1,.7,14)
    cmd=MitCommand((*LEFT_ARM_JOINTS,*RIGHT_ARM_JOINTS),pos,np.full(14,.2),np.full(14,.3),np.full(14,20),np.full(14,2),1.)
    left_old=old.arm_command('left',cmd);left_new=new.arm_command('left',cmd)
    for key in left_old:np.testing.assert_array_equal(left_old[key],left_new[key])
    a,b=old.arm_command('right',cmd),new.arm_command('right',cmd)
    for key in ('position','velocity','torque_ff'):
        expected=a[key].copy();expected[5]*=-1;np.testing.assert_allclose(b[key],expected)
    for key in ('kp','kd'):np.testing.assert_array_equal(a[key],b[key])
    q,v=new.arm_state('right',b['position'],b['velocity'])
    np.testing.assert_allclose(q,pos[7:]);np.testing.assert_allclose(v,.2)
    health={s:{'joint_limits':[[-.4,1.2]]*7} for s in ('left','right')};health['theta']={'range':[0,.422]}
    limits=new.physical_bounds(SimpleNamespace(health=health))
    assert limits[RIGHT_ARM_JOINTS[5]]==(-1.2,.4)
    assert limits[LEFT_ARM_JOINTS[5]]==(-.4,1.2)
