"""Frozen general-IK research admission; all tests are device-free."""
import copy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from hexfellow_deploy.prepare_general_research import prepare, sha
from hexfellow_deploy.robotics.control.motion_profile import load_motion_profile
from hexfellow_deploy.robotics.hardware.research_profiles import GENERAL_IK_PROFILE, HIGHSPEED_PROFILE
from hexfellow_deploy.robotics.hardware.highspeed_trial import require_research_release
from hexfellow_deploy.robotics.hardware.x7 import HardwareSafetyError

ROOT=Path(__file__).resolve().parents[1]


@pytest.fixture
def source(tmp_path):
    original=ROOT/'configs/hardware/desktop.yaml'
    config=yaml.safe_load(original.read_text())
    for key in ('controller','cameras','base_motion'):
        config[key]=str((original.parent/config[key]).resolve())
    for key in ('urdf','assembly','mapping'):
        config['robot'][key]=str((original.parent/config['robot'][key]).resolve())
    config['camera_calibration']={k:str((original.parent/v).resolve()) if v else None
                                  for k,v in config['camera_calibration'].items()}
    baseline=load_motion_profile(HIGHSPEED_PROFILE)
    release=dict(authorization='operator_requested_research_trial',hardware_qualified=False,
                 profile_id=baseline.profile_id,profile_sha256=baseline.sha256,
                 runtime_sha256=sha(config['robot']['mapping']),source_sha256={'pyproject.toml':sha(ROOT/'pyproject.toml')})
    release_path=tmp_path/'source_release.json';release_path.write_text(json.dumps(release))
    config['robot']['research_release']=str(release_path)
    source=tmp_path/'source.yaml';source.write_text(yaml.safe_dump(config))
    return source


def test_prepare_keeps_evidence_and_rejects_invalid_binding(source,tmp_path):
    output=tmp_path/'research'
    before=source.read_bytes()
    result=prepare(source,output)
    assert result['profile']==GENERAL_IK_PROFILE and result['control_period_s']==.01
    assert result['review_unchanged'] and not result['hardware_qualified'] and not result['devices_opened']
    assert source.read_bytes()==before
    old=json.loads((output/'source_mapping.json').read_text())
    new=json.loads((output/'mapping.json').read_text())
    for key in old:
        if key not in ('commissioning','default_pose'):
            assert new[key]==old[key]
    profile=load_motion_profile(GENERAL_IK_PROFILE)
    release_path=output/'research_release.json'
    require_research_release(profile,release_path,output/'mapping.json')
    from hexfellow_deploy.robotics.hardware.whole_body import RobotMapping
    mapping=RobotMapping.load(output/'mapping.json')
    assert mapping.quest_experiment and mapping.runtime_session
    release=json.loads(release_path.read_text())
    release['source_sha256']['pyproject.toml']='0'*64
    release_path.write_text(json.dumps(release))
    with pytest.raises(HardwareSafetyError,match='binding mismatch'):
        require_research_release(profile,release_path,output/'mapping.json')
    with pytest.raises(FileExistsError): prepare(source,output)


def test_preparation_does_not_accept_changed_mapping(source,tmp_path):
    config=yaml.safe_load(source.read_text())
    release_path=Path(config['robot']['research_release'])
    release=json.loads(release_path.read_text());release['runtime_sha256']='0'*64
    release_path.write_text(json.dumps(release))
    with pytest.raises(ValueError,match='existing unqualified research'):
        prepare(source,tmp_path/'bad')
    assert not (tmp_path/'bad').exists()


def test_reject_wrong_control_period_before_runtime(monkeypatch):
    from omegaconf import OmegaConf
    from hexfellow_deploy import actuation
    from hexfellow_deploy.cli import validate_hardware_configuration
    metadata=dict(profile=GENERAL_IK_PROFILE,actuation_limits={'dt_s':.01})
    monkeypatch.setattr(actuation,'deployment',lambda *a,**k:(None,None,None,metadata))
    config=OmegaConf.create(dict(workspace=str(ROOT),deployment_config='unused',
                                timing=dict(control_period_s=.02,compute_budget_s=.006)))
    with pytest.raises(ValueError,match='differs from selected profile'):
        validate_hardware_configuration(config)
    config.timing.control_period_s=.01
    assert validate_hardware_configuration(config)==metadata


def test_latest_profile_shared_mit_frames(source,tmp_path):
    import numpy as np
    from hexfellow_deploy.actuation import deployment, StreamContract, encode_mit
    from hexfellow_deploy.workspace import build_shared_executor
    from hexfellow_deploy.simulation import SimulationPlant
    from hexfellow_deploy.manual_tcp import ManualTCPControl
    from scipy.spatial.transform import Rotation
    output=tmp_path/'research';prepare(source,output)
    # No execute authorization: real shared controller, simulated device adapter.
    _,_,profile,metadata=deployment(ROOT,output/'desktop.yaml',execute=False)
    executor=build_shared_executor(metadata,.01)
    plant=SimulationPlant(ROOT/'configs/scenes/apple_on_table.yaml',tmp_path/'plant',
        effort_limits=metadata['effort_limits_nm'],profile=profile,realtime=False)
    try:
        measured=copy.deepcopy(plant.snapshot['measured']);measured['timestamp_s']=1.
        clock=[1.];executor.ik.control_clock=lambda:clock[0]
        executor.ik.initialize_reference(measured,stationary_reference=True)
        manual=ManualTCPControl(executor,.01)
        assert manual.linear_speed == 2.0
        assert manual.angular_speed == 4.0
        manual.receive(dict(id=1,operation='interactive_enable',created_s=1.),measured,1.)
        poses=manual.targets
        action={s:dict(position_m=p.translation.tolist(),quaternion_xyzw=Rotation.from_matrix(p.rotation).as_quat().tolist(),gripper_open=0)
                for s,p in poses.items()}
        start_z = float(poses['right'].translation[2])
        action['right']['position_m'][2]+=.01
        contract=StreamContract(.1,.1,metadata=metadata,clock=lambda:clock[0]);contract.seed(measured)
        for seq in range(1,21):
            clock[0]=1.+seq*.01;measured['timestamp_s']=clock[0]
            manual.receive(dict(id=seq+1,operation='interactive_target',created_s=clock[0],expires_s=clock[0]+.25,
                                session=1,action=action),measured,clock[0])
            manual.step(measured,clock[0])
            if seq == 1:
                # 10 mm in one 10 ms input tick exceeds the removed 0.1 m/s cap.
                # The downstream IK still applies its acceleration/jerk envelope.
                assert manual.targets['right'].translation[2]-start_z == pytest.approx(.01)
            command,_=executor.tick(measured);result=executor.ik.last_result
            contract.accept(dict(sequence=seq,created_s=clock[0],measured=measured,
                received=dict.fromkeys(('left','right','theta','base'),clock[0]),command=command,
                mit=encode_mit(result.mit),reference_velocity={n:float(result.velocity[i]) for n,i in executor.ik.indices.items()}),measured)
            assert command['gripper_open']==dict(left=None,right=None)
            assert command['base_velocity_body']==[0.,0.,0.]
        assert executor.ik.report['motion_profile']==GENERAL_IK_PROFILE
        assert executor.ik.report['motion_profile_sha256']==profile.sha256
        assert executor.ik.dt==.01
    finally:
        plant.close()


def test_default_runtime_uses_general_profile_without_hardware_qualification(monkeypatch, tmp_path):
    from omegaconf import OmegaConf
    from hexfellow_deploy.actuation import deployment
    from hexfellow_deploy.workspace import build_shared_executor
    monkeypatch.setenv('HEXFELLOW_WORKSPACE', str(ROOT))
    monkeypatch.setenv('HEXFELLOW_OUTPUT', str(tmp_path))
    monkeypatch.setenv('HEXFELLOW_TOPIC_PREFIX', 'default_profile_test')
    runtime = OmegaConf.load(ROOT/'configs/simulation.yaml')
    config, _, profile, metadata = deployment(ROOT, Path(runtime.deployment_config))
    assert profile.profile_id == GENERAL_IK_PROFILE
    assert runtime.timing.control_period_s == profile.control_dt_s == .01
    executor = build_shared_executor(metadata, runtime.timing.control_period_s)
    assert executor.ik.core.config.target_linear_velocity_limit == 2.
    assert executor.ik.core.config.target_angular_velocity_limit == 4.
    assert config['commissioned'] is False
    mapping = json.loads(Path(config['robot']['mapping']).read_text())
    assert mapping['read_only_candidate'] is True
    assert mapping['review']['mapping_checked'] is False
