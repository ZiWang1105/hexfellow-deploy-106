"""The relocated tools resolve their owned code and never mix wrist frame IDs."""
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import numpy as np
import pytest

ROOT=Path(__file__).resolve().parents[2]
SPEC=importlib.util.spec_from_file_location('calibration_cli', ROOT.parent/'src/hexfellow_deploy/calibration.py')
CLI=importlib.util.module_from_spec(SPEC); SPEC.loader.exec_module(CLI)

def test_calibration_overlay_uses_current_upstream_control():
    import hexfellow_deploy.robotics.calibration.wrist_handeye as calibration
    import hexfellow_deploy.robotics.hardware.session as control
    assert Path(calibration.__file__).is_relative_to(ROOT.parent/'src/hexfellow_deploy/robotics')
    assert Path(control.__file__).is_relative_to(ROOT.parent/'src/hexfellow_deploy/robotics')

def test_relocated_right_plan_payload_and_source_hashes():
    from wrist_calibration_geometry import DEFAULT_PLAN, load_plan
    historical=json.loads((ROOT/'config/camera/realsense_d435i_right_fixed33_plan.json').read_text())
    assert historical['migration']['geometry_payload_unchanged']
    plan=load_plan(DEFAULT_PLAN)
    assert plan['mount_installation']['mount_side_at_default']=='outside_right_arm'
    assert plan['mount_side_validation']['passed']
    assert plan['simulation_validation']['poses_accepted']==30

def test_superseded_inner_side_plan_cannot_launch_calibration():
    from wrist_calibration_geometry import load_plan
    with pytest.raises(ValueError, match='old inner-side wrist plan is superseded'):
        load_plan(ROOT/'config/camera/realsense_d435i_right_fixed33_plan.json')

def test_preview_cannot_relabel_a_different_controller_route(tmp_path):
    (tmp_path/'wrist-plan.json').write_text(json.dumps({'geometry_payload_sha256':'old-inner-side-route'}))
    result=subprocess.run([sys.executable,str(ROOT/'tools/camera/render_wrist_open_gripper.py'),
                           '--sequence-run',str(tmp_path),'--output',str(tmp_path/'preview')],
                          capture_output=True,text=True)
    assert result.returncode!=0
    assert 'Recorded controller sequence belongs to a different mount/pose plan' in result.stderr

def test_left_fit_rejects_right_camera_dataset(tmp_path):
    data=tmp_path/'dataset.json'
    data.write_text(json.dumps({'parent_frame':'maver_x4_asm_mount2','camera':{'serial':'243222074562'}}))
    with pytest.raises(ValueError, match='refusing to relabel'):
        CLI.build_command('left','fit',[str(data),'--output',str(tmp_path/'fit.json')])

def test_left_run_does_not_reuse_right_pose_route():
    left,_=CLI.build_command('left','run',['--mode','live'])
    right,_=CLI.build_command('right','run',['--mode','live'])
    path=Path(left[left.index('--wrist-plan')+1])
    plan=json.loads(path.read_text())
    assert plan['capture_side']=='left'
    assert plan['serial']=='230422271628'
    assert plan['parent_frame']=='maver_x4_asm_mount1'
    assert left[left.index('--wrist-plan')+1]!=right[right.index('--wrist-plan')+1]
    assert left[left.index('--port')+1]=='8774'

def test_camera_workers_use_distinct_serials():
    for side, serial in [('left','230422271628'),('right','243222074562')]:
        cmd,_=CLI.build_command(side,'camera',['--directory','/tmp/camera-only'])
        assert cmd[cmd.index('--serial')+1]==serial
        assert 'calibrate_robot.py' not in ' '.join(cmd)

def test_launcher_forwards_help_without_opening_devices():
    result=subprocess.run([sys.executable,str(ROOT/'cli.py'),'right','run','--help'],
                          capture_output=True,text=True,check=True)
    assert '--wrist-plan' in result.stdout
    assert '--confirm-onsite-ready' in result.stdout

@pytest.mark.parametrize('camera,action,expected',[
    ('head','refine','fit_landmark_extrinsics.py'),
    ('head','overlay','validate_camera_overlay.py'),
    ('head','review','verify_centered_calibration.py'),
    ('head','verify','verify_handeye_ground_truth.py'),
    ('right','plan','wrist_calibration_geometry.py'),
    ('right','preview','render_wrist_open_gripper.py'),
    ('right','verify','verify_wrist_ground_truth.py'),
    ('left','plan','plan_left_wrist_calibration.py'),
    ('left','preview','preview_left_wrist_calibration.py'),
    ('left','verify','verify_wrist_ground_truth.py'),
])
def test_offline_steps_select_owned_tools_without_robot_runner(camera,action,expected):
    command,_=CLI.build_command(camera,action,['--help'])
    assert Path(command[1])==ROOT/'tools/camera'/expected
    assert 'calibrate_robot.py' not in ' '.join(command)

def test_head_validation_targets_head_camera_with_three_left_arm_poses():
    command,_=CLI.build_command('head','validate',['--mode','sim'])
    assert command[command.index('--validation-plan')+1]==str(ROOT/'config/camera/left_hand_validation.json')
    assert command[command.index('--port')+1]=='8773'
    assert '--wrist-plan' not in command

@pytest.mark.parametrize('camera,action',[('right','refine'),('left','validate')])
def test_camera_specific_actions_do_not_borrow_another_camera_workflow(camera,action):
    with pytest.raises(ValueError,match='is not available'):
        CLI.build_command(camera,action,['--help'])

@pytest.mark.parametrize('camera',['head','right','left'])
def test_fit_help_does_not_require_a_dataset_or_open_devices(camera):
    result=subprocess.run([sys.executable,str(ROOT/'cli.py'),camera,'fit','--help'],capture_output=True,text=True,check=True)
    assert '--output' in result.stdout

def test_list_explains_available_and_missing_motion_workflows():
    result=subprocess.run([sys.executable,str(ROOT/'cli.py'),'list'],capture_output=True,text=True,check=True)
    registry=json.loads(result.stdout)
    assert {'refine','validate','overlay'}<=set(registry['head']['supported_actions'])
    assert 'plan' in registry['right']['supported_actions']
    assert registry['left']['supported_actions']==['run','camera','fit','plan','preview','verify']
