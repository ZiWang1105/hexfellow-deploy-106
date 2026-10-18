"""Physical-input admission tested without opening any device."""
import copy
import json
from pathlib import Path

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from hexfellow_deploy.actuation import deployment
from hexfellow_deploy.workspace import build_shared_executor
from hexfellow_deploy.manual_tcp import ManualTCPControl
from hexfellow_deploy.hardware_interactive import make_hardware_display, apply_encoders
from hexfellow_deploy.interactive import tcp_targets, paired_action
from hexfellow_deploy.simulation import SimulationPlant

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def setup(tmp_path):
    _, _, profile, metadata = deployment(ROOT, Path('configs/hardware/desktop.yaml'))
    plant = SimulationPlant(ROOT/'configs/scenes/apple_on_table.yaml', tmp_path/'plant',
                            effort_limits=metadata['effort_limits_nm'], profile=profile, realtime=False)
    executor = build_shared_executor(metadata, .02)
    measured = copy.deepcopy(plant.snapshot['measured'])
    measured['timestamp_s'] = 1.
    executor.ik.control_clock = lambda: measured['timestamp_s']
    executor.ik.initialize_reference(measured, stationary_reference=True)
    yield executor, measured, metadata, plant
    plant.close()


def target(executor, measured, identifier=2, created=1.01):
    executor.kinematics.update(measured['joint_positions'])
    action = {}
    for side in ('left','right'):
        pose = executor.kinematics.pose(side)
        action[side] = dict(position_m=pose.translation.tolist(),
                           quaternion_xyzw=Rotation.from_matrix(pose.rotation).as_quat().tolist(), gripper_open=0)
    action['right']['position_m'][2] += .02
    return dict(id=identifier, operation='interactive_target', created_s=created,
                expires_s=created+.25, session=1, action=action)


def test_display_uses_physical_encoders_and_same_tcp(setup, tmp_path):
    executor, measured, metadata, _ = setup
    model, data = make_hardware_display(metadata, tmp_path)
    apply_encoders(model, data, measured)
    executor.kinematics.update(measured['joint_positions'])
    for side, pose in tcp_targets(model, data).items():
        expected = executor.kinematics.pose(side)
        np.testing.assert_allclose(pose['position_m'], expected.translation, atol=1e-7)
        np.testing.assert_allclose(Rotation.from_quat(pose['quaternion_xyzw']).as_matrix(), expected.rotation, atol=1e-7)
    assert not any('mobile_base' in (model.joint(i).name or '') for i in range(model.njnt))


def test_manual_target_rates_and_no_jaw_base_or_lift_commands(setup):
    executor, measured, _, _ = setup
    manual = ManualTCPControl(executor, .02)
    manual.receive(dict(id=1, operation='interactive_enable', created_s=1.), measured, 1.)
    start = manual.targets['right'].translation.copy()
    manual.receive(target(executor, measured), measured, 1.01)
    manual.step(measured, 1.02)
    assert np.linalg.norm(manual.targets['right'].translation-start) == pytest.approx(min(.02, manual.linear_speed*.02))
    command, _ = executor.tick(measured)
    assert command['gripper_open'] == dict(left=None,right=None)
    assert command['base_velocity_body'] == [0.,0.,0.]
    from hexfellow_deploy.robot.joints import LIFT
    assert command['joint_positions'][LIFT] == pytest.approx(measured['joint_positions'][LIFT], abs=1e-8)
    manual.step(measured, 1.261)
    assert not manual.report()['enabled']
    assert 'expired' in manual.reason
    late = target(executor, measured, 3, 1.27)
    with pytest.raises(ValueError, match='fresh E'):
        manual.receive(late, measured, 1.27)


@pytest.mark.parametrize('mutation', ['stale','future','session','long_lease','navigation','replay'])
def test_invalid_input_stops_and_requires_enable(setup, mutation):
    executor, measured, _, _ = setup
    manual = ManualTCPControl(executor, .02)
    manual.receive(dict(id=1, operation='interactive_enable', created_s=1.), measured, 1.)
    request = target(executor, measured)
    if mutation == 'stale': request['created_s'] = .5
    if mutation == 'future': request['created_s'] = 2.
    if mutation == 'session': request['session'] = 22
    if mutation == 'long_lease': request['expires_s'] = 10.
    if mutation == 'navigation': request['action']['navigation'] = dict(object_xy_m=[1,0],yaw_rad=0)
    if mutation == 'replay': request['created_s'] = 1.
    with pytest.raises(ValueError): manual.receive(request, measured, 1.02)
    assert manual.session is None and manual.goals is None
    assert executor.gripper == dict(left=None,right=None)


def test_hardware_build_requires_authorization_and_forces_manual_owner(tmp_path, monkeypatch):
    from hexfellow_deploy.cli import build
    monkeypatch.setenv('HEXFELLOW_WORKSPACE', str(ROOT))
    monkeypatch.setenv('HEXFELLOW_OUTPUT', str(tmp_path))
    monkeypatch.setenv('HEXFELLOW_TOPIC_PREFIX', 'test_manual_hardware_')
    # Admission fails before component creation or device access.
    with pytest.raises(ValueError, match='Hardware requires'):
        build(ROOT/'configs/simulation.yaml', ['interface.backend.backend=hardware',
              'interface.backend.execute=true'], interactive_ik=True, camera=False)


def test_preflight_never_starts_manager_or_devices(tmp_path, monkeypatch, capsys):
    import sys
    from hexfellow_deploy import cli, actuation
    monkeypatch.setattr(sys, 'argv', ['hexfellow-deploy', '--preflight', '--output', str(tmp_path/'out')])
    monkeypatch.setattr(actuation, 'deployment', lambda *a, **k: (None,None,None,dict(profile='test', profile_sha256='abc', actuation_limits={'dt_s':.01})))
    monkeypatch.setattr(cli, 'build', lambda *a, **k: pytest.fail('Preflight constructed runtime'))
    cli.main()
    assert json.loads((tmp_path/'out/preflight.json').read_text())['devices_opened'] is False


def test_large_drag_and_reference_lag_do_not_cancel_input(setup):
    executor, measured, _, _ = setup
    manual = ManualTCPControl(executor, .02)
    manual.receive(dict(id=1, operation='interactive_enable', created_s=1.), measured, 1.)
    request = target(executor, measured)
    initial = Rotation.from_quat(request['action']['right']['quaternion_xyzw'])
    request['action']['right']['quaternion_xyzw'] = (initial*Rotation.from_rotvec([1.,0,0])).as_quat().tolist()
    request['action']['right']['position_m'][0] += .5
    manual.receive(request, measured, 1.01)
    manual.step(measured, 1.02)
    assert (initial.inv()*Rotation.from_matrix(manual.targets['right'].rotation)).magnitude() == pytest.approx(min(1., manual.angular_speed*.02))
    # An unshaped input goal is not servo lag.
    manual.targets['right'].translation[0] += .04
    manual.step(measured, 1.03)
    assert manual.session is not None
    from types import SimpleNamespace
    positions = executor.ik.core.configuration.q.copy()
    positions[executor.ik.indices['maver_x4_asm_URDF_joint_1']] += .04
    executor.ik.last_result = SimpleNamespace(positions=positions)
    manual.step(measured, 1.04)
    assert manual.session == 1 and manual.reason == 'dragging'
