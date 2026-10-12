"""Real display geometry, frame conversion and shared-controller admission."""
import copy
from pathlib import Path

import numpy as np
import pytest

from hexfellow_deploy.workspace import activate, build_shared_executor
ROOT = activate(Path(__file__).resolve().parents[1])
from hexfellow_deploy.actuation import deployment, StreamContract, encode_mit
from hexfellow_deploy.interactive import (
    make_display_model, apply_snapshot, tcp_targets, write_handles, read_drag, paired_action, changed)
from hexfellow_deploy.simulation import SimulationPlant


@pytest.fixture
def display(tmp_path):
    _, _, profile, metadata = deployment(ROOT, Path('configs/hardware/desktop.yaml'))
    plant = SimulationPlant(ROOT/'configs/scenes/apple_on_table.yaml', tmp_path,
                            effort_limits=metadata['effort_limits_nm'], profile=profile)
    try:
        original = (tmp_path/'plant.xml').read_bytes()
        model, data = make_display_model(tmp_path/'plant.xml')
        assert (tmp_path/'plant.xml').read_bytes() == original
        assert model.nq == plant.model.nq and model.nv == plant.model.nv
        apply_snapshot(model, data, plant.snapshot)
        yield plant, model, data, metadata
    finally:
        plant.close()


def test_targets_are_noncolliding_and_only_selected_mocap_can_change(display):
    plant, model, data, _ = display
    targets = tcp_targets(model, data)
    write_handles(model, data, targets)
    selected = model.body('right_drag_target')
    for side in ('left', 'right'):
        body = model.body(f'{side}_drag_target')
        geoms = model.geom_bodyid == body.id
        assert np.all(model.geom_contype[geoms] == 0)
        assert np.all(model.geom_conaffinity[geoms] == 0)
        data.mocap_pos[body.mocapid[0], 0] += .02
    for active, body, reset in [(False, selected.id, False), (True, 0, False), (True, selected.id, True)]:
        assert not changed(targets, read_drag(model, data, targets, body, active, reset=reset))
    result = read_drag(model, data, targets, selected.id, True)
    np.testing.assert_allclose(result['left']['position_m'], targets['left']['position_m'])
    np.testing.assert_allclose(result['right']['position_m'], targets['right']['position_m']+[.02, 0, 0])
    np.testing.assert_array_equal(plant.snapshot['qpos'], data.qpos)


def test_world_to_base_conversion_and_shared_ik_stream(display):
    import mujoco
    from scipy.spatial.transform import Rotation
    from hexfellow_deploy.robot.joints import JOINTS, LIFT
    from hexfellow_deploy.interfaces.contract import Action
    plant, model, data, metadata = display
    # Nonzero rendered base translation/yaw must never leak into TCP commands.
    for name, value in zip(plant.base_joints, (.4, -.2, .7)):
        data.qpos[model.joint(name).qposadr[0]] = value
    mujoco.mj_forward(model, data)
    targets = tcp_targets(model, data)
    action = paired_action(model, data, targets, dict(left=0, right=1))
    executor = build_shared_executor(metadata, .02)
    measured = copy.deepcopy(plant.snapshot['measured'])
    executor.kinematics.update(measured['joint_positions'])
    for side in ('left', 'right'):
        pose = executor.kinematics.pose(side)
        np.testing.assert_allclose(action[side]['position_m'], pose.translation, atol=1e-7)
        np.testing.assert_allclose(Rotation.from_quat(action[side]['quaternion_xyzw']).as_matrix(),
                                   pose.rotation, atol=1e-7)
    action['right']['position_m'] = (np.asarray(action['right']['position_m'])+[.02, 0, 0]).tolist()
    now = [1.]
    measured['timestamp_s'] = now[0]
    executor.ik.control_clock = lambda: now[0]
    executor.ik.initialize_reference(measured, stationary_reference=True)
    executor.submit(Action.from_dict(action), measured)
    contract = StreamContract(.1, .1, metadata=metadata, clock=lambda: now[0])
    contract.seed(measured)
    start = dict(measured['joint_positions'])
    for sequence in range(1, 21):
        now[0] += .02
        measured['timestamp_s'] = now[0]
        command, diagnostic = executor.tick(measured)
        result = executor.ik.last_result
        contract.accept(dict(sequence=sequence, created_s=now[0], measured=copy.deepcopy(measured),
            received=dict.fromkeys(('left','right','theta','base'), now[0]), command=command,
            mit=encode_mit(result.mit), reference_velocity={n:float(result.velocity[i])
                for n,i in executor.ik.indices.items()}), measured)
        assert command['joint_positions'][LIFT] == pytest.approx(start[LIFT], abs=1e-10)
    assert any(abs(command['joint_positions'][n]-start[n]) > 1e-5 for n in JOINTS[1:])


def test_window_shutdown_during_publication_preserves_unaccepted_reference(display, monkeypatch, tmp_path):
    import json
    import time
    from hexfellow_deploy import backend
    from hexfellow_deploy.components import ControlComponent
    plant, _, _, metadata = display
    class Topic:
        def get(self, **kwargs):
            return None
        def put(self, value, **kwargs):
            return True
    topic = Topic()
    component = ControlComponent('test_shutdown', topic, topic, topic, topic, topic,
                                 str(ROOT), str(tmp_path), .02, .006)
    class Proxy:
        def __init__(self, *args, **kwargs):
            self.metadata = metadata
            self.feedback_refresh = self.feedback_read = {}
        def read(self):
            measured = copy.deepcopy(plant.snapshot['measured'])
            measured['timestamp_s'] = time.monotonic()
            return measured
        def wait_ready(self, stopped):
            return self.read()
        def received_for(self, measured):
            return dict.fromkeys(('left', 'right', 'theta', 'base'), measured['timestamp_s'])
        def read_for_control(self, **kwargs):
            return self.read()
        def submit(self, *args, **kwargs):
            component.signal_stop()
            raise backend.ControlCancelled('Reference publication cancelled by stop request')
        def acknowledgement(self, sequence):
            return None
        def close(self):
            pass
    monkeypatch.setattr(backend, 'BackendProxy', Proxy)
    component.run()
    final = json.loads((tmp_path/'control_final.json').read_text())
    assert final['phase'] == 'closed' and final['fault'] is None
    assert final['generated'] == 1 and final['accepted'] == 0
    assert final['cancelled_pending'] == final['pending'] == 1
