"""Startup lift completion and admission use measured feedback on both leaves."""
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from threading import RLock
import numpy as np
import pytest

from hexfellow_deploy.startup_lift import LiftArrival, validate_target
from hexfellow_deploy.workspace import activate
ROOT = activate(Path(__file__).resolve().parents[1])


@pytest.mark.parametrize('target', [float('nan'), float('inf'), 0., .421, -.1])
def test_invalid_target_rejected(target):
    with pytest.raises(ValueError):
        validate_target(target)


def test_arrival_needs_new_stationary_feedback_and_resets_after_motion():
    arrival = LiftArrival(.3, 10.)
    assert not arrival.update(.3, 0., 10., 10.)
    assert not arrival.update(.3, 0., 10., 10.02)  # Re-reading is not a new sample.
    assert not arrival.update(.3, 0., 10.1, 10.1)
    assert not arrival.update(.3, .003, 10.2, 10.2)
    assert not arrival.update(.3, 0., 10.3, 10.3)
    assert not arrival.update(.3, 0., 10.4, 10.4)
    assert arrival.update(.3001, 0., 10.51, 10.51)


@pytest.mark.parametrize('kind', ['cancel', 'stale', 'deadline', 'regression'])
def test_startup_does_not_accept_bad_feedback_or_cancel(kind):
    arrival = LiftArrival(.3, 10.)
    arrival.update(.4, 0., 10., 10.)
    with pytest.raises(RuntimeError):
        if kind == 'cancel': arrival.update(.3, 0., 10.01, 10.01, stopped=True)
        if kind == 'stale': arrival.update(.3, 0., 10., 10.1)
        if kind == 'deadline': arrival.update(.3, 0., 35., 35.)
        if kind == 'regression': arrival.update(.3, 0., 9.99, 10.01)


@pytest.mark.parametrize('flags,target', [(['--prompt','task'], '0.3'),
    (['--prompt','task','--startup-lift-m','.25'], '0.25'),
    (['--prompt','task','--keep-lift-height'], 'null'), (['--hold-only'], 'null'),
    (['--interactive-ik'], 'null')])
def test_launcher_selects_startup_before_carry_but_not_diagnostics(monkeypatch, flags, target):
    from hexfellow_deploy import entrypoints, cli
    captured = []
    monkeypatch.setattr(cli, 'main', lambda: captured.extend(sys.argv))
    entrypoints.main('simulation', ['--output','unused', *flags])
    assert f'interface.backend.startup_lift_m={target}' in captured


def test_lift_startup_authorization_is_one_way_and_finite():
    from hexfellow_deploy.native import ContinuousGate
    from hex_device.generated.public_api_down_pb2 import APIDown
    clock = [10.]
    lift = ContinuousGate('theta', clock=lambda:clock[0], defer_activation=True, startup_lift=True)
    arm = ContinuousGate('left', clock=lambda:clock[0], defer_activation=True, startup_lift=True)
    for gate in (lift, arm):
        gate.transition('initializing'); gate.transition('active')
        assert gate.phase == 'ready'
    clock[0] += 10.
    lift.begin_lift_startup()
    packet = APIDown(); packet.linear_lift_command.target_pos = 100
    lift.authorize(0, clock[0], clock[0]+.1, packet)
    assert lift.filter(packet) is not None
    lift.finish_lift_startup()
    assert lift.phase == 'holding'
    # A copied startup packet becomes BRAKE, never a POSITION after locking.
    assert lift.filter(packet).linear_lift_command.WhichOneof('command') == 'brake'
    with pytest.raises(RuntimeError): lift.begin_lift_startup()
    with pytest.raises(RuntimeError): lift.resume_delay(2)
    arm.finish_startup()
    assert arm.phase == 'active'


def test_native_startup_reaches_target_before_enabling_arms(monkeypatch):
    from hexfellow_deploy.native import StreamEndpoint, ContinuousGate
    from hexfellow_deploy.actuation import deployment
    clock = [10.]
    monkeypatch.setattr('hexfellow_deploy.native.time.monotonic', lambda:clock[0])
    e = object.__new__(StreamEndpoint)
    e.mapping = deployment(ROOT, Path('configs/hardware/desktop.yaml'))[1]
    e.startup_lift_m = .3
    gate = ContinuousGate('theta', clock=lambda:clock[0], defer_activation=True, startup_lift=True)
    gate.transition('initializing'); gate.transition('active')
    e.direct_gates = dict(theta=gate)
    height = [.422]
    writes = []
    e.command_type = SimpleNamespace(POSITION='POSITION')
    def write(kind, value):
        writes.append((kind,value))
        gate._recorded_dispatches.add(gate.sequence)
    e.devices = dict(theta=SimpleNamespace(_pulse_per_rotation=1000000., _max_pos=422000,
        _status_lock=RLock(), _target_pos=None,
        convert_rad_to_positions=lambda q, scale: np.rint(q*scale).astype(int), motor_command=write))
    def state():
        return SimpleNamespace(q=np.r_[np.zeros(3),height[0],np.zeros(14)],
            v=np.r_[np.zeros(3), -.025 if height[0]>.3 else 0., np.zeros(14)],
            received=dict(theta=clock[0]), health=dict(theta=dict(range=[0.,.422])))
    e.state = state
    e.check_transport = lambda: None
    e.check_health = lambda *args: None
    reports = []
    e.startup_progress = lambda report: reports.append(report)
    def sleep(dt):
        clock[0] += dt
        height[0] = max(.3,height[0]-.025*dt)
    monkeypatch.setattr('hexfellow_deploy.native.time.sleep', sleep)
    final = e.position_startup_lift(state(), 'whole-body')
    assert abs(final.q[3]-.3)<1e-8 and clock[0]>15.
    assert writes and all(kind=='POSITION' and abs(value-.122)<1e-8 for kind,value in writes)
    assert reports[-1]['height_m'] < .301
    assert gate.phase == 'holding'


def test_mujoco_lowers_from_422mm_and_seeds_locked_ik_after_arrival(monkeypatch, tmp_path):
    from hexfellow_deploy.simulation import SimulationPlant
    from hexfellow_deploy.actuation import deployment, StreamContract, encode_mit
    from hexfellow_deploy.workspace import build_shared_executor
    from hexfellow_deploy.sim import scene_config
    from hexfellow_deploy.robot.joints import LIFT
    original = scene_config.load_scene
    def raised_scene(path):
        scene = original(path)
        scene.data['robot']['initial_joints'][LIFT] = .422
        return scene
    monkeypatch.setattr(scene_config, 'load_scene', raised_scene)
    _, _, profile, metadata = deployment(ROOT, Path('configs/hardware/desktop.yaml'))
    executor = build_shared_executor(metadata, .02)
    plant = SimulationPlant(ROOT/'configs/scenes/apple_on_table.yaml', tmp_path,
        effort_limits=metadata['effort_limits_nm'], profile=profile)
    try:
        samples = []
        plant.acquire(startup_lift_m=.3, progress=lambda report:samples.append(report))
        seed = plant.read()
        assert abs(seed['joint_positions'][LIFT]-.3) <= .001
        assert abs(seed['joint_velocities'][LIFT]) <= .002
        assert samples[0]['height_m'] > .421
        assert any(.32 < row['height_m'] < .39 for row in samples)
        assert plant.max_lift_speed < .026
        executor.ik.initialize_reference(seed, stationary_reference=True)
        contract = StreamContract(.1, .1, metadata=metadata)
        contract.seed(seed)
        command, _ = executor.tick(seed)
        result = executor.ik.last_result
        frame = dict(sequence=1, created_s=result.mit.timestamp_s, measured=seed,
            received=dict.fromkeys(('left','right','theta','base'), seed['timestamp_s']), command=command, mit=encode_mit(result.mit),
            reference_velocity={n:float(result.velocity[i]) for n,i in executor.ik.indices.items()})
        contract.accept(frame, plant.read())
        assert abs(command['joint_positions'][LIFT]-.3) <= .001
        assert frame['reference_velocity'][LIFT] == 0.
    finally:
        plant.close()
