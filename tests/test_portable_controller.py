"""Portable deployment parity, with no mobile-manipulator workspace imports."""
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest
import yaml

from hexfellow_deploy.actuation import StreamContract, encode_mit
from hexfellow_deploy.contracts import JOINTS, LEFT, SHORT_JOINTS, physical_measurement
from hexfellow_deploy.portable_controller import DEFAULT_PROFILE, UmiController
from hexfellow_deploy.portable_execution import UmiExecution
from hexfellow_deploy.portable_profile import actuation_metadata
from hexfellow_deploy.reference import Receipt, ReferenceOwner
from hexfellow_deploy.source import UPSTREAM_ROOT


def source_robot():
    document = yaml.safe_load((UPSTREAM_ROOT / f'config/control/motion_profiles/{DEFAULT_PROFILE}.yaml').read_text())
    model = json.loads((UPSTREAM_ROOT / document['robot_model']['path']).read_text())
    return UPSTREAM_ROOT / model['urdf']


@pytest.fixture
def controller(tmp_path):
    if importlib.util.find_spec('pinocchio') is None or importlib.util.find_spec('_policy_sim_collision_kernel') is None:
        pytest.skip('Run native parity using the bootstrapped deployment environment')
    return UmiController(source_robot(), binding_dir=tmp_path)


def measured(q, dq, timestamp):
    return physical_measurement(q, dq, timestamp, gripper_state=dict(left='closed', right='closed'))


def wire(result, source, sequence):
    q = dict(zip(JOINTS, result.native_packet.positions[3:]))
    return dict(sequence=sequence, created_s=source['timestamp_s'], measured=source,
                received=dict.fromkeys(('left', 'right', 'theta', 'base'), source['timestamp_s']),
                command=dict(joint_positions=q, base_velocity_body=[0., 0., 0.],
                             gripper_open=dict(left=None, right=None)),
                reference_velocity=dict(zip(JOINTS, result.native_packet.velocity[3:])),
                mit=encode_mit(result.native_packet.mit))


def test_physical_aliases_preserve_actual_velocity_and_require_every_encoder():
    q = dict.fromkeys(SHORT_JOINTS, .1)
    dq = dict(zip(SHORT_JOINTS, np.arange(15)*.001))
    packet = measured(q, dq, 4.)
    assert list(packet['joint_velocities'].values()) == list(dq.values())
    with pytest.raises(ValueError, match='Missing'):
        measured({}, dq, 4.)
    with pytest.raises(ValueError, match='aliases disagree'):
        measured(dict(q, **{JOINTS[1]: .2}), dq, 4.)
    with pytest.raises(ValueError, match='finite'):
        measured(q, dict(dq, right_joint_7=float('nan')), 4.)


def test_isaac_wire_and_profile_import_without_any_controller_or_workspace_import():
    code = r'''
import sys
class Forbidden:
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'policy_sim', 'pinocchio', '_policy_sim_collision_kernel', 'mobile_manipulator'}:
            raise AssertionError('Native or parent workspace import: '+fullname)
sys.meta_path.insert(0, Forbidden())
from hexfellow_deploy.actuation import StreamContract, decode_mit
from hexfellow_deploy.portable_profile import actuation_metadata
from hexfellow_deploy.contracts import LEFT, RIGHT
metadata=actuation_metadata(sys.argv[1])
assert len(metadata['actuation_limits']['velocity']) == 15
mit=decode_mit(dict(joint_names=LEFT+RIGHT,position=[0.]*14,velocity=[0.]*14,
    torque_ff=[0.]*14,kp=[1.]*14,kd=[1.]*14,timestamp_s=1.))
assert len(mit.motor_input()) == 14
'''
    subprocess.run([sys.executable, '-c', code, str(source_robot())], check=True,
                   env=dict(os.environ, PYTHONPATH=str(Path(__file__).resolve().parents[1]/'src')))


def test_profile_reader_matches_native_model_and_every_tuning_bound(controller):
    limits = actuation_metadata(source_robot(), controller.profile_path)['actuation_limits']
    core = controller.core
    for key, expected in dict(position_lower=core.model.lowerPositionLimit[3:],
                              position_upper=core.model.upperPositionLimit[3:],
                              velocity=core.model.velocityLimit[3:],
                              acceleration=core._acceleration_limits[3:], jerk=core._jerk_limits[3:],
                              kp=core.effective_mit_servo.kp, kd=core.effective_mit_servo.kd,
                              effort=core.effective_mit_servo.torque_limit_nm).items():
        np.testing.assert_allclose(limits[key], expected, atol=1e-12, rtol=0)
    document = yaml.safe_load(controller.profile_path.read_text())
    document['mit']['kp'][0] += 1
    controller.profile_path.write_text(yaml.safe_dump(document))
    with pytest.raises(ValueError, match='tuning or limits'):
        actuation_metadata(source_robot(), controller.profile_path)


def test_runtime_bounds_narrow_same_native_model(tmp_path):
    if importlib.util.find_spec('pinocchio') is None:
        pytest.skip('Native dependency unavailable')
    kwargs = dict(runtime_position_bounds={JOINTS[0]: (.02, .35), JOINTS[1]: (-2., 2.)},
                  runtime_velocity_bounds={JOINTS[0]: .1, JOINTS[1]: 1.},
                  runtime_effort_limits_nm=[5.]*14)
    native = UmiController(source_robot(), binding_dir=tmp_path, **kwargs)
    limits = actuation_metadata(source_robot(), native.profile_path, **kwargs)['actuation_limits']
    np.testing.assert_allclose(limits['position_lower'], native.core.model.lowerPositionLimit[3:])
    np.testing.assert_allclose(limits['position_upper'], native.core.model.upperPositionLimit[3:])
    np.testing.assert_allclose(limits['velocity'], native.core.model.velocityLimit[3:])
    np.testing.assert_allclose(limits['effort'], native.core.effective_mit_servo.torque_limit_nm)


def test_native_mode_changes_keep_history_and_solver_waits_for_application_receipt(controller):
    execution = UmiExecution(controller)
    home = controller.core.home_q.copy()
    q = dict(zip(SHORT_JOINTS, home[3:]))
    dq = dict.fromkeys(SHORT_JOINTS, 0.)
    execution.reset(q, dq, 1.)
    original_core, original_guard = controller.core, execution.guard
    owner = ReferenceOwner()
    clock = [1.]
    metadata = actuation_metadata(source_robot(), controller.profile_path)
    stream = StreamContract(.1, .04, metadata=metadata, clock=lambda: clock[0])
    stream.seed(measured(q, dq, 1.))
    for tick, mode in enumerate(('posture', 'cartesian', 'posture', 'cartesian')):
        clock[0] = 1.+tick*.02
        source = measured(q, dq, clock[0])
        sequence, result = owner.generate(lambda: execution.step(q, dq, q, dq, clock[0], mode=mode))
        for _ in range(3):
            assert owner.generate(lambda: pytest.fail('Solver ran while publication pending')) is None
        assert execution.sequence == tick+1
        frame = wire(result, source, sequence)
        decoded = stream.accept(frame, source)
        np.testing.assert_array_equal(decoded.position, result.native_packet.mit.position)
        assert not decoded.position.flags.writeable
        owner.acknowledge(Receipt(sequence, clock[0], 'simulator_applied'))
        assert controller.core is original_core and execution.guard is original_guard
        q, dq = result.q, result.dq
    assert owner.accepted == 4 and owner.pending is None


@pytest.mark.parametrize('fault', ['gains', 'effort', 'discontinuity', 'freshness', 'sequence'])
def test_trusted_wire_guard_rejects_corruption(controller, fault):
    execution = UmiExecution(controller)
    q = dict(zip(SHORT_JOINTS, controller.core.home_q[3:]))
    dq = dict.fromkeys(SHORT_JOINTS, 0.)
    source = measured(q, dq, 1.)
    execution.reset(q, dq, 1.)
    frame = wire(execution.step(q, dq, q, dq, 1., mode='posture'), source, 1)
    if fault == 'gains': frame['mit']['kp'][0] += 1
    if fault == 'effort': frame['mit']['torque_ff'][0] = 500.
    if fault == 'discontinuity':
        frame['mit']['position'][0] += .01
        frame['command']['joint_positions'][LEFT[0]] += .01
    if fault == 'freshness': frame['received']['right'] -= .05
    if fault == 'sequence': frame['sequence'] = 2
    stream = StreamContract(.1, .04, metadata=actuation_metadata(source_robot(), controller.profile_path), clock=lambda: 1.)
    stream.seed(source)
    with pytest.raises((ValueError, RuntimeError)):
        stream.accept(frame, source)
    assert stream.latest is None


@pytest.mark.parametrize('direction', [-1, 1])
def test_shared_source_planar_lift_brakes_at_both_bounds_without_rebasing(controller, direction):
    """Exercise 9398's measured-origin governor in PoolBot's retained 18-state mode."""
    core = controller.core
    assert core._measured_lift_boundary_governor is not None
    assert core._lift_reference_governor is None and core._persistent_start == 4
    q = core.home_q.copy()
    bound = core.model.lowerPositionLimit[3] if direction < 0 else core.model.upperPositionLimit[3]
    q[3] = bound-direction*.01
    target = q.copy()
    target[3] = bound
    # The native delayed stopping envelope deliberately reserves finite travel
    # even at zero current speed. A requested boundary is therefore not an
    # assertion that the measured lift can attain that exact planning limit.
    from hexfellow_deploy.source import upstream
    reserve = float(upstream('control.command_limits').stopping_distance(
        0., 0., core._acceleration_limits[3], core._jerk_limits[3],
        delay_s=core.config.maximum_command_delay_s+controller.control_dt))
    stopping_position = bound-direction*reserve
    v = np.zeros(18)
    poses = controller.tcp_poses(q)
    controller.reset(q, v, 0., targets=poses)
    previous_v, previous_a = np.zeros(18), np.zeros(18)
    # The fixture's 70% lift response approaches the native reserve asymptotically;
    # use a finite observation budget, not a requested motion deadline.
    for tick in range(600):
        result = controller.step(q, v, poses, state_timestamp=tick*.02, control_timestamp=tick*.02,
                                 posture_target=target, posture_velocity=np.zeros(18))
        assert result.executable, result.diagnostics.fault
        acceleration = (result.velocity-previous_v)/.02
        assert abs(result.velocity[3]) <= core.model.velocityLimit[3]+1e-7
        assert abs(acceleration[3]) <= core._acceleration_limits[3]+1e-6
        assert abs((acceleration-previous_a)[3]/.02) <= core._jerk_limits[3]+1e-4
        assert result.positions[3] == pytest.approx(q[3]+result.velocity[3]*.02, abs=1e-9)
        previous_v, previous_a = result.velocity.copy(), acceleration
        next_q = q+.7*(result.positions-q)
        v, q = (next_q-q)/.02, next_q
        assert core.model.lowerPositionLimit[3] <= q[3] <= core.model.upperPositionLimit[3]
        if abs(result.velocity[3]) < 1e-6 and abs(q[3]-stopping_position) < 1e-5:
            break
    assert abs(q[3]-stopping_position) < 1e-5 and abs(result.velocity[3]) < 1e-6
