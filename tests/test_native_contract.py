"""Native packet and shared MIT admission tests; no device connections."""
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from hexfellow_deploy.workspace import activate

ROOT = activate(Path(__file__).resolve().parents[3])
from hexfellow_deploy.actuation import StreamContract, deployment, encode_mit
from hexfellow_deploy.native import ContinuousGate, NativeDriver, StreamEndpoint
from mobile_manipulator.robot.joints import JOINTS, LEFT, RIGHT
from policy_sim.control.mit_command import MitCommand
from policy_sim.hardware.mit_wire import authorize_group
from hex_device.generated.public_api_down_pb2 import APIDown


@pytest.fixture(scope='module')
def metadata():
    return deployment(ROOT, Path('configs/hardware/desktop.yaml'))[3]


def sample():
    q = dict.fromkeys(JOINTS, 0.)
    q[JOINTS[0]] = .3
    return dict(timestamp_s=1., joint_positions=q,
                joint_velocities=dict.fromkeys(JOINTS, 0.),
                gripper_state=dict(left='closed', right='closed'))


def frame(metadata):
    measured = sample()
    limits = metadata['actuation_limits']
    mit = MitCommand(LEFT+RIGHT, np.zeros(14), np.zeros(14), np.zeros(14),
                     limits['kp'], limits['kd'], 1.)
    return dict(sequence=1, created_s=1., measured=measured,
                reference_velocity=dict.fromkeys(JOINTS, 0.),
                received=dict.fromkeys(('left', 'right', 'theta', 'base'), 1.),
                command=dict(joint_positions=dict(measured['joint_positions']),
                             base_velocity_body=[0., 0., 0.], gripper_open=dict(left=0, right=1)),
                mit=encode_mit(mit))


def seeded(metadata):
    contract = StreamContract(.1, .1, metadata=metadata, clock=lambda: 1.001)
    contract.seed(sample())
    return contract


def test_shared_profile_accepts_original_frame_and_prevents_history_reset(metadata):
    contract = seeded(metadata)
    contract.accept(frame(metadata), sample())
    with pytest.raises(RuntimeError, match='already initialized'):
        contract.seed(sample())


def test_shared_executor_uses_selected_deployment_base_motion(metadata, tmp_path):
    import yaml
    from hexfellow_deploy.workspace import build_shared_executor
    from mobile_manipulator.control.base_motion import BaseMotionConfig
    selected = tmp_path/'base.yaml'
    values = dict(BaseMotionConfig.load(metadata['base_motion']).to_dict(), approach_distance_m=.61)
    selected.write_text(yaml.safe_dump(values))
    executor = build_shared_executor(dict(metadata, base_motion=str(selected)), .02)
    assert executor.base_motion.to_dict() == values


@pytest.mark.parametrize('change, reason', [
    (lambda f: f['mit']['kp'].__setitem__(0, 1.), 'gains'),
    (lambda f: f['mit']['torque_ff'].__setitem__(0, 53.), 'effort'),
    (lambda f: f['mit'].__setitem__('timestamp_s', .99), 'timestamp'),
    (lambda f: f['received'].__setitem__('right', 1.1), 'source feedback'),
    (lambda f: f.__setitem__('sequence', 2), 'skipped'),
    (lambda f: f['command']['base_velocity_body'].__setitem__(0, .2), 'acceleration'),
])
def test_shared_profile_rejects_invalid_frame_without_committing(metadata, change, reason):
    contract = seeded(metadata)
    value = frame(metadata)
    change(value)
    with pytest.raises(ValueError, match=reason):
        contract.accept(value, sample())
    assert contract.latest is None
    np.testing.assert_array_equal(contract.previous_velocity, np.zeros(15))


def test_reference_step_uses_profile_dt_when_wall_clock_is_delayed(metadata):
    now = [1.001]
    contract = StreamContract(.1, .1, metadata=metadata, clock=lambda: now[0])
    contract.seed(sample())
    first = frame(metadata)
    contract.accept(first, sample())
    second = frame(metadata)
    second.update(sequence=2, created_s=1.06)
    second['mit']['timestamp_s'] = 1.06
    second['mit']['velocity'][0] = .01
    second['reference_velocity'][LEFT[0]] = .01
    second['mit']['position'][0] = .0002
    second['command']['joint_positions'][LEFT[0]] = .0002
    second['received'] = dict.fromkeys(second['received'], 1.06)
    second['measured']['timestamp_s'] = 1.06
    now[0] = 1.061
    contract.accept(second, second['measured'])
    assert contract.previous_velocity[1] == .01


def test_discontinuous_reference_and_physical_overspeed_are_rejected(metadata):
    value = frame(metadata)
    value['mit']['position'][0] = .1
    value['command']['joint_positions'][LEFT[0]] = .1
    with pytest.raises(ValueError, match='Discontinuous'):
        seeded(metadata).accept(value, sample())
    measured = sample()
    measured['joint_velocities'][RIGHT[-1]] = 100.
    with pytest.raises(ValueError, match='physical bounds'):
        seeded(metadata).accept(frame(metadata), measured)


def test_nonpersistent_lift_uses_source_position_and_solved_velocity(metadata):
    import copy
    metadata = copy.deepcopy(metadata)
    metadata['actuation_limits'].update(persistent_lift_reference=False, locked_joints=[])
    now = [1.001]
    contract = StreamContract(.1, .1, metadata=metadata, clock=lambda: now[0])
    contract.seed(sample())
    contract.accept(frame(metadata), sample())
    value = frame(metadata)
    value.update(sequence=2, created_s=1.02)
    value['mit']['timestamp_s'] = 1.02
    value['received'] = dict.fromkeys(value['received'], 1.02)
    value['measured']['timestamp_s'] = 1.02
    value['measured']['joint_positions'][JOINTS[0]] = .2995
    value['command']['joint_positions'][JOINTS[0]] = .29955
    value['reference_velocity'][JOINTS[0]] = .0025
    now[0] = 1.021
    contract.accept(value, value['measured'])
    assert contract.previous_velocity[0] == .0025


def test_lift_wire_reference_matches_upstream_planning_boundary_projection(metadata):
    from hexfellow_deploy.workspace import build_shared_executor
    executor = build_shared_executor(metadata, .02)
    source = sample()
    # Exact source measurement from the failed pickup, just outside the
    # artificial planning margin while still within the physical interval.
    source['joint_positions'][JOINTS[0]] = .39800006137472715
    vector = executor.ik.vector(source['joint_positions'])
    projected = executor.ik.core._validate_configuration_vector(vector, recover_margin_excursion=True)
    assert metadata['actuation_limits']['position_upper'][0] == executor.ik.core.model.upperPositionLimit[0]
    value = frame(metadata)
    value['measured'] = source
    value['command']['joint_positions'][JOINTS[0]] = float(projected[0])
    contract = StreamContract(.1, .1, metadata=metadata, clock=lambda: 1.001)
    contract.seed(source)
    contract.accept(value, source)
    assert value['measured']['joint_positions'][JOINTS[0]] == .39800006137472715
    # Projection is not permission to alter an unrelated reference.
    import copy
    value = copy.deepcopy(value)
    value['sequence'] = 2
    value['command']['joint_positions'][JOINTS[0]] -= .001
    with pytest.raises(ValueError, match='Lift position'):
        contract.accept(value, source)


def test_profile_jerk_is_checked_with_matching_position_and_velocity(metadata):
    value = frame(metadata)
    value['mit']['position'][0] = .004
    value['mit']['velocity'][0] = .2
    value['command']['joint_positions'][LEFT[0]] = .004
    value['reference_velocity'][LEFT[0]] = .2
    with pytest.raises(ValueError, match='acceleration or jerk'):
        seeded(metadata).accept(value, sample())


def packet():
    result = APIDown()
    targets = result.arm_command.arm_exclusive_command.arm_api_control_command.motor_targets.targets
    for _ in range(7):
        targets.add().mit_target.kp = 200.
    return result


def test_group_authorization_is_atomic_when_one_device_has_not_dispatched():
    gates = [ContinuousGate(n, clock=lambda: 1.) for n in ('left', 'right')]
    data = packet()
    for gate in gates:
        gate.transition('active')
        gate.authorize(1, 1., 1.1, data)
    gates[0].record_dispatch(data, 1., 1., sequence=1)
    with pytest.raises(RuntimeError, match='no confirmed SDK dispatch'):
        authorize_group([(gate, data) for gate in gates], 2, 1., 1.1)
    assert [gate.sequence for gate in gates] == [1, 1]


def test_native_receipt_requires_matching_first_dispatch_for_every_primary_device():
    gates = {n: ContinuousGate(n) for n in ('left', 'right', 'theta', 'base')}
    driver = object.__new__(NativeDriver)
    driver.endpoint = SimpleNamespace(direct_gates=gates)
    for gate in gates.values():
        gate.first_dispatch = dict(sequence=4, sdk_send_returned=2.)
    gates['base'].first_dispatch = dict(sequence=3, sdk_send_returned=3.)
    assert driver.acknowledgement(4) is None
    gates['base'].first_dispatch = dict(sequence=4, sdk_send_returned=2.5)
    receipt = driver.acknowledgement(4)
    assert receipt.sequence == 4 and receipt.accepted_at == 2.5
    assert receipt.scope == 'sdk_accepted'


def test_native_preserves_ik_source_positions_with_newer_endpoint_feedback(metadata):
    value = frame(metadata)
    driver = object.__new__(NativeDriver)
    driver.timeout = .1
    driver.state = SimpleNamespace(q=np.r_[np.zeros(3), np.full(15, .15)])
    captured = []
    driver.endpoint = SimpleNamespace(publish_frame=lambda command, state: captured.append(command))
    from hexfellow_deploy.actuation import decode_mit
    driver.submit(1, value['command'], decode_mit(value['mit']), created_s=1.,
                  received=value['received'], measured=value['measured'], reference_velocity=value['reference_velocity'])
    np.testing.assert_array_equal(captured[0].measured_q[3:],
                                  [value['measured']['joint_positions'][n] for n in JOINTS])
    assert captured[0].state_received == value['received']


@pytest.mark.parametrize('invalid_hand_pulses', [False, True])
def test_native_leaf_reuses_calibrated_mit_encoding_for_complete_frame(monkeypatch, invalid_hand_pulses):
    """Exercise production packet preparation with passive SDK setter recorders."""
    from mobile_manipulator.control.gripper import BinaryGripperServo
    from mobile_manipulator.hardware.device import BinaryMotorTargets
    from mobile_manipulator.hardware.umi_device import native_packet
    from policy_sim.hardware.whole_body import WholeBodyState

    monkeypatch.setattr('time.monotonic', lambda: 100.)
    config, mapping, profile, metadata = deployment(ROOT, Path('configs/hardware/desktop.yaml'))
    robot = config['robot']
    for settings in robot['grippers'].values():
        # Passive SDK recorders only; enable closure in this test configuration.
        settings.update(closed_motor_rad=5.5, open_motor_rad=.02, closure_enabled=True)
    measured = sample()
    measured['timestamp_s'] = 100.
    measured['joint_positions'].update({LEFT[0]: 1.57, RIGHT[0]: -1.57, LEFT[1]: 1.5, RIGHT[1]: 1.5})
    q = np.array(list(measured['joint_positions'].values()))
    health = {s: dict(joint_limits=np.column_stack((profile.sdk_position_lower_rad[i:i+7],
                  profile.sdk_position_upper_rad[i:i+7])).tolist()) for s, i in [('left', 0), ('right', 7)]}
    health.update(theta=dict(range=[0., .422]), base={})
    state = WholeBodyState(np.r_[np.zeros(3), q], np.zeros(18),
                          dict.fromkeys(('left', 'right', 'theta', 'base'), 100.), {}, health, {})

    class SetterRecorder:
        def __init__(self):
            self.calls = []
            self._last_command_send = None
            self._pulse_per_rotation = np.array([1000.])
            self._max_pos = 1000
        def motor_command(self, kind, values):
            self.calls.append((kind, values))
        def set_vehicle_speed(self, *values):
            self.calls.append(('base', values))
        def get_joint_limits(self):
            return [0., 8.]
        def convert_rad_to_positions(self, values, pulses):
            return np.asarray(values)*1000.
        def get_motor_pulse_per_rotations(self):
            return np.array([6283.])

    endpoint = object.__new__(StreamEndpoint)
    endpoint.mobile_config, endpoint.mapping = robot, mapping
    endpoint.last_base = np.zeros(3)
    endpoint.grippers = {s: BinaryMotorTargets(robot['grippers'][s]) for s in ('left', 'right')}
    endpoint.gripper_servos = {s: BinaryGripperServo(g.closed, g.opened, g.closed, 1.)
                              for s, g in endpoint.grippers.items()}
    endpoint.hands = {s: SetterRecorder() for s in ('left', 'right')}
    class PublisherRecorder:
        def __init__(self, hand): self.hand = hand
        def check(self): pass
        def request(self, opened, *, expires, opening_target):
            self.hand.motor_command('POSITION' if opened == 1 else 'BRAKE',
                                    [opening_target] if opened == 1 else [True])
    endpoint.gripper_publishers = {s:PublisherRecorder(h) for s,h in endpoint.hands.items()}

    endpoint.devices = {s: SetterRecorder() for s in ('left', 'right', 'theta', 'base')}
    endpoint.command_type = SimpleNamespace(MIT='MIT', POSITION='POSITION', BRAKE='BRAKE')
    endpoint.mit_type = SimpleNamespace
    endpoint.direct_gates = {s: ContinuousGate(s, clock=lambda: 100.) for s in endpoint.devices}
    for gate in endpoint.direct_gates.values():
        gate.transition('active')
    endpoint.write_count, endpoint.retreat_active = 0, False
    endpoint.hand_feedback = lambda: {s: dict(pos=g.closed, vel=0., states=['normal'], received=100.)
                                      for s, g in endpoint.grippers.items()}
    endpoint.check_transport = lambda: None
    endpoint.check_health = lambda *args, **kwargs: None
    endpoint.extras = dict(sequence=1, base=[.02, 0., .01], grippers=dict(left=1, right=0))
    mit = MitCommand(LEFT+RIGHT, q[1:], np.linspace(-.02, .02, 14), np.linspace(-.1, .1, 14),
                     profile.mit_servo.kp, profile.mit_servo.kd, 100.)
    bundle = dict(sequence=1, created=100., expires=100.1, received=state.received,
                  source_q=q, positions=q, velocity=np.r_[0., mit.velocity], mit=mit)
    command = native_packet(bundle, state)
    if invalid_hand_pulses:
        endpoint.hands['right'].convert_rad_to_positions = lambda *args: np.array([np.nan])
        with pytest.raises(RuntimeError, match='Invalid GP80 encoder pulse'):
            endpoint.publish_frame(command, state)
        assert all(not device.calls for device in endpoint.devices.values())
        assert all(not hand.calls for hand in endpoint.hands.values())
        return
    prepared = endpoint._prepare_dispatch(command, 'whole-body', state)
    endpoint.publish_frame(command, state)
    assert endpoint.write_count == 4
    for side in ('left', 'right'):
        kind, targets = endpoint.devices[side].calls[-1]
        assert kind == 'MIT'
        expected = mapping.arm_command(side, mit)
        for field, sdk_field in [('position', 'position'), ('velocity', 'speed'), ('torque_ff', 'torque'),
                                 ('kp', 'kp'), ('kd', 'kd')]:
            np.testing.assert_array_equal([getattr(target, sdk_field) for target in targets], expected[field])
        assert endpoint.hands[side].calls[-1][0] == ('POSITION' if side == 'left' else 'BRAKE')
        assert endpoint.direct_gates[side].hand_deadline == 100.1
    assert endpoint.devices['theta'].calls[-1][0] == 'POSITION'
    for name, (wire_packet, _) in prepared.items():
        assert endpoint.direct_gates[name].filter(wire_packet) == wire_packet
        assert endpoint.direct_gates[name].deadline == 100.1


def test_native_fault_evidence_survives_close_buffer_clear_and_identifies_device():
    from collections import deque
    import json
    import threading
    from policy_sim.hardware.x7 import HardwareSafetyError
    from hexfellow_deploy.backend import diagnostic_value
    gate = ContinuousGate('right', clock=lambda: 100.)
    gate.transition('active')
    gate.fault = 'unexpected SDK command, automatic rearm or clear-stop blocked'
    gate.hand_rejection = dict(packet_kind='arm_command', packet_text='clear_parking_stop',
                               observed_s=100.01, reason=gate.fault)
    transport = SimpleNamespace(fault=None,
        rows=deque([dict(udp_send_started=99.95, udp_send_returned=99.951)]),
        receive_rows=deque([dict(udp_received_at=100.)]),
        process_rows=deque([(100., 100.001, 128)]))
    api = SimpleNamespace(_HexDeviceApi__kcp_client=SimpleNamespace(_policy_udp=transport))
    driver = object.__new__(NativeDriver)
    driver.endpoint = SimpleNamespace(direct_gates={'right': gate}, apis=[api],
        taps={'right': SimpleNamespace(frames=deque([dict(received=100., pos=np.zeros(7))]),
            value=dict(received=100., parking_stop={'reason': 'API communication timeout'}))})
    with pytest.raises(HardwareSafetyError, match='right downlink: unexpected SDK command'):
        driver.check()
    evidence = driver.diagnostics()
    driver.endpoint.apis.clear()
    transport.rows.clear()
    assert evidence['devices']['right']['udp_send'][0]['udp_send_started'] == 99.95
    assert evidence['devices']['right']['gate']['rejected_packet']['packet_text'] == 'clear_parking_stop'
    saved = json.loads(json.dumps(evidence, allow_nan=False, default=diagnostic_value))
    assert saved['devices']['right']['feedback'][0]['pos'] == [0.]*7
    assert saved['devices']['right']['latest_health']['parking_stop']['reason'] == 'API communication timeout'
    # A stuck sender cannot prevent fault capture from returning to stop.
    locked, release = threading.Event(), threading.Event()
    def owner():
        with gate.lock:
            locked.set()
            release.wait(5)
    worker = threading.Thread(target=owner)
    worker.start()
    try:
        assert locked.wait(1)
        blocked_evidence = driver.diagnostics()
        assert 'busy' in blocked_evidence['devices']['right']['gate_error']
        assert 'capture_error' in blocked_evidence['devices']['right']
    finally:
        release.set()
        worker.join(1)


@pytest.mark.parametrize('diagnostic_failure', [False, True])
def test_backend_stops_before_writing_evidence_and_preserves_primary_fault(monkeypatch, tmp_path, diagnostic_failure):
    import json
    from hexfellow_deploy import backend, native
    from hexfellow_deploy.backend import BackendComponent
    events = []
    class Topic:
        def put(self, value, **kwargs):
            if value.get('fault'):
                component.signal_stop()
            return True
    topic = Topic()
    component = BackendComponent('probe', topic, topic, topic, topic, topic, topic,
        str(ROOT), str(tmp_path), '', '', backend='hardware', execute=True)
    class Driver:
        @property
        def snapshot(self):
            raise RuntimeError('firmware communication timeout')
        def diagnostics(self):
            events.append('capture')
            if diagnostic_failure:
                raise ValueError('capture failed')
            return {'feedback': np.zeros(7)}
        def close(self):
            events.append('stop')
            assert not (tmp_path/'native_transport.json').exists()
            component.signal_stop()
            raise RuntimeError('lift stop unconfirmed')
    monkeypatch.setattr(backend, 'deployment', lambda *a, **k: ({}, None, None, None))
    monkeypatch.setattr(native, 'NativeDriver', lambda *a, **k: Driver())
    component.run()
    assert events == ['capture', 'stop']
    final = json.loads((tmp_path/'backend_final.json').read_text())
    assert final['fault'] == 'RuntimeError: firmware communication timeout'
    assert final['shutdown_error'] == 'RuntimeError: lift stop unconfirmed'
    if diagnostic_failure:
        assert final['native_diagnostic_error'] == 'ValueError: capture failed'
    else:
        assert json.loads((tmp_path/'native_transport.json').read_text())['feedback'] == [0.]*7
