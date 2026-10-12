"""Exercise solver-to-actuator lift semantics with noisy and recorded feedback."""
import copy
import json
from pathlib import Path

import numpy as np
import pytest

from hexfellow_deploy.workspace import activate, build_shared_executor

ROOT = activate(Path(__file__).resolve().parents[1])
from hexfellow_deploy.actuation import StreamContract, deployment, encode_mit
from hexfellow_deploy.robot.joints import JOINTS, LIFT


def test_recorded_fixed_lift_frame_reproduces_old_rejection_and_passes_current_contract():
    record = json.loads((Path(__file__).parent/'fixtures/fixed_lift_rejected_232851.json').read_text())
    frame = record['frame']
    current = deployment(ROOT, Path('configs/hardware/desktop.yaml'))[3]['actuation_limits']
    assert current['persistent_lift_reference'] and LIFT in current['locked_joints']
    for corrected in (False, True):
        limits = copy.deepcopy(record['limits'])
        if corrected:
            limits.update({k: current[k] for k in ('persistent_lift_reference', 'locked_joints')})
        contract = StreamContract(.1, .1, metadata={'actuation_limits': limits},
                                  clock=lambda: frame['created_s']+.001)
        contract.seed(record['seed'])
        # Restore the saved accepted command history immediately before frame 52.
        contract.previous_position = np.array([record['previous_position'][n] for n in JOINTS])
        contract.previous_velocity = np.array([record['previous_velocity'][n] for n in JOINTS])
        preceding = np.array([record['preceding_velocity'][n] for n in JOINTS])
        contract.previous_acceleration = (contract.previous_velocity-preceding)/limits['dt_s']
        contract.latest = {'sequence': frame['sequence']-1}
        if corrected:
            contract.accept(frame, frame['measured'])
            changed = copy.deepcopy(frame)
            changed['sequence'] += 1
            for i, name in enumerate(JOINTS[1:]):
                position = contract.previous_position[i+1] + changed['mit']['velocity'][i]*limits['dt_s']
                changed['command']['joint_positions'][name] = position
                changed['mit']['position'][i] = position
            changed['command']['joint_positions'][LIFT] += .00001
            changed['reference_velocity'][LIFT] = .00001/limits['dt_s']
            with pytest.raises(ValueError, match='locked startup reference'):
                contract.accept(changed, changed['measured'])
        else:
            with pytest.raises(ValueError, match='Lift position does not match'):
                contract.accept(frame, frame['measured'])


@pytest.mark.parametrize('height', [.3, .421, .4219988120410797, .422, 563397/1335063])
def test_actual_solver_commands_pass_stream_with_changing_lift_feedback(height):
    metadata = deployment(ROOT, Path('configs/hardware/desktop.yaml'))[3]
    executor = build_shared_executor(metadata, .02)
    now = [1.]
    executor.ik.control_clock = lambda: now[0]
    positions = {n: float(executor.ik.core.home_q[i]) for n, i in executor.ik.indices.items()}
    positions[LIFT] = height
    seed = dict(timestamp_s=now[0], joint_positions=positions,
                joint_velocities=dict.fromkeys(JOINTS, 0.), gripper_state=dict(left='open', right='open'))
    executor.ik.initialize_reference(seed, stationary_reference=True)
    contract = StreamContract(.1, .1, metadata=metadata, clock=lambda: now[0])
    contract.seed(seed)
    for sequence in range(1, 31):
        now[0] += .02
        measured = copy.deepcopy(seed)
        measured['timestamp_s'] = now[0]
        measured['joint_positions'][LIFT] -= (.00003*abs(np.sin(sequence)) if height <= .422
                                              else (sequence % 4)/1335063)
        command, _ = executor.tick(measured)
        result = executor.ik.last_result
        frame = dict(sequence=sequence, created_s=now[0], measured=measured,
                     received=dict.fromkeys(('left', 'right', 'theta', 'base'), now[0]),
                     command=command, mit=encode_mit(result.mit),
                     reference_velocity={n:float(result.velocity[i]) for n,i in executor.ik.indices.items()})
        contract.accept(frame, measured)
        assert command['joint_positions'][LIFT] == pytest.approx(min(height, .422-1e-9), abs=1e-10)
        assert executor.ik.last_source['joint_positions'][LIFT] == measured['joint_positions'][LIFT]
