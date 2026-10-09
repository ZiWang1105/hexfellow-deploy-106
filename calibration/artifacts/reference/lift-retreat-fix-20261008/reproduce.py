"""Replay the recorded failed IK cycle; no robot or camera endpoint is opened."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
import pinocchio as pin

root = next(p for p in Path(__file__).resolve().parents
            if (p / 'config/camera/cameras.json').is_file())
upstream = root.parent.parent / 'UMI_Hardware'
sys.path.insert(0, str(upstream / 'src'))
sys.path.insert(0, str(root / 'tools/camera'))
from calibration_paths import desktop_robot_paths
from policy_sim.hardware import session
from policy_sim.hardware.whole_body import WholeBodyState
from policy_sim.hardware.ik_runner import initialize_core
from policy_sim.control.ik_core import PinkIKCore
from policy_sim.control.motion_profile import controller_config_for_profile

source = root / 'runs/wrist-calibration/20261008T234924.788901Z/session'
failed = json.loads((source / 'failed_cycle.json').read_text())
rows = [json.loads(line) for line in (source / 'trace.jsonl').read_text().splitlines()]
cycles = [r for r in rows if r.get('kind') == 'cycle' and r.get('executable')]
last, before = cycles[-1], cycles[-2]
runtime, release = desktop_robot_paths()
parser = argparse.ArgumentParser()
session.add_arguments(parser)
values = parser.parse_args(['--mode', 'sim', '--robot-config', str(runtime)])
mapping, profile, _, _ = session.load_configuration(values)
record = failed['state'].copy()
record['q'], record['v'] = np.asarray(record['q']), np.asarray(record['v'])
state = WholeBodyState(**record)
core = PinkIKCore(config=controller_config_for_profile(profile), base_mode='planar',
    runtime_position_bounds=mapping.physical_bounds(state), allow_upper_lift_retreat=True,
    **mapping.runtime_limits(profile, state))
initialize_core(core, state, np.array([]), stationary_reference=False)
core._last_safe_positions = np.asarray(last['positions'])
core._last_velocity = np.asarray(last['velocity'])
core._last_acceleration = (core._last_velocity - np.asarray(before['velocity'])) / .02
core.acceleration_limit.set_last_integration(core._last_velocity, .02)
core.jerk_limit.reset(core._last_velocity, core._last_acceleration)
original = state.q.copy()
targets = {s: pin.SE3(np.asarray(t)) for s, t in failed['targets'].items()}
command = core.step(state.q, targets, measured_velocity=state.v,
    state_timestamp=state.timestamp, control_timestamp=failed['control_timestamp'],
    measured_gripper_positions=np.array([]),
    posture_target=np.asarray(failed['posture_target']),
    posture_velocity=np.asarray(failed['posture_velocity']),
    control_phase=failed['phase'], **state.ik_feedback())
assert command.executable, command.diagnostics.fault
assert command.velocity[3] <= 0.
np.testing.assert_array_equal(state.q, original)
np.testing.assert_allclose(command.positions[4:], np.asarray(last['positions'])[4:], atol=1e-8)
old = failed['diagnostics']['fault_details']
upper = state.health['theta']['range'][1]
clearance = upper - state.q[3]
previous = last['velocity'][3]
reaction = old['reaction_s']
report = dict(source_run=str(source.parent), scope='Offline replay of recorded failed cycle and command histories',
    hardware_commands_sent=0, physical_devices_opened=False,
    measured_lift_m=float(state.q[3]), physical_upper_m=upper,
    physical_clearance_m=float(clearance), measured_velocity_m_s=float(state.v[3]),
    previous_velocity_m_s=previous, reaction_s=reaction,
    original_false_outward_reserve_m=max(0., state.v[3]-previous)*reaction,
    original_available_distance_m=old['boundaries'][0]['available_distance'],
    corrected_outward_reserve_m=max(0., max(state.v[3],0.)-max(previous,0.))*reaction,
    command_executable=command.executable, new_lift_command_m=float(command.positions[3]),
    new_lift_velocity_m_s=float(command.velocity[3]), arms_held=True,
    raw_measured_state_preserved=True, limits_relaxed=False,
    real_robot_rerun_performed=False,
    source_sha256={str(p.relative_to(upstream)):hashlib.sha256(p.read_bytes()).hexdigest()
        for p in [upstream/'src/policy_sim/control/command_limits.py']})
(Path(__file__).parent / 'recorded-cycle-replay.json').write_text(json.dumps(report, indent=2)+'\n')
print(json.dumps(report, indent=2))
