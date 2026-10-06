"""Offline PinkIK reference and position-plant qualification, without devices/API."""
import argparse
import json
from pathlib import Path
import subprocess
import time

import numpy as np
from scipy.spatial.transform import Rotation

from hexfellow_deploy.workspace import activate, build_executor
from hexfellow_deploy.reference import ReferenceOwner, Receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workspace', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    root = activate(args.workspace)
    import mujoco
    from mobile_manipulator.sim.scene import build_scene
    from mobile_manipulator.sim.scene_config import load_scene
    from mobile_manipulator.sim.sensors import EncoderReader
    from mobile_manipulator.interfaces.contract import Action, ArmTarget
    config = load_scene(root/'configs/scenes/apple_on_table.yaml')
    scene = build_scene(config, output/'plant.xml')
    sensor = EncoderReader(scene.model)
    action = Action(ArmTarget((.53, .18, .95), (.7071067811865476, -.7071067811865476, 0., 0.), 1),
                    ArmTarget((.53, -.18, .95), (.7071067811865476, .7071067811865476, 0., 0.), 1))
    report = dict(evidence='offline_solver_and_position_plant_not_native_MIT', cases=[])
    for profile in ('umi_pink_mobile', 'umi_pink_real'):
        for mode in ('ideal_reference', 'mujoco_position'):
            data = mujoco.MjData(scene.model)
            mujoco.mj_resetDataKeyframe(scene.model, data, scene.model.key('initial').id)
            mujoco.mj_forward(scene.model, data)
            executor = build_executor(config, root/f'configs/controller/{profile}.yaml', .02)
            packet = sensor.read(data.sensordata, data.time)
            executor.submit(action, packet)
            owner = ReferenceOwner()
            samples = []
            fault = None
            start = time.monotonic()
            try:
                for step in range(1100):
                    sequence, (command, diagnostic) = owner.generate(lambda: executor.tick(packet))
                    # The actual solver has already committed its reference. A delayed
                    # handoff must prevent additional solves, not reset this history.
                    saved = executor.ik.core._last_safe_positions.copy()
                    for _ in range(3):
                        assert owner.generate(lambda: executor.tick(packet)) is None
                    np.testing.assert_array_equal(saved, executor.ik.core._last_safe_positions)
                    if mode == 'ideal_reference':
                        positions = command['joint_positions']
                        velocity = {n:(q-packet['joint_positions'][n])/.02 for n,q in positions.items()}
                        packet = dict(packet, timestamp_s=(step+1)*.02,
                                      joint_positions=positions, joint_velocities=velocity)
                    else:
                        for name, value in command['joint_positions'].items():
                            data.ctrl[scene.model.actuator('position_'+name).id] = value
                        mujoco.mj_step(scene.model, data, nstep=10)
                        mujoco.mj_forward(scene.model, data)
                        packet = sensor.read(data.sensordata, data.time)
                    owner.acknowledge(Receipt(sequence, step*.02+1, 'simulator_applied'))
                    row = dict(step=step, errors={})
                    for label, positions in [('reference', command['joint_positions']),
                                             ('measured', packet['joint_positions'])]:
                        executor.kinematics.update(positions)
                        for side in ('left', 'right'):
                            pose, target = executor.kinematics.pose(side), getattr(action, side)
                            row['errors'][label+'_'+side] = dict(
                                position_m=float(np.linalg.norm(pose.translation-target.position_m)),
                                orientation_deg=float(np.rad2deg(np.linalg.norm(Rotation.from_matrix(
                                    Rotation.from_quat(target.quaternion_xyzw).as_matrix() @ pose.rotation.T).as_rotvec()))))
                    samples.append(row)
            except Exception as error:
                fault = f'{type(error).__name__}: {error}'
            dt_rejected = False
            try:
                executor.ik.core.step(executor.ik.vector(packet['joint_positions']), executor.ik.targets, dt=.03)
            except ValueError as error:
                dt_rejected = 'profile-bound IK timestep' in str(error)
            case = dict(profile=profile, mode=mode, ticks=owner.generated,
                        accepted=owner.accepted, wall_s=time.monotonic()-start,
                        fault=fault, variable_dt_rejected=dt_rejected,
                        final_errors=samples[-1]['errors'] if samples else None)
            report['cases'].append(case)
            (output/f'{profile}_{mode}.json').write_text(json.dumps(samples))
            print(json.dumps(case), flush=True)
    report['revisions'] = {name:subprocess.check_output(['git','rev-parse','HEAD'], cwd=path, text=True).strip()
                           for name,path in [('workspace', root), ('UMI_Hardware', root/'third_party/UMI_Hardware')]}
    report['native_qualified'] = False
    report['solver_cases_passed'] = all(
        case['fault'] is None and case['ticks'] == case['accepted'] == 1100
        and all(error['position_m'] < .003 and error['orientation_deg'] < 1.2
                for error in case['final_errors'].values()) for case in report['cases'])
    from hexfellow_deploy.provenance import collect
    report['provenance'] = collect(root)
    (output/'summary.json').write_text(json.dumps(report, indent=2)+'\n')
    if not report['solver_cases_passed']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
