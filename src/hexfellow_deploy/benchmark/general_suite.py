"""Reproducible multi-path validation; tuning and held-out splits stay explicit."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import os
from pathlib import Path
import subprocess
import sys

import qpsolvers
from hexfellow_deploy.robot.upstream import PROJECT_ROOT
from hexfellow_deploy.robot.joints import LIFT
from hexfellow_deploy.robotics.control.ik_core import PinkIKCore
from hexfellow_deploy.robotics.control.motion_profile import load_motion_profile, controller_config_for_profile
from .diverse import DiverseTrajectory
from .trajectories import envelope_report
from .run import initial_state, write_json
from .public_tcp import evaluate


def screened_speed(envelope, requested, reserve=.95):
    """Uniform time scaling; geometry and all acceptance limits stay unchanged.

    This only screens densely sampled Cartesian derivatives, not joint-space
    feasibility. Record both requested and reduced speeds so slowdown cannot
    masquerade as increased controller capability.
    """
    factor = max(1., *(ratio**(1/order) for key, ratio in envelope['ratios'].items()
                       for order in [1 if key.endswith('velocity') else 2 if key.endswith('acceleration') else 3]))
    return requested*reserve/factor if factor > reserve else requested


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True)
    parser.add_argument('--profile', action='append', required=True)
    parser.add_argument('--split', choices=('train', 'heldout', 'all'), default='train')
    parser.add_argument('--speeds', type=float, nargs='+', default=[1.2])
    parser.add_argument('--retime', action='store_true')
    parser.add_argument('--measured-pose-tracking', action='store_true')
    parser.add_argument('--plant-mass-scale', type=float, default=1.)
    parser.add_argument('--jobs', type=int, default=1)
    args = parser.parse_args()
    if args.jobs < 1:
        parser.error('jobs must be positive')
    root = Path(args.output).resolve()
    root.mkdir(parents=True, exist_ok=False)
    suite_dir = PROJECT_ROOT/'configs/benchmarks/generalization'
    specs = [suite_dir/row['path'] for row in json.loads((suite_dir/'suite.json').read_text())['cases']
             if args.split == 'all' or row['split'] == args.split]
    jobs = []
    for profile_path in args.profile:
        profile_path = Path(profile_path).resolve()
        profile = load_motion_profile(profile_path)
        model = PinkIKCore(config=controller_config_for_profile(profile, lift_position_margin=0.),
                          base_mode='fixed', locked_joint_names=(LIFT,)).model
        for spec_path in specs:
            spec = json.loads(spec_path.read_text())
            for speed in args.speeds:
                trajectory = DiverseTrajectory(model, initial_state(spec['posture']), spec, speed)
                envelope = envelope_report(trajectory, profile)
                effective = screened_speed(envelope, speed) if args.retime else speed
                name = f'{profile_path.stem}__{spec["name"]}__{speed:g}'
                command = [sys.executable, '-m', 'hexfellow_deploy.benchmark.run', '--profile', str(profile_path),
                           '--trajectory-spec', str(spec_path), '--speed', str(effective), '--derivatives',
                           '--plant-mass-scale', str(args.plant_mass_scale),
                           '--output', str(root/name)]
                if args.measured_pose_tracking:
                    command.append('--measured-pose-tracking')
                jobs.append(dict(name=name, split=spec['split'], profile=str(profile_path),
                    requested_speed_m_s=speed, effective_request_m_s=effective,
                    original_envelope=envelope, command=command))
    write_json(root/'plan.json', dict(arguments=vars(args), cases=jobs,
        timing_scope='Parallel wall times are not realtime qualification; use jobs=1 for timing'))
    def execute(row):
        result = subprocess.run(row['command'], cwd=PROJECT_ROOT, env=dict(os.environ,
            OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1'), text=True, capture_output=True)
        (root/(row['name']+'.log')).write_text(result.stdout+result.stderr)
        summary_path = root/row['name']/'summary.json'
        row.update(exit_code=result.returncode, result=json.loads(summary_path.read_text()) if summary_path.exists()
                   else dict(completed=False, tracking_accepted=False, failure=result.stderr[-3000:]))
        if summary_path.exists() and (root/row['name']/'trace.npz').exists():
            row['public_tcp'] = evaluate(root/row['name'])
        return row
    rows = []
    with ThreadPoolExecutor(max_workers=args.jobs) as pool:
        for future in as_completed([pool.submit(execute, row) for row in jobs]):
            row = future.result()
            rows.append(row)
            write_json(root/'manifest.json', sorted(rows, key=lambda x: x['name']))
            print(row['name'], 'accepted=', row.get('public_tcp', {}).get('tracking_accepted', False), flush=True)


if __name__ == '__main__':
    main()
