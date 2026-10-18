"""Offline passing-speed brackets for a frozen controller on diverse paths.

Each trial runs the shared MIT simulation. A Cartesian envelope screen supplies
only a starting ceiling. All failed trials remain recorded. Bisection assumes
local monotonicity and does not prove a global maximum or hardware feasibility.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import os
from pathlib import Path
import subprocess
import sys

import qpsolvers
import numpy as np
from hexfellow_deploy.robot.upstream import PROJECT_ROOT
from hexfellow_deploy.robot.joints import LIFT
from hexfellow_deploy.robotics.control.ik_core import PinkIKCore
from hexfellow_deploy.robotics.control.motion_profile import load_motion_profile, controller_config_for_profile
from .diverse import DiverseTrajectory
from .trajectories import envelope_report
from .general_suite import screened_speed
from .public_tcp import evaluate
from .run import initial_state, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--profile', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--split', choices=('train', 'heldout', 'all'), default='train')
    parser.add_argument('--max-speed', type=float, default=1.9)
    parser.add_argument('--min-speed', type=float, default=.2)
    parser.add_argument('--resolution', type=float, default=.04)
    parser.add_argument('--jobs', type=int, default=1)
    args = parser.parse_args()
    if not all(np.isfinite(v) and v > 0 for v in (args.max_speed, args.min_speed, args.resolution)) or args.max_speed < args.min_speed or args.jobs < 1:
        parser.error('Positive finite speeds/resolution/jobs required, max >= min')
    root = Path(args.output).resolve()
    root.mkdir(parents=True, exist_ok=False)
    profile_path = Path(args.profile).resolve()
    profile = load_motion_profile(profile_path)
    model = PinkIKCore(config=controller_config_for_profile(profile, lift_position_margin=0.),
                      base_mode='fixed', locked_joint_names=(LIFT,)).model
    specs_root = PROJECT_ROOT/'configs/benchmarks/generalization'
    plans = []
    for row in json.loads((specs_root/'suite.json').read_text())['cases']:
        if args.split != 'all' and row['split'] != args.split:
            continue
        path = specs_root/row['path']
        spec = json.loads(path.read_text())
        trajectory = DiverseTrajectory(model, initial_state(spec['posture']), spec, args.max_speed)
        envelope = envelope_report(trajectory, profile)
        plans.append(dict(path=str(path), spec=spec, ceiling=screened_speed(envelope, args.max_speed), envelope=envelope))
    write_json(root/'plan.json', dict(arguments=vars(args), profile_sha256=profile.sha256, paths=plans))
    def search(plan):
        trials = []
        def trial(speed):
            name = f'{profile_path.stem}__{plan["spec"]["name"]}__{speed:.9f}'
            command = [sys.executable, '-m', 'hexfellow_deploy.benchmark.run', '--trajectory-spec', plan['path'],
                       '--profile', str(profile_path), '--speed', str(speed), '--derivatives', '--output', str(root/name)]
            result = subprocess.run(command, cwd=PROJECT_ROOT, env=dict(os.environ,
                OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1'), text=True, capture_output=True)
            (root/(name+'.log')).write_text(result.stdout+result.stderr)
            summary = root/name/'summary.json'
            value = json.loads(summary.read_text()) if summary.exists() else dict(completed=False, failure=result.stderr[-3000:])
            public = evaluate(root/name) if summary.exists() and (root/name/'trace.npz').exists() else {}
            row = dict(name=name, split=plan['spec']['split'], profile=str(profile_path),
                requested_speed_m_s=speed, effective_request_m_s=speed,
                command=command, exit_code=result.returncode, result=value, public_tcp=public)
            trials.append(row)
            write_json(root/(plan['spec']['name']+'_trials.json'), trials)
            return bool(public.get('tracking_accepted', False))
        ceiling = plan['ceiling']
        passed, failed = None, None
        if trial(ceiling):
            passed = ceiling
        else:
            failed = ceiling
            speed = max(args.min_speed, ceiling/2)
            while speed < failed:
                if trial(speed):
                    passed = speed
                    break
                failed = speed
                if speed <= args.min_speed:
                    break
                speed = max(args.min_speed, speed/2)
            if passed is not None:
                while failed-passed > args.resolution:
                    mid = (passed+failed)/2
                    if trial(mid):
                        passed = mid
                    else:
                        failed = mid
        return dict(path=plan['spec']['name'], split=plan['spec']['split'],
                    passed_request_m_s=passed, adjacent_failed_request_m_s=failed,
                    screened_ceiling_m_s=ceiling, trials=trials)
    results = []
    with ThreadPoolExecutor(max_workers=args.jobs) as pool:
        for future in as_completed([pool.submit(search, plan) for plan in plans]):
            result = future.result()
            results.append(result)
            write_json(root/'brackets.json', sorted(results, key=lambda x: x['path']))
            write_json(root/'manifest.json', [trial for result in results for trial in result['trials']])
            print(result['path'], 'pass=', result['passed_request_m_s'], 'next_fail=', result['adjacent_failed_request_m_s'], flush=True)


if __name__ == '__main__':
    main()
