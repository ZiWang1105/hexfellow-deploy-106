"""Evaluate the public pad-center TCP from immutable fingertip-frame traces."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import qpsolvers  # Preserve the shared-library initialization order.
import pinocchio as pin

from hexfellow_deploy.robot.joints import SIDES
from .run import statistics, longest_interval, write_json


def transform_trace(trace, offsets):
    """Convert core fingertip poses and body twists to the public tool point."""
    transforms = [pin.SE3(np.asarray(offsets[side])) for side in SIDES]
    inverse = np.array([transform.inverse().homogeneous for transform in transforms])
    result = {key: trace[key]@inverse for key in ('requested', 'measured')}
    result['tcp_velocity'] = np.stack([
        trace['tcp_velocity'][:, i]@transform.action.T for i, transform in enumerate(transforms)], axis=1)
    return result


def evaluate(directory):
    directory = Path(directory)
    metadata = json.loads((directory/'metadata.json').read_text())
    summary = json.loads((directory/'summary.json').read_text())
    with np.load(directory/'trace.npz') as trace:
        public = transform_trace(trace, metadata['controller']['policy_T_umi_tcp'])
        active = (trace['time_s'] >= metadata['motion_start_s']) & (trace['time_s'] < metadata['motion_end_s'])
    arms = [SIDES.index(s) for s in metadata.get('moving_arms', list(SIDES) if metadata['arguments']['dual'] else ['right'])]
    requested, measured = public['requested'][:, arms], public['measured'][:, arms]
    error = np.linalg.norm(requested[:, :, :3, 3]-measured[:, :, :3, 3], axis=2)*1000
    speed = np.linalg.norm(public['tcp_velocity'][:, arms, :3], axis=2)
    position = statistics(error[active])
    per_arm_position = {SIDES[arm]: statistics(error[active, i]) for i, arm in enumerate(arms)}
    orientation = np.array([[np.linalg.norm(pin.log3(a[:3, :3].T@b[:3, :3]))*180/np.pi
                            for a, b in zip(r, m)] for r, m in zip(requested, measured)])
    per_arm_orientation = {SIDES[arm]: statistics(orientation[active, i]) for i, arm in enumerate(arms)}
    per_arm = {SIDES[arm]: dict(peak_m_s=float(np.max(speed[:, i])),
        time_at_or_above_1p2_m_s=float(np.sum(speed[:, i] >= 1.2)*metadata['dt_s']),
        longest_continuous_interval_at_or_above_1p2_m_s=longest_interval(speed[:, i] >= 1.2, metadata['dt_s']))
        for i, arm in enumerate(arms)}
    acceptance = dict(summary['tracking_acceptance'])
    acceptance.update(position_rms_5mm=all(p is not None and p['rms'] <= 5. for p in per_arm_position.values()),
                      position_p95_10mm=all(p is not None and p['p95'] <= 10. for p in per_arm_position.values()),
                      orientation_p95_2deg=all(p is not None and p['p95'] <= 2. for p in per_arm_orientation.values()),
                      final_position_1mm=bool(np.max(error[-1]) <= 1.))
    accepted = all(acceptance.values())
    return dict(frame='public pad-center TCP; fixed tool transform from run metadata',
        position_mm=position, final_position_mm=error[-1].tolist(),
        per_arm_position_mm=per_arm_position, per_arm_orientation_deg=per_arm_orientation,
        orientation_deg=summary['metrics']['measured']['orientation_deg'],
        moving_arm_speed=per_arm, tracking_acceptance=acceptance, tracking_accepted=accepted,
        fast_tracking_accepted=accepted and all(
            row['longest_continuous_interval_at_or_above_1p2_m_s'] >= .1 for row in per_arm.values()),
        source_sha256={name: hashlib.sha256((directory/name).read_bytes()).hexdigest()
                       for name in ('trace.npz', 'metadata.json', 'summary.json')})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', help='A benchmark suite directory containing manifest.json')
    args = parser.parse_args()
    root = Path(args.directory)
    rows = json.loads((root/'manifest.json').read_text())
    results = {row['name']: evaluate(root/row['name']) for row in rows
               if (root/row['name']/'trace.npz').exists()}
    write_json(root/'public_tcp.json', dict(
        evaluator_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), cases=results))
    print(f'{len(results)} cases evaluated at the public TCP; '
          f'{sum(row["fast_tracking_accepted"] for row in results.values())} pass the fast criterion')


if __name__ == '__main__':
    main()
