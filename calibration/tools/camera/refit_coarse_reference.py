#!/usr/bin/env python3
"""Reproduce the frozen provisional board fit offline, without camera or SDK access.

This deliberately does not replace solve_handeye.py's accepted-capture checks.
The reference pairs were recovered by marker-guided boundary PnP, and are not
automatically promoted to accepted ChArUco observations or physical qualification.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src'))
from hexfellow_deploy.robotics.calibration.handeye import solve, summarize, validate


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pairs', type=Path,
                        default=ROOT / 'artifacts/reference/camera-calibration/coarse-board-pairs.json')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    document = json.loads(args.pairs.read_text())
    if document.get('schema_version') != 1 or not document.get('offline_only'):
        parser.error('Expected the frozen offline reference-pair schema')
    poses = document['observations']
    if len({row['pose_id'] for row in poses}) != len(poses):
        parser.error('Duplicate pose IDs')
    g = validate([row['T_mount_from_hand'] for row in poses])
    c = validate([row['T_camera_from_board'] for row in poses])
    held = np.arange(len(poses)) % 5 == 4
    if held.sum() == 0:
        parser.error('Reference requires held-out observations')
    result = solve(g[~held], c[~held])
    result.update(
        held_out=summarize(g[held], c[held], np.asarray(result['T_mount_from_camera']),
                           np.asarray(result['T_hand_from_board'])),
        training_pose_ids=[row['pose_id'] for row, h in zip(poses, held) if not h],
        held_out_pose_ids=[row['pose_id'] for row, h in zip(poses, held) if h],
        source_pairs=str(args.pairs.resolve()),
        source_pairs_sha256=hashlib.sha256(args.pairs.read_bytes()).hexdigest(),
        mount_frame=document['mount_frame'], hand_frame=document['hand_frame'],
        camera_frame=document['camera_frame'], offline_only=True, hardware_commands_sent=0,
        calibration_status='provisional boundary fit; requires physical validation',
        active_configuration_changed=False)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x') as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write('\n')
    print(json.dumps(dict(converged=result['converged'],
                         training_rms_mm=result['training']['position_rms_mm'],
                         held_out_rms_mm=result['held_out']['position_rms_mm']), indent=2))
    return 0 if result['converged'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
