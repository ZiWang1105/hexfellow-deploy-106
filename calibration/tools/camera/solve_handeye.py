#!/usr/bin/env python3
"""Solve paired stationary observations, reserving every fifth pose for validation."""
import argparse
import json
from pathlib import Path
import sys
import numpy as np
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT.parent/'src'))
from hexfellow_deploy.robotics.calibration.handeye import solve, summarize, validate


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('dataset',type=Path);p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    data=json.loads(args.dataset.read_text())
    for name in ('mount_frame','hand_frame','camera_frame','board'):
        if name not in data: p.error(f'Missing dataset metadata: {name}')
    poses=data['observations']
    if len(poses)<30:p.error('Need at least 30 accepted distinct stationary poses')
    if len({r['pose_id'] for r in poses})!=len(poses):p.error('Duplicate pose IDs')
    if not all(r.get('accepted') is True and r.get('stationary') is True and r.get('synchronized') is True for r in poses):
        p.error('Every observation must pass visual, stationarity, and synchronization checks')
    g=validate([r['T_mount_from_hand'] for r in poses]);c=validate([r['T_camera_from_board'] for r in poses])
    held=np.arange(len(poses))%5==4
    result=solve(g[~held],c[~held])
    result['validation']=summarize(g[held],c[held],np.array(result['T_mount_from_camera']),np.array(result['T_hand_from_board']))
    result['training_pose_ids']=[r['pose_id'] for r,h in zip(poses,held) if not h]
    result['validation_pose_ids']=[r['pose_id'] for r,h in zip(poses,held) if h]
    result['metadata']={k:v for k,v in data.items() if k!='observations'}
    result['source_dataset']=str(args.dataset.resolve())
    # A low residual alone is not evidence of physical accuracy.
    result['calibration_status']='candidate_requires_physical_validation'
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open('x') as f: json.dump(result,f,indent=2,allow_nan=False)
    print(json.dumps({k:result[k] for k in ('converged','termination','training','validation')},indent=2))
    return 0 if result['converged'] else 1

if __name__=='__main__':raise SystemExit(main())
