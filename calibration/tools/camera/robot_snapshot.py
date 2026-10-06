#!/usr/bin/env python3
"""Read-only measured robot/FK snapshot, sharing the existing onsite lock."""
import argparse
import fcntl
import json
from pathlib import Path
import sys
import time
ROOT=Path(__file__).resolve().parents[2]; sys.path.insert(0,str(ROOT/'src'))
import numpy as np
import pinocchio as pin
from policy_sim.hardware.whole_body import RobotMapping
from policy_sim.hardware.hexfellow_endpoint import HexWholeBodyEndpoint
from policy_sim.control.robot import CONTROLLED_JOINTS, BASE_FRAME
from calibration_paths import UPSTREAM


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--robot-config',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    # Mapping location does not identify the controller checkout. Share the
    # current UMI session lock used by this deployment's calibration runner.
    lockpath=UPSTREAM/'runs/.hexfellow-onsite.lock'
    lockpath.parent.mkdir(parents=True,exist_ok=True)
    model=pin.buildModelFromUrdf(str(ROOT/'urdf/maver_x4_gp80.urdf'));data=model.createData()
    result={'motion_commands_sent':0,'read_only':True,'mount_frame':'maver_x4_asm_URDF_link_1',
            'hand_frame':'maver_x4_asm_mount2','base_frame':BASE_FRAME}
    with lockpath.open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        ep=HexWholeBodyEndpoint(RobotMapping.load(args.robot_config),read_only=True)
        try:
            samples=[];last=None
            for _ in range(100):
                s=ep.state();now=time.monotonic();s.validate(now)
                samples.append(s.q.tolist());last=s;time.sleep(.01)
            q=pin.neutral(model)
            for name,value in zip(CONTROLLED_JOINTS,last.q[3:]):
                q[model.joints[model.getJointId(name)].idx_q]=value
            pin.framesForwardKinematics(model,data,q)
            poses={name:data.oMf[model.getFrameId(name)].homogeneous.copy() for name in
                   (BASE_FRAME,result['mount_frame'],result['hand_frame'])}
            result.update(q=last.q.tolist(),v=last.v.tolist(),received=last.received,
                          host_monotonic_s=time.monotonic(),host_realtime_ns=time.time_ns(),
                          sample_position_span=np.ptp(np.array(samples),axis=0).tolist(),
                          T_base_from_mount=(np.linalg.inv(poses[BASE_FRAME])@poses[result['mount_frame']]).tolist(),
                          T_mount_from_hand=(np.linalg.inv(poses[result['mount_frame']])@poses[result['hand_frame']]).tolist(),
                          T_base_from_hand=(np.linalg.inv(poses[BASE_FRAME])@poses[result['hand_frame']]).tolist(),
                          owned_sessions=sorted(ep.owned),motion_commands_sent=ep.write_count)
        finally: result['close_errors']=ep.close()
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open('x') as f:json.dump(result,f,indent=2)
    print(json.dumps(result,indent=2))
if __name__=='__main__':main()
