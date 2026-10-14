#!/usr/bin/env python3
"""Offline right-arm IK candidates. Never connects to hardware or authorizes motion.

Mount/board geometry is approximate until measured; this is a planning seed.
"""
import argparse
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT.parent/'src'))
import numpy as np
import pinocchio as pin
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation
from hexfellow_deploy.robotics.control.robot import CONTROLLED_JOINTS, RIGHT_ARM_JOINTS


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--snapshot',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();snapshot=json.loads(args.snapshot.read_text())
    model=pin.buildModelFromUrdf(str(ROOT/'urdf/maver_x4_gp80.urdf'));data=model.createData()
    q0=pin.neutral(model)
    for name,value in zip(CONTROLLED_JOINTS,snapshot['q'][3:]):q0[model.joints[model.getJointId(name)].idx_q]=value
    indices=np.array([model.joints[model.getJointId(n)].idx_q for n in RIGHT_ARM_JOINTS])
    hand=model.getFrameId(snapshot['hand_frame']);mount=model.getFrameId(snapshot['mount_frame'])
    pin.framesForwardKinematics(model,data,q0);mount_world=data.oMf[mount].homogeneous.copy()
    lower,upper=model.lowerPositionLimit[indices]+.03,model.upperPositionLimit[indices]-.03
    rng=np.random.default_rng(87)
    def fk(arm):
        q=q0.copy();q[indices]=arm;pin.framesForwardKinematics(model,data,q)
        return data.oMf[hand].homogeneous.copy()
    def ik(target,seed):
        def residual(arm):
            t=fk(arm)
            return np.r_[5*(t[:3,3]-target[:3,3]),Rotation.from_matrix(target[:3,:3].T@t[:3,:3]).as_rotvec(),.0001*(arm-seed)]
        best=None
        for trial in range(8):
            start=seed if trial==0 else np.clip(seed+rng.normal(0,.7,7),lower+1e-5,upper-1e-5)
            fit=least_squares(residual,np.clip(start,lower+1e-5,upper-1e-5),bounds=(lower,upper),max_nfev=350,ftol=1e-10,xtol=1e-10,gtol=1e-10)
            t=fk(fit.x);dp=np.linalg.norm(t[:3,3]-target[:3,3]);dr=np.linalg.norm(Rotation.from_matrix(target[:3,:3].T@t[:3,:3]).as_rotvec())
            if best is None or dp+dr<best[1]+best[2]:best=(fit.x,dp,dr)
            if dp<.0005 and dr<.005:break
        return best
    # Flange +Z points down, nominal taped board face points toward camera.
    facing=np.array([[0,-1,0],[-1,0,0],[0,0,-1.]])
    families=[('near',[-.035,0,0]),('far',[.035,0,0]),('right',[0,-.035,0]),
              ('left',[0,.035,0]),('lower',[0,0,-.035]),('upper',[0,0,.035])]
    angles=[('front',[0,0,0]),('tilt_x_plus',[.25,0,0]),('tilt_x_minus',[-.25,0,0]),
            ('tilt_y_plus',[0,.25,0]),('tilt_y_minus',[0,-.25,0])]
    rows=[];seed=q0[indices].copy()
    # Approach waypoints keep the arm on its own side before approaching center.
    targets=[('approach_out',np.array([.28,-.32,-.30]),fk(seed)[:3,:3]),
             ('approach_up',np.array([.40,-.23,.12]),facing)]
    for family,delta in families:
        for label,rotvec in angles:
            targets.append((f'{family}_{label}',np.array([.40,-.06,.22])+delta,Rotation.from_rotvec(rotvec).as_matrix()@facing))
    for idx,(label,position,rotation) in enumerate(targets):
        local=np.eye(4);local[:3,:3]=rotation;local[:3,3]=position
        target=mount_world@local;arm,dp,dr=ik(target,seed)
        if dp>.002 or dr>.02:raise RuntimeError(f'{label} unreachable: {dp:.4f}m {dr:.4f}rad')
        q=np.asarray(snapshot['q']).copy();q[:3]=0;q[11:]=arm
        rows.append({'pose_id':label,'calibration_pose':idx>=2,'T_mount_from_hand':local.tolist(),
                     'q':q.tolist(),'ik_position_error_m':float(dp),'ik_angle_error_rad':float(dr)})
        seed=arm
    result={'simulation_only':True,'hardware_execution_authorized':False,'camera_visibility_verified':False,
            'mount_frame':snapshot['mount_frame'],'hand_frame':snapshot['hand_frame'],
            'board_pattern_m':.087,'backing_m':.100,'initial_q':snapshot['q'],
            'limitations':['Camera/board transforms are not yet measured','Taped board rigidity and payload unknown',
                'Live publisher limits, self-collision, cables and obstacles require validation'],
            'poses':rows}
    with args.output.open('x') as f:json.dump(result,f,indent=2)
    print(f'Generated {len(rows)-2} calibration poses and 2 approach waypoints; IK only so far.')
if __name__=='__main__':main()
