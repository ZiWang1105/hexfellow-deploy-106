#!/usr/bin/env python3
"""Plan and render three LEFT-hand validation views, entirely offline.

Uses deployed controller bounds and checks paths with the existing geometric
checker. This is not a live command entry point or hardware qualification.
"""
import argparse,json,sys,subprocess
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
from calibration_paths import UPSTREAM
DEPLOYMENT=UPSTREAM
sys.path.insert(0,str(ROOT/'src'))
import cv2,mujoco,numpy as np,pinocchio as pin
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation
from policy_sim.control.robot import LEFT_ARM_JOINTS
from validate_calibration_poses import Study
from calibration_path import LocalChecker
from validate_camera_overlay import OverlayRenderer,digest


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--extrinsics',type=Path,required=True)
    p.add_argument('--dataset',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--robot-config',type=Path,default=DEPLOYMENT/'config/local/hexfellow_gp80_highspeed_both_j6.json')
    p.add_argument('--snapshot',type=Path,default=ROOT/'config/camera/both_j6_sim_start.json')
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    fit=json.loads(a.extrinsics.read_text());dataset=json.loads(a.dataset.read_text())
    start=np.array(json.loads(a.snapshot.read_text())['q'])
    runtime=a.robot_config.resolve()
    # Keep deployment and camera-worktree asset resolution in separate processes,
    # just as the existing calibration geometry worker does.
    code='''import argparse,json,sys,numpy as np
from types import SimpleNamespace
sys.path.insert(0,sys.argv[1]+'/src')
from policy_sim.hardware import session
from policy_sim.control.motion_profile import controller_config_for_profile
from policy_sim.control.ik_core import PinkIKCore
from policy_sim.hardware.whole_body import WholeBodyState
from policy_sim.paths import PROJECT_ROOT
p=argparse.ArgumentParser();session.add_arguments(p)
a=p.parse_args(['--mode','sim','--robot-config',sys.argv[2]])
a.control_group='coupled';a.tracking_state='measured'
mapping,profile,_,_=session.load_configuration(a)
audit=json.loads((PROJECT_ROOT/'artifacts/reference/initial-state.json').read_text())['state']
state=WholeBodyState(**{**audit,'q':np.asarray(audit['q']),'v':np.asarray(audit['v'])})
core=PinkIKCore(config=controller_config_for_profile(profile),base_mode='planar',
    runtime_position_bounds=mapping.physical_bounds(state),measured_pose_tracking=True,
    **mapping.runtime_limits(profile,state))
print('BOUNDS '+json.dumps([core.model.lowerPositionLimit.tolist(),core.model.upperPositionLimit.tolist()]))
'''
    result=subprocess.run([sys.executable,'-c',code,str(DEPLOYMENT),str(runtime),json.dumps(start.tolist())],
        check=True,capture_output=True,text=True)
    bounds=np.array(json.loads(next(line[7:] for line in result.stdout.splitlines() if line.startswith('BOUNDS '))))
    s=Study({'q':start.tolist()},a.output/'geometry',
        attachment={'T_hand_from_board':fit['T_hand_from_board']},include_left=True)
    checker=object.__new__(LocalChecker);checker.study=s
    idx=[s.pm.joints[s.pm.getJointId(n)].idx_q for n in LEFT_ARM_JOINTS]
    hand=s.pm.getFrameId('maver_x4_asm_mount1')
    camera=s.mount_world@np.array(fit['T_mount_from_camera']);ic=np.linalg.inv(camera)
    renderer=OverlayRenderer(s.scene,fit['T_mount_from_camera'],dataset['camera']['K'],a.output/'render',
        board_transform=fit['T_hand_from_board'])
    lower,upper=bounds[:,4:11];lower=lower+.12;upper=upper-.12
    rng=np.random.default_rng(4003);base=s.q0.copy()
    center=np.array([0.,0.,.08,1.])
    R=np.array([[0,0,1],[0,1,0],[-1,0,0.]])
    desires=[(np.array([-.06,.015,.56]),R),
             (np.array([.055,.035,.60]),Rotation.from_euler('z',18,degrees=True).as_matrix()@R),
             (np.array([-.025,-.045,.50]),Rotation.from_euler('y',-18,degrees=True).as_matrix()@R)]
    def fk(joints):
        q=base.copy();q[idx]=joints;pin.framesForwardKinematics(s.pm,s.pd,q)
        return ic@s.pd.oMf[hand].homogeneous
    poses=[];previous=start.copy();path_reports=[]
    try:
        for number,(desired,orientation) in enumerate(desires,1):
            found=None
            for trial in range(100):
                seed=previous[4:11] if trial==0 else np.clip(previous[4:11]+rng.normal(0,.65,7),lower+1e-5,upper-1e-5)
                def residual(joints):
                    t=fk(joints)
                    return np.r_[12*((t@center)[:3]-desired),
                                 .5*Rotation.from_matrix(orientation.T@t[:3,:3]).as_rotvec(),.008*(joints-seed)]
                result=least_squares(residual,seed,bounds=(lower,upper),max_nfev=100)
                t=fk(result.x)
                if np.linalg.norm((t@center)[:3]-desired)>.015:continue
                target=start.copy();target[4:11]=result.x
                try:
                    checker.check_pose(target,bounds)
                    minimum=checker.edge_full(previous,target,bounds)
                    return_minimum=checker.edge_full(target,start,bounds)
                except ValueError:continue
                rgb,mask=renderer.render(target,'left')
                ys,xs=np.where(mask)
                if len(xs)<5000 or min(xs.min(),ys.min(),1919-xs.max(),1079-ys.max())<80:continue
                found=target;break
            if found is None:raise RuntimeError(f'No visible checked approach for validation pose {number}')
            report=dict(pose_id=number,q=found.tolist(),left_arm_rad=found[4:11].tolist(),
                gripper_center_camera_m=(t@center)[:3].tolist(),visible_gripper_pixels=int(mask.sum()),
                bbox_xyxy=[int(xs.min()),int(ys.min()),int(xs.max()),int(ys.max())],
                approach_minimum_clearance_m=minimum,return_to_default_minimum_clearance_m=return_minimum,
                joint_limit_margin_rad=float(np.min(np.r_[found[4:11]-bounds[0,4:11],bounds[1,4:11]-found[4:11]])))
            poses.append(report);path_reports.append(dict(from_q=previous.tolist(),to_q=found.tolist(),minimum_clearance_m=minimum))
            cv2.imwrite(str(a.output/f'pose-{number:02d}-camera.png'),cv2.cvtColor(rgb,cv2.COLOR_RGB2BGR))
            cv2.imwrite(str(a.output/f'pose-{number:02d}-left-mask.png'),np.uint8(mask)*255)
            overview=renderer.overview(found)
            cv2.imwrite(str(a.output/f'pose-{number:02d}-overview.png'),cv2.cvtColor(overview,cv2.COLOR_RGB2BGR))
            previous=found
            print('Planned pose',number,'minimum gap mm',minimum*1000,'pixels',len(xs),flush=True)
    finally:renderer.close()
    output=dict(simulation_only=True,hardware_commands_sent=0,hardware_execution_authorized=False,
        initial_q=start.tolist(),controller_bounds=bounds.tolist(),poses=poses,segments=path_reports,
        extrinsics_file=str(a.extrinsics.resolve()),extrinsics_sha256=digest(a.extrinsics),
        intrinsics=dataset['camera'],mapping_file=str(runtime),mapping_sha256=digest(runtime),
        scene=str(s.scene.resolve()),
        limitations='Sampled geometry only; desks, cables and people absent. Start is configured default, not live feedback. Must recheck actual approach before movement.')
    (a.output/'plan.json').write_text(json.dumps(output,indent=2))


if __name__=='__main__':main()
