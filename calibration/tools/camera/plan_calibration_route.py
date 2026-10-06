#!/usr/bin/env python3
"""Select visible right-arm poses and check slow interpolated paths in MuJoCo."""
import json,sys,math
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
from validate_calibration_poses import Study,ROOT
from policy_sim.calibration.handeye import jacobian
OUT=ROOT/'runs/camera-calibration-40deg/route'
s=Study(json.loads((ROOT/'runs/camera-calibration/robot-snapshot.json').read_text()),OUT)
source=json.loads((ROOT/'runs/camera-calibration-40deg/sample/vision_candidates.json').read_text())
rows=source['poses'];q0=s.q0[s.ai];current=q0.copy();selected=[];segments=[]

def edge(a,b,fine=False):
    n=max(2,int(np.ceil(np.sum(np.abs(b-a))/(.005 if fine else .035))))
    minimum=.1
    for u in np.linspace(0,1,n+1):
        d,p=s.clearance(a+(b-a)*u);minimum=min(minimum,d)
        if d<.015:return False,minimum
    return True,minimum

def bridge(a,b):
    # Two-segment random joint-space bridge; every segment checked.
    for trial in range(300):
        waypoint=np.clip((a+b)/2+s.rng.normal(0,.65,7),s.lower+1e-6,s.upper-1e-6)
        if s.clearance(waypoint)[0]<.015:continue
        if edge(a,waypoint)[0] and edge(waypoint,b)[0] and edge(a,waypoint,True)[0] and edge(waypoint,b,True)[0]:return [waypoint,b]
    return None

remaining=list(range(len(rows)))
for index in range(30):
    ordered=sorted(remaining,key=lambda j:np.linalg.norm(np.array(rows[j]['right_arm_q'])-current))
    found=None
    for candidate in ordered:
        row=rows[candidate];a=np.array(row['right_arm_q'])
        # Avoid near-duplicate camera observations.
        cb=np.array(row['T_camera_from_board'])
        if selected and min(np.linalg.norm(cb[:3,3]-np.array(v['T_camera_from_board'])[:3,3])/.04+
           np.linalg.norm(Rotation.from_matrix(np.array(v['T_camera_from_board'])[:3,:3].T@cb[:3,:3]).as_rotvec())/.2 for v in selected)<.6:continue
        if edge(current,a)[0] and edge(current,a,True)[0]:found=(candidate,[a]);break
    if found is None:
        for candidate in ordered[:5]:
            path=bridge(current,np.array(rows[candidate]['right_arm_q']))
            if path is not None:found=(candidate,path);break
    if found is None:raise RuntimeError(f'No safe modeled path at pose {index+1}')
    candidate,path=found
    for dest in path:
        passed,gap=edge(current,dest,True)
        if not passed:raise RuntimeError('Dense edge failed; route must be replanned')
        delta=np.max(np.abs(dest-current))
        # Quintic zero-v/a endpoints, limit nominal joint v<=0.20 rad/s, a<=0.30 rad/s².
        duration=max(2.,1.875*delta/.20,math.sqrt(5.774*delta/.30))
        segments.append({'from_right_q':current.tolist(),'to_right_q':dest.tolist(),'duration_s':duration,'minimum_sampled_clearance_m':gap,'destination_pose':index+1 if np.array_equal(dest,path[-1]) else None})
        current=dest.copy()
    row=dict(rows[candidate]);row.update(pose_id=index+1,split='held_out' if index%5==4 else 'fit')
    selected.append(row);remaining.remove(candidate)
    print('Selected',index+1,'candidate',candidate,'route time',round(sum(v['duration_s'] for v in segments)),flush=True)
# Same validated path played backward for return; no new unchecked shortcut.
return_segments=[{**r,'from_right_q':r['to_right_q'],'to_right_q':r['from_right_q'],'destination_pose':None} for r in segments[::-1]]
g=[];c=[]
for row in selected:
    g.append(np.linalg.inv(s.mount_world)@s.fk(np.array(row['right_arm_q'])))
    c.append(row['T_camera_from_board'])
x=np.linalg.inv(s.mount_world)@s.camera
g=np.array(g);c=np.array(c);sv=np.linalg.svd(jacobian(g,c,x,s.hb),compute_uv=False)
report={'simulation_only':True,'hardware_execution_authorized':False,'validation_kind':'Kinematic poses and densely sampled geometric paths; not real controller dynamics',
 'initial_q':s.q18.tolist(),'initial_state_source':'Saved robot snapshot; fresh robot state unavailable',
 'T_hand_from_board_assumed':s.hb.tolist(),'board_attachment_measured':False,'camera_transform_calibrated':False,
 'environment':'Floor only; desk, cable, human and room collision not modeled',
 'collision_policy':'Convex visual geometry, same-weld and two-joint mechanical housing neighbor exclusions; board checked against all other weld groups',
 'collision_pair_count':len(s.pairs),'sample_max_l1_joint_step_rad':.005,'minimum_sampled_clearance_m':min(r['minimum_sampled_clearance_m'] for r in segments),
 'nominal_max_joint_speed_rad_s':.20,'nominal_max_joint_accel_rad_s2':.30,'hold_s':1.0,
 'joint_margin_rad':.05,'poses':selected,'segments':segments,'return_segments':return_segments,
 'handeye_jacobian_singular_values':sv.tolist(),'observability_ratio':float(sv[-1]/sv[0])}
(OUT/'route.json').write_text(json.dumps(report,indent=2));np.savez(OUT/'synthetic_pairs.npz',g=g,c=c,x=x,y=s.hb)
print('READY',len(selected),'ratio',sv[-1]/sv[0],flush=True)
