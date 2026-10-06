#!/usr/bin/env python3
"""Offline centered ChArUco planning with approximate attachment and image reserve."""
import json,hashlib,sys
from pathlib import Path
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation
from validate_calibration_poses import Study,ROOT
from policy_sim.control.motion_profile import controller_config_for_profile

OUT=ROOT/'runs/centered-calibration';OUT.mkdir(parents=True,exist_ok=True)
attachment=json.loads((ROOT/'config/camera/board_right_outward.json').read_text())
# Photo-scale estimate: pattern center nearer the gripper root than previous 160 mm.
hb=np.array(attachment['T_hand_from_board']);hb[2,3]-=.04;hb[:3,3]-=hb[:3,2]*.025
attachment.update(T_hand_from_board=hb.tolist(),position_assumption='Rough photo estimate: pattern center 120 mm along flange +Z, 65 mm outward from the flange reference. Not measured. Planning tests include +/-40 mm position uncertainty.',geometry_status='approximate, not calibrated')
(OUT/'attachment.json').write_text(json.dumps(attachment,indent=2))
q=np.array(json.loads((ROOT/'runs/live-calibration/20260930T044638.695810Z/camera/request.json').read_text())['q'])
s=Study(dict(q=q.tolist()),OUT/'geometry',attachment=attachment,include_left=True)
cfg=controller_config_for_profile(ROOT/'config/control/motion_profiles/x7-gp80-highspeed-fixed-base-v1.yaml')
lo=s.pm.lowerPositionLimit[s.ai]+cfg.arm_position_margin+.15
hi=s.pm.upperPositionLimit[s.ai]-cfg.arm_position_margin-.15
if cfg.arm_position_lower_limits is not None:
 lo=np.maximum(lo,np.array(cfg.arm_position_lower_limits)[-7:]+cfg.arm_position_margin+.15)
 hi=np.minimum(hi,np.array(cfg.arm_position_upper_limits)[-7:]-cfg.arm_position_margin-.15)
s.lower=lo;s.upper=hi
pool=json.loads((ROOT/'config/camera/right_outward_candidates.json').read_text())['poses'];rng=np.random.default_rng(4030);ic=np.linalg.inv(s.camera)
center=np.array([.0435,.0435,0,1]);corners=np.array([[x,y,0] for x in [0,.087] for y in [0,.087]])
rows=[]
for i in range(1500):
 seed=np.array(pool[i%len(pool)]['right_arm_q']) if i< len(pool) else rows[int(rng.integers(len(rows)))]['right_arm_q'] if rows and i%2 else rng.uniform(lo,hi)
 seed=np.clip(seed,lo+1e-6,hi-1e-6)
 # Preserve diverse feasible board rotations while steering the center to the optical axis.
 start=ic@s.fk(seed)@hb
 desired=np.array([rng.uniform(-.035,.035),rng.uniform(-.025,.025),rng.uniform(.40,.65)])
 rotation=start[:3,:3]@Rotation.from_rotvec(rng.normal(0,.18,3)).as_matrix()
 def residual(a):
  t=ic@s.fk(a)@hb;c=(t@center)[:3]
  return np.r_[(c-desired)*12,Rotation.from_matrix(rotation.T@t[:3,:3]).as_rotvec()*.45,(a-seed)*.012]
 fit=least_squares(residual,seed,bounds=(lo,hi),max_nfev=65)
 a=fit.x;v=s.visibility(a);t=np.array(v['T_camera_from_board']);c=(t@center)[:3];pixel=s.k@c;pixel=pixel[:2]/pixel[2]
 if abs(pixel[0]-960)>140 or abs(pixel[1]-540)>100 or v['margin_px']<250 or v['facing_cosine']<.75 or not .35<v['distance_m']<.75:continue
 # Pessimistic simultaneous corner displacement (attachment/camera position) and ray rotation.
 points=corners@t[:3,:3].T+t[:3,3];worst=10000.
 for shift in np.array(np.meshgrid(*[[-.04,.04]]*3)).T.reshape(-1,3):
  for angle in [(-8,-8),(-8,8),(8,-8),(8,8)]:
   xyz=(points+shift)@Rotation.from_euler('xy',angle,degrees=True).as_matrix().T
   uv=xyz@s.k.T;uv=uv[:,:2]/uv[:,2:]
   worst=min(worst,float(np.min(np.c_[uv[:,0],1920-uv[:,0],uv[:,1],1080-uv[:,1]])))
 if worst<80:continue
 if any(np.linalg.norm(a-np.array(r['right_arm_q']))<.18 for r in rows):continue
 gap,_=s.clearance(a)
 if gap<.02:continue
 rows.append(dict(right_arm_q=a.tolist(),center_px=pixel.tolist(),uncertainty_edge_margin_px=worst,minimum_model_clearance_m=gap,**v))
 if len(rows)%10==0:print('Centered candidates',len(rows),'trials',i+1,flush=True)
 if len(rows)>=80:break
if len(rows)<30:raise RuntimeError(f'Only {len(rows)} centered poses; no replacement installed')
(OUT/'candidates.json').write_text(json.dumps(dict(poses=rows,attachment_sha256=hashlib.sha256((OUT/'attachment.json').read_bytes()).hexdigest(),simulation_only=True),indent=2))
print('GENERATED',len(rows),flush=True)
