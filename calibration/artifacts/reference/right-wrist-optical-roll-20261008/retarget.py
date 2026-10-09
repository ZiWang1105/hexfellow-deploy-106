"""Offline revalidation of the same 30 joint poses after sensor-roll correction."""
from pathlib import Path
import json,sys
import cv2,mujoco,numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation
root=next(p for p in Path(__file__).resolve().parents if (p/'config/camera/cameras.json').exists())
sys.path.insert(0,str(root/'tools/camera'));sys.path.insert(0,str(root/'src'))
from wrist_calibration_geometry import LocalWristChecker,installed_camera_transforms,geometry_payload_hash,sha
from policy_sim.calibration.charuco import detect
from plan_wrist_calibration import image_target_center
out=Path(__file__).parent
path=root/'config/camera/realsense_d435i_right_fixed33_outside_plan.json'
plan=json.loads((out/'plan-before.json').read_text())
installation,housing,optical=installed_camera_transforms()
plan['mount_installation']=installation
plan['T_parent_from_optical_nominal']=optical.tolist()
plan['source_sha256']['config/camera/right_fixed33_installation.json']=sha(root/'config/camera/right_fixed33_installation.json')
checker=LocalWristChecker(out/'revalidated-geometry',plan);study=checker.study
origin=np.asarray(plan['initial_q']);minimum=.1;count=0;segments=[]
with mujoco.Renderer(study.m,height=800,width=1400) as renderer:
 for row in plan['poses']:
  q=np.asarray(row['q']);arm=study.arm_slice
  before_q=q.copy();before_view=study.visibility(q)
  desired_camera=study.fk(q)@study.x
  target_center=image_target_center(plan)
  ray=(target_center-np.asarray(plan['K'])[:2,2])/np.diag(np.asarray(plan['K']))[:2]
  center=(study.y@np.array([.0435,.0435,0,1.]))[:3]
  def residual(a):
   candidate=before_q.copy();candidate[arm]=a
   c=study.fk(candidate)@study.x
   p=c[:3,:3].T@(center-c[:3,3])
   return np.r_[(c[:3,3]-desired_camera[:3,3])*8,
       (p[:2]-ray*p[2])*12,
       .015*Rotation.from_matrix(desired_camera[:3,:3].T@c[:3,:3]).as_rotvec(),
       .01*(a-before_q[arm])]
  solution=least_squares(residual,q[arm],bounds=(study.lower,study.upper),max_nfev=100,
      ftol=1e-8,xtol=1e-8,gtol=1e-8)
  q[arm]=solution.x
  gap,n=checker.edge(origin,q,plan['bounds'])
  row['q']=q.tolist()
  row['optical_roll_retargeting']={'before_q':before_q.tolist(),
      'target_center_px':target_center.tolist(),'max_joint_change_deg':float(np.max(np.abs(q[arm]-before_q[arm]))*180/np.pi)}
  view=study.visibility(q);image=study.render(q,renderer)
  detection,annotated=detect(image,study.boardcfg,study.k,np.zeros(5))
  print(row['pose_id'],'corners',detection['corners'],'margin',view['margin_px'],flush=True)
  if view['margin_px']<35 or view['facing_cosine']<.7 or not detection['accepted'] or detection['corners']<12:
   raise ValueError(f'Pose {row["pose_id"]} needs a new validated target')
  row['nominal_view']=view;row['nominal_detection']=detection
  segments.append(dict(destination_pose=row['pose_id'],minimum_clearance_m=gap,samples=n))
  minimum=min(minimum,gap);count+=n;origin=q
  cv2.imwrite(str(out/f'pose-{row["pose_id"]:02d}-wrist-corrected.png'),annotated)
plan['default_path']=checker.plan(np.asarray(plan['initial_q']),-2,plan['bounds'],np.asarray(plan['initial_q']))
plan['return_path']=checker.plan(origin,-1,plan['bounds'],np.asarray(plan['initial_q']))
minimum=min(minimum,plan['return_path']['minimum_sampled_clearance_m']);count+=plan['return_path']['path_samples']+plan['default_path']['path_samples']
plan['segments']=segments;plan['scene_file']=str(study.scene)
plan['simulation_validation']=dict(plan['simulation_validation'],poses_accepted=30,path_sample_count=count,
 minimum_path_clearance_m=minimum,collision_pair_count=len(study.pairs),
 minimum_detection_corners=min(r['nominal_detection']['corners'] for r in plan['poses']),
 method='Same joint route rechecked for optical-axis roll +180 degrees using fresh rendering/PnP and collision-path samples',
 source_previous_plan_sha256=sha(out/'plan-before.json'),physical_execution_validated=False,
 camera_roll_changed_without_display_flip=True)
frames=[study.fk(np.asarray(row['q']))@study.x for row in plan['poses']]
centers=np.asarray([c[:3,3] for c in frames])
rotations=np.asarray([Rotation.from_matrix(frames[0][:3,:3].T@c[:3,:3]).as_rotvec() for c in frames[1:]])
v=plan['simulation_validation']
v['camera_translation_span_m']=np.ptp(centers,axis=0).tolist()
v['rotation_excitation_singular_values_rad']=np.linalg.svd(rotations,compute_uv=False).tolist()
v['method']='Retargeted the 30-pose route for optical-axis roll +180 degrees; fresh rendered detections and all collision-path checks'
v['minimum_image_margin_px']=min(r['nominal_view']['margin_px'] for r in plan['poses'])
v['maximum_joint_retargeting_deg']=max(r['optical_roll_retargeting']['max_joint_change_deg'] for r in plan['poses'])
v['target_image_center_px']=image_target_center(plan).tolist()
plan['geometry_payload_sha256']=geometry_payload_hash(plan)
plan['optical_orientation_revision']='2026-10-08: current physical sensor optical roll corrected by +180 degrees'
(out/'revalidated-plan.json').write_text(json.dumps(plan,indent=2)+'\n')
(out/'revalidation.json').write_text(json.dumps(plan['simulation_validation'],indent=2)+'\n')
tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(plan,indent=2)+'\n');tmp.replace(path)
print('PASS',json.dumps(plan['simulation_validation']))
