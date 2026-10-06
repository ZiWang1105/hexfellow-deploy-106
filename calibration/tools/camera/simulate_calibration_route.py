#!/usr/bin/env python3
"""Offline position-servo physics check; NOT the hardware MIT execution controller."""
import json,sys
from pathlib import Path
import numpy as np,mujoco
from types import SimpleNamespace
from validate_calibration_poses import Study,ROOT
from policy_sim.control.self_collision import MuJoCoSelfCollisionOracle
OUT=ROOT/'runs/camera-calibration-40deg/route'
r=json.loads((OUT/'route.json').read_text())
s=Study(json.loads((ROOT/'runs/camera-calibration/robot-snapshot.json').read_text()),OUT)
m=s.m;d=s.d;s.update(s.q0[s.ai]);d.qvel[:]=0
from policy_sim.control.robot import CONTROLLED_JOINTS
certificate=SimpleNamespace(model=m,_convex_meshes={});ambiguous=[]
act=[m.actuator('position_'+n).id for n in CONTROLLED_JOINTS]
# All nonmoving coordinates keep their recorded references throughout.
base=s.q0[s.pi].copy();d.ctrl[act]=base
segments=[]
for v in r['segments']:
 segments.append(dict(v,hold=False))
 if v['destination_pose'] is not None:
  segments.append({**v,'from_right_q':v['to_right_q'],'duration_s':2.,'hold':True})
segments.extend(dict(v,hold=False) for v in r['return_segments'])
segments.append({'from_right_q':r['segments'][0]['from_right_q'],'to_right_q':r['segments'][0]['from_right_q'],'duration_s':2.,'hold':True,'destination_pose':None})
trace=[];stops=[];t=0.;worst_error=0.;min_gap=.1;min_margin=10.;nonmoving=0.;checks=0
for seg in segments:
 a=np.array(seg['from_right_q']);b=np.array(seg['to_right_q']);duration=seg['duration_s']
 steps=int(np.ceil(duration/m.opt.timestep));sample_every=max(1,round(.02/m.opt.timestep))
 for step in range(steps):
  u=min(1.,(step+1)/steps);w=10*u**3-15*u**4+6*u**5;ref=base.copy();ref[-7:]=a+w*(b-a);d.ctrl[act]=ref
  mujoco.mj_step(m,d)
  if step%sample_every==0 or step==steps-1:
   q=d.qpos[s.mi].copy();v=d.qvel[[m.joint(n).dofadr[0] for n in CONTROLLED_JOINTS]].copy()
   err=float(np.max(np.abs(q[-7:]-ref[-7:])));worst_error=max(worst_error,err)
   nonmoving=max(nonmoving,float(np.max(np.abs(q[:8]-base[:8]))),float(np.max(np.abs(d.qpos[:3]))))
   min_margin=min(min_margin,float(np.min(np.r_[q[-7:]-s.pm.lowerPositionLimit[s.ai],s.pm.upperPositionLimit[s.ai]-q[-7:]])))
   # Geometric check uses actual simulated q; retain physics state in original data.
   best=.1
   for g1,g2 in s.pairs:
    distance=s.distance(d,g1,g2,best)
    if distance<=0:
     proof=MuJoCoSelfCollisionOracle._mesh_separation_certificate(certificate,d,g1,g2)
     ambiguous.append({'time_s':float(d.time),'geom_ids':[g1,g2],'mujoco_distance_m':float(distance),'all_vertex_separation_bound_m':proof})
     if proof is not None:distance=proof
    if distance<best:best=float(distance)
   min_gap=min(min_gap,best);checks+=1
   trace.append((float(d.time),q.copy(),seg['destination_pose'] or 0,bool(seg['hold']),d.qpos.copy()))
 if seg['hold'] and seg['destination_pose'] is not None:
  stops.append({'pose_id':seg['destination_pose'],'q':d.qpos[s.mi].tolist(),'v':d.qvel[[m.joint(n).dofadr[0] for n in CONTROLLED_JOINTS]].tolist(),'time_s':float(d.time),'max_right_tracking_error_rad':float(np.max(np.abs(d.qpos[s.mi][-7:]-b)))})
  print('Dwell',seg['destination_pose'],'tracking rad',stops[-1]['max_right_tracking_error_rad'],flush=True)
np.savez_compressed(OUT/'physics_trace.npz',time=np.array([v[0] for v in trace]),q=np.array([v[1] for v in trace]),pose=np.array([v[2] for v in trace]),hold=np.array([v[3] for v in trace]),qpos=np.array([v[4] for v in trace]))
report={'simulation_only':True,'controller':'MuJoCo position servos with model gravity compensation; NOT deployed MIT controller','duration_s':float(d.time),'physics_timestep_s':m.opt.timestep,'collision_samples':checks,'nonpositive_distance_checks':ambiguous,'minimum_sampled_clearance_m':min_gap,'minimum_right_joint_margin_rad':min_margin,'maximum_right_tracking_error_rad':worst_error,'max_nonmoving_coordinate_deviation':nonmoving,'dwells':stops,'no_modeled_collision':min_gap>0,'holds_track_within_0p02rad':all(v['max_right_tracking_error_rad']<.02 for v in stops),'hardware_execution_authorized':False}
(OUT/'physics_report.json').write_text(json.dumps(report,indent=2));print({k:v for k,v in report.items() if k not in ('dwells','nonpositive_distance_checks')},flush=True)
