#!/usr/bin/env python3
"""Independently check/render the nominal left D405 route. Opens no hardware."""
import argparse
import json
from pathlib import Path
import sys

import cv2
import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

from plan_left_wrist_calibration import PLAN
from wrist_calibration_geometry import ROOT, load_plan, LocalWristChecker, sha
from render_wrist_open_gripper import Video, panel, label
from policy_sim.calibration.charuco import detect
from policy_sim.calibration.wrist_handeye import solve


def pose_error(estimate, truth):
    delta=np.linalg.inv(truth)@estimate
    return dict(translation_mm=float(np.linalg.norm(delta[:3,3])*1000),
                rotation_deg=float(np.degrees(Rotation.from_matrix(delta[:3,:3]).magnitude())))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan',type=Path,default=PLAN)
    parser.add_argument('--output',type=Path,default=ROOT/'runs/left-wrist-calibration-preview')
    args=parser.parse_args();out=args.output;out.mkdir(parents=True,exist_ok=True)
    plan=load_plan(args.plan);checker=LocalWristChecker(out/'geometry',plan);study=checker.study
    nodes=plan['approach_waypoints']+[row['q'] for row in plan['poses']]+plan['return_path']['targets']
    origin=np.asarray(plan['initial_q']);minimum=.1;samples=0
    for node in nodes:
        node=np.asarray(node)
        np.testing.assert_array_equal(node[:4],origin[:4]);np.testing.assert_array_equal(node[11:],origin[11:])
        gap,count=checker.edge(origin,node,plan['bounds']);minimum=min(minimum,gap);samples+=count;origin=node
    video=Video(out/'left-d405-30-poses.mp4')
    g=[];c=[];rows=[];tile_frames=[]
    with mujoco.Renderer(study.m,height=800,width=1400) as renderer:
        for row in plan['poses']:
            q=np.asarray(row['q']);image=study.render(q,renderer)
            result,annotated=detect(image,study.boardcfg,study.k,np.zeros(5))
            if not result['accepted'] or result['corners']<12:raise ValueError('Current left scene does not detect the board')
            camera=mujoco.MjvCamera();camera.type=mujoco.mjtCamera.mjCAMERA_FREE
            camera.lookat[:]=[.40,.10,.80];camera.distance=2.2;camera.azimuth=145;camera.elevation=-22
            opt=mujoco.MjvOption();opt.geomgroup[3:5]=1;opt.geomgroup[5]=0
            renderer.update_scene(study.d,camera=camera,scene_option=opt)
            robot=cv2.cvtColor(renderer.render(),cv2.COLOR_RGB2BGR)
            frame=np.full((720,1280,3),(17,21,27),np.uint8)
            label(frame,f'Left D405 calibration | pose {row["pose_id"]:02d} / 30',(24,44),.9)
            label(frame,'Fixed 32-degree CAD, outside-left assumption | verified D405 SDK intrinsics',(24,82),.65)
            label(frame,'Whole robot / open left gripper',(24,128),.65)
            label(frame,'Left D405 view / detected board corners',(654,128),.65)
            frame[148:498,16:636]=panel(robot,620,350)
            frame[148:498,644:1264]=panel(annotated,620,350)
            label(frame,f'{result["corners"]} corners | distance {result["distance_m"]:.3f} m | lift held at 300 mm',(24,554),.72)
            label(frame,'30 checked target poses; connecting paths checked separately at <=0.005 rad sum steps',(24,604),.61)
            label(frame,'Static target preview, 2 s each. No hardware motion or MIT controller replay.',(24,651),.61)
            label(frame,'Physical mounting and controller rehearsal are still required before a live route is enabled.',(24,691),.58)
            cv2.imwrite(str(out/f'pose-{row["pose_id"]:02d}.png'),frame)
            for _ in range(60):video.add(frame)
            tile_frames.append(frame)
            g.append(study.fk(q));c.append(np.asarray(result['T_camera_from_board']))
            rows.append(dict(pose_id=row['pose_id'],q=q.tolist(),T_base_from_wrist=g[-1].tolist(),
                             **result))
    metadata=video.close()
    # Only detector PnP and FK enter the solver. Nominal truth is used after fitting.
    g=np.asarray(g);c=np.asarray(c);held=np.arange(5,30,6);train=np.setdiff1d(np.arange(30),held)
    fit=solve(g[train],c[train],parent_frame=plan['parent_frame'],camera_serial=plan['serial'],
              camera_frame=plan['camera_frame'],validation_g=g[held],validation_c=c[held])
    errors=dict(camera_mount=pose_error(np.asarray(fit['T_wrist_from_camera']),study.x),
                stationary_board=pose_error(np.asarray(fit['T_base_from_board']),study.y))
    report=dict(passed=True,scope='Nominal kinematic geometry + rendered PnP; not physical calibration',
        poses=30,minimum_detected_corners=min(r['corners'] for r in rows),
        path_samples=samples,minimum_path_clearance_m=minimum,lift_and_right_fixed=True,
        plan_sha256=sha(args.plan),intrinsics_provenance=plan['intrinsics_provenance'],
        held_out_pose_ids=(held+1).tolist(),solver_vs_nominal_ground_truth=errors,
        physical_devices_opened=False,hardware_commands_sent=0,controller_replay_performed=False,video=metadata)
    if errors['camera_mount']['translation_mm']>5 or errors['camera_mount']['rotation_deg']>1:
        raise ValueError('Nominal solver recovery failed: '+json.dumps(errors))
    (out/'rendered-observations.json').write_text(json.dumps(rows,indent=2)+'\n')
    (out/'simulated-fit.json').write_text(json.dumps(fit,indent=2)+'\n')
    (out/'validation.json').write_text(json.dumps(report,indent=2)+'\n')
    for start in (0,15):
        sheet=np.full((8*180,1280,3),25,np.uint8)
        for i,frame in enumerate(tile_frames[start:start+15]):
            sheet[(i//2)*180:(i//2+1)*180,(i%2)*640:(i%2+1)*640]=cv2.resize(frame,(640,180))
        cv2.imwrite(str(out/f'poses-{start+1:02d}-{start+15:02d}.png'),sheet)
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
