#!/usr/bin/env python3
"""Generate a reproducible simulated fixed-33 D435i eye-in-hand route."""
import json
import math
from pathlib import Path

import numpy as np

from wrist_calibration_geometry import ROOT, DESIGN, PREVIEW, URDF, INSTALLATION, mapping_path, WristStudy, LocalWristChecker, sha, geometry_payload_hash, installed_camera_transforms


def configuration():
    import yaml
    MAPPING=mapping_path()
    snapshot=json.loads((PREVIEW/"robot_snapshot.json").read_text())
    seed=json.loads((PREVIEW/"planning_seed.json").read_text())
    bounds=json.loads((PREVIEW/"deployed_bounds.json").read_text())
    lo,hi=np.asarray(bounds["lower"]),np.asarray(bounds["upper"])
    lo[:3]=-1e6;hi[:3]=1e6
    default_path=ROOT/"config/hardware/hexfellow_default_pose_both_j6.yaml"
    default=yaml.safe_load(default_path.read_text())
    initial=np.r_[np.zeros(3),default["lift_m"],default["left_arm_rad"],default["right_arm_rad"]]
    data=json.loads((DESIGN/"design_and_validation.json").read_text())
    installation,body,x=installed_camera_transforms()
    y=np.asarray(seed["T_base_from_board"],float)
    center=(y@np.r_[.0435,.0435,0,1])[:3]
    table_height=float(center[2]-.012)
    # mapping_sha256 is checked against the runtime-selected mapping. Geometry
    # source references remain portable and need no old machine-local path.
    sources=[URDF,default_path,DESIGN/"design_and_validation.json",DESIGN/"assembly_mm/01_onepiece_ring_camera_33deg.stl",ROOT/"config/camera/charuco_87mm.json",PREVIEW/"planning_seed.json",PREVIEW/"deployed_bounds.json",INSTALLATION]
    return dict(schema_version=1,capture_side="right",pose_count=30,calibration_kind="eye_in_hand",
                initial_q=initial.tolist(),measured_start_q=snapshot["q"],parent_frame="maver_x4_asm_mount2",
                camera_frame="realsense_243222074562_color_optical",serial="243222074562",width=1280,height=720,
                K=[[909.8963623046875,0,643.9840698242188],[0,909.7846069335938,373.11004638671875],[0,0,1]],
                distortion_coefficients=[0,0,0,0,0],T_parent_from_optical_nominal=x.tolist(),T_base_from_board_nominal=y.tolist(),
                mount_installation=installation,
                board_placement_provenance=seed,nominal_mount_revision=data["revision"],fixed_camera_pitch_deg=33,
                bounds=[lo.tolist(),hi.tolist()],mapping_sha256=sha(MAPPING),urdf_sha256=sha(URDF),
                source_sha256={str(p.relative_to(ROOT)) if p.is_relative_to(ROOT) else str(p):sha(p) for p in sources},
                table_assumption=dict(center_m=[float(center[0]+.05),float(center[1]),table_height-.02],half_extents_m=[.32,.40,.02],planning_only=True,
                    note="Rough tabletop under head-observed board, not measured room geometry; footprint and height require physical review."),
                simulation_only=True,hardware_execution_authorized=False,poses=[])


def camera_pose(center,position,roll):
    from scipy.spatial.transform import Rotation
    z=center-position;z/=np.linalg.norm(z)
    x=np.cross([0,0,1],z)
    if np.linalg.norm(x)<.05:x=np.cross([1,0,0],z)
    x/=np.linalg.norm(x);y=np.cross(z,x)
    r=np.column_stack((x,y,z))@Rotation.from_rotvec([0,0,roll]).as_matrix()
    t=np.eye(4);t[:3,:3]=r;t[:3,3]=position
    return t


def image_target_center(plan):
    """Keep the board on the unobstructed side of the nominal gripper view."""
    rolled = np.isclose(float(plan['mount_installation'].get('optical_axis_roll_deg', 0.)) % 360., 180.)
    y_fraction = .72 if plan['capture_side'] == 'right' and rolled else (.25 if plan['capture_side'] == 'left' else 170./720.)
    return np.array([plan['width']/2, plan['height']*y_fraction])


def run(output,plan_path,configuration_factory=configuration):
    import cv2,mujoco
    from scipy.optimize import least_squares
    from scipy.spatial.transform import Rotation
    from policy_sim.calibration.charuco import detect
    from validate_calibration_poses import quat
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    plan=configuration_factory();checker=LocalWristChecker(output/"geometry",plan);s=checker.study
    left=plan['capture_side']=='left'
    arm=s.arm_slice
    target_center=image_target_center(plan)
    ray_offset=(plan['K'][1][2]-target_center[1])/plan['K'][1][1]
    initial=np.asarray(plan["initial_q"],float);current=initial.copy()
    default_wrist=s.fk(initial)
    if left:
        housing=np.asarray(plan['T_parent_from_housing_nominal'])
    else:
        _,housing,_=installed_camera_transforms()
    outside_y=float((default_wrist@housing)[1,3]-default_wrist[1,3])
    if (left and outside_y<=0) or (not left and outside_y>=0):raise RuntimeError('Camera is not outside the selected arm at default pose')
    plan['mount_side_validation']=dict(passed=True,default_camera_minus_wrist_base_y_m=outside_y,
        default_T_base_from_parent=default_wrist.tolist(),
        default_T_base_from_camera_housing=(default_wrist@housing).tolist())
    bounds=plan["bounds"]
    center=(s.y@np.r_[.0435,.0435,0,1])[:3]
    print("Initial modeled clearance",s.clearance(initial),flush=True)
    default_path=checker.plan(np.asarray(plan["measured_start_q"]),-2,bounds,initial)
    print("Default including upper-lift retreat validated",default_path["minimum_sampled_clearance_m"],flush=True)
    pool=[];attempted=0;rejections=dict(view=0,clearance=0,duplicate=0,detector=0)
    with mujoco.Renderer(s.m,height=800,width=1400) as renderer:
        for index in range(600 if left else 280):
            # Keep the board on the unobstructed image side, accounting for
            # physical sensor roll, while exciting position/viewing direction.
            # These are nominal optical targets, solved by bounded FK.
            dx=s.rng.uniform(-.20,.03) if left else s.rng.uniform(-.22,.08)
            dy=s.rng.uniform(-.12,.20) if left else s.rng.uniform(-.26,.16)
            dz=s.rng.uniform(.20,.32) if left else s.rng.uniform(.28,.46)
            desired=camera_pose(center,center+[dx,dy,dz],s.rng.uniform(-.45,.45))
            if index<4:desired=camera_pose(center,center+[-.15,.13 if left else -.13,.24+.025*index if left else .3+.025*index],0)
            def residual(a):
                q=initial.copy();q[arm]=a
                c=s.fk(q)@s.x
                to_board=c[:3,:3].T@(center-c[:3,3])
                # RGB roll is unconstrained: demanding a particular roll can
                # exhaust the short wrist joints even when the board is visible.
                # Two ray coordinates center the target while leaving that
                # redundant degree of freedom for a reachable arm posture.
                return np.r_[(c[:3,3]-desired[:3,3])*8,
                             np.array([to_board[0],to_board[1]+ray_offset*to_board[2]])*12,
                             .015*Rotation.from_matrix(desired[:3,:3].T@c[:3,:3]).as_rotvec(),
                             .003*(a-initial[arm])]
            candidates=[initial[arm],current[arm]]
            candidates.extend(s.rng.uniform(s.lower,s.upper) for _ in range(3))
            best=None
            for seed in candidates:
                fit=least_squares(residual,np.clip(seed,s.lower+1e-6,s.upper-1e-6),bounds=(s.lower,s.upper),max_nfev=110,ftol=1e-6,xtol=1e-7,gtol=1e-7)
                attempted+=1
                q=initial.copy();q[arm]=fit.x
                view=s.visibility(q)
                if view["margin_px"]<35 or view["facing_cosine"]<.7 or not (.17 if left else .25)<view["distance_m"]<(.43 if left else .55):
                    rejections['view']+=1;continue
                if abs(np.asarray(view["center_px"])-target_center).max()>90:
                    rejections['view']+=1;continue
                if s.clearance(q)[0]<.015:
                    rejections['clearance']+=1;continue
                if any(np.linalg.norm(q[arm]-r["q"][arm])<.09 for r in pool):
                    rejections['duplicate']+=1;continue
                image=s.render(q,renderer)
                result,_=detect(image,s.boardcfg,s.k,np.zeros(5))
                if not result["accepted"] or result["corners"]<12:
                    rejections['detector']+=1
                    if rejections['detector']==1:cv2.imwrite(str(output/'first-rejected-view.png'),image)
                    continue
                best=dict(q=q.tolist(),nominal_view=view,nominal_detection=result)
                break
            if best:
                pool.append(best);current=np.asarray(best["q"])
                if len(pool)%10==0:print("Sim-visible candidates",len(pool),"attempts",attempted,flush=True)
            if len(pool)>=(160 if left else 80):break
        print("Candidate pool",len(pool),"rejections",rejections,flush=True)
        (output/"candidate_pool.json").write_text(json.dumps(dict(poses=pool,attempted_ik_solves=attempted),indent=2))
        selected=[];segments=[];current=initial.copy()
        approach_segments=[]
        for waypoint in plan.get('approach_waypoints', []):
            waypoint=np.asarray(waypoint)
            gap,samples=checker.edge(current,waypoint,bounds)
            approach_segments.append(dict(destination='approach',minimum_clearance_m=gap,samples=samples))
            current=waypoint
        while len(selected)<30:
            order=sorted(range(len(pool)),key=lambda i:np.linalg.norm(np.asarray(pool[i]["q"])[arm]-current[arm]))
            found=None
            for i in order:
                row=pool[i]
                try:
                    gap,samples=checker.edge(current,np.asarray(row["q"]),bounds)
                except ValueError:continue
                found=i;break
            if found is None:raise RuntimeError(f"Only {len(selected)} directly connected valid poses")
            row=pool.pop(found);q=np.asarray(row["q"])
            row["pose_id"]=len(selected)+1
            selected.append(row);segments.append(dict(destination_pose=row["pose_id"],minimum_clearance_m=gap,samples=samples))
            current=q
        plan['poses']=selected
        return_report=checker.plan(current,-1,bounds,initial)
        plan["poses"]=selected;plan["scene_file"]=str(s.scene);plan["default_path"]=default_path
        plan["return_path"]=return_report;plan["segments"]=segments
        plan['approach_segments']=approach_segments
        plan["simulation_validation"]=dict(passed=True,poses_accepted=30,rendered_charuco_detections=30,minimum_detection_corners=min(r["nominal_detection"]["corners"] for r in selected),
            path_sample_count=sum(x["samples"] for x in segments+approach_segments)+default_path["path_samples"]+return_report["path_samples"],
            minimum_path_clearance_m=min([x["minimum_clearance_m"] for x in segments+approach_segments]+[default_path["minimum_sampled_clearance_m"],return_report["minimum_sampled_clearance_m"]]),
            target_joint_limit_reserve_rad=.15,collision_pair_count=len(s.pairs),method="Bounded optical-pose FK optimization; kinematic interpolation at <=.005 sum-joint-step, MuJoCo mesh/box separation, real ChArUco detection in 30 renders.",
            limitation="Geometric trajectory validation only. Motor dynamics, payload torque, cable routing, actual table and physical mounting are not qualified.")
        # Independent excitation screen for the eye-in-hand solver.
        camera_centers=np.array([(s.fk(np.asarray(r["q"]))@s.x)[:3,3] for r in selected])
        rotations=np.array([Rotation.from_matrix(np.asarray(selected[0]["nominal_view"]["T_base_from_parent"])[:3,:3].T@np.asarray(r["nominal_view"]["T_base_from_parent"])[:3,:3]).as_rotvec() for r in selected[1:]])
        plan["simulation_validation"]["camera_translation_span_m"]=np.ptp(camera_centers,axis=0).tolist()
        plan["simulation_validation"]["rotation_excitation_singular_values_rad"]=np.linalg.svd(rotations,compute_uv=False).tolist()
        if left:
            plan['simulation_validation'].update(intrinsics_measured=plan['intrinsics_provenance']['measured'],
                lift_and_right_arm_fixed=True,gripper_opening_m=.08,physical_devices_opened=False,hardware_commands_sent=0,
                controller_replay_performed=False,scope='Nominal kinematic pose and path validation; not MIT dynamics or physical qualification')
        plan['geometry_payload_sha256']=geometry_payload_hash(plan)
        plan_path.parent.mkdir(parents=True,exist_ok=True);plan_path.write_text(json.dumps(plan,indent=2)+"\n")
        # Full-resolution detection overlays and side-by-side robot views.
        import PIL.Image,PIL.ImageDraw
        thumbnails=[]
        for row in selected:
            q=np.asarray(row["q"]);image=s.render(q,renderer)
            _,overlay=detect(image,s.boardcfg,s.k,np.zeros(5))
            cv2.imwrite(str(output/f"pose-{row['pose_id']:02d}-wrist.png"),overlay)
            opt=mujoco.MjvOption();opt.geomgroup[3:6]=1
            camera=mujoco.MjvCamera();camera.type=mujoco.mjtCamera.mjCAMERA_FREE
            camera.lookat[:]=[.45,-.05,.8];camera.distance=2.25;camera.azimuth=145;camera.elevation=-22
            renderer.update_scene(s.d,camera=camera,scene_option=opt)
            third=renderer.render();cv2.imwrite(str(output/f"pose-{row['pose_id']:02d}-robot.png"),cv2.cvtColor(third,cv2.COLOR_RGB2BGR))
            combined=PIL.Image.new("RGB",(640,220),"white")
            combined.paste(PIL.Image.fromarray(third).resize((320,183)),(0,25))
            combined.paste(PIL.Image.fromarray(cv2.cvtColor(overlay,cv2.COLOR_BGR2RGB)).resize((320,180)),(320,25))
            PIL.ImageDraw.Draw(combined).text((8,6),f"Pose {row['pose_id']:02d}: {row['nominal_detection']['corners']} corners, rough table/board",fill="black")
            thumbnails.append(combined)
        sheet=PIL.Image.new("RGB",(1280,8*220),"white")
        for i,thumbnail in enumerate(thumbnails):sheet.paste(thumbnail,((i%2)*640,(i//2)*220))
        # Two separate 15-pose sheets retain readable scene/camera panels.
        for start in (0,15):
            sheet=PIL.Image.new("RGB",(1280,8*220),"white")
            for i,thumbnail in enumerate(thumbnails[start:start+15]):sheet.paste(thumbnail,((i%2)*640,(i//2)*220))
            sheet.save(output/f"poses-{start+1:02d}-{min(start+15,30):02d}.png")
    (output/"summary.json").write_text(json.dumps(plan["simulation_validation"],indent=2)+"\n")
    print("PASS",json.dumps(plan["simulation_validation"]),flush=True)
    return plan
