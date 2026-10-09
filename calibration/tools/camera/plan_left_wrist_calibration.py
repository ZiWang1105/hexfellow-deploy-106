#!/usr/bin/env python3
"""Plan LEFT D405 calibration offline using the dedicated fixed-32 CAD.

The device is not required. Default K is the left D405 active-profile SDK
snapshot verified on 2026-10-08. Supply SDK ready.json with --intrinsics to
regenerate against a different active-profile pinhole calibration.
No robot SDK is opened and this tool cannot execute physical motion.
"""
import argparse
import json
from pathlib import Path

import numpy as np

from plan_wrist_calibration import configuration as right_configuration, run
from wrist_calibration_geometry import ROOT, sha, LocalWristChecker, load_plan, geometry_payload_hash


PLAN = ROOT / 'config/camera/realsense_d405_left_fixed32_plan.json'
DESIGN = ROOT / 'assets_local/left_fixed32'
INSTALLATION = ROOT / 'config/camera/left_fixed32_installation.json'
INTRINSICS = ROOT / 'config/camera/realsense_d405_rgb_intrinsics_20261008.json'


def configuration(intrinsics=None, board_center=(.68, .12, .72)):
    plan = right_configuration()
    design = json.loads((DESIGN / 'design_and_validation.json').read_text())
    installation = json.loads(INSTALLATION.read_text())
    body = np.asarray(design['camera_transform_mm'], float)
    body[:3, 3] *= .001
    body = np.asarray(installation['T_parent_from_design']) @ body
    local = np.eye(4)
    # CAD: X up, Y right, Z forward. Optical: X right, Y down, Z forward.
    local[:3, :3] = [[0, -1, 0], [1, 0, 0], [0, 0, 1]]
    local[:3, 3] = [0, -.009, .0192]  # D405 left imager = RGB/depth nominal origin.
    y = np.eye(4)
    y[:3, :3] = np.diag([1., -1., -1.])
    center = np.asarray(board_center, float)
    y[:3, 3] = center - y[:3, :3] @ [.0435, .0435, 0]
    intrinsics=Path(intrinsics or INTRINSICS)
    metadata = json.loads(intrinsics.read_text())
    if str(metadata.get('serial')) != '230422271628':
        raise ValueError('Intrinsics must identify the left D405 serial 230422271628')
    if metadata.get('distortion_model') != 'none' or np.any(metadata.get('distortion', [1])):
        raise ValueError('Use zero-distortion pinhole metadata from the calibration RGB worker')
    width, height, k = int(metadata['width']), int(metadata['height']), metadata['K']
    k_array=np.asarray(k,float)
    if k_array.shape!=(3,3) or not np.isfinite(k_array).all() or min(k_array[0,0],k_array[1,1])<=0:
        raise ValueError('Invalid SDK pinhole K')
    provenance = dict(status='SDK_active_profile', measured=True, source_sha256=sha(intrinsics),
                      native_distortion_model=metadata['native_distortion_model'],
                      native_distortion=metadata['native_distortion'],
                      corner_processing=metadata['corner_processing'])
    sources = [INSTALLATION, DESIGN/'design_and_validation.json',
               DESIGN/'assembly_mm/01_D405_LEFT_fixed32_wrist_mount.stl',
               DESIGN/'reference/d405_official_geometry_reference.json',
               DESIGN/'reference/d405_official_camera_frame_mm.stl',
               ROOT/'config/hardware/hexfellow_default_pose_both_j6.yaml',
               ROOT/'config/camera/charuco_87mm.json', ROOT/'urdf/maver_x4_gp80_zed_40deg.urdf',
               ROOT/'artifacts/reference/wrist-planning/deployed_bounds.json']
    if intrinsics.is_relative_to(ROOT):sources.append(intrinsics)
    plan.update(capture_side='left', parent_frame='maver_x4_asm_mount1',
        camera_frame='realsense_230422271628_color_optical', serial='230422271628',
        width=width, height=height, K=k, intrinsics_provenance=provenance,
        T_parent_from_optical_nominal=(body@local).tolist(), T_parent_from_housing_nominal=body.tolist(),
        T_base_from_board_nominal=y.tolist(), mount_installation=installation,
        fixed_camera_pitch_deg=32, nominal_mount_revision=design['revision'],
        mount_mesh='assets_local/left_fixed32/assembly_mm/01_D405_LEFT_fixed32_wrist_mount.stl',
        camera_mesh='assets_local/left_fixed32/reference/d405_official_camera_frame_mm.stl',
        board_placement_provenance=dict(planning_only=True, source='Operator-placeable flat board on table',
                                       center_base_m=center.tolist(), patterned_size_m=.087),
        table_assumption=dict(center_m=[center[0]+.10, center[1], center[2]-.022],
                              half_extents_m=[.25, .32, .02], planning_only=True,
                              note='Nominal table. Review actual tabletop and surroundings before any physical run.'),
        gripper=dict(side='left', opening_m=.08, measured=False, model='Approximate parallel jaw slides'),
        source_sha256={str(p.relative_to(ROOT)):sha(p) for p in sources},
        simulation_only=True, hardware_execution_authorized=False)
    # This design task keeps lift/right fixed even during the default approach.
    # A physical measured-origin/default path must be independently checked later.
    plan['measured_start_q'] = plan['initial_q'].copy()
    plan['start_provenance'] = 'Nominal default at 300 mm; no hardware state sampled'
    approach=np.asarray(plan['initial_q']).copy()
    approach[5]=.5
    approach[7]=1.5
    plan['approach_waypoints']=[approach.tolist()]
    return plan


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT/'runs/left-wrist-calibration-planning')
    parser.add_argument('--plan', type=Path, default=PLAN)
    parser.add_argument('--intrinsics', type=Path, help='ready.json from the left camera-only worker')
    parser.add_argument('--regenerate', action='store_true', help='Search for a new route instead of rechecking existing checked joint poses')
    parser.add_argument('--board-center', type=float, nargs=3, default=[.68, .12, .72], metavar=('X', 'Y', 'Z'))
    args = parser.parse_args()
    if args.plan.exists() and not args.regenerate:
        revalidate(args.output,args.plan,configuration(args.intrinsics,args.board_center))
    else:
        run(args.output, args.plan, lambda: configuration(args.intrinsics, args.board_center))


def revalidate(output,plan_path,current):
    """Recheck every existing joint pose/path with a changed SDK profile.

    Changing focal lengths need not throw away a known connected joint route.
    New geometry/FOV still must pass actual clearance and rendered detections.
    """
    import cv2,mujoco
    from policy_sim.calibration.charuco import detect
    previous=load_plan(plan_path)
    np.testing.assert_array_equal(previous['initial_q'],current['initial_q'])
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    checker=LocalWristChecker(output/'geometry',current);study=checker.study
    current['poses']=[];segments=[]
    origin=np.asarray(current['initial_q']);minimum=.1;count=0
    approach_segments=[]
    for waypoint in current['approach_waypoints']:
        waypoint=np.asarray(waypoint);gap,n=checker.edge(origin,waypoint,current['bounds'])
        approach_segments.append(dict(destination='approach',minimum_clearance_m=gap,samples=n))
        origin=waypoint;minimum=min(minimum,gap);count+=n
    with mujoco.Renderer(study.m,height=800,width=1400) as renderer:
        for row in previous['poses']:
            q=np.asarray(row['q']);gap,n=checker.edge(origin,q,current['bounds'])
            view=study.visibility(q);image=study.render(q,renderer)
            result,annotated=detect(image,study.boardcfg,study.k,np.zeros(5))
            if view['margin_px']<35 or view['facing_cosine']<.7 or not result['accepted'] or result['corners']<12:
                raise ValueError(f'Pose {row["pose_id"]} no longer passes the measured-profile view checks; regenerate route')
            current['poses'].append(dict(pose_id=row['pose_id'],q=q.tolist(),nominal_view=view,nominal_detection=result))
            segments.append(dict(destination_pose=row['pose_id'],minimum_clearance_m=gap,samples=n))
            minimum=min(minimum,gap);count+=n;origin=q
            cv2.imwrite(str(output/f'pose-{row["pose_id"]:02d}-wrist.png'),annotated)
    current['default_path']=checker.plan(np.asarray(current['initial_q']),-2,current['bounds'],np.asarray(current['initial_q']))
    current['return_path']=checker.plan(origin,-1,current['bounds'],np.asarray(current['initial_q']))
    minimum=min(minimum,current['return_path']['minimum_sampled_clearance_m'])
    count+=current['return_path']['path_samples']+current['default_path']['path_samples']
    current['scene_file']=str(study.scene);current['segments']=segments;current['approach_segments']=approach_segments
    current['mount_side_validation']=previous['mount_side_validation']
    current['simulation_validation']=dict(previous['simulation_validation'],
        path_sample_count=count,minimum_path_clearance_m=minimum,collision_pair_count=len(study.pairs),
        minimum_detection_corners=min(row['nominal_detection']['corners'] for row in current['poses']),
        intrinsics_measured=True,method='Full checked joint route revalidated using active SDK K and fresh rendered detections',
        source_previous_plan_sha256=sha(plan_path))
    current['geometry_payload_sha256']=geometry_payload_hash(current)
    plan_path.write_text(json.dumps(current,indent=2)+'\n')
    (output/'summary.json').write_text(json.dumps(current['simulation_validation'],indent=2)+'\n')
    print('PASS',json.dumps(current['simulation_validation']))


if __name__ == '__main__':
    main()
