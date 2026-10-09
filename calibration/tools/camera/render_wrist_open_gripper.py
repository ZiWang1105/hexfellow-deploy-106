#!/usr/bin/env python3
"""Offline open-GP80 visibility review; never opens a physical device.

Nominal previews use the validated plan's joint targets; optional arm/lift replay
uses recorded simulation state.q from a matching plan. Jaw positions are an
explicit 80-mm-open visualization override, not recorded encoder measurements
or a new controller rehearsal. Upstream assets and original evidence stay intact.
"""
import argparse
import bisect
import hashlib
import json
import math
from pathlib import Path
import sys
import xml.etree.ElementTree as ET

import cv2
import mujoco
import numpy as np
import trimesh

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src'))
from policy_sim.calibration.charuco import detect
from wrist_calibration_geometry import DEFAULT_PLAN, installed_camera_transforms, load_plan

JOINTS = ['mobile_base_x', 'mobile_base_y', 'mobile_base_yaw', 'maver_x4_asm_URDF_joint_1']
JOINTS += [f'maver_x4_asm_7_urdf_SLDASM_joint_{i}' for i in range(1, 8)]
JOINTS += [f'maver_x4_asm_7_urdf_SLDASM2_joint_{i}' for i in range(1, 8)]
OPENING_M = .080


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def prepare_scene(source, output, side='right'):
    """Separate the three disconnected vendor CAD components; add two slides.

    Same +/-Y, 40-mm-per-jaw abstraction as the parent's derived GP80 assembly.
    This tool only renders forward kinematics; it performs no dynamics/control.
    """
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    tree = ET.parse(source)
    root = tree.getroot()
    asset = root.find('asset')
    mesh_path = Path(asset.find("mesh[@name='base_link']").get('file'))
    if not mesh_path.is_absolute():
        mesh_path = Path(source).parent / mesh_path
    components = {}
    for part in trimesh.load(mesh_path, force='mesh').split(only_watertight=False):
        name = 'palm' if part.bounds[1, 2] < .08 else ('positive' if part.center_mass[1] > 0 else 'negative')
        if name in components:
            raise ValueError('Unexpected GP80 CAD connected components')
        components[name] = part
    if set(components) != {'palm', 'positive', 'negative'}:
        raise ValueError('Expected GP80 palm and two jaws')
    files = {}
    for name, mesh in components.items():
        path = output / f'gp80_{name}.stl'
        mesh.export(path)
        ET.SubElement(asset, 'mesh', name=f'calibration_gp80_{name}', file=str(path.resolve()))
        files[name] = {'path': str(path), 'sha256': sha(path)}
    if side not in ('left', 'right'):
        raise ValueError('Expected left or right gripper')
    palm = root.find(f".//body[@name='{'robot_base' if side == 'left' else 'robot2_base'}']")
    original = [g for g in palm.findall('geom') if g.get('mesh') == 'base_link']
    if len(original) != 1:
        raise ValueError('Expected one rigid right GP80 mesh')
    properties = dict(original[0].attrib)
    palm.remove(original[0])
    properties.update(name=f'calibration_{side}_palm', mesh='calibration_gp80_palm')
    ET.SubElement(palm, 'geom', **properties)
    palm.find('inertial').set('mass', '.41')
    for name, sign in [('positive', 1), ('negative', -1)]:
        body = ET.SubElement(palm, 'body', name=f'calibration_{side}_jaw_{name}', gravcomp='1')
        center = components[name].center_mass
        ET.SubElement(body, 'inertial', pos=' '.join(map(str, center)), mass='.06', diaginertia='.00003 .00003 .000015')
        ET.SubElement(body, 'joint', name=f'calibration_{side}_jaw_{name}', type='slide',
                      axis=f'0 {sign} 0', range=f'0 {OPENING_M/2}', damping='2')
        properties.update(name=f'calibration_{side}_jaw_{name}', mesh=f'calibration_gp80_{name}')
        ET.SubElement(body, 'geom', **properties)
    # The right jaw joints follow the existing 18 joints in this scene. Preserve
    # the original keyframe values and actuator controls, adding only jaw zeros.
    for key in root.findall('keyframe/key'):
        for name in ('qpos', 'qvel'):
            if name in key.attrib:
                key.set(name, key.get(name) + ' 0 0')
    scene = output / 'scene-open-gripper.xml'
    tree.write(scene)
    return scene, {'source_scene': str(source), 'source_scene_sha256': sha(source),
                   'source_gp80_mesh': str(mesh_path), 'source_gp80_mesh_sha256': sha(mesh_path),
                   'derived_meshes': files, 'nominal_opening_m': OPENING_M,
                   'jaw_translation_per_side_m': OPENING_M/2,
                   'motion_model': 'Approximate parallel-jaw +/-Y slide; same abstraction as parent GP80 assembly',
                   'geometry_retained': 'Palm, both jaws, camera mount and housing all retained',
                   'physical_jaw_travel_measured': False, 'dynamics_or_motion_revalidated': False}


class Views:
    def __init__(self, scene, k):
        self.m = mujoco.MjModel.from_xml_path(str(scene))
        self.d = mujoco.MjData(self.m)
        self.indices = [int(self.m.joint(name).qposadr[0]) for name in JOINTS]
        self.jaws = [int(self.m.joint(f'calibration_right_jaw_{name}').qposadr[0]) for name in ('positive', 'negative')]
        k = np.asarray(k)
        camera = self.m.camera('wrist_calibration_view')
        camera.fovy[0] = np.rad2deg(2*np.arctan(800/(2*k[1, 1])))
        if hasattr(self.m, 'cam_sensorsize'):
            self.m.cam_sensorsize[camera.id] = 0.
        self.renderer = mujoco.Renderer(self.m, height=800, width=1400)
        self.opt = mujoco.MjvOption()
        self.opt.geomgroup[3:6] = 1
        self.third = mujoco.MjvCamera()
        self.third.type = mujoco.mjtCamera.mjCAMERA_FREE
        self.third.lookat[:] = [.45, -.05, .8]
        self.third.distance, self.third.azimuth, self.third.elevation = 2.25, 145, -22
        u, v = np.meshgrid(np.arange(1280, dtype=np.float32), np.arange(720, dtype=np.float32))
        self.mapx = ((1400-1)/2+k[1, 1]/k[0, 0]*(u-k[0, 2])).astype(np.float32)
        self.mapy = ((800-1)/2+v-k[1, 2]).astype(np.float32)
        self.pupil = [round(float(k[0, 2])), round(float(k[1, 2]))]
        self.gripper_ids = [self.m.geom(name).id for name in ('calibration_right_palm',
                             'calibration_right_jaw_positive', 'calibration_right_jaw_negative')]

    def set(self, q, opened):
        self.d.qpos[self.indices] = np.asarray(q)
        self.d.qpos[self.jaws] = OPENING_M/2 if opened else 0.
        mujoco.mj_forward(self.m, self.d)
        np.testing.assert_array_equal(self.d.qpos[self.indices], q)

    def render(self, third=False):
        self.opt.geomgroup[5] = int(third)
        self.renderer.update_scene(self.d, camera=self.third if third else 'wrist_calibration_view', scene_option=self.opt)
        self.renderer.scene.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = False
        rgb = self.renderer.render()
        if not third:
            rgb = cv2.remap(rgb, self.mapx, self.mapy, cv2.INTER_LINEAR)
        return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

    def occlusion(self):
        self.renderer.enable_segmentation_rendering()
        try:
            self.opt.geomgroup[5] = 0
            self.renderer.update_scene(self.d, camera='wrist_calibration_view', scene_option=self.opt)
            seg = cv2.remap(self.renderer.render(), self.mapx, self.mapy, cv2.INTER_NEAREST)
            mask = np.isin(seg[:, :, 0], self.gripper_ids) & (seg[:, :, 1] == int(mujoco.mjtObj.mjOBJ_GEOM))
            u, v = self.pupil
            return {'gripper_full_image_fraction': float(mask.mean()),
                    'gripper_center_region_fraction': float(mask[v-80:v+80, u-100:u+100].mean()),
                    'center_ray_occluded_by_gripper': bool(mask[v, u])}
        finally:
            self.renderer.disable_segmentation_rendering()

    def close(self):
        self.renderer.close()

    def verify_installation(self, plan):
        self.set(plan['initial_q'], False)
        def transform(name):
            t = np.eye(4)
            body = self.d.body(name)
            t[:3, :3] = body.xmat.reshape(3, 3)
            t[:3, 3] = body.xpos
            return t
        flange = transform('robot2_base')
        optical = transform(plan['camera_frame'])
        housing = transform('d435i_camera_housing')
        installation, expected_housing, expected_optical = installed_camera_transforms()
        np.testing.assert_allclose(np.linalg.inv(flange) @ optical, expected_optical, atol=1e-12, rtol=0)
        np.testing.assert_allclose(np.linalg.inv(flange) @ housing, expected_housing, atol=1e-12, rtol=0)
        mount = self.m.geom('d435i_fixed33_mount')
        rotation = np.empty(9)
        mesh_rotation = np.empty(9)
        mujoco.mju_quat2Mat(rotation, mount.quat)
        # MuJoCo absorbs mesh principal-axis alignment into the compiled geom.
        mujoco.mju_quat2Mat(mesh_rotation, self.m.mesh_quat[self.m.geom_dataid[mount.id]])
        np.testing.assert_allclose(rotation.reshape(3, 3) @ mesh_rotation.reshape(3, 3).T,
                                   np.asarray(installation['T_parent_from_design'])[:3, :3], atol=1e-12, rtol=0)
        if optical[1, 3] >= flange[1, 3]:
            raise ValueError('Rendered camera is not outside the right wrist at default')
        return dict(passed=True, optical_minus_wrist_base_y_m=float(optical[1, 3]-flange[1, 3]),
                    T_base_from_optical_at_default=optical.tolist(), T_base_from_wrist_at_default=flange.tolist())

    def default_preview(self, plan, output):
        """Show the actual modeled mount at default before the calibration targets."""
        self.set(plan['initial_q'], True)
        whole = self.render(third=True)
        lookat, distance, azimuth, elevation = self.third.lookat.copy(), self.third.distance, self.third.azimuth, self.third.elevation
        self.third.lookat[:] = self.d.body('robot2_base').xpos + np.array([0, -.035, 0])
        self.third.distance, self.third.azimuth, self.third.elevation = .60, 90, -16
        closeup = self.render(third=True)
        image = np.full((800, 1600, 3), (17, 21, 27), np.uint8)
        label(image, 'Corrected installation: camera OUTSIDE the right arm at default', (24, 45), .90)
        label(image, 'Same fixed 33-degree mount; complete CAD assembly rotated 180 degrees around flange Z', (24, 86), .67)
        image[125:625, 20:790] = panel(whole, 770, 500)
        image[125:625, 810:1580] = panel(closeup, 770, 500)
        label(image, 'Default pose / whole robot', (24, 660), .75)
        label(image, 'Robot right side / wrist close-up', (814, 660), .75)
        label(image, 'Camera pupil is 57.5 mm outward from the right wrist in this nominal model.', (24, 716), .72)
        label(image, 'Open-jaw visualization only. No physical motion or measured extrinsic calibration.', (24, 762), .65)
        path = Path(output)/'default-outside-mount.png'
        cv2.imwrite(str(path), image)
        self.third.lookat[:], self.third.distance, self.third.azimuth, self.third.elevation = lookat, distance, azimuth, elevation
        return str(path)


def label(image, text, at, size=.7, color=(230, 230, 230)):
    cv2.putText(image, text, at, cv2.FONT_HERSHEY_SIMPLEX, size, color, 1, cv2.LINE_AA)


def panel(image, width, height):
    scale = min(width/image.shape[1], height/image.shape[0])
    resized = cv2.resize(image, (round(image.shape[1]*scale), round(image.shape[0]*scale)), interpolation=cv2.INTER_AREA)
    result = np.full((height, width, 3), 28, np.uint8)
    y, x = (height-resized.shape[0])//2, (width-resized.shape[1])//2
    result[y:y+resized.shape[0], x:x+resized.shape[1]] = resized
    return result


def compose(robot, wrist, title, subtitle, status):
    image = np.full((720, 1280, 3), (17, 21, 27), np.uint8)
    label(image, title, (24, 43), .90)
    label(image, subtitle, (24, 78), .60)
    label(image, 'Whole robot / third-person', (24, 123), .65)
    label(image, 'Right D435i / wrist-camera view', (654, 123), .65)
    image[143:493, 16:636] = panel(robot, 620, 350)
    image[143:493, 644:1264] = panel(wrist, 620, 350)
    label(image, status, (24, 543), .68, (118, 224, 214))
    label(image, 'Right gripper fully open: nominal 80 mm | fixed 33-degree camera mount', (24, 590), .63)
    label(image, 'Jaw opening is an explicit simulation override. No physical robot movement.', (24, 634), .59)
    label(image, 'Offline visibility rendering; controller rehearsal and source provenance are reported separately.', (24, 676), .56)
    return image


class Video:
    def __init__(self, path):
        import av
        self.av, self.path = av, path
        self.container = av.open(str(path), 'w')
        self.stream = self.container.add_stream('libx264', rate=30)
        self.stream.width, self.stream.height, self.stream.pix_fmt = 1280, 720, 'yuv420p'
        self.stream.options = {'crf': '21', 'preset': 'fast'}
        self.count = 0

    def add(self, image):
        frame = self.av.VideoFrame.from_ndarray(image, format='bgr24')
        frame.pts = self.count
        for packet in self.stream.encode(frame):
            self.container.mux(packet)
        self.count += 1

    def close(self):
        for packet in self.stream.encode():
            self.container.mux(packet)
        self.container.close()
        with self.av.open(str(self.path)) as video:
            stream = video.streams.video[0]
            metadata = dict(codec=stream.codec_context.name, width=stream.width, height=stream.height,
                            fps=str(stream.average_rate), duration_s=video.duration/self.av.time_base)
            decoded = sum(1 for _ in video.decode(stream))
        if decoded != self.count:
            raise RuntimeError('MP4 frame count mismatch')
        metadata.update(path=str(self.path), frames=decoded, bytes=self.path.stat().st_size,
                        sha256=sha(self.path), full_decode_passed=True)
        return metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--sequence-run', type=Path, default=ROOT/'runs/right-wrist-outside-e2e')
    parser.add_argument('--source-scene', type=Path, help='Scene built from the same installed-camera plan')
    parser.add_argument('--poses-only', action='store_true', help='Render nominal poses without a controller replay')
    parser.add_argument('--plan', type=Path, default=DEFAULT_PLAN)
    parser.add_argument('--output', type=Path, default=ROOT/'runs/right-wrist-open-gripper')
    args = parser.parse_args()
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    plan = load_plan(args.plan)
    if not args.poses_only:
        recorded_plan = json.loads((args.sequence_run/'wrist-plan.json').read_text())
        if recorded_plan.get('geometry_payload_sha256') != plan['geometry_payload_sha256']:
            raise ValueError('Recorded controller sequence belongs to a different mount/pose plan')
    scene, report = prepare_scene(args.source_scene or args.sequence_run/'geometry/scene.xml', out/'geometry')
    views = Views(scene, plan['K'])
    cfg = json.loads((ROOT/'config/camera/charuco_87mm.json').read_text())
    report.update(scope='Offline right-wrist open-gripper visualization and board detection',
                  hardware_commands_sent=0, physical_devices_opened=False,
                  plan_path=str(args.plan), plan_sha256=sha(args.plan),
                  source_geometry_payload_sha256=plan['geometry_payload_sha256'],
                  nominal_pose_results=[], videos=[])
    try:
        report['installation_validation'] = views.verify_installation(plan)
        report['default_mount_preview'] = views.default_preview(plan, out)
        video = Video(out/'right-d435i-open-gripper-30-poses.mp4')
        for row in plan['poses']:
            pose = row['pose_id']
            views.set(row['q'], False)
            closed = views.render()
            closed_occ = views.occlusion()
            views.set(row['q'], True)
            opened = views.render()
            open_occ = views.occlusion()
            result, annotated = detect(opened, cfg, np.asarray(plan['K']), np.zeros(5))
            third = views.render(third=True)
            cv2.imwrite(str(out/f'pose-{pose:02d}-wrist-open.png'), annotated)
            cv2.imwrite(str(out/f'pose-{pose:02d}-robot-open.png'), third)
            report['nominal_pose_results'].append({'pose_id': pose, 'closed': closed_occ,
                'open': open_occ, 'detection_accepted': result['accepted'], 'corners': result['corners'],
                'reprojection_rms_px': result.get('reprojection_rms_px')})
            frame = compose(third, annotated, 'Right wrist calibration - gripper open',
                'All 30 nominal target poses | 2 seconds per pose',
                f'Pose {pose:02d} / 30 | ChArUco corners: {result["corners"]} | detection: {"PASS" if result["accepted"] else "FAIL"}')
            for _ in range(60):
                video.add(frame)
            if pose in (1, 15, 30):
                cv2.imwrite(str(out/f'preview-frame-pose-{pose:02d}.png'), frame)
            if pose == 1:
                comparison = np.full((810, 2560, 3), 25, np.uint8)
                label(comparison, 'Same pose and camera: closed gripper', (24, 42), 1.0)
                label(comparison, 'Same pose and camera: fully open gripper (80 mm nominal)', (1304, 42), .90)
                comparison[65:785, :1280] = closed
                comparison[65:785, 1280:] = opened
                cv2.imwrite(str(out/'closed-vs-open.png'), comparison)
            print(f'Open preview {pose}/30: {result["corners"]} corners; center occluded={open_occ["center_ray_occluded_by_gripper"]}', flush=True)
        report['videos'].append(video.close())

        results = report['nominal_pose_results']
        report.update(accepted_nominal_views=sum(row['detection_accepted'] for row in results),
                      minimum_charuco_corners=min(row['corners'] for row in results),
                      closed_gripper_fraction=float(np.mean([row['closed']['gripper_full_image_fraction'] for row in results])),
                      open_gripper_fraction=float(np.mean([row['open']['gripper_full_image_fraction'] for row in results])),
                      open_center_ray_occluded_count=sum(row['open']['center_ray_occluded_by_gripper'] for row in results),
                      controller_replay_performed=not args.poses_only,
                      replay_jaw_source='Explicit 40 mm per jaw override; not measured gripper feedback',
                      gripper_state_overridden=True)
        if args.poses_only:
            (out/'visibility-report.json').write_text(json.dumps(report, indent=2)+'\n')
            print(json.dumps({key: report[key] for key in ('accepted_nominal_views', 'minimum_charuco_corners',
                'closed_gripper_fraction', 'open_gripper_fraction', 'open_center_ray_occluded_count')}), flush=True)
            return

        trace = args.sequence_run/'session/trace.jsonl'
        rows = []
        for line in trace.open():
            row = json.loads(line)
            if row.get('kind') == 'cycle':
                rows.append({'at': row['at'], 'q': row['state']['q'], 'tick': row['tick'], 'state': row.get('control_state')})
        times = np.asarray([row['at'] for row in rows])
        if not np.all(np.diff(times) > 0):
            raise ValueError('Non-monotonic trace')
        events = [json.loads(line) for line in (args.sequence_run/'operator-events.jsonl').read_text().splitlines()]
        next_times = [row['at'] for row in events if row['action'] == 'next']
        return_at = next(row['at'] for row in events if row['action'] == 'return')
        video = Video(out/'right-d435i-open-gripper-arm-replay.mp4')
        count = math.ceil((times[-1]-times[0])/10*30)
        for i in range(count):
            at = times[0]+i/30*10
            index = min(int(np.searchsorted(times, at)), len(rows)-1)
            if index and abs(times[index-1]-at) < abs(times[index]-at):
                index -= 1
            row = rows[index]
            views.set(row['q'], True)
            pose = bisect.bisect_right(next_times, row['at'])
            stage = 'Return to default' if row['at'] >= return_at else (f'Pose {pose:02d}/30' if pose else 'Prepare default')
            frame = compose(views.render(third=True), views.render(), 'Recorded arm/lift replay - gripper open',
                f'Recorded simulation state.q + open-jaw override | 10x | elapsed {row["at"]-times[0]:.1f} s',
                f'{stage} | {row["state"]}')
            video.add(frame)
            if i in (0, count//3, count*2//3, count-1):
                cv2.imwrite(str(out/f'replay-frame-{i:04d}.png'), frame)
            if i % 200 == 0:
                print(f'Open replay: {i}/{count} frames', flush=True)
        report['videos'].append(video.close())
        results = report['nominal_pose_results']
        report.update(accepted_nominal_views=sum(row['detection_accepted'] for row in results),
                      minimum_charuco_corners=min(row['corners'] for row in results),
                      closed_gripper_fraction=float(np.mean([row['closed']['gripper_full_image_fraction'] for row in results])),
                      open_gripper_fraction=float(np.mean([row['open']['gripper_full_image_fraction'] for row in results])),
                      open_center_ray_occluded_count=sum(row['open']['center_ray_occluded_by_gripper'] for row in results),
                      source_trace=str(trace), source_trace_sha256=sha(trace),
                      replay_arm_lift_source='state.q (measured simulation feedback); commanded positions not used',
                      replay_jaw_source='Explicit 40 mm per jaw override; not measured gripper feedback',
                      gripper_state_overridden=True)
    finally:
        views.close()
    (out/'visibility-report.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps({key: report[key] for key in ('accepted_nominal_views', 'minimum_charuco_corners',
        'closed_gripper_fraction', 'open_gripper_fraction', 'open_center_ray_occluded_count')}), flush=True)


if __name__ == '__main__':
    main()
