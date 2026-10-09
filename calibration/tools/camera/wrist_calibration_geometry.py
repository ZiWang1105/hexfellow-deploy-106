#!/usr/bin/env python3
"""Offline/right-wrist camera planning and isolated geometry checks.

No robot SDK is imported here. Nominal fixed-33 CAD optics and a rough board
placement are planning assumptions; neither is a measured camera extrinsic.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from calibration_outcomes import InvalidCalibrationPose

DEFAULT_PLAN = ROOT / "config/camera/realsense_d435i_right_fixed33_outside_plan.json"
INSTALLATION = ROOT / "config/camera/right_fixed33_installation.json"
DESIGN = ROOT / "assets_local/right_fixed33"
PREVIEW = ROOT / "artifacts/reference/wrist-planning"
URDF = ROOT / "urdf/maver_x4_gp80_zed_40deg.urdf"
from calibration_paths import desktop_robot_paths
# Resolved lazily by the planner; offline loading needs no machine-local mapping.
def mapping_path():
    return desktop_robot_paths()[0]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def geometry_payload_hash(plan):
    keys=("initial_q","measured_start_q","parent_frame","camera_frame","serial","width","height","K",
          "T_parent_from_optical_nominal","T_base_from_board_nominal","bounds","table_assumption","poses","mount_installation")
    payload={k:plan[k] for k in keys}
    if plan.get('capture_side')=='left':
        for k in ('capture_side','mount_mesh','camera_mesh','T_parent_from_housing_nominal','approach_waypoints','gripper','intrinsics_provenance'):
            payload[k]=plan[k]
    return hashlib.sha256(json.dumps(payload,sort_keys=True,separators=(",",":"),allow_nan=False).encode()).hexdigest()


def installed_camera_transforms(side='right'):
    """One installation transform applies to the CAD, housing and optical pupil."""
    if side not in ('left','right'):
        raise ValueError('Expected left or right mount')
    installation_path=ROOT/'config/camera/left_fixed32_installation.json' if side=='left' else INSTALLATION
    design_dir=ROOT/'assets_local/left_fixed32' if side=='left' else DESIGN
    installation = json.loads(installation_path.read_text())
    transform = np.asarray(installation['T_parent_from_design'], float)
    if (transform.shape != (4, 4) or not np.isfinite(transform).all()
            or not np.allclose(transform[3], [0, 0, 0, 1])
            or not np.allclose(transform[:3, :3].T @ transform[:3, :3], np.eye(3))
            or not np.isclose(np.linalg.det(transform[:3, :3]), 1)):
        raise ValueError('Invalid right-wrist installation transform')
    data = json.loads((design_dir / 'design_and_validation.json').read_text())
    body = np.asarray(data['camera_transform_mm'], float)
    body[:3, 3] *= .001
    local = np.eye(4)
    local[:3, :3] = [[0, -1, 0], [1, 0, 0], [0, 0, 1]]
    local[:3, 3] = [0, -.009, .0192] if side=='left' else [0, -.0325, .0207547]
    optical = optical_roll(transform @ body @ local, installation.get('optical_axis_roll_deg', 0.))
    return installation, transform @ body, optical


def optical_roll(transform, degrees):
    """Rotate sensor X/Y about its own forward axis, retaining the pupil.

    This sensor orientation is independent of flange-side installation. Apply
    it before rendering/projection/PnP simulation, never as a display flip.
    Housing geometry and the nominal pupil position are unchanged assumptions.
    """
    result = np.asarray(transform, float).copy()
    if result.shape != (4, 4) or not np.isfinite(result).all() or not np.isfinite(degrees):
        raise ValueError('Invalid optical-axis roll')
    a = math.radians(float(degrees))
    rotation = np.array([[math.cos(a), -math.sin(a), 0.],
                         [math.sin(a), math.cos(a), 0.], [0., 0., 1.]])
    result[:3, :3] = result[:3, :3] @ rotation
    return result


def load_plan(path):
    path = Path(path)
    plan = json.loads(path.read_text())
    side=plan.get('capture_side')
    installation = plan.get('mount_installation')
    expected_side='outside_left_arm' if side=='left' else 'outside_right_arm'
    if not installation or installation.get('mount_side_at_default') != expected_side:
        raise ValueError('The old inner-side wrist plan is superseded; use the regenerated outside-right-arm plan')
    current, _, optical = installed_camera_transforms('left' if side=='left' else 'right')
    if (installation != current
            or not np.allclose(plan['T_parent_from_optical_nominal'], optical, atol=1e-12, rtol=0)):
        raise ValueError('Wrist installation or optical transform changed; regenerate the outside mount plan')
    if plan.get("schema_version") != 1 or side not in ('left','right'):
        raise ValueError("Expected left/right eye-in-hand plan version 1")
    if side=='left' and (plan.get('parent_frame')!='maver_x4_asm_mount1' or plan.get('serial')!='230422271628'):
        raise ValueError('Left plan must identify mount1 and D405 serial 230422271628')
    if plan.get("pose_count") != 30 or len(plan.get("poses", [])) != 30:
        raise ValueError("Expected 30 validated wrist poses")
    for i, row in enumerate(plan["poses"], 1):
        q = np.asarray(row["q"], float)
        if row["pose_id"] != i or q.shape != (18,) or not np.isfinite(q).all():
            raise ValueError("Invalid wrist pose order or joint coordinates")
    for source, digest in plan.get("source_sha256", {}).items():
        source_path=Path(source)
        if not source_path.is_absolute():source_path=ROOT/source_path
        if sha(source_path) != digest:
            raise ValueError("Wrist planning source changed: " + source)
    if not plan.get("simulation_validation", {}).get("passed", False):
        raise ValueError("Wrist plan has not passed simulation geometry/vision checks")
    if plan.get("geometry_payload_sha256") != geometry_payload_hash(plan):
        raise ValueError("Wrist geometry or simulated targets changed; regenerate the plan")
    return plan


class WristStudy:
    def __init__(self, plan, output):
        import cv2
        import mujoco
        import pinocchio as pin
        import yaml
        from scipy.spatial.transform import Rotation
        from policy_sim.control.robot import CONTROLLED_JOINTS, RIGHT_ARM_JOINTS
        from validate_calibration_poses import Study, T, fmt, quat

        self.out = Path(output)
        self.out.mkdir(parents=True, exist_ok=True)
        self.plan = plan
        self.installation = np.asarray(plan['mount_installation']['T_parent_from_design'], float)
        self.k = np.asarray(plan["K"], float)
        self.width, self.height = plan["width"], plan["height"]
        self.boardcfg = json.loads((ROOT / "config/camera/charuco_87mm.json").read_text())
        self.x = np.asarray(plan["T_parent_from_optical_nominal"], float)
        self.y = np.asarray(plan["T_base_from_board_nominal"], float)
        self.q18 = np.asarray(plan["initial_q"], float)
        self.rng = np.random.default_rng(330087)
        # Reuse the established model conversion and unchanged arm-seam policy.
        original = Study(dict(q=self.q18.tolist()), self.out / "original", include_left=True)
        self.pm, self.pd, self.pi, self.ai = original.pm, original.pd, original.pi, original.ai
        self.q0 = original.q0.copy()
        self.hand = self.pm.getFrameId(plan["parent_frame"])
        self.side = plan.get('capture_side', 'right')
        self.arm_slice = slice(4, 11) if self.side == 'left' else slice(11, 18)
        self.lower, self.upper = np.asarray(plan["bounds"][0])[self.arm_slice] + .15, np.asarray(plan["bounds"][1])[self.arm_slice] - .15
        tree = ET.parse(original.scene)
        root = tree.getroot()
        for body in root.findall(".//body"):
            for child in list(body):
                if child.tag == "body" and child.get("name") == "calibration_board":
                    body.remove(child)
        world = root.find("worldbody")
        board = ET.SubElement(world, "body", name="stationary_calibration_board", pos=fmt(self.y[:3,3]), quat=quat(self.y[:3,:3]))
        ET.SubElement(board, "geom", name="board_backing", type="box", pos=".0435 .0435 .002", size=".05 .05 .002", rgba="1 1 1 1", group="3", contype="0", conaffinity="0")
        dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
        marker_id = 0
        for yy in range(5):
            for xx in range(5):
                if (xx+yy)%2 == 0:
                    ET.SubElement(board,"geom",type="box",pos=fmt([(xx+.5)*.0174,(yy+.5)*.0174,-.0001]),size=".0087 .0087 .0001",rgba="0 0 0 1",group="4",contype="0",conaffinity="0")
                else:
                    marker = cv2.aruco.generateImageMarker(dictionary, marker_id, 6)
                    marker_id += 1
                    for y,x in zip(*np.where(marker==0)):
                        p=[(xx+.5)*.0174+((x+.5)/6-.5)*.0116,(yy+.5)*.0174+((y+.5)/6-.5)*.0116,-.0001]
                        ET.SubElement(board,"geom",type="box",pos=fmt(p),size=fmt([.0116/12,.0116/12,.0001]),rgba="0 0 0 1",group="4",contype="0",conaffinity="0")
        table = plan["table_assumption"]
        ET.SubElement(world,"geom",name="rough_table",type="box",pos=fmt(table["center_m"]),size=fmt(table["half_extents_m"]),rgba=".55 .40 .25 1",group="3",contype="0",conaffinity="0")
        side_number = 1 if self.side == 'left' else 2
        flange_body = 'robot_base' if self.side == 'left' else 'robot2_base'
        mount = root.find(f".//body[@name='{flange_body}']")
        if mount is None:
            raise ValueError("GP80 wrist flange is absent from robot scene")
        asset = root.find("asset")
        mesh = ROOT / plan['mount_mesh'] if self.side == 'left' else DESIGN / "assembly_mm/01_onepiece_ring_camera_33deg.stl"
        mount_geom_name='d405_fixed32_mount' if self.side=='left' else 'd435i_fixed33_mount'
        mount_mesh_name='d405_fixed32_visual' if self.side=='left' else 'd435i_fixed33_visual'
        ET.SubElement(asset,"mesh",name=mount_mesh_name,file=str(mesh),scale=".001 .001 .001")
        ET.SubElement(mount,"geom",name=mount_geom_name,type="mesh",mesh=mount_mesh_name,
                      pos=fmt(self.installation[:3,3]),quat=quat(self.installation[:3,:3]),
                      rgba=".08 .12 .15 1",group="3",contype="0",conaffinity="0")
        design_dir = ROOT / 'assets_local/left_fixed32' if self.side == 'left' else DESIGN
        data = json.loads((design_dir / "design_and_validation.json").read_text())
        body_t = np.asarray(data["camera_transform_mm"],float);body_t[:3,3]*=.001
        body_t = self.installation @ body_t
        camera = ET.SubElement(mount,"body",name="d435i_camera_housing",pos=fmt(body_t[:3,3]),quat=quat(body_t[:3,:3]))
        if self.side == 'left':
            camera.set('name', 'd405_camera_housing')
            ET.SubElement(asset, 'mesh', name='d405_body_visual', file=str(ROOT / plan['camera_mesh']), scale='.001 .001 .001')
            ET.SubElement(camera, 'geom', name='d405_body_visual', type='mesh', mesh='d405_body_visual', rgba='.35 .4 .43 1', group='4', contype='0', conaffinity='0')
            ET.SubElement(camera, 'geom', name='d405_camera_box', type='box', pos='0 0 .011575', size='.021075 .021075 .011575', rgba='.35 .4 .43 0', group='5', contype='0', conaffinity='0')
        else:
            ET.SubElement(camera,"geom",name="d435i_camera_box",type="box",pos="0 0 .0125275",size=".0125 .045 .0125275",rgba=".35 .4 .43 1",group="5",contype="0",conaffinity="0")
        optical = ET.SubElement(mount,"body",name=plan["camera_frame"],pos=fmt(self.x[:3,3]),quat=quat(self.x[:3,:3]))
        ET.SubElement(optical,"camera",name="wrist_calibration_view",quat="0 1 0 0",fovy=str(math.degrees(2*math.atan(800/(2*self.k[1,1])))))
        root.find("visual/global").set("offwidth","1400")
        root.find("visual/global").set("offheight","800")
        self.scene = self.out / "scene.xml"
        tree.write(self.scene)
        if self.side == 'left':
            from render_wrist_open_gripper import prepare_scene
            self.scene, self.jaw_geometry = prepare_scene(self.scene, self.out / 'open-gripper', side='left')
        self.m = mujoco.MjModel.from_xml_path(str(self.scene))
        self.d = mujoco.MjData(self.m)
        self.mi = [self.m.joint(n).qposadr[0] for n in CONTROLLED_JOINTS]
        self.geom_board = self.m.geom("board_backing").id
        cfg = yaml.safe_load((ROOT / "config/control/self_collision_proxies.v2.yaml").read_text())
        frames = {r["name"]: r["frame"] for r in cfg["links"]}
        reasons={"approved arm near-neighbor seam exclusion","fixed gripper housing and its two-joint-neighbor forearm","gripper and its directly connected wrist housings"}
        seams={frozenset([frames[r["first"]],frames[r["second"]]]) for r in cfg["excluded_pairs"] if r["reason"] in reasons}
        jids={self.m.joint(n).id for n in CONTROLLED_JOINTS}
        def moving(b):
            while b:
                if self.m.body_jntnum[b] and int(self.m.body_jntadr[b]) in jids:return True
                b=int(self.m.body_parentid[b])
            return False
        geoms=[i for i in range(self.m.ngeom) if self.m.geom_group[i] in (0,1,3,5)]
        self.pairs=[]
        self.exclusions=[]
        for ii,g1 in enumerate(geoms):
            for g2 in geoms[ii+1:]:
                b1,b2=map(int,self.m.geom_bodyid[[g1,g2]])
                w1,w2=map(int,self.m.body_weldid[[b1,b2]])
                names=[self.m.body(b).name for b in (b1,b2)]
                if not (moving(b1) or moving(b2)) or w1==w2:continue
                if frozenset(names) in seams:continue
                # Attachment contacts are intentional at the final flange. The
                # camera box itself remains tested against forearm/torso/table.
                wrist_link = 'maver_x4_asm_7_urdf_SLDASM_link_7' if self.side == 'left' else 'maver_x4_asm_7_urdf_SLDASM2_link_7'
                if mount_geom_name in [self.m.geom(g).name for g in (g1,g2)] and any(n in (wrist_link,flange_body,f"maver_x4_asm_mount{side_number}") for n in names):
                    self.exclusions.append(names);continue
                if self.side == 'left' and any(n.startswith('calibration_left_jaw_') for n in names) and any(n.startswith(('robot_base', 'maver_x4_asm_mount1', 'maver_x4_asm_7_urdf_SLDASM_link_6', 'maver_x4_asm_7_urdf_SLDASM_link_7')) for n in names if not n.startswith('calibration_left_jaw_')):
                    # Existing near-neighbor gripper seams, preserved after jaw separation.
                    self.exclusions.append(names);continue
                arm_housings=any(all((f"SLDASM{side}_" in n or n.startswith(f"robot{side}_")) for n in names) for side in (1,2))
                if arm_housings:
                    chain=[];cur=w1
                    while cur:
                        chain.append(cur);cur=int(self.m.body_weldid[self.m.body_parentid[cur]])
                    cur=w2;steps=0
                    while cur and cur not in chain:
                        steps+=1;cur=int(self.m.body_weldid[self.m.body_parentid[cur]])
                    if cur in chain and steps+chain.index(cur)<=2:continue
                protected=any(n.startswith(("zed_","d435i_","d405_","stationary_")) for n in names) or any(self.m.geom(g).name=="rough_table" for g in (g1,g2))
                if not protected:
                    pw1=int(self.m.body_weldid[self.m.body_parentid[w1]]);pw2=int(self.m.body_weldid[self.m.body_parentid[w2]])
                    if w1==pw2 or w2==pw1:continue
                self.pairs.append((g1,g2))
        self.pair_indices=np.asarray(self.pairs,int)
        self._distance = Study.distance.__get__(self)
        from convex_clearance import ConvexSeparation
        self.separation = ConvexSeparation(self.m)
        self._clearance = Study.clearance.__get__(self)

    def distance(self, data, first, second, limit=.1):
        distance = self._distance(data, first, second, limit)
        if distance < .015:
            certificate = self.separation.certificate(data, first, second)
            distance = max(distance, min(limit, certificate))
        return distance

    def update(self, q):
        import mujoco
        q=np.asarray(q,float)
        self.d.qpos[:]=0
        self.d.qpos[self.mi]=q[3:]
        if self.side == 'left':
            for name in ('positive', 'negative'):
                self.d.qpos[self.m.joint(f'calibration_left_jaw_{name}').qposadr[0]] = .04
        mujoco.mj_forward(self.m,self.d)

    def clearance(self,q):
        return self._clearance(np.asarray(q,float))

    def fk(self,q):
        import pinocchio as pin
        self.q0[self.pi]=np.asarray(q,float)[3:]
        pin.framesForwardKinematics(self.pm,self.pd,self.q0)
        return self.pd.oMf[self.hand].homogeneous.copy()

    def visibility(self,q):
        camera=self.fk(q)@self.x
        cb=np.linalg.inv(camera)@self.y
        obj=np.array([[x,y,0] for x in [0,.087] for y in [0,.087]])
        xyz=obj@cb[:3,:3].T+cb[:3,3]
        uv=xyz@self.k.T;uv=uv[:,:2]/uv[:,2:]
        center=cb@np.array([.0435,.0435,0,1])
        return dict(margin_px=float(np.min(np.c_[uv[:,0],self.width-uv[:,0],uv[:,1],self.height-uv[:,1]])),
                    facing_cosine=float(cb[:3,2]@center[:3]/np.linalg.norm(center[:3])),
                    distance_m=float(np.linalg.norm(center[:3])),center_px=np.mean(uv,axis=0).tolist(),
                    projected_corners_px=uv.tolist(),T_camera_from_board=cb.tolist(),
                    T_base_from_parent=self.fk(q).tolist())

    def render(self,q,renderer):
        import cv2,mujoco
        self.update(q)
        opt=mujoco.MjvOption();opt.geomgroup[3:5]=1;opt.geomgroup[5]=0
        renderer.update_scene(self.d,camera="wrist_calibration_view",scene_option=opt)
        renderer.scene.flags[mujoco.mjtRndFlag.mjRND_SHADOW]=False
        # MuJoCo vertical fovy enforces equal fx/fy. A tiny x-rescale gives the
        # actual per-axis SDK focal lengths and an explicit principal point.
        yy,xx=np.meshgrid(np.arange(self.height,dtype=np.float32),np.arange(self.width,dtype=np.float32),indexing="ij")
        mx=(xx-self.k[0,2])*(self.k[1,1]/self.k[0,0])+699.5
        my=yy-self.k[1,2]+399.5
        return cv2.cvtColor(cv2.remap(renderer.render(),mx,my,cv2.INTER_LINEAR),cv2.COLOR_RGB2BGR)


class LocalWristChecker:
    def __init__(self,output,plan):
        self.route = plan
        self.study = WristStudy(plan,output)

    def check_pose(self,q,bounds=None,allow_upper_retreat=False):
        q=np.asarray(q,float)
        if q.shape!=(18,) or not np.isfinite(q).all():raise ValueError("Invalid measured wrist configuration")
        if bounds is not None:
            lo,hi=map(lambda x:np.asarray(x,float),bounds)
            # Retreat starts at the measured mechanical endpoint, preserving
            # the controller's existing one-way upper-lift retreat semantics.
            if np.any(q[4:]<lo[4:]-1e-8) or np.any(q[4:]>hi[4:]+1e-8) or q[3]<lo[3]-1e-8 or q[3]>max(hi[3],.422 if allow_upper_retreat else hi[3])+1e-5:
                raise ValueError("Measured configuration outside controller bounds")
        gap,pair=self.study.clearance(q)
        if gap<.015:raise ValueError(f"Modeled clearance {gap*1000:.1f} mm below 15 mm: {pair}")
        return dict(measured_q=q.tolist(),minimum_clearance_m=float(gap),required_clearance_m=.015,
                    geometry_scope="Robot, fixed33 mount convex hull, camera box, stationary board, rough table and floor. No cables/people/room truth.")

    def edge(self,a,b,bounds=None,retreat=False):
        a,b=np.asarray(a,float),np.asarray(b,float)
        samples=max(2,int(math.ceil(np.sum(abs(b[3:]-a[3:]))/.005)))
        minimum=.1
        for u in np.linspace(0,1,samples+1):
            minimum=min(minimum,self.check_pose(a+(b-a)*u,bounds,retreat)["minimum_clearance_m"])
        return minimum,samples+1

    def plan(self,q,pose_id,bounds=None,default_target=None,reference_origin=None):
        q=np.asarray(q,float)
        arm=slice(4,11) if self.route.get('capture_side')=='left' else slice(11,18)
        def retain_fixed(target):
            result=q.copy();result[arm]=target[arm];return result
        # A bad measured state is a current-state failure, never just an
        # unreachable target that automatic calibration may skip.
        self.check_pose(q,bounds,allow_upper_retreat=pose_id==-2)
        if pose_id in (-2,-1):
            target=np.asarray(default_target,float).copy();target[:3]=q[:3]
            points=[]
            if pose_id==-2 and q[3]>.417:
                retreat=q.copy();retreat[3]=.409;points.append(retreat)
            if pose_id==-1:target[:4]=q[:4]
            points.append(target)
        elif 1<=pose_id<=len(self.route["poses"]):
            target=np.asarray(self.route["poses"][pose_id-1]["q"],float).copy()
            target=retain_fixed(target)
            points=[target]
        else:raise ValueError("Unknown wrist pose ID")
        # A returning arm can dip toward the tabletop under a straight joint
        # interpolation. Use a previously validated visible pose as a bridge,
        # checking both segments from the *current* measured/reference origins.
        if pose_id != -2:
            try:
                self.edge(q,points[0],bounds)
                if reference_origin is not None:self.edge(reference_origin,points[0],bounds)
            except ValueError:
                route_nodes=[{'q':p} for p in self.route.get('approach_waypoints', [])]+self.route["poses"]
                bridges=sorted(route_nodes,key=lambda r:np.linalg.norm(np.asarray(r["q"])[arm]-q[arm]))
                found=None
                for row in bridges:
                    bridge=retain_fixed(np.asarray(row["q"],float))
                    try:
                        self.edge(q,bridge,bounds);self.edge(bridge,target,bounds)
                        if reference_origin is not None:self.edge(reference_origin,bridge,bounds)
                    except ValueError:continue
                    found=[bridge,target];break
                if found is None:
                    if pose_id>0:raise InvalidCalibrationPose("No collision-checked approach from the current wrist pose",pose_id=pose_id)
                    # Retreat through the validated route in reverse, skipping
                    # any nodes whose direct shortcut is independently clear.
                    # This preserves a return even when no single bridge can
                    # keep the gripper above the tabletop throughout the turn.
                    reverse=[np.asarray(r['q'],float).copy() for r in route_nodes]
                    reverse=[retain_fixed(node) for node in reverse]
                    reverse.insert(0,target)
                    route=[];origin=q.copy();last=len(reverse)
                    while last:
                        next_index=None
                        for j in range(last):
                            try:
                                self.edge(origin,reverse[j],bounds)
                                if not route and reference_origin is not None:self.edge(reference_origin,reverse[j],bounds)
                            except ValueError:continue
                            next_index=j;break
                        if next_index is None:raise ValueError("No checked reverse-route return; remain stopped")
                        route.append(reverse[next_index]);origin=reverse[next_index];last=next_index
                    found=route
                points=found
        minimum=.1;count=0;origin=q.copy()
        try:
            for p in points:
                gap,n=self.edge(origin,p,bounds,retreat=pose_id==-2)
                minimum=min(minimum,gap);count+=n;origin=p
        except ValueError as error:
            if pose_id>0:raise InvalidCalibrationPose("No valid direct measured-origin path: "+str(error),pose_id=pose_id) from error
            raise
        origins=["measured"]
        if reference_origin is not None:
            reference=np.asarray(reference_origin,float)
            try:
                gap,n=self.edge(reference,points[0],bounds,retreat=pose_id==-2)
                minimum=min(minimum,gap);count+=n;origins.append("commanded_reference")
            except ValueError as error:
                if pose_id>0:raise InvalidCalibrationPose("Retained-reference approach invalid: "+str(error),pose_id=pose_id) from error
                raise
        return dict(origin=q.tolist(),targets=[p.tolist() for p in points],pose_id=pose_id,
                    reference_origin=None if reference_origin is None else np.asarray(reference_origin,float).tolist(),
                    reference_path_minimum_clearance_m=minimum if reference_origin is not None else None,
                    minimum_sampled_clearance_m=minimum,path_samples=count,path_origins_checked=origins,
                    target_joint_limit_reserve_rad=.15,nominal_view=self.study.visibility(points[-1]) if pose_id>0 else None,
                    geometry_scope="Model geometry with rough stationary table/board only; cables and actual room unmodeled.")


class WristPathChecker:
    def __init__(self,output,plan_path):
        self.process=subprocess.Popen([sys.executable,str(Path(__file__)),"--worker",str(output),str(plan_path)],stdin=subprocess.PIPE,stdout=subprocess.PIPE,text=True,bufsize=1)
        while True:
            line=self.process.stdout.readline()
            if not line:raise RuntimeError("Wrist geometry worker failed to initialize")
            if line.startswith("READY "):
                self.study=type("Scene",(),{"scene":Path(line[6:].strip())})();break
    def request(self,value):
        self.process.stdin.write(json.dumps(value)+"\n");self.process.stdin.flush()
        line=self.process.stdout.readline()
        if not line:raise ValueError("Wrist geometry worker disconnected")
        result=json.loads(line)
        if "error" in result:
            if result.get("error_kind")=="invalid_pose":raise InvalidCalibrationPose(result["error"],pose_id=result.get("pose_id"))
            raise ValueError(result["error"])
        return result
    def plan(self,q,pose_id,bounds=None,default_target=None,reference_origin=None):
        return self.request(dict(q=np.asarray(q).tolist(),pose_id=pose_id,bounds=bounds,default_target=default_target,reference_origin=reference_origin))
    def check_pose(self,q,bounds=None):
        return self.request(dict(action="check-pose",q=np.asarray(q).tolist(),bounds=bounds))
    def close(self):
        self.process.stdin.close()
        try:self.process.wait(timeout=3)
        except subprocess.TimeoutExpired:self.process.terminate();self.process.wait(timeout=3)


def main():
    if len(sys.argv)>1 and sys.argv[1]=="--worker":
        checker=LocalWristChecker(Path(sys.argv[2]),load_plan(sys.argv[3]))
        print("READY "+str(checker.study.scene),flush=True)
        for line in sys.stdin:
            try:
                request=json.loads(line)
                if request.get("action")=="check-pose":result=checker.check_pose(request["q"],request.get("bounds"))
                else:result=checker.plan(request["q"],request["pose_id"],request.get("bounds"),request.get("default_target"),request.get("reference_origin"))
            except InvalidCalibrationPose as e:result=dict(error=str(e),error_kind="invalid_pose",pose_id=e.pose_id)
            except Exception as e:result=dict(error=str(e),error_kind="failure")
            print(json.dumps(result),flush=True)
        return
    from plan_wrist_calibration import run
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path,default=ROOT/"runs/right-wrist-calibration-planning")
    parser.add_argument("--plan",type=Path,default=DEFAULT_PLAN)
    args=parser.parse_args();run(args.output,args.plan)


if __name__=="__main__":main()
