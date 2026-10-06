#!/usr/bin/env python3
"""Static MuJoCo visibility study; no hardware imports or motion commands.

Reuses GP80 robot meshes, PoolBot table dimensions, nominal PoolBot camera mount,
and measured ZED rectified intrinsics. Views are conditional on uncalibrated mount.
"""
import argparse
import copy
import hashlib
import json
import math
from pathlib import Path
import sys
import xml.etree.ElementTree as ET
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'src'))
import cv2
import mujoco
import numpy as np
import pinocchio as pin
from PIL import Image, ImageDraw, ImageFont
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation
from policy_sim.control.robot import CONTROLLED_JOINTS, RIGHT_ARM_JOINTS
from policy_sim.control.robot_model import load_robot_model
from policy_sim.scene.builder import build_tabletop_scene

FONTPATH='/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'
font=ImageFont.truetype(FONTPATH,20);small=ImageFont.truetype(FONTPATH,16)

def values(x):return ' '.join(f'{v:.9g}' for v in np.ravel(x))

def geom(parent,name,kind,pos,size,color):
    return ET.SubElement(parent,'geom',name=name,type=kind,pos=values(pos),size=values(size),rgba=values(color),contype='0',conaffinity='0',group='0')

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=ROOT/'runs/camera-view-study')
    parser.add_argument('--mounted', action='store_true', help='Render the integrated short-bracket ZED URDF with frustum and camera view pairs')
    parser.add_argument('--mount-angle',type=int,choices=(15,32,40,42,55),default=15,help='Shelf tilt for the short 65 mm bracket; use with --mounted')
    parser.add_argument('--config',type=Path,default=ROOT/'config/camera/zed_view_study.json')
    args=parser.parse_args();out=args.output.resolve();out.mkdir(parents=True,exist_ok=True)
    snapshot=json.loads(args.config.read_text());camera=snapshot
    k=np.array(camera['K']);fx,fy,cx,cy=k[0,0],k[1,1],k[0,2],k[1,2]
    width,height=1920,1080
    table_config=snapshot['table_config']
    mounted_descriptor='maver-x4-gp80-zed-view-v1.json' if args.mount_angle==15 else f'maver-x4-gp80-zed-{args.mount_angle}deg-view-v1.json'
    spec=ROOT/'config/control/robot_models'/(mounted_descriptor if args.mounted else 'maver-x4-gp80-fixed-v1.json')
    urdf_path=ROOT/json.loads(spec.read_text())['urdf']
    basefile=out/('robot_zed.xml' if args.mounted else 'robot.xml')
    if args.mounted or not basefile.exists():build_tabletop_scene(output_path=basefile,object_ids=(),include_table=False,include_box=False,include_trashcan=False,base_mode='planar',preserve_fixed_inertias=True,robot_model=load_robot_model({'path':str(spec.relative_to(ROOT)),'sha256':hashlib.sha256(spec.read_bytes()).hexdigest()}),appearance_seed=0,layout_seed=0)
    base=ET.parse(basefile).getroot()
    for elem in base.findall('./asset/*'):
        if elem.get('file'):elem.set('file',str((basefile.parent/elem.get('file')).resolve()))
    # Overscan followed by subpixel crop reproduces off-centre measured intrinsics.
    rw,rh=2000,1160
    base.find('visual/global').set('offwidth',str(rw));base.find('visual/global').set('offheight',str(rh))
    base.find('visual/quality').set('offsamples','2')
    base.find('visual/headlight').set('ambient','.35 .35 .35')
    base.find('visual/headlight').set('diffuse','.45 .45 .45')
    base.find('visual/map').set('zfar','20')
    for s in list(base.findall('.//site')):
        s.set('rgba','0 0 0 0')
    for g in base.findall('.//geom'):
        if g.get('name','').endswith('target'):g.set('rgba','0 0 0 0')
    world=base.find('worldbody')
    ET.SubElement(world,'camera',name='study_camera',pos='0 0 2',fovy=str(math.degrees(2*math.atan(rh/(2*fy)))))
    scenarios=[('bench_grasp','Near-table grasp',.42,.85,.80,.76),
               ('bench_transfer','Right-hand object transfer',.42,.85,.80,.76),
               ('pool_end','7-ft pool / close end',.60,table_config['length_m'],table_config['width_m'],table_config['cloth_height_m']),
               ('pool_end_back','7-ft pool / step back',1.00,table_config['length_m'],table_config['width_m'],table_config['cloth_height_m']),
               ('pool_side','7-ft pool / close side',.60,table_config['width_m'],table_config['length_m'],table_config['cloth_height_m']),
               ('pool_side_back','7-ft pool / side step back',1.30,table_config['width_m'],table_config['length_m'],table_config['cloth_height_m'])]
    variants=[('level',0.,0.,False,'Level / actual lens'),('cad32',32.,0.,False,'CAD 32 deg / actual lens'),
              ('tilt42',42.,0.,False,'Tilt 42 deg / actual lens'),('tilt55',55.,0.,False,'Tilt 55 deg / actual lens'),
              ('lower20',32.,-.20,False,'Lift -20 cm / 32 deg'),('legacy',32.,0.,True,'Legacy 110 deg / 32 deg')]
    if args.mounted:
        from mounted_view import render_pair
        mount_tree=ET.parse(urdf_path)
        mounted_pitch=math.degrees(float(mount_tree.find("joint[@name='zed_camera_mount']/origin").get('rpy').split()[1]))
        variants=[(f'short{args.mount_angle}',mounted_pitch,0.,False,f'Short v5 / {mounted_pitch:g} deg / actual lens')]
    pmodel=pin.buildModelFromUrdf(str(urdf_path));pdata=pmodel.createData()
    pindices=[pmodel.joints[pmodel.getJointId(n)].idx_q for n in CONTROLLED_JOINTS]
    armidx=[pmodel.joints[pmodel.getJointId(n)].idx_q for n in RIGHT_ARM_JOINTS]
    frame=pmodel.getFrameId('maver_x4_asm_mount2')
    def place_arm(qfull,target):
        pq=pin.neutral(pmodel);pq[pindices]=qfull[3:]
        initial=pq[armidx].copy();pin.framesForwardKinematics(pmodel,pdata,pq)
        desired_r=pdata.oMf[frame].rotation.copy()
        def err(arm):
            pq[armidx]=arm;pin.framesForwardKinematics(pmodel,pdata,pq);pose=pdata.oMf[frame]
            return np.r_[5*(pose.translation-target),.03*Rotation.from_matrix(desired_r.T@pose.rotation).as_rotvec(),.0005*(arm-initial)]
        fit=least_squares(err,initial,bounds=(pmodel.lowerPositionLimit[armidx]+.005,pmodel.upperPositionLimit[armidx]-.005),max_nfev=500)
        rng=np.random.default_rng(34)
        for trial in range(8):
            if np.linalg.norm(err(fit.x)[:3])/5 < .002:break
            lo=pmodel.lowerPositionLimit[armidx]+.005;hi=pmodel.upperPositionLimit[armidx]-.005
            seed=np.clip(initial+rng.normal(0,.8,7),lo+1e-5,hi-1e-5)
            candidate=least_squares(err,seed,bounds=(lo,hi),max_nfev=500)
            if np.linalg.norm(err(candidate.x)) < np.linalg.norm(err(fit.x)):fit=candidate
        answer=qfull.copy();answer[11:]=fit.x
        return answer,float(np.linalg.norm(err(fit.x)[:3])/5)
    report={'integrated_camera_urdf': str(urdf_path.relative_to(ROOT)) if args.mounted else None,
      'camera_self_body_excluded_from_rays': bool(args.mounted),
      'simulation_only':True,'motion_commands_sent':0,'intrinsics':camera['K'],
      'fov_deg':[float(np.degrees(np.arctan(cx/fx)+np.arctan((width-cx)/fx))),float(np.degrees(np.arctan(cy/fy)+np.arctan((height-cy)/fy)))],
      'camera_model':'ZED 2i serial 32701991, rectified LEFT, 1920x1080',
      'camera_mount':(f'Short triangular v5: shelf 65 mm, pitch {args.mount_angle} deg' if args.mounted else 'Historical tall PoolBot mount')+'; physical extrinsics NOT calibrated',
      'measured_lift_m':snapshot['q'][3], 'table_config':table_config,
      'limitations':['Static visibility, not trajectory or collision qualification','Camera tilt and mount offsets are nominal, not measured',
       'Simplified workbench and table geometry; 7-foot cloth dimensions match PoolBot','No cables, tape or calibration board modeled',
       'Ray checks include robot visual meshes; no depth accuracy claims','Grasp/transfer arm poses are illustrative static IK, not executed trajectories'], 'results':[]}
    frames=[]
    for scene,title,near,length,breadth,z in scenarios:
        tree=copy.deepcopy(base);world=tree.find('worldbody')
        geom(world,'surface','box',[near+length/2,0,z-.025],[length/2,breadth/2,.025],[.06,.40,.24,1] if scene.startswith('pool') else [.58,.40,.23,1])
        for x in [near+.12,near+length-.12]:
            for y in [-breadth/2+.1,breadth/2-.1]:geom(world,f'leg_{x}_{y}','box',[x,y,(z-.06)/2],[.045,.045,(z-.06)/2],[.14,.12,.10,1])
        object_targets=[]
        if scene.startswith('pool'):
            for y in [-breadth/2-.045,breadth/2+.045]:geom(world,f'rail_y{y}','box',[near+length/2,y,z+.02],[length/2+.09,.045,.045],[.19,.105,.05,1])
            for x in [near-.045,near+length+.045]:geom(world,f'rail_x{x}','box',[x,0,z+.02],[.045,breadth/2,.045],[.19,.105,.05,1])
            for x in [near,near+length]:
                for y in [-breadth/2,breadth/2]:geom(world,f'pocket{x}{y}','cylinder',[x,y,z+.027],[.05,.012],[.015,.015,.015,1])
            # Middle pockets on the long rails.
            middle=[(near+length/2,y) for y in [-breadth/2,breadth/2]] if length>breadth else [(x,0) for x in [near,near+length]]
            for i,(x,y) in enumerate(middle):geom(world,f'middle{i}','cylinder',[x,y,z+.027],[.05,.012],[.015,.015,.015,1])
            rng=np.random.default_rng(17)
            for i in range(16):
                x=near+.12+(length-.24)*rng.random();y=(breadth-.24)*(rng.random()-.5)
                color=([.96,.96,.90,1] if i==0 else [.03,.03,.04,1] if i==8 else [(.9,.7,.1),(.1,.3,.9),(.9,.12,.1),(.55,.15,.7)][i%4]+(1,))
                geom(world,f'ball{i}','sphere',[x,y,z+.028575],[.028575],color)
                object_targets.append((f'ball{i}',np.array([x,y,z+.043])))
        else:
            for i,(x,y,color) in enumerate([(.53,-.18,[.9,.15,.1,1]),(.76,.18,[.15,.35,.9,1]),(1.04,-.22,[.95,.65,.08,1])]):
                geom(world,f'block{i}','box',[x,y,z+.035],[.035,.035,.035],color)
                object_targets.append((f'block{i}',np.array([x,y,z+.071])))
            geom(world,'bottle','cylinder',[.92,.31,z+.13],[.037,.13],[.05,.7,.7,1]);object_targets.append(('bottle',np.array([.92,.31,z+.261])))
            geom(world,'tray','box',[.98,.06,z+.015],[.16,.11,.015],[.30,.32,.34,1])
        scene_path=out/f'{scene}.xml';ET.ElementTree(tree).write(scene_path)
        model=mujoco.MjModel.from_xml_path(str(scene_path));data=mujoco.MjData(model)
        renderer=mujoco.Renderer(model,height=rh,width=rw)
        cam_id=mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_CAMERA,'study_camera')
        joint_ids=[mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_JOINT,n) for n in CONTROLLED_JOINTS]
        liftbody=mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_BODY,'maver_x4_asm_URDF_link_1')
        surfaces=[]
        for ix in np.linspace(near+.005,near+length-.005,45):
            for iy in np.linspace(-breadth/2+.005,breadth/2-.005,31):surfaces.append([ix,iy,z+.001])
        surfaces=np.array(surfaces)
        corners=np.array([[x,y,z+.002] for x in [near,near+length] for y in [-breadth/2,breadth/2]])
        panel=Image.new('RGB',(1440,3*460+90),'#111c2c');draw=ImageDraw.Draw(panel)
        draw.text((20,15),title+' | simulated ZED left view',font=font,fill='white')
        draw.text((20,48),'Nominal mount geometry; real camera tilt is not yet calibrated. Captions use native 1080p metrics.',font=small,fill='#bdcbdf')
        for n,(variant,pitch,drop,legacy,label) in enumerate(variants):
            q=np.array(snapshot['q']);q[:3]=0;q[3]+=drop
            ikerr=None
            if scene.startswith('bench'):
                target=np.array([.53,-.18,z+.23]) if scene=='bench_grasp' else np.array([.48,-.07,z+.32])
                q,ikerr=place_arm(q,target)
            data.qpos[:]=0
            for jid,val in zip(joint_ids,q[3:]):data.qpos[model.jnt_qposadr[jid]]=val
            mujoco.mj_forward(model,data)
            lift=data.xpos[liftbody]
            ry=Rotation.from_euler('y',pitch,degrees=True).as_matrix()
            # PoolBot URDF: bracket (.04,0,.08), camera_link (.00159,0,.16754),
            # center (0,0,.015), left (-.01,.06,0).
            position=lift+np.array(snapshot['nominal_camera_link_in_mount_xyz_m'])+ry@np.array(snapshot['camera_link_to_left_optical_origin_m'])
            optical=ry@np.array([[0,0,1],[-1,0,0],[0,-1,0]])
            if args.mounted:
                pq=pin.neutral(pmodel);pq[pindices]=q[3:]
                pin.framesForwardKinematics(pmodel,pdata,pq)
                parent=pdata.oMf[pmodel.getFrameId('maver_x4_asm_URDF_link_1')]
                camera_pose=parent.inverse()*pdata.oMf[pmodel.getFrameId('zed_left_camera_frame_optical')]
                lift_r=data.xmat[liftbody].reshape(3,3)
                position=lift+lift_r@camera_pose.translation
                optical=lift_r@camera_pose.rotation
            gl=optical@np.diag([1,-1,-1]);quat=Rotation.from_matrix(gl).as_quat()
            model.cam_pos[cam_id]=position;model.cam_quat[cam_id]=np.r_[quat[3],quat[:3]]
            focal=width/(2*math.tan(math.radians(55))) if legacy else fx
            kview=np.array([[focal,0,width/2 if legacy else cx],[0,focal,height/2 if legacy else cy],[0,0,1.]])
            model.cam_fovy[cam_id]=math.degrees(2*math.atan(rh/(2*focal)))
            mujoco.mj_forward(model,data)
            renderer.update_scene(data,camera='study_camera');renderer.scene.flags[mujoco.mjtRndFlag.mjRND_SHADOW]=False
            raw=renderer.render()
            mapx,mapy=np.meshgrid(np.arange(width,dtype=np.float32)+rw/2-kview[0,2],np.arange(height,dtype=np.float32)+rh/2-kview[1,2])
            rgb=cv2.remap(raw,mapx,mapy,cv2.INTER_LINEAR)
            Image.fromarray(rgb).save(out/f'{scene}_{variant}.png')
            def check(points,occlusion=False,names=None):
                local=(points-position)@optical
                uv=local[:,:2]/np.maximum(local[:,2:3],1e-10)*[kview[0,0],kview[1,1]]+kview[:2,2]
                inside=(local[:,2]>.01)&(uv[:,0]>=0)&(uv[:,0]<width)&(uv[:,1]>=0)&(uv[:,1]<height)
                visible=inside.copy()
                if occlusion:
                    for j in np.flatnonzero(inside):
                        direction=points[j]-position;distance=np.linalg.norm(direction);direction/=distance
                        gid=np.zeros(1,dtype=np.int32)
                        # The decorative camera mesh has a convex ray hull enclosing its own pinhole.
                        # Exclude only the sensor housing; the bracket and robot still occlude.
                        excluded=mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_BODY,'zed_camera_center') if args.mounted else -1
                        hit=mujoco.mj_ray(model,data,position,direction,None,True,excluded,gid)
                        own=names is not None and gid[0]>=0 and mujoco.mj_id2name(model,mujoco.mjtObj.mjOBJ_GEOM,int(gid[0]))==names[j]
                        visible[j]=own or hit<0 or hit>=distance-.008
                return inside,visible,uv,local
            inside,visible,_,_=check(surfaces,True);cin,_,cuv,_=check(corners)
            oin,ovis,ouv,olocal=check(np.array([x[1] for x in object_targets]),True,[x[0] for x in object_targets])
            row={'scene':scene,'variant':variant,'pitch_deg':pitch,'camera_height_m':float(position[2]),
                'robot_q':q.tolist(),'camera_position_m':position.tolist(),'near_edge_from_camera_m':float(near-position[0]),
                'surface_in_image_pct':float(100*inside.mean()),'surface_visible_pct':float(100*visible.mean()),
                'surface_corners_in_image':int(cin.sum()),'surface_corner_min_border_px':float(np.min(np.c_[cuv[:,0],width-cuv[:,0],cuv[:,1],height-cuv[:,1]])),'objects_in_image':int(oin.sum()),'objects_unoccluded':int(ovis.sum()),
                'objects_total':len(object_targets),'illustrative_arm_ik_error_m':ikerr,
                'table_corner_pixels':cuv.tolist(),'min_ball_diameter_px':float((focal*.05715/olocal[:,2]).min()) if scene.startswith('pool') else None}
            if args.mounted:
                row.update(render_pair(renderer,data,position,optical,kview,rgb,out,scene,title,near,length,z))
            report['results'].append(row)
            thumb=Image.fromarray(rgb).resize((720,405))
            x=(n%2)*720;y=90+(n//2)*460
            panel.paste(thumb,(x,y));d=ImageDraw.Draw(panel)
            d.text((x+12,y+409),label+f' | h={position[2]:.2f} m',font=small,fill='white')
            d.text((x+12,y+432),f"Surface unoccluded {row['surface_visible_pct']:.0f}% | surface corners {row['surface_corners_in_image']}/4 | objects {row['objects_unoccluded']}/{len(object_targets)}",font=small,fill='#a8ddcc')
            print(json.dumps({key:row[key] for key in ('scene','variant','surface_visible_pct','surface_corners_in_image','objects_unoccluded')}),flush=True)
            if variant=='cad32' and not args.mounted:
                overview=mujoco.MjvCamera();overview.lookat[:]=[near+length*.3,0,.85];overview.distance=3.8 if scene.startswith('pool') else 2.8;overview.azimuth=135;overview.elevation=-25
                renderer.update_scene(data,camera=overview)
                # Draw optical axis and sensor location in an external scene view.
                geom_index=renderer.scene.ngeom;g=renderer.scene.geoms[geom_index]
                mujoco.mjv_initGeom(g,mujoco.mjtGeom.mjGEOM_ARROW,np.ones(3)*.01,position,np.eye(3).ravel(),np.array([1.,.4,.05,1.]))
                mujoco.mjv_connector(g,mujoco.mjtGeom.mjGEOM_ARROW,.015,position,position+optical[:,2]*.65);renderer.scene.ngeom+=1
                Image.fromarray(renderer.render()).resize((1000,580)).save(out/f'{scene}_overview.png')
        if not args.mounted:
            panel.save(out/f'{scene}_comparison.jpg',quality=94);frames.append(panel)
        renderer.close()
    (out/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False))
    if args.mounted:
        print('Mounted model report:',out/'report.json')
        return
    # A compact summary figure compares the same three angles for all tasks.
    sheet=Image.new('RGB',(3*640,70+len(scenarios)*420),'#111c2c');d=ImageDraw.Draw(sheet)
    d.text((20,12),'Camera view study | actual ZED intrinsics | 32 / 42 / 55 degrees downward',font=font,fill='white')
    d.text((20,40),'Nominal camera mount, measured lift height. Static renders; not a calibration or motion safety test.',font=small,fill='#b7c9df')
    for i,(scene,title,*_) in enumerate(scenarios):
        for j,var in enumerate(('cad32','tilt42','tilt55')):
            im=Image.open(out/f'{scene}_{var}.png').resize((640,360));x=j*640;y=70+i*420;sheet.paste(im,(x,y))
            row=next(r for r in report['results'] if r['scene']==scene and r['variant']==var)
            d.text((x+10,y+363),title+f" | {row['pitch_deg']:.0f} deg",font=small,fill='white')
            d.text((x+10,y+386),f"Visible surface {row['surface_visible_pct']:.0f}% | surface corners {row['surface_corners_in_image']}/4",font=small,fill='#9de1bf')
    sheet.save(out/'summary.jpg',quality=94)
    print('Report:',out/'report.json')
if __name__=='__main__':main()
