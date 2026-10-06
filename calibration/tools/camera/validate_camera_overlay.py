#!/usr/bin/env python3
"""Offline measured-joint replay through a fitted ZED optical camera.

No SDK, robot control, image registration, or extrinsic fitting. Real RGB must be
the rectified LEFT view with the supplied K and paired measured model joints.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import xml.etree.ElementTree as ET

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'src'))
import cv2
import mujoco
import numpy as np
from scipy.spatial.transform import Rotation
from policy_sim.control.robot import PLANAR_CONTROLLED_COORDINATES
from policy_sim.calibration.handeye import validate

MOUNT='maver_x4_asm_URDF_link_1'
CAMERA='extrinsic_validation'
OPTICAL_TO_GL=np.diag([1.,-1.,-1.])


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def camera_remap(k,width,height,raw_width,raw_height):
    """MuJoCo centered pinhole -> real rectified pixel coordinates."""
    k=np.asarray(k,float)
    if (k.shape!=(3,3) or not np.all(np.isfinite(k)) or min(k[0,0],k[1,1])<=0
            or not np.allclose(k[2],[0,0,1]) or abs(k[0,1])+abs(k[1,0])>1e-9):
        raise ValueError('Expected finite pinhole K with zero skew')
    x=(np.arange(width,dtype=np.float32)-k[0,2])*k[1,1]/k[0,0]+raw_width/2
    y=np.arange(height,dtype=np.float32)-k[1,2]+raw_height/2
    if min(x.min(),y.min())<0 or x.max()>raw_width-1 or y.max()>raw_height-1:
        raise ValueError('Camera intrinsics require a larger offscreen render')
    return np.meshgrid(x,y)


def inject_camera(scene,destination,x,k,width=1920,height=1080,board_transform=None):
    x=validate([x])[0]
    tree=ET.parse(scene);root=tree.getroot()
    for asset in root.findall('./asset/*'):
        if asset.get('file'):
            asset.set('file',str((Path(scene).parent/asset.get('file')).resolve()))
    raw_width=max(2000,width+160);raw_height=max(1160,height+160)
    visual=root.find('visual')
    if visual is None:visual=ET.SubElement(root,'visual')
    global_visual=visual.find('global')
    if global_visual is None:global_visual=ET.SubElement(visual,'global')
    global_visual.set('offwidth',str(raw_width));global_visual.set('offheight',str(raw_height))
    camera_parent=root.find(f".//body[@name='{MOUNT}']")
    if camera_parent is None:raise ValueError('Scene does not contain the calibrated lift frame')
    q=Rotation.from_matrix(x[:3,:3]@OPTICAL_TO_GL).as_quat()
    ET.SubElement(camera_parent,'camera',name=CAMERA,pos=' '.join(map(str,x[:3,3])),
        quat=' '.join(map(str,np.r_[q[3],q[:3]])),
        fovy=str(np.degrees(2*np.arctan(raw_height/(2*np.asarray(k)[1,1])))))
    if board_transform is not None:
        y=validate([board_transform])[0]
        b=root.find(".//body[@name='calibration_board']")
        if b is not None:
            q=Rotation.from_matrix(y[:3,:3]).as_quat()
            b.set('pos',' '.join(map(str,y[:3,3])))
            b.set('quat',' '.join(map(str,np.r_[q[3],q[:3]])))
    destination.parent.mkdir(parents=True,exist_ok=True);tree.write(destination)
    return raw_width,raw_height


class OverlayRenderer:
    def __init__(self,scene,x,k,output,width=1920,height=1080,board_transform=None):
        self.width,self.height=width,height
        self.k=np.asarray(k,float);self.x=validate([x])[0]
        self.scene=Path(output)/'overlay-scene.xml'
        self.raw_width,self.raw_height=inject_camera(scene,self.scene,x,k,width,height,board_transform)
        self.model=mujoco.MjModel.from_xml_path(str(self.scene));self.data=mujoco.MjData(self.model)
        self.indices=[self.model.joint(n).qposadr[0] for n in PLANAR_CONTROLLED_COORDINATES]
        self.option=mujoco.MjvOption();self.option.geomgroup[3:5]=1
        # Target handles/frames are annotations, not robot geometry.
        self.model.site_rgba[:,3]=0
        board_id=mujoco.mj_name2id(self.model,mujoco.mjtObj.mjOBJ_BODY,'calibration_board')
        for i in range(self.model.ngeom):
            if self.model.geom_group[i]==4 and self.model.geom_bodyid[i]!=board_id:
                self.model.geom_rgba[i,3]=0
        self.renderer=mujoco.Renderer(self.model,height=self.raw_height,width=self.raw_width)
        self.mx,self.my=camera_remap(k,width,height,self.raw_width,self.raw_height)

    def close(self):self.renderer.close()

    def set_q(self,q):
        q=np.asarray(q,float)
        if q.shape!=(18,) or not np.all(np.isfinite(q)):
            raise ValueError('Expected 18 finite measured model coordinates, not raw SDK angles')
        self.data.qpos[self.indices]=q
        mujoco.mj_forward(self.model,self.data)

    def geom_ids(self,side,whole_arm=False):
        if side not in ('left','right'):raise ValueError('Unknown arm')
        root_name=(('maver_x4_asm_7_urdf_SLDASM_base_link' if side=='left' else 'maver_x4_asm_7_urdf_SLDASM2_base_link')
                   if whole_arm else ('maver_x4_asm_mount1' if side=='left' else 'maver_x4_asm_mount2'))
        ancestor=self.model.body(root_name).id
        ids=[]
        for geom in range(self.model.ngeom):
            body=int(self.model.geom_bodyid[geom])
            # The right-hand calibration board is not part of the gripper mask.
            if self.model.body(body).name=='calibration_board':continue
            while body and body!=ancestor:body=int(self.model.body_parentid[body])
            if body==ancestor:ids.append(geom)
        return ids

    def render(self,q,side='left',whole_arm=False):
        self.set_q(q)
        self.renderer.update_scene(self.data,camera=CAMERA,scene_option=self.option)
        self.renderer.scene.flags[mujoco.mjtRndFlag.mjRND_SHADOW]=False
        rgb=cv2.remap(self.renderer.render(),self.mx,self.my,cv2.INTER_LINEAR)
        self.renderer.enable_segmentation_rendering()
        try:
            seg=cv2.remap(self.renderer.render(),self.mx,self.my,cv2.INTER_NEAREST,
                          borderMode=cv2.BORDER_CONSTANT,borderValue=(-1,-1))
        finally:self.renderer.disable_segmentation_rendering()
        mask=np.isin(seg[:,:,0],self.geom_ids(side,whole_arm))&(seg[:,:,1]==int(mujoco.mjtObj.mjOBJ_GEOM))
        return rgb,mask

    def overview(self,q):
        self.set_q(q)
        camera=mujoco.MjvCamera();camera.lookat[:]=[.30,0,1.05]
        camera.distance=2.1;camera.azimuth=140;camera.elevation=-18
        self.renderer.update_scene(self.data,camera=camera,scene_option=self.option)
        return self.renderer.render()


def overlay(real,rgb,mask):
    if real.shape!=rgb.shape:raise ValueError('Real and rendered image sizes differ')
    result=real.copy()
    result[mask]=np.uint8(.65*real[mask]+.35*rgb[mask])
    contours,_=cv2.findContours(np.uint8(mask)*255,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(result,contours,-1,(0,255,210),2)
    return result


def mask_metrics(predicted,observed):
    predicted,observed=np.asarray(predicted,bool),np.asarray(observed,bool)
    if predicted.shape!=observed.shape or not predicted.any() or not observed.any():
        raise ValueError('Both masks must be nonempty, aligned, and the same size')
    intersection=np.count_nonzero(predicted&observed);union=np.count_nonzero(predicted|observed)
    def boundary(mask):return mask&~cv2.erode(np.uint8(mask),np.ones((3,3),np.uint8)).astype(bool)
    a,b=boundary(predicted),boundary(observed)
    da=cv2.distanceTransform(np.uint8(~a),cv2.DIST_L2,cv2.DIST_MASK_PRECISE)
    db=cv2.distanceTransform(np.uint8(~b),cv2.DIST_L2,cv2.DIST_MASK_PRECISE)
    distances=np.r_[da[b],db[a]]
    return dict(iou=intersection/union,boundary_mean_px=float(distances.mean()),
                boundary_p95_px=float(np.percentile(distances,95)))


def write_comparison(renderer,row,output,side,whole_arm=False,real_mask=None):
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    real=cv2.imread(row['image'])
    if real is None:raise ValueError('Cannot read the saved real image')
    real=cv2.cvtColor(real,cv2.COLOR_BGR2RGB)
    if real.shape[:2]!=(renderer.height,renderer.width):raise ValueError('Image size does not match camera intrinsics')
    rgb,mask=renderer.render(row['q'],side,whole_arm)
    combined=overlay(real,rgb,mask)
    for name,image in [('real.png',real),('sim.png',rgb),('overlay.png',combined)]:
        cv2.imwrite(str(output/name),cv2.cvtColor(image,cv2.COLOR_RGB2BGR))
    cv2.imwrite(str(output/'predicted-mask.png'),np.uint8(mask)*255)
    ys,xs=np.where(mask)
    stats=dict(pose_id=row.get('pose_id'),side=side,mask_scope='whole arm' if whole_arm else 'gripper',
        simulated_visible_pixels=int(mask.sum()),
        simulated_bbox_xyxy=[int(xs.min()),int(ys.min()),int(xs.max()),int(ys.max())] if len(xs) else None,
        measured_q=row['q'],pairing=row.get('pairing'),
        joint_names=list(PLANAR_CONTROLLED_COORDINATES),
        real_image=row['image'],real_image_sha256=digest(row['image']),
        geometry_scene_file=str(renderer.scene),geometry_scene_sha256=digest(renderer.scene),
        K=renderer.k.tolist(),T_mount_from_camera=renderer.x.tolist(),
        camera_frame='rectified LEFT optical; x right, y down, z forward',
        image_registration_applied=False,quantitative_overlap=None)
    if real_mask is not None:
        observed=cv2.imread(str(real_mask),cv2.IMREAD_GRAYSCALE)
        if observed is None:raise ValueError('Cannot read real hand mask')
        stats['quantitative_overlap']=mask_metrics(mask,observed>127)
    (output/'comparison.json').write_text(json.dumps(stats,indent=2))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,3,figsize=(18,4.2))
    simulated_capture=bool(row.get('simulation_only',False))
    titles=['Simulated capture (workflow test)' if simulated_capture else 'Real rectified left image',
            'Simulation at measured joints',
            'Capture + predicted outline' if simulated_capture else 'Real image + predicted outline']
    for ax,image,title in zip(axes,[real,rgb,combined],titles):
        ax.imshow(image);ax.set_title(title);ax.axis('off')
    fig.suptitle(f'{side.capitalize()} {stats["mask_scope"]} · pose {row.get("pose_id", "measured")} · fitted extrinsics · no image alignment')
    fig.tight_layout();fig.savefig(output/'comparison.png',dpi=140);plt.close(fig)
    return stats


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset',type=Path,required=True)
    p.add_argument('--extrinsics',type=Path,required=True)
    p.add_argument('--pose',type=int,required=True)
    p.add_argument('--side',choices=['left','right'],default='left')
    p.add_argument('--whole-arm',action='store_true')
    p.add_argument('--real-mask',type=Path)
    p.add_argument('--scene',type=Path)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();data=json.loads(a.dataset.read_text());fit=json.loads(a.extrinsics.read_text())
    if data['camera'].get('view')!='LEFT rectified' or np.any(np.asarray(data['camera'].get('distortion',[0]),float)!=0):
        raise ValueError('Replay requires the rectified LEFT image and its rectified intrinsics')
    rows=data['observations']+data.get('ineligible',[])
    row=next(r for r in rows if r['pose_id']==a.pose)
    if row.get('pairing') and not np.array_equal(row['q'],row['pairing']['q']):
        raise ValueError('Replay joints differ from the saved image/feedback pair')
    image=cv2.imread(row['image'])
    if image is None:raise ValueError('No saved image for requested pose')
    a.output.mkdir(parents=True,exist_ok=False)
    renderer=OverlayRenderer(a.scene or a.dataset.parent/'geometry/scene.xml',fit['T_mount_from_camera'],
        data['camera']['K'],a.output,image.shape[1],image.shape[0],fit.get('T_hand_from_board'))
    try:report=write_comparison(renderer,row,a.output,a.side,a.whole_arm,a.real_mask)
    finally:renderer.close()
    report.update(extrinsics_file=str(a.extrinsics),extrinsics_sha256=digest(a.extrinsics),
                  dataset_file=str(a.dataset),dataset_sha256=digest(a.dataset),offline_only=True)
    (a.output/'comparison.json').write_text(json.dumps(report,indent=2))
    print(json.dumps({k:report[k] for k in ('side','simulated_visible_pixels','simulated_bbox_xyxy','quantitative_overlap')},indent=2))


if __name__=='__main__':main()
