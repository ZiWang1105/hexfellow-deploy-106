#!/usr/bin/env python3
"""Offline calibration pose/path study. Does not import any hardware endpoint.

Board attachment is a photo-derived planning assumption, not a calibration.
"""
import argparse,hashlib,json,math,sys,xml.etree.ElementTree as ET
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT.parent/'src'))
import cv2,mujoco,numpy as np,pinocchio as pin
import yaml
from scipy.spatial.transform import Rotation
from hexfellow_deploy.robotics.control.robot import CONTROLLED_JOINTS,RIGHT_ARM_JOINTS
from hexfellow_deploy.robotics.control.robot_model import load_robot_model
from hexfellow_deploy.robotics.scene.builder import build_tabletop_scene
from hexfellow_deploy.robotics.calibration.charuco import make_board,detect

def fmt(v):return ' '.join(map(str,np.asarray(v).ravel()))
def quat(r):
 q=Rotation.from_matrix(r).as_quat();return fmt(np.r_[q[3],q[:3]])
def T(r=np.eye(3),p=np.zeros(3)):
 t=np.eye(4);t[:3,:3]=r;t[:3,3]=p;return t

class Study:
 def __init__(self,snapshot,out,attachment=None,include_left=False):
  self.out=out;out.mkdir(parents=True,exist_ok=True)
  self.snapshot=snapshot;self.q18=np.array(snapshot['q']);self.q18[:3]=0
  self.pm=pin.buildModelFromUrdf(str(ROOT/'urdf/maver_x4_gp80_zed_40deg.urdf'));self.pd=self.pm.createData()
  self.pi=[self.pm.joints[self.pm.getJointId(n)].idx_q for n in CONTROLLED_JOINTS]
  self.ai=np.array([self.pm.joints[self.pm.getJointId(n)].idx_q for n in RIGHT_ARM_JOINTS])
  self.q0=pin.neutral(self.pm);self.q0[self.pi]=self.q18[3:]
  self.hand=self.pm.getFrameId('maver_x4_asm_mount2');self.cf=self.pm.getFrameId('zed_left_camera_frame_optical')
  self.mount=self.pm.getFrameId('maver_x4_asm_URDF_link_1')
  pin.framesForwardKinematics(self.pm,self.pd,self.q0)
  self.camera=self.pd.oMf[self.cf].homogeneous.copy();self.mount_world=self.pd.oMf[self.mount].homogeneous.copy()
  hand0=self.pd.oMf[self.hand].homogeneous.copy()
  # Pattern +Z points into the paper; photo shows front outward (+base X), top upward.
  board_rotation=np.array([[0,0,-1],[1,0,0],[0,-1,0.]])
  # Centre at approximately the closed-finger region plus 40 mm outward for backing.
  center=hand0[:3,3]+hand0[:3,:3]@np.array([.02348191,0,.16])+np.array([.04,0,0])
  bw=T(board_rotation,center-board_rotation@np.array([.0435,.0435,0]))
  self.hb=np.linalg.inv(hand0)@bw
  if attachment is not None:
   self.hb=np.asarray(attachment['T_hand_from_board'],dtype=float)
   if self.hb.shape!=(4,4) or not np.all(np.isfinite(self.hb)) or not np.allclose(self.hb[3],[0,0,0,1]) or not np.allclose(self.hb[:3,:3].T@self.hb[:3,:3],np.eye(3),atol=1e-6) or not np.isclose(np.linalg.det(self.hb[:3,:3]),1):
    raise ValueError('Invalid board attachment rigid transform')
  self.k=np.array(json.loads((ROOT/'config/camera/zed_view_study.json').read_text())['K'])
  self.boardcfg=json.loads((ROOT/'config/camera/charuco_87mm.json').read_text())
  self.lower=self.pm.lowerPositionLimit[self.ai]+.05;self.upper=self.pm.upperPositionLimit[self.ai]-.05
  spec=ROOT/'config/control/robot_models/maver-x4-gp80-zed-40deg-view-v1.json'
  base=out/'base.xml'
  build_tabletop_scene(output_path=base,object_ids=(),include_table=False,include_box=False,include_trashcan=False,base_mode='planar',preserve_fixed_inertias=True,robot_model=load_robot_model({'path':str(spec),'sha256':hashlib.sha256(spec.read_bytes()).hexdigest()}),appearance_seed=0,layout_seed=0)
  tree=ET.parse(base);root=tree.getroot()
  for e in root.findall('./asset/*'):
   if e.get('file'):e.set('file',str((base.parent/e.get('file')).resolve()))
  root.find('visual/global').set('offwidth','2000');root.find('visual/global').set('offheight','1160')
  for s in root.findall('.//site'):s.set('rgba','0 0 0 0')
  for g in root.findall('.//geom'):
   if g.get('group')=='4':g.set('rgba','0 0 0 0')
  root.find('visual/headlight').set('ambient','.55 .55 .55')
  root.find('visual/headlight').set('diffuse','.45 .45 .45')
  handbody=root.find(".//body[@name='maver_x4_asm_mount2']")
  b=ET.SubElement(handbody,'body',name='calibration_board',pos=fmt(self.hb[:3,3]),quat=quat(self.hb[:3,:3]))
  ET.SubElement(b,'geom',name='board_backing',type='box',pos='.0435 .0435 .002',size='.05 .05 .002',rgba='1 1 1 1',contype='0',conaffinity='0',group='3')
  # Draw exact square/marker cells on the front (-Z) of the rigid backing.
  dictionary=cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50);marker_id=0
  for y in range(5):
   for x in range(5):
    if (x+y)%2==0:
     ET.SubElement(b,'geom',name=f'square_{x}_{y}',type='box',pos=fmt([(x+.5)*.0174,(y+.5)*.0174,-.0001]),size='.0087 .0087 .0001',rgba='0 0 0 1',group='4',contype='0',conaffinity='0')
    else:
     marker=cv2.aruco.generateImageMarker(dictionary,marker_id,6);marker_id+=1
     for yy,xx in zip(*np.where(marker==0)):
      pos=[(x+.5)*.0174+((xx+.5)/6-.5)*.0116,(y+.5)*.0174+((yy+.5)/6-.5)*.0116,-.0001]
      ET.SubElement(b,'geom',type='box',pos=fmt(pos),size=fmt([.0116/12,.0116/12,.0001]),rgba='0 0 0 1',group='4',contype='0',conaffinity='0')
  optical=root.find(".//body[@name='zed_left_camera_frame_optical']")
  ET.SubElement(optical,'camera',name='calibration_view',quat='0 1 0 0',fovy=str(np.degrees(2*np.arctan(1160/(2*self.k[1,1])))))
  self.scene=out/'scene.xml';tree.write(self.scene)
  self.m=mujoco.MjModel.from_xml_path(str(self.scene));self.d=mujoco.MjData(self.m)
  self.mi=[self.m.joint(n).qposadr[0] for n in CONTROLLED_JOINTS]
  self.geom_board=self.m.geom('board_backing').id
  self.camera_body=self.m.body('zed_camera_center').id
  moving_names=CONTROLLED_JOINTS[1:] if include_left else RIGHT_ARM_JOINTS
  right_jids={self.m.joint(n).id for n in moving_names}
  def active_right(b):
   while b:
    if self.m.body_jntnum[b] and int(self.m.body_jntadr[b]) in right_jids:return True
    b=int(self.m.body_parentid[b])
   return False
  geoms=[i for i in range(self.m.ngeom) if self.m.geom_group[i] in (0,1,3)]
  self.pairs=[]
  # Reuse only documented arm seam exclusions for unchanged arm link meshes.
  cfg=yaml.safe_load((ROOT/'config/control/self_collision_proxies.v2.yaml').read_text())
  frames={r['name']:r['frame'] for r in cfg['links']}
  seam_reasons={'approved arm near-neighbor seam exclusion',
                'fixed gripper housing and its two-joint-neighbor forearm',
                'gripper and its directly connected wrist housings'}
  seam_pairs={frozenset([frames[r['first']],frames[r['second']]]) for r in cfg['excluded_pairs'] if r['reason'] in seam_reasons}
  for ii,g1 in enumerate(geoms):
   for g2 in geoms[ii+1:]:
    b1,b2=map(int,self.m.geom_bodyid[[g1,g2]])
    if frozenset([self.m.body(b1).name,self.m.body(b2).name]) in seam_pairs:continue
    if not(active_right(b1) or active_right(b2)):continue
    w1,w2=map(int,self.m.body_weldid[[b1,b2]])
    if w1==w2:continue
    names=[self.m.body(b).name for b in (b1,b2)]
    right_housings=any(all((f'SLDASM{side}_' in name or name.startswith(f'robot{side}_')) for name in names) for side in (1,2))
    if right_housings:
     chain=[];current=w1
     while current:
      chain.append(current);current=int(self.m.body_weldid[self.m.body_parentid[current]])
     current=w2;steps=0
     while current and current not in chain:
      steps+=1;current=int(self.m.body_weldid[self.m.body_parentid[current]])
     if current in chain and steps+chain.index(current)<=2:continue
    if g1!=self.geom_board and g2!=self.geom_board and not any(name.startswith('zed_') for name in names):
     pw1=int(self.m.body_weldid[self.m.body_parentid[w1]]);pw2=int(self.m.body_weldid[self.m.body_parentid[w2]])
     if w1==pw2 or w2==pw1:continue
    self.pairs.append((g1,g2))
  self.rng=np.random.default_rng(40087)
  self.pair_indices=np.asarray(self.pairs,dtype=int)
 def fk(self,a):
  q=self.q0.copy();q[self.ai]=a;pin.framesForwardKinematics(self.pm,self.pd,q);return self.pd.oMf[self.hand].homogeneous.copy()
 def update(self,a):
  q=self.q0.copy();q[self.ai]=a;self.d.qpos[:]=0;self.d.qpos[self.mi]=q[self.pi];mujoco.mj_forward(self.m,self.d)
 def distance(self,data,g1,g2,limit=.1):
  # Independently certify positive AABB separation before narrow-phase queries.
  # MuJoCo can return zero for separated mesh/box pairs near GJK degeneracies.
  finite=all(self.m.geom_type[g]!=mujoco.mjtGeom.mjGEOM_PLANE for g in (g1,g2))
  bound=0.
  if finite:
   boxes=[]
   for g in (g1,g2):
    r=data.geom_xmat[g].reshape(3,3);local=self.m.geom_aabb[g]
    center=data.geom_xpos[g]+r@local[:3];half=np.abs(r)@local[3:]
    boxes.append((center,half))
   bound=float(np.linalg.norm(np.maximum(0,np.abs(boxes[0][0]-boxes[1][0])-boxes[0][1]-boxes[1][1])))
   if bound>=.015:return min(limit,bound)
  distance=float(mujoco.mj_geomDistance(self.m,data,g1,g2,limit,None))
  return max(distance,bound)
 def clearance(self,a):
  self.update(a)
  # Vectorized version of the same positive-AABB certificate used by distance().
  # Mesh/plane or near pairs still take the existing narrow-phase query.
  rot=self.d.geom_xmat.reshape(-1,3,3);local=self.m.geom_aabb
  centers=self.d.geom_xpos+np.einsum('nij,nj->ni',rot,local[:,:3])
  halves=np.einsum('nij,nj->ni',np.abs(rot),local[:,3:])
  first,second=self.pair_indices.T
  bounds=np.linalg.norm(np.maximum(0,np.abs(centers[first]-centers[second])-halves[first]-halves[second]),axis=1)
  planes=(self.m.geom_type[first]==mujoco.mjtGeom.mjGEOM_PLANE)|(self.m.geom_type[second]==mujoco.mjtGeom.mjGEOM_PLANE)
  best=.1;pair=None
  for i,(g1,g2) in enumerate(self.pairs):
   dist=min(best,bounds[i]) if bounds[i]>=.015 and not planes[i] else self.distance(self.d,g1,g2,best)
   if dist<best:best=float(dist);pair=(g1,g2)
  names=None if pair is None else [self.m.body(int(self.m.geom_bodyid[g])).name for g in pair]
  return best,names
 def visibility(self,a):
  bw=self.fk(a)@self.hb;cb=np.linalg.inv(self.camera)@bw
  obj=np.array([[x,y,0] for x in [0,.087] for y in [0,.087]])
  xyz=obj@cb[:3,:3].T+cb[:3,3];uv=xyz@self.k.T;uv=uv[:,:2]/uv[:,2:]
  margin=float(np.min(np.c_[uv[:,0],1920-uv[:,0],uv[:,1],1080-uv[:,1]]))
  center=cb@np.array([.0435,.0435,0,1]);facing=float(cb[:3,2]@center[:3]/np.linalg.norm(center[:3]))
  return {'margin_px':margin,'facing_cosine':facing,'distance_m':float(np.linalg.norm(center[:3])),'projected_corners_px':uv.tolist(),'T_camera_from_board':cb.tolist()}

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--snapshot',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
 args=p.parse_args();s=Study(json.loads(args.snapshot.read_text()),args.output)
 rows=[];visible=0
 for i in range(60000):
  a=s.rng.uniform(s.lower,s.upper) if i%2 else np.clip(s.q0[s.ai]+s.rng.normal(0,1.3,7),s.lower+1e-6,s.upper-1e-6)
  view=s.visibility(a)
  if view['margin_px']<70 or view['facing_cosine']<.65 or not .3<view['distance_m']<.85:continue
  if np.array(view['T_camera_from_board'])[2,3]<.25:continue
  visible+=1;clear,pair=s.clearance(a)
  if clear<.015:continue
  rows.append({'right_arm_q':a.tolist(),'minimum_model_clearance_m':clear,**view})
  if len(rows)>=200:break
 result={'simulation_only':True,'hardware_execution_authorized':False,'snapshot_source':str(args.snapshot),
  'board_attachment':'Approximate from earlier photo; NOT measured','T_hand_from_board_assumed':s.hb.tolist(),
  'camera_world_nominal':s.camera.tolist(),'initial_q':s.q18.tolist(),'poses':rows,'collision_pair_count':len(s.pairs),
  'environment':'Floor only; actual desks/cables/people not modeled','path_validated':False,
  'random_samples':i+1,'in_image_candidates':visible}
 (args.output/'candidates.json').write_text(json.dumps(result,indent=2));print('Saved',len(rows),'candidates',flush=True)
if __name__=='__main__':main()
