#!/usr/bin/env python3
"""Render actual offline simulated joint feedback and validate synthetic ChArUco views."""
import json,sys,math
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'src'))
import av,cv2,mujoco,numpy as np,pinocchio as pin
from scipy.spatial.transform import Rotation
from PIL import Image,ImageDraw,ImageFont
from policy_sim.calibration.charuco import detect
from policy_sim.calibration.handeye import solve,summarize,inverse
from policy_sim.control.robot import CONTROLLED_JOINTS
OUT=ROOT/'runs/camera-calibration-40deg/route'
r=json.loads((OUT/'route.json').read_text());phy=json.loads((OUT/'physics_report.json').read_text());tr=np.load(OUT/'physics_trace.npz')
m=mujoco.MjModel.from_xml_path(str(OUT/'scene.xml'));d=mujoco.MjData(m)
k=np.array(json.loads((ROOT/'config/camera/zed_view_study.json').read_text())['K'])
cfg=json.loads((ROOT/'config/camera/charuco_87mm.json').read_text())
opt=mujoco.MjvOption();opt.geomgroup[3:5]=1
font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',22);small=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',17)
pmodel=pin.buildModelFromUrdf(str(ROOT/'urdf/maver_x4_gp80_zed_40deg.urdf'));pd=pmodel.createData();pi=[pmodel.joints[pmodel.getJointId(n)].idx_q for n in CONTROLLED_JOINTS]
mi=[m.joint(n).qposadr[0] for n in CONTROLLED_JOINTS]

def set_frame(idx):
 d.qpos[:]=tr['qpos'][idx];mujoco.mj_forward(m,d)

def render_camera(renderer,scale):
 renderer.update_scene(d,camera='calibration_view',scene_option=opt);renderer.scene.flags[mujoco.mjtRndFlag.mjRND_SHADOW]=False
 raw=renderer.render();rw,rh=2000//scale,1160//scale
 mx,my=np.meshgrid(np.arange(1920//scale,dtype=np.float32)+rw/2-k[0,2]/scale,np.arange(1080//scale,dtype=np.float32)+rh/2-k[1,2]/scale)
 return cv2.remap(raw,mx,my,cv2.INTER_LINEAR)

reports=[];gs=[];cs=[];crops=[]
with mujoco.Renderer(m,height=1160,width=2000) as renderer:
 for stop in phy['dwells']:
  idx=int(np.argmin(abs(tr['time']-stop['time_s'])));set_frame(idx)
  rgb=render_camera(renderer,1);res,annotated=detect(cv2.cvtColor(rgb,cv2.COLOR_RGB2BGR),cfg,k,np.zeros(5));res['pose_id']=stop['pose_id']
  reports.append(res);cv2.imwrite(str(OUT/f"pose_{stop['pose_id']:02}_camera.png"),annotated)
  if not res['accepted']:raise RuntimeError(f"Pose {stop['pose_id']}: board detection failed")
  q=pin.neutral(pmodel);q[pi]=tr['q'][idx];pin.framesForwardKinematics(pmodel,pd,q)
  mount=pd.oMf[pmodel.getFrameId('maver_x4_asm_URDF_link_1')];hand=pd.oMf[pmodel.getFrameId('maver_x4_asm_mount2')]
  gs.append((mount.inverse()*hand).homogeneous.copy());cs.append(res['T_camera_from_board'])
  pts=np.array(res['image_points']);lo=np.maximum(0,np.floor(pts.min(0)-30).astype(int));hi=np.minimum([1920,1080],np.ceil(pts.max(0)+30).astype(int))
  crops.append(Image.fromarray(rgb[lo[1]:hi[1],lo[0]:hi[0]]).resize((220,200)))
  print('Detected',stop['pose_id'],res['corners'],'corners, rms',res['reprojection_rms_px'],flush=True)
contact=Image.new('RGB',(6*240,5*240+55),'#111c2c');draw=ImageDraw.Draw(contact)
draw.text((15,15),'30 SIMULATED board views | actual servo feedback | 87 mm ChArUco print',font=font,fill='white')
for i,crop in enumerate(crops):
 x=(i%6)*240;y=55+(i//6)*240;contact.paste(crop,(x+10,y));draw.text((x+10,y+204),f"{i+1:02}: {reports[i]['corners']} corners / {reports[i]['reprojection_rms_px']:.2f}px",font=small,fill='white')
contact.save(OUT/'calibration_30_pose_views.jpg',quality=95)
g=np.array(gs);c=np.array(cs);fit=np.array([i%5!=4 for i in range(30)])
sol=solve(g[fit],c[fit]);x=np.array(sol['T_mount_from_camera']);y=np.array(sol['T_hand_from_board']);sol['held_out']=summarize(g[~fit],c[~fit],x,y);sol['synthetic_images_only']=True
nominal_camera=(pd.oMf[pmodel.getFrameId('maver_x4_asm_URDF_link_1')].inverse()*pd.oMf[pmodel.getFrameId('zed_left_camera_frame_optical')]).homogeneous
for label,estimated,truth in [('camera',x,nominal_camera),('board',y,np.array(r['T_hand_from_board_assumed']))]:
 error=inverse(truth)@estimated
 sol[label+'_ground_truth_error']={'translation_mm':float(np.linalg.norm(error[:3,3])*1000),'rotation_deg':float(np.degrees(np.linalg.norm(Rotation.from_matrix(error[:3,:3]).as_rotvec())))}
(OUT/'synthetic_handeye_check.json').write_text(json.dumps(sol,indent=2));(OUT/'rendered_board_detection.json').write_text(json.dumps(reports,indent=2))
np.savez_compressed(OUT/'rendered_synthetic_pairs.npz',g=g,c=c,fit=fit)
# Playback uses simulated measured qpos, with 6x time compression explicitly labelled.
rate=10;speed=6.;container=av.open(str(OUT/'calibration_simulation_6x.mp4'),'w');stream=container.add_stream('libx264',rate=rate);stream.width=1280;stream.height=720;stream.pix_fmt='yuv420p';stream.options={'crf':'23'}
overview=mujoco.MjvCamera();overview.lookat[:]=[.28,0,.96];overview.distance=2.45;overview.azimuth=130;overview.elevation=-18
with mujoco.Renderer(m,height=580,width=1000) as renderer:
 for count,time in enumerate(np.arange(0,tr['time'][-1],speed/rate)):
  idx=min(int(np.searchsorted(tr['time'],time)),len(tr['time'])-1);set_frame(idx)
  rgb=render_camera(renderer,2)
  renderer.update_scene(d,camera=overview,scene_option=opt);renderer.scene.flags[mujoco.mjtRndFlag.mjRND_SHADOW]=False;external=renderer.render()
  panel=Image.new('RGB',(1280,720),'#111c2c');dr=ImageDraw.Draw(panel)
  dr.text((18,14),'40 deg short mount | SIMULATION ONLY | 6x playback',font=font,fill='white')
  label=f"Pose {tr['pose'][idx]:02}" if tr['pose'][idx] else 'Approach / return'
  dr.text((18,47),f"{label} | {'settling / capture' if tr['hold'][idx] else 'moving'} | simulation t={time:.1f}s",font=small,fill='#a9d8e9')
  panel.paste(Image.fromarray(external).resize((640,371)),(0,85));panel.paste(Image.fromarray(rgb).resize((640,360)),(640,85))
  dr.text((18,472),'Third person: camera + short support + right-hand board',font=small,fill='white')
  dr.text((665,472),'Simulated ZED left view',font=small,fill='white')
  dr.text((18,525),'Position-servo simulation; deployed MIT controller not validated by this video.',font=small,fill='#ffcf85')
  dr.text((18,558),'Saved starting joints. Board attachment approximate; room obstacles not modeled.',font=small,fill='#ffcf85')
  dr.text((18,602),'No real robot motion commands sent. Robot/board clearance checked in the model.',font=small,fill='white')
  if tr['hold'][idx] and tr['pose'][idx]:
   im=crops[int(tr['pose'][idx])-1].resize((180,164));panel.paste(im,(1080,530))
  if count in (0,100,250):panel.save(OUT/f'video_preview_{count}.jpg',quality=93)
  for packet in stream.encode(av.VideoFrame.from_ndarray(np.asarray(panel),format='rgb24')):container.mux(packet)
  if count%100==0:print('Video frames',count,flush=True)
for packet in stream.encode():container.mux(packet)
container.close();print('Video complete',flush=True)
