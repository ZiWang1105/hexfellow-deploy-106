#!/usr/bin/env python3
"""Require real ChArUco detection on rendered images before route selection."""
import json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT.parent/'src'))
import cv2,mujoco,numpy as np
from hexfellow_deploy.robotics.calibration.charuco import detect
from hexfellow_deploy.robotics.control.robot import CONTROLLED_JOINTS
p=ROOT/'runs/camera-calibration-40deg/sample'
r=json.loads((p/'candidates.json').read_text());m=mujoco.MjModel.from_xml_path(str(p/'scene.xml'));d=mujoco.MjData(m)
q=np.array(json.loads((ROOT/'runs/camera-calibration/robot-snapshot.json').read_text())['q'][3:]);mi=[m.joint(n).qposadr[0] for n in CONTROLLED_JOINTS]
k=np.array(json.loads((ROOT/'config/camera/zed_view_study.json').read_text())['K']);cfg=json.loads((ROOT/'config/camera/charuco_87mm.json').read_text())
mx,my=np.meshgrid(np.arange(1920,dtype=np.float32)+1000-k[0,2],np.arange(1080,dtype=np.float32)+580-k[1,2]);opt=mujoco.MjvOption();opt.geomgroup[3:5]=1
accepted=[];results=[]
with mujoco.Renderer(m,height=1160,width=2000) as renderer:
 for i,row in enumerate(r['poses']):
  q[-7:]=row['right_arm_q'];d.qpos[mi]=q;mujoco.mj_forward(m,d)
  renderer.update_scene(d,camera='calibration_view',scene_option=opt);renderer.scene.flags[mujoco.mjtRndFlag.mjRND_SHADOW]=False
  im=cv2.remap(renderer.render(),mx,my,cv2.INTER_LINEAR);res,_=detect(cv2.cvtColor(im,cv2.COLOR_RGB2BGR),cfg,k,np.zeros(5))
  results.append({'candidate':i,'corners':res['corners'],'accepted':res['accepted']})
  if res['accepted'] and res['corners']>=12:
   accepted.append({**row,'candidate_index':i,'rendered_detection':res})
  if i%40==0:print('Checked',i,'accepted',len(accepted),flush=True)
r['poses']=accepted;r['rendered_validation']=results;(p/'vision_candidates.json').write_text(json.dumps(r,indent=2));print('Passed',len(accepted),flush=True)
