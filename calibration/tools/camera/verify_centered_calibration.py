#!/usr/bin/env python3
"""Render and check the centered route offline; sends no hardware commands."""
import argparse,json,sys
from pathlib import Path
import numpy as np,cv2,mujoco
from PIL import Image,ImageDraw
from calibration_path import LocalChecker,ROOT
sys.path.insert(0,str(ROOT/"src"))
from hexfellow_deploy.robotics.calibration.charuco import detect
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--output',type=Path,default=ROOT/'runs/centered-calibration')
parser.add_argument('--snapshot',type=Path,default=ROOT/'runs/live-calibration/20260930T044638.695810Z/camera/request.json')
parser.add_argument('--attachment',type=Path)
parser.add_argument('--candidates',type=Path)
args=parser.parse_args()
OUT=args.output;OUT.mkdir(parents=True,exist_ok=True)
q=np.array(json.loads(args.snapshot.read_text())['q'])
route=json.loads((ROOT/'artifacts/reference/camera-calibration/initial-right-route.json').read_text())
checker=LocalChecker(OUT/'validated',route,args.attachment or OUT/'attachment.json',args.candidates or OUT/'candidates.json')
checker.select_route(q)
s=checker.study;paths=[];results=[];current=q.copy()
for i in range(30):
 p=checker.plan(current,i+1);paths.append(p);current=np.array(p['targets'][-1]);print('Path checked',i+1,flush=True)
(OUT/'paths.json').write_text(json.dumps(paths,indent=2))
opt=mujoco.MjvOption();opt.geomgroup[3:5]=1
mx,my=np.meshgrid(np.arange(1920,dtype=np.float32)+1000-s.k[0,2],np.arange(1080,dtype=np.float32)+580-s.k[1,2])
sheet=Image.new('RGB',(1920,6*238),(20,25,32));draw=ImageDraw.Draw(sheet)
with mujoco.Renderer(s.m,height=1160,width=2000) as renderer:
 for i,a in enumerate(checker.selected):
  s.update(a);renderer.update_scene(s.d,camera='calibration_view',scene_option=opt);renderer.scene.flags[mujoco.mjtRndFlag.mjRND_SHADOW]=False
  rgb=cv2.remap(renderer.render(),mx,my,cv2.INTER_LINEAR);result,annotation=detect(cv2.cvtColor(rgb,cv2.COLOR_RGB2BGR),s.boardcfg,s.k,np.zeros(5))
  assert result['accepted'] and result['corners']==16,(i,result)
  v=s.visibility(a);uv=np.array(v['projected_corners_px']);center=uv.mean(0)
  assert abs(center[0]-960)<160 and abs(center[1]-540)<120
  result.update(pose_id=i+1,center_px=center.tolist(),margin_px=v['margin_px']);results.append(result)
  cv2.rectangle(annotation,(800,420),(1120,660),(0,210,255),2)
  cv2.drawMarker(annotation,(960,540),(0,210,255),cv2.MARKER_CROSS,24,2)
  cv2.imwrite(str(OUT/f'pose-{i+1:02}.jpg'),annotation)
  thumb=Image.fromarray(cv2.cvtColor(annotation,cv2.COLOR_BGR2RGB)).resize((384,216));x=i%5*384;y=i//5*238;sheet.paste(thumb,(x,y));draw.text((x+8,y+218),f'{i+1:02} | 16 corners | center ({center[0]:.0f},{center[1]:.0f})',fill='white')
sheet.save(OUT/'30-centered-views.jpg',quality=94)
# Render third-person key poses and their full camera images together.
overview=mujoco.MjvCamera();overview.lookat[:]=[.25,0,1.0];overview.distance=2.35;overview.azimuth=130;overview.elevation=-18
panel=Image.new('RGB',(1280,4*390),(20,25,32));dr=ImageDraw.Draw(panel)
with mujoco.Renderer(s.m,height=540,width=960) as renderer:
 for j,i in enumerate([0,9,19,29]):
  s.update(checker.selected[i]);renderer.update_scene(s.d,camera=overview,scene_option=opt)
  panel.paste(Image.fromarray(renderer.render()).resize((640,360)),(0,j*390))
  panel.paste(Image.open(OUT/f'pose-{i+1:02}.jpg').resize((640,360)),(640,j*390))
  dr.text((10,j*390+365),f'Pose {i+1} | modeled camera + short 40 degree mount + board | SIMULATION, estimated attachment',fill='white')
panel.save(OUT/'third-person-and-camera.jpg',quality=94)
report=dict(simulation_only=True,validation='Kinematic MuJoCo rendering and densely sampled paths; not hardware execution',poses=30,all_16_corners=True,results=results,maximum_center_offset_px=np.max(abs(np.array([r['center_px'] for r in results])-[960,540]),axis=0).tolist(),minimum_image_margin_px=min(r['margin_px'] for r in results),path_max_l1_joint_sample_step_rad=.005,path_required_clearance_m=.015,attachment_estimated=True,room_and_cables_not_modeled=True)
(OUT/'report.json').write_text(json.dumps(report,indent=2));print('VERIFIED',report['maximum_center_offset_px'],report['minimum_image_margin_px'],flush=True)
