#!/usr/bin/env python3
"""Independent render-vs-pinhole check for the view study's overscan projection."""
import json, math
from pathlib import Path
import cv2
import mujoco
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
k=np.array(json.loads((ROOT/'config/camera/zed_view_study.json').read_text())['K'])
w,h=2000,1160
fovy=np.degrees(2*np.arctan(h/(2*k[1,1])))
xml=f'''<mujoco><visual><global offwidth="{w}" offheight="{h}"/><headlight ambient="1 1 1" diffuse="0 0 0"/></visual><worldbody><camera name="test" pos="0 0 0" fovy="{fovy}"/><geom type="sphere" size=".007" pos=".2 .1 -1" rgba="1 0 0 1"/></worldbody></mujoco>'''
m=mujoco.MjModel.from_xml_string(xml);d=mujoco.MjData(m);mujoco.mj_forward(m,d)
with mujoco.Renderer(m,height=h,width=w) as r:
 r.update_scene(d,camera='test');raw=r.render()
mx,my=np.meshgrid(np.arange(1920,dtype=np.float32)+w/2-k[0,2],np.arange(1080,dtype=np.float32)+h/2-k[1,2])
im=cv2.remap(raw,mx,my,cv2.INTER_LINEAR)
y,x=np.where((im[:,:,0]>100)&(im[:,:,1]<30)&(im[:,:,2]<30))
actual=np.array([x.mean(),y.mean()]);expected=np.array([k[0,2]+.2*k[0,0],k[1,2]-.1*k[1,1]])
error=float(np.linalg.norm(actual-expected))
report={'expected_pixel':expected.tolist(),'rendered_centroid_pixel':actual.tolist(),'error_px':error,'passed':error<1.5}
print(json.dumps(report,indent=2))
assert report['passed']
