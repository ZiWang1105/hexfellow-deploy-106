"""Third-person overlays tied to the URDF's left optical frame."""
import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont

FONT='/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'

def add_line(scene, start, end, color, width=3):
    g=scene.geoms[scene.ngeom]
    mujoco.mjv_initGeom(g,mujoco.mjtGeom.mjGEOM_LINE,np.zeros(3),np.zeros(3),np.eye(3).ravel(),np.asarray(color,dtype=float))
    mujoco.mjv_connector(g,mujoco.mjtGeom.mjGEOM_LINE,width,np.asarray(start),np.asarray(end))
    scene.ngeom+=1

def frustum_corners(position, optical, k, depth):
    # Boundary pixels of the SAME 1920 x 1080 image used for the right panel.
    uv=np.array([[0,0,1],[1920,0,1],[1920,1080,1],[0,1080,1]],dtype=float)
    return position + (np.linalg.solve(k,uv.T).T*depth)@optical.T

def overlay(scene, position, optical, k, depth):
    corners=frustum_corners(position,optical,k,depth)
    for i in range(4):
        add_line(scene,position,corners[i],[.05,.85,1,1])
        add_line(scene,corners[i],corners[(i+1)%4],[.05,.85,1,1])
    add_line(scene,position,position+optical[:,2]*depth,[1,.42,.05,1],5)

def render_pair(renderer, data, position, optical, k, rgb, out, scene_name, title, near, length, height):
    overview=mujoco.MjvCamera()
    overview.lookat[:]=[near+length*.25,0,.91]
    overview.distance=3.65 if scene_name.startswith('pool') else 2.7
    overview.azimuth=120;overview.elevation=-22
    renderer.update_scene(data,camera=overview)
    renderer.scene.flags[mujoco.mjtRndFlag.mjRND_SHADOW]=False
    depth=(height-position[2])/optical[2,2]
    # Wireframe section intersects the table centre height on the optical axis.
    # This is a visualization plane, not a maximum sensing range.
    depth=min(1.3,max(.7,float(depth)))
    overlay(renderer.scene,position,optical,k,depth)
    external=Image.fromarray(renderer.render())
    external.save(out/f'{scene_name}_mounted_third_person.png')
    close=mujoco.MjvCamera();close.lookat[:]=position-[.01,0,.035]
    close.distance=.45;close.azimuth=35;close.elevation=-12
    renderer.update_scene(data,camera=close)
    renderer.scene.flags[mujoco.mjtRndFlag.mjRND_SHADOW]=False
    # Short cone segment identifies the lens origin without covering the mount.
    overlay(renderer.scene,position,optical,k,.22)
    detail=Image.fromarray(renderer.render())
    detail.save(out/f'{scene_name}_mount_detail.png')
    panel=Image.new('RGB',(2000,850),'#111c2c');d=ImageDraw.Draw(panel)
    font=ImageFont.truetype(FONT,25);small=ImageFont.truetype(FONT,20)
    d.text((24,16),title+' | ZED + SHORT v5 bracket attached to lift URDF',font=font,fill='white')
    pitch=np.degrees(np.arcsin(-optical[2,2]))
    d.text((24,54),f'Short v5: 65 mm shelf, {pitch:.0f} deg down | cyan: image boundaries | orange: optical axis | physical pose not yet calibrated',font=small,fill='#bad5e8')
    panel.paste(external.resize((1100,638)),(0,95))
    panel.paste(Image.fromarray(rgb).resize((900,506)),(1100,95))
    panel.paste(detail.resize((400,232)),(1100,605))
    d.text((24,747),'THIRD PERSON - actual camera and bracket meshes',font=small,fill='white')
    d.text((24,817),'Cone section is illustrative; its end is not a sensing-distance limit.',font=small,fill='#b8c3ce')
    d.text((24,782),'Cone drawn from left optical frame using measured ZED intrinsics.',font=small,fill='#8cdaed')
    d.text((1518,620),'MOUNT CLOSE-UP',font=small,fill='white')
    d.text((1518,659),'Teal: short v5 mount STL',font=small,fill='#8cdaed')
    d.text((1518,692),'Grey: ZED body STL',font=small,fill='#b8c3ce')
    d.text((1518,725),'Both move with the lift.',font=small,fill='#b8c3ce')
    d.text((1120,105),'MATCHING LEFT CAMERA IMAGE',font=small,fill='white',stroke_width=1,stroke_fill='black')
    panel.save(out/f'{scene_name}_mounted_pair.jpg',quality=95)
    return {'frustum_section_depth_m':depth,'frustum_world_corners_m':frustum_corners(position,optical,k,depth).tolist()}
