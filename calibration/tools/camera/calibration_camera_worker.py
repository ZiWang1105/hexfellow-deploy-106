#!/usr/bin/env python3
"""Owned camera process: ZED in live mode; MuJoCo rendering in simulation."""
import argparse
import json
from pathlib import Path
import sys
import time
import cv2
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT.parent/'src'))
from hexfellow_deploy.robotics.calibration.charuco import detect


def atomic(path, value):
    tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(value,allow_nan=False));tmp.replace(path)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--mode',choices=['live','sim'],required=True)
    p.add_argument('--directory',type=Path,required=True)
    p.add_argument('--scene',type=Path)
    p.add_argument('--serial',type=int,default=32701991)
    p.add_argument('--validation-plan',type=Path,help='Capture RGB for left-hand overlay validation; no board detection')
    a=p.parse_args();a.directory.mkdir(parents=True,exist_ok=True)
    validation=None
    if a.validation_plan:
        from left_hand_validation import load_plan,resolve
        validation=load_plan(a.validation_plan)
    cfg=json.loads((ROOT/'config/camera/charuco_87mm.json').read_text())
    camera=None;renderer=None
    try:
        if a.mode=='live':
            import pyzed.sl as sl
            camera=sl.Camera();mat=sl.Mat()
            init=sl.InitParameters(camera_resolution=sl.RESOLUTION.HD1080,camera_fps=30,depth_mode=sl.DEPTH_MODE.NONE)
            init.set_from_serial_number(a.serial)
            init.optional_settings_path=str(__import__('calibration_paths').sdk_directory()/'settings')
            result=camera.open(init)
            if result!=sl.ERROR_CODE.SUCCESS:raise RuntimeError(f'ZED open: {result}')
            cam=camera.get_camera_information().camera_configuration.calibration_parameters.left_cam
            k=np.array([[cam.fx,0,cam.cx],[0,cam.fy,cam.cy],[0,0,1.]])
        elif validation:
            from validate_camera_overlay import OverlayRenderer
            fit=json.loads(resolve(validation['extrinsics_file']).read_text())
            k=np.array(validation['intrinsics']['K'])
            renderer=OverlayRenderer(a.scene,fit['T_mount_from_camera'],k,a.directory/'simulation-render',
                board_transform=fit['T_hand_from_board'])
        else:
            import mujoco
            from hexfellow_deploy.robotics.control.robot import CONTROLLED_JOINTS
            m=mujoco.MjModel.from_xml_path(str(a.scene));d=mujoco.MjData(m)
            indices=[m.joint(n).qposadr[0] for n in CONTROLLED_JOINTS]
            renderer=mujoco.Renderer(m,height=1160,width=2000)
            opt=mujoco.MjvOption();opt.geomgroup[3:5]=1
            k=np.array(json.loads((ROOT/'config/camera/zed_view_study.json').read_text())['K'])
            mx,my=np.meshgrid(np.arange(1920,dtype=np.float32)+1000-k[0,2],np.arange(1080,dtype=np.float32)+580-k[1,2])
        atomic(a.directory/'ready.json',dict(mode=a.mode,K=k.tolist(),distortion=[0]*5,serial=a.serial,
             frame='zed_left_camera_frame_optical',view='LEFT rectified',board=cfg))
        last=0.;done=None
        while not (a.directory/'stop').exists():
            request=json.loads((a.directory/'request.json').read_text()) if (a.directory/'request.json').exists() else None
            if a.mode=='live':
                if camera.grab()!=sl.ERROR_CODE.SUCCESS:continue
                before=time.monotonic();current=camera.get_timestamp(sl.TIME_REFERENCE.CURRENT).get_nanoseconds();after=time.monotonic()
                stamp=camera.get_timestamp(sl.TIME_REFERENCE.IMAGE).get_nanoseconds()
                frame_time=(before+after)/2-(current-stamp)/1e9;uncertainty=(after-before)/2
                if time.monotonic()-last<.2:continue
                camera.retrieve_image(mat,sl.VIEW.LEFT);bgr=cv2.cvtColor(mat.get_data(),cv2.COLOR_BGRA2BGR)
            else:
                if request is None or request['id']==done:time.sleep(.02);continue
                if time.monotonic()<request['after']+.2:time.sleep(.02);continue
                if validation:
                    rgb,_=renderer.render(request['q'],'left')
                    bgr=cv2.cvtColor(rgb,cv2.COLOR_RGB2BGR)
                else:
                    d.qpos[indices]=np.array(request['q'])[3:];mujoco.mj_forward(m,d)
                    renderer.update_scene(d,camera='calibration_view',scene_option=opt)
                    renderer.scene.flags[mujoco.mjtRndFlag.mjRND_SHADOW]=False
                    bgr=cv2.cvtColor(cv2.remap(renderer.render(),mx,my,cv2.INTER_LINEAR),cv2.COLOR_RGB2BGR)
                frame_time=time.monotonic();stamp=None;uncertainty=0.
            last=time.monotonic()
            if validation:
                detection=dict(accepted=True,purpose='left_hand_overlay',board_detection_required=False)
                annotated=bgr
            else:detection,annotated=detect(bgr,cfg,k,np.zeros(5))
            preview=a.directory/'preview.tmp.jpg';cv2.imwrite(str(preview),cv2.resize(annotated,(960,540)));preview.replace(a.directory/'preview.jpg')
            atomic(a.directory/'status.json',dict(at=time.monotonic(),image_time=frame_time,**detection))
            if request and request['id']!=done and frame_time>request['after']+.2:
                folder=a.directory/str(int(request['id']));folder.mkdir(exist_ok=False)
                cv2.imwrite(str(folder/'left.png'),bgr)
                atomic(folder/'result.json',dict(detection,image_time=frame_time,image_timestamp_ns=stamp,
                    clock_mapping_uncertainty_s=uncertainty,simulation_only=a.mode=='sim'))
                done=request['id']
    except Exception as error:
        atomic(a.directory/'error.json',dict(error=f'{type(error).__name__}: {error}'))
        raise
    finally:
        if camera is not None:camera.close()
        if renderer is not None:renderer.close()


if __name__=='__main__':main()
