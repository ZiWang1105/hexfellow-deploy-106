#!/usr/bin/env python3
"""Read-only ZED capture and board visibility check; never connects to the robot."""
import argparse
import json
from pathlib import Path
import sys
import time
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT.parent/'src'))
import cv2
import numpy as np
import pyzed.sl as sl
from hexfellow_deploy.robotics.calibration.charuco import detect


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--board',type=Path,default=ROOT/'config/camera/charuco_87mm.json')
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--seconds',type=float,default=10)
    parser.add_argument('--serial',type=int,default=32701991)
    args=parser.parse_args()
    if args.seconds <= 0: parser.error('--seconds must be positive')
    args.output.mkdir(parents=True,exist_ok=False)
    config=json.loads(args.board.read_text())
    cam=sl.Camera(); mat=sl.Mat()
    init=sl.InitParameters(camera_resolution=sl.RESOLUTION.HD1080,camera_fps=30,
        depth_mode=sl.DEPTH_MODE.NONE,coordinate_units=sl.UNIT.METER)
    init.set_from_serial_number(args.serial)
    init.optional_settings_path=str(__import__('calibration_paths').sdk_directory()/'settings')
    err=cam.open(init)
    if err != sl.ERROR_CODE.SUCCESS: raise RuntimeError(f'Camera open: {err}')
    frames=0; detections=[]; best=-1; last_detect=-1.; start=time.monotonic()
    try:
        info=cam.get_camera_information(); left=info.camera_configuration.calibration_parameters.left_cam
        k=np.array([[left.fx,0,left.cx],[0,left.fy,left.cy],[0,0,1.]])
        # VIEW.LEFT is rectified: use rectified intrinsics and zero distortion.
        d=np.zeros(5)
        meta={'serial':info.serial_number,'resolution':[1920,1080],
              'view':'LEFT rectified','optical_axes':'x right, y down, z forward',
              'K':k.tolist(),'distortion':d.tolist(),'board':config,'sdk_version':sl.Camera.get_sdk_version()}
        cam.set_camera_settings(sl.VIDEO_SETTINGS.AEC_AGC,1)
        while time.monotonic()-start < args.seconds:
            if cam.grab() != sl.ERROR_CODE.SUCCESS: continue
            frames+=1
            if time.monotonic()-last_detect < .5: continue
            last_detect=time.monotonic()
            cam.retrieve_image(mat,sl.VIEW.LEFT)
            image=cv2.cvtColor(mat.get_data(),cv2.COLOR_BGRA2BGR)
            row,annotated=detect(image,config,k,d)
            row['image_timestamp_ns']=cam.get_timestamp(sl.TIME_REFERENCE.IMAGE).get_nanoseconds()
            row['host_monotonic_s']=time.monotonic()
            detections.append(row)
            cv2.imwrite(str(args.output/'latest.png'),annotated)
            if row['corners'] > best:
                best=row['corners']; meta['best_detection']=row
                cv2.imwrite(str(args.output/'best.png'),annotated)
                cv2.imwrite(str(args.output/'best_raw.png'),image)
            print(json.dumps({key:row[key] for key in ('markers','corners','accepted','reason')}),flush=True)
        meta.update(frames=frames,detections=detections,any_board_accepted=any(r['accepted'] for r in detections),
                    motion_commands_sent=0,elapsed_s=time.monotonic()-start)
        (args.output/'report.json').write_text(json.dumps(meta,indent=2))
    finally: cam.close()
    return 0

if __name__=='__main__': raise SystemExit(main())
