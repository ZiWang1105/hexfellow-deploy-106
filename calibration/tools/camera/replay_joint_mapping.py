#!/usr/bin/env python3
"""Reinterpret saved encoder measurements with a revised mapping, offline only.

Original captures are immutable. Camera intrinsics/extrinsics and geometry are
unchanged. This cannot connect to hardware or fit images to each other.
"""
import argparse
import copy
import json
from pathlib import Path
import numpy as np
from validate_camera_overlay import OverlayRenderer, digest, write_comparison


def remap_q(q, original, revised, velocity=False):
    q=np.asarray(q,float)
    if q.shape!=(18,) or not np.isfinite(q).all():
        raise ValueError('Expected 18 finite measured coordinates')
    result=q.copy()
    for side,start in [('left',4),('right',11)]:
        old,new=original['arms'][side],revised['arms'][side]
        raw=np.empty(7)
        offset=0 if velocity else np.asarray(old['offsets_rad'])
        raw[old['sdk_indices']]=(q[start:start+7]-offset)*old['signs']
        offset=0 if velocity else np.asarray(new['offsets_rad'])
        result[start:start+7]=raw[new['sdk_indices']]*new['signs']+offset
    return result.tolist()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('dataset','original-mapping','revised-mapping','extrinsics','scene','output'):
        p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--whole-arm',action='store_true',help='Overlay all visible left-arm links and the gripper')
    a=p.parse_args()
    data=json.loads(a.dataset.read_text())
    if data['mapping_sha256']!=digest(a.original_mapping):
        raise ValueError('Original mapping does not match the capture provenance')
    if data['camera']['view']!='LEFT rectified' or np.any(data['camera']['distortion']):
        raise ValueError('Expected rectified LEFT frames')
    if data['extrinsics_sha256']!=digest(a.extrinsics):
        raise ValueError('Keep the original extrinsics fixed when checking joint mappings')
    old,new=[json.loads(v.read_text()) for v in (a.original_mapping,a.revised_mapping)]
    fit=json.loads(a.extrinsics.read_text())
    a.output.mkdir(parents=True,exist_ok=False)
    import cv2
    first=cv2.imread(data['observations'][0]['image'])
    renderer=OverlayRenderer(a.scene,fit['T_mount_from_camera'],data['camera']['K'],
        a.output/'render',first.shape[1],first.shape[0],fit['T_hand_from_board'])
    provenance=dict(source_dataset=str(a.dataset.resolve()),source_dataset_sha256=digest(a.dataset),
        original_mapping=str(a.original_mapping.resolve()),original_mapping_sha256=digest(a.original_mapping),
        revised_mapping=str(a.revised_mapping.resolve()),revised_mapping_sha256=digest(a.revised_mapping),
        extrinsics_file=str(a.extrinsics.resolve()),extrinsics_sha256=digest(a.extrinsics),
        method='Invert original model mapping to SDK encoders, then apply revised mapping.',
        offline_only=True,hardware_commands_sent=0,extrinsics_refitted=False)
    results=[]
    try:
        for original in data['observations']:
            if original['q']!=original['pairing']['q']:
                raise ValueError('Capture does not match measured feedback pairing')
            row=copy.deepcopy(original)
            row['q']=remap_q(original['q'],old,new)
            # Pairing records retain their original complete payload in provenance.
            row['pairing']=dict(original_pairing=original['pairing'],q=row['q'],
                remapping=provenance)
            output=a.output/f'pose-{row["pose_id"]:02d}'
            result=write_comparison(renderer,row,output,'left',whole_arm=a.whole_arm)
            result.update(provenance=provenance,original_measured_q=original['q'])
            (output/'comparison.json').write_text(json.dumps(result,indent=2))
            results.append(dict(pose_id=row['pose_id'],mask_scope=result['mask_scope'],original_left_j6_deg=float(np.degrees(original['q'][9])),
                revised_left_j6_deg=float(np.degrees(row['q'][9])),overlay=str(output/'overlay.png')))
    finally:
        renderer.close()
    (a.output/'summary.json').write_text(json.dumps(dict(**provenance,poses=results),indent=2))
    print(json.dumps(results,indent=2))


if __name__=='__main__':main()
