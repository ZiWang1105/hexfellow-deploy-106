"""Validated left-hand paths and offline comparisons for the operator workflow.

Geometry/rendering run in a subprocess, outside the deployment control loop.
This module contains no robot SDK calls.
"""
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import numpy as np
from calibration_path import LocalChecker, PathChecker, ROOT
from calibration_outcomes import InvalidCalibrationPose


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def resolve(path):
    p=Path(path)
    return p if p.is_absolute() else ROOT/p


def load_plan(path):
    plan=json.loads(Path(path).read_text())
    if plan.get('schema_version')!=1 or plan.get('capture_side')!='left':
        raise ValueError('Expected the reviewed left-hand validation plan')
    for key in ('extrinsics_file','robot_urdf','simulation_plan','simulation_report'):
        hash_key='extrinsics_sha256' if key=='extrinsics_file' else key+'_sha256'
        if sha(resolve(plan[key]))!=plan[hash_key]:
            raise ValueError(f'Validation {key} changed; regenerate/review the simulation')
    evidence=json.loads(resolve(plan['simulation_report']).read_text())
    original=json.loads(resolve(plan['simulation_plan']).read_text())
    if (not evidence['passed'] or not evidence['simulation_only'] or evidence['hardware_commands_sent']!=0
            or not evidence['returned_to_default'] or evidence['plan_sha256']!=plan['simulation_plan_sha256']):
        raise ValueError('Missing passing simulation evidence for this route')
    if plan['initial_q']!=original['initial_q'] or plan['poses']!=original['poses']:
        raise ValueError('Validation targets differ from the simulated targets')
    if len(plan['poses'])!=3 or [p['pose_id'] for p in plan['poses']]!=[1,2,3]:
        raise ValueError('Expected exactly three reviewed validation poses')
    initial=np.array(plan['initial_q'],float)
    for pose in plan['poses']:
        q=np.asarray(pose['q'],float)
        if q.shape!=(18,) or not np.all(np.isfinite(q)) or not np.array_equal(q[np.r_[:4,11:18]],initial[np.r_[:4,11:18]]):
            raise ValueError('Validation route may only change left-arm joints')
    return plan


class LeftLocalChecker(LocalChecker):
    def __init__(self,output,plan_path):
        from validate_calibration_poses import Study
        self.plan_document=load_plan(plan_path)
        fit=json.loads(resolve(self.plan_document['extrinsics_file']).read_text())
        self.study=Study({'q':self.plan_document['initial_q']},output,
                         attachment={'T_hand_from_board':fit['T_hand_from_board']},include_left=True)
        self.selected=None

    def _plan(self,q,pose_id,bounds=None,default_target=None):
        q=np.asarray(q,float)
        if pose_id==-2:return self.check_full_default(q,default_target)
        if pose_id==-1:return self.return_default(q,default_target,bounds)
        if pose_id not in (1,2,3):raise ValueError('Unknown left-hand validation pose')
        self.check_pose(q,bounds)  # Unsafe current state is never a skippable target.
        if abs(q[3]-.300)>.010:
            raise ValueError('Left-hand validation starts at the 300 mm default lift pose')
        target=q.copy()
        target[4:11]=self.plan_document['poses'][pose_id-1]['q'][4:11]
        try:
            gap=self.edge_full(q,target,bounds)
        except ValueError as error:
            raise InvalidCalibrationPose('No checked left-hand approach: '+str(error),pose_id=pose_id) from error
        return dict(pose_id=pose_id,origin=q.tolist(),targets=[target.tolist()],
            minimum_sampled_clearance_m=gap,moving_arm='left',
            geometry_scope='Robot, fitted right-board attachment, nominal camera support and floor; room/cables not modeled')

    def plan(self,q,pose_id,bounds=None,default_target=None,reference_origin=None):
        if pose_id==-2:
            # The initial upper-lift retreat is handled by the existing default
            # controller. Do not apply normal lift bounds to that initial state.
            result=self.check_full_default(q,default_target)
            if reference_origin is not None:
                self.check_full_default(reference_origin,default_target)
                result['reference_origin']=list(reference_origin)
            return result
        return super().plan(q,pose_id,bounds,default_target,reference_origin)

    def compare(self,dataset_path,output):
        import cv2
        from validate_camera_overlay import OverlayRenderer,write_comparison
        data=json.loads(Path(dataset_path).read_text())
        if data['camera']['view']!='LEFT rectified':raise ValueError('Expected rectified LEFT captures')
        fit_path=resolve(self.plan_document['extrinsics_file'])
        if sha(fit_path)!=self.plan_document['extrinsics_sha256']:raise ValueError('Extrinsics changed during capture')
        fit=json.loads(fit_path.read_text());output=Path(output);output.mkdir(exist_ok=True)
        reports=[];renderer=None
        try:
            for row in data['observations']:
                if row['q']!=row['pairing']['q']:raise ValueError('Capture lost its measured-joint pairing')
                image=cv2.imread(row['image'])
                if image is None:raise ValueError('Saved capture cannot be read')
                if renderer is None:
                    renderer=OverlayRenderer(self.study.scene,fit['T_mount_from_camera'],data['camera']['K'],
                        output/'render',image.shape[1],image.shape[0],fit['T_hand_from_board'])
                destination=output/f'pose-{row["pose_id"]:02d}'
                report=write_comparison(renderer,row,destination,'left',whole_arm=True)
                report.update(simulation_only=data['simulation_only'],extrinsics_sha256=sha(fit_path),
                              extrinsics_file=str(fit_path),mapping_sha256=data['mapping_sha256'])
                (destination/'comparison.json').write_text(json.dumps(report,indent=2))
                reports.append(dict(pose_id=row['pose_id'],comparison=str(destination/'comparison.png'),
                    visible_simulated_hand_pixels=report['simulated_visible_pixels'],
                    status='visual_review_required',quantitative_overlap=None))
        finally:
            if renderer is not None:renderer.close()
        summary=dict(status='overlays_ready' if reports else 'no_eligible_captures',
            captures=len(reports),comparisons=reports,ineligible=data['ineligible'],
            simulation_only=data['simulation_only'],image_registration_applied=False,
            interpretation='Inspect housing/wrist overlap. No independent real-hand annotation, so no numerical overlap score. This does not refit or approve the extrinsics.')
        (output/'summary.json').write_text(json.dumps(summary,indent=2))
        from html import escape
        parts=['<!doctype html><meta charset="utf-8"><title>Left-hand validation</title>',
            '<style>body{font:18px system-ui;max-width:1400px;margin:40px auto;padding:0 20px}img{width:100%}</style>',
            '<h1>Left-hand camera validation</h1>',
            '<p>Real image / simulation at measured joints / predicted outline. No image alignment or extrinsic refitting.</p>',
            '<p>'+('SIMULATION TEST — not physical validation.' if data['simulation_only'] else 'Real ZED captures paired with measured robot joints.')+'</p>']
        for row in reports:
            n=row['pose_id'];parts.append(f'<h2>Pose {n}</h2><a href="pose-{n:02d}/comparison.png"><img src="pose-{n:02d}/comparison.png" alt="Pose {n} comparison"></a>')
        for row in data['ineligible']:
            parts.append(f'<p>Pose {row["pose_id"]}: invalid — {escape(str(row["reason"]))}</p>')
        parts.append('<p>'+escape(summary['interpretation'])+'</p>')
        (output/'index.html').write_text('\n'.join(parts))
        return summary


class LeftPathChecker(PathChecker):
    def __init__(self,output,plan_path):
        self.process=subprocess.Popen([sys.executable,str(Path(__file__)),str(output),str(plan_path)],
            stdin=subprocess.PIPE,stdout=subprocess.PIPE,text=True,bufsize=1)
        while True:
            line=self.process.stdout.readline()
            if not line:raise RuntimeError('Left-hand geometry worker failed to initialize')
            if line.startswith('READY '):self.scene=Path(line[6:].strip());break
        self.study=type('Scene',(),{'scene':self.scene})()

    def compare(self,dataset,output):
        return self.request(dict(action='compare',dataset=str(dataset),output=str(output)))


if __name__=='__main__':
    sys.path.insert(0,str(ROOT/'src'))
    checker=LeftLocalChecker(Path(sys.argv[1]),Path(sys.argv[2]))
    print('READY '+str(checker.study.scene),flush=True)
    for line in sys.stdin:
        try:
            request=json.loads(line)
            if request.get('action')=='compare':result=checker.compare(request['dataset'],request['output'])
            elif request.get('action')=='check-pose':result=checker.check_pose(request['q'],request.get('bounds'))
            else:result=checker.plan(request['q'],request['pose_id'],request.get('bounds'),request.get('default_target'),request.get('reference_origin'))
        except InvalidCalibrationPose as error:
            result=dict(error=str(error),error_kind='invalid_pose',pose_id=error.pose_id)
        except Exception as error:result=dict(error=str(error),error_kind='failure')
        print(json.dumps(result),flush=True)
