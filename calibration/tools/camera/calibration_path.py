"""Checks calibration paths against the modeled robot, board and short mount."""
import json,hashlib
import numpy as np
from pathlib import Path
import subprocess, sys
from calibration_outcomes import InvalidCalibrationPose
ROOT=Path(__file__).resolve().parents[2]


def usable_nominal_view(view):
    """Planning screen only; unknown attachment errors still need real validation."""
    values=np.asarray([view['margin_px'],view['facing_cosine'],view['distance_m']],float)
    transform=np.asarray(view['T_camera_from_board'],float)
    if 'projected_corners_px' in view:
        center=np.asarray(view['projected_corners_px'],float).mean(axis=0)
        if not np.all(np.isfinite(center)) or np.any(abs(center-[960,540])>[160,120]):
            return False
    return bool(np.all(np.isfinite(values)) and np.all(np.isfinite(transform))
                and values[0]>=150 and values[1]>=.65
                and .3<values[2]<.85 and transform[2,3]>.25)


class LocalChecker:
    def __init__(self, output, route, attachment=None, candidates=None):
        from validate_calibration_poses import Study
        self.attachment=attachment
        self.candidates=candidates
        self.study=Study(json.loads((ROOT/'artifacts/reference/camera-calibration/initial-geometry-snapshot.json').read_text()),output,
                         attachment=json.loads(Path(attachment).read_text()) if attachment else None,include_left=True)
        self.route=route
        self.selected=None

    def select_route(self,q):
        # Re-evaluate images after applying the real controller's tighter limits.
        import cv2,mujoco
        from hexfellow_deploy.robotics.calibration.charuco import detect
        s=self.study
        import pinocchio as pin
        s.q0[s.pi]=np.asarray(q)[3:]
        pin.framesForwardKinematics(s.pm,s.pd,s.q0)
        s.camera=s.pd.oMf[s.cf].homogeneous.copy()
        if self.candidates:
            pool=json.loads(Path(self.candidates).read_text())
            if not self.attachment or pool.get('attachment_sha256')!=hashlib.sha256(Path(self.attachment).read_bytes()).hexdigest():
                raise ValueError('Candidate pool does not match board attachment; regenerate poses')
            candidates=pool['poses']
        elif self.attachment:
            candidates=[]
            for i in range(100000):
                target=s.rng.uniform(s.lower,s.upper) if i%2 else np.clip(q[-7:]+s.rng.normal(0,1.4,7),s.lower,s.upper)
                view=s.visibility(target)
                if not usable_nominal_view(view):continue
                if s.clearance(target)[0]>=.015:candidates.append(dict(right_arm_q=target.tolist()))
                if len(candidates)>=250:break
            (s.out/'candidates.json').write_text(json.dumps(dict(poses=candidates,attachment=str(self.attachment))))
        else:
            candidates=json.loads((ROOT/'runs/camera-calibration-40deg/sample/vision_candidates.json').read_text())['poses']
        opt=mujoco.MjvOption();opt.geomgroup[3:5]=1
        mx,my=np.meshgrid(np.arange(1920,dtype=np.float32)+1000-s.k[0,2],np.arange(1080,dtype=np.float32)+580-s.k[1,2])
        valid=[];seen=set()
        with mujoco.Renderer(s.m,height=1160,width=2000) as renderer:
            for row in candidates:
                target=np.clip(np.array(row['right_arm_q']),s.lower,s.upper)
                key=tuple(np.round(target,5))
                if key in seen:continue
                seen.add(key)
                # Clamping joints changes both board position and orientation.
                # Old candidate visibility is not evidence for this new target.
                view=s.visibility(target)
                if not usable_nominal_view(view):continue
                if s.clearance(target)[0]<.015:continue
                renderer.update_scene(s.d,camera='calibration_view',scene_option=opt)
                renderer.scene.flags[mujoco.mjtRndFlag.mjRND_SHADOW]=False
                image=cv2.cvtColor(cv2.remap(renderer.render(),mx,my,cv2.INTER_LINEAR),cv2.COLOR_RGB2BGR)
                result,_=detect(image,s.boardcfg,s.k,np.zeros(5))
                # Reserve corner margin for measured tracking / pixel quantization.
                if result['accepted'] and result['corners']==16:valid.append((target,result))
        if len(valid)<30:raise ValueError(f'Only {len(valid)} runtime-compatible visible poses; replan required')
        current=q[-7:].copy();selected=[];reports=[]
        for _ in range(30):
            choices=sorted(range(len(valid)),key=lambda i:np.linalg.norm(valid[i][0]-current))
            index=next((i for i in choices if all(np.linalg.norm(valid[i][0]-v)>.15 for v in selected)),None)
            if index is None:raise ValueError('Too few distinct runtime-compatible camera poses')
            current,result=valid.pop(index);selected.append(current);reports.append(result)
        self.selected=selected
        (s.out/'runtime-route.json').write_text(json.dumps(dict(initial_q=q.tolist(),right_arm_targets=[v.tolist() for v in selected],nominal_detections=reports,simulation_only=True),indent=2))

    def edge(self,a,b):
        n=max(2,int(np.ceil(np.sum(abs(b-a))/.005)))
        minimum=.1
        for u in np.linspace(0,1,n+1):
            gap,_=self.study.clearance(a+(b-a)*u)
            minimum=min(minimum,gap)
            if gap<.015:return False,minimum
        return True,minimum

    def return_default(self,q,target,bounds=None):
        """Both arms return to deployment default; base and lift stay fixed."""
        q=np.asarray(q);target=np.asarray(target).copy();target[:4]=q[:4]
        if bounds is not None and (np.any(target[4:]<np.asarray(bounds[0])[4:]) or np.any(target[4:]>np.asarray(bounds[1])[4:])):
            raise ValueError('Default arm pose is outside current controller bounds')
        def clear(full):
            self.study.q0[self.study.pi]=full[3:]
            return self.study.clearance(full[-7:])[0]
        def edge(a,b):
            for u in np.linspace(0,1,max(2,int(np.ceil(np.sum(abs(b-a))/.005)))+1):
                if clear(a+(b-a)*u)<.015:return False
            return True
        def arm_path(a,b,part):
            if edge(a,b):return [b]
            # Move the shoulder away from the torso before unwinding the wrist.
            # Each candidate (including approach and final lowering) is checked.
            offsets=[]
            for j in (1,0,3):
                for amount in (-.25,.25,-.5,.5,-.8,.8):
                    offset=np.zeros(7);offset[j]=amount;offsets.append(offset)
            offsets.extend(self.study.rng.normal(0,.35,7) for _ in range(80))
            for offset in offsets:
                m1=a.copy();m2=b.copy();m1[part]+=offset;m2[part]+=offset
                if bounds is not None:
                    lo,hi=map(np.asarray,bounds)
                    if any(np.any(m[4:]<lo[4:]+.025) or np.any(m[4:]>hi[4:]-.025) for m in (m1,m2)):continue
                if clear(m1)<.015 or clear(m2)<.015:continue
                if edge(a,m1) and edge(m1,m2) and edge(m2,b):return [m1,m2,b]
            return None
        if edge(q,target):waypoints=[target]
        else:
            waypoints=None
            for part in (slice(11,18),slice(4,11)):
                mid=q.copy();mid[part]=target[part]
                first=arm_path(q,mid,part)
                other=slice(4,11) if part.start==11 else slice(11,18)
                second=arm_path(mid,target,other) if first is not None else None
                if first is not None and second is not None:waypoints=first+second;break
            if waypoints is None:raise ValueError('Return to default has no collision-checked path; remain stopped')
        return dict(origin=q.tolist(),targets=[v.tolist() for v in waypoints],pose_id=-1,
                    geometry_scope='Both arms and board checked; lift/base held; room/cables not modeled')

    def check_full_default(self,q,target):
        """Check the deployment retreat and full-default straight posture segments."""
        q=np.asarray(q);target=np.asarray(target).copy();target[:3]=q[:3]
        points=[q.copy()]
        if q[3]>.417:
            retreat=q.copy();retreat[3]=.409;points.append(retreat)
        points.append(target)
        minimum=.1
        for a,b in zip(points,points[1:]):
            for u in np.linspace(0,1,max(2,int(np.ceil(np.sum(abs(b-a))/.005)))+1):
                full=a+(b-a)*u;self.study.q0[self.study.pi]=full[3:]
                gap,_=self.study.clearance(full[-7:]);minimum=min(minimum,gap)
                if gap<.015:raise ValueError('Default-pose path has less than 15 mm modeled clearance')
        self.selected=None
        return dict(origin=q.tolist(),targets=[v.tolist() for v in points[1:]],
                    pose_id=-2,minimum_sampled_clearance_m=minimum)

    def check_pose(self,q,bounds=None):
        """Certify a measured capture configuration in the existing scene."""
        q=np.asarray(q,dtype=float)
        if q.shape!=(18,) or not np.all(np.isfinite(q)):
            raise ValueError('Invalid measured capture configuration')
        if bounds is not None:
            lower,upper=map(np.asarray,bounds)
            if np.any(q[3:]<lower[3:]) or np.any(q[3:]>upper[3:]):
                raise ValueError('Measured configuration outside controller joint bounds')
        self.study.q0[self.study.pi]=q[3:]
        gap,pair=self.study.clearance(q[-7:])
        if gap<.015:
            raise ValueError(f'Measured configuration has only {1000*gap:.1f} mm modeled clearance: {pair}')
        return dict(measured_q=q.tolist(),minimum_clearance_m=float(gap),required_clearance_m=.015,
                    geometry_scope='robot, assumed board, camera support, floor; room/cables not modeled')

    def edge_full(self,a,b,bounds=None):
        a,b=map(np.asarray,(a,b));minimum=.1
        for u in np.linspace(0,1,max(2,int(np.ceil(np.sum(abs(b[3:]-a[3:]))/.005)))+1):
            report=self.check_pose(a+(b-a)*u,bounds)
            minimum=min(minimum,report['minimum_clearance_m'])
        return minimum

    def plan(self,q,pose_id,bounds=None,default_target=None,reference_origin=None):
        payload=self._plan(q,pose_id,bounds,default_target)
        if reference_origin is not None:
            reference=np.asarray(reference_origin,dtype=float)
            # Keep the controller's support reference. Check its actual starting
            # path separately from the path planned using encoder feedback.
            minimum=self.edge_full(reference,q,bounds)
            try:
                minimum=min(minimum,self.edge_full(reference,payload['targets'][0],bounds))
            except ValueError as error:
                if pose_id>0:
                    raise InvalidCalibrationPose('Retained-reference approach is not valid: '+str(error),
                                                 pose_id=pose_id) from error
                raise
            payload.update(reference_origin=reference.tolist(),
                reference_path_minimum_clearance_m=minimum,
                path_origins_checked=['measured','commanded_reference'])
        return payload

    def _plan(self,q,pose_id,bounds=None,default_target=None):
        """Recheck from actual HOLD, preserving current left arm and lift."""
        q=np.asarray(q);s=self.study;s.q0[s.pi]=q[3:]
        if pose_id==-2:return self.check_full_default(q,default_target)
        if pose_id==-1:return self.return_default(q,default_target,bounds)
        target=np.array(self.route['poses'][pose_id-1]['right_arm_q'])
        if bounds is not None:
            s.lower=np.maximum(s.lower,np.array(bounds[0])[-7:]+.15)
            s.upper=np.minimum(s.upper,np.array(bounds[1])[-7:]-.15)
            target=np.clip(target,s.lower,s.upper)
        if self.selected is None:self.select_route(q)
        target=self.selected[pose_id-1].copy()
        a=q[-7:].copy()
        if s.clearance(a)[0]<.015:raise ValueError('Current pose has less than 15 mm modeled clearance')
        if self.edge(a,target)[0]:waypoints=[target]
        else:
            waypoints=None
            # Try original intermediate nodes first, then deterministic random bridges.
            options=[np.array(v['to_right_q']) for v in self.route['segments'] if v['destination_pose'] in (None,pose_id)]
            options.extend(np.clip((a+target)/2+s.rng.normal(0,.55,7),s.lower,s.upper) for _ in range(150))
            for mid in options:
                mid=np.clip(mid,s.lower,s.upper)
                if s.clearance(mid)[0]>=.015 and self.edge(a,mid)[0] and self.edge(mid,target)[0]:
                    waypoints=[mid,target];break
            if waypoints is None:
                raise InvalidCalibrationPose('No collision-checked approach found for this pose',pose_id=pose_id)
        targets=[]
        for w in waypoints:
            full=q.copy();full[-7:]=w;targets.append(full.tolist())
        return dict(origin=q.tolist(),targets=targets,pose_id=pose_id,
                    nominal_right_q=self.selected[pose_id-1].tolist(),
                    target_joint_limit_reserve_rad=.15,
                    geometry_scope='robot, assumed board, camera support, floor; room/cables not modeled')


class PathChecker:
    def __init__(self,output,route,attachment=None,candidates=None):
        self.process=subprocess.Popen([sys.executable,str(Path(__file__)),str(output),str(attachment or ''),str(candidates or '')],
            stdin=subprocess.PIPE,stdout=subprocess.PIPE,text=True,bufsize=1)
        self.scene=None
        while True:
            line=self.process.stdout.readline()
            if not line:raise RuntimeError('Geometry worker failed to initialize')
            if line.startswith('READY '):self.scene=Path(line[6:].strip());break
        self.study=type('Scene',(),{'scene':self.scene})()
    def request(self,value):
        self.process.stdin.write(json.dumps(value)+'\n');self.process.stdin.flush()
        line=self.process.stdout.readline()
        if not line:raise ValueError('Geometry checker disconnected')
        result=json.loads(line)
        if 'error' in result:
            if result.get('error_kind')=='invalid_pose':
                raise InvalidCalibrationPose(result['error'],pose_id=result.get('pose_id'))
            raise ValueError(result['error'])
        return result

    def plan(self,q,pose_id,bounds=None,default_target=None,reference_origin=None):
        return self.request(dict(q=q,pose_id=pose_id,bounds=bounds,default_target=default_target,
                                 reference_origin=reference_origin))

    def check_pose(self,q,bounds=None):
        return self.request(dict(action='check-pose',q=q,bounds=bounds))
    def close(self):
        self.process.stdin.close()
        try:self.process.wait(timeout=3)
        except subprocess.TimeoutExpired:self.process.terminate();self.process.wait(timeout=3)


if __name__=='__main__':
    sys.path.insert(0,str(ROOT.parent/'src'))
    route=json.loads((ROOT/'artifacts/reference/camera-calibration/initial-right-route.json').read_text())
    checker=LocalChecker(Path(sys.argv[1]),route,sys.argv[2] or None if len(sys.argv)>2 else None,sys.argv[3] or None if len(sys.argv)>3 else None)
    print('READY '+str(checker.study.scene),flush=True)
    for line in sys.stdin:
        try:
            request=json.loads(line)
            if request.get('action')=='check-pose':result=checker.check_pose(request['q'],request.get('bounds'))
            else:result=checker.plan(request['q'],request['pose_id'],request.get('bounds'),request.get('default_target'),request.get('reference_origin'))
        except InvalidCalibrationPose as e:
            result={'error':str(e),'error_kind':'invalid_pose','pose_id':e.pose_id}
        except Exception as e:result={'error':str(e),'error_kind':'failure'}
        print(json.dumps(result),flush=True)
