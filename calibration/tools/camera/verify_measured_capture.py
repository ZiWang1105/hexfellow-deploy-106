#!/usr/bin/env python3
"""Simulation-only two-pose capture with an unmodeled shoulder load.

Exercises the real launcher, UI sequence, dynamics, camera worker and measured FK.
The artificial -5.58 Nm load is a regression disturbance, not a gravity estimate.
Never launches live mode or sends robot commands.
"""
import argparse
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time
import urllib.request

ROOT=Path(__file__).resolve().parents[2]


def worker(output,port,inject_invalid=False):
    sys.path.insert(0,str(ROOT/'src'))
    from policy_sim.hardware.mit_ik_replay import SimulationPlant
    import numpy as np
    original=SimulationPlant.__init__
    def initialize(plant,*args,**kwargs):
        original(plant,*args,**kwargs)
        q_index=plant.core._mj_qpos_indices[11]
        v_index=plant.core._mj_qvel_indices[11]
        def disturbance(p):
            # Smoothly introduce the load after leaving the default arm pose.
            weight=float(np.clip((p.data.qpos[q_index]+1.2)/.2,0,1))
            p.data.qfrc_applied[v_index]=-5.58*weight
        plant.substep_hook=disturbance
    SimulationPlant.__init__=initialize
    if inject_invalid:
        # Bounded simulation fault injection: exercise both no-path and arrival
        # failure through the real sequence. No production thresholds change.
        from calibration_path import PathChecker
        from calibration_runtime import CalibrationController
        from calibration_outcomes import InvalidCalibrationPose
        original_plan=PathChecker.plan
        def plan(checker,q,pose_id,*args,**kwargs):
            if pose_id==1:
                raise InvalidCalibrationPose('Simulation-injected unreachable path',pose_id=1)
            return original_plan(checker,q,pose_id,*args,**kwargs)
        PathChecker.plan=plan
        original_segment=CalibrationController._segment
        def segment(controller,*args):
            original_segment(controller,*args)
            if controller.pose_id!=3 or controller.todo:return
            gate=controller.measured_arrival;original_update=gate.update
            def update(*args):
                original_update(*args)
                if gate.observed_s>10.:
                    raise InvalidCalibrationPose('Simulation-injected arrival timeout')
                return False
            gate.update=update
        CalibrationController._segment=segment
    import calibrate_robot
    sys.argv=[str(ROOT/'tools/camera/calibrate_robot.py'),'--mode','sim',
              '--port',str(port),'--run-seconds','300','--output',str(output/'session')]
    return calibrate_robot.main()


def verify(output,port,inject_invalid=False):
    output.mkdir(parents=True,exist_ok=False)
    url=f'http://127.0.0.1:{port}'
    env=dict(os.environ,OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',MUJOCO_GL='egl')
    token=None;observations=[]
    def post(path,body):
        request=urllib.request.Request(url+path,data=json.dumps(body).encode(),
            headers={'Content-Type':'application/json','X-Calibration-Token':token})
        return json.loads(urllib.request.urlopen(request,timeout=3).read())
    with (output/'launcher.log').open('w') as log:
        command=[sys.executable,str(Path(__file__)),'--worker','--output',str(output),'--port',str(port)]
        if inject_invalid:command.append('--inject-invalid')
        process=subprocess.Popen(command,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
        try:
            deadline=time.monotonic()+45
            while True:
                try:
                    page=urllib.request.urlopen(url,timeout=2).read().decode()
                    token=re.search("const token='([^']+)'",page)[1]
                    break
                except OSError:
                    if process.poll() is not None or time.monotonic()>deadline:
                        raise RuntimeError('Simulation startup failed; inspect launcher.log')
                    time.sleep(.2)
            status=post('/heartbeat',{})
            post('/command',dict(action='start',revision=status['revision']))
            deadline=time.monotonic()+240
            while time.monotonic()<deadline:
                status=post('/heartbeat',{})
                observations.append({k:status[k] for k in ('phase','pose','target_pose','message','captured','ineligible','pose_settling')})
                if len(status['attempted'])>=(4 if inject_invalid else 2):
                    post('/command',dict(action='quit',revision=status['revision']))
                    process.wait(timeout=25)
                    break
                if 'rejected:' in status['message'] or status['failure_detail']:
                    raise RuntimeError(status['message'])
                if process.poll() is not None:raise RuntimeError('Simulation exited before expected attempts')
                time.sleep(.15)
            else:raise TimeoutError('Capture/skip simulation did not complete')
            result=json.loads((output/'session/result.json').read_text())
            dataset=json.loads((output/'session/dataset.json').read_text())
            rows=dataset['observations']
            assert process.returncode==0 and result['passed'],result
            expected=[2,4] if inject_invalid else [1,2]
            assert [r['pose_id'] for r in rows]==expected,status
            if inject_invalid:
                assert [r['pose_id'] for r in dataset['ineligible']]==[1,3]
                assert all(r['eligibility']=='invalid' and not r['accepted'] for r in dataset['ineligible'])
                assert all(r['measured_geometry']['minimum_clearance_m']>=.015 for r in dataset['ineligible'])
                assert any(r['phase']=='stopping' and r['target_pose']==3 for r in observations)
                assert any(r['phase']=='hold' and r['target_pose']==3 for r in observations)
            assert all(r['accepted'] and r['measured_geometry']['minimum_clearance_m']>=.015 for r in rows)
            assert all(r['settling']['maximum_right_offset_deg']>1.146 for r in rows),[r['settling'] for r in rows]
            assert all(r['q']==r['pairing']['q'] for r in rows)
            paths=[json.loads(p.read_text()) for p in (output/'session').glob('path-*.json')]
            second=next(p for p in paths if p['pose_id']==expected[-1])
            assert second['path_origins_checked']==['measured','commanded_reference']
            assert max(abs(a-b) for a,b in zip(second['origin'][11:],second['reference_origin'][11:]))>.02
            report=dict(passed=True,simulation_only=True,hardware_commands_sent=0,
                artificial_right_J1_load_nm=-5.58,eligible_poses=expected,
                injected_invalid_poses=[1,3] if inject_invalid else [],
                offsets_deg=[r['settling']['maximum_right_offset_deg'] for r in rows],
                measured_clearance_m=[r['measured_geometry']['minimum_clearance_m'] for r in rows],
                second_path_checks=second['path_origins_checked'],
                limitations='Controlled regression disturbance; not a physical gravity identification or hardware validation.')
            (output/'report.json').write_text(json.dumps(report,indent=2)+'\n')
            print(json.dumps(report,indent=2),flush=True)
        finally:
            if process.poll() is None:
                try:
                    status=post('/heartbeat',{})
                    post('/command',dict(action='quit',revision=status['revision']))
                    process.wait(timeout=15)
                except Exception:
                    process.terminate();process.wait(timeout=10)
            (output/'observations.json').write_text(json.dumps(observations,indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--port',type=int,default=8778)
    parser.add_argument('--worker',action='store_true',help=argparse.SUPPRESS)
    parser.add_argument('--inject-invalid',action='store_true',help='Simulate unreachable pose 1 and arrival timeout at pose 3')
    args=parser.parse_args()
    if args.worker:raise SystemExit(worker(args.output.resolve(),args.port,args.inject_invalid))
    verify(args.output.resolve(),args.port,args.inject_invalid)
