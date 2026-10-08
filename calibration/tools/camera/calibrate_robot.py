#!/usr/bin/env python3
"""Operator-paced live/sim hand-eye calibration through the deployed session."""
import argparse
from collections import deque
from datetime import datetime,timezone
import fcntl
import hashlib
from http.server import BaseHTTPRequestHandler,ThreadingHTTPServer
import json
import os
from pathlib import Path
import queue
import secrets
import subprocess
import sys
import threading
import time
from types import SimpleNamespace

from calibration_paths import ROOT, UPSTREAM, desktop_robot_paths, camera_python, sdk_directory


def run_wrist_fit(dataset, output, *, cancelled, timeout_s=120., python=None):
    """Fit in a separate process while the deployed controller keeps HOLD.

    Bounded waits release the Python thread to serve operator cancellation.
    A paused/ended session never publishes a completed solution in the UI;
    any already-written fit remains an offline artifact. Exit code 1 with a
    valid result is the CLI's explicit nonconvergence result, not a lost fit.
    """
    dataset, output = Path(dataset), Path(output)
    if not 0 < timeout_s <= 300:
        raise ValueError('Wrist solver timeout must be positive and at most 300 seconds')
    if cancelled():
        raise RuntimeError('Wrist solve cancelled; dataset remains saved')
    if output.exists():
        raise ValueError('Wrist solver output already exists; use a new artifact path')
    output.parent.mkdir(parents=True, exist_ok=True)
    logpath = output.with_suffix('.log')
    env = os.environ.copy()
    for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
        env[name] = '1'
    command = [str(python or sys.executable), str(ROOT/'tools/camera/solve_wrist_calibration.py'),
               str(dataset), '--output', str(output)]
    deadline = time.monotonic() + timeout_s
    with logpath.open('x') as logfile:
        process = subprocess.Popen(command, stdout=logfile, stderr=subprocess.STDOUT, env=env)
        try:
            while True:
                if cancelled():
                    raise RuntimeError('Wrist solve cancelled; dataset and any completed fit remain saved')
                remaining = deadline-time.monotonic()
                if remaining <= 0:
                    raise TimeoutError(f'Wrist solve exceeded {timeout_s:g} seconds; dataset saved; log: {logpath}')
                try:
                    code = process.wait(timeout=min(.2, remaining))
                    break
                except subprocess.TimeoutExpired:
                    pass
        finally:
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=2.)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=2.)
    if cancelled():
        raise RuntimeError('Wrist solve cancelled; dataset and any completed fit remain saved')
    if code not in (0, 1) or not output.exists():
        with logpath.open() as logfile:
            logfile.seek(max(0, logpath.stat().st_size-3000))
            detail = logfile.read().strip()
        raise ValueError(f'Wrist solver rejected dataset or failed (exit {code}); log: {logpath}; {detail}')
    result = json.loads(output.read_text())
    if (not isinstance(result, dict) or not isinstance(result.get('converged'), bool)
            or any(key not in result for key in ('T_wrist_from_camera', 'T_base_from_board', 'validation'))):
        raise ValueError(f'Wrist solver returned an incomplete result: {output}')
    if result.get('source_dataset_sha256') != hashlib.sha256(dataset.read_bytes()).hexdigest():
        raise ValueError('Wrist dataset changed during fitting; completed fit retained for inspection')
    result['fit_execution'] = dict(isolated_process=True, hardware_commands_sent=0,
                                 output=str(output), log=str(logpath), exit_code=code,
                                 timeout_s=timeout_s, thread_limits=1)
    return result


def lift_preparation_status(height_m, upper_m=.417):
    """Explain the existing planning guard; never change limits or command lift."""
    blocked=height_m>upper_m
    target=upper_m-.010
    return dict(height_mm=round(height_m*1000,1),upper_mm=round(upper_m*1000,1),
                blocked=blocked,target_mm=round(target*1000,1),
                lower_by_mm=round(max(0.,height_m-target)*1000,1) if blocked else 0.)


def camera_failure_message(directory):
    try:
        error=json.loads((directory/'error.json').read_text()).get('error')
        if error:return f'Camera worker failed: {error}; requesting HOLD.'
    except (OSError,ValueError):pass
    return 'Camera stream is stale; requesting HOLD.'


def preflight_failure_detail(report):
    """Summarize the existing audit verdict without changing its acceptance rules."""
    gateway=report.get('gateway',{})
    fault=str(gateway.get('fault') or report.get('error') or '')
    if '[Errno 101]' in fault or '[Errno 99]' in fault:
        return 'Robot network connection failed during startup: '+fault
    if str(report.get('error','')).startswith('ControlDelay:'):
        return 'Feedback freshness interrupted the read-only audit: '+report['error']
    if str(report.get('error','')).startswith('FeedbackSkewSafetyError:'):
        feedback=gateway.get('feedback_observations',{})
        skew=feedback.get('max_skew_ms')
        skew_text=f'{skew:.2f} ms' if isinstance(skew,(int,float)) else 'unknown'
        ages=feedback.get('ages_ms',{})
        age_text=', '.join(f'{side} {age:.2f} ms' for side,age in ages.items()
                          if isinstance(age,(int,float)))
        return (f'Read-only startup audit interrupted by feedback skew: maximum {skew_text} '
                f'(limit 20 ms); latest feedback ages: {age_text or "unavailable"}. '
                f'{report.get("cycles",0)}/1500 cycles completed; full timing qualification incomplete. '
                +report['error'])
    if report.get('cycles',0)==0 and (gateway.get('fault') or report.get('error')):
        return ('Startup failed before timing/feedback qualification could complete: '
                +str(gateway.get('fault') or report['error']))
    reasons=[]
    if report.get('motion_readiness_error'):
        reasons.append(report['motion_readiness_error'])
    feedback=report.get('gateway',{}).get('feedback_observations',{})
    if report.get('gateway_observed_feedback_clean') is not True:
        reasons.append('feedback monitor was not clean: '
                       f"{feedback.get('invalid_samples','unknown')} invalid samples; "
                       f"maximum device skew {feedback.get('max_skew_ms','unknown')} ms")
        reasons.extend(str(e['error']) for e in feedback.get('events',[])[:3] if e.get('error'))
    if report.get('performance_passed') is False:
        limits=report.get('performance_acceptance_limits',{})
        metrics=report.get('timing_ms_p50_p95_p99_max',{})
        exceeded=[]
        for name,limit in limits.items():
            values=metrics.get(name.removesuffix('_p99_ms')+'_ms')
            if isinstance(values,(list,tuple)) and len(values)>2 and values[2]>limit:
                exceeded.append(f"{name}={values[2]:.2f} ms (limit {limit:g} ms)")
        reasons.append('timing acceptance limits exceeded: '+(', '.join(exceeded) if exceeded
                       else f"{limits}; measured values unavailable or other timing failures"))
    if report.get('error'):
        reasons.append(str(report['error']))
    return '; '.join(dict.fromkeys(reasons)) or 'Audit acceptance conditions were not met; inspect the saved report.'


def retryable_feedback_audit(report):
    """Retry a fresh read-only audit, never accept or resume a failed attempt."""
    gateway=report.get('gateway',{})
    events=gateway.get('feedback_observations',{}).get('events',[])
    closed_readonly=(report.get('mode')=='audit' and report.get('motion_commands_sent')==0
            and report.get('motion_readiness_passed') is True
            and gateway.get('kind')=='closed' and not gateway.get('fault')
            and gateway.get('motion_commands_sent')==0 and not report.get('close_error')
            and not gateway.get('stop_failures') and bool(report.get('initial_health')))
    if not closed_readonly:return False
    # An early stale-feedback abort cannot yield a completed timing audit or
    # post-event diagnostics. It can only trigger a NEW full read-only attempt.
    if str(report.get('error','')).startswith('ControlDelay: stale robot feedback'):
        return True
    # FeedbackSkewSafetyError is another ControlDelay subtype. Its class name
    # survives serialization, so matching only "ControlDelay:" missed it.
    # Retry only a fully closed, motion-free audit, never a running controller.
    if str(report.get('error','')).startswith('FeedbackSkewSafetyError:'):
        return True
    return (
            report.get('performance_passed') is True
            and report.get('cycles')==1500 and report.get('failed_cycles')==0
            and report.get('skipped_slots')==0 and report.get('trace_complete') is True
            and report.get('diagnostics_complete') is True and not report.get('trace_dropped_rows')
            and report.get('gateway_observed_feedback_clean') is False
            and bool(events) and all(e.get('error')=='robot feedback skew exceeds 20 ms'
                                    and e.get('recovered_at') is not None for e in events))


def startup_audit(run,accepted,output,attempts=3):
    """Keep every failed audit; acquisition still requires a wholly passing audit."""
    history=[]
    for attempt in range(1,attempts+1):
        directory=output/('audit' if attempt==1 else f'audit-retry-{attempt}')
        print(f'Read-only startup audit {attempt}/{attempts} (30 seconds; no motion).',flush=True)
        report=run(directory)
        passed=accepted(report)
        history.append(dict(attempt=attempt,passed=passed,report=str(directory/'summary.json'),
                            reason=None if passed else preflight_failure_detail(report)))
        (output/'startup-audits.json').write_text(json.dumps(history,indent=2))
        if passed or not retryable_feedback_audit(report) or attempt==attempts:
            return report,directory/'summary.json'
        print('Feedback timing interrupted this attempt. Repeating the full audit with unchanged limits; '
              'robot control remains unacquired.',flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--mode',choices=['sim','live'],default='sim')
    p.add_argument('--deployment',type=Path,default=UPSTREAM)
    p.add_argument('--robot-config',type=Path)
    p.add_argument('--research-release',type=Path)
    p.add_argument('--confirm-onsite-ready',action='store_true')
    p.add_argument('--port',type=int,default=8772)
    p.add_argument('--probe',action='store_true',help='read-only startup check; acquire no robot control')
    p.add_argument('--scheduling',choices=['responsive','legacy'],default='responsive',
                   help='responsive separates SDK/control/background CPUs and requests SDK thread priority before ownership')
    p.add_argument('--auto',action='store_true',help='simulation-only end-to-end test')
    p.add_argument('--run-seconds',type=float,default=3600)
    p.add_argument('--output',type=Path)
    p.add_argument('--snapshot',type=Path,default=ROOT/'config/camera/both_j6_sim_start.json')
    p.add_argument('--attachment',type=Path,default=ROOT/'config/camera/board_right_centered.json')
    p.add_argument('--candidates',type=Path,default=ROOT/'config/camera/right_centered_candidates.json',help='Previously simulated candidate pool for this attachment')
    p.add_argument('--validation-plan',type=Path,help='Run three left-hand RGB/extrinsic validation poses instead of board calibration')
    p.add_argument('--wrist-plan',type=Path,help='Simulated stationary-board, right-wrist RealSense eye-in-hand route')
    a=p.parse_args()
    if a.wrist_plan and a.validation_plan:p.error('Select wrist calibration or head-camera validation')
    if a.output is None:
        a.output=ROOT/'runs'/('wrist-calibration' if a.wrist_plan else 'left-hand-validation' if a.validation_plan else 'live-calibration')/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S.%fZ')
    a.output=a.output.resolve()
    if a.mode=='live' and (a.auto or (not a.confirm_onsite_ready and not a.probe)):
        p.error('live requires --confirm-onsite-ready and never allows --auto')
    os.environ['HEXFELLOW_CALIBRATION_UMI']=str(a.deployment.resolve())
    sys.path.insert(0,str(ROOT/'src'))
    import policy_sim
    policy_sim.__path__.insert(0,str(ROOT/'src/policy_sim'))
    import numpy as np
    import pinocchio as pin
    from policy_sim.hardware import session
    if a.mode=='live' and a.scheduling=='responsive':
        from policy_sim.hardware.responsive_scheduling import configure
        scheduling=configure()
        print('Calibration CPU pools: '+json.dumps(scheduling),flush=True)
    from policy_sim.hardware.verification import software_binding,device_binding
    from policy_sim.hardware.quest_trial import audit_passed
    from policy_sim.hardware.verify import timing_probe
    from policy_sim.hardware.highspeed_trial import require_research_release
    from policy_sim.control.motion_profile import controller_config_for_profile
    from policy_sim.hardware.mit_ik_replay import SimulationPlant
    from policy_sim.calibration.handeye import solve,summarize
    from policy_sim.control.robot import CONTROLLED_JOINTS,BASE_FRAME
    from calibration_runtime import CalibrationSource,stationary_pair,capture_position_check
    from calibration_path import PathChecker
    from calibration_outcomes import InvalidCalibrationPose
    from calibration_sequence import Sequence

    if a.robot_config and a.research_release:
        runtime,release=a.robot_config,a.research_release
    else:
        defaults=desktop_robot_paths()
        runtime=a.robot_config or defaults[0]
        release=a.research_release or defaults[1]
    parser=argparse.ArgumentParser();session.add_arguments(parser)
    values=parser.parse_args(['--mode',a.mode,'--robot-config',str(runtime),'--run-seconds',str(a.run_seconds)])
    values.control_group='coupled';values.tracking_state='measured'
    values.demo_authorized=a.mode=='live';values.research_release=release
    values.confirm_onsite_ready=a.confirm_onsite_ready;values.save_trace=a.output/'session'
    mapping,profile,default_pose,_=session.load_configuration(values)
    print(f'Mapping: {runtime} | left signs {mapping.data["arms"]["left"]["signs"]} | right signs {mapping.data["arms"]["right"]["signs"]}',flush=True)
    if a.mode=='live':require_research_release(profile,release,runtime)
    validation=None
    wrist=None
    if a.wrist_plan:
        from wrist_calibration_geometry import load_plan,WristPathChecker
        wrist=load_plan(a.wrist_plan.resolve())
        if hashlib.sha256(runtime.read_bytes()).hexdigest()!=wrist['mapping_sha256']:
            raise ValueError('Robot mapping differs from the simulated wrist calibration plan')
        if not np.allclose(session.pose_vector(default_pose,np.zeros(3)),wrist['initial_q'],rtol=0,atol=1e-10):
            raise ValueError('Default pose differs from the simulated wrist preparation')
    if a.validation_plan:
        from left_hand_validation import load_plan,LeftPathChecker,resolve
        validation=load_plan(a.validation_plan)
        if hashlib.sha256(runtime.read_bytes()).hexdigest()!=validation['mapping_sha256']:
            raise ValueError('Robot mapping differs from the simulated validation plan')
        if not np.allclose(session.pose_vector(default_pose,np.zeros(3)),validation['initial_q'],rtol=0,atol=1e-10):
            raise ValueError('Default pose differs from the simulated validation start')
    pose_count=len(wrist['poses']) if wrist else 3 if validation else 30
    capture_side='left' if validation else 'right'
    a.output.mkdir(parents=True,exist_ok=False)
    if wrist:checker=WristPathChecker(a.output/'geometry',a.wrist_plan.resolve())
    elif validation:checker=LeftPathChecker(a.output/'geometry',a.validation_plan.resolve())
    else:
        route=json.loads((ROOT/'artifacts/reference/camera-calibration/initial-right-route.json').read_text())
        checker=PathChecker(a.output/'geometry',route,a.attachment,a.candidates)
    source=CalibrationSource(live=a.mode=='live',capture_side=capture_side)
    sequence=Sequence(pose_count=pose_count,preparation='prepare-default' if validation or wrist else 'prepare',
                      return_before_solve=bool(validation),return_after_solve=bool(wrist))
    values.allow_uncommissioned_profile_sha256=profile.sha256
    # Preserve the current deployment checkout; record exact source/config binding.
    binding=software_binding(mapping)
    (a.output/'binding.json').write_text(json.dumps(dict(deployment=str(a.deployment),binding=binding,
        attachment_sha256=hashlib.sha256(a.attachment.read_bytes()).hexdigest(),
        adapter_sha256={str(f):hashlib.sha256(f.read_bytes()).hexdigest() for f in Path(__file__).parent.glob('calibration*.py')}),indent=2))
    if validation:
        (a.output/'validation-plan.json').write_text(json.dumps(validation,indent=2))
        (a.output/'validation-code.json').write_text(json.dumps({f.name:hashlib.sha256(f.read_bytes()).hexdigest()
                for f in [Path(__file__),ROOT/'tools/camera/left_hand_validation.py',ROOT/'tools/camera/validate_camera_overlay.py']},indent=2))
    if wrist:
        (a.output/'wrist-plan.json').write_text(json.dumps(wrist,indent=2))
        (a.output/'wrist-code.json').write_text(json.dumps({str(f.relative_to(ROOT)):hashlib.sha256(f.read_bytes()).hexdigest()
            for f in [Path(__file__),ROOT/'tools/camera/wrist_calibration_geometry.py',
                      ROOT/'tools/camera/wrist_camera_worker.py',ROOT/'tools/camera/solve_wrist_calibration.py',
                      ROOT/'src/policy_sim/calibration/wrist_handeye.py']},indent=2))
    robot_lock=None;camproc=None;server=None;log=None
    source_error=None
    stop_workers=threading.Event()
    jobs=queue.Queue(maxsize=1)
    captured={};ineligible={};revision=0;busy=False;solution=None;fault=None;cancel_epoch=0;activity=None
    guard=threading.RLock()
    camera_dir=a.output/'camera';camera_dir.mkdir()
    pm=pin.buildModelFromUrdf(str(ROOT/'urdf/maver_x4_gp80_zed_40deg.urdf'));pd=pm.createData()
    pi=[pm.joints[pm.getJointId(n)].idx_q for n in CONTROLLED_JOINTS]

    def status():
        row=source.last_display
        lift=lift_preparation_status(row['q'][3]) if row else None
        return dict(lift=lift,mode=a.mode,task='wrist_calibration' if wrist else 'left_hand_validation' if validation else 'calibration',pose_count=pose_count,
            phase=(source.last_display or {}).get('phase','starting'),
            pose=source.arrived_pose,target_pose=source.target_pose,captured=sorted(captured),ineligible=ineligible,attempted=sorted(set(captured)|set(ineligible)),
            message=fault or source.message,revision=revision,busy=busy or source.pending_command is not None,
            finished=source.finished,paused=source.paused,output=str(a.output),solution=solution,sequence=sequence.mode,activity=activity,
            failure_detail=source.failure_detail,pose_settling=source.pose_settling,
            pending_invalid=source.pending_invalid)

    def snapshot():
        row=source.last_display
        if row is None or time.monotonic()-row['at']>.15:raise ValueError('No fresh robot feedback')
        return row

    def save_dataset():
        meta=json.loads((camera_dir/'ready.json').read_text())
        value=dict(mount_frame='maver_x4_asm_URDF_link_1',hand_frame='maver_x4_asm_mount1' if validation else 'maver_x4_asm_mount2',
            camera_frame='zed_left_camera_frame_optical',board=meta['board'],camera=meta,
            board_attachment_assumption=json.loads(a.attachment.read_text()),
            simulation_only=a.mode=='sim',observations=[captured[k] for k in sorted(captured)],ineligible=[ineligible[k] for k in sorted(ineligible)])
        if validation:
            value.update(purpose='independent_left_hand_validation',mapping_sha256=hashlib.sha256(runtime.read_bytes()).hexdigest(),
                extrinsics_file=str(resolve(validation['extrinsics_file'])),extrinsics_sha256=validation['extrinsics_sha256'],
                board_attachment_assumption={'source':'fitted candidate','not_used_for_rgb_eligibility':True})
        if wrist:
            value.update(purpose='right_wrist_eye_in_hand',mount_frame=wrist['parent_frame'],
                parent_frame=wrist['parent_frame'],hand_frame=wrist['parent_frame'],
                camera_frame=meta['frame'],base_frame=BASE_FRAME,
                board_attachment_assumption={'stationary_on_table':True,'not_attached_to_hand':True},
                intrinsics={'matrix':meta['K'],'distortion':meta['distortion']},
                mapping_sha256=hashlib.sha256(runtime.read_bytes()).hexdigest(),
                equation='T_base_from_wrist_i @ T_wrist_from_camera @ T_camera_from_board_i = T_base_from_board')
            value['observations']=[(captured if k in captured else ineligible)[k]
                                   for k in sorted(set(captured)|set(ineligible))]
        tmp=a.output/'dataset.tmp';tmp.write_text(json.dumps(value,indent=2));tmp.replace(a.output/'dataset.json')

    def record_invalid(pose,reason,epoch,details=None):
        if not 1<=pose<=pose_count:raise ValueError('Only capture poses can be skipped')
        row=snapshot()
        if row['phase'] not in ('hold','ready'):
            raise ValueError('Invalid pose must stop in HOLD before advancing')
        # Check a complete stationary interval, including feedback after its
        # midpoint. A failed arrival is never treated as a successful capture.
        pair=stationary_pair(list(source.rows),row['at']-.15)
        if np.any(abs(np.asarray(row['v']))>session.SETTLED_SPEED):
            raise ValueError('Wait for stationary HOLD before skipping invalid pose')
        geometry=checker.check_pose(row['q'],source.bounds)
        rejected=dict(pose_id=pose,accepted=False,eligibility='invalid',reason=reason,
            image=None,q=row['q'],hold_pairing=pair,measured_geometry=geometry,
            details=details,at=row['at'])
        with guard:
            if epoch!=cancel_epoch or source.quit:raise ValueError('Skip cancelled')
            ineligible[pose]=rejected;captured.pop(pose,None)
            (a.output/f'pose-{pose:02d}-invalid.json').write_text(json.dumps(rejected,indent=2))
            save_dataset()
            source.pending_invalid=None;source.arrived_pose=0;source.target_pose=pose
            source.paused=False
        source.message=f'Pose {pose}: invalid — {reason}. Continuing to next pose.'
        print(source.message,flush=True)

    def process(action,epoch):
        try:return process_action(action,epoch)
        except InvalidCalibrationPose as error:
            pose=error.pose_id if error.pose_id is not None else source.target_pose
            row=snapshot()
            if not 1<=pose<=pose_count or row['phase'] not in ('hold','ready'):
                raise ValueError('Rejected target requires stationary HOLD before skipping') from error
            with guard:
                if epoch!=cancel_epoch or source.quit:raise ValueError('Skip cancelled')
                source.pending_invalid=dict(pose_id=pose,reason=str(error),ready=False,
                    measured_q=row['q'],settling=source.pose_settling)
                source.arrived_pose=0;source.target_pose=pose
            source.message=f'Pose {pose}: invalid — {error}. Checking stationary HOLD before skipping.'

    def process_action(action,epoch):
        nonlocal solution
        if epoch!=cancel_epoch or source.quit:raise ValueError('Action cancelled')
        row=snapshot()
        if a.mode=='live':
            status_file=camera_dir/'status.json'
            if camproc.poll() is not None or not status_file.exists() or time.monotonic()-json.loads(status_file.read_text())['at']>1:
                raise ValueError('Camera stream is not fresh; motion/capture disabled')
        if action=='skip-invalid':
            invalid=source.pending_invalid
            if invalid is None or not invalid['ready']:
                raise ValueError('Invalid pose is still stopping or settling')
            record_invalid(invalid['pose_id'],invalid['reason'],epoch,invalid)
            return
        if action in ('full-default','prepare-default'):
            if row['phase'] not in ('readonly','hold','ready'):
                raise ValueError('Pause and wait for HOLD before moving to default')
            if np.any(abs(np.array(row['v']))>session.SETTLED_SPEED):raise ValueError('Wait for stationary robot')
            target=session.pose_vector(default_pose,np.array(row['q'][:3])).tolist()
            payload=checker.plan(row['q'],-2,source.bounds,target,
                reference_origin=row.get('reference_q') if validation or wrist else None)
            (a.output/f'default-path-{time.time_ns()}.json').write_text(json.dumps(payload,indent=2))
            with guard:
                if epoch!=cancel_epoch or source.quit:raise ValueError('Default motion cancelled')
                source.pending_command='full-default'
                source.commands.put_nowait(('default',dict(full_default=True,origin=row['q'])))
            return
        if action=='prepare':
            if wrist:return process_action('prepare-default',epoch)
            lift=lift_preparation_status(row['q'][3])
            if lift['blocked']:
                raise ValueError(f"Lift reads {lift['height_mm']:.1f} mm; planning maximum is {lift['upper_mm']:.1f} mm. "
                    "Click Move to default pose to run the controlled retreat and move to the 300 mm default.")
            if source.prepared:raise ValueError('Already prepared')
            if not (camera_dir/'ready.json').exists():raise ValueError('Camera is not ready')
            with guard:
                if epoch!=cancel_epoch or source.quit:raise ValueError('Preparation cancelled')
                source.pending_command='default'
                source.commands.put_nowait(('default',None))
            return
        if action in ('next','resume','return'):
            if not source.prepared or row['phase'] not in ('hold','ready'):raise ValueError('Prepare and wait for HOLD first')
            if action=='next':
                if source.paused:raise ValueError('Resume the interrupted pose first')
                if source.arrived_pose>0 and source.arrived_pose not in captured and source.arrived_pose not in ineligible:raise ValueError('Capture current pose first')
                pose=next((i for i in range(1,pose_count+1) if i not in captured and i not in ineligible),pose_count+1)
            elif action=='resume':
                if not source.paused or not source.target_pose:raise ValueError('No interrupted pose to resume')
                pose=source.target_pose
            else:pose=-1
            if pose>pose_count:raise ValueError(f'All {pose_count} poses complete')
            if np.any(abs(np.array(row['v']))>session.SETTLED_SPEED):raise ValueError('Wait for stationary robot')
            source.message='Checking the path from current measured HOLD…'
            default_target=session.pose_vector(default_pose,np.array(row['q'][:3])).tolist()
            try:
                payload=checker.plan(row['q'],pose,source.bounds,default_target,reference_origin=row.get('reference_q'))
            except InvalidCalibrationPose as error:
                error.pose_id=pose
                raise
            (a.output/f'path-{time.time_ns()}.json').write_text(json.dumps(payload,indent=2))
            with guard:
                if epoch!=cancel_epoch or source.quit:raise ValueError('Path cancelled')
                source.pending_command='calibration-path'
                source.commands.put_nowait(('calibration-path',payload))
            return
        if action=='capture':
            if source.paused or row['phase']!='hold' or source.arrived_pose<=0:raise ValueError('Capture requires arrival and stationary HOLD')
            if source.arrived_pose!=source.target_pose:raise ValueError('Current pose is not ready for capture')
            settling=source.pose_settling
            if not settling or settling['status']!='stable_measured_pose':raise ValueError('Measured settling check has not passed')
            capture_position_check(row['q'],settling['target_q'],capture_side)
            checker.check_pose(row['q'],source.bounds)
            ident=time.time_ns();start=time.monotonic();pose=source.arrived_pose
            request=dict(id=ident,after=start,q=row['q'])
            tmp=camera_dir/'request.tmp';tmp.write_text(json.dumps(request));tmp.replace(camera_dir/'request.json')
            resultfile=camera_dir/str(ident)/'result.json'
            deadline=start+10
            while not resultfile.exists():
                if stop_workers.is_set() or time.monotonic()>deadline:raise ValueError('Camera capture timeout')
                time.sleep(.02)
            result=json.loads(resultfile.read_text())
            if wrist and result.get('timing_accepted') is not True:
                rejected=dict(result,pose_id=pose,accepted=False,eligibility='not eligible',
                    reason=result.get('reason','Camera timestamp could not be paired with measured joints'),
                    image=str(resultfile.parent/'left.png'),q=row['q'],
                    measured_geometry=checker.check_pose(row['q'],source.bounds),
                    synchronized=False,settling=settling)
                with guard:
                    if epoch!=cancel_epoch or source.quit:raise ValueError('Capture cancelled')
                    ineligible[pose]=rejected;captured.pop(pose,None)
                    (resultfile.parent/'not-eligible.json').write_text(json.dumps(rejected,indent=2));save_dataset()
                source.message=f'Pose {pose}: not eligible — {rejected["reason"]}. Continuing to next pose.'
                return
            while time.monotonic()<result['image_time']+.15:time.sleep(.01)
            pair=stationary_pair(list(source.rows),result['image_time'],uncertainty=result['clock_mapping_uncertainty_s'])
            capture_offset=capture_position_check(pair['q'],settling['target_q'],capture_side)
            geometry=checker.check_pose(pair['q'],source.bounds)
            if validation:
                observation=dict(result,pose_id=pose,accepted=True,eligibility='paired_rgb',
                    stationary=True,image=str(resultfile.parent/'left.png'),q=pair['q'],pairing=pair,
                    measured_geometry=geometry,settling=settling,capture_offset=capture_offset,
                    board_detection_required=False,requires_visual_review=True)
                with guard:
                    if epoch!=cancel_epoch or source.quit:raise ValueError('Capture cancelled')
                    (resultfile.parent/'paired.json').write_text(json.dumps(observation,indent=2))
                    captured[pose]=observation;ineligible.pop(pose,None);save_dataset()
                source.message=f'Captured left-hand pose {pose}/{pose_count} with measured joints.'
                print(source.message,flush=True)
                return
            if not result.get('accepted') or result.get('corners',0)<12:
                rejected=dict(result,pose_id=pose,accepted=False,eligibility='not eligible',
                    reason=f"{result.get('corners',0)} corners (need 12); {result.get('reason')}",
                    image=str(resultfile.parent/'left.png'),pairing=pair,q=pair['q'],
                    measured_geometry=geometry,settling=settling,capture_offset=capture_offset)
                with guard:
                    if epoch!=cancel_epoch or source.quit:raise ValueError('Capture cancelled')
                    ineligible[pose]=rejected;captured.pop(pose,None)
                    (resultfile.parent/'not-eligible.json').write_text(json.dumps(rejected,indent=2))
                    save_dataset()
                source.message=f'Pose {pose}: not eligible — {rejected["reason"]}. Continuing to next pose.'
                return
            q=pin.neutral(pm);q[pi]=np.array(pair['q'])[3:];pin.framesForwardKinematics(pm,pd,q)
            g=(pd.oMf[pm.getFrameId('maver_x4_asm_URDF_link_1')].inverse()*pd.oMf[pm.getFrameId('maver_x4_asm_mount2')]).homogeneous
            observation=dict(result,pose_id=pose,accepted=True,stationary=True,synchronized=True,
                pairing=pair,q=pair['q'],T_mount_from_hand=g.tolist(),image=str(resultfile.parent/'left.png'),
                measured_geometry=geometry,settling=settling,capture_offset=capture_offset)
            if wrist:
                base_pose=pd.oMf[pm.getFrameId(BASE_FRAME)]
                wrist_pose=pd.oMf[pm.getFrameId(wrist['parent_frame'])]
                observation.pop('T_mount_from_hand')
                observation.update(T_base_from_wrist=(base_pose.inverse()*wrist_pose).homogeneous.tolist(),
                    synchronization='stationary host-time pairing; no hardware exposure trigger')
            with guard:
                if epoch!=cancel_epoch or source.quit:raise ValueError('Capture cancelled')
                (resultfile.parent/'paired.json').write_text(json.dumps(observation,indent=2))
                captured[pose]=observation;ineligible.pop(pose,None);save_dataset()
            source.message=f'Captured pose {pose}: {result["corners"]} corners, {result["reprojection_rms_px"]:.3f} px RMS.'
            return
        if action=='solve':
            if len(captured)+len(ineligible)!=pose_count:raise ValueError(f'Finish all {pose_count} pose attempts first')
            if validation:
                if source.arrived_pose!=-1 or source.target_pose!=-1 or row['phase']!='hold':
                    raise ValueError('Return to default before generating overlays')
                compared=checker.compare(a.output/'dataset.json',a.output/'comparisons')
                with guard:
                    if epoch!=cancel_epoch or source.quit:raise ValueError('Comparison cancelled; files remain saved')
                    solution=compared
                source.message=f'Validation finished: {len(captured)} captures. Real/sim overlays ready below.'
                print(source.message+' '+str(a.output/'comparisons'),flush=True)
                return
            if len(captured)<12:
                solution=dict(converged=False,status='insufficient_eligible_images',eligible=len(captured),not_eligible=len(ineligible))
                (a.output/'solve-status.json').write_text(json.dumps(solution,indent=2))
                source.message='Sequence finished; fewer than 12 eligible images. Dataset saved; no extrinsics produced.'
                return
            rows=[captured[k] for k in sorted(captured)]
            g=np.array([r['T_base_from_wrist' if wrist else 'T_mount_from_hand'] for r in rows]);c=np.array([r['T_camera_from_board'] for r in rows])
            fit=np.arange(len(rows))%5!=4
            try:
                if wrist:
                    source.message='Solving saved wrist observations in a separate process; robot remains in HOLD.'
                    result=run_wrist_fit(a.output/'dataset.json',a.output/f'wrist-fit-{time.time_ns()}.json',
                        cancelled=lambda:stop_workers.is_set() or epoch!=cancel_epoch or source.quit)
                else:
                    result=solve(g[fit],c[fit])
            except ValueError as error:
                solution=dict(converged=False,status='solver_rejected_dataset' if wrist else 'insufficient_pose_diversity',
                    reason=str(error),eligible=len(rows))
                (a.output/'solve-status.json').write_text(json.dumps(solution,indent=2))
                source.message=('Sequence finished; wrist solver rejected the dataset. Dataset and solver log saved.'
                                if wrist else 'Sequence finished; solver rejected pose diversity. Dataset saved.')
                return
            validation_result=(result['validation']
                if wrist else summarize(g[~fit],c[~fit],np.array(result['T_mount_from_camera']),np.array(result['T_hand_from_board'])))
            result.update(validation=validation_result,
                simulation_only=a.mode=='sim',calibration_status='candidate_requires_physical_validation')
            if wrist:
                result.update(camera=json.loads((camera_dir/'ready.json').read_text()),
                    mapping_sha256=hashlib.sha256(runtime.read_bytes()).hexdigest())
            dest=a.output/f'extrinsics-{time.time_ns()}.json';dest.write_text(json.dumps(result,indent=2))
            with guard:
                if epoch!=cancel_epoch or source.quit:raise RuntimeError('Solve cancelled; completed fit remains saved')
                solution=dict(file=str(dest),converged=result['converged'],validation=result['validation'],
                              status='candidate' if result['converged'] else 'fit_did_not_converge')
                if wrist:solution['validation_reprojection']=result.get('validation_reprojection')
            source.message=('Extrinsics saved; held-out validation available. No transform applied automatically.'
                            if result['converged'] else
                            'Solver did not converge. Dataset and provisional fit saved; do not use this transform.')
            return
        raise ValueError('Unknown action')

    def worker():
        nonlocal busy,revision,fault,activity
        while not stop_workers.is_set():
            try:action,epoch=jobs.get(timeout=.1)
            except queue.Empty:continue
            activity=action
            try:process(action,epoch)
            except Exception as e:
                if not (sequence.mode=='returning' and 'cancelled' in str(e).lower()):sequence.pause()
                source.stop=True
                source.message=f'{action} rejected: {e}'
                print(source.message,flush=True)
                if a.auto:fault=source.message;source.quit=True
            finally:
                with guard:busy=False;activity=None;revision+=1

    def enqueue(action,version):
        nonlocal busy,revision,cancel_epoch
        with guard:
            if action in ('pause','quit'):
                sequence.pause()
                cancel_epoch+=1;revision+=1
                if action=='quit':source.quit=True
                else:source.stop=True
                return
            if action=='return-default':
                if source.finished or version!=revision:raise ValueError('Stale action or session ended')
                cancel_epoch+=1;revision+=1;source.stop=True
                sequence.start(source.interruption_count,returning=True)
                return
            if version!=revision or busy or source.pending_command or source.finished:raise ValueError('Busy, stale action, or session ended')
            if action=='start':
                sequence.start(source.interruption_count);revision+=1;return
            if action=='full-default':sequence.pause()
            busy=True;revision+=1;jobs.put_nowait((action,cancel_epoch))
            with (a.output/'operator-events.jsonl').open('a') as f:
                f.write(json.dumps(dict(action=action,at=time.monotonic(),revision=revision))+'\n')

    token=secrets.token_urlsafe(32)
    page=(Path(__file__).parent/('wrist_calibration_live.html' if wrist else 'left_hand_validation.html' if validation else 'calibration_live.html')).read_text().replace('__TOKEN__',token)
    class Handler(BaseHTTPRequestHandler):
        def log_message(self,*args):pass
        def send(self,code,value,mime='application/json'):
            content=value if isinstance(value,bytes) else (value.encode() if isinstance(value,str) else json.dumps(value).encode())
            self.send_response(code);self.send_header('Content-Type',mime);self.send_header('Content-Length',str(len(content)));self.send_header('Cache-Control','no-store');self.end_headers();self.wfile.write(content)
        def local(self):return self.headers.get('Host') in (f'127.0.0.1:{a.port}',f'localhost:{a.port}')
        def do_GET(self):
            if not self.local():return self.send(403,{})
            if self.path=='/':return self.send(200,page,'text/html')
            if self.path=='/status':return self.send(200,status())
            if self.path.startswith('/camera.jpg'):
                try:return self.send(200,(camera_dir/'preview.jpg').read_bytes(),'image/jpeg')
                except OSError:return self.send(404,{})
            if validation and self.path.split('?')[0] in [f'/comparison-{n}.png' for n in range(1,pose_count+1)]:
                n=int(self.path.split('?')[0].split('-')[1].split('.')[0])
                try:return self.send(200,(a.output/f'comparisons/pose-{n:02d}/comparison.png').read_bytes(),'image/png')
                except OSError:return self.send(404,{})
            return self.send(404,{})
        def do_POST(self):
            if not self.local() or self.headers.get('X-Calibration-Token')!=token:return self.send(403,{})
            try:
                n=int(self.headers.get('Content-Length','0'))
                if not 0<n<4096:raise ValueError('Invalid request')
                v=json.loads(self.rfile.read(n))
                if self.path=='/heartbeat':source.heartbeat=time.monotonic()
                elif self.path=='/command':enqueue(v['action'],v['revision'])
                else:return self.send(404,{})
                return self.send(200,status())
            except (ValueError,KeyError,queue.Full) as e:return self.send(409,dict(error=str(e)))

    def autopilot():
        pause_tested=False;pause_at=None;returned=False
        if a.auto:sequence.start(source.interruption_count)
        while not stop_workers.is_set() and not source.finished:
            if a.auto:source.heartbeat=time.monotonic()
            if source.last_display and not busy and not source.pending_command and source.commands.empty():
                st=status()
                if a.auto and st['phase']=='calibration' and not pause_tested:
                    enqueue('pause',revision);pause_tested=True;pause_at=time.monotonic()
                if a.auto and source.paused and st['phase']=='hold' and pause_at and time.monotonic()-pause_at>1:
                    sequence.start(source.interruption_count);pause_at=None
                if a.auto and sequence.mode=='complete' and not returned:
                    if validation or wrist:
                        enqueue('quit',revision);break
                    sequence.start(source.interruption_count,returning=True);returned=True
                if a.auto and returned and source.arrived_pose==source.target_pose==-1 and st['phase']=='hold':
                    enqueue('quit',revision);break
                action=sequence.step(now=time.monotonic(),prepared=source.prepared,phase=st['phase'],
                    arrived=source.arrived_pose,target=source.target_pose,captured=captured,ineligible=ineligible,solved=solution is not None,
                    interrupted=source.interruption_count,paused=source.paused,
                    busy=busy or source.stop,fresh_operator=time.monotonic()-source.heartbeat<=2,
                    pending_invalid=source.pending_invalid)
                if action:
                    try:enqueue(action,revision)
                    except ValueError:pass
            time.sleep(.1)

    try:
        if a.mode=='live':
            lockpath=a.deployment/'runs/.hexfellow-onsite.lock'
            lockpath.parent.mkdir(parents=True,exist_ok=True)
            robot_lock=lockpath.open('a')
            try:fcntl.flock(robot_lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            except BlockingIOError:
                print('Another robot controller or read-only check is active. Wait for it to exit before restarting. No control acquired.',file=sys.stderr,flush=True)
                return 1
            mapping.authorize(profile,'whole-body',profile.sha256)
            audit,report_path=startup_audit(
                lambda directory:timing_probe(mapping,profile,directory,30,sdk_sink=True,allow_lift_retreat=True),
                audit_passed,a.output)
            if not audit_passed(audit):
                detail=preflight_failure_detail(audit)
                (a.output/'result.json').write_text(json.dumps(dict(
                    session='preflight_rejected',passed=False,captures=0,solution=None,
                    fault=detail,audit_report=str(report_path),
                    motion_commands_sent=audit.get('motion_commands_sent')),indent=2))
                print(f'Calibration startup blocked: {detail}\nAudit report: {report_path}\n'
                      'No robot control acquired. Correct the reported condition before restarting.',
                      file=sys.stderr,flush=True)
                return 1
            source.binding=binding;values.demo_device_binding=device_binding(SimpleNamespace(health=audit['initial_health']))
            if a.probe:
                # Read-only means no live session, acquisition, camera worker or
                # movement, even when the startup audit passes.
                (a.output/'result.json').write_text(json.dumps(dict(
                    session='readonly_probe',passed=True,captures=0,solution=None,
                    motion_commands_sent=0,audit_report=str(report_path)),indent=2))
                print(f'Read-only calibration check passed. No robot control acquired. {report_path}',flush=True)
                return 0
        else:
            # Simulation starts from the explicit snapshot, just like the head
            # workflow. The route's historical measured_start_q is a planning
            # input, not a current robot observation or mandatory start state.
            initial=np.array(json.loads(a.snapshot.read_text())['q']);initial[:3]=0
            (a.output/'simulation-start.json').write_text(json.dumps(dict(
                snapshot=str(a.snapshot.resolve()), q=initial.tolist(),
                snapshot_sha256=hashlib.sha256(a.snapshot.read_bytes()).hexdigest(),
                simulation_only=True, hardware_commands_sent=0),indent=2))
            source.simulation_plant=SimulationPlant(SimpleNamespace(summary={}),controller_config_for_profile(profile),
                np.array(mapping.data['fixed_gripper_positions']),initial)
        python=camera_python('realsense' if wrist else 'zed') if a.mode=='live' else Path(sys.executable)
        env=os.environ.copy();sdk=sdk_directory()
        env['ZED_DIR']=str(sdk)
        env['LD_LIBRARY_PATH']=f'{sdk}/lib:{sdk}/dependencies/ai_dependencies/lib:'+env.get('LD_LIBRARY_PATH','')
        env['MUJOCO_GL']='egl'
        log=(a.output/'camera.log').open('w')
        camera_command=[str(python),str(Path(__file__).parent/('wrist_camera_worker.py' if wrist else 'calibration_camera_worker.py')),'--mode',a.mode,'--directory',str(camera_dir),'--scene',str(checker.study.scene)]
        if wrist:camera_command.extend(['--serial',str(wrist['serial'])])
        if validation:camera_command.extend(['--validation-plan',str(a.validation_plan.resolve())])
        camproc=subprocess.Popen(camera_command,env=env,stdout=log,stderr=subprocess.STDOUT)
        deadline=time.monotonic()+30
        while not (camera_dir/'ready.json').exists():
            if camproc.poll() is not None or time.monotonic()>deadline:raise RuntimeError(f'Camera startup failed; {a.output / "camera.log"}')
            time.sleep(.1)
        if validation and json.loads((camera_dir/'ready.json').read_text())['serial']!=validation['intrinsics']['serial']:
            raise RuntimeError('Connected camera serial differs from the calibrated camera')
        if wrist and str(json.loads((camera_dir/'ready.json').read_text())['serial'])!=str(wrist['serial']):
            raise RuntimeError('Connected RealSense serial differs from the wrist plan')
        server=ThreadingHTTPServer(('127.0.0.1',a.port),Handler)
        threading.Thread(target=server.serve_forever,daemon=True).start()
        threading.Thread(target=worker,daemon=True).start()
        def camera_health():
            nonlocal cancel_epoch
            while not stop_workers.wait(.2):
                if a.mode!='live':continue
                try:healthy=camproc.poll() is None and time.monotonic()-json.loads((camera_dir/'status.json').read_text())['at']<1
                except (OSError,ValueError):healthy=False
                if not healthy and source.prepared:
                    sequence.pause()
                    source.stop=True;cancel_epoch+=1;source.message=camera_failure_message(camera_dir)
        threading.Thread(target=camera_health,daemon=True).start()
        threading.Thread(target=autopilot,daemon=True).start()
        task='Left-hand validation' if validation else 'Calibration'
        print(f'{task} {a.mode}: http://127.0.0.1:{a.port} | Click Start {"validation" if validation else "calibration"} to begin; Pause stops automatic advance.',flush=True)
        if a.probe:source.quit=True
        result=session.run(values,input_source=source)
        source.finished=True
        if source.failure_detail:
            (a.output/'failure-detail.json').write_text(json.dumps(source.failure_detail,indent=2))
        (a.output/'result.json').write_text(json.dumps(dict(session=result.get('finish_reason'),passed=result['passed'],captures=len(captured),solution=solution,
            fault=fault or (result.get('finish_reason') if not result['passed'] else None),failure_detail=source.failure_detail),indent=2))
        print(f'Finished: {result.get("finish_reason")}; captures={len(captured)}; {a.output}',flush=True)
        return 0 if result['passed'] and not fault else 1
    finally:
        stop_workers.set();source.quit=True
        if server:server.shutdown();server.server_close()
        (camera_dir/'stop').touch()
        if camproc:
            try:camproc.wait(timeout=5)
            except subprocess.TimeoutExpired:camproc.terminate();camproc.wait(timeout=5)
        if log:log.close()
        if robot_lock:robot_lock.close()
        checker.close()
        if source.simulation_plant is not None:source.simulation_plant.close()


if __name__=='__main__':raise SystemExit(main())
