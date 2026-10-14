#!/usr/bin/env python3
"""Run a left-hand overlay plan through deployment MIT dynamics, OFFLINE only.

The worker hard-codes sim mode and rejects every hardware gateway construction.
Measured simulation states are checked and rendered by the camera worktree in a
separate process so each checkout resolves its own model assets correctly.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]


def worker(plan_path, output):
    import time
    import numpy as np
    sys.path.insert(0, str(ROOT.parent/'src'))
    from hexfellow_deploy.robotics.hardware import session
    from calibration_runtime import CalibrationSource
    plan = json.loads(plan_path.read_text())
    parser = argparse.ArgumentParser()
    session.add_arguments(parser)
    args = parser.parse_args(['--mode', 'sim', '--robot-config', plan['mapping_file'],
        '--initial-state-report', str(plan_path), '--save-trace', str(output / 'session'),
        '--run-seconds', '180'])
    args.control_group = 'coupled'
    args.tracking_state = 'measured'
    _, profile, _, _ = session.load_configuration(args)
    args.allow_uncommissioned_profile_sha256 = profile.sha256

    class Source(CalibrationSource):
        def __init__(self):
            super().__init__(live=False)
            self.initialized = False
            self.sent = 0
            self.captured = []
            self.trace = []
            self.hold_since = None
            self.return_sent = False

        def validate_session(self, args, mapping, profile):
            if args.mode != 'sim':
                raise ValueError('This verifier is simulation only')
            super().validate_session(args, mapping, profile)

        def poll(self, state, engine, now, paused):
            self.heartbeat = time.monotonic()  # Automated SIM input only.
            request = super().poll(state, engine, now, paused)
            self.trace.append(dict(t=now, q=state.q.tolist(), v=state.v.tolist(),
                                   phase=engine.phase if engine else 'readonly'))
            if request is not None:
                return request
            if not self.initialized:
                self.initialized = True
                return 'default', None  # Existing adapter acquires current HOLD.
            if engine is None or engine.phase != 'hold':
                self.hold_since = None
                return None
            if self.hold_since is None:
                self.hold_since = now
            if now - self.hold_since < 1.:
                return None
            if self.sent and len(self.captured) < self.sent:
                if self.arrived_pose != self.sent:
                    raise RuntimeError('Simulation did not arrive at the requested pose')
                self.captured.append(dict(pose_id=self.sent, q=state.q.tolist(),
                    v=state.v.tolist(), target_q=plan['poses'][self.sent-1]['q'],
                    settling=self.pose_settling, t=now))
                print(f'Simulation captured measured left-hand pose {self.sent}', flush=True)
            if self.sent < len(plan['poses']):
                target = plan['poses'][self.sent]['q']
                self.sent += 1
                pose = self.sent
            elif not self.return_sent:
                target = plan['initial_q']
                pose = -1
                self.return_sent = True
            else:
                if self.arrived_pose != -1:
                    raise RuntimeError('Return to default did not complete')
                return 'quit', None
            self.hold_since = None
            return 'calibration-path', dict(pose_id=pose, origin=state.q.tolist(),
                reference_origin=engine.core._last_safe_positions.tolist(), targets=[target])

    def no_hardware(*args, **kwargs):
        raise RuntimeError('Hardware gateway forbidden in offline validation')
    source = Source()
    report = session.run(args, input_source=source, gateway_factory=no_hardware)
    passed = bool(report['passed'] and len(source.captured) == 3 and source.arrived_pose == -1)
    (output / 'measured-simulation.json').write_text(json.dumps(dict(
        passed=passed, simulation_only=True, hardware_commands_sent=0,
        finish_reason=report['finish_reason'], captures=source.captured,
        returned_to_default=source.arrived_pose == -1, trace=source.trace), indent=2))
    if not passed:
        raise RuntimeError(f"Simulation failed: {report['finish_reason']}")


def verify(plan_path, output):
    output.mkdir(parents=True, exist_ok=False)
    env = dict(os.environ, OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1',
               MKL_NUM_THREADS='1', MUJOCO_GL='egl')
    with (output / 'simulation.log').open('w') as log:
        subprocess.run([sys.executable, str(Path(__file__).resolve()), '--worker',
            '--plan', str(plan_path), '--output', str(output)], cwd=ROOT,
            env=env, stdout=log, stderr=subprocess.STDOUT, check=True, timeout=240)
    import numpy as np
    import cv2
    from validate_calibration_poses import Study
    from calibration_path import LocalChecker
    from validate_camera_overlay import OverlayRenderer, digest
    plan = json.loads(plan_path.read_text())
    measured = json.loads((output / 'measured-simulation.json').read_text())
    fit = json.loads(Path(plan['extrinsics_file']).read_text())
    if digest(plan['extrinsics_file']) != plan['extrinsics_sha256']:
        raise ValueError('Extrinsics changed since planning')
    study = Study({'q': plan['initial_q']}, output / 'geometry',
        attachment={'T_hand_from_board': fit['T_hand_from_board']}, include_left=True)
    checker = object.__new__(LocalChecker)
    checker.study = study
    bounds = np.array(plan['controller_bounds'])
    minimum = 1.
    previous = None
    for row in measured['trace']:
        q = np.asarray(row['q'])
        gap = checker.check_pose(q, bounds)['minimum_clearance_m']
        minimum = min(minimum, gap)
        if previous is not None and np.sum(abs(q-previous)) > .005:
            minimum = min(minimum, checker.edge_full(previous, q, bounds))
        previous = q
    renderer = OverlayRenderer(study.scene, fit['T_mount_from_camera'],
        plan['intrinsics']['K'], output / 'render', board_transform=fit['T_hand_from_board'])
    results = []
    try:
        for row in measured['captures']:
            rgb, mask = renderer.render(row['q'], 'left')
            ys, xs = np.where(mask)
            if len(xs) < 5000 or min(xs.min(), ys.min(), 1919-xs.max(), 1079-ys.max()) < 80:
                raise ValueError('Measured simulated gripper is not sufficiently visible')
            number = row['pose_id']
            cv2.imwrite(str(output / f'pose-{number:02d}-camera.png'), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
            cv2.imwrite(str(output / f'pose-{number:02d}-overview.png'), cv2.cvtColor(renderer.overview(row['q']), cv2.COLOR_RGB2BGR))
            results.append(dict(pose_id=number, visible_gripper_pixels=len(xs),
                maximum_left_target_error_deg=float(np.degrees(max(abs(np.array(row['q'])[4:11]-np.array(row['target_q'])[4:11]))))))
    finally:
        renderer.close()
    report = dict(passed=True, simulation_only=True, hardware_commands_sent=0,
        poses=results, returned_to_default=measured['returned_to_default'],
        minimum_measured_path_clearance_m=minimum, measured_samples=len(measured['trace']),
        plan_file=str(plan_path), plan_sha256=digest(plan_path),
        limitations='Nominal simulated dynamics, sampled geometry, no desks/cables/people. Actual starting state and room clearance must be checked before a physical run.')
    (output / 'report.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--worker', action='store_true', help=argparse.SUPPRESS)
    args = parser.parse_args()
    (worker if args.worker else verify)(args.plan.resolve(), args.output.resolve())
