#!/usr/bin/env python3
"""Local operator rehearsal using the existing validated simulation artifacts.

No hardware backend is exposed. Captures are explicitly synthetic and use the
paired image/FK observations from the independently verified simulation run.
"""
import argparse
from datetime import datetime, timezone
import hashlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import secrets
import shutil
import sys
import threading
import time
from urllib.parse import urlparse

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src'))
from hexfellow_deploy.robotics.calibration.handeye import solve, summarize
from hexfellow_deploy.robotics.calibration.operator import Rehearsal


class Console:
    def __init__(self, route, verified, output):
        self.route, self.verified, self.output = route, verified, output
        physics = json.loads((route / 'physics_report.json').read_text())
        if not physics['simulation_only'] or not physics['no_modeled_collision']:
            raise ValueError('Expected passing, simulation-only physics report')
        self.trace = np.load(route / 'physics_trace.npz')
        self.pairs = np.load(verified / 'observations.npz')
        self.detections = json.loads((verified / 'detections.json').read_text())
        if len(physics['dwells']) != 30 or len(self.detections) != 30:
            raise ValueError('Expected the verified 30-pose dataset')
        self.machine = Rehearsal(physics['dwells'], physics['duration_s'])
        self.lock = threading.RLock()
        self.observations = {}
        self.attempts = {}
        self.solution = None
        self.output.mkdir(parents=True, exist_ok=False)
        self.metadata = dict(simulation_only=True, mode='simulation_rehearsal',
            mount_frame='maver_x4_asm_URDF_link_1', hand_frame='maver_x4_asm_mount2',
            camera_frame='zed_left_camera_frame_optical',
            board=json.loads((ROOT / 'config/camera/charuco_87mm.json').read_text()),
            camera=json.loads((ROOT / 'config/camera/zed_view_study.json').read_text()),
            source_hashes={str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in (
                route / 'route.json', route / 'physics_trace.npz', verified / 'observations.npz')})
        self.save_manifest()

    def event(self, action, **details):
        with (self.output / 'events.jsonl').open('a') as f:
            f.write(json.dumps(dict(at=datetime.now(timezone.utc).isoformat(), action=action,
                                    phase=self.machine.phase, pose=self.machine.pose, **details)) + '\n')

    def save_manifest(self):
        document = dict(self.metadata, observations=[self.observations[k] for k in sorted(self.observations)])
        temp = self.output / 'dataset.tmp'
        temp.write_text(json.dumps(document, indent=2, allow_nan=False))
        temp.replace(self.output / 'dataset.json')

    def capture(self):
        if not self.machine.status()['can_capture']:
            raise ValueError('Capture disabled until the pose has settled.')
        i = self.machine.pose - 1
        row = self.detections[i]
        stop = self.machine.dwells[i]
        if row['pose_id'] != self.machine.pose or not row['accepted'] or row['corners'] < 12:
            raise ValueError('Board detection did not pass the capture gate.')
        if np.max(np.abs(stop['v'])) >= .02:
            raise ValueError('Recorded robot velocity did not pass the stationary gate.')
        idx = int(np.argmin(abs(self.trace['time'] - stop['time_s'])))
        attempt = self.attempts.get(self.machine.pose, 0) + 1
        folder = self.output / f'pose_{self.machine.pose:02}_attempt_{attempt:02}'
        folder.mkdir(exist_ok=False)
        shutil.copyfile(self.verified / f'pose_{self.machine.pose:02}_raw.png', folder / 'left.png')
        observation = dict(row, stationary=True, synchronized=True, simulation_only=True,
            clock='simulation_seconds', image_time_s=float(self.trace['time'][idx]),
            joint_time_s=float(self.trace['time'][idx]),
            q=self.trace['q'][idx].tolist(), v=stop['v'],
            T_mount_from_hand=self.pairs['g'][i].tolist(),
            image=str((folder / 'left.png').relative_to(self.output)), attempt=attempt)
        (folder / 'observation.json').write_text(json.dumps(observation, indent=2, allow_nan=False))
        self.observations[self.machine.pose] = observation
        self.attempts[self.machine.pose] = attempt
        self.save_manifest()

    def command(self, action, version, now):
        if action != 'pause' and version != self.machine.version:
            raise ValueError('Status changed; stale or duplicate action rejected.')
        if action == 'capture':
            self.capture()
        if action == 'solve':
            if not self.machine.status()['can_solve']:
                raise ValueError('Capture all 30 poses before solving.')
            rows = [self.observations[k] for k in sorted(self.observations)]
            g = np.array([r['T_mount_from_hand'] for r in rows])
            c = np.array([r['T_camera_from_board'] for r in rows])
            fit = np.arange(len(rows)) % 5 != 4
            result = solve(g[fit], c[fit])
            x, y = np.array(result['T_mount_from_camera']), np.array(result['T_hand_from_board'])
            result.update(validation=summarize(g[~fit], c[~fit], x, y), simulation_only=True,
                          calibration_status='synthetic_rehearsal_only',
                          training_pose_ids=[r['pose_id'] for r, use in zip(rows, fit) if use],
                          validation_pose_ids=[r['pose_id'] for r, use in zip(rows, fit) if not use])
            path = self.output / f'solution_{time.time_ns()}.json'
            path.write_text(json.dumps(result, indent=2, allow_nan=False))
            self.solution = dict(file=path.name, converged=result['converged'], validation=result['validation'])
            self.machine.version += 1
            self.machine.message = 'Synthetic solution saved; no real calibration has been applied.'
        else:
            self.machine.command(action, now=now, version=version)
        self.event(action)

    def status(self):
        result = self.machine.status()
        result.update(output=str(self.output), solution=self.solution,
                      corners=self.detections[self.machine.pose - 1]['corners'] if self.machine.pose else None)
        return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--port', type=int, default=8770)
    parser.add_argument('--route', type=Path, default=ROOT / 'runs/camera-calibration-40deg/route')
    parser.add_argument('--verified', type=Path, default=ROOT / 'runs/camera-calibration-40deg/ground-truth-verification')
    parser.add_argument('--output', type=Path, default=ROOT / 'runs/calibration-console' /
                        datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S.%fZ'))
    args = parser.parse_args()
    console = Console(args.route, args.verified, args.output)
    token = secrets.token_urlsafe(32)
    page = (ROOT.parent / 'src/hexfellow_deploy/robotics/calibration/operator.html').read_text().replace('__TOKEN__', token)

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def send(self, code, content, mime='application/json'):
            if not isinstance(content, bytes):
                content = content.encode() if isinstance(content, str) else json.dumps(content).encode()
            self.send_response(code)
            self.send_header('Content-Type', mime)
            self.send_header('Content-Length', str(len(content)))
            self.send_header('Cache-Control', 'no-store')
            self.send_header('X-Content-Type-Options', 'nosniff')
            self.end_headers()
            self.wfile.write(content)

        def local_host(self):
            return self.headers.get('Host') in (f'127.0.0.1:{args.port}', f'localhost:{args.port}')

        def do_GET(self):
            if not self.local_host():
                return self.send(403, {'error': 'Local browser only'})
            path = urlparse(self.path).path
            if path == '/':
                return self.send(200, page, 'text/html; charset=utf-8')
            if path == '/status':
                with console.lock:
                    return self.send(200, console.status())
            if path == '/video.mp4':
                return self.send(200, (args.route / 'calibration_simulation_6x.mp4').read_bytes(), 'video/mp4')
            if path == '/image.png':
                with console.lock:
                    pose = console.machine.pose
                    if console.machine.phase != 'at_pose' or not pose:
                        return self.send(404, {'error': 'No current settled-pose image'})
                return self.send(200, (args.verified / f'pose_{pose:02}_raw.png').read_bytes(), 'image/png')
            return self.send(404, {'error': 'Not found'})

        def do_POST(self):
            if not self.local_host() or self.headers.get('X-Calibration-Token') != token:
                return self.send(403, {'error': 'Invalid local session token'})
            try:
                size = int(self.headers.get('Content-Length', '0'))
                if not 0 < size <= 4096:
                    raise ValueError('Invalid request size')
                body = json.loads(self.rfile.read(size))
                with console.lock:
                    if self.path == '/heartbeat':
                        console.machine.heartbeat(time.monotonic())
                    elif self.path == '/command':
                        console.command(body['action'], body['version'], time.monotonic())
                    else:
                        return self.send(404, {'error': 'Not found'})
                    result = console.status()
                return self.send(200, result)
            except (ValueError, KeyError, OSError) as e:
                return self.send(409, {'error': str(e)})

    server = ThreadingHTTPServer(('127.0.0.1', args.port), Handler)
    server.daemon_threads = True
    worker = threading.Thread(target=server.serve_forever, daemon=True)
    worker.start()
    print(f'Calibration rehearsal: http://127.0.0.1:{args.port}', flush=True)
    print(f'Synthetic captures: {args.output}', flush=True)
    previous = time.monotonic()
    try:
        while True:
            now = time.monotonic()
            with console.lock:
                old = console.machine.phase
                console.machine.tick(now - previous, now)
                if old != console.machine.phase:
                    console.event('phase', previous=old)
            previous = now
            time.sleep(.02)
    except KeyboardInterrupt:
        pass
    finally:
        server.shutdown()
        server.server_close()
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
