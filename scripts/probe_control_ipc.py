"""Inject contention into the shared MuJoCo runtime; never open hardware."""
import argparse
import json
from multiprocessing import resource_tracker
import os
from pathlib import Path
import subprocess
import sys
import time

from hexfellow_deploy.control_topics import FeedbackTopic, ReferenceTopic, ReceiptTopic
from hexfellow_deploy.topics import JsonTopic


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workspace', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--channel', choices=['feedback','frames','receipts'], required=True)
    parser.add_argument('--hold-ms', type=float, required=True)
    parser.add_argument('--port', type=int, default=18775)
    args = parser.parse_args()
    if not 0 < args.hold_ms <= 60:
        parser.error('Use a bounded 0..60 ms diagnostic hold')
    root, output = args.workspace.resolve(), args.output.resolve()
    if output.exists():
        parser.error('Use a new output directory')
    output.parent.mkdir(parents=True, exist_ok=True)
    command = [sys.executable, '-m', 'hexfellow_deploy.cli', '--workspace', str(root),
        '--duration-s','15', '--action','third_party/hexfellow-deploy/tests/fixtures/paired_pose.json',
        '--set','interface.control.startup_carry=false', '--set',f'interface.dashboard.port={args.port}',
        '--output',str(output)]
    attached, injections = [], []
    log_path = output.with_name(output.name+'_console.log')
    with log_path.open('w') as log:
        process = subprocess.Popen(command, cwd=root, stdout=log, stderr=subprocess.STDOUT,
            env=dict(os.environ, MUJOCO_GL='egl', OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1'))
        try:
            until = time.monotonic()+25.
            while not (output/'topics.json').exists():
                if process.poll() is not None or time.monotonic() > until:
                    raise RuntimeError('Simulator failed to initialize topics')
                time.sleep(.02)
            names = json.loads((output/'topics.json').read_text())
            for cls,name in (({'feedback':FeedbackTopic,'frames':ReferenceTopic,'receipts':ReceiptTopic}[args.channel],
                               args.channel), (JsonTopic,'status')):
                while True:
                    try:
                        topic = cls.attach(names[name])
                        break
                    except FileNotFoundError:
                        if process.poll() is not None or time.monotonic() > until:
                            raise RuntimeError('Simulator topic did not start')
                        time.sleep(.02)
                attached.append(topic)
                # This external process is not the manager's segment owner.
                if sys.version_info < (3,13):
                    resource_tracker.unregister(topic._shm._name, 'shared_memory')
            channel, status = attached
            while True:
                state = status.get()
                if state and state.get('phase') == 'running':
                    break
                if process.poll() is not None or time.monotonic() > until:
                    raise RuntimeError('Simulator control did not start')
                time.sleep(.01)
            finish = time.monotonic()+8.
            while time.monotonic() < finish:
                state = status.get()
                if state and state.get('phase') in ('fault','closed'):
                    break
                started = time.monotonic()
                with channel._hold(.01) as acquired:
                    if acquired:
                        time.sleep(args.hold_ms/1000.)
                injections.append(dict(started_s=started, ended_s=time.monotonic(), acquired=acquired))
                time.sleep(max(0., started+.15-time.monotonic()))
            result = process.wait(timeout=25)
        finally:
            if process.poll() is None:
                process.send_signal(2)
                process.wait(timeout=15)
            for topic in attached:
                topic.clean_up()
    log_path.replace(output/'probe_console.log')
    control = json.loads((output/'control_final.json').read_text())
    backend = json.loads((output/'backend_final.json').read_text())
    ticks = [json.loads(x) for x in (output/'control.jsonl').read_text().splitlines()]
    summary = dict(scope='Shared MIT MuJoCo paired-pose fixture with camera/dashboard; no API or hardware',
        channel=args.channel, hold_ms=args.hold_ms, exit_code=result,
        injections=sum(x['acquired'] for x in injections), generated=control['generated'], accepted=control['accepted'],
        cancelled_pending=control.get('cancelled_pending'),
        feedback_retries=sum(x['feedback_refresh']['retries'] for x in ticks),
        publication_retries=sum(x['publication']['retries'] for x in ticks),
        max_source_age_ms=max(max(x['feedback_refresh']['source_age_ms'].values()) for x in ticks),
        max_previous_acceptance_ms=max(x['previous_acceptance_ms'] or 0 for x in ticks),
        control_fault=control['fault'], backend_fault=backend['fault'], shutdown_error=backend.get('shutdown_error'))
    (output/'contention_injection.json').write_text(json.dumps(injections,indent=2)+'\n')
    (output/'contention_summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps(summary))
    if result or control['fault'] or backend['fault'] or backend.get('shutdown_error') or not summary['injections']:
        raise RuntimeError('IPC probe failed; inspect saved diagnostics')


if __name__ == '__main__':
    main()
