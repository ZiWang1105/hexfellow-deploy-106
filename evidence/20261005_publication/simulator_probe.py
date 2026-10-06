"""Simulator-only reproduction using the real DA reference-topic semaphore."""
import json, os
from pathlib import Path
import subprocess, sys, time
from multiprocessing import resource_tracker
from hexfellow_deploy.topics import JsonTopic
root=Path.cwd(); output=root/'runs/shm_retry_204438_verified'
command=[sys.executable,'-m','hexfellow_deploy.cli','--workspace',str(root),'--duration-s','15',
 '--action','third_party/hexfellow-deploy/tests/fixtures/paired_pose.json',
 '--set','interface.control.startup_carry=false','--set','interface.dashboard.port=8875','--output',str(output)]
rows=[];frames=status=None
with (root/'runs/shm_retry_204438_harness/console.log').open('w') as log:
 process=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT,env=dict(os.environ,MUJOCO_GL='egl'))
 try:
  until=time.monotonic()+20.
  while not (output/'topics.json').exists():
   if process.poll() is not None or time.monotonic()>until:raise RuntimeError('Simulation startup failed')
   time.sleep(.02)
  names=json.loads((output/'topics.json').read_text())
  while time.monotonic()<until:
   try:
    frames=JsonTopic.attach(names['frames']);break
   except FileNotFoundError:time.sleep(.02)
  status=JsonTopic.attach(names['status'])
  # This external attacher does not own the manager's segments. Python 3.10
  # otherwise unlinks them on attacher exit, unlike DA's shared fork tracker.
  for topic in (frames,status):resource_tracker.unregister(topic._shm._name,'shared_memory')
  while time.monotonic()<until:
   state=status.get()
   if state and state.get('phase')=='running':break
   if process.poll() is not None:raise RuntimeError('Simulation ended before injection')
   time.sleep(.01)
  finish=time.monotonic()+8.
  while time.monotonic()<finish:
   state=status.get()
   if state and state.get('phase') in ('fault','closed'):break
   started=time.monotonic()
   with frames._hold(.01) as acquired:
    if acquired:time.sleep(.020)
   rows.append(dict(started_s=started,ended_s=time.monotonic(),acquired=acquired))
   time.sleep(max(0.,started+.08-time.monotonic()))
  result=process.wait(timeout=25)
 finally:
  if process.poll() is None:
   process.send_signal(2);process.wait(timeout=15)
  for topic in (frames,status):
   if topic is not None:topic.clean_up()
control=json.loads((output/'control_final.json').read_text());backend=json.loads((output/'backend_final.json').read_text())
ticks=[json.loads(x) for x in (output/'control.jsonl').read_text().splitlines()]
summary=dict(scope='Shared MIT MuJoCo stack, camera and dashboard enabled; no model or physical devices',exit_code=result,
 injections=len(rows),retries=sum(x['publication']['retries'] for x in ticks),
 retried_commands=sum(x['publication']['retries']>0 for x in ticks),generated=control['generated'],accepted=control['accepted'],
 control_fault=control['fault'],backend_fault=backend['fault'],shutdown_error=backend.get('shutdown_error'),
 cancelled_pending=control.get('cancelled_pending'))
(output/'contention_injection.json').write_text(json.dumps(rows,indent=2))
(output/'contention_summary.json').write_text(json.dumps(summary,indent=2))
print(json.dumps(summary))
assert result==0 and not control['fault'] and not backend['fault'] and not backend.get('shutdown_error')
assert summary['retries']>0 and summary['injections']>0
