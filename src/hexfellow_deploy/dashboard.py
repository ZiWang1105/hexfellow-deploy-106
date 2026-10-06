"""Read-only diagnostic server, independent from the control component."""
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import threading
import time
from deploy_anything.components.interface.base_interface import DABaseInterface

PAGE = br'''<!doctype html><html><meta charset="utf-8"><title>Hexfellow Deploy</title>
<style>body{font:16px system-ui;background:#111827;color:#e5e7eb;margin:32px}
pre{white-space:pre-wrap;overflow-wrap:anywhere;background:#1f2937;padding:16px;border-radius:12px}
h1{font-size:24px}#phase{font-size:22px;color:#93c5fd}
.agent{border:1px solid #3b82f6;border-radius:12px;padding:20px;margin:16px 0;background:#17243b}
.agent h2{margin-top:0}#agent-phase{font-size:22px;color:#93c5fd}#agent-message{font-size:18px}
.targets{display:grid;grid-template-columns:repeat(auto-fit,minmax(250px,1fr));gap:12px}
.muted{color:#cbd5e1}details{margin:16px 0}summary{cursor:pointer;font-size:18px}
[hidden]{display:none!important}
#failure{background:#7f1d1d;padding:16px;white-space:pre-wrap}.views{display:flex;flex-wrap:wrap;gap:16px}
figure{margin:0}img{max-width:100%;height:auto}figcaption{margin:8px 0;color:#cbd5e1}</style>
<h1>Hexfellow Deploy / Deploy Anything runtime</h1>
<p id="phase">Waiting for control telemetry</p><p>Read-only diagnostic view</p>
<p id="failure" hidden></p>
<section class="agent"><h2>Agent activity</h2><div id="agent-phase">Waiting for agent telemetry</div>
<p id="agent-message">No decision received yet.</p><p id="agent-meta" class="muted"></p>
<p id="instruction"></p><h3 id="action-title">Latest agent decision</h3>
<p id="no-action">No model output yet.</p><div class="targets" id="targets" hidden>
<div><b>Base navigation</b><pre id="navigation"></pre></div>
<div><b>Left TCP / gripper</b><pre id="left-target"></pre></div>
<div><b>Right TCP / gripper</b><pre id="right-target"></pre></div></div>
<details><summary>Recent agent events</summary><pre id="events"></pre></details></section>
<div class="views"><figure>
<img id="camera" width="640" alt="Latest RGB observation"><figcaption>RGB observation</figcaption></figure><figure>
<img id="depth" width="640" alt="Registered metric depth"><figcaption>Registered depth: 0-3 m; black = invalid</figcaption></figure></div>
<details><summary>Component health</summary><pre id="health"></pre></details>
<details><summary>Control and actuator acceptance</summary><pre id="control"></pre></details>
<details><summary>Agent decision / raw telemetry</summary><pre id="decision"></pre></details>
<details><summary>Camera matching diagnostics</summary><pre id="camera-health"></pre></details>
<script>async function refresh(){try{let r=await fetch('/status');let s=await r.json();
document.getElementById('phase').textContent=s.control?.phase||'Starting';
let failure=s.control?.fault||s.backend?.fault||s.health?.fault;
document.getElementById('failure').hidden=!failure;
document.getElementById('failure').textContent=failure||'';
let a=s.decision||{}, d=s.control?.diagnostic||{};
let stage=d.mode==='startup_carry'?'Preparing carrying pose: stage '+(d.startup_stage||'?')+'/'+(d.startup_stages||'?'):null;
document.getElementById('agent-phase').textContent=(a.phase||'waiting_for_control').replaceAll('_',' ');
document.getElementById('agent-message').textContent=failure
 ? ((a.policy_calls||0)===0?'Stopped before the first model call. ':'Execution stopped. ')+failure
 : stage||(a.message||'Waiting for agent readiness; no decision received yet.');
let thinking=a.phase==='thinking'&&a.thinking_started_s?'; waiting '+Math.max(0,s.observed_s-a.thinking_started_s).toFixed(1)+' s':'';
document.getElementById('agent-meta').textContent='Model: '+(a.model||'pending')+' / '+(a.reasoning_effort||'pending')
 +' | Calls: '+(a.policy_calls||0)+' | Cost: $'+(a.cost_usd||0).toFixed(4)+thinking;
document.getElementById('instruction').textContent=a.prompt?'Task: '+a.prompt:'';
document.getElementById('action-title').textContent='Latest agent decision'+(a.action_observation_id?' (observation '+a.action_observation_id+')':'');
document.getElementById('no-action').hidden=!!a.action;
document.getElementById('targets').hidden=!a.action;
if(a.action){document.getElementById('navigation').textContent=JSON.stringify(a.action.navigation,null,2)||'No base motion';
const vector=v=>'['+v.map(x=>Number(x).toFixed(3)).join(', ')+']';
for(let side of ['left','right']){let t=a.action[side];document.getElementById(side+'-target').textContent=t
 ? 'Position (m): '+vector(t.position_m)+'\nQuaternion (xyzw): '+vector(t.quaternion_xyzw)+'\nGripper: '+(t.gripper_open===1?'OPEN':'CLOSE'):'';}}
document.getElementById('events').textContent=(a.events||[]).map(e=>e.phase+': '+(e.message||e.error||'')).join('\n')||'No agent events yet.';
document.getElementById('health').textContent=JSON.stringify(s.health,null,2);
document.getElementById('control').textContent=JSON.stringify(s.control,null,2);
document.getElementById('decision').textContent=JSON.stringify(s.decision,null,2);
document.getElementById('camera-health').textContent=JSON.stringify(s.camera,null,2);
document.getElementById('camera').src='/rgb.jpg?t='+Date.now();
document.getElementById('depth').src='/depth.jpg?t='+Date.now();
}catch(e){document.getElementById('phase').textContent='Diagnostic connection unavailable'}
setTimeout(refresh,500)}refresh();</script></html>'''


class DashboardComponent(DABaseInterface):
    def __init__(self, name, status, decision, images, port=8765, time_budget=.01, health=None,
                 backend_status=None, output=None):
        super().__init__(name, [status, decision, images]+[t for t in (health,backend_status) if t is not None], time_budget)
        self.status, self.decision, self.images, self.port = status, decision, images, port
        self.health = health
        self.backend_status, self.output = backend_status, output
        self._latest = {}
        self._snapshot_lock = threading.Lock()

    def initialize(self):
        self._initialized.value = True
        return True

    def clean_up(self):
        self._cleaned_up.value = True
        return True

    def snapshot(self):
        # A transient try-lock miss must not erase the last visible decision.
        with self._snapshot_lock:
            for name, topic in [('control',self.status),('decision',self.decision),
                                ('health',self.health),('backend',self.backend_status)]:
                value = None if topic is None else topic.get(timeout=.001)
                if value is not None:
                    self._latest[name] = value
            if self.output is not None:
                for name, filename in [('camera','camera_health.json'),('control','control_final.json'),
                                       ('backend','backend_final.json')]:
                    try:
                        self._latest[name] = json.loads((Path(self.output)/filename).read_text())
                    except (OSError, ValueError):
                        pass
                # Policy events are durable even if their last SHM write was busy.
                try:
                    with (Path(self.output)/'decisions.jsonl').open('rb') as stream:
                        stream.seek(0, 2)
                        stream.seek(max(0, stream.tell()-65536))
                        lines = stream.read().splitlines()
                    for line in reversed(lines):
                        try:
                            candidate = json.loads(line)
                        except ValueError:
                            continue
                        if candidate.get('published_s',0) >= self._latest.get('decision',{}).get('published_s',0):
                            self._latest['decision'] = candidate
                        break
                except OSError:
                    pass
            return dict(self._latest, observed_s=time.monotonic())

    def run(self):
        owner = self
        class Handler(BaseHTTPRequestHandler):
            def do_GET(self):
                if self.path == '/':
                    body, content_type = PAGE, 'text/html; charset=utf-8'
                elif self.path == '/status':
                    body = json.dumps(owner.snapshot()).encode()
                    content_type = 'application/json'
                elif self.path.startswith(('/rgb.jpg', '/depth.jpg')):
                    import cv2
                    frame = owner.images.get()
                    if frame is None:
                        self.send_error(503, 'Waiting for camera')
                        return
                    if self.path.startswith('/depth.jpg'):
                        import numpy as np
                        depth = frame['depth_m']
                        valid = np.isfinite(depth) & (depth > 0)
                        intensity = np.zeros(depth.shape, dtype=np.uint8)
                        intensity[valid] = np.clip(depth[valid]/3.*255, 0, 255).astype(np.uint8)
                        pixels = cv2.applyColorMap(intensity, cv2.COLORMAP_TURBO)
                        pixels[~valid] = 0
                    else:
                        pixels = cv2.cvtColor(frame['rgb'], cv2.COLOR_RGB2BGR)
                    body = cv2.imencode('.jpg', pixels)[1].tobytes()
                    content_type = 'image/jpeg'
                else:
                    self.send_error(404)
                    return
                self.send_response(200)
                self.send_header('Content-Type', content_type)
                self.send_header('Content-Length', str(len(body)))
                self.send_header('Cache-Control', 'no-store')
                self.end_headers()
                try:
                    self.wfile.write(body)
                except (BrokenPipeError, ConnectionResetError):
                    pass
            def log_message(self, *_):
                pass
        server = ThreadingHTTPServer(('127.0.0.1', self.port), Handler)
        server.daemon_threads = True
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        self._ready.value = True
        print(f'Hexfellow deployment dashboard: http://127.0.0.1:{server.server_port}/', flush=True)
        try:
            while not self.stopping:
                time.sleep(.1)
        finally:
            server.shutdown()
            server.server_close()
            thread.join(2)
