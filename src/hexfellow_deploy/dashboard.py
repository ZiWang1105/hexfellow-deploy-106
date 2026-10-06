"""Read-only diagnostic server, independent from the control component."""
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import threading
import time
from deploy_anything.components.interface.base_interface import DABaseInterface

PAGE = b'''<!doctype html><html><meta charset="utf-8"><title>Hexfellow Deploy</title>
<style>body{font:16px system-ui;background:#111827;color:#e5e7eb;margin:32px}
pre{white-space:pre-wrap;background:#1f2937;padding:20px;border-radius:12px}
h1{font-size:24px}#phase{font-size:22px;color:#93c5fd}
#failure{background:#7f1d1d;padding:16px;white-space:pre-wrap}.views{display:flex;flex-wrap:wrap;gap:16px}
figure{margin:0}img{max-width:100%;height:auto}figcaption{margin:8px 0;color:#cbd5e1}</style>
<h1>Hexfellow Deploy / Deploy Anything runtime</h1>
<p id="phase">Waiting for control telemetry</p><p>Read-only diagnostic view</p>
<p id="failure" hidden></p><div class="views"><figure>
<img id="camera" width="640" alt="Latest RGB observation"><figcaption>RGB observation</figcaption></figure><figure>
<img id="depth" width="640" alt="Registered metric depth"><figcaption>Registered depth: 0-3 m; black = invalid</figcaption></figure></div>
<h2>Component health</h2><pre id="health"></pre>
<h2>Control and actuator acceptance</h2><pre id="control"></pre>
<h2>Agent decision</h2><pre id="decision"></pre>
<script>async function refresh(){try{let r=await fetch('/status');let s=await r.json();
document.getElementById('phase').textContent=s.control?.phase||'Starting';
let failure=s.health?.fault||s.control?.fault;
document.getElementById('failure').hidden=!failure;
document.getElementById('failure').textContent=failure||'';
document.getElementById('health').textContent=JSON.stringify(s.health,null,2);
document.getElementById('control').textContent=JSON.stringify(s.control,null,2);
document.getElementById('decision').textContent=JSON.stringify(s.decision,null,2);
document.getElementById('camera').src='/rgb.jpg?t='+Date.now();
document.getElementById('depth').src='/depth.jpg?t='+Date.now();
}catch(e){document.getElementById('phase').textContent='Diagnostic connection unavailable'}
setTimeout(refresh,500)}refresh();</script></html>'''


class DashboardComponent(DABaseInterface):
    def __init__(self, name, status, decision, images, port=8765, time_budget=.01, health=None):
        super().__init__(name, [status, decision, images]+([] if health is None else [health]), time_budget)
        self.status, self.decision, self.images, self.port = status, decision, images, port
        self.health = health

    def initialize(self):
        self._initialized.value = True
        return True

    def clean_up(self):
        self._cleaned_up.value = True
        return True

    def snapshot(self):
        return dict(control=self.status.get(timeout=.001), decision=self.decision.get(timeout=.001),
                    health=None if self.health is None else self.health.get(timeout=.001))

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
