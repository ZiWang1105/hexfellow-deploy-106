"""Read-only diagnostic server, independent from the control component."""
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import threading
import time
from deploy_anything.components.interface.base_interface import DABaseInterface

PAGE = b'''<!doctype html><html><meta charset="utf-8"><title>Hexfellow Deploy</title>
<style>body{font:16px system-ui;background:#111827;color:#e5e7eb;margin:32px}
pre{white-space:pre-wrap;background:#1f2937;padding:20px;border-radius:12px}
h1{font-size:24px}#phase{font-size:22px;color:#93c5fd}</style>
<h1>Hexfellow Deploy / Deploy Anything runtime</h1>
<p id="phase">Waiting for control telemetry</p><p>Read-only diagnostic view</p>
<img id="camera" width="640" alt="Latest RGB observation">
<h2>Control and actuator acceptance</h2><pre id="control"></pre>
<h2>Agent decision</h2><pre id="decision"></pre>
<script>async function refresh(){try{let r=await fetch('/status');let s=await r.json();
document.getElementById('phase').textContent=s.control?.phase||'Starting';
document.getElementById('control').textContent=JSON.stringify(s.control,null,2);
document.getElementById('decision').textContent=JSON.stringify(s.decision,null,2);
document.getElementById('camera').src='/rgb.jpg?t='+Date.now();
}catch(e){document.getElementById('phase').textContent='Diagnostic connection unavailable'}
setTimeout(refresh,500)}refresh();</script></html>'''


class DashboardComponent(DABaseInterface):
    def __init__(self, name, status, decision, images, port=8765, time_budget=.01):
        super().__init__(name, [status, decision, images], time_budget)
        self.status, self.decision, self.images, self.port = status, decision, images, port

    def initialize(self):
        self._initialized.value = True
        return True

    def clean_up(self):
        return True

    def run(self):
        owner = self
        class Handler(BaseHTTPRequestHandler):
            def do_GET(self):
                if self.path == '/':
                    body, content_type = PAGE, 'text/html; charset=utf-8'
                elif self.path == '/status':
                    body = json.dumps(dict(control=owner.status.get(timeout=.001),
                                           decision=owner.decision.get(timeout=.001))).encode()
                    content_type = 'application/json'
                elif self.path.startswith('/rgb.jpg'):
                    import cv2
                    frame = owner.images.get()
                    if frame is None:
                        self.send_error(503, 'Waiting for camera')
                        return
                    body = cv2.imencode('.jpg', cv2.cvtColor(frame['rgb'], cv2.COLOR_RGB2BGR))[1].tobytes()
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
