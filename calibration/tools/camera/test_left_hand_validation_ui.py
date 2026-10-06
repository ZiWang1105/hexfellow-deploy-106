#!/usr/bin/env python3
"""Exercise only an already-running SIM validation page in isolated headless Chrome.

Uses real DOM button clicks, tests Pause/Resume, captures a screenshot, and quits
the SIM controller on completion. Refuses to interact with a live-mode page.
"""
import argparse
import base64
import json
from pathlib import Path
import subprocess
import tempfile
import time
import urllib.request
from websockets.sync.client import connect


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--port',type=int,default=8773)
    parser.add_argument('--debug-port',type=int,default=9773)
    parser.add_argument('--output',type=Path,required=True)
    a=parser.parse_args();a.output.mkdir(parents=True,exist_ok=True)
    url=f'http://127.0.0.1:{a.port}'
    # Refuse live BEFORE creating a browser (which supplies operator heartbeats).
    initial=json.load(urllib.request.urlopen(url+'/status',timeout=3))
    if initial['mode']!='sim' or initial['task']!='left_hand_validation':
        raise RuntimeError('UI test only supports the simulation validation page')
    with tempfile.TemporaryDirectory(prefix='left-validation-chrome-') as profile:
        browser=subprocess.Popen(['/usr/bin/google-chrome','--headless','--no-sandbox',
            '--disable-dev-shm-usage',f'--remote-debugging-port={a.debug_port}',
            f'--user-data-dir={profile}','--window-size=1400,1100',url],
            stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
        try:
            deadline=time.monotonic()+20
            while True:
                try:
                    targets=json.load(urllib.request.urlopen(f'http://127.0.0.1:{a.debug_port}/json',timeout=2))
                    target=next(t for t in targets if t.get('type')=='page' and t.get('url','').startswith(url))
                    break
                except (OSError,StopIteration):
                    if time.monotonic()>deadline:raise RuntimeError('Headless browser did not start')
                    time.sleep(.2)
            with connect(target['webSocketDebuggerUrl'],open_timeout=5,max_size=16*1024*1024) as websocket:
                sequence=0
                def call(method,params=None):
                    nonlocal sequence
                    sequence+=1;ident=sequence
                    websocket.send(json.dumps(dict(id=ident,method=method,params=params or {})))
                    while True:
                        result=json.loads(websocket.recv(timeout=10))
                        if result.get('id')!=ident:continue
                        if 'error' in result:raise RuntimeError(result['error'])
                        return result['result']
                def evaluate(expression):
                    result=call('Runtime.evaluate',dict(expression=expression,returnByValue=True,awaitPromise=True))
                    if result.get('exceptionDetails'):raise RuntimeError(result['exceptionDetails'])
                    return result['result'].get('value')
                while not evaluate("typeof s!=='undefined'&&s!==null"):
                    if time.monotonic()>deadline:raise RuntimeError('Validation JS did not load')
                    time.sleep(.1)
                if evaluate('s.mode')!='sim':raise RuntimeError('Refusing live UI interaction')
                evaluate("document.getElementById('start').click()")
                paused=False;resumed=False;pause_hold=None;history=[]
                deadline=time.monotonic()+200
                while time.monotonic()<deadline:
                    s=evaluate('s')
                    history.append({k:s.get(k) for k in ('phase','pose','target_pose','sequence','captured','message')})
                    if s['failure_detail']:raise RuntimeError(s['failure_detail'])
                    if not paused and s['phase']=='calibration':
                        evaluate("document.querySelector(\"button[onclick=\\\"act('pause')\\\"]\").click()")
                        paused=True
                    elif paused and not resumed and s['phase']=='hold' and s['sequence']=='paused':
                        if pause_hold is None:pause_hold=time.monotonic()
                        if time.monotonic()-pause_hold>=1:
                            assert not s['captured'],'Pause must not capture or advance'
                            evaluate("document.getElementById('start').click()")
                            resumed=True
                    if s['sequence']=='complete':
                        assert paused and resumed and s['captured']==[1,2,3] and s['target_pose']==-1,s
                        deadline_images=time.monotonic()+10
                        while not evaluate("document.querySelectorAll('#comparisons img').length===3&&Array.from(document.querySelectorAll('#comparisons img')).every(i=>i.complete&&i.naturalWidth>0)"):
                            if time.monotonic()>deadline_images:raise RuntimeError('Comparison images did not load')
                            time.sleep(.1)
                        screenshot=call('Page.captureScreenshot',dict(format='png',captureBeyondViewport=True))
                        (a.output/'completed-ui.png').write_bytes(base64.b64decode(screenshot['data']))
                        (a.output/'ui-report.json').write_text(json.dumps(dict(passed=True,simulation_only=True,
                            mode_verified_before_browser=True,pause_resume_tested=True,captures=s['captured'],
                            returned_to_default=True,loaded_comparisons=3,status=s),indent=2))
                        evaluate("document.querySelector(\"button[onclick=\\\"act('quit')\\\"]\").click()")
                        print('PASS: browser Start, Pause, Resume, 3 captures, return, 3 loaded comparisons, Quit',flush=True)
                        break
                    time.sleep(.15)
                else:raise TimeoutError('Simulation UI did not complete')
                (a.output/'ui-history.json').write_text(json.dumps(history,indent=2))
        finally:
            browser.terminate()
            try:browser.wait(timeout=5)
            except subprocess.TimeoutExpired:browser.kill();browser.wait(timeout=5)


if __name__=='__main__':main()
