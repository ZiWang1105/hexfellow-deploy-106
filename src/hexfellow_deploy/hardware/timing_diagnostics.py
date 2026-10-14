"""Opt-in out-of-process stall capture; never sends robot commands.

Heartbeat words are advisory diagnostics, not synchronization or safety state.
SIGUSR2 is handled by faulthandler's native handler, including during GIL stalls.
"""
from collections import deque
import faulthandler
import json
import multiprocessing as mp
import os
from pathlib import Path
import resource
import signal
import threading
import time


PHASES = {0: 'inactive', 1: 'mailbox_read', 2: 'native_tick',
          3: 'gripper_feedback', 4: 'state_publication', 5: 'wake_wait'}


def thread_scheduling(pid):
    result = {}
    for path in Path(f'/proc/{pid}/task').glob('*'):
        try:
            result[path.name] = dict(name=(path / 'comm').read_text().strip(),
                schedstat=(path / 'schedstat').read_text().strip(),
                wchan=(path / 'wchan').read_text().strip())
        except OSError:
            pass
    return result


def _watch(pid, heartbeat, stop, ready, directory, threshold):
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    last_capture, count = 0., 0
    previous_threads = thread_scheduling(pid)
    previous_time = time.monotonic()
    with (Path(directory) / 'stalls.jsonl').open('w', buffering=1) as output:
        ready.set()
        while not stop.wait(.003):
            now = time.monotonic()
            phase, stamp, source = int(heartbeat[0]), heartbeat[1], heartbeat[2]
            stalled = now - stamp > threshold
            stale = source > 0 and now - source > threshold
            if phase and (stalled or stale) and now - last_capture > .1 and count < 32:
                current_threads = thread_scheduling(pid)
                output.write(json.dumps(dict(observed_s=now, phase=PHASES.get(phase),
                    heartbeat_s=stamp, age_ms=1000*(now-stamp),
                    reason='gateway_phase_stall' if stalled else 'feedback_age',
                    source_age_ms=1000*(now-source) if source else None,
                    baseline_s=previous_time, baseline_threads=previous_threads,
                    threads=current_threads)) + '\n')
                try:
                    os.kill(pid, signal.SIGUSR2)
                except ProcessLookupError:
                    break
                last_capture, count = now, count + 1
            if now - previous_time >= .1:
                previous_threads, previous_time = thread_scheduling(pid), now


class GatewayTiming:
    def __init__(self, directory, threshold_s=.025):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.events = deque(maxlen=4096)
        self.snapshots = deque(maxlen=64)
        self.slow_snapshots = deque(maxlen=128)
        self.snapshot_count = 0
        self.slow_snapshot_count = 0
        self.max_source_age_ms = {}
        self.thread_ids = []
        self.context = mp.get_context('spawn')
        self.heartbeat = self.context.RawArray('d', 3)
        self.stop, ready = self.context.Event(), self.context.Event()
        self.stack_file = (self.directory / 'stacks.txt').open('w')
        faulthandler.register(signal.SIGUSR2, file=self.stack_file, all_threads=True)
        self.process = self.context.Process(target=_watch,
            args=(os.getpid(), self.heartbeat, self.stop, ready, str(self.directory), threshold_s),
            name='gateway-stall-observer')
        self.process.start()
        if not ready.wait(15):
            self.close()
            raise RuntimeError('Timing observer did not start')

    def mark(self, phase):
        now = time.monotonic()
        self.events.append((now, PHASES[phase]))
        self.heartbeat[1] = now
        self.heartbeat[0] = phase

    def instrument_snapshot(self, read):
        """Separate snapshot CPU work from elapsed waits without changing reads."""
        def measured_read():
            before = resource.getrusage(resource.RUSAGE_THREAD)
            cpu = time.thread_time()
            started = time.monotonic()
            try:
                return read()
            finally:
                ended = time.monotonic()
                cpu_ms = 1000 * (time.thread_time() - cpu)
                after = resource.getrusage(resource.RUSAGE_THREAD)
                wall_ms = 1000 * (ended - started)
                row = dict(started_s=started, ended_s=ended,
                    thread_name=threading.current_thread().name, native_id=threading.get_native_id(),
                    wall_ms=wall_ms, thread_cpu_ms=cpu_ms,
                    non_cpu_ms=max(0., wall_ms-cpu_ms),
                    voluntary_switches=after.ru_nvcsw-before.ru_nvcsw,
                    involuntary_switches=after.ru_nivcsw-before.ru_nivcsw)
                self.snapshots.append(row)
                self.snapshot_count += 1
                if wall_ms >= 2:
                    self.slow_snapshot_count += 1
                    self.slow_snapshots.append(row)
        return measured_read

    def observe(self, received):
        now = time.monotonic()
        if not self.thread_ids:
            self.thread_ids = [dict(name=t.name, python_id=t.ident, native_id=t.native_id)
                               for t in threading.enumerate()]
        self.heartbeat[2] = min(received.values())
        for name, stamp in received.items():
            self.max_source_age_ms[name] = max(self.max_source_age_ms.get(name, 0.), 1000*(now-stamp))

    def close(self):
        self.heartbeat[0] = 0
        self.stop.set()
        self.process.join(3)
        if self.process.is_alive():
            self.process.terminate()
            self.process.join(3)
        faulthandler.unregister(signal.SIGUSR2)
        self.stack_file.close()

    def save(self, endpoint, supervisor):
        from hexfellow_deploy.robotics.hardware.ik_runner import jsonable
        window = (getattr(supervisor, 'fault_window', None)
                  or getattr(endpoint, 'stop_window', None))
        if window is None and endpoint is not None:
            window = endpoint.diagnostic_window()
        result = dict(pid=os.getpid(), phases=list(self.events),
            feedback_worker=getattr(endpoint, 'feedback_worker_report', None),
            snapshot_timing=list(self.snapshots),
            slow_snapshots=list(self.slow_snapshots), snapshot_count=self.snapshot_count,
            slow_snapshot_count=self.slow_snapshot_count,
            max_source_age_ms=self.max_source_age_ms,
            feedback_latency={n: t.latencies.report() for n, t in getattr(endpoint, 'taps', {}).items()},
            last_health=(supervisor.last_state.health if supervisor and supervisor.last_state is not None else None),
            threads=self.thread_ids,
            command_timing=list(supervisor.command_timing) if supervisor else [],
            transport=window, clock='host monotonic; kernel conversions are estimates',
            note='Diagnostic sampling can perturb timing; bounded windows are not a full packet capture.')
        (self.directory / 'gateway.json').write_text(json.dumps(jsonable(result), indent=2)+'\n')
