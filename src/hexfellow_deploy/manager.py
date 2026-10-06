"""DA lifecycle and process health, independent of the control clock."""
import json
import multiprocessing as mp
import os
from pathlib import Path
import queue
import time
import traceback
from deploy_anything.manager import DAManager
from .scheduling import python_scheduling, runtime_state, validate_settings


class InterfaceManager(DAManager):
    def __init__(self, *args, **kwargs):
        self._python_scheduling = kwargs.pop('python_scheduling', {})
        for settings in self._python_scheduling.values():
            validate_settings(settings)
        self._health_topic = kwargs.get('topics', {}).get('health')
        output = os.environ.get('HEXFELLOW_OUTPUT')
        self._health_path = Path(output)/'component_health.json' if output else None
        self._health_events = mp.Queue(maxsize=64)
        self._health_state = dict(fault=None, components={})
        self._first_failed_component = None
        self._shutdown_requested = False
        self._health_saved = None
        super().__init__(*args, **kwargs)

    def _run_with_priority(self, name, func, priority=10, affinity=None):
        """Report lifecycle events, including exceptions outside component.run()."""
        def report(phase, **details):
            event = dict(name=name, phase=phase, pid=os.getpid(), observed_s=time.monotonic(), **details)
            try:
                self._health_events.put_nowait(event)
            except queue.Full:
                # Process exit detection still works when diagnostic delivery fails.
                self._logger.error('Lifecycle event queue full: %s', event)
        report('starting')
        try:
            with python_scheduling(self._python_scheduling.get(name, {})):
                def configured_run():
                    report('configured', python_runtime=runtime_state())
                    func()
                super()._run_with_priority(name, configured_run, priority, affinity)
        except KeyboardInterrupt:
            report('interrupted')
        except BaseException as error:
            report('failed', error=f'{type(error).__name__}: {error}',
                   traceback=traceback.format_exc()[-6000:])
            raise
        else:
            component = next((item for item in self._all_components if item.name == name), None)
            expected = (component.stopping if component else all(item.stopping for item in self._all_components))
            if expected:
                report('stopped')
            else:
                report('failed', error='UnexpectedExit: component returned without a stop request')

    def _poll_health(self):
        """Called by the process that started the children, never by a forked watcher."""
        rows = self._health_state['components']
        while True:
            try:
                event = self._health_events.get_nowait()
            except queue.Empty:
                break
            name = event.pop('name')
            row = rows.setdefault(name, {})
            # A late queue event must not erase a previously observed exit/error.
            if row.get('error') and not event.get('error'):
                event.pop('phase', None)
            row.update(event)
        by_name = {item.name: item for item in self._all_components}
        for process in self._processes:
            if process.pid is None:
                continue
            row = rows.setdefault(process.name, {})
            row.update(pid=process.pid, exitcode=process.exitcode)
            component = by_name.get(process.name)
            if component is not None:
                row.update(ready=component.ready, stopping=component.stopping)
            if process.exitcode is not None:
                expected = self._shutdown_requested or (component is not None and component.stopping)
                if not expected and row.get('phase') != 'interrupted' and not row.get('error'):
                    row.update(phase='failed', error=f'UnexpectedExit: process exited with code {process.exitcode}')
                elif not row.get('error'):
                    row['phase'] = 'stopped'
            elif not row.get('error'):
                row['phase'] = 'running' if component is None or component.ready else 'starting'
        if self._first_failed_component is None:
            failed = next(((name, row) for name, row in rows.items() if row.get('error')), None)
            if failed:
                self._first_failed_component = failed[0]
                self._logger.error('Component failure: %s: %s', failed[0], failed[1]['error'])
        if self._first_failed_component is not None:
            name = self._first_failed_component
            self._health_state['fault'] = f"{name}: {rows[name]['error']}"
        snapshot = dict(self._health_state, published_s=time.monotonic())
        if self._health_topic is not None:
            self._health_topic.put(snapshot, timeout=.001)
        serialized = json.dumps(self._health_state, indent=2)
        if self._health_path is not None and serialized != self._health_saved:
            self._health_path.parent.mkdir(parents=True, exist_ok=True)
            temporary = self._health_path.with_suffix('.tmp')
            temporary.write_text(json.dumps(snapshot, indent=2)+'\n')
            temporary.replace(self._health_path)
            self._health_saved = serialized
        return snapshot

    def _shutdown(self):
        self._shutdown_requested = True
        for component in self._all_components:
            component.signal_stop()
        for comm in self._comms:
            comm.signal_stop()
        self._stopping = True
        for process in self._processes:
            if process.pid is None:
                continue
            process.join(timeout=self._shutdown_timeout_s)
            if process.is_alive():
                self._logger.warning('Component %s did not stop in time. Terminating.', process.name)
                self._health_state['components'].setdefault(process.name, {}).update(
                    phase='failed', error='ShutdownTimeout: component did not stop after stop request')
                process.terminate()
                process.join(timeout=2.)
        self._poll_health()
        for component in self._all_components:
            component.clean_up()
        for comm in reversed(self._comms):
            comm.clean_up()
        self._health_events.close()
        self._health_events.join_thread()

    def run_all(self):
        if self._monitor_node is not None:
            raise ValueError('InterfaceManager uses shared-memory diagnostics, not ROS monitoring')
        try:
            for process in self._processes:
                process.start()
            # This loop observes process lifecycle only. Control, camera and policy
            # retain their own clocks; an ordinary budget miss is not an exit.
            while True:
                self._poll_health()
                time.sleep(.05)
        except KeyboardInterrupt:
            self._logger.info('Stopping deployment components...')
        finally:
            self._shutdown()

    def _run(self):
        # DA's parent _stopping is a plain bool, not shared after fork. For an
        # interface-only deployment use the components' shared stopping flags.
        # There is no inference/output barrier to schedule in this mode.
        if any((self._preprocess, self._input, self._inference, self._output)):
            raise ValueError("This manager owns independently clocked DA interfaces only")
        while not all(component.stopping for component in self._all_components):
            time.sleep(.01)
