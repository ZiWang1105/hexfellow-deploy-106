"""JSON conversion and disk writes run in a separate, bounded logger process."""
import json
import multiprocessing as mp
import pickle
import queue
import time

from .x7 import HardwareSafetyError


def write_rows(directory, rows, errors):
    from .ik_runner import jsonable
    from .stream_diagnostics import StreamDiagnostics
    diagnostics = StreamDiagnostics()
    last_report = 0.
    try:
        with (directory / 'trace.jsonl').open('x') as output:
            while True:
                value = rows.get()
                if value is None:
                    break
                row = jsonable(pickle.loads(value))
                output.write(json.dumps(row, allow_nan=False) + '\n')
                diagnostics.record(row)
                if time.monotonic() - last_report >= 1.:
                    temporary = directory / 'stream-status.tmp'
                    temporary.write_text(json.dumps(diagnostics.report(), allow_nan=False))
                    temporary.replace(directory / 'stream-status.json')
                    last_report = time.monotonic()
        (directory / 'stream-status.json').write_text(json.dumps(diagnostics.report(), allow_nan=False))
    except BaseException as error:
        errors.send(f'{type(error).__name__}: {error}')
    finally:
        errors.close()


class TraceWriter:
    def __init__(self, directory):
        directory.mkdir(parents=True, exist_ok=False)
        self.directory = directory
        context = mp.get_context('spawn')
        self.rows = context.Queue(maxsize=512)
        self.errors, sender = context.Pipe(duplex=False)
        self.process = context.Process(target=write_rows, args=(directory, self.rows, sender),
                                       name='hardware-trace', daemon=True)
        self.process.start()
        sender.close()
        self.error = None
        self.drop_on_full = False
        self.dropped = 0

    def check(self):
        if self.errors.poll():
            try:
                self.error = self.errors.recv()
            except EOFError:
                pass
        if self.error or not self.process.is_alive():
            raise HardwareSafetyError(f'trace writer failed: {self.error or self.process.exitcode}')

    def record(self, value):
        self.check()
        try:
            self.rows.put_nowait(pickle.dumps(value, protocol=pickle.HIGHEST_PROTOCOL))
        except queue.Full as error:
            if self.drop_on_full:
                self.dropped += 1
                return
            raise HardwareSafetyError('trace buffer full; refusing unrecorded motion') from error

    def close(self, summary):
        from .ik_runner import jsonable
        summary['trace_dropped_rows'] = self.dropped
        summary['trace_complete'] = self.dropped == 0
        try:
            if self.process.is_alive():
                self.rows.put(None, timeout=2)
                self.process.join(timeout=5)
            if self.errors.poll():
                try:
                    self.error = self.errors.recv()
                except EOFError:
                    pass
            if self.process.is_alive() or self.error or self.process.exitcode:
                summary['trace_error'] = str(self.error or 'trace writer did not finish successfully')
        except Exception as error:
            summary['trace_error'] = str(error)
        finally:
            if self.process.is_alive():
                self.process.terminate()  # Logger only; never the SDK brake process.
                self.process.join(timeout=1)
            self.rows.cancel_join_thread()
            self.rows.close()
            self.errors.close()
        if summary.get('trace_error'):
            summary['passed'] = False
        with (self.directory / 'summary.json').open('x') as output:
            json.dump(jsonable(summary), output, indent=2, allow_nan=False)
