"""Bounded latest-value IPC without multiprocessing.Queue feeder races.

Only trusted parent/child processes exchange pickled snapshots. Serialization
is outside the lock. State publication overwrites and is nonblocking; command
publication never overwrites and bounds lock contention to half a millisecond.
"""

import pickle
import os
import queue
import time


class MailboxBusy(queue.Empty):
    """The mailbox could not be inspected; different from an empty mailbox."""


class LatestStateMailbox:
    def __init__(self, context, capacity=65536):
        self.buffer = context.RawArray("B", capacity)
        self.length = context.RawValue("I", 0)
        self.lock = context.Lock()
        self.overwrites = context.RawValue("Q", 0)
        self.publication_sequence = context.RawValue("Q", 0)

    def put_nowait(self, value):
        payload = pickle.dumps(value, protocol=pickle.HIGHEST_PROTOCOL)
        if len(payload) > len(self.buffer):
            raise ValueError("hardware state exceeds bounded mailbox capacity")
        if not self.lock.acquire(False):
            raise queue.Full
        try:
            if self.length.value:
                self.overwrites.value += 1
            memoryview(self.buffer).cast("B")[:len(payload)] = payload
            self.length.value = len(payload)
            self.publication_sequence.value += 1
        finally:
            self.lock.release()

    def get_nowait(self):
        if not self.lock.acquire(False):
            raise queue.Empty
        try:
            if not self.length.value:
                raise queue.Empty
            payload = bytes(memoryview(self.buffer).cast("B")[:self.length.value])
            self.length.value = 0
        finally:
            self.lock.release()
        return pickle.loads(payload)

    def get_latest(self, timeout=.0005):
        """One atomic read of the latest publication, with bounded lock wait."""
        started = time.monotonic()
        if not self.lock.acquire(timeout=max(0., min(timeout, .0005))):
            raise MailboxBusy
        try:
            if not self.length.value:
                raise queue.Empty
            payload = bytes(memoryview(self.buffer).cast("B")[:self.length.value])
            sequence = self.publication_sequence.value
            self.length.value = 0
        finally:
            self.lock.release()
        return pickle.loads(payload), {"publication_sequence": sequence,
            "mailbox_read_ms": (time.monotonic() - started) * 1000}

    def cancel_join_thread(self):
        # No feeder thread exists. Keep gateway cleanup protocol compatible.
        pass


class CommandMailbox(LatestStateMailbox):
    """One reference, no overwrite and no asynchronous Queue feeder."""

    def __init__(self, context, wake=None, capacity=65536):
        super().__init__(context, capacity)
        self.wake = wake or PipeWake(context)
        self.hold_requested = context.Event()
        self.estop_armed_at = context.Value('d', 0.)

    def put_nowait(self, value):
        payload = pickle.dumps(value, protocol=pickle.HIGHEST_PROTOCOL)
        if len(payload) > len(self.buffer):
            raise ValueError("command exceeds bounded mailbox capacity")
        if not self.lock.acquire(timeout=.0005):
            raise queue.Full
        try:
            if self.length.value:
                raise queue.Full
            memoryview(self.buffer).cast("B")[:len(payload)] = payload
            self.length.value = len(payload)
        finally:
            self.lock.release()
        self.wake.set()


class PipeWake:
    """Coalescing nonblocking wake pipe, without a shared condition lock.

    Data stays in the guarded mailbox. A full pipe already means the consumer
    is wakeable, so extra notifications may coalesce without losing references.
    Both Connection handles are transferred by multiprocessing spawn itself.
    """
    def __init__(self, context):
        self.reader, self.writer = context.Pipe(duplex=False)
        os.set_blocking(self.reader.fileno(), False)
        os.set_blocking(self.writer.fileno(), False)

    def set(self):
        try:
            os.write(self.writer.fileno(), b'x')
        except BlockingIOError:
            pass

    def clear(self):
        for _ in range(4):
            try:
                if not os.read(self.reader.fileno(), 65536):
                    break
            except BlockingIOError:
                break

    def wait(self, timeout=None):
        from multiprocessing.connection import wait
        return bool(wait([self.reader], timeout))
