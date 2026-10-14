"""Single-owner feedback construction, independent of command dispatch.

The builder owns coordinate-conversion history. Readers borrow a completed
snapshot; they must not mutate it. Publication replaces one object reference
under CPython's GIL and never holds a reader lock across builder work. Sensor
receipt timestamps are preserved; a new publication does not renew feedback.
"""
from contextlib import nullcontext
from dataclasses import dataclass, replace
import math
import threading
import time

from .x7 import ControlDelay, HardwareSafetyError


@dataclass(frozen=True)
class Publication:
    state: object
    sequence: int
    published_s: float
    build_ms: float = 0.
    build_cpu_ms: float = 0.
    max_build_ms: float = 0.
    error: BaseException | None = None


class FeedbackSnapshotWorker:
    def __init__(self, build, seed, *, period_s=.004, context=nullcontext,
                 notify=None, clock=time.monotonic):
        if not math.isfinite(period_s) or period_s <= 0:
            raise ValueError("feedback period must be positive and finite")
        self.build, self.period_s, self.context = build, period_s, context
        self.notify, self.clock = notify, clock
        self._publication = Publication(seed, 0, clock())
        self._stop = threading.Event()
        self.thread = threading.Thread(target=self._run, name="umi-feedback-snapshot", daemon=True)

    def start(self):
        self.thread.start()

    def _run(self):
        try:
            with self.context():
                while not self._stop.is_set():
                    started, cpu = self.clock(), time.thread_time()
                    try:
                        state = self.build()
                    except ControlDelay as error:
                        # Preserve upstream recoverable-delay semantics; keep
                        # collecting until a valid snapshot replaces this error.
                        self._publication = replace(self._publication, error=error)
                        if self.notify is not None:
                            self.notify()
                        self._stop.wait(self.period_s)
                        continue
                    ended = self.clock()
                    # Arrays belong to the builder, not mutable SDK buffers.
                    state.q.flags.writeable = False
                    state.v.flags.writeable = False
                    prior = self._publication
                    self._publication = Publication(state, prior.sequence + 1, ended,
                        1000*(ended-started), 1000*(time.thread_time()-cpu),
                        max(prior.max_build_ms, 1000*(ended-started)))
                    if self.notify is not None:
                        self.notify()
                    self._stop.wait(max(0., self.period_s-(self.clock()-started)))
        except BaseException as error:
            # Never hide a failed builder behind a previously healthy snapshot.
            self._publication = replace(self._publication, error=error)
            if self.notify is not None:
                self.notify()

    def read(self):
        publication = self._publication
        if publication.error is not None:
            if isinstance(publication.error, ControlDelay):
                raise publication.error
            raise HardwareSafetyError(
                f"feedback snapshot worker failed: {type(publication.error).__name__}: {publication.error}"
            ) from publication.error
        state = publication.state
        return replace(state, timing={**(state.timing or {}),
            "snapshot_mode": "background", "snapshot_sequence": publication.sequence,
            "snapshot_published_s": publication.published_s,
            "snapshot_build_ms": publication.build_ms,
            "snapshot_build_cpu_ms": publication.build_cpu_ms})

    def report(self):
        p = self._publication
        return dict(mode="background", sequence=p.sequence, published_s=p.published_s,
            build_ms=p.build_ms, build_cpu_ms=p.build_cpu_ms, max_build_ms=p.max_build_ms,
            error=None if p.error is None else f"{type(p.error).__name__}: {p.error}",
            thread_alive=self.thread.is_alive(), native_id=self.thread.native_id)

    def close(self, timeout_s=.5):
        self._stop.set()
        if self.thread.ident is not None:
            self.thread.join(timeout_s)
        if self.thread.is_alive():
            raise HardwareSafetyError("feedback snapshot worker did not stop")
