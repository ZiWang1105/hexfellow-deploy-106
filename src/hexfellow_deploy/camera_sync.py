"""Bounded encoder history for host receipt-time RGB-D matching."""
from collections import deque
from copy import deepcopy
import math
import threading
import time


class EncoderHistory:
    """Retain physical samples; never interpolate or refresh their timestamps."""
    def __init__(self, horizon_s, capacity=512):
        if not math.isfinite(horizon_s) or horizon_s <= 0 or capacity < 1:
            raise ValueError('Encoder history requires positive bounds')
        self.horizon = horizon_s
        self.rows = deque(maxlen=capacity)
        self.lock = threading.Lock()
        self.rejected = 0

    def add(self, row, now):
        stamp = row['measured']['timestamp_s']
        if not math.isfinite(stamp) or not 0 <= now-stamp <= self.horizon:
            with self.lock:
                self.rejected += 1
            return False
        owned = deepcopy(row)
        with self.lock:
            while self.rows and now-self.rows[0]['measured']['timestamp_s'] > self.horizon:
                self.rows.popleft()
            if self.rows:
                previous = self.rows[-1]['measured']['timestamp_s']
                if stamp < previous:
                    self.rejected += 1
                    return False
                if stamp == previous:
                    self.rows[-1] = owned
                    return True
            self.rows.append(owned)
        return True

    def closest(self, timestamp_s, now, max_skew_s):
        if (not math.isfinite(timestamp_s) or not 0 <= now-timestamp_s <= self.horizon
                or not math.isfinite(max_skew_s) or max_skew_s < 0):
            return None
        with self.lock:
            rows = tuple(self.rows)
        candidates = [row for row in rows
                      if 0 <= now-row['measured']['timestamp_s'] <= self.horizon
                      and abs(row['measured']['timestamp_s']-timestamp_s) <= max_skew_s]
        if not candidates:
            return None
        return deepcopy(min(candidates, key=lambda row: abs(row['measured']['timestamp_s']-timestamp_s)))

    def stats(self):
        with self.lock:
            return dict(samples=len(self.rows), rejected=self.rejected,
                        oldest_s=self.rows[0]['measured']['timestamp_s'] if self.rows else None,
                        newest_s=self.rows[-1]['measured']['timestamp_s'] if self.rows else None)


class FeedbackSampler:
    """Sample the existing feedback SHM without coupling to image processing."""
    def __init__(self, topic, history, period_s=.004):
        if not math.isfinite(period_s) or period_s <= 0:
            raise ValueError('Feedback sampling period must be finite and positive')
        self.topic, self.history, self.period = topic, history, period_s
        self.stop = threading.Event()
        self.thread = None
        self.error = None
        self.reads = self.misses = 0

    def start(self):
        if self.thread is not None:
            raise RuntimeError('Feedback sampler already started')
        self.thread = threading.Thread(target=self._run, name='camera-encoder-history', daemon=True)
        self.thread.start()
        return self

    def _run(self):
        try:
            while not self.stop.is_set():
                started = time.monotonic()
                row = self.topic.get()
                if row is None:
                    self.misses += 1
                else:
                    self.reads += 1
                    self.history.add(row, time.monotonic())
                self.stop.wait(max(0., started+self.period-time.monotonic()))
        except Exception as error:
            self.error = f'{type(error).__name__}: {error}'

    def close(self):
        self.stop.set()
        if self.thread is not None:
            self.thread.join(timeout=1.)
            if self.thread.is_alive():
                raise RuntimeError('Camera encoder sampler did not stop')
