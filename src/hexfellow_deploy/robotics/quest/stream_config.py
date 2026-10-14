"""Source timing configuration; independent of motor/feedback safety limits."""
import hashlib
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path

from ..paths import CONFIG_DIR

DEFAULT_STREAM_CONFIG = CONFIG_DIR / 'quest/target_stream.json'


@dataclass(frozen=True)
class StreamConfig:
    playout_delay_ms: float
    age_warning_ms: float
    age_stale_ms: float
    age_timeout_ms: float
    gap_warning_ms: float
    gap_stale_ms: float
    gap_timeout_ms: float
    clock_uncertainty_ms: float
    rejoin_blend_ms: float
    recovery_stable_ms: float
    recovery_max_gap_ms: float
    sync_exchanges: int
    sync_stable_ms: float
    minimum_receive_hz: float
    history_capacity: int
    history_ms: float
    max_batch: int

    def __post_init__(self):
        if any(isinstance(v, bool) or not math.isfinite(v) or v <= 0 for v in asdict(self).values()):
            raise ValueError('stream parameters must be finite and positive')
        for name in ('sync_exchanges', 'history_capacity', 'max_batch'):
            if not isinstance(getattr(self, name), int):
                raise ValueError(f'{name} must be an integer')
        for prefix in ('age', 'gap'):
            if not (getattr(self, prefix + '_warning_ms') < getattr(self, prefix + '_stale_ms')
                    < getattr(self, prefix + '_timeout_ms')):
                raise ValueError(f'{prefix} thresholds must increase')
        if not (self.playout_delay_ms < self.age_stale_ms < self.history_ms):
            raise ValueError('playout < stale age < history required')
        if not 2 <= self.max_batch <= self.history_capacity <= 4096:
            raise ValueError('require 2 <= max_batch <= history_capacity <= 4096')

    def report(self):
        data = asdict(self)
        return dict(parameters=data, sha256=hashlib.sha256(
            json.dumps(data, sort_keys=True).encode()).hexdigest())


def load_stream_config(path=None):
    if isinstance(path, StreamConfig):
        return path  # A running session uses its frozen, reported configuration.
    return StreamConfig(**json.loads(Path(path or DEFAULT_STREAM_CONFIG).read_text()))
