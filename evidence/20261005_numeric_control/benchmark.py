"""Uncontended same-process DA put/get cost, not a hardware latency benchmark."""
import argparse
import json
from pathlib import Path
import time
import uuid
import numpy as np
from hexfellow_deploy.control_topics import FeedbackTopic, ReferenceTopic, ReceiptTopic
from hexfellow_deploy.topics import JsonTopic

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--run', type=Path, required=True)
parser.add_argument('--output', type=Path, required=True)
args = parser.parse_args()
with (args.run/'control.jsonl').open() as stream:
    row = json.loads(next(stream))
samples = [
    (FeedbackTopic, dict(measured=row['measured'], received=row['source_received'],
                        publication_sequence=1, published_s=row['submitted_s'])),
    (ReferenceTopic, dict(sequence=row['sequence'], created_s=row['mit']['timestamp_s'],
                         command=row['reference'], mit=row['mit'], measured=row['measured'],
                         received=row['source_received'], reference_velocity=row['reference_velocity'])),
    (ReceiptTopic, dict(receipt=dict(sequence=1, accepted_at=row['submitted_s'], scope='simulator_applied'),
                       pending=None, latest_sequence=1, fault=False))]
results = []
for cls, value in samples:
    for implementation in (JsonTopic, cls):
        topic = implementation('bench_'+uuid.uuid4().hex)
        topic.initialize()
        try:
            for _ in range(100):
                assert topic.put(value) and topic.get() == value
            times = []
            for _ in range(3000):
                start = time.perf_counter_ns()
                assert topic.put(value)
                topic.get()
                times.append((time.perf_counter_ns()-start)/1000.)
            results.append(dict(message=cls.__name__, implementation=implementation.__name__,
                payload_bytes=int(np.prod(topic.shape))*np.dtype(topic.dtype).itemsize,
                samples=len(times), median_us=float(np.median(times)), p95_us=float(np.percentile(times,95)),
                p99_us=float(np.percentile(times,99)), max_us=max(times)))
        finally:
            topic.clean_up()
args.output.write_text(json.dumps(dict(scope=__doc__, results=results), indent=2)+'\n')
print(json.dumps(results))
