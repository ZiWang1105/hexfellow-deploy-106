"""Summarize measured scheduling and acceptance data, including delayed intervals."""
import argparse
import json
from pathlib import Path
import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run')
    args = parser.parse_args()
    directory = Path(args.run)
    rows = [json.loads(line) for line in (directory/'control.jsonl').read_text().splitlines()]
    final = json.loads((directory/'control_final.json').read_text())
    result = {k:final[k] for k in ('phase','fault','ticks','generated','accepted','evidence_kind')}
    result['overruns'] = sum(row['timing']['budget_overrun'] for row in rows)
    result['history_advanced_without_acceptance'] = sum(row['accepted_before_solve'] != row['sequence']-1 for row in rows)
    for key, values in [('compute_and_submit_ms', [r['timing']['elapsed_ms'] for r in rows]),
                        ('interval_ms', [r['interval_ms'] for r in rows if r['interval_ms'] is not None]),
                        ('acceptance_ms', [r['previous_acceptance_ms'] for r in rows if r['previous_acceptance_ms'] is not None])]:
        result[key] = dict(zip(('p50', 'p95', 'p99', 'max'), np.percentile(values, [50,95,99,100]).tolist())) if values else None
    (directory/'summary.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
