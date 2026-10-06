# Fixed numeric control IPC validation

Scope: DA transport alignment and contention recovery in the shared MuJoCo MIT
runtime. No physical device access, policy API calls or pickup qualification.
See the [alignment audit](../../docs/DEPLOY_ANYTHING_ALIGNMENT.md) for remaining
differences and corrections to earlier claims.

Tests: `../../.venv/bin/python -m pytest tests -q`, 140 passed in 7.71 seconds.
Run this test command from the deployment repository; commands below run from
the parent `agentic-mobile-manipulator` workspace.

The public entry point passed 10 seconds of HOLD, 502/502 accepted frames:

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MUJOCO_GL=egl \
  .venv/bin/python scripts/run_agent.py --hold-only --duration-s 10 \
  --output runs/numeric_control_smoke --viewer-port 18769
```

The reproducible probe starts the actual shared simulator runtime, camera and
dashboard. An external process attaches through DA to a numeric topic's actual
named semaphore, holds it repeatedly, and checks final reports. It does not
replace the transport with a mock or run privileged object control. Use new
output directories when repeating:

```bash
.venv/bin/python third_party/hexfellow-deploy/scripts/probe_control_ipc.py \
  --workspace . --output runs/numeric_feedback_contention_02 \
  --channel feedback --hold-ms 45
.venv/bin/python third_party/hexfellow-deploy/scripts/probe_control_ipc.py \
  --workspace . --output runs/numeric_reference_contention \
  --channel frames --hold-ms 20
.venv/bin/python third_party/hexfellow-deploy/scripts/probe_control_ipc.py \
  --workspace . --output runs/numeric_receipt_contention \
  --channel receipts --hold-ms 20
```

All three 15-second fixtures passed, with 54 acquired holds per run. Feedback
contention recovered through 500 refresh retries; reference contention through
283 exact-frame publication retries. Receipt contention paused generation until
acceptance. No control/backend/shutdown faults. The reference fixture generated
728 and accepted 727; its last pending sequence 728 was canceled on scheduled
shutdown, explicitly recorded in `control_final.json`.

An earlier probe launch (`runs/numeric_feedback_contention`) failed before
starting simulation because the harness pre-created the runner's output folder.
The corrected probe writes its launch log alongside the output until the runner
has created the directory. This harness failure is not counted as a passing run.

`benchmark.json` compares the previous 64 KiB JSON topic and the new numeric
layouts on the same recorded payloads, 3,000 uncontended same-process put/get
samples each. It measures local serialization/copy overhead, not cross-process
latency, SDK delivery or hardware control reliability. Receipt timing is almost
unchanged; feedback/reference improve by roughly 6/15 microseconds in the median.
This does not establish JSON as the cause of the physical stale-feedback event.

```bash
.venv/bin/python third_party/hexfellow-deploy/evidence/20261005_numeric_control/benchmark.py \
  --run runs/numeric_control_smoke --output /tmp/control_ipc_benchmark.json
```

Compact final reports are stored here; `artifacts.json` records SHA-256 and size
of the complete parent run artifacts including per-cycle timing and original
feedback clocks. Tests additionally cover held-lock expiry, unchanged reference
history, unknown schema, corrupt fields, and refusal to leak extra sensor fields.
The native snapshot, SDK callback, UDP/KCP and firmware paths require separate
physical evidence. Lift stop confirmation remains unresolved.
