# Hexfellow Deploy

One Deploy Anything (DA) runtime for the RGB-D agent, PinkIK controller and MIT
actuator stream. MuJoCo and the real robot select different device adapters;
they use the same controller, reference ownership, serialized commands, feedback
contract and command-health checks.

The native SDK adapter and physical RGB-D adapter are implemented. **This change
has not been tested on the physical robot.** Simulator results establish software
and modeled-plant behavior, not firmware dynamics, network reliability or hardware
stopping. See [architecture](docs/ARCHITECTURE.md) and [validation](docs/VALIDATION.md).

DA is pinned to `6c2bda40ebee56ab3a50b8dd62aaf4df5814e5a8`. Robot assets,
calibration, sensor contracts, policy client and PinkIK are imported from the
parent `agentic-mobile-manipulator` workspace. This repository is not yet a
standalone robot distribution. The sibling source checkouts are not modified.

## Install in the existing workspace

Use the parent Python 3.10 environment with MuJoCo, Pinocchio and the UMI native
extension already installed. From this repository:

```bash
git submodule update --init --recursive
uv pip install --python ../../.venv/bin/python \
  'hydra-core==1.3.7' 'posix_ipc==1.3.2' 'rich==15.0.0' log-throttling
uv pip install --python ../../.venv/bin/python --no-deps -e third_party/deploy-anything -e .
uv pip install --python ../../.venv/bin/python 'matplotlib>=3.8,<3.10' 'numpy<2'
```

The tested environment uses NumPy 1.26.4 because the parent Pinocchio environment
requires NumPy below 2. DA's package metadata requests NumPy 2; `--no-deps` is an
intentional compatibility exception for its tested manager/shared-memory subset.
Do not upgrade the robot environment to NumPy 2 to satisfy that metadata. ROS,
YAM, CAN and DA's optional learning backends are not used here.

## Run

From `agentic-mobile-manipulator`, the simulator entry point is:

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 .venv/bin/python scripts/run_agent.py \
  --config configs/local/desktop.yaml \
  --prompt 'Pick up the apple and hold it above the table.' \
  --output "runs/sim_$(date +%Y%m%d_%H%M%S)"
```

The corresponding native adapter is selected by the real-robot entry point:

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 .venv/bin/python scripts/run_real_agent.py \
  --config configs/local/desktop.yaml \
  --execute --research --confirm-onsite-ready --live-viewer \
  --prompt 'Pick up the orange using a top-side grasp and hold it above the table.' \
  --output "runs/real_$(date +%Y%m%d_%H%M%S)"
```

The second command opens physical devices and can move the robot. Its availability
is not a claim of onsite validation of this revision. Both entry points route to
this same DA implementation; historical runners require `--legacy-runtime`.
The default preparation bends both elbows before the first policy observation.
A `--hold-only` diagnostic keeps the measured initial posture instead.

The terminal prints the clickable dashboard URL, normally
<http://127.0.0.1:8765/>. It displays RGB, control state, actuator acceptance,
latest agent action, model-call state and errors. The last action remains visible
during inference. API requests, sensor images, responses and reported costs are
archived under `policy/`. Only the API client reads the external key file.

Astra with medium effort is the default. There is no default execution duration,
call count or API spending cap. Ctrl-C closes the run. `--duration-s` optionally
limits the **whole diagnostic run after control startup**, not one model call.
`--settle-s` optionally adds a final constrained HOLD interval after that budget.

## Controller diagnostics and offline analysis

A paired-pose test with deliberate scheduling and actuator delays, without API calls:

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 .venv/bin/python -m hexfellow_deploy.cli \
  --workspace . --duration-s 28 --no-camera \
  --action third_party/hexfellow-deploy/tests/fixtures/paired_pose.json \
  --set interface.control.startup_carry=false \
  --set interface.control.inject_every=10 \
  --set interface.control.inject_delay_s=0.03 \
  --set interface.backend.submit_delay_s=0.03 \
  --output "runs/delay_$(date +%Y%m%d_%H%M%S)"
```

Use `--set interface.dashboard.port=8875` to avoid another viewer's port.
Overrides use OmegaConf YAML syntax; topics resolve through the actual DA registry.

From this repository:

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 ../../.venv/bin/python -m pytest tests -q
../../.venv/bin/python scripts/analyze_mit_run.py ../../runs/COMPLETED_RUN
MUJOCO_GL=egl ../../.venv/bin/python scripts/analyze_mit_run.py ../../runs/COMPLETED_RUN --video
```

The analyzer requires both owner final records before reading evaluation data.
It writes `analysis/analysis.json`, `tcp_samples.npz` and `tcp_tracking.png`,
separating actual-to-target, reference-to-target and actual-to-reference errors.
`--video` replays the entire recorded trajectory into
`analysis/replay/pickup.mp4`; it does not rerun physics or affect the policy.
Every exported movie labels the overall run outcome and termination separately
from per-frame grasp retention. A retained grasp in a failed episode stays labeled
as a failed run.
The older `qualify_ik.py` remains explicitly labeled as position-plant evidence.

Outputs also include source/dependency provenance, command/control records,
backend metadata, termination reason and separate shutdown errors. Simulator
base/object state is recorded for rendering and offline scoring only; it never
enters policy/control observations.
