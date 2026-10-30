# Hexfellow Deploy

One owned deployment runtime for the RGB-D agent, PinkIK controller and MIT
actuator stream. MuJoCo and the real robot select different device adapters;
they use the same controller, reference ownership, serialized commands, feedback
contract and command-health checks.

The native SDK adapter and physical RGB-D adapter are implemented. The physical
RGB-D/encoder matcher has passed a read-only connected-device test; the full
physical agent task remains unqualified. Simulator results establish software and
modeled-plant behavior, not firmware dynamics, network reliability or hardware
stopping. See [architecture](ARCHITECTURE.md) and [validation](VALIDATION.md).
The [historical alignment audit](DEPLOY_ANYTHING_ALIGNMENT.md) records the
original design comparison. The current implementation owns its process/topic
runtime and PinkIK/native code; no external deployment or solver repository is required.
Both physical GP80 hands use this repository's [feedback gripper controller](GRIPPER_CONTROL.md),
shared by the native policy adapter and the parent keyboard console.
Native policy observations include [both calibrated wrist RGB views](WRIST_RGB.md)
alongside head RGB-D: four images per observation. Each wrist uses its own
October 8 candidate extrinsics, active-profile intrinsics and matched encoder FK.

Camera calibration code for the ZED 2i head and both RealSense wrists is now
[consolidated here](../calibration/README.md), with a common
`./scripts/calibrate_camera.sh` launcher for collection, fitting, head landmark
PnP refinement, measured-arm overlays and simulation checks. `list` shows the
actions supported for each device. Both wrists have dedicated 30-pose routes,
with measured-joint capture, simulation validation and automatic fitting.

Saved head and both wrist extrinsics, SDK intrinsics, board scale and validation
evidence are versioned in the [camera parameter guide](../calibration/docs/camera_parameters.md)
and [current parameter snapshot](../calibration/config/camera/camera_parameters_20261008.json).
The [latest wrist results](../calibration/docs/wrist_parameters_20261008.md) preserve
their candidate status: held-out RMS is 7.92 px for the left D405 and 13.62 px
for the right D435i. These exports do not change production sensor selection.

## Default IK

The default runtime selects `x7-gp80-general-trajectory-sim-v2` at **100 Hz**,
with target velocity limits of **2.0 m/s** and **4.0 rad/s**. The profile's
acceleration, jerk, joint and effort constraints remain active. The name retains
`sim` for benchmark provenance; selecting it does not qualify hardware safety.

The original profile remains available by explicitly selecting
`deployment_config=configs/hardware/desktop.yaml` and
`timing.control_period_s=0.02`. Physical execution requires a machine-local
configuration and matching research binding; repository templates remain
uncommissioned. See [interactive hardware IK](HARDWARE_INTERACTIVE_IK.md).

## Standalone installation

Clone this repository normally; it has no Git submodules. The controller, robot contracts,
hardware adapters, MuJoCo scene builder/assets, skill executor and draggable TCP
viewer are owned here. No agent application checkout or API key is needed for IK.

On Ubuntu, install `build-essential` and `libeigen3-dev`, plus `uv`, then:

```bash
git clone git@github.com:ZiWang1105/hexfellow-deploy.git
cd hexfellow-deploy
./scripts/setup_sim.sh
```

The script creates a Python 3.10 `.venv`, installs the simulation dependencies,
builds the owned native controller extension, and installs the owned process/topic
runtime. No additional repository access is required. NumPy stays below 2 for Pinocchio.
Source lineage and preserved license terms are recorded in [NOTICE.md](../NOTICE.md).
For native wire-format tests, also install the optional SDK (this opens no devices):

```bash
uv pip install --python .venv/bin/python -e '.[hardware]'
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 .venv/bin/python -m pytest tests -q
```

See [standalone ownership and migration](STANDALONE.md). Calibration candidates
retain their recorded qualification status; this extraction changes no IK gains,
actuator limits, feedback checks, ownership rules, or hardware authorization.

## Run

From this repository in a desktop terminal, open the interactive simulator:

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 .venv/bin/hexfellow-deploy \
  --interactive-ik --no-camera \
  --output "runs/interactive_ik_$(date +%Y%m%d_%H%M%S)"
```

No `--workspace` is required. Defaults resolve this checkout's
`configs/hardware/desktop.yaml`, controller profile, and example scene.
That uncommissioned template is for simulation, not physical execution.

For a headless MIT control diagnostic with a paired TCP target:

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 .venv/bin/hexfellow-deploy \
  --no-camera --duration-s 8 \
  --action tests/fixtures/paired_pose.json \
  --set interface.control.startup_carry=false \
  --output "runs/ik_$(date +%Y%m%d_%H%M%S)"
```

An agent application can supply its installed component via `policy.component`,
or an explicit `--agent-plugin` module with `AGENT_COMPONENT` and optional task
catalog/evaluation hooks. Agent requests and task scoring are not runtime dependencies.
Existing application launchers pass their configuration workspace and plugin explicitly.
For real-robot mouse dragging, see [interactive hardware IK](HARDWARE_INTERACTIVE_IK.md).
Physical operation still requires a locally configured deployment file and both
`--execute --confirm-onsite-ready`; the default command above never opens hardware.

The terminal prints the clickable dashboard URL, normally
<http://127.0.0.1:8765/>. It displays RGB, control state, actuator acceptance,
startup stage, model-call state and errors. The Agent activity panel appears above
the camera images and shows the latest left/right TCP, binary grippers and base
XY/yaw decision. The last action remains visible during inference; a failure
before the first model call is stated explicitly. Raw telemetry is collapsible. API requests, sensor images, responses and reported costs are
archived under `policy/`. Physical camera matching diagnostics are saved to
`camera_health.json`, including frame ages, encoder skew and publication failures.
Only the API client reads the external key file.

Astra with medium effort is the default. There is no default execution duration,
call count or API spending cap. Ctrl-C closes the run. `--duration-s` optionally
limits the **whole diagnostic run after control startup**, not one model call.
`--settle-s` optionally adds a final constrained HOLD interval after that budget.

## Controller diagnostics and offline analysis

The [fast X7 IK benchmark](IK_BENCHMARK.md) compares the current profiles,
an AgileX-style nonlinear endpoint solver, and a 100 Hz differential IK candidate
in the shared MIT plant. It includes analytic trajectories, per-joint constraint
checks, a speed sweep, retained failures, plots, and a video replay command.
The candidate exceeds 1.2 m/s on tested feasible paths; it remains a simulation
experiment and does not change the default viewer or hardware profile.

Open a native MuJoCo window with **draggable red TCP targets**, using the shared
controller and simulator actuator process (run from a desktop terminal):

```bash
cd ~/projects/hexfellow-deploy
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 .venv/bin/hexfellow-deploy \
  --interactive-ik --no-camera \
  --output "runs/interactive_ik_$(date +%Y%m%d_%H%M%S)"
```

Double-click a red sphere to select the left or right TCP. Hold Ctrl and drag
with the right mouse button to translate, or the left button to rotate. Space
or R requests constrained HOLD and resets both handles to measured TCP poses;
drag again to resume. Q or closing the window shuts down the simulation. The
overlay shows the selected arm, measured position/orientation tracking error,
and controller rejections. The web dashboard remains available on port 8765
(`--set interface.dashboard.port=8875` changes the dashboard port).

This mode starts at the scene's initial pose without the carrying routine and
makes no policy/API calls. It holds the lift reference fixed and preserves the
initial binary gripper commands. It cannot be combined with an agent prompt,
action fixture or hardware execution. No camera capture is needed. The display
mirrors plant snapshots; its mocap handles never apply forces or alter plant
state. Mouse input becomes an ordinary paired base-frame TCP request through
the same PinkIK, reference ownership, command validation and MIT simulation
path. Only the operator renderer sees world coordinates. `interactive.jsonl`
records submitted input; `control.jsonl` records measured/reference behavior.
No successful grasp or hardware qualification is implied by this diagnostic.

A paired-pose test with deliberate scheduling and actuator delays, without API calls:

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 .venv/bin/python -m hexfellow_deploy.cli \
  --workspace . --duration-s 28 --no-camera \
  --action tests/fixtures/paired_pose.json \
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
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 .venv/bin/python -m pytest tests -q
.venv/bin/python scripts/analyze_mit_run.py runs/COMPLETED_RUN
MUJOCO_GL=egl .venv/bin/python scripts/analyze_mit_run.py runs/COMPLETED_RUN --video
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

Solver/runtime source attribution and dependency-removal details are in [NOTICE.md](../NOTICE.md) and [standalone setup](STANDALONE.md).
