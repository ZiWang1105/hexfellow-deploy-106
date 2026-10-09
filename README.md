# Hexfellow Deploy

One Deploy Anything (DA) runtime for the RGB-D agent, PinkIK controller and MIT
actuator stream. MuJoCo and the real robot select different device adapters;
they use the same controller, reference ownership, serialized commands, feedback
contract and command-health checks.

The native SDK adapter and physical RGB-D adapter are implemented. The physical
RGB-D/encoder matcher has passed a read-only connected-device test; the full
physical agent task remains unqualified. Simulator results establish software and
modeled-plant behavior, not firmware dynamics, network reliability or hardware
stopping. See [architecture](docs/ARCHITECTURE.md) and [validation](docs/VALIDATION.md).
The [DA alignment audit](docs/DEPLOY_ANYTHING_ALIGNMENT.md) lists implemented
parity, deliberate robot-specific differences and unresolved recovery behavior.
Both physical GP80 hands use this repository's [feedback gripper controller](docs/GRIPPER_CONTROL.md),
shared by the native policy adapter and the parent keyboard console.
Native policy observations include [both calibrated wrist RGB views](docs/WRIST_RGB.md)
alongside head RGB-D: four images per observation. Each wrist uses its own
October 8 candidate extrinsics, active-profile intrinsics and matched encoder FK.

Camera calibration code for the ZED 2i head and both RealSense wrists is now
[consolidated here](calibration/README.md), with a common
`./scripts/calibrate_camera.sh` launcher for collection, fitting, head landmark
PnP refinement, measured-arm overlays and simulation checks. `list` shows the
actions supported for each device. Both wrists have dedicated 30-pose routes,
with measured-joint capture, simulation validation and automatic fitting.

Saved head and both wrist extrinsics, SDK intrinsics, board scale and validation
evidence are versioned in the [camera parameter guide](calibration/docs/camera_parameters.md)
and [current parameter snapshot](calibration/config/camera/camera_parameters_20261008.json).
The [latest wrist results](calibration/docs/wrist_parameters_20261008.md) preserve
their candidate status: held-out RMS is 7.92 px for the left D405 and 13.62 px
for the right D435i. These exports do not change production sensor selection.

DA is pinned to `6c2bda40ebee56ab3a50b8dd62aaf4df5814e5a8`. Robot assets,
active calibration configuration, sensor contracts, policy client and PinkIK are imported from the
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
  --prompt 'Pick up the gray cuboid using a top-down grasp with the right hand and hold it above the table.' \
  --output "runs/real_$(date +%Y%m%d_%H%M%S)"
```

Camera use, point grounding, offset selection and reobservation are specified in
the shared system prompt. User task prompts need only the desired outcome and
any object, hand or grasp-style preferences.

The second command opens physical devices and can move the robot. Its availability
is not a claim of onsite validation of this revision. Both entry points route to
this same DA implementation; historical runners require `--legacy-runtime`.
The default preparation bends both elbows before the first policy observation.
A `--hold-only` diagnostic keeps the measured initial posture instead.

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

Open a native MuJoCo window with **draggable red TCP targets**, using the shared
controller and simulator actuator process (run from a desktop terminal):

```bash
cd ~/projects/agentic-mobile-manipulator
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 .venv/bin/python scripts/run_agent.py \
  --config configs/local/desktop.yaml --interactive-ik \
  --output "runs/interactive_ik_$(date +%Y%m%d_%H%M%S)"
```

Double-click a red sphere to select the left or right TCP. Hold Ctrl and drag
with the right mouse button to translate, or the left button to rotate. Space
or R requests constrained HOLD and resets both handles to measured TCP poses;
drag again to resume. Q or closing the window shuts down the simulation. The
overlay shows the selected arm, measured position/orientation tracking error,
and controller rejections. The web dashboard remains available on port 8765
(`--viewer-port` changes the dashboard port).

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
