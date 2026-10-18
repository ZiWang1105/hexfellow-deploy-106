# Interactive hardware IK

The standalone launcher supports the mouse-drag viewer with the native hardware
backend. The display is a robot-only MuJoCo rendering of physical joint feedback;
it never steps physics, synthesizes feedback, or sends rendered joint positions.
It uses the deployment's URDF, assembly, and pad-center TCP convention. The base
is drawn at the origin. No table, object, obstacle, or world localization is sensed
by this display. Jaw geometry is illustrative, not an aperture measurement.

## Launch

Select an existing locally configured hardware deployment and its matching
operator-research source binding. The repository example remains unconfigured.
The selected deployment controls the profile. The original local deployment
retains its historical profile. The latest benchmark candidate can be selected
explicitly for unqualified hardware research using the preparation below.

From this checkout in a desktop terminal:

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 .venv/bin/python \
  -m hexfellow_deploy.cli \
  --interactive-ik --no-camera --preflight \
  --set deployment_config=/absolute/path/to/local/desktop.yaml \
  --output "runs/interactive_preflight_$(date +%Y%m%d_%H%M%S)"
```

Preflight checks configuration, SDK availability, and source binding, writes
`preflight.json`, and exits without opening devices. A failed source binding is
not a network failure. Select a research binding for the exact code being tested;
do not set qualification flags or disable the validator to make it pass.

With the robot's motion area clear and the physical E-stop accessible:

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 .venv/bin/python \
  -m hexfellow_deploy.cli \
  --interactive-ik --no-camera --execute --confirm-onsite-ready \
  --set deployment_config=/absolute/path/to/local/desktop.yaml \
  --output "runs/interactive_real_$(date +%Y%m%d_%H%M%S)"
```

The launcher acquires the existing hardware control sessions and starts HOLD.
There is no automatic carrying pose or lift positioning. The lift reference
stays at startup height, base velocity stays zero, and both jaws receive BRAKE.
Do not launch while using the grippers to support an object. No policy API runs.

## Controls

1. Check that the displayed pose matches the robot and both TCP handles.
2. **Double-click a red TCP sphere** to select it and enable one drag.
3. Hold **Ctrl + right mouse button** and move the mouse to translate it;
   **Ctrl + left drag** rotates it. Alternatively, press **1 / 2** to select
   left / right and **E** to enable. Selection alone sends no movement target.
4. Release the drag to request constrained HOLD. Double-click the TCP again
   to enable another drag (or press E). Selecting robot geometry does not enable.
5. **Space / R** requests HOLD and resets handles. **Q / window close / Ctrl-C**
   shuts down the session. These software controls are not a physical E-stop.

The control owner uses the selected IK profile target velocity limits.
The latest general profile allows **2.0 m/s** and **4.0 rad/s**; no extra
low-speed mouse limit is applied.
These are target-rate limits, not guarantees on instantaneous measured TCP speed.
The shared controller's trajectory shaping, joint/effort constraints, command
validation, native watchdogs and stop/rearm restrictions remain active.
The interactive layer adds no goal-distance rejection or Cartesian tracking-lag
HOLD threshold. Reachability and actuator constraints remain the shared
controller’s responsibility; a draggable goal is not a guarantee of reachability.

The GUI publishes fresh target leases only during an active drag. Leases are
at most **250 ms** and are checked independently in the control process. Window
stall/crash, expired input, invalid targets, or publication failure cannot renew
an old target indefinitely. After lease expiry, another E press is required.
HOLD uses the controller's constrained braking; it is not an instantaneous stop.
The existing hardware stream watchdog remains separate from the GUI lease.

## Evidence

`interactive.jsonl` records input timestamps, deadlines, goals and publication
results. Its binary gripper fields only satisfy the public Action schema; this
operator mode ignores them and sends BRAKE to both jaws. `control.jsonl` records
measured joints, MIT references, actual control timing and the manual-controller
state, including HOLD reasons. `control_final.json` records shutdown/faults.
The dashboard remains available on the usual port 8765.

Offline tests cover encoder-display FK consistency, target translation/rotation
rates, source/lease expiry, replay/session rejection, lag stopping, fixed lift,
zero base velocity, jaw BRAKE and preflight without runtime construction.
A real desktop mouse-drag exercise uses this same hardware-viewer component with
the shared simulated MIT backend. This checks input and display behavior, not
hardware motion, contact safety or high-speed qualification.

## Latest benchmark tuning on hardware

The exact frozen `x7-gp80-general-trajectory-sim-v2` profile is registered for
explicit hardware research. Its filename retains `sim` for identity/provenance;
registration does not claim physical qualification. The benchmark file and its
parameters are unchanged. The controller runs at **100 Hz**, not the historical
50 Hz. Startup prints the selected profile, canonical profile SHA-256 and cadence.

Prepare a separate local configuration from an existing fixed-base-v1 research
deployment. This copies hardware mapping/review evidence, changes only the
selected profile identity and controller configuration, and binds current source
hashes. It does not change device calibration, MIT gains, physical joint/effort
limits, review flags or hardware qualification. Existing files are never overwritten.

```bash
.venv/bin/python -m hexfellow_deploy.prepare_general_research \
  --source-deployment /absolute/path/to/existing/research/desktop.yaml \
  --output configs/local/general_ik_research_20261017
```

For this worktree, `configs/local/general_ik_research_20261017/desktop.yaml` has already
been prepared from the existing local research configuration. Run file-only
validation before opening devices:

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 .venv/bin/python \
  -m hexfellow_deploy.cli --interactive-ik --no-camera --preflight \
  --set deployment_config=configs/local/general_ik_research_20261017/desktop.yaml \
  --set timing.control_period_s=0.01 \
  --output "runs/general_preflight_$(date +%Y%m%d_%H%M%S)"
```

Then launch from the desktop terminal:

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 .venv/bin/python \
  -m hexfellow_deploy.cli --interactive-ik --no-camera \
  --execute --confirm-onsite-ready \
  --set deployment_config=configs/local/general_ik_research_20261017/desktop.yaml \
  --set timing.control_period_s=0.01 \
  --output "runs/general_ik_real_$(date +%Y%m%d_%H%M%S)"
```

Expected canonical profile SHA-256:
`ac1785da8217414b1a96e7ffad2d7a8045374fbe10aa46e8f522645ea4a98d4b`.
The raw YAML file SHA-256 differs from the parsed canonical profile hash used
by hardware admission. Preflight also rejects a mismatched control period before
any device connection. Source edits invalidate the research binding as before.

The mouse input uses the selected profile limits, including the latest profile's
2.0 m/s and 4.0 rad/s envelope. The IK still shapes acceleration and jerk and
enforces joint and effort constraints. Mouse dragging does not establish a
repeatable benchmark or guarantee the requested measured speed.
