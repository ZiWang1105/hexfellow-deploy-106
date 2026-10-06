# Live, operator-paced ZED calibration

See [the complete calibration procedure](camera_calibration.md) for both the
coarse board fit and later landmark PnP refinement, current matrices and numerical
reproduction. This page describes the live operator controls and startup checks.

This launcher uses the current deployment at `../UMI_Hardware` without modifying
its working files. It imports that checkout's session, controller, gateway and
research checks. Camera/calibration additions stay on `zi/dev_camera`.

## Start

Keep the board rigidly attached to the right hand and the 40-degree camera mount
fixed. Provide clearance for the right arm and board, cable slack, a disconnected
charger and an accessible physical E-stop. The geometric model includes the robot,
assumed board placement, camera support and floor; it does not represent the room,
desks or cable. The extra taped board's actual payload is not identified.

Keep the lift and arms stationary at startup. If preparation reports the lift
near its upper boundary, use **Move to default pose · lift 300 mm** in the web
interface. This delegates the checked retreat and default movement to the existing
deployment controller. **Return arms to default** is a separate action that keeps
the lift at its current height.

The desktop robot Ethernet profile is `hexfellow-discovery` on `enp3s0`, with
IPv4 disabled and IPv6 link-local addressing. It now auto-connects with priority
100. The generic DHCP profile can time out and tear down the link-local address
during a session, so verify that `hexfellow-discovery` is the active profile.
To activate it manually:

```bash
nmcli connection up hexfellow-discovery
```

Read-only startup verification, with no control acquisition:

```bash
cd ~/projects/agentic-mobile-manipulator/third_party/hexfellow-deploy/calibration
./tools/camera/run_calibration.sh --mode live --probe
```

Start the operator session after checking the physical workspace:

```bash
cd ~/projects/agentic-mobile-manipulator/third_party/hexfellow-deploy/calibration
./tools/camera/run_calibration.sh --mode live --confirm-onsite-ready
```

Open **http://127.0.0.1:8772** after the feedback audit finishes. Each attempt
takes approximately 30 seconds. If the failure is a recovered feedback-skew event or an early stale-feedback
abort with clean connection shutdown, startup repeats the full read-only audit
up to three times. Every attempt
is retained in `startup-audits.json`; acquisition still requires a wholly passing
audit with unchanged limits. Charger/readiness faults, transport faults and failed shutdowns do not trigger
retries. An early stale-feedback abort is never accepted as a passed audit; only
a new complete 30-second audit can permit startup. Only
one application can own the ZED; stop the previous RGB/depth preview first.
The calibration worker supplies its own rectified-left preview and acquisition.
There is no depth computation because the current solve uses RGB corners and FK.

## Controls — automatic operation

The right-hand board is planned with its printed face pointing toward the robot's
right side (-base Y) when the arm is resting. `config/camera/board_right_centered.json`
records this orientation and the approximate attachment offset. The taped offset
and printed top direction are assumptions until physically checked/calibrated.
The candidate pool is bound to that attachment's hash; each session renders and
checks the poses again using its current joint state and limits.

1. If needed, click **Move to default pose · lift 300 mm** and wait for completion.
   Then click **Start calibration**. This acquires HOLD at the current pose, checks
   the first path, then automatically moves, settles for at least one second,
   captures, and advances through **30 poses**. You do not press Capture 30 times.
2. Watch **Pose N / 30**, the moving/settling/capturing label, and the accepted
   capture count. Keep this page open and stay near the physical E-stop.
3. **Pause / Space** requests controlled braking and disables automatic advance.
   **Start / Resume calibration** explicitly resumes. Browser loss, camera loss,
   capture synchronization/geometry failure, command rejection, or feedback interruption stops automatic
   progress; a recovered connection alone never resumes the sequence.
4. All 30 poses are attempted. Poor board detections are saved as **not eligible**
   and skipped. With at least 12 eligible images, **Solve runs automatically**,
   fitting approximately 80% and testing the remaining images. Insufficient data
   or pose diversity produces a report instead of extrinsics. Transforms are not
   applied automatically.
5. **Return arms to default** stops the sequence, waits for stationary feedback,
   and collision-checks a path to both arms' existing deployment default angles.
   **The lift and base remain at their current positions.** It can be used before
   or after completing the dataset. If no checked path exists, it remains stopped.
   A paused return is continued by pressing Return again. Return is not automatic
   after a live solve, and it does not discard saved captures.
6. **Stop & quit** stops and releases control without returning automatically.
   The browser is not a physical E-stop.

Manual Prepare / Next / Resume movement / Capture / Solve remain in the collapsed
**Manual controls** section. They are disabled while an automatic sequence runs.

Failures retain evidence under `runs/live-calibration/<timestamp>/`. The session
does not automatically restart after a controller fault. Recheck the physical
state and reports before launching again. New launches create new datasets; do not
combine data across a changed camera mount or shifted board attachment.

## Measured-pose capture acceptance

For the final endpoint of each positive calibration pose, capture readiness uses
one second of new, fresh measured feedback after the commanded reference has
settled. All joint speeds must remain within the existing stationary limits;
the observed configuration range must be within half the existing arrival
tolerance. Camera capture then independently brackets the image with stationary
feedback, checks synchronization and computes FK from the measured joint angles.

A right-arm target offset of up to **5 degrees per joint** is allowed for this
capture check. This is a bounded capture acceptance choice, not a claim that
arbitrary tracking error is safe. The default/return/intermediate-waypoint arrival
checks, base/lift/left-arm tolerances, feedback/torque limits and guarded stopping
remain unchanged. A calibration pose that cannot settle within 10 seconds is
marked invalid: the controller runs its controlled stop, verifies one second of
fresh stationary HOLD, and the sequence advances to a newly checked next path.
A right-arm intermediate-waypoint arrival timeout uses the same skip path.
Default/return timeouts and other hardware safety faults retain their guards. Joint bounds and at least
15 mm of modeled clearance are checked for the measured capture pose. The board
attachment is estimated; room obstacles, cables and compliance are not modeled.

The controller retains its commanded holding reference and gains; it does not
reset the reference to the encoder angle and remove the PD support torque. Each
next path is checked from both measured and retained reference configurations.
Changed origins invalidate that check. A capture rechecks its actual offset,
geometry and synchronization before saving it; old readiness is not sufficient.

The page displays the target offset and stationary window. Saved observations
include `settling`, `capture_offset`, `measured_geometry`, and `pairing`, with
`q` taken from the image-paired measured feedback. Poor image detections still
advance as not eligible. An unreachable target path or right-arm capture offset
over 5 degrees is saved as `eligibility: invalid`, with pose ID and reason, and
is excluded from the solver. Invalid poses count toward the 30 attempts.
Unsafe current geometry, failure to stop/stay stationary, stale feedback, network
or torque faults still pause or stop. Operator Pause/Stop overrides continuation.

Simulation regression with an unmodeled shoulder load (no robot connection):

```bash
../../../.venv/bin/python tools/camera/verify_measured_capture.py \
  --output runs/measured-pose-check-new
```

This tests two automatic captures with approximately 1.28 degrees of J1 tracking
error and checks that the second path includes the retained reference. It is not
hardware qualification or a gravity identification procedure.

Add `--inject-invalid` to reject the first approach and simulate an arrival
timeout at pose 3. The test requires invalid records for poses 1 and 3, successful
captures at poses 2 and 4, controlled stopping, and a stationary HOLD before
continuation. No real robot is connected by this regression.

## Implementation and timing

- `calibrate_robot.py`: launcher, localhost controls, capture/solve worker and
  deployment session integration. `--deployment` selects a different checkout;
  runtime mapping and existing research binding must match that checkout.
- `calibration_runtime.py`: adapter for `SessionController`/`session.run` and
  stationary timestamp pairing. It owns no SDK endpoint and sends no direct
  motor packets. The canonical deployment lock, source/config binding, device
  identity check, audit, publisher guards and stop handling remain required.
- `calibration_path.py`: separate geometry process. Selects 30 distinct poses with all 16 nominal ChArUco corners
  from the prior candidate pool after applying current controller limits with a
  0.15-radian additional target reserve and rendering ChArUco detections again.
  This reserve avoids the controller's near-limit constraints; no limit is widened.
  Each Next/Resume path is checked at maximum 0.005-radian L1 joint increments
  with at least 15 mm modeled clearance. Checked paths are saved separately.
- The deployment preparation generator retains acceleration/jerk history and
  caps requested arm segment speed at 0.15 rad/s. Pause invokes
  `core.request_stop()` through the normal session. Resume never teleports a
  commanded reference. All path targets are validated before starting a segment.
- `calibration_camera_worker.py`: separately owned SDK process (the SDK Python
  environment differs from the controller environment). Camera loss blocks new
  movement/capture and requests HOLD while active. Camera frames, board detection,
  filesystem I/O and solving run outside the control loop.

SDK IMAGE time is mapped to host monotonic time using a bracketed SDK CURRENT
timestamp query. The pairing records mapping uncertainty and requires a fresh
frame after the request, a stationary history spanning approximately 0.5 seconds
before and 0.05 seconds after the image, no feedback gap over 80 ms, and nearest
sample skew plus mapping uncertainty below 40 ms. Position-span and velocity
checks use the deployment's per-coordinate tolerances. Captures require at least
12 corners and the existing positive-depth/reprojection checks.

This is **stationary host-time pairing**, not hardware-triggered exposure
synchronization. It is designed for stopped-arm calibration. It must not be used
as an accurate moving-arm timestamp calibration. Stereolabs documents the SDK's
[IMAGE timestamp in Epoch nanoseconds](https://docs.stereolabs.com/docs/tutorials/image-capture)
and [CURRENT–IMAGE comparison](https://docs.stereolabs.com/docs/development/zed-sdk/modules/camera/multi-camera).

Outputs include `binding.json`, checked paths, the session's full measured/control
trace and stop report, camera metadata and original left images, all attempted
detections, accepted `dataset.json`, and `extrinsics-*.json`. Camera-to-lift and
board-to-right-flange are the fixed unknowns; compose lift FK to get camera-to-base.

## Simulation check

```bash
./tools/camera/run_calibration.sh --mode sim --auto --port 8773 \
  --snapshot runs/camera-calibration-40deg/sim-start-lift-lowered.json
```

`--auto` is rejected in live mode. Simulation uses the actual deployment session,
MIT physics plant, stop logic and publisher checks, including a deliberate
pause/resume and a return to the default arm pose. Each capture renders the measured simulated joints and runs the
same ChArUco/PnP detector; no ground-truth board pose enters the solver.
The physics plant is the deployment's nominal robot; camera/support/board geometry
is included in the separate path and rendering model, not a measured payload model.

## Verification results (2026-09-29)

- Calibration tests: **34 passed**.
- Full deployed-controller simulation: **30 accepted captures**, deliberate
  pause/resume, solve and controlled shutdown. Held-out error: **0.330 mm /
  0.157 degrees RMS**. Ground-truth recovery: camera-to-lift **0.324 mm /
  0.033 degrees**, board-to-hand **0.119 mm / 0.048 degrees**.
- Live read-only startup: passed, ZED opened, all connections closed cleanly,
  **zero motion commands**. The timing audit met the deployment acceptance limits;
  its stricter performance target was not met (reported grade: relaxed).
- The real board was outside the detected view during this check (zero corners).
  No real calibration dataset or physical-motion qualification is claimed.

Evidence: `runs/live-calibration/20260930T024655.190579Z/` (simulation) and
`runs/live-calibration/20260930T025647.899665Z/` (physical read-only startup).

## Automatic outward-board verification

The revised right-facing board completed 30 automatic captures and solving in the
actual deployment-controller simulation. Held-out RMS: 0.522 mm / 0.193 degrees.
Camera-to-lift ground-truth error: 0.411 mm / 0.023 degrees; board-to-hand error:
0.204 mm / 0.059 degrees. A post-solve feedback interruption paused continuation;
the simulation was then explicitly closed normally.

Separate simulation checks passed for Return during first-path preparation
(cancelling the pending path), and Return from the final calibration pose. Both
arms reached their deployment default angles with the lift unchanged, followed
by clean shutdown. Physical motion was not tested or commanded by these checks.
The geometry uses the deployment's existing wrist/gripper seam exclusions on
both arms. The faster broad-phase check matched the scalar check over 50 poses
(maximum numerical difference 1.4e-17 m). Calibration unit tests: 43 passed.

Evidence:
- `runs/live-calibration/20260930T032432.079967Z/verification-summary.json`
- `runs/automatic-interface-verification/result.json`
- `runs/automatic-return-final-verification/result.json`
- `runs/clearance-parity/result.json`

The `--auto` flag remains a simulation test driver. For real automatic capture,
use the normal live launch command and click **Start calibration** in the browser.

### Centered replacement route (2026-09-29)

Default attachment/candidate files are now `config/camera/board_right_centered.json`
and `config/camera/right_centered_candidates.json`. The attachment is a rough
photo estimate (120 mm along flange Z, 65 mm outward reference offset), not a
measured extrinsic. `plan_centered_calibration.py` generates centered candidates;
`verify_centered_calibration.py` validates dense sampled paths and renders them.
Current evidence is in `runs/centered-calibration/`: 30 views, 16 detected corners
each, maximum nominal center offset 75 px horizontally / 69 px vertically,
minimum image edge margin 359 px. Candidate projection screening includes combined
+/-40 mm Cartesian shifts and +/-8 degree X/Y ray rotations; this is a sensitivity
screen, not a guarantee about unknown real geometry or occlusion. Paths require
15 mm modeled clearance at <=0.005 rad L1 joint sampling. This is kinematic
simulation validation, not new deployed-controller dynamic qualification.
Room obstacles, cable routing and real attachment errors remain unmodeled.
The first approach is checked from recorded failed-capture joints; runtime
rechecks from current measured joints and controller limits.

The sequence now records image-quality failures as `not eligible`, retains their
images/reasons, and proceeds to the next pose. `dataset.json` separates eligible
`observations` from `ineligible`; progress counts all attempts. At the end it
requires at least 12 eligible images, fits on approximately 80% and checks the
held-out remainder. Existing solver diversity checks still apply. Insufficient
images/diversity produces a saved status, not invented extrinsics. Robot feedback,
stationarity, camera availability, cancellation and watchdog failures still pause.
No real robot commands were sent during this replacement-route validation.

### Full default-pose button

`Move to default pose · lift 300 mm` is available while stationary, including
before Prepare. It checks the modeled full-body posture path, then invokes the
existing deployment `default` command, including its upper-endpoint lift retreat.
It returns both arms and the lift to the configured default. Calibration uses the
coupled deployment session so the lift can execute this explicit command; normal
calibration paths retain the current lift/base coordinates and use preparation
control semantics. The base remains at its current pose. The controller's limits,
retreat guards, feedback watchdogs, and arrival tolerances are unchanged.

Wait for `Default pose reached`, then click `Start calibration`. The displayed
lift reading is measured; the default target is 300 mm with the deployment's
3 mm lift arrival tolerance. No separate lift terminal is needed. Pause cancels
the motion; a new explicit default command is required to retry it. The previous
`Return arms to default` remains an arms-only return preserving current lift.

### Current mapping: both J6 signs reversed (2026-09-30)

The camera calibration and left-hand validation launchers now default to
`config/local/hexfellow_gp80_highspeed_both_j6.json` and the matching
`hexfellow_gp80_highspeed_both_j6_research.json` in the deployment checkout.
Both arms use `[1,1,-1,1,-1,-1,-1]`. This adds the operator-requested **left J6**
correction to the earlier right J6 correction. Other signs, offsets and SDK
indices are unchanged. The shared mapping converts measured position/velocity,
effort feedback, position/velocity/feedforward-torque commands and joint limits;
this is not a display-only sign change.

The converted `config/hardware/hexfellow_default_pose_both_j6.yaml` preserves
the previous physical SDK default targets exactly, with the lift at 300 mm.
Left J6's saved model default changes from -0.1025657904 to +0.1025657904 rad.
Old mapping files and captured datasets remain available unchanged. Historical
left-hand captures require remapping through their original mapping before replay;
do not reinterpret their stored model coordinates as already corrected.

Evidence is in `runs/both-j6-correction/`: conversion tests, regenerated left-hand
simulation plan, controller replay, right-hand geometry checks and corrected
overlays. This records an operator-selected direction correction and offline
checks, not a new physical direction sweep or hardware qualification. Restart
the launcher to load it; no running controller is hot-patched.

### Earlier operator-selected right-wrist mapping B (2026-09-30)

The calibration launcher previously defaulted to the deployment-local
`config/local/hexfellow_gp80_highspeed_right_j6.json` and matching
`hexfellow_gp80_highspeed_right_j6_research.json`. B reverses **right J6 only**:
left signs remain `[1,1,-1,1,-1,1,-1]`; right signs are
`[1,1,-1,1,-1,-1,-1]`. This follows the operator's observation of reversed
real/sim wrist motion and explicit selection of comparison B. It is not an
independent physical direction or torque qualification.

The old J357 mapping and research binding remain unchanged for reproducibility;
other entrypoints explicitly selecting that old file still use its old signs.
The read-only portable template and converted default pose are under
`config/hardware/hexfellow_gp80_highspeed_right_j6.json` and
`config/hardware/hexfellow_default_pose_right_j6.yaml` in this camera worktree.
The default pose was converted through original SDK coordinates; its SDK motor
position targets are unchanged, including the wrist, and lift target remains
300 mm. Recorded snapshots must be reinterpreted from SDK telemetry when using B.
The default simulation snapshot uses the converted default configuration.

Offline evidence: `runs/right-j6-correction/`. Conversion tests cover feedback,
position/velocity/torque commands, asymmetric joint limits, unchanged left-arm
mapping, and default-pose capture metadata/hashes. All 30 poses render 16 corners
and pass sampled model path checks from the converted default. The deployed
simulation session passed upper-lift retreat, full default, and the first
centered automatic capture, then shut down normally. No physical commands were
sent. Historical calibration datasets/poses are not silently rewritten.

### OpenCV 5 camera-worker crash fix (2026-09-30)

The live ZED environment returns ChArUco corners as `(N,2)` and IDs as `(N,)`;
the simulation environment uses `(N,1,2)` and `(N,1)`. OpenCV 5's corner overlay
still requires the two-channel shape, so drawing previously raised an assertion
and terminated the camera worker. The watchdog correctly paused the robot, but
its generic stale/disconnected message hid this cause.

Detection now normalizes point/ID shapes before drawing and PnP, and flattens both
projected and observed coordinates before computing reprojection RMS (preventing
cross-version broadcasting). Overlay failures are recorded as warnings. OpenCV
processing failures reject that frame while leaving the stream alive. The
watchdog now reports the worker's saved error when available. Robot/camera
freshness and capture acceptance thresholds are unchanged.

Regression tests `test_charuco_compat.py` passed in both installed OpenCV
runtimes. The full calibration suite passed 73 tests. An owned live-camera check
processed 20 frames and closed normally, without any robot command; evidence is
`runs/camera-opencv5-fix/report.json`. Existing sessions need a restart to load the
corrected camera worker. Poor board views can still be marked not eligible.
