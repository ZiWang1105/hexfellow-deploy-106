# Operator-paced camera calibration

For the implemented live deployment workflow, see [camera_calibration_live.md](camera_calibration_live.md).
The following describes the earlier playback-only rehearsal.

## Playback rehearsal

The local browser rehearsal is at `http://127.0.0.1:8770`. It runs the operator
procedure against the previously validated MuJoCo trace, shows the third-person
and camera views, saves explicitly synthetic image/joint pairs, and invokes the
existing hand-eye solver. It has **no robot connection or physical motion backend**.
The real ZED preview on port 8767 is a separate application.

Start from this checkout with the existing simulation environment:

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  ../../../.venv/bin/python tools/camera/calibration_console.py
```

Ctrl-C in its terminal stops the rehearsal server. This command requires the
existing route/trace and ground-truth-verification artifacts documented in
`camera_calibration_40deg.md`; the route is not regenerated or changed at startup.
Each launch creates a new output directory, never resumes motion automatically.

## Operator procedure

1. **Next pose:** move along the recorded approach to pose 1. Intermediate
   waypoints and the settling interval are included. Progress stops at that pose.
2. Inspect the full-resolution board view and detected-corner count.
3. **Capture (C):** save the paired image, measured simulated joints, FK,
   detected corners and timestamps. The robot view stays at the current pose.
4. **Next pose:** proceed only when the current pose has a saved capture.
5. **Pause (Space):** pause at any point in recorded travel. **Resume** is a
   separate deliberate action. Capture remains disabled between planned poses.
6. **Capture again:** retain the previous attempt and save another. The latest
   successful attempt is the one listed in the dataset. In rehearsal, this uses
   the same recorded image; a physical implementation must acquire a fresh image.
7. After all 30: **Solve & validate** fits 24 and tests six held-out poses.
   **Return to start** plays the previously checked reverse route. Neither action
   is implicit in capture, and solving does not apply a calibration to hardware.

The 2-second browser heartbeat timeout pauses playback; reconnecting does not
resume it. Duplicate/stale movement actions are rejected using a state version.
Pause takes priority even when submitted with an old version. The server binds
only to loopback and requires a per-session token for actions.

Output contains `dataset.json`, append-only `events.jsonl`, per-pose attempt
directories with `left.png` and `observation.json`, and versioned solver results.
No arbitrary pose jumps or Skip button are provided: the current validation
covers the specified sequence and reverse route, not untested shortcuts.

## Design for the existing physical deployment stack

This is the integration design, **not an implemented live-control mode**. The
original `~/projects/UMI_Hardware` checkout currently contains uncommitted changes
to session, drag, measured tracking and independent control. They were inspected
read-only and preserved. Integration must target that reviewed controller version,
not silently replace it with the older worktree copy.

Use one guarded control session and one camera worker. The browser never owns a
robot SDK endpoint or sends joint commands. The camera worker must own the ZED
exclusively and supply both preview and capture, replacing the separate preview
process when real acquisition begins. Rendering, image encoding, board detection,
disk writes and optimization stay outside the real-time control loop.

| Operator action | Existing stack integration | Completion condition |
| --- | --- | --- |
| Prepare | Highspeed launcher profile/mapping/source binding, onsite process lock, read-only preflight, existing gateway acquisition | Fresh feedback and validated current-to-first-pose approach; no automatic D/init movement |
| Next pose | `session.run(..., input_source=...)` with a dedicated calibration source/controller adapter | Follow the reviewed segment through existing reference/MIT publisher limits; then measured settling |
| Pause | `SessionController.command('stop', ...)`, existing `core.request_stop()` and gateway stop handling | Display STOPPING until measured stationary HOLD; never label a frozen UI as a stopped robot |
| Resume | Explicit adapter command from stationary HOLD | Replan/revalidate from the measured stop state; preserve velocity/acceleration histories, no jump to an old reference |
| Capture | One-shot capture request to camera worker plus a timestamped feedback ring buffer published by the active session | Fresh detected image, matched FK, stationarity over a time window; otherwise rejection with reason |
| Fault / browser loss | Existing stop/release and fault-paused path | No automatic restart; stale targets discarded; retain existing confirmed-stop recovery rules |
| Solve | Existing `calibration.handeye.solve`, running outside control | Save candidate matrices plus held-out metrics, without installing them |

Do not start `robot_snapshot.py` alongside an active controller: it takes the
onsite lock and creates another endpoint. Instead, publish copies of feedback
already measured by the active session. All processes must share the same
canonical lock across worktrees. The browser is an operator convenience, not a
replacement for the physical E-stop.

### Capture contract for real observations

The scene route currently starts from saved joints and estimates the taped board
attachment. Before enabling motion, obtain fresh joints and a visible stationary
board observation; update the board placement, room/cable geometry and payload,
and validate the deployed controller and current-state approach. The rehearsal's
instant pause does not qualify that controller's braking or resume trajectory.
Keep the base, lift and left-arm references fixed during collection.

At every pose, the acquisition worker should require:

- Confirmed stationary HOLD for a continuous interval (initial design: 0.5 s),
  using fresh joint velocities and joint-position span, with separate angular
  and linear thresholds. Thresholds must be checked against actual sensor noise.
- A **new** rectified-left ZED image taken after the request and within the same
  stationary interval; correct serial, resolution, intrinsics and 87 mm pattern.
- At least 12 well-distributed ChArUco corners, positive depth, and reprojection
  RMS below 1 pixel, matching the current simulated capture criteria.
- Image time bracketed by fresh measured joint samples. Establish the ZED
  timestamp-to-host-clock mapping and uncertainty explicitly. Do not equate SDK
  image timestamps with Python monotonic time or label a latest-sample lookup
  synchronized. Reject stale feedback, excessive skew or movement during exposure.
- Measured joint positions and FK in the declared lift/right-flange frames;
  never commanded targets. Save timestamps, clock mapping, raw image, camera
  intrinsics, board config, corner IDs, pixel coordinates and quality reasons.

The physical UI should show connection age, measured speed, image/joint skew,
corner count, reprojection error, current/next pose, capture count and stop/fault
state. Capture stays disabled with an explicit reason whenever a gate fails.
Depth/right RGB can be saved for diagnostics but do not enter the current solve.

### State transitions

```text
DISCONNECTED → READ_ONLY → PREPARED_HOLD
PREPARED_HOLD --Next--> MOVING → SETTLING → AT_POSE
AT_POSE --Capture--> SAVED_HOLD --Next--> MOVING
MOVING --Pause--> STOPPING → PAUSED_HOLD
PAUSED_HOLD --Resume + validated continuation--> MOVING
any active state --fault--> existing FAULT_PAUSED / STOP_UNCONFIRMED
30 captures --Solve--> candidate extrinsics + held-out report
```

Capture never advances the route; Next never captures implicitly. Pausing retains
completed captures. If the board shifts or the camera mount changes, start a new
dataset because the assumed constant unknown transform has changed.

## Verification of this rehearsal implementation

- 22 calibration tests passed, including capture-in-motion rejection, required
  capture before Next, stale commands, pause/resume, heartbeat loss and return.
- End-to-end rehearsal exercised 30 captures, one retained retake, solve and
  reverse-route completion. The synthetic solver converged, with held-out closure
  RMS 0.366 mm / 0.214 degrees, matching the independent verification dataset.
- No physical robot command was sent; deployment control files were unchanged.
