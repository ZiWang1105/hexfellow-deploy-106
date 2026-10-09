# Left D405 calibration pose design

The left wrist has a dedicated **30-pose plan and calibration runner**, using the actual
`D405_LEFT_V1_FIXED32` mount mesh and the D405 housing reference. It does not
reuse right-arm joint targets, right-camera intrinsics or the D435i pupil offset.
No robot was commanded. Subsequent camera-only streaming verified the actual
D405/D435i profiles and replaced the initial nominal D405 K.

## Installation and board assumptions

- Camera identity: D405 `230422271628`; parent `maver_x4_asm_mount1`.
- Optical frame: `realsense_230422271628_color_optical`, X right / Y down / Z forward.
- Fixed 32-degree CAD; **outside-left orientation is a planning assumption**.
  The full mesh/housing/pupil is rotated 180 degrees around flange Z. At default
  the housing center is 70.975 mm farther toward base +Y than the left wrist.
  This must match the physical installation. Edit `left_fixed32_installation.json`
  and regenerate if the actual camera is on the other side.
- Keep the existing **87 x 87 mm printed pattern** flat and stationary on its
  rigid 100 x 100 mm backing, on the table. Do not attach the board to the wrist
  for this eye-in-hand calibration. Square 17.4 mm, marker 11.6 mm, 5 x 5 squares,
  `DICT_4X4_50`.
- Nominal board center `[0.68, 0.12, 0.72]` m in the URDF/IK base frame:
  forward +X, robot left +Y, up +Z. This is an operator-placeable planning
  location, not a measured location or an extrinsic estimate. The board stays
  fixed; the solver estimates its actual base transform.
- Nominal table footprint 0.50 x 0.64 m; surface Z = 0.718 m; backing front at
  Z = 0.720 m. Actual furniture, cables and room clearance require inspection.
- Lift remains at **300 mm**, right arm remains at its documented default, and
  the left gripper is open (approximate 80 mm parallel-jaw model).

Default intrinsics now use **actual D405 SDK calibration**, verified on
2026-10-08 by simultaneous camera-only RGB streaming: 1280 x 720 at 30 Hz,
`fx=651.4862670898438`, `fy=650.7327880859375`,
`cx=643.0418701171875`, `cy=362.6119689941406`.
The initial nominal horizontal focal length was 710.792 px and is superseded.
Native distortion is inverse Brown-Conrady with nonzero coefficients; the
worker converts detected corner pixels using SDK deprojection before
zero-distortion PnP. Raw images remain native. The saved profile is
`config/camera/realsense_d405_rgb_intrinsics_20261008.json`.
This verifies intrinsics/image dimensions, not the CAD optical-to-wrist
extrinsic. A later physical acquisition now provides a separate measured-FK
candidate; see [the saved wrist results](wrist_parameters_20261008.md).
The CAD remains nominal geometry, and this export does not enable left-camera
geometry in the production policy.

## Offline commands

From `~/projects/agentic-mobile-manipulator/third_party/hexfellow-deploy`:

```bash
./scripts/calibrate_camera.sh left plan
./scripts/calibrate_camera.sh left preview
```

The first command creates or revalidates
`calibration/config/camera/realsense_d405_left_fixed32_plan.json`. Revalidation
retains the checked connected joint route and tests it with the current profile;
`--regenerate` performs a new candidate search. The second
independently checks the full route, renders/detects every board view and tests
the hand-eye solver against nominal simulator truth.

The checked route includes default -> raised approach waypoint -> poses 1–30 ->
checked return to default. Endpoint-only checks would miss the tabletop crossing
on the direct approach; the extra waypoint avoids that modeled obstruction.
Only left-arm joints change. The lift and right-arm coordinates remain fixed.

Outputs:

- `calibration/runs/left-wrist-calibration-preview/left-d405-30-poses.mp4`:
  60-second static pose preview, 2 seconds per pose, robot and camera side by side.
  It is **not** a dynamic MIT controller replay.
- `.../pose-01.png` through `.../pose-30.png`: full-size paired views.
- `.../validation.json`: independent path and solver results.
- `.../rendered-observations.json` and `.../simulated-fit.json`: simulated evidence
  and fit, kept separate from physical calibration candidates.
- `calibration/artifacts/reference/left-d405-sdk-planning-20261008/validation.json`:
  portable validation summary, with source/plan hashes.

## Verification results (2026-10-08)

30/30 rendered views detect at least 14 ChArUco corners using the measured SDK
intrinsics. Camera-to-board distance is 0.245–0.359 m. The upper-center placement
avoids the open jaws below the board. Nominal rotation excitation has three nonzero singular values:
4.1252, 1.9561, 0.5562 rad; camera-center spans are 202, 217, 109 mm.

The independent checker evaluates **7,872 path samples** (the planner's 7,875
also includes repeated default-state samples), at no more than 0.005 rad summed
joint change between samples, against 547 modeled collision pairs. Minimum
sampled separation is **15.00018 mm**, close to the configured 15 mm requirement.
This is nominal sampled model clearance, not physical safety qualification.

Using rendered corner PnP and FK only, the existing Park initialization plus
robust gradient-descent solver fits poses excluding held-out IDs 6, 12, 18, 24,
30. Recovery against nominal simulator truth:

| Transform | Translation error | Rotation error |
| --- | ---: | ---: |
| Left optical camera -> left wrist/flange | 0.241 mm | 0.086 degrees |
| Stationary board -> robot base | 0.101 mm | 0.043 degrees |

No nominal ground-truth matrices are provided to the fit. These numbers establish
internal rendering/solver consistency, not accuracy on the installed camera.

## Before physical acquisition

1. Connect the D405 and check its camera-only view:
   `./scripts/calibrate_camera.sh left camera --directory /tmp/left-d405-board`.
   This does not move the robot. Confirm the actual serial and active profile
   in the resulting `ready.json`.
2. Confirm the 32-degree/outside-left installation, inspect the actual tabletop,
   and place the board near the planned workspace. Keep it fixed for the sequence.
3. Regenerate and preview using the camera worker's actual pinhole metadata:
   `./scripts/calibrate_camera.sh left plan --intrinsics /tmp/left-d405-board/ready.json`.
   A different board location can be supplied with `--board-center X Y Z`.
4. Stop any camera-only worker and other robot-control sessions before running
   the left calibration. It uses its dedicated left plan and D405 serial; it
   does not reuse the right-arm targets.

## Live command and website order

```bash
cd ~/projects/agentic-mobile-manipulator/third_party/hexfellow-deploy
./scripts/calibrate_camera.sh left run --mode live --confirm-onsite-ready
```

Open `http://127.0.0.1:8774` and click **Start calibration** once. The guarded
startup audit is read-only; control is acquired only after it passes. Start
restores the default arms/lift, then uses the checked raised approach waypoint,
attempts all 30 left-arm views, captures stationary measured-angle observations,
fits the extrinsics, and returns through checked waypoints. **Pause** stops
automatic advance; **Start** resumes. **Move to default pose** sets the lift
and arms; **Return arms to default** keeps the current lift height. The page
shows the left D405 identity, pose ID, eligible/invalid counts and active-left
joint settling error.

Keep the board fixed on the tabletop throughout. Use the left workspace above,
with the gripper open. Keep this page open for the operator heartbeat. Invalid
capture targets/images are saved and skipped after stationary HOLD; network,
ownership, feedback and physical-limit faults still stop control. Capture permits
5 degrees of target offset per active left joint and uses the actual angles,
not the targets, for FK. Other joints keep their existing tolerances.

Each run saves under `calibration/runs/wrist-calibration/<UTC timestamp>/`:
`dataset.json` records all attempted poses, raw images are under
`camera/<capture-id>/left.png` (a historical IPC filename, not the right/left
camera selection), and `extrinsics-*.json` contains the fit and held-out results.
Candidates are never silently applied to the policy. The runner checks actual
SDK serial, optical frame, resolution, FPS and K against the checked plan before
starting the motion session; changed profiles require revalidating the plan.

For simulation-only rehearsal from the planned default, make an explicit
snapshot and run the same controller:

```bash
python3 - <<'SNAPSHOT'
import json
from pathlib import Path
plan = json.loads(Path('calibration/config/camera/realsense_d405_left_fixed32_plan.json').read_text())
Path('/tmp/left-d405-start.json').write_text(json.dumps({'q': plan['initial_q']}))
SNAPSHOT
./scripts/calibrate_camera.sh left run --mode sim --auto --snapshot /tmp/left-d405-start.json
```

`--auto` cannot run live. Simulation uses the recorded D405 pinhole K and 80 mm modeled
open jaws; live uses native SDK RGB and SDK corner deprojection. Physical mount
orientation and actual furniture remain separate from nominal simulation.

The physical solve uses
`FK_base_from_mount1(q_measured) @ T_mount1_from_camera @ T_camera_from_board = T_base_from_board`.
The fixed extrinsic is optical camera -> left flange, not camera -> robot base
or camera -> TCP. Runtime camera-to-base is measured FK times that fixed fit.

## Full deployment-controller rehearsal (2026-10-08)

The left runner completed the automatic 30-pose sequence in MIT simulation,
including Pause/Resume, real stationary-feedback admission, matched RGB/FK,
fitting before return, checked return to default and controlled shutdown.
All 30 captures were eligible with 16 corners; no capture pose was rejected.
Camera-to-wrist recovery error against simulator truth was **0.43047 mm /
0.12461 degrees**. Six held-out views had **0.15488 px RMS** reprojection error.
Measured FK independently matches the recorded FK. The solver never receives
the truth matrices. Calibration tests: 202 passed; deployment tests: 261 passed.

One scheduler-delay HOLD occurred after pose 11 when all simulation processes
were restricted to one background CPU. Expanding only these simulation processes
to CPUs 20–23 and explicitly resuming through the existing path check completed
the run; no timing, ownership or motion limit changed. The earlier closed-jaw
preliminary run was stopped deliberately and is kept separately. These are
model/controller results; no physical robot or camera was opened for this
rehearsal and no physical left extrinsics have been accepted or applied.

Evidence: `artifacts/reference/left-d405-live-runner-20261008/validation.json`
and `ground-truth.json`. Full raw session: 
`runs/left-d405-live-runner-20261008/open-jaw-sequence/`.

## Startup failure diagnosis

A rejected audit may have latency percentiles below their limits but still
fail a single IK/braking check. The runner now saves `audit/cycle-failures.json`
and prints that cycle's cause, measured lift state and actual latency values;
it no longer calls an incomplete audit an exceeded percentile limit. No
failed audit is accepted and no guard is weakened. Recorded run
`20261009T024041.597628Z` had one lift-boundary rejection at cycle1203 near
the422mm upper endpoint; a fresh motion-free30-second probe passed1500/1500
cycles. See `artifacts/reference/left-d405-startup-diagnosis-20261008/validation.json`.


## 2026-10-08: False torso/jaw collision during default preparation

The measured hanging-arm start from `wrist-calibration/20261009T025124.986613Z`
was clear. At 7.6% of interpolation to default, MuJoCo returned zero for the
torso/left-positive-jaw convex-mesh pair. An independent separating plane
certified 30.0706 mm of clearance using all compiled mesh vertices. The wrist
checker now resolves these degenerate queries with this conservative certificate;
no collision pair or 15-mm requirement was removed. FCL only supplies a candidate
axis, whose separation is independently checked against vertex extrema.

From that measured start, default, all 30 poses and return pass 15,434 sampled
checks with measured/reference origins (minimum 15.00018 mm); 208 calibration
tests pass, including real overlaps and the recorded failing transition. This is
offline model validation, not new physical motion or calibration acceptance.
Evidence: `calibration/artifacts/reference/left-default-clearance-20261008/`.
Restart the existing runner to load the correction, then click Start calibration.


## 2026-10-08: RealtimeKit startup burst denial

RealtimeKit is the desktop service that grants selected control/SDK threads
round-robin CPU priority. The 20:18:37 daemon log explicitly reported
`Reached burst limit for user '1000', denying request.` The installed daemon
default is 25 actions per 20-second burst. Two closely spaced starts requested
priority for compute, gateway and SDK threads within that window. The affected
audit had zero cycles and zero motion commands; no robot control was acquired.
Source audit: `runs/wrist-calibration/20261009T031835.850384Z/audit/summary.json`.

The runner now preserves the failed report, waits 21 seconds, and retries a
new complete read-only audit once only after verified motion-free shutdown.
A persistent permission denial remains a startup failure. It does not resume a
live controller, accept a failed audit, bypass priority verification, or change
feedback, motor, stopping or timing limits. The D-Bus message alone cannot
distinguish rate limiting from other permission denials; this run's daemon log
establishes the cause. The same live command loads the correction on restart.
Evidence: `artifacts/reference/rtkit-startup-20261008/`.
