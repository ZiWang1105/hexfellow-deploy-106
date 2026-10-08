# Robot camera calibration

The owned calibration toolset for the **ZED 2i head**, **D435i right wrist**
and **D405 left wrist** lives here. It includes the board detector, hand-eye
solvers, measured-joint capture/eligibility logic, browser controls, robot pose
planning, simulator checks, coarse head calibration, landmark PnP refinement,
and full-arm real/simulation overlays. The original source worktree is retained.

The saved head and right-wrist parameter snapshot, frame/TCP conventions, SDK
intrinsics, board dimensions and physical validation evidence are documented in
[Camera parameters](docs/camera_parameters.md). The snapshot is
[config/camera/camera_parameters_20261006.json](config/camera/camera_parameters_20261006.json);
the left D405 matrix remains `null` until calibrated.

Run commands from `third_party/hexfellow-deploy`:

```bash
cd ~/projects/agentic-mobile-manipulator/third_party/hexfellow-deploy
./scripts/calibrate_camera.sh list
./scripts/calibrate_camera.sh right run --mode sim --auto
```

Open `http://127.0.0.1:8772` for interactive runs; omit `--auto` to use the page.
`--auto` is available only in simulation. Live runs require an explicit operator
click to begin movement and the existing read-only startup audit.

| Camera | Serial | Constant calibrated transform | Workflow |
| --- | --- | --- | --- |
| ZED 2i head | 32701991 | LEFT rectified optical → lift link | Coarse board/hand-eye fit, landmark PnP refinement, measured-arm overlays |
| D435i right wrist | 243222074562 | color optical → right flange | Fixed 33° mount, 30 simulated board views, automatic collection/fit/guarded return |
| D405 left wrist | 230422271628 | color optical → left flange | Camera-only board detection and shared offline eye-in-hand fit; no validated automatic left motion route yet |

The source is split into `tools/camera/` (commands and web UI),
`src/policy_sim/calibration/` (detectors and solvers), `config/camera/` (board,
camera registry, candidate transforms and pose plans), `tests/calibration/`,
`docs/` and compact `artifacts/reference/` inputs. `assets_local/right_fixed33/`
contains the right camera mount used for the route study. `SOURCE_MANIFEST.json`
records original hashes, migrated hashes and dependency provenance.

The active right-wrist installation is **outside the right arm at default**.
`config/camera/right_fixed33_installation.json` rotates the whole CAD assembly
180° around flange Z while retaining its fixed 33° tilt. The regenerated
`realsense_d435i_right_fixed33_outside_plan.json` passes 30/30 geometric/board-view
checks. Old inward-side plans and videos are superseded; the launcher rejects
that route. See the right-wrist guide for current evidence and limitations.

## Commands

All commands below run from `third_party/hexfellow-deploy`. Use `list` to see
each camera's available actions and `<camera> <action> --help` for its options.
`--dry-run` prints the selected command without opening a device. The source
launcher uses the existing workspace environment; no conda activation is needed.

| Step | Unified command | Board/robot setup |
| --- | --- | --- |
| Head coarse route review | `./scripts/calibrate_camera.sh head review --output calibration/runs/head-review` | Offline; board attached to the outward face of the right hand |
| Head coarse collection | `./scripts/calibrate_camera.sh head run --mode live --confirm-onsite-ready` | Same rigid board attachment; Start once at port 8772 |
| Head coarse fit | `./scripts/calibrate_camera.sh head fit DATASET --output FIT` | Offline measured-FK/board-PnP dataset |
| Head independent arm views | `./scripts/calibrate_camera.sh head validate --mode live --confirm-onsite-ready` | Three left-arm poses; Start validation at port 8773 |
| Head landmark PnP refinement | `./scripts/calibrate_camera.sh head refine --points POINTS --refine-all --output FIT` | Offline matched real pixels / measured-FK 3D model anchors |
| Head measured-arm overlay | `./scripts/calibrate_camera.sh head overlay --dataset DATASET --extrinsics FIT --pose 1 --side left --whole-arm --output DIR` | Offline; repeat for captured pose IDs |
| Right wrist planning | `./scripts/calibrate_camera.sh right plan --output calibration/runs/right-wrist-outside/planning` | Offline; corrected outside-right 33° mount, board on table |
| Right wrist automatic collection | `./scripts/calibrate_camera.sh right run --mode live --confirm-onsite-ready` | Stationary table board; Start once at port 8772 |
| Either wrist board view | `./scripts/calibrate_camera.sh right camera --directory DIR` or `left camera --directory DIR` | Camera-only RGB/detection; does not supply robot FK |
| Either wrist extrinsic fit | `./scripts/calibrate_camera.sh right fit DATASET --output FIT` or `left fit DATASET --output FIT` | Offline synchronized stationary measured-FK/PnP pairs |

For head refinement, **two sets of 2D clicks are insufficient**: the point file
must pair real image pixels with 3D model anchors transformed using the measured
joint angles. The [head guide](docs/camera_calibration.md) documents this procedure,
the actual coarse/refined transforms and omitted-pose validation. The D405
camera/PnP and fit steps are included, but a camera-only image is not a complete
calibration observation. Its own geometry and automatic arm route still need
validation; `left run` does not reuse the right route.

**Right wrist:** Keep the existing 87 × 87 mm pattern stationary on the table.
The 33° route assumes its patterned-area center near `(0.722, 0.033, 0.725) m` in
the robot base frame. That location is a rough planning estimate, not measured
room geometry. The full operation and pose limitations are in
[the right-wrist guide](docs/realsense_wrist_calibration.md).

```bash
./scripts/calibrate_camera.sh right run --mode live --confirm-onsite-ready
./scripts/calibrate_camera.sh right fit /path/to/dataset.json --output /path/to/right-extrinsics.json
```

Live calibration now defaults to `--scheduling responsive`. It separates the
IK process, gateway owner, SDK workers and our camera/UI/planning workers onto
different physical CPU pools. Before ownership, it requests round-robin
priority 10 for the gateway and each existing SDK thread, using the user's
existing RT permission or Ubuntu RealtimeKit. Grant failure stops startup;
no automatic stop clearing or watchdog relaxation is added. The broker's
bounded continuous-CPU RTTIME limit persists until the worker process exits.
Priority and CPU affinity are recorded in each audit/session `cpu_scheduling`
report. New snapshot workers use the SDK CPU pool with ordinary priority.
These are application CPU pools; other programs and IRQs are not excluded.
Responsive mode also prefers feedback no older than 10 ms, waiting at most
4.5 ms within the original cycle/completion budget for a newer real sample.
The 40 ms validity, 20 ms skew and 8/15 ms P99 acceptance limits stay unchanged.

To check the connected robot without acquiring control or moving it:

```bash
./scripts/calibrate_camera.sh right run --mode live --probe
```

The probe exits after the read-only audit, without launching a live session or
camera worker. `--scheduling legacy` reproduces the previous scheduling for
diagnosis. A passing probe does not prove loaded motion reliability or physical
stop confirmation. RealtimeKit's policy is documented in its
[upstream README](https://github.com/heftig/rtkit/blob/master/README).

The UI shows pose IDs, attempts and eligibility. Start performs controlled
preparation, attempts all 30 views, saves a fit, then returns the arms through
checked waypoints. Pause requests HOLD. Invalid poses and poor images are
recorded and skipped after a stationary check; robot/control faults remain stops.
Capture FK uses **measured** joints, with up to 5° target offset per moving arm
joint. A result is a calibration candidate, with held-out closure/reprojection
checks; it is not automatically selected by the production deployment.

**Head:** The board is rigidly attached to the right hand during coarse
calibration. Use the original head workflow, not the table-board wrist route.

```bash
./scripts/calibrate_camera.sh head run --mode live --confirm-onsite-ready
./scripts/calibrate_camera.sh head fit /path/to/dataset.json --output /path/to/head-coarse.json
./scripts/calibrate_camera.sh head validate --mode live --confirm-onsite-ready
./scripts/calibrate_camera.sh head refine --points /path/to/landmarks.json --refine-all --output /path/to/head-refined.json
```

For manual landmark refinement and overlays, run the commands in
[the complete head guide](docs/camera_calibration.md). That guide reproduces the
original coarse and refined fits, their frame conventions and their measured
limitations. Saved candidate extrinsics are in `config/camera/`; acceptance
annotations and existing accuracy reports are preserved. A source report's
absolute paths identify its original evidence location, not runtime dependencies.

**Camera-only view for either wrist:** This opens the selected RGB device and
runs ChArUco detection; it does not connect to robot control. A directory stores
`ready.json`, `status.json` and `preview.jpg`. The worker's existing request IPC
can save an image/result. A saved camera frame alone is not a synchronized
robot calibration observation.

```bash
./scripts/calibrate_camera.sh right camera --directory calibration/runs/right-board-view
./scripts/calibrate_camera.sh left camera --directory calibration/runs/left-board-view
```

Stop with Ctrl-C, or create the directory's `stop` file. D405 nonzero RGB
inverse-Brown distortion is handled by SDK deprojection of each detected corner
into pinhole pixels before PnP. Saved RGB stays native; `raw_image_points` labels
native pixels and `image_points` labels the transformed PnP pixels. Metadata
retains both the native distortion and the effective zero-distortion PnP model.
Reports score the declared PnP coordinates. This uses the
[RealSense SDK projection/deprojection API](https://github.com/realsenseai/librealsense/blob/master/include/librealsense2/rsutil.h).

For a left wrist dataset with `parent_frame: maver_x4_asm_mount1`, serial
`230422271628`, stationary synchronized measured-FK/PnP observations and explicit
intrinsics:

```bash
./scripts/calibrate_camera.sh left fit /path/to/left-dataset.json --output /path/to/left-extrinsics.json
```

The shared wrist fit needs at least 12 eligible views and holds out every fifth
view. `left run` intentionally reports the missing mount/route rather than using
the right camera's 33° geometry or commanding the wrong arm. A left automatic
route needs its own mount model, tabletop assumptions, rendered visibility,
reachability and connecting-path checks.

## Frames and deployment integration

`T_A_from_B` maps B coordinates into A. The base is
`maver_x4_asm_maver_x4_base_link`, the exact URDF/IK base. Head calibration stores
`T_mount_from_camera` relative to `maver_x4_asm_URDF_link_1` (moving lift). Wrist
calibration stores `T_parent_from_optical` / `T_wrist_from_camera`, relative to
`maver_x4_asm_mount2` on the right or `maver_x4_asm_mount1` on the left.

```text
Head:   X · C_i = G_i · Y    G_i = T_lift_from_hand(q_measured)
Wrist:  G_i · X · C_i = Y    G_i = T_base_from_wrist(q_measured)
Runtime camera pose: T_base_from_camera(q) = FK_base_from_parent(q) · X
```

The lift height is part of FK; a camera-to-base matrix copied from one capture
is not constant while the lift or wrist moves. The two wrist serials and parent
frames are checked before fitting through the unified launcher.

Control/model imports come from the current sibling `../UMI_Hardware` submodule;
only the calibration package is overlaid here. Relative `urdf`, `assets`, and
`config/control` symlinks reuse its robot assets. No controller copy or old IK
implementation is introduced. The calibration runner uses the existing UMI
session API as a dedicated operator tool. It is separate from the production DA
policy/actuation processes: close those and other device viewers before starting
it. The production uncalibrated-wrist RGB contract remains unchanged until a
separate explicit calibration selection/integration is made.

The launcher uses the parent workspace `.venv`. It reads the current mapping
and research-release paths from `../../configs/local/desktop.yaml`; explicit
`--robot-config` and `--research-release` override them. Public mappings under
`config/hardware/` are historical test fixtures, not live defaults. Robot/source
bindings and startup guards are preserved, not automatically regenerated.
Camera workers use their existing SDK environments, without copying proprietary
SDK binaries. Optional location overrides:

```text
HEXFELLOW_CALIBRATION_PYTHON     controller/solver interpreter
HEXFELLOW_CALIBRATION_UMI        shared UMI_Hardware checkout
HEXFELLOW_CALIBRATION_DESKTOP    desktop configuration YAML
HEXFELLOW_ZED_PYTHON             interpreter with pyzed.sl
HEXFELLOW_REALSENSE_PYTHON       interpreter with pyrealsense2
ZED_DIR                         local ZED SDK directory
```

## Outputs and offline verification

Outputs are under `calibration/runs/` and are ignored by Git. Run manifests retain
input/config/source hashes. Coarse/PnP reference pairs are copied; large source
videos, trace histories and raw device-health dumps are not duplicated.

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 MUJOCO_GL=egl \
  ../../.venv/bin/python -m pytest -q calibration/tests/calibration -c calibration/pytest.ini
../../.venv/bin/python calibration/tools/camera/refit_coarse_reference.py \
  --output calibration/runs/head-coarse.json
../../.venv/bin/python calibration/tools/camera/fit_landmark_extrinsics.py \
  --points calibration/artifacts/reference/camera-calibration/landmark-points.json \
  --refine-all --output calibration/runs/head-pnp.json
../../.venv/bin/python calibration/tools/camera/verify_wrist_ground_truth.py \
  calibration/runs/<simulated-session> --output calibration/runs/wrist-ground-truth.json
```

The corresponding unified simulation commands are `head review`, `head verify
--route DIR --output DIR`, `head validate --mode sim --auto`, `right run --mode
sim --auto`, and `right verify SESSION --output FILE`. Verify accepts recorded
simulation evidence, not physical data. To preview the corrected right target
poses with an open gripper after planning:

```bash
./scripts/calibrate_camera.sh right preview \
  --source-scene calibration/runs/right-wrist-outside/planning/geometry/scene.xml \
  --poses-only --output calibration/runs/right-wrist-outside/open-gripper
```

Simulation defaults to the explicit `both_j6_sim_start.json` snapshot at 300 mm.
`--snapshot FILE` selects another simulation-only start. During migration, the
current UMI controller rejected the historical 422 mm start during lift retreat
with `BoundaryStateUnrecoverable`; the failed test is retained. A passing test at
300 mm does not qualify that endpoint retreat or physical motion. No shared
controller limits were relaxed. See `VALIDATION.json` for the actual migration
checks and outcome.
