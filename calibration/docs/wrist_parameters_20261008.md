# Measured wrist camera parameters, 2026-10-08

The current [parameter snapshot](../config/camera/camera_parameters_20261008.json)
contains the recorded physical **left D405** and **right D435i** fits. The camera
registry selects these versioned candidates for inspection. Original fit reports
are copied byte-for-byte, and the older 2026-10-06 snapshot remains unchanged.
Publishing parameters does not update the parent policy's runtime sensor selection.

| Parameter | Left wrist | Right wrist |
| --- | --- | --- |
| Device / serial | D405 / `230422271628` | D435i / `243222074562` |
| Parent flange frame | `maver_x4_asm_mount1` | `maver_x4_asm_mount2` |
| Optical frame | `realsense_230422271628_color_optical` | `realsense_243222074562_color_optical` |
| Fixed mount | Outside left, 32° | Outside right, 33° |
| Physical source run (UTC) | `20261009T033555.843603Z` | `20261009T003232.406454Z` |
| Eligible / recorded poses | 26 / 30 | 26 / 30 |
| Training / held-out poses | 21 / 5 | 21 / 5 |
| Held-out pose IDs | 5, 10, 15, 20, 29 | 9, 14, 19, 24, 29 |
| Held-out reprojection RMS / max | 7.9218814693 / 12.1092423498 px | 13.6236857923 / 25.5603414995 px |
| Held-out closure position RMS | 3.6359085105 mm | 6.7171380458 mm |
| Held-out closure angle RMS | 0.6116540280° | 2.1400617662° |
| Run-level result | Controlled session exit, passed | Saved fit; later arm-arrival failure, failed |
| Calibration status | Candidate requiring physical validation | Candidate requiring physical validation |

The left run excluded poses 21, 22, 23 and 27 because only 5–7 board corners
were detected; the capture gate requires at least 12. Both solvers converged.
The right run's later arrival failure does not erase its 26 saved usable pairs
or its completed offline fit. Its original failure is retained in the
[capture summary](../artifacts/reference/right-wrist-calibration-20261008/capture-summary.json).
Convergence and run completion are separate from calibration accuracy.
The visible held-out errors have not been removed or accepted as fine-positioning
qualification. Closure measures consistency, not error against ground truth.

## Files and provenance

Paths below are relative to the repository's `calibration/` directory:

| Data | Left | Right |
| --- | --- | --- |
| Full fit / extrinsics | `config/camera/realsense_d405_left_candidate_20261008.json` | `config/camera/realsense_d435i_right_candidate_20261008.json` |
| Numeric measured-FK / PnP dataset | `artifacts/reference/left-wrist-calibration-20261008/dataset.json` | `artifacts/reference/right-wrist-calibration-20261008/dataset.json` |
| Acquisition / quality summary | `artifacts/reference/left-wrist-calibration-20261008/capture-summary.json` | `artifacts/reference/right-wrist-calibration-20261008/capture-summary.json` |
| SDK RGB profile | `config/camera/realsense_d405_rgb_intrinsics_20261008.json` | `config/camera/realsense_d435i_rgb_intrinsics_20261008.json` |

The snapshot records hashes for both candidates, portable datasets and summaries.
Reports retain their original source paths and fit-input hash. The export also
records the final source dataset hash separately: acquisition logs can change
after a fit is saved. Room photographs, raw health/control logs, operator tokens
and machine-local research bindings are not included in the numeric exports.
Raw images and traces remain in ignored `calibration/runs/wrist-calibration/`.

## Transform direction and measured lift

Both reports contain identical aliases `T_parent_from_optical` and
`T_wrist_from_camera`. This is a **camera optical → wrist/flange** transform,
not camera → TCP or a fixed camera → base transform. It maps homogeneous column
vectors, with translation in meters and optical axes X right, Y down, Z forward.
The robot base is the URDF/IK frame `maver_x4_asm_maver_x4_base_link`.

```python
T_base_from_camera = T_base_from_wrist_measured @ T_wrist_from_camera
p_base = T_base_from_camera @ p_camera
T_TCP_from_camera = np.linalg.inv(T_wrist_from_TCP) @ T_wrist_from_camera
```

`T_base_from_wrist_measured` must come from FK using synchronized measured arm
angles and lift height. Camera-to-flange stays fixed; camera-to-base changes with
arm posture and lift. The both-J6 sign/order/offset mapping is preserved in the
snapshot. Neither chassis world localization nor a TCP offset is estimated here.

The board remains fixed on the table. For each recorded image the solver uses
`G_i @ X @ C_i = Y`, where G is measured base-from-wrist FK, X is the fixed
wrist-from-camera estimate, C is camera-from-board PnP and Y is the fitted
base-from-board pose. Y describes this acquisition's board placement, not a
permanent robot sensor parameter. Park initialization plus robust pseudo-Huber
gradient descent with Armijo search fits X and Y using only training poses.
Held-out predictions use `inverse(X) @ inverse(G_i) @ Y`; their image PnP poses
are not used to construct those predictions.

## SDK intrinsics and native image convention

Both cameras streamed native 1280 × 720 RGB at 30 FPS. K was read from each
camera's SDK profile and held fixed during extrinsic fitting. No 180° rotation
is applied to native images. The right nominal simulator optical-roll correction
is separate from the measured fit and the physical housing installation.

| Intrinsic | D405 | D435i |
| --- | ---: | ---: |
| fx | 651.4862670898438 | 909.8963623046875 |
| fy | 650.7327880859375 | 909.7846069335938 |
| cx | 643.0418701171875 | 643.9840698242188 |
| cy | 362.6119689941406 | 373.11004638671875 |

D405 native distortion is inverse Brown-Conrady with coefficients
`[-0.05051824077963829, 0.06193859130144119, -0.00009105065691983327,
0.0008188839419744909, -0.02104456163942814]`. Detected native corners are converted
through SDK deprojection into pinhole pixel coordinates. The saved PnP/projector
therefore uses zero distortion on those processed points; this does not mean
native D405 images are distortion-free. D435i native coefficients were all zero.
Neither run exports a calibrated depth profile or factory depth-to-color matrix.
The physical board is a 5 × 5 ChArUco pattern, 87 × 87 mm patterned area,
17.4-mm squares, 11.6-mm markers, `DICT_4X4_50`, on a 100-mm backing.

## Load or reproduce without opening devices

From the deployment repository:

```python
import json
from pathlib import Path
import numpy as np

root = Path('calibration')
parameters = json.loads((root / 'config/camera/camera_parameters_20261008.json').read_text())
camera = parameters['cameras']['left']  # Use 'right' for D435i.
T_wrist_from_camera = np.asarray(camera['T_parent_from_optical'])
K = np.asarray(camera['intrinsics']['K'])
```

```bash
./scripts/calibrate_camera.sh left fit \
  calibration/artifacts/reference/left-wrist-calibration-20261008/dataset.json \
  --output /tmp/left-wrist-refit.json
./scripts/calibrate_camera.sh right fit \
  calibration/artifacts/reference/right-wrist-calibration-20261008/dataset.json \
  --output /tmp/right-wrist-refit.json
```

These commands are offline. Fresh output files preserve the published candidates.
To collect new physical data, use the dedicated side-specific commands:

```bash
./scripts/calibrate_camera.sh left run --mode live --confirm-onsite-ready
./scripts/calibrate_camera.sh right run --mode live --confirm-onsite-ready
```

Run one physical controller at a time. The operator clicks Start at port 8774
for left or 8772 for right after the read-only startup audit passes. Both routes
use measured joints, stationary capture, up to 5° target offset, invalid-pose
recording and checked default return. Controller/network/health faults still
stop the session; poor images are skipped. Further independent overlay checks
are needed before treating either candidate as validated fine-positioning geometry.
