# Both wrist RGB views and measured calibration geometry

The default native policy observation contains four image blocks, in this order:
head RGB, registered head depth visualization, `left_wrist` RGB, and `right_wrist`
RGB. Metric head depth samples, head calibration and measured proprioception
remain unchanged. Both wrists are RGB-only. The action schema is unchanged.

## Selected physical cameras

| View | Device / serial | Candidate | Held-out RMS at 1280 × 720 |
| --- | --- | --- | --- |
| left_wrist | D405 / 230422271628 | realsense_d405_left_candidate_20261008.json | 7.9218814693 px |
| right_wrist | D435i / 243222074562 | realsense_d435i_right_candidate_20261008.json | 13.6236857923 px |

Candidates are under `calibration/config/camera/`. The right selection supersedes
its October 6 extrinsics for new runs. Historical records remain unchanged.
Both retain `candidate_requires_physical_validation`; neither published fit is
independent physical accuracy certification. See
[measured parameters](../calibration/docs/wrist_parameters_20261008.md).

Every wrist observation includes its serial/frame identity, calibration SHA-256,
quality metadata, active-profile K, encoder timestamp and matched physical joints.
`camera_T_base_from_optical` maps optical points into the robot base using measured
FK through the corresponding flange (`mount1` left, `mount2` right).
`left_tcp_T_from_wrist_optical` or `right_tcp_T_from_wrist_optical` maps that
camera's optical points into its corresponding configured TCP. This fixed
transform includes the actual tool offset. Its composition with the matched TCP
pose equals the camera-to-base transform. Translations are meters.
Board poses, calibration corners and dataset landmarks are never policy inputs.

## Image geometry

Each component requests 640 × 480 RGB at 30 FPS by default and reads its own
active SDK intrinsics. It does not scale the saved 1280 × 720 K across aspect
ratios. The D405 has nonzero inverse Brown-Conrady distortion: at startup, SDK
projection constructs a destination-pinhole to source-native pixel map. RGB is
remapped into the supplied pinhole K without changing optical orientation.
Inverse coefficients are not passed to OpenCV as forward Brown coefficients.
Black remap borders contain no observation; images have no annotations.

Metadata identifies `sdk_rectified_pinhole` or `native_pinhole`. The right D435i's
zero-distortion stream passes through unchanged. Health records preserve native
SDK K, model and coefficients separately from the processed image geometry.
RGB does not provide metric depth along a wrist pixel ray; head depth cannot be
indexed with wrist pixels. Candidate extrinsics do not remove range uncertainty.

## Runtime and recordings

The shared runtime YAML now uses `wrist_cameras.left_wrist` and
`wrist_cameras.right_wrist`. Each block selects enabled, serial, calibration,
width, height and fps. Native capture has a separate process, RGB shared-memory
topic and feedback sampler per camera. Simulation does not open these devices
or fabricate wrist imagery. An explicitly selected historical `wrist_camera`
configuration remains readable.

Frames without measured encoder feedback within the configured bound (at most
80 ms) are dropped. Policy requests require all enabled wrists to be at most
0.5 s old and within 0.15 s of the head observation timestamp. Missing or stale
views cause a named waiting status, never silent removal. These are bounded
host receipt-time matches, not hardware exposure synchronization.

Each run writes:

- `cameras/left_wrist/wrist_calibration.json` and `wrist_camera_health.json`.
- `cameras/right_wrist/wrist_calibration.json` and `wrist_camera_health.json`.
- `policy/left_wrist_rgb_NNN.png` and `policy/right_wrist_rgb_NNN.png`.
- Exact image bytes and side-specific geometry in the saved API requests.

The dashboard at port 8765 shows head RGB/depth and both labeled wrist views.
`/left_wrist.jpg` and `/right_wrist.jpg` return HTTP 503 for unavailable/stale
images. `/status` lists each view under `wrists`. Legacy `/wrist.jpg` remains
available when a legacy single-view configuration is explicitly used.
The offline request auditor checks both calibration hashes and measured FK.

To intentionally omit a physical view, use `--set
wrist_cameras.left_wrist.enabled=false` or the corresponding right key. This
changes the input set and must not be described as a four-image run. The normal
real-agent command requires no new flags. Camera settings do not alter robot
ownership, IK, MIT actuation or gripper execution.

## Validation scope

Offline tests cover both physical candidates, FK/optical-to-TCP composition,
serial binding, missing-view waiting, four-image API serialization, image archives,
calibration auditing, independent dashboard endpoints and SDK rectification.
The installed SDK deprojection agrees with sampled D405 remap rays to less than
0.001 pixel. This checks numerical geometry, not physical calibration accuracy.
No live robot command or remote model call was made for this integration.

## Historical uncalibrated validation (2026-10-05)

196 targeted tests passed: policy boundary, policy process, camera viewer, and
the complete deployment suite. Tests inspect the actual outgoing request body
with a stub API: exactly three images, unchanged head metadata/action schema,
pixel-identical archived wrist PNG, and no wrist depth/calibration fields.
Offline request audit checks the auxiliary image/history against the archive.
Tests cover stale/future/missing wrist frames, semaphore contention, native-only
YAML wiring and a dashboard 503 response for stale wrist pixels.

The actual RGB-only component captured the connected D435i through the real DA
RGB topic. 100 distinct frames were observed at 29.98 FPS; maximum consumer age
was 5.66 ms, 101 frames were published, and there were no publication misses or
capture errors. It exited normally and released the camera. Robot SDK and ZED
imports were blocked in that probe. There were no model calls or robot commands.
Raw evidence: parent `runs/wrist_rgb_policy_check`; compact report in
`evidence/20261005_wrist_rgb/probe.json`.

An initially broader test invocation also encountered an existing base-pulse
duration assertion (1.03 s versus 1.0 s), reproduced after loading the unchanged
HEAD contract module. That controller behavior is outside this sensor change.
The first invocation lacked a headless OpenGL setting; rendering tests require
`MUJOCO_GL=egl`. This validation is not a new agent pickup or simultaneous
head/wrist physical-control trial.
