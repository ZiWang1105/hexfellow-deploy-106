# Camera calibration tools

All owned ZED 2i and wrist calibration tools are consolidated in
[`../calibration/`](../calibration/README.md). The entry point is
`./scripts/calibrate_camera.sh`, with explicit `head`, `right` and `left` camera
selection. The registry is
[`cameras.json`](../calibration/config/camera/cameras.json).

Run `./scripts/calibrate_camera.sh list` to see supported actions. The common
launcher covers head `run`, `fit`, `refine`, `validate`, `overlay`, `review` and
`verify`; right-wrist `run`, `fit`, `plan`, `preview` and `verify`; and camera-only
RGB/board detection for all three devices. Both RealSense wrists use the shared
eye-in-hand fit with explicit serial and flange checks. The
[workflow table](../calibration/README.md#commands) gives the operation order,
board placement and output requirements.

The head workflow includes coarse hand-eye calibration, landmark PnP refinement
and measured-joint arm overlays. The right D435i workflow includes the fixed 33°
mount, 30 rendered/sampled targets, automatic capture, eligibility records,
held-out fitting and guarded return. The left D405 uses the shared RGB/PnP worker
and eye-in-hand solver; its automatic motion plan has not been validated.

The right fixed-33° mount is installed **outside the right arm at default**.
Its explicit installation transform rotates the supporter, housing and optics
together by 180° about flange Z. The active outside-side route passes all 30
geometric/rendered board-view checks; earlier inward-side videos and the old
27/30 rehearsal are superseded for this mount. A new controller rehearsal and
physical calibration have not yet been completed for the corrected geometry.

Results remain explicit candidates and are not automatically activated in the
production native camera or policy adapter. Keep the current head RGB-D and
uncalibrated wrist RGB contracts as documented in [WRIST_RGB.md](WRIST_RGB.md).
