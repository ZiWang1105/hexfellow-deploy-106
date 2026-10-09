# Lift retreat correction, 2026-10-08

The right-wrist run `20261008T234924.788901Z` stopped during its initial lift retreat, before capturing any images. The controller's signed tracking-lag calculation incorrectly consumed outward boundary clearance while the lift was stationary and its command was downward.

The measured lift position was 0.4219995610694027 m; the device's upper bound was 0.4220003100977257 m. Actual clearance was 0.749028323 micrometers, one encoder pulse. Measured lift velocity was zero. The previous reference velocity was -0.00001978781328327739 m/s. Over the 0.04529894500505179 s reaction window, the old calculation reserved 0.896367066 micrometers of outward travel. That produced a false negative distance of -0.147338743 micrometers and the reported `BoundaryStateUnrecoverable`.

The shared controller in `third_party/UMI_Hardware/src/policy_sim/control/command_limits.py` now reserves only the excess positive closing component: `max(0, max(measured_closing_speed, 0) - max(previous_closing_speed, 0)) * reaction_s`. The measured and reference positions remain reserved, and reference velocity/acceleration history, physical bounds, uncertainty, acceleration/jerk limits, and rejection of real outward travel remain active. No threshold was widened.

Verification opened no physical devices and sent no robot commands. Offline replay used the exact recorded failed state, posture target, camera-independent TCP targets, and preceding reference histories. It produced an executable downward command at -0.000025062594454705406 m/s and held both arms fixed. A regression also keeps the endpoint encoder stationary for the first 30 control ticks, then follows the references and verifies transition from lift retreat into the normal default phase. The focused controller suite passed 52 tests; the calibration suite passed 196 tests. These results do not establish a successful physical retreat or completed calibration.

Evidence: [recorded cycle replay](../artifacts/reference/lift-retreat-fix-20261008/recorded-cycle-replay.json), [controller tests](../artifacts/reference/lift-retreat-fix-20261008/control-tests.xml), and [calibration tests](../artifacts/reference/lift-retreat-fix-20261008/calibration-tests.xml). The offline reproduction script uses the retained raw source run and the existing desktop configuration. The local research source binding was updated for this one changed file, with its previous value backed up under `runs/lift-retreat-fix-20261008/`; review flags, profile, and mapping were preserved.

Restart the right-wrist session in a desktop terminal:

```bash
cd ~/projects/agentic-mobile-manipulator/third_party/hexfellow-deploy
./scripts/calibrate_camera.sh right run --mode live --confirm-onsite-ready
```

After the read-only startup audit succeeds, open `http://127.0.0.1:8772` and click **Start calibration**. The initial default preparation performs the lift retreat before arm motion. Do not run a second robot controller alongside it. A new session still performs the normal startup checks; this correction does not turn a failed hardware audit into a passing one.
