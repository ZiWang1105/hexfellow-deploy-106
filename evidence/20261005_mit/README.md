# Shared MIT evidence, 2026-10-05

Compact copied records and TCP plots from completed simulator runs. The manifest
identifies the exact original controller, trajectory, contact, configuration and
video records by SHA-256. Full recordings remain in the parent workspace's
`runs/` directories; large videos and API requests are not copied into Git.
No physical hardware validation is claimed.

- `da_mit_pickup_05`: **passed** the 360-second agent budget and 14-second final
  HOLD, with 18,646 accepted MIT frames, 30.33 cm final elevation, 209.06 seconds
  retained grasp, and zero robot/static contacts across 181,408 audited physics steps. All 41
  request inputs passed the policy-boundary audit: 40 completed responses and one
  explicitly canceled at the budget transition. The full recorded-state video is
  retained in the parent run; this directory includes its outcome-labeled frame,
  TCP plot, complete MIT statistics and dense physics evidence.
- `da_mit_final_nominal`: final-revision normal paired-pose controller case,
  1,200 accepted frames and one explicitly canceled pending frame, no fault.
- `da_mit_final_delayed`: final-revision 30 ms actuator delay plus periodic 30 ms
  compute delay, all 695 frames accepted, 70 warnings and no fault.
- `da_mit_unify_03`: normal paired-pose MIT controller case.
- `da_mit_delay_01`: preserved failing case before the lift model correction.
- `da_mit_compute_jitter_02`: periodic 30 ms work delay; passed controller case.
- `da_mit_delay_diagnosis_02`: 30 ms actuator delay plus periodic work delay; passed controller case.
- `da_mit_pickup_01`: full 240-second agent run; pickup failed (insufficient elevation, drift and contact).
- `da_mit_pickup_03`: retained grasp but early lift admission failure; full experiment failed.
- `da_mit_pickup_04`: retained grasp but incompatible lift QP bounds; full experiment failed.
- `native_preflight.json`: configuration-only check and source-binding refresh;
  no devices opened and `hardware_qualified` remains false.

See [validation notes](../../docs/VALIDATION.md) for criteria, observed issues and
model limitations. Retained object contact alone is not a completed pickup pass.
