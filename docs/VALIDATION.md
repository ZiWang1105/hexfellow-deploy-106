# Shared MIT validation — 2026-10-05

Fixed numeric DA control transport (2026-10-05)

Replaced high-frequency feedback/reference/measurement JSON topics with fixed
float64 DA ArrayCommWrapper layouts and added a separate numeric receipt topic.
Backend diagnostic JSON is now 10 Hz plus lifecycle/fault updates. Finite lock
waits, exact pending-frame retry, source-clock preservation and pre-IK feedback
refresh remain; no motion history reset or hardware watchdog relaxation.

140 deployment tests pass. Public run_agent.py HOLD completed 502/502 accepted
cycles. Three shared MuJoCo paired-pose fixtures with camera/dashboard survived
54 injected semaphore holds each: feedback 45 ms (500 retries), reference 20 ms
(283 retries), receipt 20 ms (acceptance waiting). No control/backend/shutdown
faults. The reference fixture canceled its final pending sequence at shutdown.
Local put/get microbenchmark savings are microseconds, not proof that JSON caused
the original 50 ms hardware feedback gap. No API or hardware run in this change.

The DA alignment audit retracts complete-parity claims and explicitly preserves
the unaligned recovery semantics: YAM resets motion history after chunk failure;
stateful PinkIK cannot safely copy that operation. Native SDK/firmware timing,
physical freshness root cause and lift stop confirmation remain unqualified.
See third_party/hexfellow-deploy/docs/DEPLOY_ANYTHING_ALIGNMENT.md and its
evidence/20261005_numeric_control directory.

Feedback refresh before stateful IK (2026-10-05, 21:12 run)

Physical real_20261005_211146 reused the same per-device feedback in references
2175/2176 and the failing solve. IK then rejected 45.94–50.88 ms ages under its
40 ms strict receive-age limit. Native receive records continued throughout;
arm telemetry maximum gaps in the saved window were about 10 ms, and no arm
parking stop was present at diagnostic capture. This localizes the missing
freshness between SDK receipt and controller selection, but does not prove
which of snapshot construction, SHM publication or SHM reading stalled.

Zero-timeout SHM reads can miss the semaphore and reuse cache; zero-timeout
feedback writes previously ignored failure. The shared backend now gives
feedback publication a bounded 1 ms lock wait, records publication outcomes
and snapshot timing, and the controller refreshes feedback before stateful IK.
Retries preserve original sensor times and remain bounded by the existing
command deadline, with a solve margin; no deadline or validity limit is raised.
Control records distinguish a lock miss from an old publication. 129 tests pass.
A 15-second shared MuJoCo run with camera/dashboard survived 54 injected 45 ms
feedback semaphore holds: 487 refresh retries, 724 generated/accepted commands,
no control/backend/shutdown fault. No physical rerun or API calls in validation;
physical root-cause attribution and lift-stop confirmation remain unresolved.
Evidence: deployment repo `evidence/20261005_feedback_refresh` and parent
`runs/feedback_retry_211231`.

Firmware timeout investigation (2026-10-05, 20:58 run)

Both arm firmware logs reported API communication timeout, followed by SDK
attempted auto-clear/reinitialization blocked by the wire guard. A 58.03 ms
reference-to-SDK-acceptance delay is recorded; no UDP history was saved, so this
is not proof of a wire outage or its CPU/network cause. Camera failure followed
loss of encoder publications, while capture remained healthy. Restored omitted
process-local GIL/GC settings through shared DA YAML and record actual runtime
settings. Native failures now retain bounded gate/SDK/UDP/KCP evidence before
close, written after stopping. 124 tests pass; 10-second shared MuJoCo HOLD with
camera/dashboard completed 504 accepted cycles, no faults or API calls. No
physical rerun or automatic stop clearing. Firmware root cause and lift stop
confirmation remain unresolved. Evidence: deployment repo
`evidence/20261005_firmware_timeout` and parent `runs/da_scheduling_205821`.

Current simulator runs use the same DA controller, actuator owner, command schema
and deployment profile as the native adapter. Arm physics now evaluates MIT
PD-plus-feedforward torque. No physical robot was operated for this validation.
Compact records, plots and SHA-256 hashes of the full original artifacts are in
[`evidence/20261005_mit`](../evidence/20261005_mit/README.md).

## Shared reference publication contention

The physical 20:44 attempt exposed a bug also present in the shared simulator
path: a single 1 ms reference-topic lock miss was treated as fatal. The fix
retains and retries the exact pending reference. A 15-second MuJoCo fixture with
camera/dashboard and 100 injected 20 ms semaphore holds completed with 257
recovered publication attempts, no control/backend/shutdown fault and exit 0.
115 deployment tests and recorded-data browser checks passed. The original run
had zero model calls; startup failure and agent waiting/output are now visible at
the top of the dashboard. [Evidence and scope](../evidence/20261005_publication/README.md).

## Physical RGB-D matching regression

The 20:33 real run failed in the camera component before any policy call. A
read-only reproduction found 180 fresh frames rejected by latest-only encoder
matching (median skew 94.74 ms versus the configured 80 ms). Encoder history
rescued all 180. The revised actual camera component completed a 20-second
passive run with 260 publications, zero matching failures and maximum absolute
encoder skew 5.27 ms. Freshness/skew limits were unchanged; no motion was sent.
See [sensor evidence](../evidence/20261005_camera/README.md). This is a sensor-path
test with real SHM and a concurrent image consumer, not complete agent execution.

## Completed pickup validation

**`da_mit_pickup_05` passed the complete simulated pickup experiment.** It ran the
fixed 360-second budget plus an uninterrupted 14-second final HOLD, closed normally
at 374.06 wall seconds and accepted all 18,646 generated MIT frames. The apple
finished 30.33 cm above its initial center of mass, held by opposing right jaws
without support for 209.06 simulated seconds. All 181,408 physics steps were
audited at 2 ms with complete coverage and zero robot/static contact steps.

The agent initiated 41 API requests: 40 completed responses and one explicitly
interrupted at the fixed-budget transition to HOLD. The boundary audit passed all
41 recorded inputs and 195 observation/history entries, including exact RGB/depth
bytes and physical-encoder FK. No action was invented for the canceled request.
Recorded API cost was $10.5734045. Native hardware remains unqualified.

Compute/submission p99 was 3.983 ms, solve interval p99 20.083 ms (maximum 26.820 ms),
and application latency p99 5.905 ms. Eleven budget overruns warned without
terminating control. Complete MIT vectors are recorded for every frame, with zero
position-reference mismatch and zero predicted effort-limit exceedances at the
source feedback. Actual actuator torque maxima are separately recorded.

At the end of HOLD, measured TCP-to-MIT-reference errors were 0.337 mm / 0.0559°
for the left arm and 3.065 mm / 0.3795° for the loaded right arm. The last active
right TCP target error before HOLD was 3.602 mm / 0.3063°; plots leave inactive
target intervals blank rather than assigning a fictitious HOLD target. Maximum
lift speed was 0.025213 m/s with a 0.025 m/s internal drive limit; this small
mechanical transient is retained in the evidence.

The full outcome-labeled replay and TCP plot are in
`runs/da_mit_pickup_05/analysis/replay/pickup.mp4` and
`runs/da_mit_pickup_05/analysis/tcp_tracking.png`. This successful episode follows
the preserved failed attempts below; it does not relabel them as passes.
The movie contains all 5,443 recorded frames at 15 Hz (362.867 simulated seconds).
It replays the exact compiled `plant.mjb`; final recorded-state evaluation matches
the original record exactly. The outcome label and loaded grasp were inspected.

## Completed controller cases

| Run under parent `runs/` | Accepted / generated | Compute + submit p99 | Solve interval p99 | Outcome |
|---|---:|---:|---:|---|
| `da_mit_final_nominal`, final revision, paired target, 24 s | 1200 / 1201 | 4.093 ms | 20.083 ms | Normal close, one warning, final pending frame canceled |
| `da_mit_final_delayed`, final revision, 30 ms application delay plus periodic 30 ms work delay, 28 s | 695 / 695 | 33.782 ms | 70.372 ms | Normal close, 70 warnings |
| `da_mit_unify_03`, paired target, 24 s | 1201 / 1202 | 3.189 ms | 20.075 ms | Normal close, one budget warning, final pending frame canceled |
| `da_mit_compute_jitter_02`, 30 ms work delay every 10 solves, 28 s | 1281 / 1282 | 32.483 ms | 39.459 ms | Normal close, 129 warnings, final pending frame canceled |
| `da_mit_delay_diagnosis_02`, 30 ms application delay plus periodic 30 ms work delay, 28 s | 704 / 704 | 33.015 ms | 69.576 ms | Normal close, 72 warnings |

Each run records `accepted_before_solve == sequence - 1` throughout. The baseline
ended below 0.000260 mm / 0.000052 degrees actual TCP target error. Both delayed
cases ended below 0.000267 mm / 0.000052 degrees. These tiny steady-state values
reflect the consistent idealized model, not achievable physical robot accuracy.
Transient target/reference/actual errors are retained in the plots rather than
hidden by reporting only the endpoint.

The deliberate delays produce warnings and longer intervals, not the old 40 ms
source-lease termination. This is not a worst-case scheduling guarantee. The
velocity source still comes from the fixed 20 ms PinkIK profile; the owner waits
for acceptance before the next solve, so delayed execution slows trajectory
progress rather than solving speculative intermediate references.
The shared command watchdog allows 100 ms from command creation. Upstream strict
feedback checks still impose 40 ms sensor age and 20 ms coupled-device skew;
those independent limits were not removed. Simulation provides synchronous
per-device snapshot timestamps, so these cases do not test real asynchronous
device feedback or establish hardware timeout reliability.

`scripts/analyze_mit_run.py` writes robot-relative TCP plots, compressed samples,
reference/actual/target error statistics, gripper states, commanded navigation and
compute/acceptance timing. New logs record active target poses directly. Older logs
only infer a submitted target when both independently computed TCP errors match
the controller's recorded error magnitudes; HOLD/navigation gaps remain blank.

## Bugs exposed by MIT physics

The initial `da_mit_delay_01` failed after 71 accepted references with PinkIK
`NoSolutionFound`; it is preserved as a failed attempt. The position-actuator tests
from the initial migration did not expose this physical-model behavior.

The original lift used an underdamped position motor without the native SDK's
speed-limited inner controller. Under a carried load, v1's lift HOLD request
followed the measured height, and the plain servo could continue descending while
PinkIK remained in BRAKING. The lift model now uses a bounded velocity demand and
PI force servo. Correcting it also exposed the source URDF's 10 N joint-level
clamp, inconsistent with the assembly's 1800 N actuator and approximately 158 N
robot gravity load. Both force clamps now use the explicit assembly limit.

Four motor regressions exercise actual MuJoCo forces, not a duplicate formula:

- Actual arm actuator **and transmitted joint** torques match upstream `mit_torque`
  for unsaturated and positive/negative saturated references.
- Reloaded configured XML preserves those torque semantics and matches the exact
  compiled model's gains, limits and gravity-compensation flags.
- A 25-second scalar lift test covers unknown 0/10/100 N loads, incremental position
  commands and HOLD targets updated from measurement. Drift is below 1 mm and final
  speed below 0.1 micrometer/s, with finite motor effort and bounded drive velocity.
- A 25-second full-robot test adds elbow motion and a 10 N downward wrist force;
  lift HOLD drift remains below 1 mm and final speed below 1 micrometer/s.

The coupled DA jitter run observed a maximum lift speed of 0.025103 m/s with a
0.025 m/s internal command limit. The small mechanical transient is recorded;
there is no hidden joint-position or joint-velocity clamp. Lift servo gains remain
explicit simulator assumptions, not measurements of the physical controller.

Runs through pickup attempt 04 saved the scene XML before runtime actuator
conversion. Their recorded-state replays reproduce geometry and contacts but that
XML alone cannot restart the original MIT dynamics. Subsequent runs export the
configured XML plus the exact compiled model; a regression verifies the reloaded
torque law and flags. Historical artifacts are preserved without rewriting them.

## Agent pickup attempts

`da_mit_pickup_01` completed its fixed 240-second wall-clock budget with 16 model
responses and 11,964 generated / 11,963 accepted controller frames. It navigated,
positioned the right arm and closed its gripper. Control compute/submission p99 was
2.975 ms, solve interval p99 20.080 ms and application latency p99 5.921 ms. There
was one budget warning and no runtime fault or premature reference-history advance.

**Pickup acceptance failed.** The apple reached a maximum 8.73 cm elevation and
ended at 2.36 cm, with opposing right-jaw contacts. It never met the 10 cm / 10 s
retention criterion; three sampled robot/table contact frames also reject the run.
After the last lift, both TCP heights fell approximately 6 cm while the lift target
followed measured height and the controller remained BRAKING. This prompted the
lift inner-servo correction above. A successful motor test does not relabel this
failed agent episode as successful.

The full 232.867-second simulated trajectory is exported at
`runs/da_mit_pickup_01/analysis/replay/pickup.mp4`. Replay final evaluation exactly
matches the original record. `tcp_tracking.png` shows desired, reference and
measured trajectories through the final descent. Video includes all recorded
frames, including the later loss of elevation.

`da_mit_pickup_03` used the corrected lift model. The policy grasped the apple with
its right hand, raised it 24.77 cm and retained it for 35.60 simulated seconds, with
zero sampled robot/static contact frames. Nevertheless, **the full experiment
failed**: after 9,609 generated / 9,608 accepted references, command admission
rejected a lift position/source-velocity mismatch at the 0.398 m model margin.
The final source was 0.398000061 m and reference 0.3979999999 m. It ended after
192.82 wall seconds, before its fixed 360-second budget and 14-second final HOLD.
This is retained-grasp evidence, not a completed-run pass. The cause was a
coordinate mismatch in command admission: PinkIK clips the measured lift to its
planning bounds before integration, while admission reconstructed the velocity
from the raw encoder value. At the upper margin, a 61 nm difference became a
3 micrometer/s mismatch. Admission now uses the same explicit planning projection;
a regression compares the real upstream projection and still rejects an altered
1 mm target. This correction was included in the completed attempt 05 above.

Its compute/submission p99 was 2.729 ms, interval p99 20.081 ms and application
latency p99 5.933 ms. Full replay video at
`runs/da_mit_pickup_03/analysis/replay/pickup.mp4` contains all 2,804 frames at 15 Hz
(186.93 simulated seconds), including the final fault interval; re-evaluated final
contacts exactly match the original record. RGB-D request audits passed for both
pickup runs: 16 calls / 70 history entries for attempt 01 and 19 calls / 85 entries
for attempt 03. The auditor checks exact sensor PNG bytes, metric depth,
robot-only encoder FK, allowed fields and observation/action history. Older
non-contiguous observation IDs are mapped to their original call-indexed archives
without changing recordings.

`da_mit_pickup_04` also **failed the full experiment**. It accepted all 12,156
generated references and retained a right-hand grasp for 97.13 simulated seconds,
ending 26.18 cm above the initial object height, with zero recorded robot/static
contact frames. It stopped at 243.93 wall seconds, before the 360-second budget
and final HOLD. The captured QP proves incompatible lift constraints: acceleration
required velocity at least 0.015 m/s while the position limit allowed at most
0.012913 m/s. This is a feasible-deceleration problem near a limit, not an inference
timeout or gateway queue delay. Compute/submission p99 was 2.764 ms, solve interval
p99 20.082 ms and application latency p99 5.935 ms. All 28 model requests and 130
observation/history entries passed the offline boundary audit. The entire recorded
trajectory is retained in `runs/da_mit_pickup_04/analysis/replay/pickup.mp4`.
The replay displays a persistent overall `RUN FAILED` label, QP error and missing
budget/final HOLD completion even while the per-frame retained-grasp flag is true.
The prior unannotated movie is preserved as `pickup_unannotated.mp4`.

Offline pickup acceptance requires the specified body to finish at least 10 cm
above its initial height, held by opposing jaws without support for at least 10 s,
a completed fixed budget and uninterrupted final constrained HOLD, zero sampled
robot/static contacts, and no control/backend/shutdown fault. Tests explicitly
reject early stops, wrong objects, short retention, support contacts and incomplete
HOLD. Privileged evaluator data never stops the live episode or enters policy.

The historical recorder samples contact evidence at 15 Hz. Passing those records
would not prove the absence of contact between samples. Subsequent runs add a
separate plant-only audit after every 2 ms integration step, including a coverage
check from the initial state. A transient robot/table contact or loss of one jaw
between video frames is captured and invalidates contact/retention evidence.
Tests exercise those missed-frame cases and missing initial/interior samples.
The analyzer prefers this dense evidence whenever present. No evaluation field
is published to the controller or agent. This remains discrete simulation evidence;
hardware calibration, transport timing, native stopping and real pickup remain
unqualified.

The final focused analyzer, physics-audit and MuJoCo actuator regression command
passed 31 tests (23 analyzer, four physics-audit and four actuator tests). The
parent policy-boundary suite passed 31 tests, including non-contiguous historical
observation IDs and explicit final-request cancellation at a completed budget.
These are separate suites; the counts are not an additional whole-project total.

## Historical position-plant evidence

The original migration's position-actuator controller cases and short navigation
agent runs remain archived in [`evidence/20261005`](../evidence/20261005/). They
predate shared MIT physics and are not evidence that this native actuator path or
a full pickup succeeds. `scripts/qualify_ik.py` keeps that historical scope in its
output label. Do not combine its position-plant results with the MIT cases above.
