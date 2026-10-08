# GP80 feedback control

`hexfellow-deploy` owns the gripper control implementation:

- `src/hexfellow_deploy/gripper_effort.py`: measured-feedback control law,
  state machine and shared `closing_effort_settings` profile. Standard library only.
- `src/hexfellow_deploy/gripper_sdk.py`: exact SDK POSITION encoding, per-hand
  publisher, motor-angle admission and finite command leases. SDK 1.4.10 is
  available through the `hardware` installation extra.
- `src/hexfellow_deploy/gripper_contract.py`: binary request validation.

The parent workspace supplies device connections, timestamped feedback taps,
configuration and policy/keyboard requests. Both native deployment and keyboard
control use `GP80FeedbackPublisher`; there is no separate production closing
algorithm in the parent. Its historical import paths are compatibility aliases.
Bench workflows remain in the parent and use the same law and closing profile.

## Behavior

Each hand has an independent stateful controller in its existing 250-Hz SDK
publication loop. The 50-Hz command owner renews its lease. Repeating a close
request preserves the current regulation phase rather than restarting approach.
Policy requests remain `0 = close`, `1 = open`; transport-only `None` brakes.

Unobstructed closing advances the previous reference at up to 5 motor rad/s,
with steps no larger than 0.02 rad. Obstruction requires a 60-ms window of
stationary, lagging measured feedback, plus load and low speed. Once detected,
the controller latches effort regulation. Its reference velocity is
`direction * 0.5 * (0.1 - filtered_closing_effort_nm)`, with a +/-0.005 Nm
deadband, +/-0.05 rad/s speed limit and +/-0.005 rad reference-error envelope.
The filter is a three-sample median. Excess approach error reconciles toward
the regulation envelope at the SDK slew limit. High effort opens; low effort
closes. There is no automatic return to fast approach or fixed settling timeout.

Opening preserves the normal SDK position interpolation and its separately
configured opening torque threshold. Production regulation continues until an
open/stop request or a fault. The bench's ten-second release criterion and
45-second overall trial limit do not apply to production. The keyboard console
still runs one selected action at a time: another keyboard action preempts the
current one and releases its session. Native policy deployment regulates both
hands alongside arm motion.

Commands retain a lease of at most 100 ms, measured feedback must be no older
than 50 ms, and health/position/speed checks remain active. Faults latch and
produce BRAKE; renewing a lease cannot clear one. Native wire filtering admits
the generated hand packets only while their command and source feedback remain
fresh. Stopping or changing request invalidates old packet authorizations.

The operator-selected **0.1 Nm is a motor-effort regulation target**, not a peak
torque cap or calibrated jaw force. The default optional absolute software trip
is disabled. The normal-opening SDK threshold of 1 Nm is not a hard force cap.
The local desktop configuration enables closure for both hands after the user
confirmed replacement/repair. The unconfigured template remains disabled.

## Diagnostics and validation

Each publisher retains the last 4,096 samples (about 16 seconds at 250 Hz), with
raw measured angle/speed/effort, firmware target when reported, command, issued
reference, phase, tracking window and regulation diagnostics. Native diagnostic
snapshots include `gripper_controllers`; keyboard `run.json` includes the latest
per-hand controller report. Snapshot capture does not wait on a busy publisher
lock. Older samples are evicted; these are bounded diagnostic traces, not a
complete recording of every motor cycle in a long run.

Controller and recorded-feedback regression tests live in this repository:
`tests/test_gripper_effort.py`, `tests/test_gripper_tracking_window.py` and their
fixtures. Parent tests exercise real SDK packet construction, both wire IDs,
keyboard ownership/cancellation, native group dispatch and wire filtering.
The relocated controller tests pass from this repository (98 tests). Combined
targeted controller/bench/keyboard/native coverage comprises 372 passing tests;
file-only local deployment/keyboard checks also pass. A separate isolated Python
check runs the controller without the parent package or third-party libraries.

These changes involved no device connection or robot motion. The user accepted
the left-hand regulation observed in `left_gripper_close_20261007_210012`; that
bench run still ended at its 45-second budget. The repaired right hand has not
been physically tested with this integration. The simulator's jaw actuator model
is not a calibrated GP80 motor-effort model, so these tests do not establish
hardware force limits or full simulated/physical gripper-dynamics parity.
