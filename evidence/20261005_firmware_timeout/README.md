# Firmware communication timeout at 20:58:21 UTC

Source: parent `runs/real_20261005_205758` and the operator's terminal log.
The run failed during carrying startup, before any policy API call.

## Established evidence

- Sequence 1038 started at 257161.482560361. Its first SDK acceptance was
  257161.543904784; the control record reports 58.030579 ms from submission to
  acceptance. Sequence 1039 started at 257161.548479972: a 65.919611 ms start gap.
- Both arm firmware logs report `PscAPICommunicationTimeout`. SDK 1.4.10
  `Arm._periodic` responds by attempting clear-parking-stop and reinitialization.
  The wire gate rejects automatic recovery. The backend's generic gate error
  previously hid the affected device and rejected packet.
- Camera capture was still running without capture errors, after 300 registered
  publications. Encoder updates ended at 257161.548355199. Later unmatched frames
  caused the camera publication deadline failure. This is a downstream symptom.
- The new runtime did not enter the previous process-local Python scheduling
  context: default 5 ms GIL switching and automatic cyclic GC replaced the prior
  0.2 ms / GC-disabled control settings. This is a migration gap, not proof that
  GC or the GIL caused this specific incident.

SDK acceptance is not firmware receipt. SDK periodic retransmissions run
independently of 50 Hz IK. The saved run does not contain UDP/send history, so the
58 ms reference delay cannot be relabeled as a measured 58 ms wire outage.
Host scheduling, sender-lock contention and network delivery are not resolved by
these logs. Existing finite stream checks and no-auto-rearm behavior remain.

## Changes and validation

- Configure process-local GIL/GC through shared DA YAML for control/backend in
  both simulation and hardware. Collect before device construction; restore on
  process exit. Record effective settings, affinity and actual scheduler in
  `component_health.json`. This does not grant real-time priority or isolate CPUs.
- Freeze bounded per-device gate, rejected-packet, first-SDK-dispatch, UDP send,
  UDP receive, KCP processing and latest firmware health evidence before close.
  Skip busy sender locks instead of waiting. Write `native_transport.json` only
  after stop/close; keep it out of policy observations and periodic SHM status.
  Diagnostic failures cannot replace the original control or shutdown fault.
- 124 tests pass, including process-local settings restoration, blocked-lock
  diagnostic capture, numeric telemetry serialization and stop-before-file-write.
- `runs/da_scheduling_205821`: 10-second shared MuJoCo HOLD with camera/dashboard,
  504 generated/accepted cycles, no control/backend fault, zero API calls. Both
  workers report 0.2 ms GIL switching and cyclic GC disabled.

No physical motion, stop clearing or hardware task rerun was performed for these
changes. The firmware timeout cause and lift stop confirmation remain unresolved.
MuJoCo does not implement the vendor firmware watchdog or real network.
