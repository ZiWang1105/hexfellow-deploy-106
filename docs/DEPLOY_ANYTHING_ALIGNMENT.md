# Deploy Anything alignment audit, 2026-10-05

The earlier claim of complete alignment was too broad. Reusing DAManager and
DA shared memory did not establish matching serialization, lock handling,
feedback freshness or recovery semantics. Likewise, sharing a simulator/native
controller does not make a passing simulator run evidence of native reliability.

The reference inspected here is `deploy-anything` commit
`6c2bda40ebee56ab3a50b8dd62aaf4df5814e5a8`, both the sibling checkout and the
pinned dependency. Neither checkout was modified. The comparison is with its
`MinkEEFChunkExecutor` and `YAMEEFHardwareInterface`, not every historical driver.

| Area | Reference behavior | Hexfellow implementation and difference |
| --- | --- | --- |
| Lifecycle/configuration | DA manager, interface workers, Hydra topic construction | Reused. Control, actuator, camera, policy and dashboard are independent processes. |
| Control feedback and commands | Fixed NumPy joint state/command wrappers on DA shared memory | Replaced custom JSON frames with versioned float64 layouts on the same `ArrayCommWrapper`. Extra fields carry Hexfellow lift, binary grippers, original source clocks and complete MIT vectors. |
| Acceptance | Separate numeric synchronization/status messages | Added a 56-byte numeric receipt topic. SDK acceptance, UDP send and simulator application remain explicitly different scopes. |
| Diagnostic status | Separate from motor command arrays | Backend JSON status reduced from every 4 ms iteration to 10 Hz, plus startup/fault updates. Control JSON status already publishes in a background worker at 20 Hz. Policy requests remain JSON; this is not a claim that all IPC is numeric. |
| Lock waiting | DA wrapper defaults to blocking; some executor calls use 10 ms. Its SHM documentation recommends finite waits in real-time loops because dead owners can strand the semaphore. | Numeric topics use a finite 1 ms acquisition. A miss writes nothing. Feedback misses may retain the previous snapshot, but freshness is checked and refreshed before entering stateful IK. |
| Reference contention | The executor has explicit handoff/acknowledgement waits | Retry the exact pending frame within its original command deadline. No new solve, fabricated acknowledgement or refreshed creation time. A wedged lock still fails. |
| Timing | EEF executor warns on compute overruns and sleeps only the remaining period | Same remaining-period calculation in control/backend. No additional full-period sleep after work; ordinary budget overruns do not end the session. The reference YAM hardware loop itself uses a fixed 2 ms sleep, so not every reference loop has identical scheduling. |
| Feedback freshness | Robot-specific state/health policies | Retain PinkIK's original 40 ms age / 20 ms skew checks. Refresh before solving with a compute margin; keep each device's original receipt time. These limits are not copied from YAM. |
| Reference history | Mink failure handling resets motion history and rejects a chunk | PinkIK commits inside `step()` and has no rollback. One outstanding reference is retained until acknowledged. A rejected/failed solve does not automatically reset history or resume motion. This recovery behavior is **not aligned** with YAM and requires a separate explicitly designed recovery state transition. |
| Low-level control | YAM SDK and its own actuation contract | Native Hexfellow MIT, lift/base/GP80 protocol and firmware watchdog remain. They cannot be replaced by YAM joint-position writes. The old gateway supervisor is not instantiated; the shared actuator process still owns the SDK. |
| Fault visibility | Chunk failure can retain the worker | Invalid policy requests and short contention are recoverable; hard device, stale-stream and stateful IK faults stop the control session. Control/backend retain diagnostic workers. Native camera availability errors can still fail that component; automatic camera/session recovery is not implemented. |
| Simulation boundary | Application-specific plant adapters | Both Hexfellow backends use the same numeric messages, reference owner, PinkIK, admission and clocks. MuJoCo does not reproduce SDK threads, asynchronous device clocks, physical network, firmware watchdog or real stopping. |

## What changed in this revision

`control_topics.py` defines the fixed layouts. Feedback payload is 320 bytes,
reference payload 1,168 bytes, measured camera-proprioception payload 272 bytes,
and receipt payload 56 bytes. Previous JSON topics copied 65,536-byte payloads.
All four use DA's existing named semaphore, coherent array copy and lifecycle;
no new shared-memory implementation or unbounded command queue was introduced.

Original sensor/command times are losslessly represented as float64. Physical
joint names are encoded in a fixed order; MIT order, binary grippers, exact
sensor fields, schema version and finite values are checked. Root/object truth
does not enter these schemas. Native and simulation select the same YAML topics.

The earlier feedback-refresh and exact-reference retry corrections are retained
and now tested against the actual numeric schemas. Startup still reads the
separate metadata/seed message; steady actuator acknowledgements do not depend
on diagnostic JSON publication. Tests explicitly cover that distinction.

## Evidence and limits

140 deployment tests pass. The public `run_agent.py` completed a 10-second
MuJoCo HOLD with 502 generated/accepted frames. Three separate 15-second shared
MuJoCo paired-pose runs with camera/dashboard passed actual semaphore contention:

| Topic held by another process | Hold duration | Injections | Recovered feedback/reference retries | Generated / accepted |
| --- | ---: | ---: | ---: | ---: |
| Feedback | 45 ms | 54 | 500 / 0 | 725 / 725 |
| Reference | 20 ms | 54 | 0 / 283 | 728 / 727 |
| Receipt | 20 ms | 54 | 0 / 0 | 724 / 724 |

The reference run canceled its final pending sequence at the requested test
shutdown; it did not fabricate acceptance. All three exited zero with no
control/backend/shutdown fault. Receipt contention waits on acceptance without
generating another IK reference; the lower generated count reflects waiting.
Unit tests also verify that a permanently held lock expires without a partial
publication or another solve.

An uncontended, same-process 3,000-sample put/get microbenchmark measured median
feedback 21.05 to 15.30 microseconds, reference 43.26 to 28.15 microseconds, and
receipt 9.26 to 9.23 microseconds. These are small local savings, **not evidence
that JSON caused the observed 50 ms stale-feedback event**. Contention recovery
and precise source clocks matter separately from serialization speed.

No hardware was opened and no model API was called in this revision's validation.
These fixtures do not constitute another agent pickup test. Full raw runs are
in the parent workspace; compact summaries, hashes, benchmark and reproduction
commands are in [numeric control evidence](../evidence/20261005_numeric_control/README.md).

Unresolved physical issues remain the original SDK-to-controller freshness gap,
the earlier firmware communication timeout, and lift stop confirmation. Existing
snapshot/publication/read and native wire instrumentation is needed to distinguish
them in a physical run. It is not valid to label this revision hardware-qualified
or claim that Deploy Anything cannot time out.
