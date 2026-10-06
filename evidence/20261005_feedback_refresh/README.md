# Feedback refresh and fault localization

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

