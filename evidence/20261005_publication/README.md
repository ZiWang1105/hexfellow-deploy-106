# Reference publication contention and operator diagnostics

The real run `real_20261005_204438` stopped during carrying startup: sequence 210
was generated but its DA reference-topic semaphore could not be acquired within
1 ms. Sequences 1–209 had been accepted. The shared BackendProxy converted this
known no-write result directly into a fatal error. Subsequent actuator liveness
expiration stopped feedback, and the camera then reported no encoder match. The
agent made zero API calls. Backend shutdown also reported unconfirmed lift stop;
this work does not establish a fix for that separate physical shutdown issue.

BackendProxy now retries the exact pending serialized reference within the
existing finite health deadline. It never re-solves PinkIK, changes sequence,
refreshes the command timestamp or ignores an unknown publication failure. Stop
requests, backend faults and a persistently wedged semaphore still terminate the
attempt. Control telemetry and records include retry counts and publication time.

A full shared MIT MuJoCo controller fixture with camera and dashboard completed
15 seconds with 100 deliberate 20 ms semaphore holds from a separate process:
257 recovered publication attempts across 98 commands, no control/backend/shutdown
fault, exit 0. 751 references were generated, 750 acknowledged by the control
owner, and final pending 751 was explicitly canceled at normal closure. No model
API or physical devices were used. This tests contention, not task success.

The initial injector trial also recovered publication contention but its external
Python 3.10 resource tracker incorrectly unlinked manager-owned shared memory,
causing teardown exit 1. Raw `runs/shm_retry_204438` remains preserved as that
failed harness run. The corrected external probe unregisters its attachments and
is preserved here; it does not change deployment source or hardware cleanup.

115 deployment tests pass, including real DA semaphore delay/wedge/stop cases,
no extra IK advancement, startup-before-API visibility, and durable decision
fallback. Browser tests use recorded simulator decisions and the failed physical
run as clearly labeled replay data, not a new policy execution. They verify
visible arm/gripper/base decisions above the fold, retained decisions during
inference, explicit zero-call startup failure and carrying-stage display.
