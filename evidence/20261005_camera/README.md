# Physical camera synchronization — 2026-10-05

Read-only sensor probes; no acquisition, MIT commands or policy calls. Original
metadata and probe scripts remain in the parent `runs/camera_publication_203311/`.
The JSON summaries and hashes here preserve the scope and source identities.

The original latest-only matcher rejected 180 fresh image frames in 15 seconds:
median image/encoder skew 94.74 ms exceeded the unchanged 80 ms allowance. Every
rejected frame had a recorded encoder-history match within 15.38 ms. Images were
at most 111.73 ms old, below the unchanged 250 ms freshness limit. The old
publication watchdog would fail after 2.67 seconds. This reproduces the failure
class; the user's original run lacks per-image matching records.

The revised actual `NativeCamera.run` then completed its 20-second timer and
clean shutdown (20.626 seconds total), with 260 publications, no matching failures
and no camera error. Last 256 frame ages: median 58.97 ms, maximum 74.81 ms;
encoder skew ranged from -4.77 to +5.27 ms. Longest publication interval was
116.09 ms. Feedback used real DA shared memory and a concurrent image consumer.
All endpoints remained unowned, command writes were zero, and close had no errors.
This does not validate a full physical pickup or simultaneous IK/policy load.
