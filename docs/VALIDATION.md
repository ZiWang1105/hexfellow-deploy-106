# Initial migration evidence — 2026-10-05

All runs below used MuJoCo position actuators on the development workstation.
No physical device was opened. There is no native MIT, hardware stopping or
pickup qualification claim. No before/after hardware latency comparison was run.

| Run | Accepted / generated | Compute + submit p99 | Actual solve interval p99 | Outcome |
|---|---:|---:|---:|---|
| Paired target, camera active, 24 s | 1201 / 1201 | 3.74 ms | 20.10 ms | Closed normally, no budget overruns |
| Paired target, 30 ms acceptance delay and 30 ms compute delay every 10 solves, 26 s | 689 / 689 | 33.73 ms | 66.00 ms | 69 warnings; closed normally |
| HOLD, 30 ms compute delay every 10 solves, 8 s | 378 / 378 | 32.48 ms | 34.36 ms | 37 warnings; closed normally |
| Astra medium agent, camera active, 60 s | 2995 / 2995 | 3.10 ms | 20.09 ms | 7 completed model calls; closed normally |
| Astra medium agent, decision archive/provenance, 25 s | 1252 / 1252 | 3.11 ms | 20.09 ms | 3 completed calls; one request interrupted at diagnostic stop |
| Injected 200 ms acceptance delay | 0 / 1 | Not meaningful | Not meaningful | Expected acceptance-health fault; no second solve |

These are different workloads, not controlled throughput comparisons. The
delayed paired run had no camera; it also overlapped part of the separate agent
run. No real-time scheduling privilege was available. Worst-case latency is not
bounded by these finite samples.

Every recorded solve had `accepted_before_solve == sequence - 1`. The delayed
paired run ended with maximum measured TCP error 0.144 mm and 0.0791 degrees;
the baseline converged below 0.001 mm / 0.001 degrees in this idealized model.
The acceptance timeout test retained the unresolved pending reference and fault.
It did not declare that reference accepted or advance the solver.

The 60-second agent run returned seven navigation estimates, decreasing from
2.32 m to 0.707 m, and emitted 343 control samples with nonzero base velocity.
Those distances are model outputs, not privileged measurements. No pickup
completion was evaluated. Reported API cost was $1.050468. The later 25-second
run recorded $0.232375 for its three completed responses; the fourth request was
interrupted and its eventual provider cost is unknown. Production has no default
duration/call/spend cap; these durations were explicit test limits.

The actual saved API request used `openai/gpt-6-astra` / `medium`. Model input
contained unannotated RGB, registered depth, physical encoder/gripper feedback,
robot-only TCP/camera FK and static intrinsics. No root pose, scene objects,
contacts or evaluator fields were included. HTTP `/status`, `/rgb.jpg` and `/`
were fetched successfully during the baseline run; `/status` showed live agent
inference in the agent run. The component contract test additionally checks
decision persistence during inference and after stopping, without API access.

## PinkIK qualification

`scripts/qualify_ik.py` ran four 1,100-step cases: precision and real-profile
settings, each with ideal reference following and MuJoCo position actuation.
All four passed the 3 mm / 1.2 degree terminal error criteria. Both reference
TCP errors and measured TCP errors are archived separately. Holding submission
pending never advanced the actual core's committed reference array.

Both profiles rejected a 30 ms `dt`: their motion profile fixes the integration
step to 20 ms. This is a real integration constraint, not evidence that PinkIK
cannot solve the target. It means the new native adapter still needs a tested
missed-cycle MIT position/velocity/torque contract. Position-plant tests alone
cannot qualify that behavior.

Seven focused tests pass: remaining-budget sleep, nonterminal cycle overrun,
reference ownership while pending/rejected, malformed acceptance receipts,
cross-process DA shared-memory coherence, and policy observation/decision flow.
The initial smoke run exposed an incorrect `BASE_JOINTS` import and was corrected;
the initial policy archive assertion compared JSON lists to tuples and was also
corrected. Failed raw attempts remain in ignored `runs/`.

Compact machine-readable results and raw-file hashes are in
[`evidence/20261005`](../evidence/20261005/). Large raw logs, XML assets and API
requests remain in the corresponding local `runs/` directories. The initial
60-second run preceded the addition of code-provenance recording and persistent
decision JSONL; the later agent run verifies that addition. Current output always
records invocation, code hashes and dependency versions.
