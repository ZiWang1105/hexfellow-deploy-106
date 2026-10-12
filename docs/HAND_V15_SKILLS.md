# hand_v15 integration

This branch adds an **opt-in adaptation of 29 named skills** from hand_v15 3.11.0 to Hexfellow. The exact upstream Python source is retained as a non-executable reference under `third_party/hand_v15`, pinned to `facc274ff998d1545dc75f6ebe107e6da666dacd`. The source checkout is unchanged. This is not a claim that all upstream skills or benchmark successes work on this robot.

## Runtime ownership

The robot-relative skill adapter lives in `src/hexfellow_deploy/skills/` and
uses the same paired IK/MIT runtime in simulation and hardware. It requires no
agent application for execution. A model application may opt into its schema and
prompt, supply measured observations, and submit validated actions through the
optional plugin interface described in [STANDALONE.md](STANDALONE.md).
No physical skill success or upstream benchmark success is established by this
source extraction.

## Policy and execution

The optional policy schema adds `skill` to the existing paired target, navigation, grounding and public decision-summary fields. It is either null (existing direct behavior) or:

```json
{
  "name": "set_gripper",
  "arms": ["right"],
  "waypoints": [],
  "gripper_open": 0,
  "duration_s": 2,
  "resume": null,
  "geometry": null
}
```

A waypoint contains `left` and `right`, each with `position_m` and normalized `quaternion_xyzw`. Every coordinate is in the current observation's **robot base frame**. All geometry comes from the model's sensor interpretation, not the source library's world frame, table height or automatic detections. Outer action poses are compatibility placeholders and are ignored for skill execution. Skill calls cannot simultaneously request navigation or head-pixel grounding. Inspect a selected depth point in a prior call when needed.

The skill instructions and JSON schema are supplied on every model call. Requests archive the entire skill call; control diagnostics record the skill name, current stage and result. Allowlisted execution feedback returns to the next model call. There is no model-generated code execution.

`SkillExecutor` sequences operations **above** `PoseExecutor`; PinkIK, MIT commands, fixed-lift behavior, actuator receipts and feedback checks remain shared. Subsequent stages run only after measured arrival. Gripper steps wait for constrained braking to reach HOLD before changing a binary gripper command. Both arm references remain in HOLD during that step. Motion-only steps retain the current gripper commands, including transport-only HOLD when no command exists yet. Single-arm moves use reference FK for the other arm's TCP task; this is not a hard lock on its redundant joints.

A controller stall, external cancellation or 30-second stage deadline cancels the remaining plan into HOLD, preserving the current gripper command. There is no automatic reopen/retry or automatic base movement. The opt-in action horizon is 180 seconds. All waypoints are validated before a plan starts. No swept-path collision checking is added.

## Executable adaptations

| Skill | Hexfellow behavior |
|---|---|
| `move_to` | One selected arm and one paired waypoint; measured TCP arrival gate. |
| `move_above` | Same motion primitive with an explicitly supplied above-object pose; no known table plane. |
| `move_both` | One paired target, both arms active. |
| `bimanual_lift` | Explicit paired target with both gripper commands retained. No automatic grasp or retention claim. |
| `run_sequence` | 1–8 explicit paired motion waypoints; no arbitrary nested calls, gripper transitions or scripts. |
| `set_gripper` | Selected hands open/close after both arms reach reference HOLD; wait 1–10 seconds, then reobserve. |
| `wait` | Reference HOLD for 0–10 seconds, preserving grippers. |
| `pick` | Open in HOLD; reach approach; descend open to pinch; close in HOLD; wait 1–10 seconds; trial-lift vertically by at most 3 cm; return for fresh visual verification. |
| `place` | Keep an existing close command through transit and release poses; open only after measured release-pose arrival and HOLD; return for visual verification. |

Pick requires three explicit waypoints, with vertical approach/descent and trial lift. It refuses to silently reopen a hand already commanded closed. Place requires an existing close command. Neither precondition proves object retention. For uncertain pinch alignment, use separate move/reobserve/set-gripper calls instead of a multi-stage pick.

Every completed sequence is `completed_unverified`, with `effect_verified:false`. A motor close command, elapsed dwell, measured TCP rise or completed motion does **not** demonstrate object contact/retention. The source library's calibrated jaw-width classifier and proprioception-only grasp-success fallback are intentionally not reproduced. Our input currently lacks that verified jaw-width classification. Use fresh head/wrist imagery to decide whether to transport or recover; offline scoring stays separate.

## Extended workflows and visual checkpoints

Twenty additional names adapt source workflow structure to this robot. They accept
policy-selected paths, not source object names, automatic segmentation, YAM grasp
angles, known table heights, or calibrated source jaw widths. This is an adaptation
of behaviors, **not API compatibility or demonstrated task competence**.

| Workflow | Ordered paired waypoints / behavior |
|---|---|
| `pick_and_place` | Approach, pinch, trial lift, transit, release. Verify retention before transit and support before opening. |
| `transfer_between_arms` | `arms:[giver,receiver]`; giver transit, release, retreat; receiver approach, pinch, trial lift. Table-mediated transfer, matching the source strategy. |
| `pull_toward_base` | One explicitly chosen target with smaller horizontal base radius; retain grippers. |
| `pick_tall_object` | High clearance, pitched pregrasp, grasp, trial lift; explicit Hexfellow orientations. |
| `pick_rim` | Rim approach, pinch, trial lift; policy identifies the rim. |
| `extract_from_container` | Approach, pinch, trial lift, straight lift above rim, lateral clearance. Check retention and that the container stayed. |
| `place_into_bin` | Raised clearance, above opening, lower release, retreat. Check opening alignment and support. |
| `pick_and_place_tall_object` | Four tall-pick poses followed by four bin-placement poses. |
| `insert_into_container`, `insert_onto_peg` | Above opening/peg, vertical descent, upward retreat. Check alignment and support; no blind search. |
| `set_down` | Transit and supported release; check before and after opening. |
| `pinch_cloth` | Both arms: approach, pinch, trial lift; visually assess secured corners. |
| `fold_cloth`, `repair_corner`, `align_corner` | One arm: approach, pinch, trial lift, raised carry/fold point, laid-down corner, retreat. |
| `bimanual_fold` | Same six paired waypoints, both arms. |
| `push`, `push_until_stop` | Approach and contact-stroke endpoint. Check empty pushing hand/path first and displacement afterward. |
| `sweep_area` | 2–8 explicitly selected sweep poses. No automatic coverage planner. |
| `hold_and_push` | `arms:[holder,pusher]`; pusher approach, stroke, retreat. Holder remains at its current reference. |

All acquisition trial lifts retain pinch orientation and rise vertically by at most
3 cm. Ordinary approach/pinch pairs are vertically aligned. Tall-object pregrasps
may use explicit pitch changes before closure. Intermediate motion uses the existing
PinkIK limits; there is no new Cartesian microstep planner or swept-path checker.
Cloth skills do not include an autonomous corner detector or cloth physics estimator.

Composite workflows enter `awaiting_verification` at designated boundaries. Both
arms remain in reference HOLD and gripper commands persist. The deployment loop
waits for head and all configured wrist observations newer than that checkpoint,
then calls the policy with the checkpoint question and a single-use token. Example
continuation inside the ordinary action's `skill` field:

```json
{
  "name": "pick_and_place",
  "arms": ["right"],
  "waypoints": [],
  "gripper_open": null,
  "duration_s": 0,
  "geometry": null,
  "resume": {
    "token": "<exact token from harness_feedback.execution.skill>",
    "verdict": "confirmed",
    "evidence": "The target follows the hand; the planned destination remains clear."
  }
}
```

The verdict can also be `uncertain` or `failed`; either cancels the remainder into
HOLD without opening. A new action or ordinary HOLD cancels the previous token.
Only the internal reobservation HOLD preserves it. No geometry may change inside a
continuation: if visual alignment needs correction, submit a new plan. Even a
confirmed assessment remains a model interpretation; `effect_verified` stays false.
The policy's public evidence is archived, not hidden reasoning.

Container/bin/peg workflows require `geometry` with `rim_z_m` (peg-top height for a
peg) and `object_bottom_in_tcp_m:[x,y,z]`. These are policy estimates from allowed
sensors. The offset is expressed in the grasping TCP frame, not base coordinates.
Planned object-bottom height must clear the rim/peg by 1 cm before lateral transport
or insertion. Extraction closure additionally checks the approximate configured
GP80 pad envelope against `rim_z_m + 0.002`, both at the requested pinch pose and
again using measured FK immediately before closure. The pad check covers jaw travel,
not the entire hand or arm, and is not a calibrated physical clearance guarantee.

A stalled contact stroke becomes `blocked_unverified` and cancels remaining steps;
it never triggers automatic release, force escalation, search, or a success claim.

## Source features not enabled

The manifest inventories all 38 upstream entries. The remaining nine are:

- `get_state`, `capture_image`, `ask_runtime_model`, `verify_grasp`: existing observation/model loop supplies the allowed sensors and visual judgment; no separately callable source API or automatic verifier is added.
- `pixel_to_world`: only the existing explicit model-selected **head pixel** grounding is available, returning base-frame provenance. No world localization, automatic batch sampling or wrist depth is added.
- `solve_ik`, `reset_arms`: no isolated source solver or YAM reset pose is exposed. Shared PinkIK and explicit Hexfellow poses remain authoritative.
- `segment_objects`, `locate_objects`: source SAM/color detectors are not enabled. They conflict with the current raw-sensor policy boundary.

Thus 29 of the 38 source names are callable adaptations. Five observation/model/
grounding names are supplied by the existing loop rather than callable skills;
four source-only solver/reset/detector names remain unavailable.

The new controller does not claim collision-free execution, physical grasp success, calibrated contact force or simulator/native contact equivalence. Existing feedback-based gripper control remains responsible for native closure; its motor effort target is not a jaw-force guarantee.

## Validation evidence

The first nine-skill integration established the following baseline:

- 451 relevant regression tests passed; one known base-pulse test was excluded after
  reproducing its 1.03 s versus 1.0 s assertion failure with the original `cce0f30`
  executor. It is not reported as passing.
- A shared PinkIK / stream-guard / MuJoCo MIT smoke run accepted 385 frames, exercised
  `set_gripper`, `move_to`, `wait`, `pick` and `place`, and kept the lift reference at
  0.3 m. All five calls returned `completed_unverified`. Grippers were empty.
- No hardware endpoint or remote model API was opened for these checks. This is
  execution-path validation, not visual task completion or physical grasp validation.

See [validation.json](../experiments/hand_v15_integration_20261009/validation.json),
[the reproducible smoke script](../experiments/hand_v15_integration_20261009/check_sim.py),
and [the baseline reproduction](../experiments/hand_v15_integration_20261009/check_baseline.py).

Expanded checks: **487 passed** across the regression run and a headless EGL
render rerun; the same pre-existing base-pulse assertion remains excluded. The
MuJoCo workflow smoke accepted **996 MIT frames** and kept the lift at **0.3 m**.
Evidence is recorded separately in
[workflow validation](../experiments/hand_v15_workflows_20261009/validation.json).
The shared MuJoCo smoke script exercises composite pick/place, table-mediated
transfer and set-down with empty grippers and scripted checkpoint responses.
It validates control sequencing and token handling, **not image interpretation,
object retention, cloth manipulation, insertion success or physical safety**.
No hardware or remote model API is opened by these checks.


## Gemini 3.8 Flash response compatibility

Select Gemini without changing the skill configuration:

```bash
--agent configs/agent/hand_v15.yaml \
--set policy.model=google/gemini-3.8-flash \
--set policy.reasoning_effort=low
```

The client automatically uses OpenRouter JSON-object mode for this model and
includes the complete output schema in the system message. A local JSON Schema
validator enforces required fields, types, enums, and additional-property rules
before the existing motion and skill validators run. Other models retain the
existing JSON-schema response mode. `structured_output_mode` in the agent YAML
can explicitly select `auto`, `json_schema`, or `json_object`.

October 9 replay evidence showed repeated empty left/right objects in the schema
response mode, including with skills disabled and medium reasoning. Explicit
prompt corrections alone did not resolve it. JSON-object mode produced a valid
non-actuating inspection followed by a valid skill response on archived images.
These outputs were never sent to hardware. This verifies response compatibility,
not perception quality, grasp success, or physical execution.

Turn records now preserve public response content, finish reason, and the precise
validation error. Rejected output never becomes an action. Subsequent attempts
receive the recorded rejection through `harness_feedback.validation`; three
consecutive failures still pause requests in HOLD. Hidden reasoning and
credentials are not recorded. Evidence: `experiments/gemini_schema_20261009/`.
