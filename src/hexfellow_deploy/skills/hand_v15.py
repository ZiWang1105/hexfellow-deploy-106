"""Typed hand_v15 adaptation; no upstream world-frame controller is imported.

The source library and exact provenance live in third_party/hand_v15. These
plans adapt its ordered motion/closure/verification pattern to paired Hexfellow
TCP actions. All geometry is explicitly chosen by the policy, never detected
or loaded from the scene by this module.
"""
from copy import deepcopy
from dataclasses import dataclass
import math

import numpy as np

from .workflows import COUNTS, RESUME_SCHEMA, GEOMETRY_SCHEMA, validate_extensions, validate_workflow

LEGACY_NAMES = ('move_to', 'move_both', 'move_above', 'set_gripper', 'wait',
         'run_sequence', 'pick', 'place', 'bimanual_lift')
NAMES = LEGACY_NAMES + tuple(COUNTS)
POSE = {'type': 'object', 'properties': {
    'position_m': {'type': 'array', 'items': {'type': 'number'}, 'minItems': 3, 'maxItems': 3},
    'quaternion_xyzw': {'type': 'array', 'items': {'type': 'number'}, 'minItems': 4, 'maxItems': 4}},
    'required': ['position_m', 'quaternion_xyzw'], 'additionalProperties': False}
PAIR = {'type': 'object', 'properties': {'left': POSE, 'right': POSE},
        'required': ['left', 'right'], 'additionalProperties': False}
SKILL_SCHEMA = {'anyOf': [{'type': 'null'}, {'type': 'object', 'properties': {
    'name': {'type': 'string', 'enum': list(NAMES)},
    'arms': {'type': 'array', 'items': {'type': 'string', 'enum': ['left', 'right']},
             'minItems': 1, 'maxItems': 2},
    'waypoints': {'type': 'array', 'items': PAIR, 'maxItems': 8},
    'gripper_open': {'anyOf': [{'type': 'null'}, {'type': 'integer', 'enum': [0, 1]}]},
    'duration_s': {'type': 'number', 'minimum': 0, 'maximum': 10},
    'resume': RESUME_SCHEMA, 'geometry': GEOMETRY_SCHEMA},
    'required': ['name', 'arms', 'waypoints', 'gripper_open', 'duration_s', 'resume', 'geometry'],
    'additionalProperties': False}]}


def validate_skill(value):
    from ..interfaces.contract import ArmTarget
    from ..control.gripper import validate_gripper
    fields = {'name', 'arms', 'waypoints', 'gripper_open', 'duration_s'}
    if (not isinstance(value, dict) or not fields <= set(value)
            or set(value)-fields-{'resume','geometry'} or value['name'] not in NAMES):
        raise ValueError('Unknown skill or invalid skill fields')
    arms, points, duration = value['arms'], value['waypoints'], value['duration_s']
    if (not isinstance(arms, list) or not 1 <= len(arms) <= 2
            or any(s not in ('left', 'right') for s in arms) or len(set(arms)) != len(arms)):
        raise ValueError('Skill arms must be unique left/right names')
    if not isinstance(points, list) or len(points) > 8:
        raise ValueError('Skills accept at most eight paired waypoints')
    if isinstance(duration, bool) or not isinstance(duration, (int, float)) or not math.isfinite(duration) or not 0 <= duration <= 10:
        raise ValueError('Invalid skill duration')
    for pair in points:
        if not isinstance(pair, dict) or set(pair) != {'left', 'right'}:
            raise ValueError('Skill waypoints require both TCP poses')
        for pose in pair.values():
            if not isinstance(pose, dict) or set(pose) != {'position_m', 'quaternion_xyzw'}:
                raise ValueError('Invalid skill pose')
            ArmTarget.from_dict({**pose, 'gripper_open': 1})
    name = value['name']
    if validate_extensions(value):
        return deepcopy(value)
    if name in COUNTS:
        validate_workflow(value)
        return deepcopy(value)
    if value.get('geometry') is not None:
        raise ValueError('This skill does not accept geometric constraints')
    count = {'pick': 3, 'place': 2, 'wait': 0, 'set_gripper': 0}.get(name, 1)
    if (name == 'run_sequence' and not points) or (name != 'run_sequence' and len(points) != count):
        raise ValueError(f'{name} requires {count} waypoints (run_sequence: 1..8)')
    if name in ('move_both', 'bimanual_lift') and set(arms) != {'left', 'right'}:
        raise ValueError(f'{name} requires both arms')
    if name in ('move_to', 'move_above', 'pick', 'place') and len(arms) != 1:
        raise ValueError(f'{name} requires one explicitly selected arm')
    if name == 'set_gripper':
        validate_gripper(value['gripper_open'])
    elif value['gripper_open'] is not None:
        raise ValueError('Only set_gripper accepts gripper_open')
    if name in ('pick', 'place', 'set_gripper') and duration < 1:
        raise ValueError('Gripper commands need at least one second before the next step')
    # The trial lift is a bounded diagnostic, not transport or proof of retention.
    if name == 'pick':
        side = arms[0]
        approach, pinch, lift = [np.asarray(p[side]['position_m']) for p in points]
        if (np.linalg.norm((approach-pinch)[:2]) > 1e-6 or approach[2] <= pinch[2]
                or np.linalg.norm((lift-pinch)[:2]) > 1e-6 or not 0 < lift[2]-pinch[2] <= .030001):
            raise ValueError('Pick requires vertical approach/descent and a trial lift of at most 3 cm')
    # No movement of the unselected arm is authorized by a single-arm skill.
    if len(arms) == 1 and points:
        other = 'right' if arms[0] == 'left' else 'left'
        if any(p[other] != points[0][other] for p in points[1:]):
            raise ValueError('Unselected arm pose must be identical across skill waypoints')
    return deepcopy(value)


@dataclass(frozen=True)
class Step:
    kind: str
    waypoint: dict | None = None
    gripper: int | None = None
    duration_s: float = 0.
    arms: tuple[str, ...] | None = None
    checkpoint: str | None = None
    contact: bool = False
    pad_floor_m: float | None = None


def compile_skill(value):
    value = validate_skill(value)
    name, points, duration = value['name'], value['waypoints'], value['duration_s']
    if value.get('resume') is not None:
        raise ValueError('Resume requires a matching live checkpoint, not a new compiled plan')
    if name in COUNTS:
        from .workflows import compile_workflow
        return compile_workflow(value)
    if name == 'pick':
        return [Step('gripper', gripper=1, duration_s=duration), Step('move', points[0]),
                Step('move', points[1]), Step('gripper', gripper=0, duration_s=duration),
                Step('move', points[2])]
    if name == 'place':
        # Preserve an existing grasp. Never open before arrival at release pose.
        return [Step('move', p) for p in points] + [Step('gripper', gripper=1, duration_s=duration)]
    if name == 'set_gripper':
        return [Step('gripper', gripper=value['gripper_open'], duration_s=duration)]
    if name == 'wait':
        return [Step('wait', duration_s=duration)]
    return [Step('move', p) for p in points]


PROMPT = """
HAND_V15 ADAPTED SKILL MODE (takes precedence over direct-action sequencing):
The output also requires skill: null or {name, arms, waypoints, gripper_open, duration_s, resume, geometry}.
Use skill:null for the existing direct pose/navigation/head-pixel interface. Otherwise
navigation and grasp_target MUST be null. Outer left/right fields are compatibility
placeholders and are not executed; every skill motion uses waypoints instead.
These placeholders must still be complete objects with position_m (3 numbers),
quaternion_xyzw (4 numbers), and binary gripper_open. Never use {} or null.
Copy the measured TCP poses and previous gripper commands; without a prior command,
use measured open=1/closed=0, or 1 for 'between' as an ignored skill-only placeholder.
Waypoints contain left/right {position_m, quaternion_xyzw} in the current base frame,
meters and normalized xyzw, never world coordinates. Select all geometry from the
current RGB-D, calibrated wrist RGB and measured FK. No known table height, automatic
object detector, copied YAM wrist yaw, jaw-width grasp classifier or scene oracle exists.
Available calls:
- move_to / move_above: one waypoint and one arm. move_above's height is explicit in
  the waypoint, not an automatically computed offset from a known table.
- move_both / bimanual_lift: one paired waypoint, arms:[left,right]. Preserve grippers.
- run_sequence: 1..8 paired waypoints, selected arms; preserves gripper commands.
- set_gripper: no waypoints, selected arms, binary gripper_open, duration_s 1..10.
  Both arms brake into reference HOLD before gripper actuation. No TCP retargeting.
- wait: no waypoints, duration_s 0..10; reference HOLD, preserving grippers.
- pick: one arm, three paired waypoints: approach, pinch, vertical trial lift <=3 cm.
  Approach must be vertically above pinch. The runner opens in HOLD, reaches approach,
  descends open, reaches pinch, brakes both arms, closes, waits duration_s (1..10),
  then trial-lifts. It returns for fresh visual verification. Motion completion is
  NOT grasp success. Use only when the path and pinch placement are already clear;
  for uncertain final alignment prefer separate moves, reobserve, then set_gripper.
- place: one arm, two waypoints: transit and release. Retains existing grasp until
  measured arrival at release, then opens in HOLD and returns for visual verification.
For all calls except set_gripper, gripper_open must be null. For motion-only calls
use duration_s:0. Gripper duration is an observation delay, not contact confirmation.
Non-selected arm waypoint poses must repeat its CURRENT measured TCP pose; the runner
holds that arm's current reference pose. Each move waits for measured arrival; a stall
or stage timeout aborts the remaining steps into HOLD without opening or retrying.
Select short clear paths: no swept collision checking exists. Composite workflows
stop at enforced fresh-image checkpoints before transport or release.
get_state/capture_image are provided on every observation. verify_grasp remains your
visual judgment, with uncertainty explicit, not a motor or simulation truth label.
Extended workflows (all geometry selected by you, not a host detector):
- pick_and_place: 5 poses = approach, pinch, <=3 cm trial lift, transit, release.
- transfer_between_arms: arms:[giver,receiver]; 6 poses = giver transit, release,
  retreat; receiver approach, pinch, <=3 cm trial lift. TABLE-MEDIATED transfer.
  The giver releases on the table and retreats before receiver regrasp. Inspect
  the staged object and revise the plan if it moved; this is not an in-air handover.
- pick_tall_object: 4 poses = high clearance, pitched pregrasp, grasp, trial lift.
  Use explicit Hexfellow quaternions, not the original robot's fixed yaw/pitch map.
- pick_rim: 3 poses = rim approach, rim pinch, trial lift. Identify a real graspable
  rim from images; no automatic rim detector or filled-container assumption.
- extract_from_container: 5 poses = approach, pinch, trial lift, straight lift
  above rim, lateral clearance. Supply geometry; pinch has a measured pad-floor gate.
- place_into_bin: 4 poses = straight raised clearance, above opening, lower release,
  retreat. Supply geometry. Object bottom must clear the rim before lateral motion.
- pick_and_place_tall_object: the 4 tall-pick poses followed by the 4 bin-place poses.
- insert_into_container / insert_onto_peg: 3 poses = above opening/peg, vertical
  insertion, upward retreat. Supply geometry. Stall aborts; never force/search blindly.
- set_down: 2 poses = transit, supported release; inspect support before opening.
- pinch_cloth: arms:[left,right]; 3 pairs = approach, pinch, vertical trial lift.
- fold_cloth / repair_corner / align_corner: one arm; 6 pairs = approach, pinch,
  trial lift, raised fold/carry point, laid-down corner, retreat.
- bimanual_fold: the same 6 paired poses with both arms; choose the two corners.
- push / push_until_stop: 2 poses = approach and stroke endpoint. A stall is an
  obstruction needing observation, NOT verified object travel or end-stop success.
- sweep_area: 2..8 explicit sweep waypoints. Inspect the patch and choose spacing
  from this robot's fingers and observed debris; no original-robot blade dimensions.
- hold_and_push: arms:[holder,pusher]; 3 poses = pusher approach, stroke, retreat.
  Holder must already be commanded closed and visually secure the object.
- pull_toward_base: one nearer-base waypoint; retain current grippers.
Extended workflows use duration_s:1..10 for gripper dwell (pull_toward_base:0).
Initial calls use resume:null. geometry:null except the container/bin/peg skills,
which require {rim_z_m, object_bottom_in_tcp_m:[x,y,z]}; rim_z_m is also used for
peg-top height. These are YOUR estimates from permitted images/depth, not truth.
The bottom offset is rigidly expressed in the selected TCP frame after grasp.

CHECKPOINT PROTOCOL:
When harness_feedback.execution.skill.status is awaiting_verification, both arms
remain in reference HOLD and the active gripper commands persist. Inspect the NEW
head/wrist images against checkpoint/question. To proceed, return the SAME name and
ordered arms, waypoints:[], gripper_open:null, duration_s:0, geometry:null,
resume:{token:<exact returned token>,verdict:confirmed|uncertain|failed,evidence:<brief
visible evidence>}. Confirm only when the condition AND remaining geometry are still
supported. Uncertain/failed cancels the remainder without releasing the grasp. New
motion, navigation, cancellation or another skill invalidates the old continuation.
Use a new plan for visual alignment corrections; do not reuse pre-correction tokens.
Tokens are single-use and require a newer observation. Visual confirmation remains
a model assessment, never sensor-certified contact or physical success.
Source detector APIs, isolated reachability queries and robot-specific reset search
remain unavailable. State/capture/model/verification are supplied by this observation
loop, and explicit head-pixel grounding remains available with skill:null.
"""
