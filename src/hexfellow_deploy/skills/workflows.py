"""Sensor-guided hand_v15 workflows adapted to Hexfellow.

Original algorithms are indexed in third_party/hand_v15/PROVENANCE.json.
Robot-specific search grids, table constants, aperture classifiers and source
perception calls are replaced by explicit policy geometry and resumable visual
checkpoints. The source transfer is table-mediated, not an aerial handover.
"""
import math

import numpy as np
from scipy.spatial.transform import Rotation

# Number/order of paired poses is part of each skill's public contract.
COUNTS = {
    'pick_and_place': 5, 'transfer_between_arms': 6, 'pull_toward_base': 1,
    'pinch_cloth': 3, 'fold_cloth': 6, 'repair_corner': 6,
    'bimanual_fold': 6, 'align_corner': 6, 'hold_and_push': 3,
    'extract_from_container': 5, 'insert_into_container': 3, 'sweep_area': (2, 8),
    'push_until_stop': 2, 'push': 2, 'pick_tall_object': 4,
    'place_into_bin': 4, 'pick_and_place_tall_object': 8,
    'pick_rim': 3, 'set_down': 2, 'insert_onto_peg': 3,
}
ACQUIRE = {'pick', 'pick_and_place', 'pinch_cloth', 'fold_cloth', 'repair_corner',
           'bimanual_fold', 'align_corner', 'extract_from_container',
           'pick_tall_object', 'pick_and_place_tall_object', 'pick_rim'}
CARRY = {'place', 'place_into_bin', 'insert_into_container', 'set_down', 'insert_onto_peg'}
BOTH = {'transfer_between_arms', 'pinch_cloth', 'bimanual_fold', 'hold_and_push'}
GEOMETRIC = {'extract_from_container', 'insert_into_container', 'place_into_bin',
             'pick_and_place_tall_object', 'insert_onto_peg'}

CHECKPOINTS = {
    'held_target_only': 'Confirm the intended object alone follows the hand; exclude a mixed grasp or dropped object.',
    'supported_for_release': 'Confirm stable support at the intended destination before opening; do not drop from a hover.',
    'placement_visible': 'Inspect the released object and destination; report uncertainty or displacement.',
    'staged_object_visible': 'Confirm the object rests in the shared staging area, the giver is clear, and receiver pinch poses still align.',
    'container_stayed': 'Confirm only the target lifted, the container stayed in place, and the held object clears the rim.',
    'opening_aligned': 'Confirm the opening or peg is aligned with the held object and the planned descent remains clear.',
    'cloth_secured': 'Confirm the selected cloth corner(s) lifted with the finger pads before transport.',
    'cloth_laid_down': 'Confirm the moved cloth edge/corner is laid at the intended fold or repair destination before release.',
    'cloth_result': 'Inspect fold/corner placement; execution alone does not establish alignment or fold quality.',
    'empty_push_path': 'Confirm the pushing hand holds no object and the selected contact path is appropriate and clear of unintended obstacles.',
    'push_result': 'Inspect object displacement. TCP motion or a stall does not prove the object reached an end-stop.',
    'holder_secured': 'Confirm the holding hand already secures the intended object and the other hand can push without striking it.',
}

RESUME_SCHEMA = {'anyOf': [{'type': 'null'}, {'type': 'object', 'properties': {
    'token': {'type': 'string', 'minLength': 1, 'maxLength': 64},
    'verdict': {'type': 'string', 'enum': ['confirmed', 'uncertain', 'failed']},
    'evidence': {'type': 'string', 'minLength': 1, 'maxLength': 600}},
    'required': ['token', 'verdict', 'evidence'], 'additionalProperties': False}]}
GEOMETRY_SCHEMA = {'anyOf': [{'type': 'null'}, {'type': 'object', 'properties': {
    'rim_z_m': {'type': 'number', 'minimum': .3, 'maximum': 1.8},
    'object_bottom_in_tcp_m': {'type': 'array', 'items': {'type': 'number'}, 'minItems': 3, 'maxItems': 3}},
    'required': ['rim_z_m', 'object_bottom_in_tcp_m'], 'additionalProperties': False}]}


def validate_extensions(value):
    resume, geometry = value.get('resume'), value.get('geometry')
    if resume is not None:
        if (not isinstance(resume, dict) or set(resume) != {'token','verdict','evidence'}
                or not isinstance(resume['token'], str) or not 1 <= len(resume['token']) <= 64
                or resume['verdict'] not in ('confirmed','uncertain','failed')
                or not isinstance(resume['evidence'], str) or not resume['evidence'].strip()
                or len(resume['evidence']) > 600):
            raise ValueError('Invalid visual-checkpoint response')
        if value['waypoints'] or geometry is not None or value['gripper_open'] is not None or value['duration_s'] != 0:
            raise ValueError('Resume carries only the existing token and visual verdict; it cannot replace a plan')
    if geometry is not None:
        if not isinstance(geometry, dict) or set(geometry) != {'rim_z_m','object_bottom_in_tcp_m'}:
            raise ValueError('Invalid policy-estimated container/peg geometry')
        rim, offset = geometry['rim_z_m'], geometry['object_bottom_in_tcp_m']
        if (isinstance(rim, bool) or not isinstance(rim, (int,float)) or not math.isfinite(rim)
                or not .3 <= rim <= 1.8 or not isinstance(offset, list) or len(offset) != 3
                or any(isinstance(x, bool) or not isinstance(x, (int,float)) or not math.isfinite(x)
                       or abs(x) > .5 for x in offset)):
            raise ValueError('Geometry requires bounded finite meters, not scene identifiers')
    return resume is not None


def xyz(pair, side):
    return np.asarray(pair[side]['position_m'], float)


def bottom_z(pair, side, geometry):
    pose = pair[side]
    return float(xyz(pair, side)[2] + (Rotation.from_quat(pose['quaternion_xyzw']).as_matrix()
                                     @ np.asarray(geometry['object_bottom_in_tcp_m']))[2])


def validate_workflow(value):
    name, arms, points = value['name'], value['arms'], value['waypoints']
    count = COUNTS[name]
    low, high = count if isinstance(count, tuple) else (count,count)
    if not low <= len(points) <= high:
        raise ValueError(f'{name} requires {low}..{high} ordered waypoints')
    if name in BOTH and len(arms) != 2:
        raise ValueError(f'{name} requires two ordered arms')
    if name not in BOTH and name != 'sweep_area' and len(arms) != 1:
        raise ValueError(f'{name} requires one selected arm')
    if name != 'pull_toward_base' and value['duration_s'] < 1:
        raise ValueError('Workflow gripper dwell must be at least one second')
    if value['gripper_open'] is not None:
        raise ValueError('Workflow gripper transitions are defined by the skill')
    if name in GEOMETRIC and value.get('geometry') is None:
        raise ValueError(f'{name} requires policy-estimated rim/peg height and object bottom offset')
    if name not in GEOMETRIC and value.get('geometry') is not None:
        raise ValueError('Geometry is only accepted by container/bin/peg workflows')
    side = arms[0]
    # A trial lift must be straight up; transport is gated on a fresh visual check.
    lift_indices = {'pick_and_place':(1,2), 'pick_rim':(1,2), 'pinch_cloth':(1,2),
                    'fold_cloth':(1,2), 'repair_corner':(1,2), 'bimanual_fold':(1,2),
                    'align_corner':(1,2), 'extract_from_container':(1,2),
                    'pick_tall_object':(2,3), 'pick_and_place_tall_object':(2,3),
                    'transfer_between_arms':(4,5)}
    if name in lift_indices:
        a,b = lift_indices[name]
        selected = arms if name in ('pinch_cloth','bimanual_fold') else [arms[-1] if name == 'transfer_between_arms' else side]
        for s in selected:
            delta = xyz(points[b],s)-xyz(points[a],s)
            if np.linalg.norm(delta[:2]) > 1e-6 or not 0 < delta[2] <= .030001:
                raise ValueError('Trial lift must be vertical and at most 3 cm')
        # Preserve orientation during the diagnostic lift; a rotating gripper
        # can sweep into a rim even when its TCP moves straight upward.
        for s in selected:
            qa = Rotation.from_quat(points[a][s]['quaternion_xyzw'])
            qb = Rotation.from_quat(points[b][s]['quaternion_xyzw'])
            if (qa.inv()*qb).magnitude() > 1e-6:
                raise ValueError('Trial lift must retain the pinch orientation')
    if name in ('pick_and_place', 'pick_rim', 'pinch_cloth', 'fold_cloth',
                'repair_corner', 'bimanual_fold', 'align_corner', 'extract_from_container',
                'transfer_between_arms'):
        a, b = (3,4) if name == 'transfer_between_arms' else (0,1)
        selected = arms if name in ('pinch_cloth','bimanual_fold') else [arms[-1] if name == 'transfer_between_arms' else side]
        for s in selected:
            delta = xyz(points[a],s)-xyz(points[b],s)
            if np.linalg.norm(delta[:2]) > 1e-6 or delta[2] <= 0:
                raise ValueError('Approach must be vertically above the pinch')
    if name == 'extract_from_container':
        if np.linalg.norm((xyz(points[3],side)-xyz(points[2],side))[:2]) > 1e-6:
            raise ValueError('Extraction must rise vertically before lateral clearance')
        if min(bottom_z(points[i],side,value['geometry']) for i in (3,4)) < value['geometry']['rim_z_m']+.01:
            raise ValueError('Estimated object bottom must clear the rim by 1 cm before lateral extraction')
    if name in ('place_into_bin','pick_and_place_tall_object'):
        p = points if name == 'place_into_bin' else points[4:]
        if (np.linalg.norm((xyz(p[1],side)-xyz(p[0],side))[:2]) < 1e-6
                or xyz(p[2],side)[2] >= xyz(p[1],side)[2]
                or np.linalg.norm((xyz(p[2],side)-xyz(p[1],side))[:2]) > 1e-6
                or xyz(p[3],side)[2] < xyz(p[1],side)[2]):
            raise ValueError('Bin placement needs raised transit, lateral opening alignment and lower release')
        if min(bottom_z(p[i],side,value['geometry']) for i in (0,1)) < value['geometry']['rim_z_m']+.01:
            raise ValueError('Bin transit requires 1 cm estimated object-bottom clearance above the rim')
    if name in ('insert_into_container','insert_onto_peg'):
        if (np.linalg.norm((xyz(points[1],side)-xyz(points[0],side))[:2]) > 1e-6
                or xyz(points[1],side)[2] >= xyz(points[0],side)[2]
                or xyz(points[2],side)[2] < xyz(points[0],side)[2]):
            raise ValueError('Insertion requires vertical descent and upward retreat')
        if bottom_z(points[0],side,value['geometry']) < value['geometry']['rim_z_m']+.01:
            raise ValueError('Insertion starts with estimated object bottom above the rim/peg')


def compile_workflow(value):
    from .hand_v15 import Step
    name, arms, p, dwell = value['name'], tuple(value['arms']), value['waypoints'], value['duration_s']
    def move(i, selected=arms, contact=False, floor=None):
        return Step('move', p[i], arms=selected, contact=contact, pad_floor_m=floor)
    def grip(g, selected=arms, floor=None):
        return Step('gripper', gripper=g, duration_s=dwell, arms=selected, pad_floor_m=floor)
    def verify(key): return Step('verify', checkpoint=key)
    def pick(start=0, tall=False, selected=arms, floor=None):
        end = start+(2 if tall else 1)
        return ([grip(1,selected)] + [move(i,selected) for i in range(start,end+1)]
                + [grip(0,selected,floor),move(end+1,selected),verify('held_target_only')])
    def bin_place(start):
        return [move(start),move(start+1),verify('opening_aligned'),
                move(start+2,contact=True),verify('supported_for_release'),grip(1),
                move(start+3),verify('placement_visible')]
    if name == 'pull_toward_base': return [move(0)]
    if name == 'pick_and_place':
        return pick()+[move(3),move(4),verify('supported_for_release'),grip(1),verify('placement_visible')]
    if name == 'transfer_between_arms':
        giver, receiver = (arms[0],), (arms[1],)
        return [verify('held_target_only'),move(0,giver),move(1,giver),verify('supported_for_release'),
                grip(1,giver),move(2,giver),verify('staged_object_visible')]+pick(3,selected=receiver)
    if name == 'pick_tall_object': return pick(tall=True)
    if name == 'pick_and_place_tall_object': return pick(tall=True)+bin_place(4)
    if name == 'pick_rim': return pick()
    if name == 'pinch_cloth':
        return [grip(1),move(0),move(1),grip(0),move(2),verify('cloth_secured')]
    if name in ('fold_cloth','repair_corner','align_corner','bimanual_fold'):
        return [grip(1),move(0),move(1),grip(0),move(2),verify('cloth_secured'),
                move(3),move(4),verify('cloth_laid_down'),grip(1),move(5),verify('cloth_result')]
    if name == 'extract_from_container':
        return pick(floor=value['geometry']['rim_z_m']+.002)+[
            move(3),verify('container_stayed'),move(4),verify('held_target_only')]
    if name == 'place_into_bin': return [verify('held_target_only')]+bin_place(0)
    if name in ('insert_into_container','insert_onto_peg'):
        return [verify('held_target_only'),move(0),verify('opening_aligned'),move(1,contact=True),
                verify('supported_for_release'),grip(1),move(2),verify('placement_visible')]
    if name == 'set_down':
        return [verify('held_target_only'),move(0),move(1,contact=True),
                verify('supported_for_release'),grip(1),verify('placement_visible')]
    if name == 'hold_and_push':
        holder, pusher = (arms[0],), (arms[1],)
        return [verify('holder_secured'),grip(0,pusher),move(0,pusher),move(1,pusher,contact=True),
                verify('push_result'),move(2,pusher)]
    if name in ('push','push_until_stop','sweep_area'):
        return [verify('empty_push_path'),grip(0),move(0)]+[
            move(i,contact=True) for i in range(1,len(p))]+[verify('push_result')]
    raise ValueError('Workflow has no executable recipe')
