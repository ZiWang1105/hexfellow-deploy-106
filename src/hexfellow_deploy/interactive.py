"""Mouse targets for the shared simulator; the display never steps physics.

Plant coordinates stay in this operator-only renderer. Only paired base-frame
TCP poses and physical encoder observations enter the ordinary request topic.
"""
import copy
import json
import os
from pathlib import Path
import queue
import time
import xml.etree.ElementTree as ET

import numpy as np
from deploy_anything.components.interface.base_interface import DABaseInterface

SIDES = ('left', 'right')
ROOT_BODY = 'maver_x4_asm_maver_x4_base_link'


def make_display_model(plant_path):
    """Add collision-free mocap handles to a copy, leaving the plant untouched."""
    import mujoco
    tree = ET.parse(plant_path)
    world = tree.getroot().find('worldbody')
    for side in SIDES:
        body = ET.SubElement(world, 'body', name=f'{side}_drag_target', mocap='true')
        ET.SubElement(body, 'geom', type='sphere', size='.027', rgba='1 .12 .12 .8',
                      contype='0', conaffinity='0', group='5')
        for axis, end in enumerate(((.085, 0, 0), (0, .085, 0), (0, 0, .085))):
            color = ['.2', '.2', '.2', '1']
            color[axis] = '1'
            ET.SubElement(body, 'geom', type='capsule', size='.003',
                          fromto='0 0 0 '+' '.join(map(str, end)), rgba=' '.join(color),
                          contype='0', conaffinity='0', group='5')
    # Old keyframes have mocap arrays of the original size. This display uses
    # authoritative snapshots instead of keyframes, including after GUI Reset.
    keys = tree.getroot().find('keyframe')
    if keys is not None:
        tree.getroot().remove(keys)
    path = Path(plant_path).with_name('interactive_display.xml')
    tree.write(path, encoding='unicode')
    model = mujoco.MjModel.from_xml_path(str(path))
    return model, mujoco.MjData(model)


def apply_snapshot(model, data, snapshot):
    import mujoco
    data.qpos[:] = snapshot['qpos']
    data.qvel[:] = snapshot['qvel']
    data.time = 1.  # Native Reset is recognizable after sync().
    data.xfrc_applied[:] = 0
    data.qfrc_applied[:] = 0
    mujoco.mj_forward(model, data)


def tcp_targets(model, data):
    """Measured TCP sites, in world coordinates for the display only."""
    from scipy.spatial.transform import Rotation
    return {side: dict(position_m=data.site(f'{side}_tcp').xpos.copy(),
        quaternion_xyzw=Rotation.from_matrix(data.site(f'{side}_tcp').xmat.reshape(3, 3)).as_quat())
        for side in SIDES}


def write_handles(model, data, targets):
    for side, pose in targets.items():
        index = model.body(f'{side}_drag_target').mocapid[0]
        data.mocap_pos[index] = pose['position_m']
        data.mocap_quat[index] = np.asarray(pose['quaternion_xyzw'])[[3, 0, 1, 2]]


def read_drag(model, data, targets, selected, active, *, reset=False):
    """Ignore Reset and attempts to perturb robot or scene geometry."""
    result = copy.deepcopy(targets)
    if reset or not active:
        return result
    for side in SIDES:
        body = model.body(f'{side}_drag_target')
        if selected == body.id:
            index = body.mocapid[0]
            result[side] = dict(position_m=data.mocap_pos[index].copy(),
                quaternion_xyzw=data.mocap_quat[index][[1, 2, 3, 0]].copy())
    return result


def paired_action(model, data, targets, grippers):
    from scipy.spatial.transform import Rotation
    from hexfellow_deploy.interfaces.contract import Action
    base = data.body(ROOT_BODY)
    rotation = base.xmat.reshape(3, 3)
    action = {}
    for side, pose in targets.items():
        action[side] = dict(position_m=(rotation.T @ (pose['position_m']-base.xpos)).tolist(),
            quaternion_xyzw=Rotation.from_matrix(rotation.T @
                Rotation.from_quat(pose['quaternion_xyzw']).as_matrix()).as_quat().tolist(),
            gripper_open=grippers[side])
    return Action.from_dict(action).to_dict()


def changed(a, b):
    return any(not np.allclose(a[s][key], b[s][key], atol=1e-7, rtol=0)
               for s in SIDES for key in ('position_m', 'quaternion_xyzw'))


class InteractiveIKViewer(DABaseInterface):
    def __init__(self, name, plant_state, feedback, request, status, decision, viewer_status,
                 output, backend='simulation', fps=30):
        super().__init__(name, [plant_state, feedback, request, status, decision, viewer_status], 1/fps)
        if backend != 'simulation':
            raise ValueError('Interactive IK viewer is simulation-only')
        self.source, self.request, self.control = plant_state, request, status
        self.feedback = feedback
        self.decision, self.viewer_status = decision, viewer_status
        self.output, self.period = Path(output), 1/fps

    def initialize(self):
        self._initialized.value = True
        return True

    def clean_up(self):
        self._cleaned_up.value = True
        return True

    def run(self):
        if not os.environ.get('DISPLAY'):
            raise RuntimeError('Interactive IK needs a desktop DISPLAY; run this command in a desktop terminal')
        import mujoco
        import mujoco.viewer
        from scipy.spatial.transform import Rotation
        events = queue.SimpleQueue()
        deadline = time.monotonic()+30
        snapshot = None
        while not self.stopping:
            state = self.control.get()
            snapshot = self.source.get()
            if (snapshot and (self.output/'backend_metadata.json').exists() and state
                    and state.get('phase') == 'running'):
                break
            if time.monotonic() > deadline:
                raise RuntimeError('Interactive viewer timed out waiting for simulator startup')
            time.sleep(.02)
        if self.stopping:
            return
        model, data = make_display_model(self.output/'plant.xml')
        apply_snapshot(model, data, snapshot)
        targets = tcp_targets(model, data)
        grippers = {s: int(snapshot['measured']['gripper_state'][s] != 'closed') for s in SIDES}
        write_handles(model, data, targets)
        sequence = 0
        message = 'Drag a red target to move either arm. Lift stays fixed.'
        last_sent = copy.deepcopy(targets)
        paused = False
        def send(operation, action=None):
            nonlocal sequence, message
            sequence += 1
            row = dict(id=sequence, operation=operation)
            if action is not None:
                # Rendering can take longer than a control tick. Read the fast
                # encoder topic immediately before publication, not an old GUI frame.
                latest = self.feedback.get(timeout=.005)
                if latest is None or time.monotonic()-latest['measured']['timestamp_s'] > .1:
                    message = 'Waiting for fresh feedback before submitting target'
                    return False
                row.update(action=action, observed=latest['measured'])
            if not self.request.put(row, timeout=.005):
                message = 'Input publication busy; retrying'
                return False
            event = dict(kind='interactive_ik', request_id=sequence, operation=operation,
                         action=action, policy_calls=0, at=time.monotonic())
            self.decision.put(event)
            record.write(json.dumps(event)+'\n')
            record.flush()
            return True
        with (self.output/'interactive.jsonl').open('w') as record:
            with mujoco.viewer.launch_passive(model, data, key_callback=events.put,
                    show_left_ui=False, show_right_ui=False) as viewer:
                viewer.cam.lookat[:] = data.body(ROOT_BODY).xpos+np.array([.25, 0, .7])
                viewer.cam.distance = 2.7
                viewer.cam.azimuth = 135
                viewer.cam.elevation = -20
                with viewer.lock():
                    viewer.opt.geomgroup[5] = 1
                    viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_PERTFORCE] = False
                self._ready.value = True
                self.viewer_status.put(dict(phase='running'))
                try:
                    while not self.stopping and viewer.is_running():
                        began = time.monotonic()
                        incoming = self.source.get()
                        if incoming:
                            snapshot = incoming
                        state = self.control.get() or {}
                        fresh = began-snapshot['measured']['timestamp_s'] < .25
                        usable = fresh and not state.get('fault') and not state.get('final_hold')
                        if not usable and not paused:
                            send('hold')
                            paused = True
                        reset = False
                        while not events.empty():
                            key = events.get()
                            if key in (ord('Q'), 256):
                                return
                            if key in (32, ord('R')):
                                send('hold')
                                reset = True
                                paused = not usable
                                message = 'Holding; targets reset to measured TCPs. Drag to resume.'
                        with viewer.lock():
                            apply_snapshot(model, data, snapshot)
                            if reset:
                                targets = tcp_targets(model, data)
                                last_sent = copy.deepcopy(targets)
                                viewer.perturb.active = 0
                                viewer.perturb.select = 0
                            write_handles(model, data, targets)
                        viewer.sync(state_only=True)
                        with viewer.lock():
                            targets = read_drag(model, data, targets, viewer.perturb.select,
                                viewer.perturb.active, reset=data.time != 1. or not usable or paused)
                            # Restore authoritative state after native GUI reset/load
                            # or attempts to perturb the robot instead of a handle.
                            apply_snapshot(model, data, snapshot)
                            write_handles(model, data, targets)
                            if viewer.perturb.select not in [model.body(f'{s}_drag_target').id for s in SIDES]:
                                viewer.perturb.active = 0
                            retry = (state.get('request_id') == sequence and
                                     state.get('request_error') == 'Posture changed during inference; reobserve')
                            if usable and not paused and (changed(last_sent, targets) or retry):
                                try:
                                    action = paired_action(model, data, targets, grippers)
                                    if send('target', action):
                                        last_sent = copy.deepcopy(targets)
                                        message = f'Target submitted ({sequence})'
                                except ValueError as error:
                                    message = str(error)
                            actual = tcp_targets(model, data)
                            errors = [1000*np.linalg.norm(targets[s]['position_m']-actual[s]['position_m']) for s in SIDES]
                            angles = [np.degrees((Rotation.from_quat(targets[s]['quaternion_xyzw']).inv()*
                                Rotation.from_quat(actual[s]['quaternion_xyzw'])).magnitude()) for s in SIDES]
                        detail = (state.get('fault') or state.get('request_error') or
                                  ('Feedback stale; press R after recovery' if paused else message))
                        selected_side = next((s for s in SIDES if
                            model.body(f'{s}_drag_target').id == viewer.perturb.select), 'none')
                        self.viewer_status.put(dict(phase='running', selected_side=selected_side,
                            perturb_active=int(viewer.perturb.active), request_id=sequence,
                            position_error_mm=errors, rotation_error_deg=angles))
                        viewer.set_texts([(mujoco.mjtFontScale.mjFONTSCALE_100, mujoco.mjtGridPos.mjGRID_TOPLEFT,
                            'SIMULATION | Shared PinkIK / MIT | no policy API',
                            f"{detail}\nTCP position error L/R: {errors[0]:.1f} / {errors[1]:.1f} mm\n"
                            f"TCP rotation error L/R: {angles[0]:.1f} / {angles[1]:.1f} deg\n"
                            f"Selected target: {selected_side}\n"
                            f"Controller: {state.get('diagnostic', {}).get('mode', 'starting')}"),
                            (mujoco.mjtFontScale.mjFONTSCALE_100, mujoco.mjtGridPos.mjGRID_BOTTOMLEFT,
                             'Double-click red sphere; Ctrl+right drag moves; Ctrl+left drag rotates',
                             'Space / R: hold and reset targets | Q / window close: quit')])
                        time.sleep(max(0., began+self.period-time.monotonic()))
                finally:
                    send('hold')
                    self.viewer_status.put(dict(phase='closed'), timeout=.05)
                    self.signal_stop()
