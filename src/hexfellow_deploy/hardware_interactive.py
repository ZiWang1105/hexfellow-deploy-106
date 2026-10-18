"""Measured-encoder hardware display. MuJoCo renders only; never steps a plant."""
import copy
import json
import queue
import time
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np

from .interactive import (InteractiveIKViewer, make_display_model, tcp_targets,
                          write_handles, read_drag, paired_action, ROOT_BODY)


def make_hardware_display(metadata, output):
    from .robotics.control.robot import build_scene
    from .robot.assembly import apply_assembly, load_assembly
    from .robot.upstream import RESOURCE_ROOT
    output = Path(output)
    path = output/'hardware_display_robot.xml'
    build_scene(Path(metadata['robot']['urdf']), path, base_mode='fixed', preserve_fixed_inertias=True)
    tree = ET.parse(path)
    root = tree.getroot()
    for parent in root.iter():
        for site in list(parent.findall('site')):
            if site.get('name') in ('left_tcp', 'right_tcp'):
                parent.remove(site)
    keys = root.find('keyframe')
    if keys is not None:
        root.remove(keys)
    apply_assembly(root, RESOURCE_ROOT, output/'display_assets',
                   load_assembly(metadata['robot']['assembly']))
    tree.write(path, encoding='unicode')
    return make_display_model(path)


def apply_encoders(model, data, measured):
    import mujoco
    from .robot.joints import JOINTS
    from .interfaces.contract import validate_sensor_packet
    validate_sensor_packet(measured)
    data.qpos[:] = model.qpos0
    data.qvel[:] = 0.
    for name in JOINTS:
        joint = model.joint(name)
        data.qpos[joint.qposadr[0]] = measured['joint_positions'][name]
        data.qvel[joint.dofadr[0]] = measured['joint_velocities'][name]
    data.time = 1.
    data.xfrc_applied[:] = 0.
    data.qfrc_applied[:] = 0.
    mujoco.mj_forward(model, data)


class HardwareIKViewer(InteractiveIKViewer):
    def __init__(self, name, feedback, request, status, decision, viewer_status, output, fps=30):
        # The base class provides lifecycle flags only. No simulated plant topic.
        super().__init__(name, feedback, feedback, request, status, decision, viewer_status,
                         output, fps=fps)

    def run(self):
        import mujoco
        import mujoco.viewer
        from scipy.spatial.transform import Rotation
        events = queue.SimpleQueue()
        until = time.monotonic()+30
        while not self.stopping:
            row = self.feedback.get()
            state = self.control.get() or {}
            path = self.output/'backend_metadata.json'
            if row and path.exists() and state.get('phase') == 'running':
                break
            if time.monotonic() >= until:
                raise RuntimeError('Hardware viewer timed out waiting for measured feedback and controller')
            time.sleep(.02)
        if self.stopping:
            return
        model, data = make_hardware_display(json.loads(path.read_text()), self.output)
        apply_encoders(model, data, row['measured'])
        targets = tcp_targets(model, data)
        write_handles(model, data, targets)
        sequence, session = 0, None
        was_dragging = False
        previous_selection = None
        message = 'Double-click a red TCP to select and enable. Ctrl + right drag moves; release: HOLD.'
        with (self.output/'interactive.jsonl').open('w') as record:
            def send(operation, action=None):
                nonlocal sequence
                sequence += 1
                now = time.monotonic()
                request = dict(id=sequence, operation=operation, created_s=now)
                if action is not None:
                    request.update(action=action, session=session, expires_s=now+.25)
                sent = self.request.put(request, timeout=.005)
                event = dict(request, kind='hardware_interactive_ik', published=sent, policy_calls=0)
                record.write(json.dumps(event)+'\n')
                record.flush()
                self.decision.put(event)
                return sequence if sent else None
            with mujoco.viewer.launch_passive(model, data, key_callback=events.put,
                    show_left_ui=False, show_right_ui=False) as viewer:
                viewer.cam.lookat[:] = data.body(ROOT_BODY).xpos+np.array([.25, 0, .7])
                viewer.cam.distance, viewer.cam.azimuth, viewer.cam.elevation = 2.7, 135, -20
                with viewer.lock():
                    viewer.opt.geomgroup[5] = 1
                    viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_PERTFORCE] = False
                self._ready.value = True
                try:
                    while not self.stopping and viewer.is_running():
                        began = time.monotonic()
                        incoming = self.feedback.get(timeout=.005)
                        if incoming is not None:
                            row = incoming
                        state = self.control.get() or {}
                        status_age = began-state.get('published_s', -float('inf'))
                        usable = (0 <= began-row['measured']['timestamp_s'] < .1
                                  and status_age < .25 and not state.get('fault')
                                  and not state.get('final_hold'))
                        reset = not usable
                        preserve_selection = False
                        select_side = None
                        while not events.empty():
                            key = events.get()
                            if key in (ord('Q'), 256):
                                return
                            if key in (32, ord('R')):
                                send('hold')
                                session = None
                                reset = True
                            elif key in (ord('1'), ord('2')):
                                select_side = 'left' if key == ord('1') else 'right'
                            elif key == ord('E') and usable:
                                session = send('interactive_enable')
                                reset = True
                                preserve_selection = True
                                message = 'Enabled. Select 1 (left) / 2 (right), then Ctrl + right drag. Release: HOLD.'
                        manual = state.get('manual_tcp', {})
                        if session is not None and (state.get('request_id') or -1) >= session and not manual.get('enabled'):
                            session = None
                            reset = True
                            message = manual.get('reason', state.get('request_error') or 'Disabled')
                        if not usable and session is not None:
                            send('hold')
                            session = None
                            message = 'Feedback/control status stale. Inspect, then press E.'
                        with viewer.lock():
                            apply_encoders(model, data, row['measured'])
                            if reset or session is None:
                                targets = tcp_targets(model, data)
                            if reset:
                                viewer.perturb.active = 0
                                if not preserve_selection:
                                    viewer.perturb.select = 0
                                was_dragging = False
                                previous_selection = None
                            write_handles(model, data, targets)
                            if select_side is not None:
                                viewer.perturb.active = 0
                                viewer.perturb.select = model.body(f'{select_side}_drag_target').id
                                viewer.perturb.localpos[:] = 0.
                                previous_selection = select_side
                        viewer.sync(state_only=True)
                        with viewer.lock():
                            native_reset = data.time != 1.
                            selected = next((s for s in ('left','right') if
                                model.body(f'{s}_drag_target').id == viewer.perturb.select), None)
                            # Native MuJoCo double-click selects a body. A newly
                            # selected TCP is explicit operator enable, even if no
                            # keyboard key was pressed. Selecting robot geometry
                            # never enables control. Clear selection after HOLD so
                            # another double-click on the same TCP is detectable.
                            if (selected is not None and selected != previous_selection
                                    and usable and not native_reset and not reset):
                                session = send('interactive_enable')
                                message = f'{selected.capitalize()} TCP enabled. Ctrl + right drag moves; release: HOLD.'
                            previous_selection = selected
                            dragging = bool(viewer.perturb.active and selected and usable and session is not None)
                            targets = read_drag(model, data, targets, viewer.perturb.select,
                                dragging, reset=native_reset)
                            apply_encoders(model, data, row['measured'])
                            if native_reset or (was_dragging and not dragging):
                                send('hold')
                                session = None
                                dragging = False
                                targets = tcp_targets(model, data)
                                viewer.perturb.select = 0
                                previous_selection = selected = None
                                message = 'HOLD. Double-click a red TCP to enable the next drag.' 
                            if dragging:
                                # Binary values are placeholders required by Action's
                                # schema. The control owner never applies jaw commands.
                                action = paired_action(model, data, targets, dict(left=0,right=0))
                                if send('interactive_target', action) is None:
                                    send('hold')
                                    session = None
                                    dragging = False
                                    message = 'Input publication failed; press E after inspection.'
                            write_handles(model, data, targets)
                            actual = tcp_targets(model, data)
                            errors = [float(1000*np.linalg.norm(targets[s]['position_m']-actual[s]['position_m']))
                                      for s in ('left','right')]
                            angles = [float(np.degrees((Rotation.from_quat(targets[s]['quaternion_xyzw']).inv()*
                                      Rotation.from_quat(actual[s]['quaternion_xyzw'])).magnitude())) for s in ('left','right')]
                            was_dragging = dragging
                        self.viewer_status.put(dict(phase='running', enabled=session is not None,
                            dragging=dragging, selected_side=selected, request_id=sequence,
                            position_error_mm=errors, rotation_error_deg=angles))
                        viewer.set_texts([(mujoco.mjtFontScale.mjFONTSCALE_100, mujoco.mjtGridPos.mjGRID_TOPLEFT,
                            'REAL ROBOT | encoder FK | no scene/obstacle sensing',
                            f"{state.get('request_error') or message}\n"
                            f"Selected: {selected or 'none'} | Input: {'ENABLED' if session is not None else 'DISABLED - double-click TCP'}\n"
                            f"Profile target rate: {manual.get('target_speed_m_s', '?')} m/s, "
                            f"{manual.get('target_speed_rad_s', '?')} rad/s | jaws braked, base/lift fixed\n"
                            f'TCP error L/R: {errors[0]:.1f}/{errors[1]:.1f} mm, {angles[0]:.1f}/{angles[1]:.1f} deg'),
                            (mujoco.mjtFontScale.mjFONTSCALE_100, mujoco.mjtGridPos.mjGRID_BOTTOMLEFT,
                            'Double-click TCP: select + enable | Ctrl+right: move | Ctrl+left: rotate | 1/2 + E: keyboard alternative',
                            'Release drag / Space / R: HOLD | Q / close: shutdown | physical E-stop for emergency')])
                        time.sleep(max(0., began+self.period-time.monotonic()))
                finally:
                    send('hold')
                    self.viewer_status.put(dict(phase='closed'), timeout=.05)
                    self.signal_stop()
