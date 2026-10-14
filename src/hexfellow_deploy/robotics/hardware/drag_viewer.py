"""Isolated MuJoCo input/display process; it never opens a hardware gateway."""

import json
import multiprocessing as mp
import queue
import tempfile
import time
from pathlib import Path


class LatestValue:
    """One bounded snapshot. Neither writer nor reader waits for the peer."""

    def __init__(self, context, capacity=16384):
        self.storage = context.Array('B', capacity, lock=False)
        self.length = context.Value('i', 0, lock=False)
        self.lock = context.Lock()

    def put(self, value):
        raw = json.dumps(value, allow_nan=False, separators=(',', ':')).encode()
        if len(raw) > len(self.storage):
            raise ValueError('viewer snapshot exceeds bounded capacity')
        if not self.lock.acquire(False):
            return False
        try:
            self.storage[:len(raw)] = raw
            self.length.value = len(raw)
        finally:
            self.lock.release()
        return True

    def get(self):
        if not self.lock.acquire(False):
            return None
        try:
            raw = bytes(self.storage[:self.length.value])
        finally:
            self.lock.release()
        return json.loads(raw) if raw else None


class DragViewer:
    def __init__(self, grippers, mode, *, robot_model=None, collision_enabled=True):
        self.robot_model = robot_model
        self.collision_enabled = collision_enabled
        context = mp.get_context('spawn')
        self.display = LatestValue(context)
        self.targets = LatestValue(context)
        self.errors = LatestValue(context)
        self.commands = context.Queue(maxsize=8)
        self.stop = context.Event()
        self.quit = context.Event()
        self.ready = context.Event()
        self.closed = context.Event()
        self.process = context.Process(target=viewer_main, args=(self, list(grippers), mode),
                                       name='mujoco-drag-viewer', daemon=True)

    def start(self):
        self.process.start()
        deadline = time.monotonic() + 60
        while not self.ready.wait(.05):
            if not self.process.is_alive() or time.monotonic() > deadline:
                detail = self.errors.get()
                self.close()
                raise RuntimeError(f'MuJoCo viewer failed to start: {detail}')

    def close(self):
        if getattr(self, '_parent_closed', False):
            return
        self._parent_closed = True
        self.closed.set()
        if self.process.pid is not None:
            self.process.join(2)
            if self.process.is_alive():
                self.process.terminate()
                self.process.join(2)
        self.commands.cancel_join_thread()
        self.commands.close()


def dragged_targets(previous, candidate, selected_body, active, target_ids, *, reset=False):
    """Only a native mouse perturbation of a TCP handle is an input command."""
    if reset or not active:
        return previous
    return {s: candidate[s].copy() if target_ids[s] == selected_body else pose.copy()
            for s, pose in previous.items()}


def set_drag_render_flags(options):
    """Keep robot controls D/I/T from changing the native viewer's visibility.

    MuJoCo also handles these keys after invoking the application callback;
    returning from that callback does not consume the native shortcuts.
    """
    import mujoco

    options.flags[mujoco.mjtVisFlag.mjVIS_STATIC] = True
    options.flags[mujoco.mjtVisFlag.mjVIS_INERTIA] = False
    options.flags[mujoco.mjtVisFlag.mjVIS_TRANSPARENT] = False


BODY_KEYS = {265: 'UP', 264: 'DOWN', 263: 'LEFT', 262: 'RIGHT',
             266: 'PAGEUP', 267: 'PAGEDOWN', 44: ',', 46: '.', 257: 'ENTER', 335: 'ENTER'}


def route_viewer_key(channel, state, key, now):
    """GLFW key codes; body selection cannot trigger the arm D/I/T controls."""
    if key == ord('Q'):
        channel.quit.set()
    elif key == 32 or (key == ord('T') and state and state['phase'] in {'drag', 'drag-arming'}):
        channel.stop.set()
    elif state:
        body_mode = state.get('keyboard_group') in {'lift', 'chassis'}
        name = BODY_KEYS.get(key) if body_mode else (chr(key) if key in (68, 73, 84) else None)
        if name is not None:
            try:
                channel.commands.put_nowait(dict(key=name, at=now, epoch=state['epoch']))
            except queue.Full:
                pass


def viewer_main(channel, grippers, mode):
    # Imports and model compilation are deliberately outside the IK process.
    import mujoco
    import mujoco.viewer
    import numpy as np
    import pinocchio as pin

    from ..control.ik import read_mocap_targets, set_mocap_pose, snap_targets_to_tcp
    from ..control.robot_model import load_robot_model
    from ..scene.builder import build_tabletop_scene
    from .state_mirror import apply_state

    spec = load_robot_model(channel.robot_model)
    state = None
    sequence = 0
    epoch = -1
    last_key = {}

    def key_callback(key):
        nonlocal state
        now = time.monotonic()
        if now - last_key.get(key, -1.) < .25:
            return
        last_key[key] = now
        route_viewer_key(channel, state, key, now)

    try:
        with tempfile.TemporaryDirectory(prefix='hexfellow-drag-') as directory:
            path = build_tabletop_scene(
                output_path=Path(directory) / 'viewer.xml', base_mode='planar', object_ids=(),
                include_table=False, include_box=False, include_trashcan=False,
                appearance_seed=0, preserve_fixed_inertias=True, robot_model=spec)
            model = mujoco.MjModel.from_xml_path(str(path))
            data = mujoco.MjData(model)
            mujoco.mj_resetDataKeyframe(model, data, model.key('home').id)
            mujoco.mj_forward(model, data)
            targets = snap_targets_to_tcp(model, data)
            target_ids = {s: model.body(f'{s}_ik_target').id for s in targets}
            with mujoco.viewer.launch_passive(model, data, key_callback=key_callback,
                                             show_left_ui=False, show_right_ui=False) as viewer:
                viewer.cam.lookat[:] = (.4, 0, .7)
                viewer.cam.distance = 2.6
                viewer.cam.azimuth = 135
                viewer.cam.elevation = -20
                with viewer.lock():
                    viewer.opt.geomgroup[:] = (1, 1, 1, 0, 1, 0)
                    set_drag_render_flags(viewer.opt)
                channel.ready.set()
                while viewer.is_running() and not channel.closed.is_set():
                    began = time.monotonic()
                    incoming = channel.display.get()
                    if incoming is not None:
                        state = incoming
                    if state is not None:
                        from types import SimpleNamespace
                        with viewer.lock():
                            # Kinematic measured-state display only: never mj_step.
                            apply_state(model, data, SimpleNamespace(q=np.asarray(state['q'])),
                                        grippers, spec.auxiliary_joints)
                            # Native Reset/Load-key changes time to zero. Detect it
                            # independently of keyboard shortcuts or hidden panels.
                            data.time = 1.
                        # MuJoCo applies native mocap perturbations in either sync
                        # mode; avoid copying static model/mesh data every frame.
                        viewer.sync(state_only=True)
                        with viewer.lock():
                            if epoch != state['epoch']:
                                viewer.perturb.active = 0
                                viewer.perturb.select = 0
                                targets = ({s: pin.SE3(np.asarray(m)) for s, m in state['anchors'].items()}
                                           if state['anchors'] else snap_targets_to_tcp(model, data))
                                epoch = state['epoch']
                            else:
                                targets = dragged_targets(
                                    targets, read_mocap_targets(model, data), viewer.perturb.select,
                                    viewer.perturb.active, target_ids,
                                    reset=data.time != 1. or bool(state.get('keyboard_group')))
                            if state.get('keyboard_group'):
                                viewer.perturb.active = 0
                                viewer.perturb.select = 0
                                targets = snap_targets_to_tcp(model, data)
                            for side, pose in targets.items():
                                set_mocap_pose(model, data, f'{side}_ik_target', pose)
                            data.xfrc_applied[:] = 0
                            data.qfrc_applied[:] = 0
                        sequence += 1
                        channel.targets.put(dict(at=time.monotonic(), sequence=sequence, epoch=epoch,
                                                 selected_body=int(viewer.perturb.select),
                                                 perturb_active=int(viewer.perturb.active),
                                                 targets={s: p.homogeneous.tolist() for s, p in targets.items()}))
                        label = 'LIVE: MEASURED ROBOT' if mode == 'live' else 'SIMULATION: NO HARDWARE'
                        if state.get('disconnected'):
                            label = 'PAUSED: LAST RECORDED POSE / NO LIVE FEEDBACK'
                        label += f' | {spec.model_id}'
                        if not channel.collision_enabled:
                            label += ' | COLLISION OFF'
                        details = (f"{state['phase']} | {state['message']}\n"
                                   f"Input {state['age_ms']:.1f} ms | IK {state['ik_ms']:.2f} ms\n"
                                   f"Reference error L/R: {state['errors_mm'][0]:.1f}/{state['errors_mm'][1]:.1f} mm\n"
                                   f"Input error L/R: {state.get('raw_errors_mm', [0,0])[0]:.1f}/{state.get('raw_errors_mm', [0,0])[1]:.1f} mm\n"
                                   f"Input rotation L/R: {state.get('raw_rotation_deg', [0,0])[0]:.1f}/{state.get('raw_rotation_deg', [0,0])[1]:.1f} deg")
                        if state.get('keyboard_group'):
                            details = f"{state['keyboard_group'].upper()} | {state['phase']}\n{state['message']}"
                        viewer.set_texts([(mujoco.mjtFontScale.mjFONTSCALE_100,
                                           mujoco.mjtGridPos.mjGRID_TOPLEFT, label, details),
                                          (mujoco.mjtFontScale.mjFONTSCALE_100,
                                           mujoco.mjtGridPos.mjGRID_BOTTOMLEFT,
                                           state.get('controls', 'D default | I init | T follow/stop | Space stop | Q quit'),
                                           state.get('input_help', 'Double click target; Ctrl+right drag move; Ctrl+left drag rotate'))])
                    else:
                        viewer.sync()
                    # Native keyboard and panel events are applied by sync.
                    # Restore these three flags even before a session is ready
                    # and while fault-paused; other view settings stay editable.
                    with viewer.lock():
                        set_drag_render_flags(viewer.opt)
                    time.sleep(max(0., 1 / 60 - (time.monotonic() - began)))
                viewer.close()
                deadline = time.monotonic() + 2
                while viewer.m is not None and time.monotonic() < deadline:
                    time.sleep(.01)
    except BaseException as error:
        channel.errors.put(dict(error=f'{type(error).__name__}: {error}'))
    finally:
        channel.quit.set()
