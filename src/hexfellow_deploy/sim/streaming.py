"""Plant-owned rendering consumers. Privileged snapshots never enter control."""
from pathlib import Path
import queue
import time


def render_worker(scene_path, directory, states, images, stop, ready, video=False):
    from ..hardware.scheduling import runtime_cpu
    import mujoco
    import numpy as np
    import cv2
    from .scene import build_scene, policy_renderer, policy_camera_maps, update_policy_camera, read_policy_depth
    from .scene_config import load_scene
    from .sensors import EncoderReader
    from .recording import SceneRecorder
    recorder = sensor = None
    try:
        with runtime_cpu('background'):
            directory = Path(directory)
            directory.mkdir(parents=True, exist_ok=True)
            scene = build_scene(load_scene(scene_path), directory/'scene.xml')
            model, data = scene.model, scene.data
            if video:
                recorder = SceneRecorder(scene, directory, video=True)
            else:
                sensor = policy_renderer(scene)
                maps = policy_camera_maps(scene)
                option = mujoco.MjvOption()
                option.geomgroup[3] = 0
                option.sitegroup[:] = 0
                encoders = EncoderReader(model)
            ready.send({'ok': True})
            last = 0
            while True:
                started = time.monotonic()
                try:
                    sequence, values = states.get()
                except queue.Empty:
                    if stop.wait(.002):
                        break
                    continue
                if sequence != last:
                    last = sequence
                    nq,nv,nu = model.nq,model.nv,model.nu
                    data.time = values[0]
                    data.qpos[:] = values[1:1+nq]
                    data.qvel[:] = values[1+nq:1+nq+nv]
                    data.ctrl[:] = values[1+nq+nv:1+nq+nv+nu]
                    mujoco.mj_forward(model, data)
                    if video:
                        recorder.step()
                    else:
                        update_policy_camera(sensor, scene, option)
                        rgb = cv2.remap(sensor.render(), *maps, interpolation=cv2.INTER_LINEAR)
                        depth = read_policy_depth(sensor, scene, maps)
                        images.put(dict(encoders.read(data.sensordata, data.time), rgb=rgb, depth_m=depth), started)
                if stop.is_set():
                    break
                stop.wait(max(0, 1/15-(time.monotonic()-started)))
    except BaseException as error:
        try:
            ready.send({'error': f'{type(error).__name__}: {error}'})
        except (OSError, EOFError):
            pass
    finally:
        if recorder:
            recorder.close()
        if sensor:
            sensor.close()
        ready.close()


class PlantStreams:
    def __init__(self, scene, scene_path, directory, images, video):
        import multiprocessing as mp
        import numpy as np
        from ..interfaces.mailbox import ArrayMailbox
        self.context = mp.get_context('spawn')
        self.states = ArrayMailbox(self.context, 1+scene.model.nq+scene.model.nv+scene.model.nu)
        self.stop = self.context.Event()
        self.workers = []
        self.publish(scene.data)
        for mode in ([False, True] if video else [False]):
            parent, child = self.context.Pipe(duplex=False)
            name = 'video' if mode else 'camera'
            process = self.context.Process(target=render_worker, name=f'sim-{name}',
                args=(str(scene_path), str(Path(directory)/name), self.states, images,
                      self.stop, child, mode))
            process.start()
            child.close()
            self.workers.append((process,parent))
            if not parent.poll(60):
                self.close()
                raise RuntimeError(f'Simulator {name} startup timeout')
            message = parent.recv()
            if 'error' in message:
                self.close()
                raise RuntimeError(message['error'])
        # First published sensor packet must exist before controller ownership.
        import queue
        import time
        deadline = time.monotonic()+10
        while True:
            try:
                images.get()
                break
            except queue.Empty:
                if time.monotonic() > deadline:
                    self.close()
                    raise RuntimeError('Simulator camera did not publish')
                time.sleep(.01)

    def publish(self, data):
        import numpy as np
        self.states.put(np.r_[data.time, data.qpos, data.qvel, data.ctrl])
        for process, pipe in getattr(self, 'workers', []):
            if pipe.poll():
                try:
                    message = pipe.recv()
                except EOFError as error:
                    raise RuntimeError('Simulator stream exited') from error
                if 'error' in message:
                    raise RuntimeError(message['error'])
            if not process.is_alive():
                raise RuntimeError('Simulator stream exited')

    def close(self):
        self.stop.set()
        errors = []
        for process, pipe in self.workers:
            process.join(timeout=35)
            if process.is_alive():
                process.terminate()
                process.join()
                errors.append('Simulator rendering shutdown timeout')
            if process.exitcode:
                errors.append(f'Simulator rendering exited {process.exitcode}')
            if pipe.poll():
                try:
                    message = pipe.recv()
                    if 'error' in message:
                        errors.append(message['error'])
                except EOFError:
                    pass
            pipe.close()
        if errors:
            raise RuntimeError('; '.join(errors))
