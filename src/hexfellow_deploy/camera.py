"""Independent rendering component; privileged plant topic ends here."""
from pathlib import Path
import time
from deploy_anything.components.interface.base_interface import DABaseInterface
from .workspace import activate


class SimulationCamera(DABaseInterface):
    def __init__(self, name, plant_state, images, workspace, scene, output, fps=15, time_budget=.04):
        super().__init__(name, [plant_state, images], time_budget)
        self.source, self.images = plant_state, images
        self.workspace, self.scene, self.output, self.fps = workspace, scene, output, fps

    def initialize(self):
        if self.fps <= 0:
            raise ValueError("Camera frequency must be positive")
        self._initialized.value = True
        return True

    def clean_up(self):
        return True

    def run(self):
        activate(self.workspace)
        import cv2
        import mujoco
        from mobile_manipulator.sim.scene import (build_scene, policy_renderer, policy_camera_maps,
                                                   update_policy_camera, read_policy_depth)
        from mobile_manipulator.sim.scene_config import load_scene
        output = Path(self.output)/'camera'
        output.mkdir(parents=True, exist_ok=True)
        scene = build_scene(load_scene(self.scene), output/'scene.xml')
        renderer = policy_renderer(scene)
        maps = policy_camera_maps(scene)
        option = mujoco.MjvOption()
        option.geomgroup[3] = 0
        option.sitegroup[:] = 0
        self._ready.value = True
        try:
            while not self.stopping:
                started = time.monotonic()
                row = self.source.get()
                if row:
                    scene.data.qpos[:] = row['qpos']
                    scene.data.qvel[:] = row['qvel']
                    scene.data.time = row['sim_s']
                    mujoco.mj_forward(scene.model, scene.data)
                    update_policy_camera(renderer, scene, option)
                    rgb = cv2.remap(renderer.render(), *maps, interpolation=cv2.INTER_LINEAR)
                    depth = read_policy_depth(renderer, scene, maps)
                    self.images.put(row['measured'], rgb, depth, scene.camera_k.tolist())
                time.sleep(max(0., 1/self.fps-(time.monotonic()-started)))
        finally:
            renderer.close()
