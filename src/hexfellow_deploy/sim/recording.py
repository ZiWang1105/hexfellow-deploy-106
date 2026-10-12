"""Operator video and separate privileged evaluation. Never policy inputs."""
import json
import subprocess
from pathlib import Path

import cv2
import imageio_ffmpeg
import mujoco
import numpy as np

from ..robot.joints import SIDES
from ..interfaces.depth import visualize_depth
from .contacts import is_static_body
from .scene import policy_camera_maps, update_policy_camera, policy_renderer, read_policy_depth


class SceneRecorder:
    def __init__(self, scene, directory, *, video=True, sensors=True, operator_annotation=()):
        self.scene, self.directory = scene, Path(directory)
        self.video = video
        self.operator_annotation = tuple(operator_annotation)
        self.next_frame = 0.
        self.fps = 15
        self.option = mujoco.MjvOption()
        self.option.geomgroup[3] = 0
        self.option.sitegroup[:] = 0
        self.sensor = policy_renderer(scene) if sensors else None
        self.maps = policy_camera_maps(scene) if sensors else None
        self.operator = mujoco.Renderer(scene.model, height=720, width=960) if video else None
        self.detail = mujoco.Renderer(scene.model, height=270, width=480) if video else None
        self.overview = mujoco.MjvCamera()
        self.overview.lookat[:] = [1.25, 0, .75]
        self.overview.distance = 3.9
        self.overview.azimuth, self.overview.elevation = 132, -24
        self.closeup = mujoco.MjvCamera()
        self.closeup.distance = .62
        self.closeup.azimuth, self.closeup.elevation = 115, -22
        self.movie = None
        self.video_frames = 0
        self.last_video_frame = None
        if video:
            log = (self.directory / "encoder.log").open("wb")
            self.encoder_log = log
            self.movie = subprocess.Popen([
                imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-loglevel", "error", "-f", "rawvideo",
                "-vcodec", "rawvideo", "-pix_fmt", "rgb24", "-s", "1440x900", "-r", str(self.fps),
                "-i", "-", "-an", "-vcodec", "libx264", "-preset", "veryfast", "-crf", "20",
                "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(self.directory/"pickup.mp4")],
                stdin=subprocess.PIPE, stderr=log)
        self.truth = (self.directory / "evaluation.jsonl").open("w")
        self.trajectory = (self.directory / "trajectory.jsonl").open("w")
        self.initial = {}
        self.retained_since = {}
        self.success = None
        self.last_evaluation = None
        self.robot_static_contact_frames = 0
        self.first_robot_static_contact = None
        # Evaluation records every dynamic scene body generically. Selection of
        # the requested object is checked later by the independent run evaluator.
        flex_bodies = set(scene.model.flex_vertbodyid)
        self.objects = [body for body in range(scene.model.nbody)
                        if scene.model.body(body).name.startswith("entity_")
                        and scene.model.body_jntnum[body] > 0 and body not in flex_bodies]
        for body in self.objects:
            self.initial[body] = scene.data.xpos[body].copy()

    def rgb(self):
        update_policy_camera(self.sensor, self.scene, self.option)
        return cv2.remap(self.sensor.render(), *self.maps, interpolation=cv2.INTER_LINEAR)

    def rgbd(self):
        """One frozen capture state, identical optics and registration for both images."""
        rgb = self.rgb()
        return rgb, read_policy_depth(self.sensor, self.scene, self.maps)

    def step(self):
        model, data = self.scene.model, self.scene.data
        if data.time+1e-9 < self.next_frame:
            return
        self.next_frame += 1/self.fps
        self.trajectory.write(json.dumps({"time_s": float(data.time), "qpos": data.qpos.tolist(),
                                         "qvel": data.qvel.tolist(), "ctrl": data.ctrl.tolist()})+"\n")
        tcps = {side: data.site_xpos[model.site(side+"_tcp").id] for side in SIDES}
        objects = []
        for body in self.objects:
            touching = {side: set() for side in SIDES}
            support = False
            for c in data.contact:
                if c.geom1 < 0 or c.geom2 < 0:
                    continue  # Deformables are evaluated from offline flex vertices.
                a, b = int(model.geom_bodyid[c.geom1]), int(model.geom_bodyid[c.geom2])
                if body not in [a, b]:
                    continue
                other = b if a == body else a
                name = model.body(other).name
                for side in SIDES:
                    if name.startswith(side+"_jaw_"):
                        touching[side].add(name)
                if name.startswith("entity_") or other == 0:
                    support = True
            height = float(data.xpos[body, 2]-self.initial[body][2])
            grasping_arms = [side for side in SIDES if len(touching[side]) == 2
                            and np.linalg.norm(data.xpos[body] - tcps[side]) < .09]
            retained = height >= .10 and bool(grasping_arms) and not support
            if retained:
                self.retained_since.setdefault(body, float(data.time))
            else:
                self.retained_since.pop(body, None)
            held = float(data.time-self.retained_since.get(body, data.time))
            item = {"name": model.body(body).name, "position_m": data.xpos[body].tolist(),
                    "height_gain_m": height, "opposing_jaw_contacts": any(len(t) == 2 for t in touching.values()),
                    "grasping_arms": grasping_arms,
                    "support_contact": support, "retained_s": held}
            objects.append(item)
            if held >= 10 and self.success is None:
                self.success = {"timestamp_s": float(data.time), **item,
                                "no_welds": model.neq == 0,
                                "criterion": "10 cm lift, opposing jaws, no support, retained for 10 s"}
                (self.directory / "physical_lift.json").write_text(json.dumps(self.success, indent=2)+"\n")
        # Evaluation only: reject collision-assisted demonstrations. Static
        # scene entities are generic; this does not select a target or task.
        forbidden_contacts = []
        for contact in data.contact:
            if contact.geom1 < 0 or contact.geom2 < 0:
                continue
            a, b = (int(model.geom_bodyid[g]) for g in (contact.geom1, contact.geom2))
            if contact.dist > 0:
                continue
            for static, other in ((a, b), (b, a)):
                if (model.body(static).name.startswith("entity_") and is_static_body(model, static)
                        and other != 0 and not model.body(other).name.startswith("entity_")):
                    forbidden_contacts.append([model.body(static).name, model.body(other).name])
        if forbidden_contacts:
            self.robot_static_contact_frames += 1
            if self.first_robot_static_contact is None:
                self.first_robot_static_contact = {"time_s": float(data.time), "bodies": forbidden_contacts}
        self.last_evaluation = {"time_s": float(data.time), "objects": objects,
                                "robot_static_contact_frames": self.robot_static_contact_frames,
                                "first_robot_static_contact": self.first_robot_static_contact}
        self.truth.write(json.dumps(self.last_evaluation)+"\n")
        if self.movie:
            self.operator.update_scene(data, camera=self.overview, scene_option=self.option)
            left = self.operator.render().copy()
            self.closeup.lookat[:] = (data.xpos[self.objects[0]] if self.objects else np.mean(list(tcps.values()), axis=0))
            self.detail.update_scene(data, camera=self.closeup, scene_option=self.option)
            closeup = self.detail.render().copy()
            rgb, depth = self.rgbd()
            frame = np.zeros((900, 1440, 3), dtype=np.uint8)
            frame[:720, :960] = left
            frame[:270, 960:] = cv2.resize(rgb, (480, 270))
            frame[270:540, 960:] = cv2.resize(visualize_depth(depth), (480, 270))
            frame[540:810, 960:] = closeup
            for text, origin in [("OPERATOR VIEW", (16, 30)), ("AGENT RGB", (976, 26)),
                                 ("AGENT DEPTH: optical Z in meters", (976, 298)),
                                 ("CONTACT DETAIL - OPERATOR ONLY", (976, 568)),
                                 (f"SIMULATION  t = {data.time:.1f} s", (16, 762)),
                                 ("RGB-D + physical proprioception -> two TCPs + grippers", (16, 800)),
                                 ("Physical contacts / no object attachment", (16, 838)),
                                 ("Depth: fixed TURBO scale 0.2 to 4.0 m", (976, 846)),
                                 ("Black = invalid; metric samples also provided", (976, 877)),
                                 ("RETAINED PICKUP" if any(x["retained_s"] >= 10 for x in objects)
                                  else "Pose execution", (16, 876))]:
                cv2.putText(frame, text, origin, cv2.FONT_HERSHEY_SIMPLEX, .52, (245, 245, 245), 1, cv2.LINE_AA)
            # This annotation is only on the composed operator movie. Neither
            # the raw sensor image nor any policy observation is modified.
            if self.operator_annotation:
                bottom = 45+25*len(self.operator_annotation)
                cv2.rectangle(frame, (8, 42), (951, bottom), (15, 15, 15), -1)
                for index, text in enumerate(self.operator_annotation):
                    cv2.putText(frame, text, (16, 63+25*index), cv2.FONT_HERSHEY_SIMPLEX,
                                .55, (255, 205, 125), 1, cv2.LINE_AA)
            target_frame = round(float(data.time) * self.fps)
            while self.last_video_frame is not None and self.video_frames < target_frame:
                self.movie.stdin.write(self.last_video_frame)
                self.video_frames += 1
            self.last_video_frame = frame.tobytes()
            self.movie.stdin.write(self.last_video_frame)
            self.video_frames += 1

    def close(self):
        (self.directory / "final_evaluation.json").write_text(json.dumps(self.last_evaluation, indent=2)+"\n")
        self.truth.close()
        self.trajectory.close()
        if self.sensor is not None:
            self.sensor.close()
        if self.movie:
            self.movie.stdin.close()
            if self.movie.wait(timeout=30) != 0:
                raise RuntimeError("Video encoder failed; inspect encoder.log")
            self.encoder_log.close()
            self.operator.close()
            self.detail.close()
