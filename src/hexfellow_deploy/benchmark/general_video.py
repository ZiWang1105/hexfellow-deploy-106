"""Replay four qualified paths from saved MuJoCo states, at recorded 1x time."""
import argparse
import json
from pathlib import Path

import imageio_ffmpeg
import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from .public_tcp import transform_trace


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('report', help='Directory containing general_report results.json')
    parser.add_argument('--profile', default='x7-gp80-general-trajectory-sim-v2')
    args = parser.parse_args()
    root = Path(args.report)
    rows = json.loads((root/'results.json').read_text())
    names = ['shoulder_sweep', 'seven_joint_arc', 'dual_wave', 'rounded_corner']
    resources = []
    font = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', 17)
    small = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', 14)
    for name in names:
        row = max([r for r in rows if r['path'] == name and r['profile'] == args.profile and r['accepted']],
                  key=lambda r: r['effective_request_m_s'])
        directory = Path(row['run'])
        trace = np.load(directory/'trace.npz')
        metadata = json.loads((directory/'metadata.json').read_text())
        public = transform_trace(trace, metadata['controller']['policy_T_umi_tcp'])
        model = mujoco.MjModel.from_binary_path(str(directory/'plant/plant.mjb'))
        model.vis.global_.offwidth, model.vis.global_.offheight = 640, 290
        data = mujoco.MjData(model)
        renderer = mujoco.Renderer(model, height=290, width=640)
        camera = mujoco.MjvCamera()
        camera.lookat[:] = [.3, 0., 1.05]
        camera.distance, camera.azimuth, camera.elevation = 2.8, 135, -12
        resources.append((row, trace, metadata, public, model, data, renderer, camera))
    fps = 24
    duration = max(float(r[1]['time_s'][-1]) for r in resources)
    writer = imageio_ffmpeg.write_frames(str(root/'general_ik_montage.mp4'), (1280, 768), fps=fps,
        codec='libx264', pix_fmt_in='rgb24', pix_fmt_out='yuv420p', quality=8,
        output_params=['-movflags', '+faststart'])
    writer.send(None)
    try:
        for frame in range(int(np.ceil(duration*fps))+1):
            timestamp = frame/fps
            canvas = Image.new('RGB', (1280, 768), '#f7f8fa')
            draw = ImageDraw.Draw(canvas)
            draw.text((16, 8), 'General X7 IK | recorded shared MIT simulation | 1x playback | public pad-center TCP', font=font, fill='#17283b')
            draw.text((16, 34), 'Red: requested. Green: measured. Frozen controller; different path-specific speeds. No hardware validation.', font=small, fill='#17283b')
            for panel, (row, trace, metadata, public, model, data, renderer, camera) in enumerate(resources):
                index = min(len(trace['time_s'])-1, np.searchsorted(trace['time_s'], timestamp))
                data.qpos[:], data.qvel[:] = trace['plant_qpos'][index], trace['plant_qvel'][index]
                mujoco.mj_forward(model, data)
                renderer.update_scene(data, camera)
                base = data.body('maver_x4_asm_maver_x4_base_link')
                for side in metadata['moving_arms']:
                    arm = ['left', 'right'].index(side)
                    for field, color in [('requested', [.95, .1, .1, .9]), ('measured', [.1, .9, .3, .9])]:
                        point = base.xpos+base.xmat.reshape(3, 3)@public[field][index, arm, :3, 3]
                        geom = renderer.scene.geoms[renderer.scene.ngeom]
                        mujoco.mjv_initGeom(geom, mujoco.mjtGeom.mjGEOM_SPHERE, np.full(3, .012), point, np.eye(3).ravel(), np.array(color))
                        renderer.scene.ngeom += 1
                x, y = (panel%2)*640, 60+(panel//2)*350
                canvas.paste(Image.fromarray(renderer.render()), (x, y))
                draw.text((x+12, y+5), row['path']+' | '+row['split'], font=font, fill='white', stroke_width=1, stroke_fill='black')
                speed = min(v['peak_m_s'] for v in row['public_tcp']['moving_arm_speed'].values())
                rms = row['public_tcp']['position_mm']['rms']
                draw.text((x+12, y+297), f'Passing peak {speed:.3f} m/s | RMS {rms:.3f} mm | t={min(timestamp, trace["time_s"][-1]):.2f} s', font=small, fill='#17283b')
            writer.send(np.asarray(canvas))
            if frame == int(1.5*fps):
                canvas.save(root/'video_preview.png')
    finally:
        writer.close()
        for _, _, _, _, _, _, renderer, _ in resources:
            renderer.close()


if __name__ == '__main__':
    main()
