"""Generate static plots and a real-time-speed replay from recorded plant state."""
import argparse
import json
from pathlib import Path

import numpy as np


def plots(root):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import pinocchio as pin
    root = Path(root)
    selected = [('baseline_v1', 'Current v1, 50 Hz'),
                ('derivatives_100hz', 'Derivatives, 100 Hz'),
                ('candidate', 'Candidate + inertia FF'),
                ('nonlinear_gain10', 'AgileX-style NLP + shared QP')]
    colors = ['#a83b32', '#b98b19', '#16744a', '#435ca0']
    fig, axes = plt.subplots(2, 2, figsize=(12, 7), sharex=True)
    for (name, label), color in zip(selected, colors):
        trace = np.load(root/name/'trace.npz')
        t = trace['time_s']
        p, q = trace['requested'][:, 1], trace['measured'][:, 1]
        error = np.linalg.norm(p[:, :3, 3]-q[:, :3, 3], axis=1)*1000
        orientation = np.array([np.linalg.norm(pin.log3(a[:3, :3].T@b[:3, :3]))*180/np.pi
                                for a, b in zip(p, q)])
        axes[0, 0].plot(t, q[:, 2, 3], color=color, label=label)
        axes[0, 1].semilogy(t, np.maximum(error, .01), color=color)
        axes[1, 0].plot(t, np.linalg.norm(trace['tcp_velocity'][:, 1, :3], axis=1), color=color)
        axes[1, 1].semilogy(t, np.maximum(orientation, .001), color=color)
    axes[0, 0].plot(t, p[:, 2, 3], 'k--', label='Requested')
    axes[1, 0].plot(t, np.linalg.norm(trace['target_twist'][:, 1, :3], axis=1), 'k--')
    axes[1, 0].axhline(1.2, color='gray', ls=':', label='1.2 m/s')
    axes[0, 0].set_ylabel('TCP height (m)')
    axes[0, 1].set_ylabel('Time-aligned position error (mm)')
    axes[1, 0].set_ylabel('Measured TCP speed (m/s)')
    axes[1, 1].set_ylabel('Orientation error (degrees)')
    for ax in axes.flat:
        ax.grid(alpha=.2)
        ax.set_xlabel('Simulation time (s)')
    axes[0, 0].legend(fontsize=8)
    fig.suptitle('X7 / shared MIT plant / 1.3 m/s requested swing / no time alignment fitted')
    fig.tight_layout()
    fig.savefig(root/'comparison.png', dpi=180)
    plt.close(fig)
    trace = np.load(root/'candidate/trace.npz')
    fig = plt.figure(figsize=(12, 4))
    ax = fig.add_subplot(121, projection='3d')
    for field, color in [('requested', 'black'), ('measured', '#16744a')]:
        points = trace[field][:, 1, :3, 3]
        ax.plot(*points.T, label=field, color=color)
    ax.set(xlabel='Base X (m)', ylabel='Base Y (m)', zlabel='Base Z (m)', title='Candidate trajectory')
    ax.legend()
    ax = fig.add_subplot(122)
    error = (trace['measured'][:, 1, :3, 3]-trace['requested'][:, 1, :3, 3])*1000
    for i, name in enumerate('XYZ'):
        ax.plot(trace['time_s'], error[:, i], label=name)
    ax.set(xlabel='Simulation time (s)', ylabel='Signed tracking error (mm)', title='Candidate, unchanged MIT gains')
    ax.grid(alpha=.2)
    ax.legend()
    fig.tight_layout()
    fig.savefig(root/'candidate_tracking.png', dpi=180)
    plt.close(fig)


def video(root):
    import mujoco
    import imageio_ffmpeg
    from PIL import Image, ImageDraw, ImageFont
    root = Path(root)
    names = [('baseline_v1', 'Current v1 | 50 Hz'), ('candidate', 'Candidate | 100 Hz + inertia FF')]
    resources = []
    font_path = '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'
    font = ImageFont.truetype(font_path, 18)
    small = ImageFont.truetype(font_path, 15)
    for name, label in names:
        trace = np.load(root/name/'trace.npz')
        model = mujoco.MjModel.from_binary_path(str(root/name/'plant/plant.mjb'))
        model.vis.global_.offwidth, model.vis.global_.offheight = 640, 580
        data = mujoco.MjData(model)
        renderer = mujoco.Renderer(model, height=580, width=640)
        camera = mujoco.MjvCamera()
        camera.lookat[:] = [.35, 0., .95]
        camera.distance, camera.azimuth, camera.elevation = 2.7, 135, -15
        summary = json.loads((root/name/'summary.json').read_text())
        resources.append((trace, model, data, renderer, camera, label, summary))
    fps = 30
    duration = max(float(r[0]['time_s'][-1]) for r in resources)
    writer = imageio_ffmpeg.write_frames(str(root/'comparison.mp4'), (1280, 720), fps=fps,
        codec='libx264', pix_fmt_in='rgb24', pix_fmt_out='yuv420p', quality=8,
        output_params=['-movflags', '+faststart'])
    writer.send(None)
    try:
        for frame_index in range(int(np.ceil(duration*fps))+1):
            timestamp = min(duration, frame_index/fps)
            canvas = Image.new('RGB', (1280, 720), '#f7f8fa')
            draw = ImageDraw.Draw(canvas)
            draw.text((20, 10), 'X7 IK comparison | recorded MIT simulation | playback 1x', font=font, fill='#15202b')
            draw.text((20, 39), 'Red marker: requested fingertip TCP. Same target, geometry, MIT gains and joint limits.',
                      font=small, fill='#25364a')
            for col, (trace, model, data, renderer, camera, label, summary) in enumerate(resources):
                index = min(len(trace['time_s'])-1, int(np.searchsorted(trace['time_s'], timestamp)))
                data.qpos[:], data.qvel[:] = trace['plant_qpos'][index], trace['plant_qvel'][index]
                mujoco.mj_forward(model, data)
                renderer.update_scene(data, camera)
                base = data.body('maver_x4_asm_maver_x4_base_link')
                target = base.xpos+base.xmat.reshape(3, 3)@trace['requested'][index, 1, :3, 3]
                geom = renderer.scene.geoms[renderer.scene.ngeom]
                mujoco.mjv_initGeom(geom, mujoco.mjtGeom.mjGEOM_SPHERE, np.array([.022]*3),
                                   target, np.eye(3).ravel(), np.array([.9, .12, .12, .85]))
                renderer.scene.ngeom += 1
                canvas.paste(Image.fromarray(renderer.render()), (col*640, 73))
                draw.text((col*640+18, 75), label, font=font, fill='#ffffff', stroke_width=1, stroke_fill='#151515')
                error = np.linalg.norm(trace['requested'][index, 1, :3, 3]-trace['measured'][index, 1, :3, 3])*1000
                speed = np.linalg.norm(trace['tcp_velocity'][index, 1, :3])
                metrics = summary['metrics']
                draw.text((col*640+18, 658), f't={timestamp:.2f} s   speed={speed:.2f} m/s   error={error:.1f} mm',
                          font=small, fill='#15202b')
                draw.text((col*640+18, 684), f'Motion RMS: {metrics["measured"]["position_mm"]["rms"]:.2f} mm   '
                          f'Peak speed: {metrics["achieved_speed_m_s"]["max"]:.2f} m/s', font=small, fill='#15202b')
            writer.send(np.asarray(canvas))
            if frame_index == int(fps*1.4):
                canvas.save(root/'video_preview.png')
    finally:
        writer.close()
        for _, _, _, renderer, _, _, _ in resources:
            renderer.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory')
    parser.add_argument('--video', action='store_true')
    args = parser.parse_args()
    plots(args.directory)
    if args.video:
        video(args.directory)


if __name__ == '__main__':
    main()
