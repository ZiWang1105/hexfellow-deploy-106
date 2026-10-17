"""Export compact evidence and plots without hiding failed path/speed cases."""
import argparse
import json
from pathlib import Path

import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', action='append', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    rows = []
    sources = {}
    for manifest in args.manifest:
        source = Path(manifest)
        for case in json.loads(source.read_text()):
            public = case.get('public_tcp', {})
            directory = source.parent/case['name']
            meta_path = directory/'metadata.json'
            metadata = json.loads(meta_path.read_text()) if meta_path.exists() else {}
            spec = metadata.get('trajectory_spec') or {}
            path_name = spec.get('name', case['name'])
            # Early logs mislabeled the saved XZ rotation basis as XY. Numeric
            # geometry is unchanged; preserve the original identifier too.
            canonical_name = 'circle_xz' if path_name == 'circle_xy' else path_name
            row = dict(profile=Path(case['profile']).stem, path=canonical_name, recorded_path_name=path_name,
                split=case['split'], requested_speed_m_s=case['requested_speed_m_s'],
                effective_request_m_s=case['effective_request_m_s'],
                accepted=public.get('tracking_accepted', False), fast_accepted=public.get('fast_tracking_accepted', False),
                public_tcp=public, failure=case['result'].get('failure'),
                constraints=case['result'].get('constraint_checks'),
                envelope=metadata.get('request_envelope'),
                run=str(directory), profile_sha256=metadata.get('profile_sha256'))
            rows.append(row)
            if row['path'] not in sources and (directory/'trace.npz').exists():
                sources[row['path']] = (directory, metadata)
    (output/'results.json').write_text(json.dumps(rows, indent=2)+'\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    names = sorted(sources)
    fig = plt.figure(figsize=(18, 4.2*((len(names)+4)//5)))
    for index, name in enumerate(names):
        directory, metadata = sources[name]
        trace = np.load(directory/'trace.npz')
        ax = fig.add_subplot((len(names)+4)//5, 5, index+1, projection='3d')
        for side in metadata['moving_arms']:
            arm = ['left', 'right'].index(side)
            points = trace['requested'][:, arm, :3, 3]
            ax.plot(*points.T, label=side)
        points = np.concatenate([trace['requested'][:, ['left', 'right'].index(s), :3, 3] for s in metadata['moving_arms']])
        center = (points.max(axis=0)+points.min(axis=0))/2
        radius = max(.025, float(np.ptp(points, axis=0).max()/2)) * 1.12
        ax.set_xlim(center[0]-radius, center[0]+radius)
        ax.set_ylim(center[1]-radius, center[1]+radius)
        ax.set_zlim(center[2]-radius, center[2]+radius)
        ax.set_xticks(np.linspace(center[0]-radius, center[0]+radius, 3))
        ax.set_yticks(np.linspace(center[1]-radius, center[1]+radius, 3))
        ax.set_zticks(np.linspace(center[2]-radius, center[2]+radius, 3))
        ax.set_box_aspect((1, 1, 1))
        for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
            axis.get_major_formatter().set_useOffset(False)
        ax.set_title(name+'\n'+metadata['trajectory_spec']['split'], fontsize=10, pad=12)
        ax.set_xlabel('X m', fontsize=8, labelpad=0)
        ax.set_ylabel('Y m', fontsize=8, labelpad=0)
        ax.set_zlabel('Z m', fontsize=8, labelpad=0)
        ax.tick_params(labelsize=6)
        ax.view_init(elev=22, azim=-60)
    fig.suptitle('Frozen Cartesian requests in robot base | 8 training / 12 held-out paths')
    fig.subplots_adjust(left=.025, right=.96, top=.94, bottom=.04, hspace=.32, wspace=.18)
    fig.savefig(output/'trajectory_gallery.png', dpi=150)
    plt.close(fig)
    profiles = sorted(set(row['profile'] for row in rows))
    fig, ax = plt.subplots(figsize=(14, 6))
    width = .8/max(1, len(profiles))
    for index, profile in enumerate(profiles):
        values = []
        for name in names:
            valid = [r for r in rows if r['profile'] == profile and r['path'] == name and r['accepted']]
            values.append(max((min(v['peak_m_s'] for v in r['public_tcp']['moving_arm_speed'].values())
                               for r in valid), default=0.))
        ax.bar(np.arange(len(names))+index*width, values, width, label=profile)
    ax.axhline(1.2, color='black', ls='--', lw=1, label='1.2 m/s')
    ax.set_xticks(np.arange(len(names))+width*(len(profiles)-1)/2, names, rotation=60, ha='right')
    ax.set_ylabel('Highest passing sampled public TCP peak (m/s); slower arm for dual')
    ax.set_title('Observed passing speeds, not global maxima | zero means no sampled pass')
    ax.legend(fontsize=8)
    ax.grid(axis='y', alpha=.2)
    fig.tight_layout()
    fig.savefig(output/'passing_speeds.png', dpi=160)
    plt.close(fig)


if __name__ == '__main__':
    main()
