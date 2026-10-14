"""Explicit tool assets shared by kinematics, dynamics and the two viewers."""

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..paths import PROJECT_ROOT, URDF_DIR


@dataclass(frozen=True)
class RobotModel:
    model_id: str
    urdf: Path
    tcp_offset_m: tuple[float, ...]
    auxiliary_joints: tuple[str, ...]
    fixed_auxiliary_positions: tuple[float, ...]
    assets: tuple[tuple[str, str], ...]
    descriptor: Path | None = None
    descriptor_sha256: str | None = None

    def report(self):
        return dict(model_id=self.model_id, urdf=str(self.urdf.relative_to(PROJECT_ROOT)),
                    tcp_offset_m=self.tcp_offset_m, auxiliary_joints=self.auxiliary_joints,
                    fixed_auxiliary_positions=self.fixed_auxiliary_positions,
                    assets=dict(self.assets), descriptor_sha256=self.descriptor_sha256)


def load_robot_model(reference=None):
    if isinstance(reference, RobotModel):
        return reference
    if reference is None:
        return RobotModel('maver-x4-das-v3', URDF_DIR / 'maver_x4_asm_asm.urdf',
                          (.00948191, 0., .262),
                          tuple(f'robot{s}_joint{i}' for s in ('', '2') for i in range(1, 7)),
                          (0.,) * 12, ())
    path = PROJECT_ROOT / reference['path']
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != reference['sha256']:
        raise ValueError('robot model descriptor hash mismatch')
    data = json.loads(raw)
    if data['schema_version'] != 1:
        raise ValueError('unsupported robot model descriptor')
    names, positions = tuple(data['auxiliary_joints']), tuple(data['fixed_auxiliary_positions'])
    tcp = tuple(data['tcp_offset_m'])
    if (len(tcp) != 3 or not np.all(np.isfinite(tcp)) or len(names) != len(positions)
            or len(set(names)) != len(names) or not np.all(np.isfinite(positions))):
        raise ValueError('invalid robot tool coordinates')
    if data['urdf'] not in data['assets']:
        raise ValueError('robot model must bind its URDF')
    for asset, digest in data['assets'].items():
        if hashlib.sha256((PROJECT_ROOT / asset).read_bytes()).hexdigest() != digest:
            raise ValueError(f'robot model asset hash mismatch: {asset}')
    return RobotModel(data['model_id'], PROJECT_ROOT / data['urdf'], tcp, names, positions,
                      tuple(sorted(data['assets'].items())), path, reference['sha256'])
