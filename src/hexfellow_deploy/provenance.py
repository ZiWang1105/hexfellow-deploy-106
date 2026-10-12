"""Record code identities without secrets or full environment dumps."""
import hashlib
import importlib.metadata
from pathlib import Path
import subprocess


def code_identity(root):
    root = Path(root)
    def git(*args):
        result = subprocess.run(['git', *args], cwd=root, capture_output=True, text=True)
        return result.stdout.strip() if result.returncode == 0 else None
    digest = hashlib.sha256()
    for group in ('src', 'configs', 'scripts'):
        for path in sorted((root/group).rglob('*')):
            if path.is_file() and path.suffix in ('.py', '.yaml', '.json'):
                digest.update(str(path.relative_to(root)).encode()+b'\0'+path.read_bytes())
    return dict(commit=git('rev-parse','HEAD'), dirty=bool(git('status','--porcelain')),
                implementation_sha256=digest.hexdigest())


def collect(workspace):
    root = Path(__file__).resolve().parents[2]
    return dict(repositories={name:code_identity(path) for name,path in (
        ('hexfellow-deploy',root), ('deploy-anything',root/'third_party/deploy-anything'),
        ('workspace',Path(workspace)), ('UMI_Hardware',root/'third_party/UMI_Hardware'))},
        packages={name:importlib.metadata.version(name) for name in
                  ('numpy','mujoco','hydra-core','posix_ipc','deploy-anything')})
