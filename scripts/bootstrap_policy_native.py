#!/usr/bin/env python3
"""Reuse a locally qualified policy-sim extension with explicit provenance.

Run using the destination environment's Python. This does not compile or claim
to reproduce a build: it checks the pinned clean source, ABI and native library
identity, copies the existing artifact, and relocates its library search path.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata as metadata
import json
import shutil
import subprocess
import sys
import sysconfig
import tempfile
from pathlib import Path

PIN = "9398b61972bf983c7ab5609591b7eb4c95c6357d"
# These revisions have byte-identical native build inputs and dependency pins.
# Runtime Python safety improvements in PIN still require separate qualification.
NATIVE_COMPATIBLE_PINS = ("32d373cb584783046349f53f0343f518cad56d6c", PIN)
DEPENDENCIES = ("numpy", "pin", "hpp-fcl", "eigenpy", "cmeel-boost", "pybind11")
BUILD_INPUTS = ("setup.py", "pyproject.toml", "src/policy_sim/control/collision_kernel.cpp")


def run(*args: str) -> str:
    return subprocess.check_output(args, text=True).strip()


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def revalidate_installed(source):
    """Qualify the existing, already-audited binary against unchanged inputs.

    Preserve prior provenance instead of asserting that the binary was rebuilt
    from the new Python-controller revision. No dependency or library is replaced.
    """
    import _policy_sim_collision_kernel as native

    record_path = Path(sys.prefix) / 'policy_native_provenance.json'
    previous = json.loads(record_path.read_text())
    prior_revision = previous['policy_source_pin']
    if prior_revision not in NATIVE_COMPATIBLE_PINS:
        raise RuntimeError('Unqualified prior controller source revision')
    if run('git', '-C', str(source), 'rev-parse', 'HEAD') != PIN:
        raise RuntimeError('Nested source must match the deployment pin')
    subprocess.run(['git', '-C', str(source), 'diff', '--quiet', 'HEAD', '--'], check=True)
    installed = Path(native.__file__).resolve()
    site = Path(sysconfig.get_paths()['purelib']).resolve()
    if (installed.parent != site or Path(previous['installed_extension']).resolve() != installed
            or sha(installed) != previous['installed_extension_sha256']
            or previous['soabi'] != sysconfig.get_config_var('SOABI')
            or native.KERNEL_ABI_VERSION != previous['native_abi'] or native.KERNEL_ABI_VERSION != 3
            or native.TRAJECTORY_KERNEL_VERSION != previous['trajectory_version']):
        raise RuntimeError('Existing native artifact/ABI no longer matches its provenance')
    for relative in BUILD_INPUTS:
        prior_content = subprocess.check_output(['git', '-C', str(source), 'show', f'{prior_revision}:{relative}'])
        digest = sha(source / relative)
        if (hashlib.sha256(prior_content).hexdigest() != digest
                or previous['source_hashes'].get(relative) != digest):
            raise RuntimeError(f'Native build input changed: {relative}')
    for name in DEPENDENCIES:
        if metadata.version(name) != previous['dependency_versions'][name]:
            raise RuntimeError(f'Dependency identity changed: {name}')
    libraries = previous['native_library_sha256']
    if not libraries or 'libhpp-fcl.so' not in libraries:
        raise RuntimeError('Missing native library provenance')
    for name, digest in libraries.items():
        if sha(site / 'cmeel.prefix/lib' / name) != digest:
            raise RuntimeError(f'Native library changed: {name}')
    linked = run('ldd', str(installed))
    if 'not found' in linked:
        raise RuntimeError('Native artifact has unresolved libraries')
    backup = record_path.with_name(f'policy_native_provenance.{prior_revision[:7]}.json')
    if not backup.exists():
        shutil.copy2(record_path, backup)
    record = dict(previous, method='verified_existing_artifact_requalification_not_rebuild',
        policy_source_pin=PIN, policy_source=str(source), linked_libraries=linked,
        source_artifact_revision=previous.get('source_artifact_revision', prior_revision),
        prior_qualification=dict(policy_source_pin=prior_revision, manifest=str(backup),
                                 manifest_sha256=sha(backup)),
        native_build_inputs_equal_revisions=[prior_revision, PIN],
        limitation='Existing native artifact retained: build inputs, ABI and dependency/library bytes match; no rebuild performed.')
    record_path.write_text(json.dumps(record, indent=2)+'\n')
    print(record_path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--reuse-from-python", type=Path)
    mode.add_argument("--revalidate-installed", action='store_true')
    args = parser.parse_args()
    target_source = Path(__file__).resolve().parents[1] / "third_party/policy-sim"
    if args.revalidate_installed:
        revalidate_installed(target_source)
        return
    probe = """
import importlib.metadata as m, json, sys, sysconfig
import _policy_sim_collision_kernel as k
print(json.dumps(dict(extension=k.__file__, abi=k.KERNEL_ABI_VERSION,
    trajectory_version=k.TRAJECTORY_KERNEL_VERSION,
    soabi=sysconfig.get_config_var('SOABI'),
    purelib=sysconfig.get_paths()['purelib'],
    versions={name:m.version(name) for name in %r})))
""" % (DEPENDENCIES,)
    old = json.loads(run(str(args.reuse_from_python), "-c", probe))
    original = Path(old["extension"]).resolve()
    original_source = original.parent.parent
    for source in (original_source, target_source):
        revision = run("git", "-C", str(source), "rev-parse", "HEAD")
        if revision not in (NATIVE_COMPATIBLE_PINS if source == original_source else (PIN,)):
            raise RuntimeError(f"Unqualified policy-sim source revision: {source}")
        subprocess.run(
            ["git", "-C", str(source), "diff", "--quiet", "HEAD", "--"], check=True
        )
    source_hashes = {}
    for relative in BUILD_INPUTS:
        source_hashes[relative] = sha(target_source / relative)
        if sha(original_source / relative) != source_hashes[relative]:
            raise RuntimeError(f"Source identity mismatch: {relative}")
    if old["soabi"] != sysconfig.get_config_var("SOABI") or old["abi"] != 3:
        raise RuntimeError("Python/native ABI mismatch")
    for name, version in old["versions"].items():
        if metadata.version(name) != version:
            raise RuntimeError(f"Dependency identity mismatch: {name}")
    site = Path(sysconfig.get_paths()["purelib"])
    old_lib = Path(old["purelib"]) / "cmeel.prefix/lib"
    new_lib = site / "cmeel.prefix/lib"
    library_hashes = {}
    for original_lib in sorted(old_lib.glob("*.so*")):
        if not original_lib.is_file():
            continue
        destination_lib = new_lib / original_lib.name
        digest = sha(original_lib)
        if not destination_lib.is_file() or sha(destination_lib) != digest:
            raise RuntimeError(f"Native library identity mismatch: {original_lib.name}")
        library_hashes[original_lib.name] = digest
    if not library_hashes or "libhpp-fcl.so" not in library_hashes:
        raise RuntimeError("Missing HPP-FCL library identity")
    patchelf = Path(sys.prefix) / "bin/patchelf"
    if not patchelf.is_file():
        raise RuntimeError("Install patchelf==0.17.2.4 in the destination environment first")
    destination = site / original.name
    with tempfile.TemporaryDirectory(prefix="policy-native-") as directory:
        staged = Path(directory) / original.name
        shutil.copy2(original, staged)
        subprocess.run(
            [str(patchelf), "--set-rpath", "$ORIGIN/cmeel.prefix/lib", str(staged)], check=True
        )
        shutil.copy2(staged, destination)
    # The pure Python code is always imported from the nested pinned checkout.
    (site / "hexfellow_policy_source.pth").write_text(str(target_source / "src") + "\n")
    new = json.loads(run(sys.executable, "-c", probe))
    if Path(new["extension"]).resolve() != destination.resolve() or new["abi"] != old["abi"]:
        raise RuntimeError("Relocated native import failed identity verification")
    linked = run("ldd", str(destination))
    if "not found" in linked or str(Path(old["purelib"]).parent.parent) in linked:
        raise RuntimeError("Relocated extension retains missing or old-environment libraries")
    report = dict(
        method="verified_existing_artifact_reuse_not_rebuild", policy_source_pin=PIN,
        source_artifact_revision=run('git', '-C', str(original_source), 'rev-parse', 'HEAD'),
        policy_source=str(target_source), source_hashes=source_hashes,
        source_extension=str(original), source_extension_sha256=sha(original),
        installed_extension=str(destination), installed_extension_sha256=sha(destination),
        soabi=new["soabi"], native_abi=new["abi"],
        trajectory_version=new["trajectory_version"], dependency_versions=new["versions"],
        native_library_sha256=library_hashes, linked_libraries=linked,
        limitation="Source/ABI/library consistency verified; original build was not reproduced.",
    )
    report_path = Path(sys.prefix) / "policy_native_provenance.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    print(report_path)


if __name__ == "__main__":
    main()
