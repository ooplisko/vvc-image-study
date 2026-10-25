"""Build pinned VVenC encoders and the shared VVdeC decoder."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shutil

from .artifacts import read_json, sha256, write_json


def _fingerprint(value: dict) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode("utf-8")).hexdigest()


def _verify_source(source: Path, commit: str) -> str:
    """Reject edits that would otherwise survive a cached checkout marker."""
    from .build import _run

    if _run(["git", "rev-parse", "HEAD"], cwd=source) != commit:
        raise ValueError(f"Source commit changed: {source}")
    if _run(["git", "diff", "--name-only", "HEAD"], cwd=source):
        raise ValueError(f"Tracked codec source changed: {source}")
    extra = set(_run(["git", "ls-files", "--others", "--exclude-standard"], cwd=source).splitlines())
    extra.discard(".prepared.json")
    if extra:
        raise ValueError(f"Unexpected source files in {source}: {sorted(extra)}")
    return _run(["git", "rev-parse", "HEAD^{tree}"], cwd=source)


def _flags(prefix: str, trace: bool, compiler: dict) -> list[str]:
    cxx = Path(compiler["paths"]["g++"])
    args = ["-G", "MinGW Makefiles", "-DCMAKE_BUILD_TYPE=Release",
            "-DCMAKE_POLICY_VERSION_MINIMUM=3.5",
            f"-DCMAKE_CXX_COMPILER={cxx.as_posix()}",
            f"-DCMAKE_C_COMPILER={(cxx.parent / 'gcc.exe').as_posix()}",
            "-DBUILD_SHARED_LIBS=OFF", f"-D{prefix}_ENABLE_LINK_TIME_OPT=OFF",
            f"-D{prefix}_TOPLEVEL_OUTPUT_DIRS=OFF",
            f"-D{prefix}_ENABLE_TRACING={'ON' if trace else 'OFF'}"]
    if prefix == "VVENC":
        # Preserve the selected source version's floating-point contraction mode.
        args.append("-DVVENC_FFP_CONTRACT_OFF=OFF")
    return args


def _build_program(root: Path, name: str, item: dict, compiler: dict,
                   parallel: int, prefix: str, target: str) -> dict:
    from .build import _checkout, _run, compile_targets

    if parallel < 1:
        raise ValueError("Positive build parallelism required.")
    if item.get("patch") is not None:
        raise ValueError("VVenC/VVdeC builds require a pinned source commit without a separate patch.")
    source = _checkout(root, name, item)
    tree = _verify_source(source, item["source_commit"])
    flags = _flags(prefix, bool(item.get("trace", False)), compiler)
    recipe = {"source_url": item["source_url"], "source_commit": item["source_commit"],
              "source_tree": tree, "cmake_flags": flags, "toolchain": compiler,
              "target": target, "platform": "windows-x86_64"}
    recipe_hash = _fingerprint(recipe)
    build_dir = root / ".build" / name
    manifest_path = build_dir / "program_manifest.json"
    if manifest_path.exists():
        record = read_json(manifest_path)
        if record["recipe_sha256"] != recipe_hash:
            raise ValueError(f"Different codec build recipe already exists: {build_dir}")
        if sha256(Path(record["program"])) != record["program_sha256"]:
            raise ValueError(f"Built codec executable changed: {record['program']}")
        return record

    build_dir.mkdir(parents=True, exist_ok=True)
    recipe_path = build_dir / "recipe.json"
    if recipe_path.exists() and read_json(recipe_path) != recipe:
        raise ValueError(f"An interrupted build uses a different recipe: {build_dir}")
    write_json(recipe_path, recipe)
    print(f"Building {name} from {item['source_commit'][:12]} ({parallel} compiler jobs)", flush=True)
    _run(["cmake", "-S", str(source), "-B", str(build_dir), *flags])
    compile_targets(build_dir, [target], parallel)

    # TOPLEVEL_OUTPUT_DIRS=OFF gives a deterministic path in these pinned sources.
    program = build_dir / "bin" / f"{target}.exe"
    if not program.is_file():
        raise RuntimeError(f"Expected CMake target output is absent: {program}")
    license_path = build_dir / "LICENSE.txt"
    shutil.copy2(source / "LICENSE.txt", license_path)
    record = {**recipe, "recipe_sha256": recipe_hash, "program": str(program),
              "build_log_sha256": sha256(build_dir / "build.log"),
              "program_sha256": sha256(program), "license": str(license_path),
              "license_sha256": sha256(license_path),
              "build_method": "source"}
    write_json(manifest_path, record)
    return record


def build_vvenc(root: Path, name: str, item: dict, compiler: dict, parallel: int) -> dict:
    """Build one encoder variant and reuse a shared pinned VVdeC decoder."""
    lock = read_json(root / "manifests" / "codecs.lock.json")
    decoder_source = lock["vvdec"]
    commit = decoder_source.get("source_commit")
    if not commit or len(commit) != 40:
        raise RuntimeError("Pin the full VVdeC commit before building VVenC.")
    decoder_item = {"source_url": decoder_source["source_url"], "source_commit": commit,
                    "patch": None, "patch_sha256": None, "trace": False}
    decoder_name = "vvdec-" + commit[:12]
    decoder = _build_program(root, decoder_name, decoder_item, compiler, parallel,
                             "VVDEC", "vvdecapp")
    encoder = _build_program(root, name, item, compiler, parallel, "VVENC", "vvencFFapp")
    manifest_path = root / ".build" / name / "build_manifest.json"
    record = {"encoder": encoder["program"], "encoder_sha256": encoder["program_sha256"],
              "decoder": decoder["program"], "decoder_sha256": decoder["program_sha256"],
              "source_commit": item["source_commit"], "source_tree": encoder["source_tree"],
              "decoder_source_commit": commit, "decoder_source_tree": decoder["source_tree"],
              "patch_sha256": item.get("patch_sha256"), "trace": item["trace"],
              "cmake_flags": encoder["cmake_flags"], "toolchain": compiler,
              "encoder_recipe_sha256": encoder["recipe_sha256"],
              "decoder_recipe_sha256": decoder["recipe_sha256"],
              "build_method": "source",
              "build_manifest": str(manifest_path)}
    write_json(manifest_path, record)
    return record
