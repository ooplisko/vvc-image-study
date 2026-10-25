"""Prepare pinned Windows codec builds and portable bundles."""

from __future__ import annotations

from pathlib import Path
import os
import hashlib
import shutil
import subprocess

from .artifacts import read_json, sha256, write_json


def _run(args: list[str], *, cwd: Path | None = None) -> str:
    result = subprocess.run(args, cwd=cwd, capture_output=True, text=True,
                            encoding="utf-8", errors="replace")
    if result.returncode:
        raise RuntimeError(f"Command failed ({result.returncode}): {args[0]}\n"
                           + (result.stdout + "\n" + result.stderr)[-6000:])
    return result.stdout.strip()


def _lock(root: Path) -> dict:
    return read_json(root / "manifests" / "codecs.lock.json")


def compile_targets(build_dir: Path, targets: list[str], parallel: int) -> None:
    """Show native build progress and retain diagnostics for a resumed session."""
    command = ["cmake", "--build", str(build_dir), "--target", *targets,
               "--config", "Release", "--parallel", str(parallel)]
    log_path = build_dir / "build.log"
    with log_path.open("w", encoding="utf-8") as log:
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                   text=True, encoding="utf-8", errors="replace")
        for line in process.stdout:
            log.write(line)
            log.flush()
            print(line, end="", flush=True)
        if process.wait():
            raise RuntimeError(f"Codec compilation failed; see {log_path}")


def _verify_programs(record: dict) -> None:
    for kind in ("encoder", "decoder"):
        if sha256(Path(record[kind])) != record[kind + "_sha256"]:
            raise ValueError(f"Prepared {kind} changed; prepare codecs again.")


def toolchain() -> dict:
    if os.name != "nt":
        raise RuntimeError("Only Windows x64 is supported.")
    tools = {name: shutil.which(name) for name in ("git", "cmake", "g++", "mingw32-make")}
    missing = [name for name, path in tools.items() if path is None]
    if missing:
        raise RuntimeError("Missing build prerequisites on PATH: " + ", ".join(missing))
    versions = {name: _run([path, "--version"]).splitlines()[0] for name, path in tools.items()}
    gcc = _run([tools["g++"], "-dumpfullversion"])
    if gcc != "13.2.0":
        raise RuntimeError(f"The pinned Windows build requires GCC 13.2.0; found {gcc}.")
    return {"paths": tools, "versions": versions, "gcc": gcc}


def _checkout(root: Path, name: str, item: dict) -> Path:
    source = root / ".cache" / "sources" / name
    source.mkdir(parents=True, exist_ok=True)
    marker = source / ".prepared.json"
    wanted = {"source_commit": item["source_commit"], "patch_sha256": item["patch_sha256"]}
    if marker.exists():
        saved = read_json(marker)
        if any(saved.get(key) != value for key, value in wanted.items()):
            raise ValueError(f"Different source preparation already exists: {source}")
        if _run(["git", "rev-parse", "HEAD"], cwd=source) != item["source_commit"]:
            raise ValueError("Pinned source commit changed.")
        diff = subprocess.run(["git", "diff", "--binary", "HEAD"], cwd=source,
                              check=True, capture_output=True).stdout
        if hashlib.sha256(diff).hexdigest() != saved.get("diff_sha256"):
            raise ValueError("Prepared codec source was modified.")
        return source
    _run(["git", "init", "--quiet"], cwd=source)
    _run(["git", "fetch", "--depth", "1", item["source_url"], item["source_commit"]], cwd=source)
    _run(["git", "checkout", "--detach", "FETCH_HEAD"], cwd=source)
    if _run(["git", "rev-parse", "HEAD"], cwd=source) != item["source_commit"]:
        raise ValueError("Fetched source does not match the locked commit.")
    if item["patch"]:
        patch = root / item["patch"]
        if sha256(patch) != item["patch_sha256"]:
            raise ValueError("Codec patch checksum changed.")
        _run(["git", "apply", "--check", str(patch)], cwd=source)
        _run(["git", "apply", str(patch)], cwd=source)
    diff = subprocess.run(["git", "diff", "--binary", "HEAD"], cwd=source,
                          check=True, capture_output=True).stdout
    write_json(marker, {**wanted, "diff_sha256": hashlib.sha256(diff).hexdigest()})
    return source


def _build_vtm(root: Path, name: str, item: dict, compiler: dict, parallel: int) -> dict:
    source = _checkout(root, name, item)
    build_dir = root / ".build" / name
    output = build_dir / "programs"
    flags = ["-G", "MinGW Makefiles", "-DCMAKE_BUILD_TYPE=Release",
             "-DCMAKE_POLICY_VERSION_MINIMUM=3.5",
             "-DCMAKE_CXX_COMPILER=" + Path(compiler["paths"]["g++"]).as_posix(),
             "-DENABLE_TRACING=" + ("ON" if item["trace"] else "OFF")]
    print(f"Building {name} ({parallel} compiler jobs)", flush=True)
    _run(["cmake", "-S", str(source), "-B", str(build_dir), *flags])
    compile_targets(build_dir, ["EncoderApp", "DecoderApp"], parallel)
    # The target directories differ between upstream CMake versions. Reject ambiguity.
    output.mkdir(parents=True, exist_ok=True)
    record = {"source_commit": item["source_commit"], "patch_sha256": item["patch_sha256"],
              "toolchain": compiler, "cmake_flags": flags, "trace": item["trace"],
              "build_method": "source"}
    shutil.copy2(source / "COPYING", output / "COPYING")
    for kind, filename in (("encoder", "EncoderApp.exe"), ("decoder", "DecoderApp.exe")):
        candidates = [p for tree in (source / "bin", build_dir) if tree.exists()
                      for p in tree.rglob(filename) if output not in p.parents]
        if len(candidates) != 1:
            raise RuntimeError(f"Expected one fresh {filename}, found {len(candidates)}: {candidates}")
        target = output / filename
        shutil.copy2(candidates[0], target)
        record[kind], record[kind + "_sha256"] = str(target), sha256(target)
    # CSF streams use the clean base decoder, even if this build also produced a decoder.
    if item["csf"]:
        baseline = prepare_codecs(root, names=["vtm23-baseline"], parallel=parallel)["codecs"]["vtm23-baseline"]
        record["decoder"], record["decoder_sha256"] = baseline["decoder"], baseline["decoder_sha256"]
    record["build_manifest"] = str(build_dir / "build_manifest.json")
    record["build_log_sha256"] = sha256(build_dir / "build.log")
    write_json(Path(record["build_manifest"]), record)
    return record


def prepare_codecs(root: Path, *, mode: str = "source", names: list[str] | None = None,
                   parallel: int = 4) -> dict:
    if os.name != "nt":
        raise RuntimeError("Only Windows x64 is supported.")
    lock = _lock(root)
    if mode == "bundle":
        from .codec_bundle import prepared_bundle
        from .codec_validation import CODECS
        return prepared_bundle(root, names or list(CODECS))
    names = names or list(lock["codecs"])
    if set(names) - lock["codecs"].keys():
        raise ValueError("Unknown codec selection.")
    if mode != "source":
        raise ValueError("Codec mode must be source or bundle.")
    if parallel < 1:
        raise ValueError("Positive build parallelism required.")
    records = {}
    compiler = None
    for name in names:
        manifest = root / ".build" / name / "build_manifest.json"
        if manifest.exists():
            record = read_json(manifest)
            if record["source_commit"] != lock["codecs"][name]["source_commit"] or record["patch_sha256"] != lock["codecs"][name]["patch_sha256"]:
                raise ValueError("Build manifest belongs to a different source recipe.")
            if (lock["codecs"][name]["family"] == "vvenc"
                    and record.get("decoder_source_commit") != lock["vvdec"]["source_commit"]):
                raise ValueError("Build manifest belongs to a different decoder source recipe.")
            _verify_programs(record)
        else:
            compiler = compiler or toolchain()
            if lock["codecs"][name]["family"] == "vvenc":
                from .codec_source import build_vvenc
                record = build_vvenc(root, name, lock["codecs"][name], compiler, parallel)
            else:
                record = _build_vtm(root, name, lock["codecs"][name], compiler, parallel)
        from .codec_bundle import RUNTIMES
        runtime_dir = Path(record["toolchain"]["paths"]["g++"]).parent
        record["runtime_hashes"] = {dll: sha256(runtime_dir / dll) for dll in RUNTIMES}
        records[name] = record
    return {"platform": "windows-x86_64", "mode": mode, "codecs": records}
