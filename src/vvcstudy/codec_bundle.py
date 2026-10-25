"""Local, checksum-verified Windows codec archives; no download or publication."""

from __future__ import annotations

import json
from pathlib import Path, PurePosixPath
import shutil
import tempfile
import zipfile

from .artifacts import identity, read_json, sha256, write_json
from .codec_validation import CODECS


RUNTIMES = ("libgcc_s_seh-1.dll", "libstdc++-6.dll", "libwinpthread-1.dll")
LOCKED_FILES = ("manifests/codecs.lock.json", "configs/codecs/vtm_intra.cfg",
                "tests/fixtures/codecs/expected.json", "patches/vtm-csf.patch")


def _safe_name(name: str) -> str:
    parts = name.split("/")
    if (not name or "\\" in name or ":" in name or any(p in ("", ".", "..") for p in parts)
            or any(p.endswith((".", " ")) or any(ord(c) < 32 for c in p) for p in parts)
            or PurePosixPath(name).is_absolute()
            or any(p.split(".")[0].upper() in {"CON", "PRN", "AUX", "NUL", *(
                f"{prefix}{i}" for prefix in ("COM", "LPT") for i in range(1, 10))} for p in parts)):
        raise ValueError(f"Unsafe bundle path: {name!r}")
    return name


def _qualified(root: Path, codecs: dict, control: dict) -> None:
    if (control.get("accepted") is not True or control.get("controls") != 10
            or control.get("errors") != []
            or control.get("expected_fixture_sha256") != sha256(root / LOCKED_FILES[2])
            or control.get("config_sha256") != sha256(root / LOCKED_FILES[1])):
        raise ValueError("Bundle requires the accepted ten-control comparison for this protocol.")
    for name in CODECS:
        for kind, field in (("encoder", "codec_hashes"), ("decoder", "decoder_hashes")):
            if control.get(field, {}).get(name) != codecs[name][kind + "_sha256"]:
                raise ValueError("Codec qualification belongs to different executables.")


def _portable_recipe(record: dict) -> dict:
    # Compiler installation directories are machine-local, not portable recipe fields.
    flags = [flag.split("=", 1)[0] + "=" + Path(flag.split("=", 1)[1]).name
             if flag.startswith(("-DCMAKE_CXX_COMPILER=", "-DCMAKE_C_COMPILER=")) else flag
             for flag in record["cmake_flags"]]
    return {"cmake_flags": flags, "toolchain_versions": record["toolchain"]["versions"],
            **{key: record[key] for key in ("source_commit", "source_tree", "patch_sha256") if key in record}}


def package_codecs(root: Path, output: Path | None = None) -> dict:
    """Package existing qualified builds only; never start compilation here."""
    from .build import _verify_programs

    records = {name: read_json(root / ".build" / name / "build_manifest.json") for name in CODECS}
    lock = read_json(root / LOCKED_FILES[0])
    files = {path: root / path for path in LOCKED_FILES}
    key = identity({name: {kind: record[kind + "_sha256"] for kind in ("encoder", "decoder")}
                    for name, record in records.items()})[:16]
    control = read_json(root / "runs/codec_checks" / key / "comparison.json")
    _qualified(root, records, control)
    codecs = {}
    for name, record in records.items():
        _verify_programs(record)
        item = lock["codecs"][name]
        if any(record[field] != item[field] for field in ("source_commit", "patch_sha256", "trace")):
            raise ValueError("Build and source lock differ.")
        is_vvenc = item["family"] == "vvenc"
        decoder_name = ("vvdec-" + lock["vvdec"]["source_commit"][:12] if is_vvenc
                        else "vtm23-baseline" if item["csf"] else name)
        decoder = read_json(root / ".build" / decoder_name /
                            ("program_manifest.json" if is_vvenc else "build_manifest.json"))
        if record["decoder_sha256"] != decoder["program_sha256" if is_vvenc else "decoder_sha256"]:
            raise ValueError("Decoder build manifest differs from the qualified decoder.")
        if is_vvenc and decoder["source_commit"] != lock["vvdec"]["source_commit"]:
            raise ValueError("Decoder source lock changed.")
        entry = {"trace": item["trace"], "source_commit": item["source_commit"],
                 "patch_sha256": item["patch_sha256"], "encoder_build": _portable_recipe(record),
                 "decoder_build": _portable_recipe(decoder),
                 "build_method": "bundle"}
        runtime_dir = Path(record["toolchain"]["paths"]["g++"]).parent
        runtime_hashes = {dll: sha256(runtime_dir / dll) for dll in RUNTIMES}
        if control.get("runtime_hashes", {}).get(name) != runtime_hashes:
            raise ValueError("Runtime libraries differ from the qualified build.")
        for kind in ("encoder", "decoder"):
            source = Path(record[kind])
            relative = f"codecs/{name}/{source.name}"
            files[relative] = source
            entry[kind], entry[kind + "_sha256"] = relative, record[kind + "_sha256"]
        for dll in RUNTIMES:
            files[f"codecs/{name}/{dll}"] = runtime_dir / dll
        for source_name in {name, decoder_name}:
            source_dir = root / ".cache/sources" / source_name
            licenses = [source_dir / file for file in ("COPYING", "LICENSE.txt", "AUTHORS.md")
                        if (source_dir / file).is_file()]
            if not licenses:
                raise ValueError(f"Missing codec license: {source_name}")
            for path in licenses:
                files[f"licenses/{source_name}/{path.name}"] = path
        for component in ("gcc", "winpthreads", "mingw-w64"):
            folder = runtime_dir.parent / "licenses" / component
            notices = [path for path in folder.rglob("*") if path.is_file()]
            if not notices:
                raise ValueError(f"Missing runtime notices: {component}")
            for path in notices:
                files[f"licenses/runtime/{component}/{path.relative_to(folder).as_posix()}"] = path
        codecs[name] = entry
    manifest = {"schema_version": 1, "platform": "windows-x86_64", "codecs": codecs,
                "qualification": control, "locked_files": {path: sha256(root / path) for path in LOCKED_FILES},
                "files": {name: {"bytes": path.stat().st_size, "sha256": sha256(path)}
                          for name, path in sorted(files.items())},
                "scope": "Windows x64 encoders, decoders and runtime libraries. "
                         "FFmpeg and Windows system libraries are external prerequisites."}
    output = (output or root / "dist/vvc-codecs-windows-x64.zip").resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    partial = output.with_suffix(output.suffix + ".partial")
    with zipfile.ZipFile(partial, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name, path in sorted(files.items()):
            info = zipfile.ZipInfo(_safe_name(name))
            info.compress_type = zipfile.ZIP_DEFLATED
            archive.writestr(info, path.read_bytes())
        archive.writestr("bundle.json", json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    partial.replace(output)
    checksum = sha256(output)
    output.with_suffix(output.suffix + ".sha256").write_text(f"{checksum}  {output.name}\n", encoding="ascii")
    return {"archive": str(output), "sha256": checksum, "bytes": output.stat().st_size,
            "codecs": list(codecs), "controls": control["controls"], "published": False}


def _validate_metadata(root: Path, manifest: dict) -> None:
    if manifest["schema_version"] != 1 or manifest["platform"] != "windows-x86_64":
        raise ValueError("Unsupported codec bundle.")
    if set(manifest["codecs"]) != set(CODECS):
        raise ValueError("Incomplete codec bundle.")
    if manifest["locked_files"] != {path: sha256(root / path) for path in LOCKED_FILES}:
        raise ValueError("Bundle belongs to a different source lock or control protocol.")
    if any(manifest["files"][path]["sha256"] != checksum for path, checksum in manifest["locked_files"].items()):
        raise ValueError("Bundled protocol files differ from the source lock.")
    _qualified(root, manifest["codecs"], manifest["qualification"])
    lock = read_json(root / LOCKED_FILES[0])
    for name, codec in manifest["codecs"].items():
        item = lock["codecs"][name]
        if any(codec[field] != item[field] for field in ("source_commit", "patch_sha256", "trace")):
            raise ValueError("Bundle codec recipe differs from the source lock.")
        decoder_commit = (lock["vvdec"]["source_commit"]
                          if item["family"] == "vvenc" else item["source_commit"])
        if (codec["encoder_build"]["source_commit"] != item["source_commit"]
                or codec["decoder_build"]["source_commit"] != decoder_commit):
            raise ValueError("Bundle build provenance differs from the source lock.")
        for kind in ("encoder", "decoder"):
            path = _safe_name(codec[kind])
            if not path.startswith(f"codecs/{name}/") or manifest["files"][path]["sha256"] != codec[kind + "_sha256"]:
                raise ValueError("Bundle executable manifest is inconsistent.")
        for dll in RUNTIMES:
            if f"codecs/{name}/{dll}" not in manifest["files"]:
                raise ValueError("Missing bundled runtime library.")
        if manifest["qualification"].get("runtime_hashes", {}).get(name) != {
                dll: manifest["files"][f"codecs/{name}/{dll}"]["sha256"] for dll in RUNTIMES}:
            raise ValueError("Runtime library qualification differs from the bundle.")


def _verify_payload(folder: Path, manifest: dict) -> None:
    for name, expected in manifest["files"].items():
        path = (folder / _safe_name(name)).resolve()
        if (not path.is_relative_to(folder.resolve()) or not path.is_file()
                or path.stat().st_size != expected["bytes"] or sha256(path) != expected["sha256"]):
            raise ValueError(f"Bundle payload checksum mismatch: {name}")


def import_bundle(root: Path, archive: Path, expected_sha256: str) -> dict:
    if sha256(archive) != expected_sha256.lower():
        raise ValueError("Archive SHA256 mismatch; bundle was not imported.")
    target = root / ".cache/bundles" / expected_sha256.lower()
    target.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive) as zipped:
        names = [_safe_name(info.filename) for info in zipped.infolist()]
        if len(names) != len({name.casefold() for name in names}):
            raise ValueError("Duplicate archive paths.")
        manifest = json.loads(zipped.read("bundle.json"))
        _validate_metadata(root, manifest)
        if set(names) != {"bundle.json", *manifest["files"]}:
            raise ValueError("Unlisted or missing bundle files.")
        for name, expected in manifest["files"].items():
            if zipped.getinfo(name).file_size != expected["bytes"]:
                raise ValueError("Bundle file length mismatch.")
        if not target.exists():
            with tempfile.TemporaryDirectory(dir=target.parent, prefix=".import-") as temporary:
                staging = Path(temporary) / "payload"
                staging.mkdir()
                for name in names:
                    destination = staging / name
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    with zipped.open(name) as source, destination.open("wb") as output:
                        shutil.copyfileobj(source, output)
                _verify_payload(staging, manifest)
                staging.rename(target)
    _verify_payload(target, manifest)
    write_json(root / ".cache/bundle_codecs.json", {"archive_sha256": expected_sha256.lower(),
                                                    "manifest_sha256": sha256(target / "bundle.json")})
    return {"imported": True, "archive_sha256": expected_sha256.lower(), "codecs": list(manifest["codecs"])}


def prepared_bundle(root: Path, names: list[str]) -> dict:
    mapping = read_json(root / ".cache/bundle_codecs.json")
    key = mapping["archive_sha256"]
    if len(key) != 64 or any(c not in "0123456789abcdef" for c in key):
        raise ValueError("Invalid bundle cache identity.")
    folder = root / ".cache/bundles" / key
    if sha256(folder / "bundle.json") != mapping["manifest_sha256"]:
        raise ValueError("Imported bundle manifest changed.")
    manifest = read_json(folder / "bundle.json")
    _validate_metadata(root, manifest)
    _verify_payload(folder, manifest)
    records = {}
    for name in names:
        record = manifest["codecs"][name].copy()
        for kind in ("encoder", "decoder"):
            record[kind] = str(folder / record[kind])
        record["runtime_hashes"] = {dll: manifest["files"][f"codecs/{name}/{dll}"]["sha256"] for dll in RUNTIMES}
        records[name] = record
    return {"platform": "windows-x86_64", "mode": "bundle", "codecs": records}
