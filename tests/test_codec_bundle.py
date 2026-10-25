import json
from pathlib import Path
import shutil
import zipfile

import pytest

from vvcstudy.artifacts import identity, read_json, sha256, write_json
from vvcstudy.codec_bundle import LOCKED_FILES, RUNTIMES, import_bundle, package_codecs, prepared_bundle
from vvcstudy.codec_validation import CODECS


@pytest.fixture
def bundled(tmp_path):
    root = tmp_path / "repo"
    real = Path(__file__).resolve().parents[1]
    for relative in LOCKED_FILES:
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(real / relative, path)
    lock = read_json(root / LOCKED_FILES[0])
    runtime = tmp_path / "compiler/bin"
    runtime.mkdir(parents=True)
    for dll in RUNTIMES:
        (runtime / dll).write_bytes(dll.encode())
    for component in ("gcc", "winpthreads", "mingw-w64"):
        folder = runtime.parent / "licenses" / component
        folder.mkdir(parents=True)
        (folder / "COPYING").write_text("fixture notice")
    records = {}
    decoder_name = "vvdec-" + lock["vvdec"]["source_commit"][:12]
    for name in (*CODECS, "vtm23-baseline", decoder_name):
        folder = root / ".build" / name
        folder.mkdir(parents=True)
        encoder, decoder = folder / "encoder.exe", folder / "decoder.exe"
        encoder.write_bytes((name + "encoder").encode())
        decoder.write_bytes((name + "decoder").encode())
        item = lock["codecs"].get(name, {"source_commit": lock["vvdec"]["source_commit"],
                                       "patch_sha256": None, "trace": False})
        record = {field: item[field] for field in ("source_commit", "patch_sha256", "trace")}
        record.update(encoder=str(encoder), decoder=str(decoder), encoder_sha256=sha256(encoder),
                      decoder_sha256=sha256(decoder), cmake_flags=[f"-DCMAKE_CXX_COMPILER={runtime / 'g++.exe'}"],
                      toolchain={"paths": {"g++": str(runtime / "g++.exe")}, "versions": {"g++": "13.2.0"}})
        records[name] = record
        source = root / ".cache/sources" / name
        source.mkdir(parents=True)
        (source / "COPYING").write_text("fixture notice")
    for name in CODECS:
        decoder = records[decoder_name if name.startswith("vvenc") else "vtm23-baseline" if "csf" in name else name]
        records[name].update(decoder=decoder["decoder"], decoder_sha256=decoder["decoder_sha256"])
    for name, record in records.items():
        write_json(root / ".build" / name / "build_manifest.json", record)
    decoder = records[decoder_name]
    write_json(root / ".build" / decoder_name / "program_manifest.json",
               {**decoder, "program_sha256": decoder["decoder_sha256"]})
    control = {"accepted": True, "controls": 10, "errors": [],
               "expected_fixture_sha256": sha256(root / LOCKED_FILES[2]),
               "config_sha256": sha256(root / LOCKED_FILES[1]),
               "codec_hashes": {name: records[name]["encoder_sha256"] for name in CODECS},
               "decoder_hashes": {name: records[name]["decoder_sha256"] for name in CODECS},
               "runtime_hashes": {name: {dll: sha256(runtime / dll) for dll in RUNTIMES} for name in CODECS}}
    key = identity({name: {kind: records[name][kind + "_sha256"] for kind in ("encoder", "decoder")}
                    for name in CODECS})[:16]
    control_path = root / "runs/codec_checks" / key / "comparison.json"
    write_json(control_path, control)
    result = package_codecs(root)
    return root, Path(result["archive"]), result["sha256"], control_path


def test_bundle_roundtrip_and_runtime_corruption(bundled):
    root, archive, checksum, _ = bundled
    assert import_bundle(root, archive, checksum)["imported"]
    assert import_bundle(root, archive, checksum)["imported"]
    value = prepared_bundle(root, list(CODECS))
    assert value["mode"] == "bundle"
    for codec in value["codecs"].values():
        assert Path(codec["encoder"]).is_file()
        assert len(codec["runtime_hashes"]) == 3
    with zipfile.ZipFile(archive) as zipped:
        assert str(root).replace("\\", "\\\\") not in zipped.read("bundle.json").decode()
    codec = value["codecs"][CODECS[0]]
    (Path(codec["encoder"]).parent / RUNTIMES[0]).write_bytes(b"changed")
    with pytest.raises(ValueError, match="checksum"):
        prepared_bundle(root, list(CODECS))


def test_bundle_requires_matching_archive_hash_and_protocol(bundled):
    root, archive, checksum, _ = bundled
    with pytest.raises(ValueError, match="Archive SHA256"):
        import_bundle(root, archive, "0" * 64)
    (root / LOCKED_FILES[1]).write_text("different protocol")
    with pytest.raises(ValueError, match="different source lock"):
        import_bundle(root, archive, checksum)
    assert not (root / ".cache/bundle_codecs.json").exists()


def test_packaging_rejects_failed_qualification(bundled):
    root, _, _, control_path = bundled
    control = read_json(control_path)
    control["accepted"] = False
    write_json(control_path, control)
    with pytest.raises(ValueError, match="ten-control"):
        package_codecs(root)


@pytest.mark.parametrize("extra", ["../outside.exe", "C:/outside.exe", "codecs\\bad.exe", "NUL.exe", "CODECS.LOCK.JSON"])
def test_archive_rejects_unsafe_or_unlisted_members(bundled, extra):
    root, archive, _, _ = bundled
    with zipfile.ZipFile(archive, "a") as zipped:
        zipped.writestr(extra, b"bad")
    with pytest.raises(ValueError, match="Unsafe|Unlisted"):
        import_bundle(root, archive, sha256(archive))


def test_corrupt_payload_is_rejected_before_registration(bundled):
    root, archive, _, _ = bundled
    with zipfile.ZipFile(archive) as zipped:
        payload = {name: zipped.read(name) for name in zipped.namelist()}
    name = next(name for name in payload if name.endswith(".exe"))
    payload[name] = b"X" * len(payload[name])
    with zipfile.ZipFile(archive, "w") as zipped:
        for name, data in payload.items():
            zipped.writestr(name, data)
    with pytest.raises(ValueError, match="checksum"):
        import_bundle(root, archive, sha256(archive))
    assert not (root / ".cache/bundle_codecs.json").exists()


def test_case_aliases_are_rejected(bundled):
    root, archive, _, _ = bundled
    with zipfile.ZipFile(archive, "a") as zipped:
        zipped.writestr("BUNDLE.JSON", json.dumps({}))
    with pytest.raises(ValueError, match="Duplicate"):
        import_bundle(root, archive, sha256(archive))
