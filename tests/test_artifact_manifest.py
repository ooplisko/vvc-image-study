import pytest

from vvcstudy.artifacts import file_record, read_json, verify_artifact_manifest, write_json
from vvcstudy.cli import _parser, result_exit_code


def bundle(folder):
    path = folder / "stream.vvc"
    path.write_bytes(b"abcd")
    write_json(folder / "manifest.json", {"schema_version": 1, "files": [file_record(path, folder)]})
    return path


def test_bundle_integrity_and_same_length_corruption(tmp_path):
    path = bundle(tmp_path)
    result = verify_artifact_manifest(tmp_path)
    assert result["accepted"] and result["checked_files"] == 1 and result["checked_bytes"] == 4
    path.write_bytes(b"abce")
    result = verify_artifact_manifest(tmp_path)
    assert not result["accepted"] and "checksum" in result["errors"][0]
    assert result_exit_code({"accepted": True, "artifacts": result}) == 1


@pytest.mark.parametrize("case", ["missing_file", "empty", "duplicate", "outside", "absolute"])
def test_incomplete_or_invalid_manifest_is_rejected(tmp_path, case):
    path = bundle(tmp_path)
    manifest = read_json(tmp_path / "manifest.json")
    if case == "missing_file":
        path.unlink()
    elif case == "empty":
        manifest["files"] = []
    elif case == "duplicate":
        manifest["files"] *= 2
    elif case == "outside":
        manifest["files"][0]["path"] = "../stream.vvc"
    else:
        manifest["files"][0]["path"] = str(path.resolve())
    write_json(tmp_path / "manifest.json", manifest)
    assert not verify_artifact_manifest(tmp_path)["accepted"]


def test_missing_manifest_and_cli_option(tmp_path):
    assert not verify_artifact_manifest(tmp_path)["accepted"]
    assert _parser().parse_args(["verify", "--artifacts"]).artifacts
