import subprocess

import pytest

from vvcstudy.artifacts import sha256, write_json
from vvcstudy.build import _checkout, _verify_programs, prepare_codecs


def test_prepared_source_rejects_later_tracked_edits(tmp_path):
    upstream = tmp_path / "upstream"
    upstream.mkdir()
    subprocess.run(["git", "init", "-q", str(upstream)], check=True)
    (upstream / "source.txt").write_text("fixed source\n")
    subprocess.run(["git", "add", "source.txt"], cwd=upstream, check=True)
    subprocess.run(["git", "-c", "user.name=Fixture", "-c", "user.email=fixture@example.invalid",
                    "commit", "-qm", "Test source"], cwd=upstream, check=True)
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=upstream, text=True).strip()
    item = {"source_url": str(upstream), "source_commit": commit, "patch": None, "patch_sha256": None}
    source = _checkout(tmp_path, "fixture", item)
    assert _checkout(tmp_path, "fixture", item) == source
    (source / "source.txt").write_text("modified source\n")
    with pytest.raises(ValueError, match="modified"):
        _checkout(tmp_path, "fixture", item)


def test_prepared_executables_reject_hash_changes(tmp_path):
    binary = tmp_path / "program.exe"
    binary.write_bytes(b"fixture executable")
    checksum = sha256(binary)
    record = {"encoder": str(binary), "decoder": str(binary),
              "encoder_sha256": checksum, "decoder_sha256": checksum}
    _verify_programs(record)
    binary.write_bytes(b"changed executable")
    with pytest.raises(ValueError, match="changed"):
        _verify_programs(record)


def test_cached_encoder_cannot_silently_reuse_a_different_decoder_recipe(tmp_path):
    item = {"family": "vvenc", "source_commit": "encoder-source", "patch_sha256": None}
    write_json(tmp_path / "manifests/codecs.lock.json", {
        "codecs": {"fixture": item}, "vvdec": {"source_commit": "new-decoder-source"}})
    write_json(tmp_path / ".build/fixture/build_manifest.json", {
        "source_commit": item["source_commit"], "patch_sha256": None,
        "decoder_source_commit": "old-decoder-source"})
    with pytest.raises(ValueError, match="decoder source recipe"):
        prepare_codecs(tmp_path, names=["fixture"])
