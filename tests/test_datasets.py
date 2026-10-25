import hashlib
import io
import json
import zipfile
from pathlib import Path

import pytest
from PIL import Image

from vvcstudy import datasets


def _png(color=(12, 34, 56)):
    stream = io.BytesIO()
    Image.new("RGB", (7, 5), color).save(stream, format="PNG")
    return stream.getvalue()


def _manifest(root, name="kodak"):
    content = _png()
    ids = [f"kodim{i:02d}" for i in range(1, 25)] if name == "kodak" else [f"{i:04d}" for i in range(801, 901)]
    entries = [{"id": image_id, "sha256": hashlib.sha256(content).hexdigest(), "width": 7, "height": 5,
                "url": f"https://example.invalid/{image_id}.png", "archive_member": f"DIV2K_valid_HR/{image_id}.png"}
               for image_id in ids]
    path = root / "manifests/datasets" / f"{name}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"images": entries}), encoding="utf-8")
    return content, entries, path


def test_existing_originals_are_verified_without_network_or_changes(tmp_path, monkeypatch):
    content, entries, _ = _manifest(tmp_path)
    destination = tmp_path / "data/originals/kodak"
    destination.mkdir(parents=True)
    for entry in entries:
        (destination / f"{entry['id']}.png").write_bytes(content)
    before = {p.name: p.stat().st_mtime_ns for p in destination.iterdir()}
    monkeypatch.setattr(datasets.urllib.request, "urlopen", lambda *a, **k: pytest.fail("Unexpected download"))
    result = datasets.prepare_datasets(tmp_path, ["kodak"])
    assert len(result["kodak"]) == 24
    assert result["kodak"][0]["width"] == 7
    assert {p.name: p.stat().st_mtime_ns for p in destination.iterdir()} == before
    assert all(Path(row["path"]).read_bytes() == content for row in result["kodak"])


def test_changed_original_is_rejected_without_overwriting(tmp_path, monkeypatch):
    _, _, _ = _manifest(tmp_path)
    path = tmp_path / "data/originals/kodak/kodim01.png"
    path.parent.mkdir(parents=True)
    changed = _png((100, 0, 0))
    path.write_bytes(changed)
    monkeypatch.setattr(datasets.urllib.request, "urlopen", lambda *a, **k: pytest.fail("Unexpected download"))
    with pytest.raises(ValueError, match="checksum mismatch"):
        datasets.prepare_datasets(tmp_path, ["kodak"])
    assert path.read_bytes() == changed


def test_kodak_download_is_atomic_and_byte_exact(tmp_path, monkeypatch):
    content, _, _ = _manifest(tmp_path)
    monkeypatch.setattr(datasets.urllib.request, "urlopen", lambda *a, **k: io.BytesIO(content))
    result = datasets.prepare_datasets(tmp_path, ["kodak"])
    assert len(result["kodak"]) == 24
    assert all(Path(row["path"]).read_bytes() == content for row in result["kodak"])
    assert not list((tmp_path / "data/originals").rglob("*.partial"))


def test_bad_download_never_becomes_original(tmp_path, monkeypatch):
    _manifest(tmp_path)
    monkeypatch.setattr(datasets.urllib.request, "urlopen", lambda *a, **k: io.BytesIO(b"corrupt"))
    with pytest.raises(ValueError, match="Download checksum mismatch"):
        datasets.prepare_datasets(tmp_path, ["kodak"])
    assert not (tmp_path / "data/originals/kodak/kodim01.png").exists()


def test_div2k_extracts_only_exact_original_members(tmp_path, monkeypatch):
    content, entries, path = _manifest(tmp_path, "div2k")
    stream = io.BytesIO()
    with zipfile.ZipFile(stream, "w") as bundle:
        for entry in entries:
            bundle.writestr(entry["archive_member"], content)
        bundle.writestr("../../outside.txt", b"must not extract")
    archive = stream.getvalue()
    path.write_text(json.dumps({"images": entries, "archive": {"filename": "test.zip", "url": "https://example.invalid/test.zip",
                    "sha256": hashlib.sha256(archive).hexdigest()}}), encoding="utf-8")
    monkeypatch.setattr(datasets.urllib.request, "urlopen", lambda *a, **k: io.BytesIO(archive))
    result = datasets.prepare_datasets(tmp_path, ["div2k"])
    assert len(result["div2k"]) == 100
    assert all(Path(row["path"]).read_bytes() == content for row in result["div2k"])
    assert not list(tmp_path.rglob("outside.txt"))


def test_manifest_cannot_silently_select_a_subset(tmp_path):
    _, entries, path = _manifest(tmp_path)
    path.write_text(json.dumps({"images": entries[:3]}), encoding="utf-8")
    with pytest.raises(ValueError, match="exact ordered"):
        datasets.prepare_datasets(tmp_path, ["kodak"])


def test_dataset_names_are_explicit(tmp_path):
    with pytest.raises(ValueError, match="Unknown datasets"):
        datasets.prepare_datasets(tmp_path, ["div2k48"])


def test_committed_manifests_cover_full_original_collections():
    root = Path(__file__).resolve().parents[1]
    for name, count in (("kodak", 24), ("div2k", 100)):
        data = json.loads((root / "manifests/datasets" / f"{name}.json").read_text("utf-8"))
        assert len(data["images"]) == count
        assert len({row["sha256"] for row in data["images"]}) == count
        assert all(len(row["sha256"]) == 64 and row["width"] > 0 and row["height"] > 0 for row in data["images"])
    assert all(row["width"] > 512 and row["height"] > 512 for row in data["images"])
