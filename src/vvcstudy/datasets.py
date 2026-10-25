"""Download and verify original PNG datasets without preprocessing."""

from __future__ import annotations

import hashlib
import json
import shutil
import urllib.request
import zipfile
from pathlib import Path

from PIL import Image


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _check_original(path: Path, entry: dict) -> None:
    if _sha256(path) != entry["sha256"]:
        raise ValueError(f"Original PNG checksum mismatch; file was not changed: {path}")
    with Image.open(path) as image:
        if image.format != "PNG" or image.mode != "RGB":
            raise ValueError(f"Expected original RGB PNG: {path}")
        if image.size != (entry["width"], entry["height"]):
            raise ValueError(f"Original PNG dimensions differ from manifest: {path}")
        image.verify()


def _download(url: str, target: Path, expected_sha256: str) -> None:
    """Publish a download only after validating it; never overwrite an original."""
    if target.exists():
        if _sha256(target) != expected_sha256:
            raise ValueError(f"Existing download has an unexpected checksum: {target}")
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_name(target.name + ".partial")
    request = urllib.request.Request(url, headers={"User-Agent": "vvc-image-study/1"})
    with urllib.request.urlopen(request, timeout=60) as response, partial.open("wb") as output:
        shutil.copyfileobj(response, output, length=1024 * 1024)
    if _sha256(partial) != expected_sha256:
        raise ValueError(f"Download checksum mismatch; original not installed: {url}")
    partial.replace(target)


def prepare_datasets(root: Path, names: list[str]) -> dict[str, list[dict]]:
    """Download/verify Kodak24 and DIV2K validation100 into data/originals.

    ``root`` is the repository root. Accepted names are ``kodak`` and ``div2k``.
    Already present originals are validated and reused without modification.
    Returned paths are absolute; all committed manifests contain relative paths.
    DIV2K extraction copies the original PNG byte streams without cropping,
    scaling, re-encoding, or padding. Codec padding belongs to the encoder stage.
    """
    root = Path(root).resolve()
    unknown = set(names) - {"kodak", "div2k"}
    if unknown:
        raise ValueError(f"Unknown datasets: {sorted(unknown)}")
    result: dict[str, list[dict]] = {}
    for name in dict.fromkeys(names):
        manifest = json.loads((root / "manifests" / "datasets" / f"{name}.json").read_text("utf-8"))
        entries = manifest["images"]
        expected_ids = ([f"kodim{i:02d}" for i in range(1, 25)] if name == "kodak"
                        else [f"{i:04d}" for i in range(801, 901)])
        if [entry["id"] for entry in entries] != expected_ids:
            raise ValueError(f"Dataset manifest must contain the exact ordered {name} collection")
        destination = root / "data" / "originals" / name
        destination.mkdir(parents=True, exist_ok=True)
        missing = []
        for entry in entries:
            path = destination / f"{entry['id']}.png"
            if path.exists():
                _check_original(path, entry)
            else:
                missing.append(entry)
        if missing and "archive" in manifest:
            archive = manifest["archive"]
            archive_path = root / "data" / "downloads" / archive["filename"]
            _download(archive["url"], archive_path, archive["sha256"])
            with zipfile.ZipFile(archive_path) as bundle:
                for entry in missing:
                    path = destination / f"{entry['id']}.png"
                    partial = path.with_name(path.name + ".partial")
                    # Read only named members: archive paths never become filesystem paths.
                    with bundle.open(entry["archive_member"]) as source, partial.open("wb") as output:
                        shutil.copyfileobj(source, output, length=1024 * 1024)
                    _check_original(partial, entry)
                    partial.replace(path)
        else:
            for entry in missing:
                path = destination / f"{entry['id']}.png"
                _download(entry["url"], path, entry["sha256"])
                _check_original(path, entry)
        result[name] = [
            {"id": entry["id"], "path": str(destination / f"{entry['id']}.png"),
             "sha256": entry["sha256"], "width": entry["width"], "height": entry["height"]}
            for entry in entries
        ]
    return result
