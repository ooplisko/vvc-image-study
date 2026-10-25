"""Small, explicit records shared by experiments and verification."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def identity(spec: dict) -> str:
    raw = json.dumps(spec, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _json_values(value: Any) -> Any:
    if isinstance(value, float) and not math.isfinite(value):
        return str(value)
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {key: _json_values(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_values(item) for item in value]
    return value


def write_json(path: Path, value: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".partial")
    with temporary.open("w", encoding="utf-8", newline="\n") as stream:
        json.dump(_json_values(value), stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def read_json(path: Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def read_csv(path: Path) -> list[dict]:
    with Path(path).open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def write_csv(path: Path, rows: list[dict]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".partial")
    keys = list(dict.fromkeys(key for row in rows for key in row))
    with temporary.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def file_record(path: Path, base: Path) -> dict:
    return {"path": path.relative_to(base).as_posix(), "bytes": path.stat().st_size,
            "sha256": sha256(path)}


def verified_result(folder: Path, spec: dict) -> dict | None:
    """An existing file is insufficient: identity, lengths and hashes must agree."""
    result_path = folder / "result.json"
    if not result_path.is_file():
        return None
    try:
        result = read_json(result_path)
        if result["job_id"] != identity(spec) or result["spec"] != spec:
            return None
        if result.get("status") != "complete" or not result.get("artifacts"):
            return None
        for item in result["artifacts"]:
            path = (folder / item["path"]).resolve()
            if not path.is_relative_to(folder.resolve()):
                return None
            if not path.is_file() or path.stat().st_size != item["bytes"]:
                return None
            if sha256(path) != item["sha256"]:
                return None
        return result
    except (OSError, ValueError, KeyError, TypeError):
        return None


def verify_artifact_manifest(folder: Path) -> dict:
    """Check every retained file listed in a raw-output bundle manifest."""
    result = {"accepted": False, "checked_files": 0, "checked_bytes": 0, "errors": [],
              "scope": "files listed in the raw-output manifest; not the full experiment grid"}
    try:
        folder = folder.resolve()
        manifest = read_json(folder / "manifest.json")
        if manifest["schema_version"] != 1 or not manifest["files"]:
            raise ValueError("Unsupported or empty raw-output manifest")
        seen = set()
        for record in manifest["files"]:
            path = (folder / record["path"]).resolve()
            if Path(record["path"]).is_absolute() or not path.is_relative_to(folder):
                raise ValueError(f"Artifact path is outside its bundle: {record['path']}")
            if path in seen:
                raise ValueError(f"Duplicate artifact path: {record['path']}")
            seen.add(path)
            if not path.is_file() or path.stat().st_size != record["bytes"]:
                raise ValueError(f"Missing artifact or incorrect size: {record['path']}")
            if sha256(path) != record["sha256"]:
                raise ValueError(f"Artifact checksum mismatch: {record['path']}")
            result["checked_files"] += 1
            result["checked_bytes"] += record["bytes"]
        result["accepted"] = True
    except (OSError, ValueError, KeyError, TypeError) as error:
        result["errors"].append(str(error))
    return result
