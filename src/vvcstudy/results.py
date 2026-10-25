"""Complete experiment tables and resumable result-stage verification."""

from __future__ import annotations

import itertools
import json
import math
from pathlib import Path

from .artifacts import identity, read_csv, read_json, sha256, write_json
from .partition_analysis import normalize_measurements, validate_measurements, descriptive_tables


STUDIES = ("partitions", "div2k", "matrices", "jpeg-vvc")
QPS = ("22", "27", "32", "37")
FEATURES = ("sobel_si", "luma_sd", "edge_fraction", "glcm_contrast", "glcm_entropy", "glcm_homogeneity")
METRICS = ("psnr_rgb_db", "msssim_rgb")
CONDITIONS = ("clean", "awgn_30_seed_20260811")
SOURCES = {"kodak": tuple(f"kodim{i:02d}.png" for i in range(1, 25)),
           "div2k": tuple(f"{i:04d}.png" for i in range(801, 901))}


def _integer(value) -> str:
    return "" if value in (None, "", "None") else str(int(float(value)))


def encoding_key(row: dict) -> tuple:
    return (str(row["dataset"]), str(row["source"]), str(row["distortion"]),
            _integer(row["level"]), _integer(row.get("seed")), _integer(row["qp"]),
            str(row["codec"]), str(row["conversion"]))


def stage_rows(rows: list[dict], study: str) -> list[dict]:
    if study in {"partitions", "div2k"}:
        dataset = "kodak" if study == "partitions" else "div2k"
        return [r for r in rows if r["dataset"] == dataset and r["conversion"] == "opencv_444"]
    if study == "matrices":
        return [r for r in rows if r["conversion"].startswith("ffmpeg")]
    if study == "jpeg-vvc":
        return [r for r in rows if r["conversion"] == "opencv_444" and
                (r["distortion"] == "clean" or (r["distortion"] == "awgn" and
                 _integer(r["level"]) == "30" and _integer(r.get("seed")) == "20260811"))]
    raise ValueError(f"Unknown result stage: {study}")


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _close(actual, expected: float, label: str) -> None:
    _require(math.isfinite(float(actual)) and math.isclose(float(actual), expected, rel_tol=1e-9, abs_tol=1e-10),
             f"Inconsistent {label}: {actual} != {expected}")


def _grid(rows: list[dict], fields: tuple[str, ...], expected: set[tuple], label: str) -> None:
    keys = [tuple(str(row[field]) for field in fields) for row in rows]
    _require(len(keys) == len(set(keys)), f"Duplicate {label} keys")
    actual = set(keys)
    _require(actual == expected, f"Incomplete {label}: missing {len(expected-actual)}, unexpected {len(actual-expected)}")


def _partition_integrity(rows: list[dict]) -> None:
    for row in rows:
        label = f"{row['source']} QP{row['qp']} {row['distortion']}"
        for key in ("reconstruction_verified", "cu_coverage_verified"):
            if row.get(key):
                _require(str(row[key]).lower() == "true", f"Failed {key}: {label}")
        if row.get("reconstruction_sha256") and row.get("decoded_sha256"):
            _require(row["reconstruction_sha256"] == row["decoded_sha256"], f"Decode hash differs: {label}")
        histogram = json.loads(row["size_histogram_json"])
        count = sum(int(n) for n in histogram.values())
        area = sum(math.prod(map(int, shape.split("x"))) * int(n) for shape, n in histogram.items())
        _require(count == int(row["cu_count"]), f"CU histogram count differs: {label}")
        _require(area == int(row["coded_area"]), f"CU histogram coverage differs: {label}")
        _require(area == int(row["coded_width"]) * int(row["coded_height"]), f"Coded geometry differs: {label}")
        _close(row["cu_density_per_mpixel"], count * 1e6 / area, f"CU density {label}")


def _check_partitions(rows: list[dict], dataset: str) -> dict:
    _require(bool(rows), f"Missing {dataset} results")
    normalized = normalize_measurements(rows)
    _require(set(normalized) == {dataset}, "Unexpected partition dataset")
    selected = normalized[dataset]
    if dataset == "kodak":
        validate_measurements(selected)
    else:
        _require({row["source"] for row in selected} == set(SOURCES[dataset]), "Expected all 100 DIV2K images")
        descriptive_tables(selected)
    _partition_integrity(rows)
    return {"rows": len(rows), "sources": len(SOURCES[dataset])}


def _check_matrices(rows: list[dict]) -> dict:
    expected = {(source, qp, mode, codec) for source, qp, mode, codec in itertools.product(
        SOURCES["kodak"], QPS, ("baseline", "csf"), ("vtm", "vvenc"))}
    keys = [{**row, "family": "vvenc" if row["codec"].startswith("vvenc") else "vtm"} for row in rows]
    _grid(keys, ("source", "qp", "mode", "family"), expected, "matrix points384")
    for row in keys:
        expected_conversion = "ffmpeg_420" if row["family"] == "vvenc" else "ffmpeg_444"
        _require(row["dataset"] == "kodak" and row["conversion"] == expected_conversion,
                 "Unexpected matrix color protocol")
        _require(row["distortion"] == "clean" and _integer(row["level"]) == "0" and not _integer(row.get("seed")),
                 "Matrix comparisons require clean images")
        _close(row["file_bpp"], int(row["bitstream_bytes"]) * 8 / (int(row["width"]) * int(row["height"])), "matrix bpp")
        for metric in ("psnr_y_db", "psnr_rgb_db", "msssim_rgb"):
            _require(math.isfinite(float(row[metric])), f"Missing matrix metric {metric}")
        if _integer(row["qp"]) == "32":
            _require(0 < int(row["cu_count"]) <= int(row["width"]) * int(row["height"]), "Invalid matrix CU count")
    return {"encodes": len(rows), "partition_summaries": 96, "partition_qp": 32}


def _check_jpeg(base: Path) -> dict:
    cohorts = [(dataset, condition, source) for dataset, sources in SOURCES.items()
               for condition, source in itertools.product(CONDITIONS, sources)]
    rows = read_csv(base / "tables/measurements.csv")
    points = [("JPEG", str(qf)) for qf in range(1, 101)] + [("VTM", qp) for qp in QPS]
    _grid(rows, ("dataset", "condition", "source", "codec", "parameter"),
          {cohort + point for cohort, point in itertools.product(cohorts, points)}, "JPEG/VTM measurements25792")
    index = {(r["dataset"], r["condition"], r["source"], r["codec"], r["parameter"]): r for r in rows}
    for row in rows:
        _close(row["bpp"], int(row["bytes"]) * 8 / (int(row["width"]) * int(row["height"])), "JPEG/VTM bpp")
    matched = read_csv(base / "tables/actual_by_qp.csv")
    _grid(matched, ("dataset", "condition", "source", "metric", "qp"),
          {cohort + (metric, qp) for cohort, metric, qp in itertools.product(cohorts, METRICS, QPS)}, "matched1984")
    for row in matched:
        _require(row["status"] == "ok", "Unsuccessful quality match")
        cohort = (row["dataset"], row["condition"], row["source"])
        vtm = index[cohort + ("VTM", row["qp"])]
        metric = row["metric"]
        target = float(vtm[metric])
        _close(row["target_quality"], target, "target quality")
        _close(row["vtm_bpp"], float(vtm["bpp"]), "VTM bpp")
        _require(row["vtm_sha256"] == vtm["sha256"], "VTM match hash differs")
        measured = [index[cohort + ("JPEG", str(qf))] for qf in range(1, 101)]
        candidates = []
        for item in sorted(measured, key=lambda r: (-float(r[metric]), float(r["bpp"]), int(r["parameter"]))):
            if not candidates or float(item["bpp"]) < float(candidates[-1]["bpp"]):
                candidates.append(item)
        distances = [abs(float(item[metric]) - target) for item in candidates]
        for variant in ("nearest", "lower", "upper"):
            jpeg = index[cohort + ("JPEG", row[variant + "_parameter"])]
            quality = float(jpeg[metric])
            _close(row[variant + "_quality"], quality, variant + " quality")
            _close(row[variant + "_quality_delta"], quality - target, variant + " quality delta")
            _close(row[variant + "_bpp"], float(jpeg["bpp"]), variant + " bpp")
            _close(row["ratio_" + variant], float(jpeg["bpp"]) / float(vtm["bpp"]), variant + " ratio")
            _require(row[variant + "_sha256"] == jpeg["sha256"], "JPEG match hash differs")
            if variant == "nearest":
                _close(abs(quality - target), min(distances), "nearest JPEG distance")
            elif variant == "lower":
                _close(quality, max(float(item[metric]) for item in candidates if float(item[metric]) <= target), "lower JPEG neighbor")
            else:
                _close(quality, min(float(item[metric]) for item in candidates if float(item[metric]) >= target), "upper JPEG neighbor")
    return {"measurements": len(rows), "matched_rows": len(matched), "inputs": len(cohorts), "interpolation": False}


def _projection(rows: list[dict]) -> dict:
    # Empty columns introduced by another stage do not change existing rows.
    payload = [{key: str(value) for key, value in row.items() if value not in (None, "")}
               for row in rows]
    payload.sort(key=lambda row: json.dumps(row, sort_keys=True))
    return {"rows": len(payload), "sha256": identity({"rows": payload})}


def _stage_record(root: Path, profile: str, study: str) -> dict:
    if profile != "full":
        raise ValueError("Completed result stages use the full profile; verify smoke jobs separately.")
    base = root / "results" / profile
    rows = stage_rows(read_csv(base / "tables/encodings.csv"), study)
    if study in {"partitions", "div2k"}:
        counts = _check_partitions(rows, "kodak" if study == "partitions" else "div2k")
    elif study == "matrices":
        counts = _check_matrices(rows)
    else:
        counts = _check_jpeg(base)
        # Require the exact matched VTM operating points alongside the JPEG table.
        expected = {(dataset, source, condition, qp) for dataset, sources in SOURCES.items()
                    for source, condition, qp in itertools.product(sources, CONDITIONS, QPS)}
        keyed = [{**row, "condition": "clean" if row["distortion"] == "clean" else CONDITIONS[1]} for row in rows]
        _grid(keyed, ("dataset", "source", "condition", "qp"), expected, "matched VTM points992")
    tables = {"tables/encodings.csv": _projection(rows)}
    if study == "jpeg-vvc":
        for name in ("measurements.csv", "actual_by_qp.csv"):
            path = base / "tables" / name
            tables["tables/" + name] = {"bytes": path.stat().st_size, "sha256": sha256(path)}
    return {"config_sha256": sha256(root / "configs" / f"{profile}.toml"),
            "counts": counts, "tables": tables}


def seal_results(root: Path, profile: str = "full", studies=None) -> dict:
    """Validate complete tables and record their current contents for later reuse."""
    selected = tuple(studies or STUDIES)
    path = root / "results" / profile / "manifest.json"
    manifest = read_json(path) if path.exists() else {"schema_version": 1, "profile": profile, "studies": {}}
    records = {study: _stage_record(root, profile, study) for study in selected}
    manifest["studies"].update(records)
    write_json(path, manifest)
    return {"accepted": True, "studies": list(records)}


def verify_results(root: Path, profile: str = "full", studies=None) -> dict:
    result = {"accepted": False, "checks": {}, "errors": [],
              "scope": "complete result grids, numerical consistency and saved table checksums"}
    try:
        manifest = read_json(root / "results" / profile / "manifest.json")
        _require(manifest["schema_version"] == 1 and manifest["profile"] == profile, "Unsupported result manifest")
    except (OSError, ValueError, KeyError) as error:
        result["errors"].append(str(error))
        return result
    for study in tuple(studies or STUDIES):
        try:
            record = _stage_record(root, profile, study)
            _require(manifest["studies"].get(study) == record, f"Result checksum or configuration mismatch: {study}")
            result["checks"][study] = record["counts"]
        except (OSError, ValueError, KeyError, TypeError) as error:
            result["errors"].append(f"{study}: {error}")
    result["accepted"] = not result["errors"]
    return result


def completed_study(root: Path, profile: str, study: str) -> bool:
    return verify_results(root, profile, [study])["accepted"]
