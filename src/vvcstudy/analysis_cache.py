"""Integrity records for completed numerical analysis outputs; regenerable figures are optional."""

from pathlib import Path
import hashlib
import inspect

from .artifacts import file_record, read_json, sha256, write_json


def _signature(root: Path, profile: str, resamples: int) -> dict:
    base = root / "results" / profile
    from .plots import matrix_tables
    from .anchors import anchor_points, compare_points, analyze_anchors
    modules = ("jpeg_analysis.py", "partition_analysis.py", "statistics.py")
    calculations = (matrix_tables, anchor_points, compare_points, analyze_anchors)
    return {"resamples": resamples,
            "inputs": {name: sha256(base / "tables" / name)
                       for name in ("encodings.csv", "actual_by_qp.csv")},
            "anchors": sha256(root / "manifests/anchors.json"),
            "validation_code": {name: sha256(Path(__file__).parent / name) for name in modules},
            "descriptive_code": {f.__name__: hashlib.sha256(inspect.getsource(f).encode()).hexdigest()
                                 for f in calculations}}


def seal_analysis(root: Path, profile: str = "full", resamples: int = 99_999) -> None:
    base = root / "results" / profile / "analysis"
    required = ["partitions/kodak/correlations.csv", "partitions/kodak/contrasts.csv",
                "partitions/div2k/correlations_with_ci.csv",
                "partitions/div2k/noise_correlation_changes.csv",
                "partitions/div2k/bootstrap_metadata.json",
                "partitions/div2k/descriptor_correlations.csv", "matrices/same_qp_deltas.csv",
                "jpeg-vvc/correlations_with_ci.csv", "jpeg-vvc/matching_choice_differences.csv",
                "jpeg-vvc/noise_correlation_changes.csv", "anchors/nearest_bpp.csv",
                "anchors/context.json", "anchors/cross_anchor.csv",
                "matrices/conversion_deltas.csv", "matrices/mean_rd.csv"]
    if any(not (base / name).is_file() for name in required):
        raise ValueError("Statistical outputs are incomplete.")
    files = [file_record(path, base) for path in sorted(base.rglob("*"))
             if path.suffix == ".csv" or path.name in {"bootstrap_metadata.json", "context.json"}]
    write_json(base / "manifest.json", {"schema_version": 2, "signature": _signature(root, profile, resamples),
                                         "files": files})


def completed_analysis(root: Path, profile: str, resamples: int) -> bool:
    base = root / "results" / profile / "analysis"
    try:
        manifest = read_json(base / "manifest.json")
        if manifest["signature"] != _signature(root, profile, resamples) or not manifest["files"]:
            return False
        for record in manifest["files"]:
            path = (base / record["path"]).resolve()
            if (not path.is_relative_to(base.resolve()) or not path.is_file()
                    or path.stat().st_size != record["bytes"] or sha256(path) != record["sha256"]):
                return False
        return True
    except (OSError, ValueError, KeyError):
        return False
