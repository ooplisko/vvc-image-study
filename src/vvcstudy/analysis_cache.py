"""Integrity records for completed statistical tables and figures."""

from pathlib import Path

from .artifacts import file_record, read_json, sha256, write_json


def _signature(root: Path, profile: str, resamples: int) -> dict:
    base = root / "results" / profile
    modules = ("jpeg_analysis.py", "partition_analysis.py", "statistics.py", "plots.py")
    return {"resamples": resamples,
            "inputs": {name: sha256(base / "tables" / name)
                       for name in ("encodings.csv", "actual_by_qp.csv")},
            "validation_code": {name: sha256(Path(__file__).parent / name) for name in modules}}


def seal_analysis(root: Path, profile: str = "full", resamples: int = 99_999) -> None:
    base = root / "results" / profile / "analysis"
    required = ["partitions/kodak/correlations.csv", "partitions/kodak/contrasts.csv",
                "partitions/div2k/descriptor_correlations.csv", "matrices/same_qp_deltas.csv",
                "jpeg-vvc/correlations_with_ci.csv", "jpeg-vvc/matching_choice_differences.csv",
                "jpeg-vvc/noise_correlation_changes.csv"]
    if any(not (base / name).is_file() for name in required):
        raise ValueError("Statistical outputs are incomplete.")
    files = [file_record(path, base) for path in sorted(base.rglob("*"))
             if path.suffix in {".csv", ".png"}]
    write_json(base / "manifest.json", {"schema_version": 1, "signature": _signature(root, profile, resamples),
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
