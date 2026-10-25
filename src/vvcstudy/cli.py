"""Command entry points; data preparation and encoding are always explicit."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="vvc-study")
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[2])
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("doctor")
    build = sub.add_parser("build-codecs")
    build.add_argument("--parallel", type=int, default=4)
    control = sub.add_parser("check-codecs")
    control.add_argument("--codec-mode", choices=("source", "bundle"), default="source")
    bundle = sub.add_parser("package-codecs")
    bundle.add_argument("--output", type=Path)
    imported_bundle = sub.add_parser("import-bundle")
    imported_bundle.add_argument("--archive", type=Path, required=True)
    imported_bundle.add_argument("--sha256", required=True)
    for name in ("plan", "prepare", "reproduce", "run", "analyze", "verify", "status", "stop"):
        part = sub.add_parser(name)
        part.add_argument("--profile", choices=("full", "smoke"), default="full")
        if name in {"prepare", "reproduce", "run"}:
            part.add_argument("--codec-mode", choices=("source", "bundle"), default="source")
            part.add_argument("--workers", type=int, default=4)
        if name in {"run", "reproduce", "analyze"}:
            part.add_argument("--recompute", action="store_true")
        if name == "run":
            part.add_argument("--study", choices=("partitions", "div2k", "matrices", "jpeg-vvc", "timing"), required=True)
            part.add_argument("--matched", type=Path)
            part.add_argument("--cpu", type=int, default=0)
        if name == "verify":
            part.add_argument("--jobs", action="store_true")
            part.add_argument("--artifacts", action="store_true",
                              help="Also verify retained raw files in runs/<profile>/artifacts")
        if name == "analyze":
            part.add_argument("--study", choices=("jpeg-vvc", "partitions", "matrices", "all"), default="all")
            part.add_argument("--resamples", type=int, default=99_999)
            part.add_argument("--check-only", action="store_true")
    timing = sub.add_parser("timing-plan")
    timing.add_argument("--matched", type=Path)
    return parser


def _matched_file(root: Path, profile: str = "full") -> Path:
    return root / "results" / profile / "tables/actual_by_qp.csv"


def _codec_names(profile: str, study: str | None = None) -> list[str]:
    if profile == "smoke" or study in {"partitions", "div2k", "jpeg-vvc", "timing"}:
        return ["vtm23-baseline-trace"]
    return ["vtm23-baseline-trace", "vtm23-csf-trace", "vvenc-baseline-trace", "vvenc-csf-trace"]


def verify_run(root: Path, profile: str) -> dict:
    from .artifacts import identity, read_json, verified_result
    directory = root / "runs" / profile
    errors = []
    missing = []
    planned = {}
    for path in (directory / "plans").glob("*.json"):
        for job in read_json(path)["jobs"]:
            if job["job_id"] != identity(job["spec"]):
                errors.append("plan identity: " + job["job_id"])
            planned[job["job_id"]] = job["spec"]
    for job_id, spec in planned.items():
        if not (directory / "jobs" / job_id / "result.json").exists():
            missing.append(job_id)
        elif verified_result(directory / "jobs" / job_id, spec) is None:
            errors.append(job_id)
    files = list((directory / "jobs").glob("*/result.json"))
    for path in files:
        record = read_json(path)
        if (path.parent.name not in planned
                and verified_result(path.parent, record["spec"]) is None):
            errors.append(path.parent.name)
    timing_jobs = []
    timing_plan = directory / "timing" / "plan.json"
    if timing_plan.exists():
        timing_jobs = read_json(timing_plan)["jobs"]
        for job in timing_jobs:
            if verified_result(directory / "timing" / job["job_id"], job["spec"]) is None:
                missing.append("timing/" + job["job_id"])
    return {"accepted": bool(planned or timing_jobs) and not errors and not missing, "checked_jobs": len(files),
            "planned_jobs": len(planned), "timing_jobs": len(timing_jobs), "errors": errors, "missing": missing,
            "scope": "registered study grids and artifact integrity; unrequested studies are not implied"}


def result_exit_code(value) -> int:
    if isinstance(value, dict):
        if value.get("accepted") is False or value.get("status") == "failed":
            return 1
        nested = [result_exit_code(item) for item in value.values()]
        if 1 in nested:
            return 1
        if value.get("status") == "stopped" or 2 in nested:
            return 2
    elif isinstance(value, list):
        codes = [result_exit_code(item) for item in value]
        return 1 if 1 in codes else 2 if 2 in codes else 0
    return 0


def analyze(root: Path, profile: str, study: str, *, resamples=99_999, check_only=False, recompute=False) -> dict:
    from .artifacts import read_csv
    from .analysis_cache import completed_analysis, seal_analysis
    from .jpeg_analysis import analyze_jpeg
    from .plots import matrix_plots, partition_plots, partition_examples, jpeg_plots
    from .results import verify_results
    base = root / "results" / profile
    output = base / "analysis"
    selected = None if study == "all" else ["partitions", "div2k"] if study == "partitions" else [study]
    if profile == "full":
        checked = verify_results(root, profile, selected)
        if not checked["accepted"] or check_only:
            return checked
        if not recompute and completed_analysis(root, profile, resamples):
            return {"status": "complete", "reused": True, "output": str(output)}
    result = {}
    if study in {"all", "jpeg-vvc"}:
        matched = _matched_file(root, profile)
        result["jpeg-vvc"] = analyze_jpeg([matched], output / "jpeg-vvc", resamples=resamples,
                                          check_only=check_only)
        if not check_only:
            jpeg_plots(read_csv(matched), output / "figures")
    if study in {"all", "partitions"}:
        from .partition_analysis import analyze_partitions, load_validated_measurements
        paths = [base / "tables/encodings.csv"]
        if not paths:
            raise ValueError("No partition measurements available.")
        result["partitions"] = []
        for index, path in enumerate(paths):
            if check_only:
                checked = load_validated_measurements(path)
                result["partitions"].append({"path": str(path), "datasets": {
                    name: {"rows": len(rows), "sources": len({row["source"] for row in rows})}
                    for name, rows in checked.items()}})
            else:
                result["partitions"].append(analyze_partitions(path, output / "partitions", resamples=resamples))
                partition_plots(read_csv(path), output / "figures" / "partitions")
        if not check_only:
            result["partition_examples"] = partition_examples(root, root / "runs" / profile,
                                                              output / "figures" / "partition_maps")
    if study in {"all", "matrices"}:
        paths = [base / "tables/encodings.csv"]
        result["matrices"] = {"tables": len(paths)}
        if not paths:
            raise ValueError("No matrix measurements available.")
        if not check_only:
            matrix_plots(paths, output / "matrices")
    if profile == "full" and study == "all" and not check_only:
        seal_analysis(root, profile, resamples)
    return result


def _prepared(root: Path, profile: str, study: str | None, mode: str) -> dict:
    from .build import prepare_codecs
    from .codec_validation import check_codecs
    codecs = prepare_codecs(root, mode=mode, names=_codec_names(profile, study))["codecs"]
    if profile == "full":
        checked = check_codecs(root, codecs)
        if not checked.get("accepted"):
            raise ValueError("Codec controls did not pass: " + json.dumps(checked))
    return codecs


def execute_study(root: Path, profile: str, study: str, mode: str, workers: int,
                  *, recompute=False, reset_stop=True) -> dict:
    from .results import completed_study
    from .runner import exclusive_run
    from .studies import run_study
    folder = root / "runs" / profile
    # A reused stage must still obey a stop requested at the preceding stage boundary.
    with exclusive_run(folder):
        if reset_stop:
            (folder / "STOP").unlink(missing_ok=True)
        if (folder / "STOP").exists():
            return {"status": "stopped"}
        if profile == "full" and not recompute and completed_study(root, profile, study):
            return {"status": "complete", "reused_results": True, "study": study}
    codecs = _prepared(root, profile, study, mode)
    return run_study(root, profile, study, codecs, workers, reset_stop=False)


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    root = args.root.resolve()
    # Set before importing numerical libraries; timings additionally pin one logical CPU.
    for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[key] = "1"
    try:
        from .build import prepare_codecs
        from .runner import status, stop
        from .studies import plan
        if args.command == "doctor":
            result = {"platform": platform.platform(), "supported": os.name == "nt",
                      "tools": {name: shutil.which(name) for name in ("git", "cmake", "g++", "mingw32-make", "ffmpeg")}}
        elif args.command == "build-codecs":
            from .codec_validation import CODECS
            result = prepare_codecs(root, names=list(CODECS), parallel=args.parallel)
        elif args.command == "check-codecs":
            from .codec_validation import CODECS, check_codecs
            codecs = prepare_codecs(root, mode=args.codec_mode, names=list(CODECS))["codecs"]
            result = check_codecs(root, codecs)
        elif args.command == "package-codecs":
            from .codec_bundle import package_codecs
            result = package_codecs(root, args.output)
        elif args.command == "import-bundle":
            from .codec_bundle import import_bundle
            result = import_bundle(root, args.archive, args.sha256)
        elif args.command == "plan":
            result = plan(root, args.profile)
        elif args.command == "status":
            result = status(root / "runs" / args.profile)
            timing_progress = root / "runs" / args.profile / "timing/progress.json"
            if timing_progress.exists():
                from .artifacts import read_json
                result["timing"] = read_json(timing_progress)
        elif args.command == "stop":
            stop(root / "runs" / args.profile)
            result = {"requested": "stop after active jobs finish"}
        elif args.command == "timing-plan":
            from .artifacts import read_csv
            from .timing import timing_plan
            result = timing_plan(read_csv(args.matched or _matched_file(root)))
        elif args.command == "verify":
            if args.jobs or args.profile == "smoke":
                result = verify_run(root, args.profile)
            else:
                from .results import verify_results
                result = verify_results(root, args.profile)
                from .timing import verify_timing
                result["timing"] = verify_timing(root, args.profile)
            if args.artifacts:
                from .artifacts import verify_artifact_manifest
                result["artifacts"] = verify_artifact_manifest(root / "runs" / args.profile / "artifacts")
        elif args.command == "analyze":
            result = analyze(root, args.profile, args.study, resamples=args.resamples,
                             check_only=args.check_only, recompute=args.recompute)
        else:
            study = getattr(args, "study", None)
            if study == "timing" and (args.profile != "full" or args.workers != 1):
                raise ValueError("Timing requires --profile full --workers 1.")
            if args.command == "prepare":
                from .datasets import prepare_datasets
                from .studies import synthetic_inputs
                codecs = prepare_codecs(root, mode=args.codec_mode, names=_codec_names(args.profile))["codecs"]
                datasets = ({"synthetic": synthetic_inputs(root)} if args.profile == "smoke"
                            else prepare_datasets(root, ["kodak", "div2k"]))
                result = {"codecs": list(codecs), "datasets": {key: len(value) for key, value in datasets.items()}}
            elif args.command == "run" and study == "timing":
                from .timing import run_timing
                codecs = _prepared(root, args.profile, study, args.codec_mode)
                result = run_timing(root, args.matched or _matched_file(root),
                                    codecs["vtm23-baseline-trace"], cpu=args.cpu)
            elif args.command == "run":
                result = execute_study(root, args.profile, study, args.codec_mode, args.workers,
                                       recompute=args.recompute)
            else:
                order = ["jpeg-vvc"] if args.profile == "smoke" else ["partitions", "div2k", "matrices", "jpeg-vvc"]
                result = {}
                for item in order:
                    result[item] = execute_study(root, args.profile, item, args.codec_mode, args.workers,
                                                 recompute=args.recompute, reset_stop=item == order[0])
                    if result[item]["status"] != "complete":
                        break
                else:
                    if args.profile == "full":
                        from .timing import run_timing
                        codecs = _prepared(root, args.profile, "timing", args.codec_mode)
                        result["timing"] = run_timing(root, _matched_file(root),
                                                      codecs["vtm23-baseline-trace"], reset_stop=False)
                        if result["timing"]["status"] == "complete":
                            result["analysis"] = analyze(root, "full", "all", recompute=args.recompute)
                        from .results import verify_results
                        result["verification"] = verify_results(root, args.profile)
                    else:
                        result["verification"] = verify_run(root, args.profile)
        print(json.dumps(result, ensure_ascii=False, indent=2, default=str))
        return result_exit_code(result)
    except (ValueError, RuntimeError, OSError, KeyError, subprocess.CalledProcessError) as exc:
        print(json.dumps({"error": str(exc)}, ensure_ascii=False))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
