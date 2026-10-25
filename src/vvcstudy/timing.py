"""Serial RGB-to-compressed-file timing on all 24 clean Kodak images."""

from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
import ctypes
import os
import platform
import shutil
import statistics
import time

import cv2

from .artifacts import file_record, identity, read_csv, read_json, sha256, verified_result, write_csv, write_json
from .datasets import prepare_datasets
from .encoding import encode_jpeg, encoder_command, execute_command, jpeg_environment, prepare_yuv, read_rgb
from .runner import exclusive_run, session_metadata


def affinity() -> set[int]:
    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel.GetCurrentProcess.restype = ctypes.c_void_p
    kernel.GetProcessAffinityMask.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_size_t),
                                            ctypes.POINTER(ctypes.c_size_t)]
    mask, system = ctypes.c_size_t(), ctypes.c_size_t()
    if not kernel.GetProcessAffinityMask(kernel.GetCurrentProcess(), ctypes.byref(mask), ctypes.byref(system)):
        raise ctypes.WinError(ctypes.get_last_error())
    return {index for index in range(mask.value.bit_length()) if mask.value & (1 << index)}


@contextmanager
def one_cpu(cpu: int):
    if os.name != "nt":
        raise RuntimeError("Timing is implemented for Windows only.")
    previous = affinity()
    if cpu not in previous:
        raise ValueError(f"Logical CPU {cpu} is unavailable.")
    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel.GetCurrentProcess.restype = ctypes.c_void_p
    kernel.SetProcessAffinityMask.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
    process = kernel.GetCurrentProcess()
    if not kernel.SetProcessAffinityMask(process, 1 << cpu):
        raise ctypes.WinError(ctypes.get_last_error())
    try:
        yield
    finally:
        if not kernel.SetProcessAffinityMask(process, sum(1 << item for item in previous)):
            raise ctypes.WinError(ctypes.get_last_error())


def selected_points(rows: list[dict]) -> tuple[dict, dict]:
    clean = [row for row in rows if row.get("dataset", "kodak") == "kodak"
             and row.get("condition", "clean") == "clean"]
    expected = {(f"kodim{i:02d}.png", qp, metric) for i in range(1, 25)
                for qp in (22, 27, 32, 37) for metric in ("psnr_rgb_db", "msssim_rgb")}
    keys = [(row["source"], int(row["qp"]), row["metric"]) for row in clean]
    if len(keys) != len(expected) or set(keys) != expected or any(row["status"] != "ok" for row in clean):
        raise ValueError("Timing requires the complete accepted 24-image clean Kodak matching table.")
    vtm, jpeg = {}, {}
    for row in clean:
        for target, key, checksum in (
            (vtm, (row["source"], int(row["qp"])), row["vtm_sha256"]),
            (jpeg, (row["source"], int(row["nearest_parameter"])), row["nearest_sha256"]),
        ):
            if key in target and target[key] != checksum:
                raise ValueError("The same measured operating point has conflicting bitstream hashes.")
            target[key] = checksum
    return vtm, jpeg


def timing_plan(rows: list[dict], repeats: int = 3, jpeg_batch: int = 32) -> dict:
    vtm, jpeg = selected_points(rows)
    if repeats < 1 or jpeg_batch < 1:
        raise ValueError("Positive repeat and batch sizes required.")
    return {"images": 24, "vtm_points": len(vtm), "jpeg_unique_qfs": len(jpeg),
            "vtm_measurements": len(vtm) * repeats, "jpeg_measurements": len(jpeg) * repeats,
            "jpeg_encodes_per_measurement": jpeg_batch, "workers": 1,
            "boundary": "loaded RGB to compressed file; VTM process startup included",
            "warmups": "one VTM and one JPEG per resumed session, excluded from summaries"}


def verify_timing(root: Path, profile: str = "full") -> dict:
    base = root / "results" / profile
    folder = base / "timing"
    if not (folder / "manifest.json").exists():
        return {"status": "pending"}
    try:
        manifest = read_json(folder / "manifest.json")
        if not manifest["complete"] or manifest["matched_sha256"] != sha256(base / "tables/actual_by_qp.csv"):
            raise ValueError("Timing result does not match the selected quality pairs.")
        if not {"context.json", "repeats.csv", "paired.csv", "summary.csv"}.issubset(
                {record["path"] for record in manifest["files"]}):
            raise ValueError("Incomplete timing summary.")
        for record in manifest["files"]:
            path = (folder / record["path"]).resolve()
            if (not path.is_relative_to(folder.resolve()) or not path.is_file()
                    or path.stat().st_size != record["bytes"] or sha256(path) != record["sha256"]):
                raise ValueError("Timing summary checksum mismatch.")
        return {"status": "complete", "accepted": True, "measurements": manifest["measurements"]}
    except (OSError, ValueError, KeyError) as error:
        return {"status": "failed", "accepted": False, "error": str(error)}


def _measure(job: dict, folder: Path, session: str) -> dict:
    spec = job["spec"]
    folder.mkdir(parents=True, exist_ok=True)
    rgb = read_rgb(Path(job["input"]))  # PNG loading is deliberately outside the timer.
    if sha256(Path(job["input"])) != spec["input_sha256"]:
        raise ValueError("Timing input changed.")
    output = folder / ("stream.vvc" if spec["codec"] == "VTM" else "image.jpg")
    if spec["codec"] == "VTM":
        shutil.copyfile(job["config"], folder / "encoder.cfg")
        command = encoder_command({"width": rgb.shape[1], "height": rgb.shape[0],
                                   "qp": spec["parameter"], "codec": "vtm", "mode": "baseline"},
                                  job["encoder"], timing=True)
        begin = time.perf_counter()
        prepare_yuv(rgb, folder / "input.yuv", "opencv_444")
        execute_command(command, folder, "encoder.log")
        elapsed = time.perf_counter() - begin
        batch = 1
    else:
        batch = spec["batch"]
        begin = time.perf_counter()
        for _ in range(batch):
            output.write_bytes(encode_jpeg(rgb, spec["parameter"]))
        elapsed = time.perf_counter() - begin
    if sha256(output) != spec["expected_stream_sha256"]:
        raise ValueError("Timed file differs from its quality-matched reference; do not merge this timing.")
    return {"job_id": identity(spec), "spec": spec, "status": "complete",
            "elapsed_seconds": elapsed / batch, "batch_elapsed_seconds": elapsed,
            "batch": batch, "session": session, "artifacts": [file_record(output, folder)]}


def run_timing(root: Path, matched: Path, codec: dict, *, cpu: int = 0,
               repeats: int = 3, jpeg_batch: int = 32, reset_stop: bool = True) -> dict:
    rows = read_csv(matched)
    vtm, jpeg = selected_points(rows)
    plan = timing_plan(rows, repeats, jpeg_batch)
    originals = {row["id"] + ".png": row for row in prepare_datasets(root, ["kodak"])["kodak"]}
    config = root / "configs" / "codecs" / "vtm_intra.cfg"
    folder = root / "runs" / "full"
    timing_dir = folder / "timing"
    import winreg
    with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"HARDWARE\DESCRIPTION\System\CentralProcessor\0") as key:
        cpu_model = winreg.QueryValueEx(key, "ProcessorNameString")[0].strip()
    context = {"host": platform.node(), "cpu_model": cpu_model, "logical_cpus": os.cpu_count(),
               "platform": platform.platform(), "cpu_affinity": cpu, "repeats": repeats,
               "jpeg_batch": jpeg_batch, "opencv": cv2.__version__, "jpeg": jpeg_environment(),
               "encoder_sha256": sha256(Path(codec["encoder"])), "config_sha256": sha256(config),
               "timing_code_sha256": sha256(Path(__file__)),
               "encoding_code_sha256": sha256(Path(__file__).with_name("encoding.py"))}
    if "runtime_hashes" in codec:
        context["runtime_hashes"] = codec["runtime_hashes"]
    jobs = []
    for name, points in (("VTM", vtm), ("JPEG", jpeg)):
        for (source, parameter), checksum in sorted(points.items()):
            for repeat in range(repeats):
                spec = {"kind": "timing", "protocol": "loaded_rgb_to_file_v1", "codec": name,
                        "context_sha256": identity(context),
                        "source": source, "parameter": parameter, "repeat": repeat, "cpu": cpu,
                        "input_sha256": originals[source]["sha256"], "expected_stream_sha256": checksum,
                        "batch": 1 if name == "VTM" else jpeg_batch,
                        "implementation": {"encoder_sha256": sha256(Path(codec["encoder"])),
                                           "config_sha256": sha256(config)} if name == "VTM" else jpeg_environment()}
                jobs.append({"spec": spec, "input": originals[source]["path"],
                             "config": str(config), "encoder": codec["encoder"]})
    cv2.setNumThreads(1)
    with exclusive_run(folder), one_cpu(cpu):
        if reset_stop:
            (folder / "STOP").unlink(missing_ok=True)
        timing_dir.mkdir(parents=True, exist_ok=True)
        context_file = timing_dir / "context.json"
        if context_file.exists() and read_json(context_file) != context:
            raise ValueError("Timing hardware, software or protocol changed; keep this series separate.")
        write_json(context_file, context)
        write_json(timing_dir / "plan.json", {"jobs": [
            {"job_id": identity(job["spec"]), "spec": job["spec"]} for job in jobs]})
        meta = session_metadata(root)
        session = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
        meta.update(cpu_affinity=sorted(affinity()), opencv_threads=cv2.getNumThreads(), protocol=plan)
        write_json(timing_dir / "sessions" / (session + ".json"), meta)
        pending = [job for job in jobs if verified_result(timing_dir / identity(job["spec"]), job["spec"]) is None]
        for name in ("VTM", "JPEG"):
            warmup = next((job for job in pending if job["spec"]["codec"] == name), None)
            if warmup and not (folder / "STOP").exists():
                _measure(warmup, timing_dir / "warmup" / session / name, session)
        completed = len(jobs) - len(pending)
        for job in pending:
            if (folder / "STOP").exists():
                break
            target = timing_dir / identity(job["spec"])
            result = _measure(job, target, session)
            write_json(target / "result.json", result)
            completed += 1
            print(f"[{completed}/{len(jobs)}] timed {job['spec']['source']} "
                  f"{job['spec']['codec']} {job['spec']['parameter']} repeat {job['spec']['repeat'] + 1}", flush=True)
        result = {**plan, "status": "complete" if completed == len(jobs) and not (folder / "STOP").exists() else "stopped",
                  "completed": completed, "expected": len(jobs), "session": session}
        write_json(timing_dir / "progress.json", result)
        if result["status"] == "complete":
            output = root / "results/full/timing"
            output.mkdir(parents=True, exist_ok=True)
            summarize_timing(timing_dir, rows, jobs, output=output)
            write_json(output / "context.json", context)
            write_json(output / "manifest.json", {
                "complete": True, "measurements": len(jobs), "matched_sha256": sha256(matched),
                "files": [file_record(path, output) for path in sorted(output.iterdir())
                          if path.name != "manifest.json" and path.suffix in {".csv", ".json"}]})
        return result


def summarize_timing(folder: Path, matches: list[dict], jobs: list[dict], *, output: Path | None = None) -> None:
    output = output or folder
    measured = {}
    repeats = []
    for job in jobs:
        spec = job["spec"]
        result = verified_result(folder / identity(spec), spec)
        if result is None:
            raise ValueError("Timing is incomplete or its artifacts changed.")
        key = spec["codec"], spec["source"], spec["parameter"]
        measured.setdefault(key, []).append(result["elapsed_seconds"])
        repeats.append({**spec, "elapsed_seconds": result["elapsed_seconds"], "session": result["session"]})
    medians = {key: statistics.median(values) for key, values in measured.items()}
    paired = []
    for row in matches:
        if row.get("dataset", "kodak") != "kodak" or row.get("condition", "clean") != "clean":
            continue
        vtm = medians["VTM", row["source"], int(row["qp"])]
        jpeg = medians["JPEG", row["source"], int(row["nearest_parameter"])]
        paired.append({"source": row["source"], "qp": row["qp"], "metric": row["metric"],
                       "qf": row["nearest_parameter"], "R": row["ratio_nearest"],
                       "quality_delta": row["nearest_quality_delta"], "vtm_seconds": vtm,
                       "jpeg_seconds": jpeg, "additional_vtm_seconds": vtm - jpeg})
    summary = []
    for qp in (22, 27, 32, 37):
        for metric in ("psnr_rgb_db", "msssim_rgb"):
            group = [row for row in paired if int(row["qp"]) == qp and row["metric"] == metric]
            summary.append({"qp": qp, "metric": metric, "n_images": len(group), **{
                key + suffix: operation([float(row[key]) for row in group])
                for key in ("vtm_seconds", "jpeg_seconds", "additional_vtm_seconds", "R")
                for suffix, operation in (("_mean", statistics.mean), ("_sd", statistics.stdev),
                                          ("_median", statistics.median))}})
    write_csv(output / "repeats.csv", repeats)
    write_csv(output / "paired.csv", paired)
    write_csv(output / "summary.csv", summary)
