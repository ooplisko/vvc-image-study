"""Bounded Windows job scheduling with finish-current stop and verified resume."""

from __future__ import annotations

from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from contextlib import contextmanager, nullcontext
from datetime import datetime, timezone
from pathlib import Path
import os
import platform
import subprocess
import sys
import time

from .artifacts import identity, read_json, verified_result, write_json


@contextmanager
def exclusive_run(folder: Path):
    if os.name != "nt":
        raise RuntimeError("This release supports Windows only.")
    import msvcrt
    folder.mkdir(parents=True, exist_ok=True)
    with (folder / ".run.lock").open("a+b") as stream:
        if os.fstat(stream.fileno()).st_size == 0:
            stream.write(b"0")
            stream.flush()
        stream.seek(0)
        try:
            msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
        except OSError as exc:
            raise RuntimeError("Another experiment is using this run directory.") from exc
        try:
            yield
        finally:
            stream.seek(0)
            msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)


def session_metadata(root: Path) -> dict:
    commit = subprocess.run(["git", "-C", str(root), "rev-parse", "HEAD"],
                            capture_output=True, text=True)
    status = subprocess.run(["git", "-C", str(root), "status", "--porcelain"],
                            capture_output=True, text=True)
    return {"started_utc": datetime.now(timezone.utc).isoformat(),
            "platform": platform.platform(), "processor": platform.processor(),
            "logical_cpus": os.cpu_count(), "python": sys.version,
            "commit": commit.stdout.strip() if commit.returncode == 0 else None,
            "dirty": bool(status.stdout.strip()), "pid": os.getpid()}


def stop(run_dir: Path) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "STOP").write_text("Finish active jobs; do not dispatch more.\n")


def run_jobs(jobs: list[dict], execute, run_dir: Path, *, workers: int = 1,
             lock_held: bool = False, reset_stop: bool = True) -> dict:
    """execute(job, folder) must return artifacts and values; results commit atomically."""
    if not 1 <= workers <= (os.cpu_count() or 1):
        raise ValueError("Workers must be between 1 and the logical CPU count.")
    ids = [identity(job["spec"]) for job in jobs]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate job specifications in this study.")
    stop_path = run_dir / "STOP"
    started = time.monotonic()
    with nullcontext() if lock_held else exclusive_run(run_dir):
        if reset_stop:
            stop_path.unlink(missing_ok=True)
        completed = 0
        pending = []
        for job, job_id in zip(jobs, ids):
            folder = run_dir / "jobs" / job_id
            if verified_result(folder, job["spec"]) is not None:
                completed += 1
            else:
                pending.append((job, folder))
        initial = completed
        iterator = iter(pending)
        errors = []
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {}

            def dispatch():
                while len(futures) < workers and not stop_path.exists():
                    item = next(iterator, None)
                    if item is None:
                        break
                    job, folder = item
                    folder.mkdir(parents=True, exist_ok=True)
                    futures[pool.submit(execute, job, folder)] = (job, folder)

            dispatch()
            while futures:
                try:
                    done, _ = wait(futures, timeout=0.25, return_when=FIRST_COMPLETED)
                except KeyboardInterrupt:
                    stop(run_dir)
                    print("Stop requested; waiting for active jobs to finish.", flush=True)
                    continue
                for future in done:
                    job, folder = futures.pop(future)
                    try:
                        value = future.result()
                        value.update(job_id=identity(job["spec"]), spec=job["spec"],
                                     status="complete")
                        write_json(folder / "result.json", value)
                        if verified_result(folder, job["spec"]) is None:
                            raise RuntimeError("Saved result failed integrity verification.")
                        completed += 1
                        print(f"[{completed}/{len(jobs)}] completed {job.get('label', value['job_id'][:12])}",
                              flush=True)
                    except Exception as exc:
                        errors.append({"job_id": identity(job["spec"]), "error": str(exc)})
                        stop(run_dir)
                dispatch()
        result = {"status": "failed" if errors else "stopped" if stop_path.exists() or completed < len(jobs) else "complete",
                  "expected": len(jobs), "completed": completed, "reused": initial,
                  "elapsed_seconds": time.monotonic() - started, "errors": errors}
        write_json(run_dir / "progress.json", result)
        return result


def status(run_dir: Path) -> dict:
    records = list((run_dir / "jobs").glob("*/result.json"))
    counts = {}
    for path in records:
        result = read_json(path)
        key = result.get("spec", {}).get("kind", "unknown")
        counts[key] = counts.get(key, 0) + 1
    return {"result_records": len(records), "by_kind": counts,
            "stop_requested": (run_dir / "STOP").exists(),
            "integrity_checked": False,
            "last_stage": read_json(run_dir / "progress.json")
            if (run_dir / "progress.json").exists() else None}
