"""Duan VTM 18.0 replication with author settings and resumable measured points."""

from pathlib import Path
import hashlib
import inspect
import shutil
import statistics

import cv2
import numpy as np

from .artifacts import file_record, identity, read_csv, read_json, sha256, verified_result, write_csv, write_json
from .encoding import execute_command, prepare_yuv, read_rgb
from .metrics import psnr
from .runner import exclusive_run, run_jobs, session_metadata


def replication_plan(root: Path, profile: str = "full") -> dict:
    return {"reference": "lossy-vae", "profile": profile,
            "images": 24 if profile == "full" else 2,
            "qps": [22, 27, 32, 37] if profile == "full" else [32],
            "expected": 96 if profile == "full" else 2,
            "codec": "VTM 18.0", "input": "OpenCV RGB8 to planar YUV444",
            "decoded_output": "YUV444 8-bit, then OpenCV COLOR_YUV2RGB",
            "metrics": ["bpp", "PSNR-RGB relative to original RGB8"],
            "comparison": "same QP; arithmetic means over all 24 Kodak images",
            "interpolation": False, "extrapolation": False,
            "run_directory": str(root / "runs" / profile / "lossy-vae")}


def replication_status(root: Path, profile: str = "full") -> dict:
    folder = root / "runs" / profile / "lossy-vae"
    path = folder / "plan.json"
    jobs = read_json(path)["jobs"] if path.exists() else []
    completed = sum(verified_result(folder / "jobs" / identity(j["spec"]), j["spec"]) is not None
                    for j in jobs)
    return {"status": "not_started" if not jobs else "complete" if completed == len(jobs) else "incomplete",
            "expected": len(jobs) if jobs else replication_plan(root, profile)["expected"],
            "completed": completed, "integrity_checked": True,
            "stop_requested": (folder / "STOP").exists()}


def prepare_replication_codec(root: Path) -> dict:
    from .build import _build_vtm, _verify_programs, toolchain
    recipe = read_json(root / "manifests/validation.json")["lossy_vae"]["codec"]
    path = root / ".build/vtm18-replication/build_manifest.json"
    if path.exists():
        record = read_json(path)
        if (record["source_commit"] != recipe["source_commit"] or record["patch_sha256"] is not None
                or "-DCMAKE_CXX_FLAGS=" + recipe["cxx_flags"] not in record["cmake_flags"]):
            raise ValueError("VTM 18.0 build differs from the replication recipe.")
        _verify_programs(record)
    else:
        record = _build_vtm(root, "vtm18-replication", recipe, toolchain(), 4)
    return record


def replication_jobs(root: Path, profile: str, codec: dict) -> list[dict]:
    from .datasets import prepare_datasets
    from .studies import synthetic_inputs
    inputs = (prepare_datasets(root, ["kodak"])["kodak"] if profile == "full"
              else synthetic_inputs(root))
    plan = replication_plan(root, profile)
    manifest = read_json(root / "manifests/validation.json")
    config = root / manifest["lossy_vae"]["configuration"]
    if sha256(config) != manifest["lossy_vae"]["configuration_sha256"]:
        raise ValueError("Author configuration checksum changed.")
    jobs = []
    retained_dir = root / "runs" / profile / "lossy-vae/retained"
    retained_path = retained_dir / "manifest.json"
    retained = []
    if retained_path.exists():
        saved_manifest = read_json(retained_path)
        if saved_manifest["schema_version"] != 1 or saved_manifest["codec_version"] != "VTM 18.0":
            raise ValueError("Unsupported retained-stream protocol.")
        retained = saved_manifest["streams"]
    calculation = hashlib.sha256("\n".join(inspect.getsource(f) for f in
        (execute_replication, prepare_yuv, read_rgb, psnr)).encode()).hexdigest()
    for item in inputs:
        for qp in plan["qps"]:
            spec = {"kind": "lossy-vae", "protocol": 1, "profile": profile,
                    "source": item["id"], "width": item["width"], "height": item["height"],
                    "input_sha256": item["sha256"], "qp": qp,
                    "codec_source_commit": manifest["lossy_vae"]["codec"]["source_commit"],
                    "encoder_sha256": codec["encoder_sha256"],
                    "decoder_sha256": codec["decoder_sha256"], "config_sha256": sha256(config),
                    "numpy": np.__version__, "opencv": cv2.__version__,
                    "calculation_sha256": calculation}
            job = {"spec": spec, "input": item["path"], "config": str(config),
                         "encoder": codec["encoder"], "decoder": codec["decoder"],
                         "label": f"lossy-vae {item['id']} QP{qp}"}
            candidates = [r for r in retained if r["source"] == item["id"] and r["qp"] == qp]
            if len(candidates) > 1:
                raise ValueError("Ambiguous retained replication stream.")
            if candidates:
                saved = candidates[0]
                if saved["input_sha256"] != item["sha256"] or saved["config_sha256"] != sha256(config):
                    raise ValueError("Retained stream belongs to a different input or configuration.")
                files = {}
                for name in ("stream", "encoder_log"):
                    path = (retained_dir / saved[name]["path"]).resolve()
                    if not path.is_relative_to(retained_dir.resolve()) or sha256(path) != saved[name]["sha256"]:
                        raise ValueError("Retained replication file checksum/path mismatch.")
                    files[name] = str(path)
                job["retained"] = files
                spec["retained_stream_sha256"] = saved["stream"]["sha256"]
                spec["retained_log_sha256"] = saved["encoder_log"]["sha256"]
            jobs.append(job)
    return jobs


def execute_replication(job: dict, folder: Path) -> dict:
    spec = job["spec"]
    source = Path(job["input"])
    if sha256(source) != spec["input_sha256"]:
        raise ValueError("Replication input checksum changed.")
    for name in ("encoder", "decoder", "config"):
        if sha256(Path(job[name])) != spec[name + "_sha256"]:
            raise ValueError(f"Replication {name} checksum changed.")
    rgb = read_rgb(source)
    height, width = rgb.shape[:2]
    if (height, width) != (spec["height"], spec["width"]):
        raise ValueError("Replication dimensions changed.")
    prepare_yuv(rgb, folder / "input.yuv", "opencv_444")
    shutil.copyfile(job["config"], folder / "encoder.cfg")
    # Match the author's image command: do not inherit VTM23 flags or tracing.
    command = [job["encoder"], "-c", "encoder.cfg", "--InputFile=input.yuv",
               "--BitstreamFile=stream.vvc", f"--SourceWidth={width}", f"--SourceHeight={height}",
               "--InputChromaFormat=444", "--FrameRate=1", "--FramesToBeEncoded=1", f"--QP={spec['qp']}"]
    if spec["profile"] == "smoke":
        # Synthetic odd-sized input only. Kodak dimensions need no conformance padding.
        command.append("--ConformanceWindowMode=1")
    decoder = [job["decoder"], "--BitstreamFile=stream.vvc", "--ReconFile=decoded.yuv",
               "--OutputBitDepth=8"]
    retained = job.get("retained")
    if retained:
        for name, target, field in (("stream", "stream.vvc", "retained_stream_sha256"),
                                     ("encoder_log", "encoder.log", "retained_log_sha256")):
            source_path = Path(retained[name])
            if sha256(source_path) != spec[field]:
                raise ValueError("Retained replication artifact changed.")
            shutil.copyfile(source_path, folder / target)
        log = (folder / "encoder.log").read_text(encoding="utf-8", errors="replace")
    else:
        log = execute_command(command, folder, "encoder.log")
    write_json(folder / "commands.json", {"encoder": None if retained else command, "decoder": decoder,
                                           "encoding_action": "retained_stream" if retained else "encoded"})
    if "VTM Encoder Version 18.0" not in log:
        raise ValueError("Expected VTM Encoder Version 18.0.")
    execute_command(decoder, folder, "decoder.log")
    data = np.fromfile(folder / "decoded.yuv", dtype=np.uint8)
    if data.size != width * height * 3:
        raise ValueError("Expected one visible 8-bit YUV444 frame.")
    reconstructed = cv2.cvtColor(data.reshape(3, height, width).transpose(1, 2, 0), cv2.COLOR_YUV2RGB)
    size = (folder / "stream.vvc").stat().st_size
    if size <= 0:
        raise ValueError("Empty replication bitstream.")
    values = {"source": spec["source"], "qp": spec["qp"], "file_bytes": size,
              "bpp": size * 8 / (height * width), "psnr_rgb_db": psnr(rgb, reconstructed),
              "input_sha256": spec["input_sha256"], "stream_sha256": sha256(folder / "stream.vvc"),
              "encoding_action": "retained_stream" if retained else "encoded"}
    files = ["input.yuv", "encoder.cfg", "stream.vvc", "decoded.yuv",
             "encoder.log", "decoder.log", "commands.json"]
    if not retained:
        files.append("rec.yuv")
    return {"values": values, "artifacts": [file_record(folder / name, folder) for name in files]}


def compare_replication(rows: list[dict], published: list[dict]) -> list[dict]:
    """Require the complete 24-image grid; never compare partial means to full means."""
    expected = {(f"kodim{i:02d}", q) for i in range(1, 25) for q in (22, 27, 32, 37)}
    keys = [(row["source"], int(row["qp"])) for row in rows]
    if len(keys) != len(set(keys)) or set(keys) != expected:
        raise ValueError("Comparison requires exactly all 24 Kodak images at all four QPs.")
    result = []
    for qp in (22, 27, 32, 37):
        anchors = [p for p in published if p["anchor"] == "duan_vtm18" and p["qp"] == qp]
        if len(anchors) != 1:
            raise ValueError("Missing or ambiguous published QP.")
        anchor = anchors[0]
        group = [r for r in rows if int(r["qp"]) == qp]
        bpp = statistics.mean(float(r["bpp"]) for r in group)
        value = statistics.mean(float(r["psnr_rgb_db"]) for r in group)
        result.append({"qp": qp, "n_images": 24, "local_bpp": bpp, "published_bpp": anchor["bpp"],
                       "bpp_gap": bpp-anchor["bpp"], "local_psnr_rgb_db": value,
                       "published_psnr_rgb_db": anchor["psnr_rgb_db"],
                       "psnr_gap_db": value-anchor["psnr_rgb_db"]})
    return result


def analyze_replication(root: Path, profile: str = "full") -> dict:
    from .anchors import anchor_points
    folder = root / "runs" / profile / "lossy-vae"
    plan = read_json(folder / "plan.json")
    rows = []
    for job in plan["jobs"]:
        record = verified_result(folder / "jobs" / identity(job["spec"]), job["spec"])
        if record is not None:
            rows.append(record["values"])
    output = root / "results" / "validation" / "lossy-vae" / profile
    write_csv(output / "measurements.csv", rows)
    if len(rows) != len(plan["jobs"]):
        # These are disposable outputs of this module, never retained raw inputs.
        for name in ("same_qp.csv", "psnr_rgb_db_by_qp.png", "bpp_by_qp.png"):
            (output / name).unlink(missing_ok=True)
        result = {"status": "incomplete", "accepted": False, "completed": len(rows),
                  "expected": len(plan["jobs"]), "comparison_available": False}
        write_json(output / "context.json", result)
        return result
    if profile == "full":
        published = anchor_points(read_json(root / "manifests/anchors.json"))
        comparisons = compare_replication(rows, published)
        write_csv(output / "same_qp.csv", comparisons)
        _replication_figures(comparisons, output)
    result = {"status": "complete", "completed": len(rows), "expected": len(plan["jobs"]),
              "retained_streams": sum(r.get("encoding_action") == "retained_stream" for r in rows),
              "comparison_available": profile == "full",
              "scope": "Measured replication differences; completion is not a claim of numerical equality.",
              "configuration_sha256": plan["jobs"][0]["spec"]["config_sha256"],
              "manifest_sha256": sha256(root / "manifests/validation.json"),
              "measurements_sha256": sha256(output / "measurements.csv"),
              "plan_sha256": sha256(folder / "plan.json"), "output": str(output)}
    write_json(output / "context.json", result)
    return result


def plot_replication(root: Path) -> dict:
    """Recreate the two QP plots from a verified compact 96-point measurement grid."""
    from .anchors import anchor_points
    output = root / "results/validation/lossy-vae/full"
    context = read_json(output / "context.json")
    if (context.get("status") != "complete"
            or context.get("measurements_sha256") != sha256(output / "measurements.csv")
            or context.get("manifest_sha256") != sha256(root / "manifests/validation.json")):
        raise ValueError("Replication measurements or configuration failed integrity verification.")
    published = anchor_points(read_json(root / "manifests/anchors.json"))
    comparisons = compare_replication(read_csv(output / "measurements.csv"), published)
    write_csv(output / "same_qp.csv", comparisons)
    _replication_figures(comparisons, output)
    return {"status": "complete", "measured_points": 96, "figures": 2, "encoding": False}


def _replication_figures(rows, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    for metric, label in (("psnr_rgb_db", "PSNR-RGB (dB)"), ("bpp", "BPP (Bits Per Pixel)")):
        fig, axis = plt.subplots(figsize=(10, 6))
        qps = [r["qp"] for r in rows]
        axis.plot(qps, [r["published_"+metric] for r in rows], "o-", color="#111111", label="Duan et al. VTM 18.0")
        axis.plot(qps, [r["local_"+metric] for r in rows], "D--", color="#ff7f0e", label="Local VTM 18.0 replication")
        axis.set(xlabel="QP", ylabel=label, xticks=qps)
        axis.grid(True, linestyle="--", alpha=0.7)
        axis.legend()
        fig.tight_layout()
        fig.savefig(output / (metric + "_by_qp.png"), dpi=150)
        plt.close(fig)


def run_replication(root: Path, profile: str = "full", workers: int = 4) -> dict:
    folder = root / "runs" / profile / "lossy-vae"
    with exclusive_run(folder):
        (folder / "STOP").unlink(missing_ok=True)
        codec = prepare_replication_codec(root)
        jobs = replication_jobs(root, profile, codec)
        write_json(folder / "plan.json", {"jobs": jobs, "session": session_metadata(root)})
        result = run_jobs(jobs, execute_replication, folder, workers=workers, lock_held=True, reset_stop=False)
        if result["status"] == "complete":
            result["analysis"] = analyze_replication(root, profile)
        return result
