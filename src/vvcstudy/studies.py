"""Experiment grids, deterministic inputs and resumable execution."""

from __future__ import annotations

from pathlib import Path
import shutil
import tomllib

import cv2
import numpy as np
from PIL import Image

from .artifacts import identity, read_csv, read_json, sha256, write_csv, write_json
from .datasets import prepare_datasets
from .descriptors import glcm_features, spatial_complexity_metrics
from .encoding import execute_encode, execute_jpeg, jpeg_environment, read_rgb
from .matching import actual_match, frontier
from .runner import exclusive_run, run_jobs, session_metadata
from .stimuli import _to_luma, add_achromatic_awgn, add_sinusoidal_interference, derive_image_seed, luma_rms_difference


def calculation_identity() -> dict:
    """Invalidate cached measurements when their calculation implementation changes."""
    folder = Path(__file__).parent
    return {"numpy": np.__version__, "opencv": cv2.__version__,
            "modules": {name: sha256(folder / (name + ".py")) for name in
                        ("stimuli", "descriptors", "metrics", "partitions", "encoding")}}


def config(root: Path, profile: str) -> dict:
    if profile not in {"full", "smoke"}:
        raise ValueError("Profile must be full or smoke.")
    with (root / "configs" / f"{profile}.toml").open("rb") as stream:
        return tomllib.load(stream)


def plan(root: Path, profile: str) -> dict:
    settings = config(root, profile)
    if profile == "smoke":
        return {"profile": profile, "synthetic_inputs": 2, "VTM_encodes": 4,
                "JPEG_encodes": 12, "full_reproduction": False}
    div2k = read_json(root / "manifests/datasets/div2k.json")["images"]
    div_area = sum(row["width"] * row["height"] for row in div2k)
    # Input8 + encoder reconstruction10 + decoded10; trace/PNG/JPEG/builds are extra.
    yuv_bytes = 1248 * 393216 * 15 + div_area * 8 * 15 + 192 * 393216 * (15 + 7.5)
    return {"profile": profile, "kodak_images": 24, "div2k_full_images": 100,
            "kodak_partition_encodes": 1248,
            "div2k_partition_encodes": 800, "matrix_encode_points": 384,
            "matrix_trace_points_included": 96, "jpeg_inputs": 248,
            "jpeg_encodes": 24800, "quality_matches": 1984,
            "timing_vtm_encodes_excluding_warmups": 288,
            "bootstrap_resamples": settings["bootstrap_resamples"],
            "uncompressed_yuv_storage_gib_excluding_traces_and_other_files": round(yuv_bytes / 1024**3, 2),
            "current_free_gib": round(shutil.disk_usage(root).free / 1024**3, 2),
            "encoding_repeated_by_plan": False}


def synthetic_inputs(root: Path) -> list[dict]:
    result = []
    for index, (width, height) in enumerate(((96, 96), (97, 99))):
        y, x = np.indices((height, width))
        rgb = np.stack(((3 * x + y) % 256, (x + 2 * y) % 256, (x * y) % 256), axis=-1).astype("uint8")
        path = root / "data" / "synthetic" / f"smoke{index}.png"
        path.parent.mkdir(parents=True, exist_ok=True)
        if not path.exists():
            Image.fromarray(rgb).save(path)
        if not np.array_equal(read_rgb(path), rgb):
            raise ValueError("Synthetic test fixture changed.")
        result.append({"id": path.stem, "path": str(path), "sha256": sha256(path),
                       "width": width, "height": height})
    return result


def prepare_inputs(root: Path, dataset: str, conditions: list[tuple], profile: str) -> list[dict]:
    originals = (synthetic_inputs(root) if dataset == "synthetic"
                 else prepare_datasets(root, [dataset])[dataset])
    result = []
    for index, original in enumerate(originals):
        original_rgb = read_rgb(Path(original["path"]))
        source_sobel = spatial_complexity_metrics(original_rgb[:, :, ::-1])["sobel_si"]
        for distortion, level, seed in conditions:
            key = {"dataset": dataset, "source": original["id"] + ".png", "distortion": distortion,
                   "level": level, "seed": seed, "original_sha256": original["sha256"],
                   "width": original["width"], "height": original["height"]}
            if distortion == "clean":
                rgb, path = original_rgb, Path(original["path"])
            else:
                path = root / "data" / "stimuli" / (identity(key) + ".png")
                marker = path.with_suffix(".json")
                if (path.exists() and marker.exists() and read_json(marker).get("spec") == key
                        and read_json(marker).get("sha256") == sha256(path)):
                    rgb = read_rgb(path)
                else:
                    if distortion == "awgn":
                        rgb = add_achromatic_awgn(original_rgb, level, derive_image_seed(seed, index))
                    elif distortion == "stripes":
                        rgb = add_sinusoidal_interference(original_rgb, level)
                    else:
                        raise ValueError(f"Unknown distortion: {distortion}")
                    path.parent.mkdir(parents=True, exist_ok=True)
                    ok, data = cv2.imencode(".png", rgb[:, :, ::-1], [cv2.IMWRITE_PNG_COMPRESSION, 3])
                    if not ok:
                        raise RuntimeError("PNG stimulus encoding failed.")
                    temporary = path.with_suffix(".partial")
                    temporary.write_bytes(data.tobytes())
                    temporary.replace(path)
                    write_json(marker, {"spec": key, "sha256": sha256(path)})
            sensitivity = {}
            if dataset == "kodak":
                luma = _to_luma(rgb[:, :, ::-1])
                for suffix, values in (("levels32", glcm_features(luma, 32)),
                                       ("horizontal", glcm_features(luma, offsets=((0, 1),)))):
                    sensitivity.update({name + "_" + suffix: value for name, value in values.items()})
            result.append({**key, "path": str(path), "input_sha256": sha256(path),
                           "features": {**spatial_complexity_metrics(rgb[:, :, ::-1]),
                                        **sensitivity, "source_sobel_si": source_sobel,
                                        "actual_luma_rms": luma_rms_difference(original_rgb[:, :, ::-1], rgb[:, :, ::-1])}})
    return result


def conditions_for(study: str, settings: dict) -> list[tuple]:
    if study == "matrices":
        return [("clean", 0, None)]
    if study in {"div2k", "jpeg-vvc"} or settings["name"] == "smoke":
        return [("clean", 0, None), ("awgn", 30, 20260811)]
    if study == "partitions":
        return [("clean", 0, None)] + [
            ("awgn", sigma, seed) for seed in settings["awgn_seeds"] for sigma in settings["awgn_sigmas"]
        ] + [("stripes", amplitude, None) for amplitude in settings["sine_amplitudes"]]
    raise ValueError(f"Unknown study: {study}")


def encode_jobs(root: Path, inputs: list[dict], settings: dict, codecs: dict, *, matrices=False) -> list[dict]:
    jobs = []
    cfg = root / "configs" / "codecs" / "vtm_intra.cfg"
    modes = [("vtm23-baseline-trace", "baseline", "opencv_444")]
    if matrices:
        modes = [("vtm23-baseline-trace", "baseline", "ffmpeg_444"),
                 ("vtm23-csf-trace", "csf", "ffmpeg_444"),
                 ("vvenc-baseline-trace", "baseline", "ffmpeg_420"),
                 ("vvenc-csf-trace", "csf", "ffmpeg_420")]
    for item in inputs:
        for qp in settings["qps"]:
            for name, mode, conversion in modes:
                codec = codecs[name]
                spec = {key: item[key] for key in ("dataset", "source", "distortion", "level", "seed",
                                                   "input_sha256", "width", "height")}
                spec.update(kind="encode", protocol=1, qp=qp, codec=name, mode=mode,
                            calculation=calculation_identity(),
                            conversion=conversion, trace=not matrices or qp == 32,
                            padding=item["dataset"] in {"div2k", "synthetic"},
                            encoder_sha256=sha256(Path(codec["encoder"])),
                            decoder_sha256=sha256(Path(codec["decoder"])),
                            config_sha256=sha256(cfg) if name.startswith("vtm") else None)
                if "runtime_hashes" in codec:
                    spec["runtime_hashes"] = codec["runtime_hashes"]
                if conversion.startswith("ffmpeg"):
                    ffmpeg = shutil.which("ffmpeg")
                    if not ffmpeg:
                        raise RuntimeError("FFmpeg is required for the matrix protocol.")
                    spec["ffmpeg_sha256"] = sha256(Path(ffmpeg))
                else:
                    ffmpeg = None
                jobs.append({"spec": spec, "input": item["path"], "features": item["features"],
                             "config": str(cfg), "encoder": codec["encoder"], "decoder": codec["decoder"],
                             "ffmpeg": ffmpeg or "ffmpeg",
                             "label": f"{item['source']} {item['distortion']} QP{qp} {name}"})
    return jobs


def jpeg_jobs(inputs: list[dict], settings: dict) -> list[dict]:
    qfs = (list(range(settings["jpeg_qfs"][0], settings["jpeg_qfs"][1] + 1))
           if settings["name"] == "full" else settings["jpeg_qfs"])
    result = []
    for item in inputs:
        spec = {key: item[key] for key in ("dataset", "source", "distortion", "level", "seed",
                                           "input_sha256", "width", "height")}
        spec.update(kind="jpeg", protocol=1, qfs=qfs, implementation=jpeg_environment(),
                    calculation=calculation_identity())
        result.append({"spec": spec, "input": item["path"], "features": item["features"],
                       "label": f"JPEG {item['source']} {item['distortion']}"})
    return result


def merge_rows(path: Path, rows: list[dict], key) -> list[dict]:
    """Update measured points without removing results from other completed stages."""
    merged = {key(row): row for row in read_csv(path)} if path.exists() else {}
    merged.update({key(row): row for row in rows})
    values = [merged[name] for name in sorted(merged)]
    write_csv(path, values)
    return values


def export_results(run_dir: Path, output: Path | None = None) -> dict:
    from .results import encoding_key
    output = output or run_dir
    encodes, jpegs = [], {}
    planned = {job["job_id"] for path in (run_dir / "plans").glob("*.json")
               for job in read_json(path)["jobs"]}
    for path in sorted((run_dir / "jobs").glob("*/result.json")):
        record = read_json(path)
        if record.get("status") != "complete" or (planned and record["job_id"] not in planned):
            continue
        spec = record["spec"]
        if spec["kind"] == "encode":
            values = record["values"].copy()
            if "psnr_rgb" in values:
                values["psnr_rgb_db"] = values.pop("psnr_rgb")
            encodes.append({**values, "job_id": record["job_id"],
                            "input_sha256": spec["input_sha256"],
                            "stream": (path.parent / "stream.vvc").relative_to(run_dir.parent.parent).as_posix(),
                            "stream_sha256": next(x["sha256"] for x in record["artifacts"] if x["path"] == "stream.vvc")})
        elif spec["kind"] == "jpeg":
            jpegs[spec["dataset"], spec["source"], spec["input_sha256"]] = (record, path.parent)
    encodes = merge_rows(output / "tables/encodings.csv", encodes, encoding_key)
    matched = []
    measurements = []
    for key, (jpeg, folder) in jpegs.items():
        candidates = [row for row in encodes if (row["dataset"], row["source"], row["input_sha256"]) == key
                      and row["mode"] == "baseline" and row["conversion"] == "opencv_444"]
        vtm_points = [{"parameter": row["qp"], "bpp": row["file_bpp"], "file": row["stream"],
                       "sha256": row["stream_sha256"], "psnr_rgb_db": float(row["psnr_rgb_db"]),
                       "msssim_rgb": float(row["msssim_rgb"])} for row in candidates]
        jpeg_points = [{**row, "file": (folder / row["file"]).relative_to(run_dir.parent.parent).as_posix(),
                        "psnr_rgb_db": float(row["psnr_rgb_db"]), "msssim_rgb": float(row["msssim_rgb"])}
                       for row in jpeg["measurements"]]
        spec = jpeg["spec"]
        condition = "clean" if spec["distortion"] == "clean" else "awgn_30_seed_20260811"
        common = {"dataset": spec["dataset"], "condition": condition, "source": spec["source"],
                  "width": spec["width"], "height": spec["height"]}
        measurements.extend({**common, **point} for point in jpeg_points)
        for row in candidates:
            measurements.append({**common, "codec": "VTM", "parameter": row["qp"],
                                 "bytes": row["bitstream_bytes"], "bpp": row["file_bpp"],
                                 "file": row.get("stream", ""), "sha256": row["stream_sha256"],
                                 "psnr_rgb_db": row["psnr_rgb_db"], "msssim_rgb": row["msssim_rgb"]})
        for metric in ("psnr_rgb_db", "msssim_rgb"):
            jpeg_curve, _ = frontier(jpeg_points, metric)
            vtm_curve, _ = frontier(vtm_points, metric)
            for point in vtm_points:
                spec = jpeg["spec"]
                values = (actual_match(jpeg_curve, metric, point[metric]) if point in vtm_curve
                          else {"status": "vtm_point_excluded_from_frontier"})
                row = {"dataset": spec["dataset"], "condition": "clean" if spec["distortion"] == "clean" else "awgn_30_seed_20260811",
                       "source": spec["source"], "metric": metric, "qp": point["parameter"],
                       "target_quality": point[metric], "vtm_bpp": point["bpp"],
                       "vtm_file": point["file"], "vtm_sha256": point["sha256"],
                       "stimulus_sha256": spec["input_sha256"], **jpeg["features"], **values}
                if row["status"] == "ok":
                    for label in ("nearest", "lower", "upper"):
                        row["ratio_" + label] = row[label + "_bpp"] / point["bpp"]
                matched.append(row)
    if matched:
        merge_rows(output / "tables/actual_by_qp.csv", matched, lambda row: (
            row["dataset"], row["condition"], row["source"], row["metric"], int(row["qp"])))
    if measurements:
        merge_rows(output / "tables/measurements.csv", measurements, lambda row: (
            row["dataset"], row["condition"], row["source"], row["codec"], int(row["parameter"])))
    return {"encode_rows": len(encodes), "matched_rows": len(matched)}


def run_study(root: Path, profile: str, study: str, codecs: dict, workers: int,
              *, reset_stop: bool = True) -> dict:
    run_dir = root / "runs" / profile
    with exclusive_run(run_dir):
        if reset_stop:
            (run_dir / "STOP").unlink(missing_ok=True)
        if (run_dir / "STOP").exists():
            return {"status": "stopped"}
        return _run_study_locked(root, profile, study, codecs, workers)


def _run_study_locked(root: Path, profile: str, study: str, codecs: dict, workers: int) -> dict:
    settings = config(root, profile)
    run_dir = root / "runs" / profile
    session = session_metadata(root)
    write_json(run_dir / "sessions" / (session["started_utc"].replace(":", "-") + ".json"), session)
    datasets = (["synthetic"] if profile == "smoke" else
                ["div2k"] if study == "div2k" else ["kodak", "div2k"] if study == "jpeg-vvc" else ["kodak"])
    inputs = [item for dataset in datasets for item in prepare_inputs(
        root, dataset, conditions_for(study, settings), profile)]
    jobs = encode_jobs(root, inputs, settings, codecs, matrices=study == "matrices")
    all_jobs = jobs + (jpeg_jobs(inputs, settings) if study == "jpeg-vvc" else [])
    write_json(run_dir / "plans" / (study + ".json"), {
        "profile": profile, "study": study, "config": settings,
        "jobs": [{"job_id": identity(job["spec"]), "spec": job["spec"]} for job in all_jobs]})
    result = run_jobs(jobs, execute_encode, run_dir, workers=workers, lock_held=True, reset_stop=False)
    stages = {"encoding": result}
    if study == "jpeg-vvc" and result["status"] == "complete":
        result = run_jobs(jpeg_jobs(inputs, settings), execute_jpeg, run_dir, workers=workers,
                          lock_held=True, reset_stop=False)
        stages["jpeg"] = result
    result = {**result, "stages": stages}
    result["tables"] = export_results(run_dir, root / "results" / profile)
    if profile == "full" and result["status"] == "complete":
        from .results import seal_results
        result["verification"] = seal_results(root, profile, [study])
    return result
