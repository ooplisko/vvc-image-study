"""Scientific CSV summaries and figures; no generated Markdown reports."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
import statistics

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .artifacts import read_csv, read_json, write_csv


def jpeg_plots(rows: list[dict], output: Path) -> None:
    output.mkdir(parents=True, exist_ok=True)
    for metric in ("psnr_rgb_db", "msssim_rgb"):
        fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
        for axis, dataset in zip(axes, ("kodak", "div2k")):
            for condition in ("clean", "awgn_30_seed_20260811"):
                group = [r for r in rows if r.get("dataset", "kodak") == dataset
                         and r.get("condition", "clean") == condition and r["metric"] == metric
                         and r["status"] == "ok"]
                qps = sorted({int(r["qp"]) for r in group})
                if qps:
                    axis.plot(qps, [statistics.median(float(r["ratio_nearest"]) for r in group
                                                     if int(r["qp"]) == qp) for qp in qps],
                              marker="o", label="Clean" if condition == "clean" else "AWGN, sigma = 30")
            axis.set(title=dataset.upper(), xlabel="QP", ylabel="Median bpp JPEG / bpp VVC")
            axis.set_xticks([22, 27, 32, 37])
            axis.axhline(1, color="gray", linewidth=0.8)
            if axis.get_legend_handles_labels()[0]:
                axis.legend()
        fig.suptitle("Actual JPEG/VVC pairs matched by " +
                     ("PSNR-RGB" if metric == "psnr_rgb_db" else "MS-SSIM-RGB"))
        fig.savefig(output / f"jpeg_vvc_{metric}.png", dpi=180)
        plt.close(fig)


def matrix_plots(paths: list[Path], output: Path) -> None:
    output.mkdir(parents=True, exist_ok=True)
    indexed = {}
    for path in paths:
        for row in read_csv(path):
            image = row.get("image") or row.get("source", "")
            if not image.startswith("kodim") or row.get("mode") not in {"baseline", "csf"}:
                continue
            if row.get("conversion") and not row["conversion"].startswith("ffmpeg"):
                continue
            codec = ("vvenc" if "vvenc" in row.get("codec", "") else "vtm") if row.get("codec") else (
                "vvenc" if "vvenc" in path.parts else "vtm")
            key = codec, Path(image).stem, int(row["qp"]), row["mode"]
            indexed[key] = row
    deltas = []
    for (codec, image, qp, mode), baseline in indexed.items():
        if mode != "baseline" or (codec, image, qp, "csf") not in indexed:
            continue
        csf = indexed[codec, image, qp, "csf"]
        b0, b1 = (float(r.get("bpp") or r["file_bpp"]) for r in (baseline, csf))
        y0, y1 = (float(r.get("psnr_y") or r["psnr_y_db"]) for r in (baseline, csf))
        deltas.append({"codec": codec, "image": image, "qp": qp,
                       "bpp_change_percent": 100 * (b1 / b0 - 1), "psnr_y_change_db": y1 - y0})
    write_csv(output / "same_qp_deltas.csv", deltas)
    groups = defaultdict(list)
    for row in deltas:
        groups[row["codec"], row["qp"]].append(row)
    write_csv(output / "same_qp_summary.csv", [
        {"codec": key[0], "qp": key[1], "n_images": len(group),
         "mean_bpp_change_percent": statistics.mean(r["bpp_change_percent"] for r in group),
         "mean_psnr_y_change_db": statistics.mean(r["psnr_y_change_db"] for r in group)}
        for key, group in sorted(groups.items())])
    for codec, image in sorted({(key[0], key[1]) for key in indexed}):
        fig, axes = plt.subplots(1, 2, figsize=(9, 4), constrained_layout=True)
        for axis, metric in zip(axes, ("psnr_rgb", "msssim_rgb")):
            for mode in ("baseline", "csf"):
                rows = [row for key, row in sorted(indexed.items()) if key[0] == codec and key[1] == image and key[3] == mode]
                axis.scatter([float(r.get("bpp") or r["file_bpp"]) for r in rows],
                             [float(r.get(metric, r.get(metric + "_db"))) for r in rows], label=mode)
            axis.set(xlabel="bpp", ylabel=metric)
            axis.legend()
        fig.suptitle(f"{codec}: {image}; actual encoded points")
        fig.savefig(output / f"{codec}_{image}.png", dpi=150)
        plt.close(fig)


def partition_plots(rows: list[dict], output: Path) -> None:
    output.mkdir(parents=True, exist_ok=True)
    descriptors = ("sobel_si", "luma_sd", "edge_fraction", "glcm_contrast", "glcm_entropy", "glcm_homogeneity")
    for dataset in sorted({row.get("dataset", "div2k") for row in rows}):
        group = [r for r in rows if r.get("dataset", "div2k") == dataset and r.get("distortion") == "clean"
                 and int(r["qp"]) == 32 and r.get("mode", "baseline") == "baseline"
                 and r.get("conversion", "opencv_444") == "opencv_444"]
        if not group:
            continue
        fig, axes = plt.subplots(2, 3, figsize=(11, 6), constrained_layout=True)
        for axis, feature in zip(axes.ravel(), descriptors):
            axis.scatter([float(r[feature]) for r in group], [float(r["cu_count"]) for r in group], s=15)
            axis.set(xlabel=feature, ylabel="Final luma CU count")
        fig.suptitle(f"{dataset}: clean images, QP 32, n={len(group)}")
        fig.savefig(output / f"{dataset}_clean_qp32.png", dpi=180)
        plt.close(fig)


def partition_overlay(rgb: np.ndarray, rows: list[dict], path: Path) -> None:
    from matplotlib.collections import LineCollection
    lines = []
    for row in rows:
        x, y, w, h = (int(row[key]) for key in ("x", "y", "width", "height"))
        lines.extend([[(x, y), (x + w, y)], [(x, y), (x, y + h)],
                      [(x + w, y), (x + w, y + h)], [(x, y + h), (x + w, y + h)]])
    fig, axis = plt.subplots(figsize=(8, 6), constrained_layout=True)
    axis.imshow(rgb)
    axis.add_collection(LineCollection(lines, colors="red", linewidths=0.4))
    axis.set(xlim=(0, rgb.shape[1]), ylim=(rgb.shape[0], 0), xlabel="x (pixels)", ylabel="y (pixels)")
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def partition_examples(root: Path, run_dir: Path, output: Path) -> int:
    """Render fixed Kodak08 examples from actual new traces and stimulus pixels."""
    from .encoding import read_rgb
    planned = {job["job_id"] for plan in (run_dir / "plans").glob("*.json")
               for job in read_json(plan)["jobs"]}
    count = 0
    for path in (run_dir / "jobs").glob("*/result.json"):
        record = read_json(path)
        spec = record["spec"]
        if (record["job_id"] not in planned or spec["kind"] != "encode" or not spec["trace"]
                or spec["dataset"] != "kodak" or spec["source"] != "kodim08.png"):
            continue
        distortion = spec["distortion"]
        if distortion != "clean" and not (spec["qp"] == 32 and (
                (distortion == "awgn" and spec["level"] == 30 and spec["seed"] == 20260811)
                or (distortion == "stripes" and spec["level"] == 32))):
            continue
        image = (root / "data/originals/kodak/kodim08.png" if distortion == "clean" else
                 root / "data/stimuli" / (record["values"]["stimulus"] + ".png"))
        label = f"{spec['codec']}_{spec['conversion']}_{distortion}{spec['level']}_QP{spec['qp']}"
        partition_overlay(read_rgb(image), read_csv(path.parent / "partitions.csv"), output / (label + ".png"))
        count += 1
    return count
