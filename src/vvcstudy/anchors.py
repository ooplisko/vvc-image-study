"""Compare measured mean RD points with attributed, published Kodak anchors."""

from pathlib import Path
import math
import statistics

from .artifacts import read_csv, read_json, sha256, write_csv, write_json


def anchor_points(manifest: dict) -> list[dict]:
    points = []
    for source in manifest["sources"]:
        values = source["results"]
        bpps = values["bpp"]
        psnrs = values.get("psnr", values.get("psnr-rgb"))
        if not bpps or len(bpps) != len(psnrs) or len(set(bpps)) != len(bpps):
            raise ValueError("Invalid published anchor grid.")
        msssim = values.get("ms-ssim-rgb")
        if msssim is not None and (len(msssim) != len(bpps) or any(
                not math.isfinite(score) or not 0 <= score <= 1 for score in msssim)):
            raise ValueError("Invalid published MS-SSIM grid.")
        for index, (bpp, psnr) in enumerate(zip(bpps, psnrs, strict=True)):
            if not (math.isfinite(bpp) and bpp > 0 and math.isfinite(psnr)):
                raise ValueError("Nonfinite or nonpositive published measurement.")
            points.append({"anchor": source["id"], "point": index + 1,
                           "qp": index + source["qp_start"] if "qp_start" in source else "",
                           "bpp": bpp, "psnr_rgb_db": psnr, "url": source["url"]})
            if msssim is not None:
                points[-1]["msssim_rgb"] = msssim[index]
    return points


def compare_points(local: list[dict], published: list[dict]) -> list[dict]:
    """Use actual nearest-bpp points only inside each anchor's bpp range."""
    comparisons = []
    for row in local:
        for name in sorted({point["anchor"] for point in published}):
            points = [point for point in published if point["anchor"] == name]
            rate = float(row["bpp"])
            comparison = {**row, "anchor": name}
            if rate < min(p["bpp"] for p in points) or rate > max(p["bpp"] for p in points):
                comparisons.append({**comparison, "status": "outside_published_bpp_range"})
                continue
            point = min(points, key=lambda p: (abs(p["bpp"] - rate), p["bpp"]))
            comparisons.append({**comparison, "status": "ok", "anchor_point": point["point"],
                                "anchor_qp": point["qp"], "anchor_bpp": point["bpp"],
                                "anchor_psnr_rgb_db": point["psnr_rgb_db"],
                                "bpp_gap": rate - point["bpp"],
                                "psnr_gap_db": float(row["psnr_rgb_db"]) - point["psnr_rgb_db"]})
    return comparisons


def analyze_anchors(root: Path, profile: str = "full") -> dict:
    from .results import QPS, SOURCES
    manifest_path = root / "manifests/anchors.json"
    manifest = read_json(manifest_path)
    published = anchor_points(manifest)
    base = root / "results" / profile
    encodings = read_csv(base / "tables/encodings.csv")
    rows = [row for row in encodings
            if row["dataset"] == "kodak" and row["distortion"] == "clean"
            and row["mode"] == "baseline" and row["codec"] == "vtm23-baseline-trace"]
    local = []
    for conversion in ("opencv_444", "ffmpeg_444"):
        for qp in QPS:
            group = [r for r in rows if r["conversion"] == conversion and str(r["qp"]) == qp]
            if len(group) != 24 or {r["source"] for r in group} != set(SOURCES["kodak"]):
                raise ValueError("External anchor comparison requires all 24 clean Kodak images per QP.")
            local.append({"codec": "vtm23", "conversion": conversion, "qp": int(qp), "n_images": 24,
                          "bpp": statistics.mean(float(r["file_bpp"]) for r in group),
                          "psnr_rgb_db": statistics.mean(float(r["psnr_rgb_db"]) for r in group)})
    output = base / "analysis/anchors"
    write_csv(output / "published_points.csv", published)
    write_csv(output / "local_means.csv", local)
    comparisons = compare_points(local, published)
    write_csv(output / "nearest_bpp.csv", comparisons)
    cross_anchor = compare_points(
        [{"compressai_point": p["point"], "bpp": p["bpp"], "psnr_rgb_db": p["psnr_rgb_db"]}
         for p in published if p["anchor"] == "compressai_vtm"],
        [p for p in published if p["anchor"] == "duan_vtm18"])
    write_csv(output / "cross_anchor.csv", cross_anchor)
    write_json(output / "context.json", {
        "anchor_manifest_sha256": sha256(manifest_path),
        "encodings_sha256": sha256(base / "tables/encodings.csv"),
        "scope": "cross-version descriptive comparison, not bit-exact anchor reproduction",
        "aggregation": manifest["aggregation"], "reference": "original RGB8",
        "matching": "actual nearest mean bpp inside published range; retain signed rate and quality gaps",
        "interpolation": False, "extrapolation": False,
        "protocol_limitations": "Codec versions, encoder settings, RGB inverse and output quantization can differ. A PSNR gap at unequal bpp is not an equal-rate gain."})
    comparison_figures(encodings, published, output)
    return {"published_points": len(published), "local_points": len(local), "comparisons": len(comparisons)}


def comparison_figures(encodings: list[dict], published: list[dict], output: Path) -> None:
    """Render the documented comparisons with consistent colors, markers and dashes."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from .results import QPS, SOURCES

    # Labels retain the codec option name CSF; it denotes the experimental matrix.
    styles = (
        ("vtm23-baseline-trace", "ffmpeg_444", "Local VTM 23.0 Baseline (FFmpeg 4:4:4)", "#17becf", "^", "-"),
        ("vtm23-csf-trace", "ffmpeg_444", "Local VTM 23.0 CSF (FFmpeg 4:4:4)", "#8c564b", "P", (0, (5, 2))),
        ("vtm23-baseline-trace", "opencv_444", "Local VTM 23.0 Baseline (OpenCV 4:4:4)", "#ff7f0e", "D", "-"),
        ("vtm23-csf-trace", "opencv_444", "Local VTM 23.0 CSF (OpenCV 4:4:4)", "#9467bd", "X", "-."),
        ("vvenc-baseline-trace", "ffmpeg_420", "Local VVenC Baseline (FFmpeg 4:2:0)", "#2ca02c", "*", (0, (1, 1))),
        ("vvenc-csf-trace", "ffmpeg_420", "Local VVenC CSF (FFmpeg 4:2:0)", "#7f7f7f", "h", (0, (3, 1, 1, 1))),
    )
    local = []
    for codec, conversion, label, color, marker, line in styles:
        points = []
        for qp in QPS:
            group = [r for r in encodings if r["dataset"] == "kodak" and r["distortion"] == "clean"
                     and r["codec"] == codec and r["conversion"] == conversion and str(r["qp"]) == qp]
            if len(group) != 24 or {r["source"] for r in group} != set(SOURCES["kodak"]):
                raise ValueError(f"Incomplete Kodak figure series: {codec} {conversion} QP{qp}")
            points.append({"bpp": statistics.mean(float(r["file_bpp"]) for r in group),
                           **{metric: statistics.mean(float(r[metric]) for r in group)
                              for metric in ("psnr_rgb_db", "msssim_rgb")}})
        local.append((label, points, color, marker, line))
    duan = ("Duan et al. VTM 18.0 Anchor",
            [p for p in published if p["anchor"] == "duan_vtm18"], "#111111", "o", "-")
    compressai = ("CompressAI VTM Anchor (paper: VTM 9.1)",
                  [p for p in published if p["anchor"] == "compressai_vtm"], "#111111", "o", "-")

    def draw(name, title, series, metric="psnr_rgb_db"):
        fig, axis = plt.subplots(figsize=(10, 6), constrained_layout=True)
        for label, points, color, marker, line in series:
            ordered = sorted((p for p in points if metric in p), key=lambda p: float(p["bpp"]))
            if not ordered:
                continue
            axis.plot([float(p["bpp"]) for p in ordered], [float(p[metric]) for p in ordered],
                      color=color, marker=marker, linestyle=line, linewidth=2,
                      markersize=8, label=label, zorder=9 if "Anchor" in label else 6)
        axis.set_title(title, fontsize=14)
        axis.set_xlabel("BPP (Bits Per Pixel)", fontsize=12)
        axis.set_ylabel("PSNR-RGB (dB)" if metric == "psnr_rgb_db" else "MS-SSIM-RGB", fontsize=12)
        axis.grid(True, linestyle="--", alpha=0.7)
        axis.legend(fontsize=9, loc="upper left" if metric == "psnr_rgb_db" else "lower right")
        fig.savefig(output / name, dpi=150)
        plt.close(fig)

    duan_four = (duan[0], [p for p in duan[1] if str(p["qp"]) in QPS], *duan[2:])
    draw("rd_psnr_duan_anchor.png", "Rate-Distortion Comparison: Duan Anchor and Local Codecs (Kodak)",
         [duan_four, *local[2:]])
    draw("rd_psnr_color_conversion.png", "Rate-Distortion Comparison: FFmpeg and OpenCV Paths (Kodak)", local)
    draw("rd_psnr_compressai_anchor.png", "Rate-Distortion Comparison: CompressAI Anchor and Local Codecs (Kodak)",
         [compressai, *local])
    draw("rd_msssim_compressai_anchor.png", "MS-SSIM Comparison: CompressAI Anchor and Local Codecs (Kodak)",
         [compressai, *local], "msssim_rgb")
    secondary = (compressai[0], compressai[1], "#d62728", "s", "--")
    draw("kodak_psnr.png", "Rate-Distortion Comparison: Published VTM Anchors and VTM 23.0 (Kodak)",
         [duan, secondary, local[0], local[2]])
