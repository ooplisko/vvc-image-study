"""Fixed-condition CU associations for Kodak and full-resolution DIV2K.

Kodak retains paired image bootstrap, permutation families, oriented contrasts,
and sensitivity analyses. DIV2K CU associations use paired image bootstrap.
Input tables already contain
measured descriptors; this module neither encodes images nor generates reports.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np

from .artifacts import read_csv, sha256, write_csv, write_json
from .descriptors import FEATURES
from .statistics import (
    holm_adjust, paired_spearman_bootstrap, percentile_intervals,
    simultaneous_basic_intervals, spearman_independence_permutation,
    spearman_last_axis,
)

QPS = (22, 27, 32, 37)
SIGNS = dict(zip(FEATURES, (1, 1, 1, 1, 1, -1), strict=True))
CONDITIONS = (("clean", "0", ""), ("awgn", "30", "20260811"))
TARGETS = ("cu_density_per_mpixel", "file_bpp")
SETTINGS = {
    "expected_sources": 24,
    "expected_correlation_cells": 312,
    "expected_rq2_contrasts": 144,
    "permutation_holm_family_size": 168,
    "bootstrap_resamples": 99_999,
    "bootstrap_seed": 20260905,
    "permutation_resamples": 99_999,
    "permutation_seed": 20260906,
    "contrast_family_alpha": 0.025,
    "pointwise_confidence": 0.95,
    "glcm_sensitivity": ({"name": "levels32"}, {"name": "horizontal"}),
}


def normalize_measurements(rows: list[dict]) -> dict[str, list[dict]]:
    """Select baseline OpenCV VTM rows and validate numerical measurement schemas.

    This validates the numeric table, not the underlying encoded artifacts.
    Artifact hashes and decoder equality remain the producer's responsibility.
    Existing producer verification flags, when supplied, must be true.
    """
    selected: dict[str, list[dict]] = {}
    identities = {}
    for raw in rows:
        if raw.get("mode", "baseline") != "baseline" or raw.get("conversion", "opencv_444") != "opencv_444":
            continue
        if raw.get("codec") and not str(raw["codec"]).startswith("vtm"):
            continue
        row = dict(raw)
        dataset = str(row.get("dataset", ""))
        if dataset == "div2k_validation_full100":
            dataset = "div2k"
        if dataset not in {"kodak", "div2k", "synthetic"}:
            raise ValueError(f"Unsupported partition dataset: {dataset}")
        row["dataset"] = dataset
        row["seed"] = "" if row.get("seed") in (None, "", "None") else str(int(float(row["seed"])))
        row["level"] = str(int(float(row["level"])))
        row["qp"] = int(row["qp"])
        for key in (*FEATURES, "coded_area", "cu_count", "cu_density_per_mpixel"):
            row[key] = float(row[key])
            if not math.isfinite(row[key]):
                raise ValueError(f"Non-finite partition measurement: {key}")
        area, count, density = (row[key] for key in ("coded_area", "cu_count", "cu_density_per_mpixel"))
        if area <= 0 or not area.is_integer() or count <= 0 or not count.is_integer() or not math.isclose(
                density, 1_000_000 * count / area, rel_tol=1e-12, abs_tol=0):
            raise ValueError("CU count, coded area and density are inconsistent")
        row["cu_count"] = int(count)
        if not row.get("source_area") and row.get("width") and row.get("height"):
            row["source_area"] = int(row["width"]) * int(row["height"])
        if row.get("source_area"):
            row["source_area"] = int(row["source_area"])
            if row["source_area"] <= 0 or row["source_area"] > area:
                raise ValueError("Invalid source/coded area")
        if not row.get("file_bpp") and row.get("bitstream_bytes") and row.get("source_area"):
            row["file_bpp"] = 8 * int(row["bitstream_bytes"]) / row["source_area"]
        if row.get("file_bpp"):
            row["file_bpp"] = float(row["file_bpp"])
            if not math.isfinite(row["file_bpp"]) or row["file_bpp"] <= 0:
                raise ValueError("Invalid bitstream bpp")
            if row.get("bitstream_bytes") and row.get("source_area") and not math.isclose(
                    row["file_bpp"], 8 * int(row["bitstream_bytes"]) / row["source_area"], rel_tol=1e-12):
                raise ValueError("bpp differs from full file bytes per source pixel")
        for key in ("reconstruction_verified", "cu_coverage_verified"):
            if row.get(key) not in (None, "") and str(row[key]).lower() != "true":
                raise ValueError(f"Producer verification failed: {key}")
        identity_key = (dataset, row["source"], row["distortion"], row["level"], row["seed"])
        image_hash = row.get("image_sha256") or row.get("input_sha256") or ""
        identity = (image_hash, *(row[key] for key in FEATURES))
        if identities.setdefault(identity_key, identity) != identity:
            raise ValueError("Input identity or descriptors changed between QPs")
        selected.setdefault(dataset, []).append(row)
    if not selected:
        raise ValueError("No baseline OpenCV VTM partition measurements")
    return selected


def validate_measurements(rows: list[dict]) -> None:
    """Require the complete Kodak experiment grid before inferential statistics."""
    conditions = {("clean", 0, "", qp) for qp in QPS}
    conditions |= {("stripes", level, "", qp) for level in (8, 16, 32) for qp in QPS}
    conditions |= {("awgn", level, seed, qp) for level in (5, 15, 30)
                   for seed in ("20260811", "20260812", "20260813") for qp in QPS}
    expected = {(f"kodim{i:02d}.png", *condition) for i in range(1, 25) for condition in conditions}
    keys = [(row["source"], *condition_key(row)) for row in rows]
    if len(keys) != len(expected) or set(keys) != expected:
        raise ValueError("Expected every Kodak image/condition at QP 22, 27, 32 and 37 exactly once")
    if any(float(row["coded_area"]) != 393216 for row in rows):
        raise ValueError("Kodak coded area must be 393216 pixels")


def condition_key(row: dict[str, object]) -> tuple[str, int, str, int]:
    return str(row["distortion"]), int(float(row["level"])), str(row["seed"]), int(row["qp"])


def build_cells(
    rows: list[dict[str, object]], expected_sources: int = 24
) -> tuple[list[dict[str, object]], list[str], np.ndarray, np.ndarray]:
    """Align each feature/condition with the same complete, sorted image sample."""

    sources = sorted({str(row["source"]) for row in rows})
    if len(sources) != expected_sources:
        raise ValueError("Incomplete source set")
    groups: dict[tuple[str, int, str, int], dict[str, dict[str, object]]] = {}
    for row in rows:
        group = groups.setdefault(condition_key(row), {})
        source = str(row["source"])
        if source in group:
            raise ValueError("Duplicate source in an encoding condition")
        group[source] = row
    cells, x, y = [], [], []
    for condition, group in sorted(groups.items()):
        if sorted(group) != sources:
            raise ValueError(f"Incomplete condition: {condition}")
        distortion, level, seed, qp = condition
        primary = distortion != "awgn" or seed == "20260811"
        for feature in FEATURES:
            cells.append({"distortion": distortion, "level": level, "seed": seed, "qp": qp,
                          "feature": feature, "primary": primary, "n_images": len(sources)})
            x.append([float(group[source][feature]) for source in sources])
            y.append([float(group[source]["cu_density_per_mpixel"]) for source in sources])
    return cells, sources, np.asarray(x), np.asarray(y)


def contrasts(
    cells: list[dict[str, object]], observed: np.ndarray, distribution: np.ndarray
) -> tuple[list[dict[str, object]], np.ndarray, np.ndarray]:
    lookup = {(*condition_key(cell), cell["feature"]): i for i, cell in enumerate(cells)}
    metadata, values, sampled = [], [], []
    for i, cell in enumerate(cells):
        if not cell["primary"]:
            continue
        feature, qp = str(cell["feature"]), int(cell["qp"])
        sign = SIGNS[feature]
        if cell["distortion"] == "clean":
            if feature == "sobel_si":
                continue
            reference = lookup[("clean", 0, "", qp, "sobel_si")]
            family, reference_sign = "RQ1", 1
        else:
            reference = lookup[("clean", 0, "", qp, feature)]
            family, reference_sign = "RQ2", sign
        metadata.append({**cell, "family": family, "reference_index": reference,
                         "cell_index": i, "raw_rho_difference": observed[i] - observed[reference]})
        values.append(sign * observed[i] - reference_sign * observed[reference])
        sampled.append(sign * distribution[:, i] - reference_sign * distribution[:, reference])
    return metadata, np.asarray(values), np.asarray(sampled).T


def interval_row(intervals: dict, index: int, prefix: str = "") -> dict[str, object]:
    return {f"{prefix}{key}": value[index].item() for key, value in intervals.items()
            if isinstance(value, np.ndarray)}


def run_statistics(output: Path, config: dict[str, object]) -> None:
    rows = read_csv(output / "joined_measurements.csv")
    validate_measurements(rows)
    cells, sources, x, y = build_cells(rows, int(config["expected_sources"]))
    primary = np.array([cell["primary"] for cell in cells], dtype=bool)
    expected_cells = int(config.get("expected_correlation_cells", 168))
    expected_primary = int(config["permutation_holm_family_size"])
    if len(cells) != expected_cells or primary.sum() != expected_primary:
        raise ValueError("Correlation cells do not match the registered primary/supplementary counts")
    print(f"Computing paired image bootstrap for {len(cells)} correlation cells", flush=True)
    observed, distribution = paired_spearman_bootstrap(
        x, y, n_resamples=int(config["bootstrap_resamples"]), seed=int(config["bootstrap_seed"]))
    np.savez_compressed(output / "bootstrap_correlations.npz", observed=observed, distribution=distribution)
    ci = percentile_intervals(observed, distribution, confidence_level=float(config["pointwise_confidence"]))
    print(f"Computing independence permutations for {expected_primary} primary cells", flush=True)
    permutation = spearman_independence_permutation(
        x[primary], y[primary], n_resamples=int(config["permutation_resamples"]),
        seed=int(config["permutation_seed"]))
    np.savez_compressed(output / "permutation_correlations.npz", **permutation)
    p, adjusted = np.full(len(cells), np.nan), np.full(len(cells), np.nan)
    p[primary], adjusted[primary] = permutation["pvalue"], holm_adjust(permutation["pvalue"])
    correlation_rows = []
    for i, cell in enumerate(cells):
        correlation_rows.append({**cell, "rho": observed[i], "oriented_rho": SIGNS[cell["feature"]] * observed[i],
                                 **interval_row(ci, i, "ci_"), "permutation_p": p[i], "holm_p": adjusted[i],
                                 "unique_feature_values": np.unique(x[i]).size,
                                 "unique_density_values": np.unique(y[i]).size})
    write_csv(output / "correlations.csv", correlation_rows)
    meta, values, draws = contrasts(cells, observed, distribution)
    pointwise = percentile_intervals(values, draws)
    simultaneous_low, simultaneous_high = np.full(values.shape, np.nan), np.full(values.shape, np.nan)
    critical_values = {}
    for family, expected in (("RQ1", 20), ("RQ2", int(config.get("expected_rq2_contrasts", 108)))):
        selected = np.array([row["family"] == family for row in meta])
        if selected.sum() != expected:
            raise ValueError(f"Unexpected {family} contrast count")
        result = simultaneous_basic_intervals(values[selected], draws[:, selected],
                                              alpha=float(config["contrast_family_alpha"]))
        simultaneous_low[selected], simultaneous_high[selected] = result["low"], result["high"]
        critical_values[family] = result["critical_value"]
    contrast_rows = [{**row, "oriented_difference": values[i], **interval_row(pointwise, i, "pointwise_"),
                      "simultaneous_low": simultaneous_low[i], "simultaneous_high": simultaneous_high[i]}
                     for i, row in enumerate(meta)]
    write_csv(output / "contrasts.csv", contrast_rows)

    influence = []
    for source_index, source in enumerate(sources):
        keep = np.arange(len(sources)) != source_index
        omitted = spearman_last_axis(x[:, keep], y[:, keep])
        leave_meta, leave_values, _ = contrasts(cells, omitted, omitted[None, :])
        influence.extend({"omitted_source": source, **row, "n_images": int(keep.sum()),
                          "oriented_difference": leave_values[i]}
                         for i, row in enumerate(leave_meta))
    write_csv(output / "leave_one_out_contrasts.csv", influence)

    # Supplementary paired changes use the same shared index sequence, no formal tests.
    lookup = {(*condition_key(cell), cell["feature"]): i for i, cell in enumerate(cells)}
    delta_cells, dx, dy, delta_rows = [], [], [], []
    for i, cell in enumerate(cells):
        if not cell["primary"] or cell["distortion"] == "clean":
            continue
        ref = lookup[("clean", 0, "", int(cell["qp"]), cell["feature"])]
        delta_x = SIGNS[cell["feature"]] * (x[i] - x[ref])
        delta_y = np.log(y[i] / y[ref])
        delta_cells.append(cell)
        dx.append(delta_x)
        dy.append(delta_y)
        delta_rows.extend({**cell, "source": source, "oriented_feature_change": delta_x[j],
                           "log_density_ratio": delta_y[j], "density_change_percent": 100 * (y[i, j] / y[ref, j] - 1)}
                          for j, source in enumerate(sources))
    write_csv(output / "paired_changes.csv", delta_rows)
    print("Computing supplementary paired-change bootstrap", flush=True)
    delta_observed, delta_draws = paired_spearman_bootstrap(
        np.asarray(dx), np.asarray(dy), n_resamples=int(config["bootstrap_resamples"]),
        seed=int(config["bootstrap_seed"]))
    np.savez_compressed(output / "bootstrap_paired_changes.npz", observed=delta_observed, distribution=delta_draws)
    delta_ci = percentile_intervals(delta_observed, delta_draws)
    write_csv(output / "paired_change_correlations.csv", [
        {**cell, "rho": delta_observed[i], **interval_row(delta_ci, i, "ci_")}
        for i, cell in enumerate(delta_cells)])

    sensitivity_rows = []
    by_condition = {(str(row["source"]), *condition_key(row)): row for row in rows}
    for i, cell in enumerate(cells):
        if not cell["primary"] or not str(cell["feature"]).startswith("glcm_"):
            continue
        for variant in config["glcm_sensitivity"]:
            feature = f"{cell['feature']}_{variant['name']}"
            if not all(by_condition[(source, *condition_key(cell))].get(feature) not in (None, "") for source in sources):
                continue
            variant_x = [float(by_condition[(source, *condition_key(cell))][feature]) for source in sources]
            sensitivity_rows.append({**cell, "variant": variant["name"], "rho_primary": observed[i],
                                     "rho_variant": float(spearman_last_axis(variant_x, y[i]))})
    write_csv(output / "parameter_sensitivity.csv", sensitivity_rows)
    # Inter-feature redundancy on all clean images, separate from endpoint tests.
    clean = [row for row in rows if row["distortion"] == "clean" and int(row["qp"]) == 22]
    feature_associations = [{"feature_a": a, "feature_b": b,
                             "rho": float(spearman_last_axis([float(r[a]) for r in clean], [float(r[b]) for r in clean]))}
                            for a in FEATURES for b in FEATURES]
    write_csv(output / "clean_feature_associations.csv", feature_associations)
    print(f"Saved {len(cells)} correlations and {len(meta)} comparisons; "
          f"simultaneous critical values: {critical_values}", flush=True)


def descriptive_tables(rows: list[dict], qps: tuple[int, ...] = QPS) -> tuple[list[dict], list[dict], list[dict]]:
    """Paired DIV2K density/bpp summaries; QPs and conditions are not pooled."""
    sources = sorted({row["source"] for row in rows})
    indexed = {(row["source"], int(row["qp"]), *condition_key(row)[:3]): row for row in rows}
    expected = {(source, qp, condition, int(level), seed) for source in sources
                for qp in qps for condition, level, seed in CONDITIONS}
    if len(sources) < 2 or len(indexed) != len(rows) or set(indexed) != expected:
        raise ValueError("Missing or duplicate paired observations")
    correlations, pairs, summaries = [], [], []
    for qp in qps:
        for condition, level, seed in CONDITIONS:
            group = [indexed[source, qp, condition, int(level), seed] for source in sources]
            for feature in FEATURES:
                for target in TARGETS:
                    rho = float(spearman_last_axis([float(r[feature]) for r in group],
                                                   [float(r[target]) for r in group]))
                    correlations.append(dict(feature=feature, target=target, qp=qp, condition=condition,
                                             n_images=len(group), spearman_rho=rho, evaluable=math.isfinite(rho)))
        for source in sources:
            clean, noisy = indexed[source, qp, "clean", 0, ""], indexed[source, qp, "awgn", 30, "20260811"]
            pairs.append(dict(source=source, qp=qp,
                              cu_clean=int(clean["cu_count"]), cu_noisy=int(noisy["cu_count"]),
                              cu_ratio=float(noisy["cu_count"])/float(clean["cu_count"]),
                              density_clean=float(clean["cu_density_per_mpixel"]),
                              density_noisy=float(noisy["cu_density_per_mpixel"]),
                              bpp_clean=float(clean["file_bpp"]), bpp_noisy=float(noisy["file_bpp"])))
        these = [row for row in pairs if row["qp"] == qp]
        summaries.append(dict(qp=qp, n_images=len(these),
                              cu_increased=sum(r["cu_noisy"] > r["cu_clean"] for r in these),
                              cu_decreased=sum(r["cu_noisy"] < r["cu_clean"] for r in these),
                              cu_equal=sum(r["cu_noisy"] == r["cu_clean"] for r in these),
                              cu_ratio_min=min(r["cu_ratio"] for r in these),
                              cu_ratio_max=max(r["cu_ratio"] for r in these)))
    return correlations, pairs, summaries


def run_div2k_statistics(rows: list[dict], output: Path, *, resamples: int = 99_999,
                         seed: int = 20260905) -> dict:
    """CU correlations and noisy-minus-clean changes, paired across all 100 images."""
    descriptive_tables(rows)  # Validate the complete paired condition/QP grid.
    cells, sources, x, y = build_cells(rows, expected_sources=100)
    observed, draws = paired_spearman_bootstrap(x, y, n_resamples=resamples, seed=seed)
    ci = percentile_intervals(observed, draws)
    write_csv(output / "correlations_with_ci.csv", [
        {**cell, "target": "cu_density_per_mpixel", "rho": observed[i],
         **interval_row(ci, i, "ci_")} for i, cell in enumerate(cells)])

    lookup = {(cell["distortion"], cell["qp"], cell["feature"]): i
              for i, cell in enumerate(cells)}
    clean = [lookup["clean", qp, feature] for qp in QPS for feature in FEATURES]
    noisy = [lookup["awgn", qp, feature] for qp in QPS for feature in FEATURES]
    differences = observed[noisy] - observed[clean]
    difference_draws = draws[:, noisy] - draws[:, clean]
    pointwise = percentile_intervals(differences, difference_draws)
    simultaneous = simultaneous_basic_intervals(differences, difference_draws, alpha=0.05)
    write_csv(output / "noise_correlation_changes.csv", [
        {"qp": cells[a]["qp"], "feature": cells[a]["feature"],
         "target": "cu_density_per_mpixel", "n_images": len(sources),
         "contrast": "awgn_30_minus_clean", "rho_clean": observed[a],
         "rho_awgn": observed[b], "rho_difference": differences[i],
         **interval_row(pointwise, i, "ci_"),
         "simultaneous_low": simultaneous["low"][i],
         "simultaneous_high": simultaneous["high"][i]}
        for i, (a, b) in enumerate(zip(clean, noisy, strict=True))])
    details = {"inferential_statistics": True, "target": "cu_density_per_mpixel",
               "bpp_correlations": "descriptive", "n_images": len(sources),
               "resamples": resamples, "seed": seed, "rng": "PCG64",
               "resampling_unit": "source image; shared indices for all conditions and QPs",
               "ranks_recomputed_per_resample": True, "sources": sources,
               "correlation_cells_with_ci": len(cells), "noise_contrasts": len(clean),
               "pointwise_confidence": 0.95, "pointwise_method": "percentile",
               "simultaneous_confidence": 0.95, "simultaneous_family_size": len(clean),
               "simultaneous_method": "basic bootstrap maximum absolute error",
               "simultaneous_critical_value": simultaneous["critical_value"]}
    write_json(output / "bootstrap_metadata.json", details)
    return details


def load_validated_measurements(measurements_path: Path) -> dict[str, list[dict]]:
    """Check full registered grids without any resampling or output generation."""
    selected = normalize_measurements(read_csv(measurements_path))
    # Reject incomplete tables before creating any statistical outputs.
    for dataset, rows in selected.items():
        if dataset == "kodak":
            validate_measurements(rows)
        else:
            if dataset == "div2k" and {r["source"] for r in rows} != {f"{i:04d}.png" for i in range(801, 901)}:
                raise ValueError("Expected all 100 full DIV2K validation images")
            qps = QPS if dataset == "div2k" else tuple(sorted({int(r["qp"]) for r in rows}))
            descriptive_tables(rows, qps)
    return selected


def analyze_partitions(measurements_path: Path, output: Path, resamples: int = 99_999,
                       seed: int = 20260905) -> dict:
    """Analyze complete registered grids; synthetic data remain a functionality test."""
    if not isinstance(resamples, int) or resamples < 1:
        raise ValueError("resamples must be a positive integer")
    measurements_path, output = Path(measurements_path), Path(output)
    selected = load_validated_measurements(measurements_path)
    result = {"input_sha256": sha256(measurements_path), "datasets": {},
              "artifact_verification": "numeric table checks; encoded artifacts are verified by the producer"}
    for dataset, rows in sorted(selected.items()):
        destination = output / dataset
        write_csv(destination / "joined_measurements.csv", rows)
        details = {"rows": len(rows), "n_images": len({r["source"] for r in rows})}
        if dataset == "kodak":
            settings = {**SETTINGS, "bootstrap_resamples": resamples, "bootstrap_seed": seed,
                        "permutation_resamples": resamples, "permutation_seed": seed + 1}
            run_statistics(destination, settings)
            sensitivity_columns = [f"{feature}_{variant['name']}" for feature in FEATURES
                                   if feature.startswith("glcm_") for variant in SETTINGS["glcm_sensitivity"]]
            details.update(inferential_statistics=True, resamples=resamples, seed=seed,
                           permutation_seed=seed + 1, pointwise_confidence=0.95,
                           contrast_family_alpha=0.025, permutation_holm_family_size=168,
                           correlation_cells=312, rq1_contrasts=20, rq2_contrasts=144,
                           glcm_sensitivity_complete=all(r.get(key) not in (None, "")
                                                         for r in rows for key in sensitivity_columns))
        else:
            qps = QPS if dataset == "div2k" else tuple(sorted({int(r["qp"]) for r in rows}))
            tables = descriptive_tables(rows, qps)
            for name, table in zip(("descriptor_correlations", "paired_noise_effects", "noise_summary"), tables):
                write_csv(destination / f"{name}.csv", table)
            details.update(inferential_statistics=False, correlation_cells=len(tables[0]),
                           full_dataset_sample=dataset == "div2k")
            if dataset == "div2k":
                details.update(run_div2k_statistics(rows, destination, resamples=resamples, seed=seed))
        details["outputs"] = {path.name: sha256(path) for path in sorted(destination.iterdir())
                              if path.suffix in {".csv", ".npz"}}
        result["datasets"][dataset] = details
    result["complete"] = True
    write_json(output / "analysis.json", result)
    return result
