import numpy as np
import pytest
from scipy.stats import spearmanr

from vvcstudy.artifacts import read_csv, write_csv
from vvcstudy.partition_analysis import (
    FEATURES, QPS, analyze_partitions, build_cells, contrasts,
    descriptive_tables, normalize_measurements,
)


def small_measurements():
    rows = []
    for distortion, seed in (("clean", ""), ("awgn", "20260811"), ("awgn", "20260812")):
        for index, source in enumerate(("a", "b", "c")):
            rows.append({"source": source, "distortion": distortion, "level": 0 if distortion == "clean" else 5,
                         "seed": seed, "qp": 32, "cu_density_per_mpixel": 10 + index,
                         **{feature: index + len(feature) for feature in FEATURES}})
    return rows


def test_cells_align_images_and_do_not_pool_noise_seeds():
    cells, sources, x, y = build_cells(small_measurements()[::-1], expected_sources=3)
    assert sources == ["a", "b", "c"]
    assert len(cells) == 18 and sum(cell["primary"] for cell in cells) == 12
    np.testing.assert_array_equal(y, np.tile([10, 11, 12], (18, 1)))
    assert np.all(np.diff(x, axis=1) == 1)


def test_missing_and_duplicate_sources_fail_before_statistics():
    rows = small_measurements()
    with pytest.raises(ValueError, match="Incomplete condition"):
        build_cells(rows[:-1], expected_sources=3)
    with pytest.raises(ValueError, match="Duplicate"):
        build_cells(rows + [rows[0]], expected_sources=3)


def test_contrasts_keep_orientation_and_same_qp_reference():
    cells, _, _, _ = build_cells(small_measurements(), expected_sources=3)
    observed = np.array([-0.4 if cell["feature"] == "glcm_homogeneity" else 0.4 for cell in cells])
    observed += np.array([0.2 if cell["distortion"] == "clean" else 0 for cell in cells])
    metadata, values, draws = contrasts(cells, observed, np.tile(observed, (4, 1)))
    assert len(values) == 11
    assert all(row["seed"] != "20260812" for row in metadata)
    index = next(i for i, row in enumerate(metadata)
                 if row["family"] == "RQ2" and row["feature"] == "glcm_homogeneity")
    assert values[index] == pytest.approx(0.2)
    np.testing.assert_array_equal(draws, np.tile(values, (4, 1)))


def kodak_rows():
    conditions = [("clean", 0, None)]
    conditions += [("stripes", level, None) for level in (8, 16, 32)]
    conditions += [("awgn", level, seed) for level in (5, 15, 30)
                   for seed in (20260811, 20260812, 20260813)]
    rows = []
    for i in range(1, 25):
        for condition_index, (distortion, level, seed) in enumerate(conditions):
            features = {feature: float((i*(j+1) + condition_index*(i % 5)) % 37)
                        for j, feature in enumerate(FEATURES)}
            for feature in FEATURES:
                if feature.startswith("glcm_"):
                    features[feature + "_levels32"] = features[feature] * 3
                    features[feature + "_horizontal"] = features[feature] + i % 3
            for qp in QPS:
                count = 30 + i*3 + condition_index*(i % 3) + qp
                rows.append(dict(dataset="kodak", source=f"kodim{i:02d}.png", distortion=distortion,
                                 level=level, seed=seed, qp=qp, mode="baseline", codec="vtm23-baseline-trace",
                                 conversion="opencv_444", width=768, height=512, coded_area=393216,
                                 cu_count=count, cu_density_per_mpixel=count*1e6/393216,
                                 input_sha256=f"fixture-{i}-{condition_index}", bitstream_bytes=1000+i,
                                 **features))
    return rows


def test_fresh_kodak_csv_reproduces_full_cell_and_contrast_families(tmp_path):
    path, output = tmp_path / "encodings.csv", tmp_path / "analysis"
    rows = kodak_rows()
    # Matrix observations coexist in fresh runs but must not enter CU inference.
    matrix = {**rows[0], "conversion": "ffmpeg_444", "cu_count": 999}
    write_csv(path, rows + [matrix])
    result = analyze_partitions(path, output, resamples=19)
    assert result["datasets"]["kodak"]["correlation_cells"] == 312
    assert result["datasets"]["kodak"]["glcm_sensitivity_complete"] is True
    directory = output / "kodak"
    correlations = read_csv(directory / "correlations.csv")
    contrast_rows = read_csv(directory / "contrasts.csv")
    assert len(correlations) == 312
    assert sum(r["family"] == "RQ1" for r in contrast_rows) == 20
    assert sum(r["family"] == "RQ2" for r in contrast_rows) == 144
    assert len(read_csv(directory / "leave_one_out_contrasts.csv")) == 24*164
    assert len(read_csv(directory / "paired_changes.csv")) == 24*144
    assert len(read_csv(directory / "parameter_sensitivity.csv")) == 168
    clean = [r for r in rows if r["distortion"] == "clean" and r["qp"] == 22]
    target = next(r for r in correlations if r["distortion"] == "clean" and r["qp"] == "22"
                  and r["feature"] == "sobel_si")
    assert float(target["rho"]) == pytest.approx(spearmanr(
        [r["sobel_si"] for r in clean], [r["cu_density_per_mpixel"] for r in clean]).statistic, abs=1e-14)
    with np.load(directory / "bootstrap_correlations.npz") as archive:
        assert archive["distribution"].shape == (19, 312)
    assert not list(output.rglob("*.md"))


@pytest.mark.parametrize("damage", ["missing", "duplicate", "density", "identity", "verification"])
def test_damaged_kodak_input_fails_before_outputs(tmp_path, damage):
    rows = kodak_rows()
    if damage == "missing":
        rows.pop()
    elif damage == "duplicate":
        rows[-1] = dict(rows[0])
    elif damage == "density":
        rows[0]["cu_density_per_mpixel"] = 0
    elif damage == "identity":
        rows[0]["input_sha256"] = "changed"
    else:
        rows[0]["cu_coverage_verified"] = False
    path, output = tmp_path / "input.csv", tmp_path / "output"
    write_csv(path, rows)
    with pytest.raises(ValueError):
        analyze_partitions(path, output, resamples=7)
    assert not output.exists()


def div2k_rows(n=100):
    rows = []
    for i in range(n):
        area, count = (i+10)**2, 30+i
        for condition, level, seed in (("clean", 0, None), ("awgn", 30, 20260811)):
            cu = count*(2 if condition == "awgn" else 1)
            for qp in QPS:
                rows.append(dict(dataset="div2k_validation_full100", source=f"{801+i:04d}.png",
                                 distortion=condition, level=level, seed=seed, qp=qp, mode="baseline",
                                 conversion="opencv_444", coded_area=area, source_area=area,
                                 cu_count=cu, cu_density_per_mpixel=cu*1e6/area, file_bpp=3-i/100,
                                 **{feature: i+1 for feature in FEATURES}))
    return rows


def test_div2k_full_sample_remains_descriptive_and_uses_density(tmp_path):
    path, output = tmp_path / "div2k.csv", tmp_path / "analysis"
    write_csv(path, list(reversed(div2k_rows())))
    result = analyze_partitions(path, output, resamples=7)
    assert result["datasets"]["div2k"]["n_images"] == 100
    assert result["datasets"]["div2k"]["inferential_statistics"] is False
    correlations = read_csv(output / "div2k/descriptor_correlations.csv")
    assert len(correlations) == 96
    assert all(float(r["spearman_rho"]) == pytest.approx(-1) and r["n_images"] == "100" for r in correlations)
    assert all(r["cu_increased"] == "100" for r in read_csv(output / "div2k/noise_summary.csv"))
    assert len(read_csv(output / "div2k/paired_noise_effects.csv")) == 400
    assert not list(output.rglob("*.npz"))


def test_div2k_missing_original_is_not_a_full_sample(tmp_path):
    path = tmp_path / "div2k.csv"
    write_csv(path, div2k_rows(99))
    with pytest.raises(ValueError, match="all 100"):
        analyze_partitions(path, tmp_path / "analysis", resamples=7)


def test_mixed_fresh_csv_keeps_kodak_and_div2k_samples_separate(tmp_path):
    rows = kodak_rows() + [{**r, "dataset": "div2k"} for r in div2k_rows()]
    path = tmp_path / "encodings.csv"
    write_csv(path, rows)
    result = analyze_partitions(path, tmp_path / "analysis", resamples=3)
    assert set(result["datasets"]) == {"kodak", "div2k"}
    assert result["datasets"]["kodak"]["n_images"] == 24
    assert result["datasets"]["div2k"]["n_images"] == 100
    assert result["datasets"]["kodak"]["rows"] == 1248
    assert result["datasets"]["div2k"]["rows"] == 800


def test_constant_descriptor_retains_undefined_correlation():
    rows = div2k_rows(3)
    for row in rows:
        row["glcm_entropy"] = 0
    normalized = normalize_measurements(rows)["div2k"]
    correlations, _, _ = descriptive_tables(normalized)
    assert not any(r["evaluable"] for r in correlations if r["feature"] == "glcm_entropy")
    with pytest.raises(ValueError, match="Missing or duplicate"):
        descriptive_tables(normalized[:-1])


def test_synthetic_smoke_is_explicitly_not_full_dataset_analysis(tmp_path):
    rows = [{**r, "dataset": "synthetic", "source": "smoke" + r["source"]} for r in div2k_rows(2) if r["qp"] == 32]
    path = tmp_path / "smoke.csv"
    write_csv(path, rows)
    result = analyze_partitions(path, tmp_path / "analysis", resamples=3)
    assert result["datasets"]["synthetic"]["full_dataset_sample"] is False
    assert result["datasets"]["synthetic"]["n_images"] == 2
