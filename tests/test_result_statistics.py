"""Check statistical tables against the supplied measurements."""
from pathlib import Path

import numpy as np

from vvcstudy.artifacts import read_csv
from vvcstudy.jpeg_analysis import load_rows, prepare, safe_rho
from vvcstudy.partition_analysis import build_cells, normalize_measurements
from vvcstudy.statistics import spearman_last_axis

ROOT = Path(__file__).resolve().parents[1] / "results/full"


def test_all_kodak_cu_correlations_match_measurements():
    rows = normalize_measurements(read_csv(ROOT / "tables/encodings.csv"))["kodak"]
    cells, _, x, y = build_cells(rows)
    observed = spearman_last_axis(x, y)
    expected = {(r["distortion"], int(r["level"]), r["seed"], int(r["qp"]), r["feature"]): float(r["rho"])
                for r in read_csv(ROOT / "analysis/partitions/kodak/correlations.csv")}
    assert len(cells) == len(expected) == 312
    np.testing.assert_allclose(observed, [expected[
        r["distortion"], r["level"], r["seed"], r["qp"], r["feature"]] for r in cells],
        rtol=0, atol=1e-12)


def test_all_primary_jpeg_vvc_correlations_match_measurements():
    cells, *_ = prepare(load_rows([ROOT / "tables/actual_by_qp.csv"]))
    fields = ("dataset", "condition", "metric", "qp", "feature")
    expected = {tuple(str(r[key]) for key in fields): float(r["rho"])
                for r in read_csv(ROOT / "analysis/jpeg-vvc/correlations_with_ci.csv")
                if r["variant"] == "nearest" and r["subset"] == "all_matched"}
    chosen = [cell for cell in cells if cell.meta["variant"] == "nearest"
              and cell.meta["subset"] == "all_matched"]
    assert len(expected) == len(chosen) == 192
    np.testing.assert_allclose([safe_rho(c.x, c.y) for c in chosen],
                               [expected[tuple(str(c.meta[k]) for k in fields)] for c in chosen],
                               rtol=0, atol=1e-12)
