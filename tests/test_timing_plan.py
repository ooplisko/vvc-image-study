from pathlib import Path

import pytest

from vvcstudy.artifacts import file_record, read_csv, sha256, write_json
from vvcstudy.timing import selected_points, timing_plan, verify_timing


def matched_rows():
    return read_csv(Path(__file__).parents[1] / "results/full/tables/actual_by_qp.csv")


def test_timing_uses_all_clean_kodak_once_per_qp_not_per_metric():
    rows = matched_rows()
    vtm, jpeg = selected_points(rows)
    assert len(vtm) == 96
    assert all(source.startswith("kodim") for source, _ in jpeg)
    assert 24 <= len(jpeg) <= 192
    plan = timing_plan(rows)
    assert plan["vtm_measurements"] == 288 and plan["workers"] == 1


def test_incomplete_matching_and_inconsistent_bitstreams_are_rejected():
    rows = [r for r in matched_rows() if r["dataset"] == "kodak" and r["condition"] == "clean"]
    with pytest.raises(ValueError, match="complete"):
        selected_points(rows[:-1])
    rows[0] = {**rows[0], "vtm_sha256": "different"}
    with pytest.raises(ValueError, match="conflicting"):
        selected_points(rows)


def test_timing_summary_integrity_and_pending_state(tmp_path):
    assert verify_timing(tmp_path)["status"] == "pending"
    base = tmp_path / "results/full"
    (base / "tables").mkdir(parents=True)
    matched = base / "tables/actual_by_qp.csv"
    matched.write_text("fixture pairs")
    folder = base / "timing"
    folder.mkdir()
    for name in ("context.json", "repeats.csv", "paired.csv", "summary.csv"):
        (folder / name).write_text("fixture")
    write_json(folder / "manifest.json", {"complete": True, "measurements": 837,
                                          "matched_sha256": sha256(matched),
                                          "files": [file_record(p, folder) for p in folder.iterdir()]})
    assert verify_timing(tmp_path)["accepted"]
    (folder / "summary.csv").write_text("changed")
    assert verify_timing(tmp_path)["accepted"] is False
