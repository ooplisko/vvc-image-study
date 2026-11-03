from pathlib import Path
import shutil

import pytest

from vvcstudy.artifacts import read_csv, write_csv
from vvcstudy.results import completed_study, seal_results, verify_results
from vvcstudy.studies import merge_rows


ROOT = Path(__file__).resolve().parents[1]


def test_complete_results_validate_and_reuse_without_encoders(monkeypatch):
    from vvcstudy import cli
    checked = verify_results(ROOT)
    assert checked["accepted"], checked["errors"]
    assert checked["checks"]["partitions"]["rows"] == 1248
    assert checked["checks"]["div2k"]["rows"] == 800
    assert checked["checks"]["matrices"]["encodes"] == 576
    assert checked["checks"]["jpeg-vvc"]["matched_rows"] == 1984

    def forbidden(*args, **kwargs):
        pytest.fail("A completed result stage must not prepare codecs or encode images")

    monkeypatch.setattr(cli, "_prepared", forbidden)
    for study in ("partitions", "div2k", "matrices", "jpeg-vvc"):
        result = cli.execute_study(ROOT, "full", study, "source", 1)
        assert result["status"] == "complete" and result["reused_results"]


def test_table_tampering_or_missing_points_prevent_reuse(tmp_path):
    base = tmp_path / "results/full"
    (base / "tables").mkdir(parents=True)
    (tmp_path / "configs").mkdir()
    shutil.copyfile(ROOT / "configs/full.toml", tmp_path / "configs/full.toml")
    shutil.copyfile(ROOT / "results/full/manifest.json", base / "manifest.json")
    rows = read_csv(ROOT / "results/full/tables/encodings.csv")
    rows[0]["cu_count"] = "999999"
    write_csv(base / "tables/encodings.csv", rows)
    assert not completed_study(tmp_path, "full", "div2k")
    with pytest.raises(ValueError):
        seal_results(tmp_path, studies=["div2k"])


def test_yuv_derived_matrix_reference_cannot_pass_as_original_rgb(tmp_path):
    base = tmp_path / "results/full/tables"
    base.mkdir(parents=True)
    (tmp_path / "configs").mkdir()
    shutil.copyfile(ROOT / "configs/full.toml", tmp_path / "configs/full.toml")
    rows = read_csv(ROOT / "results/full/tables/encodings.csv")
    next(row for row in rows if row["conversion"] == "ffmpeg_444")["rgb_protocol"] = ""
    write_csv(base / "encodings.csv", rows)
    with pytest.raises(ValueError, match="original RGB8"):
        seal_results(tmp_path, studies=["matrices"])


def test_merge_keeps_unrelated_result_points(tmp_path):
    path = tmp_path / "table.csv"
    write_csv(path, [{"id": "a", "value": 1}, {"id": "b", "value": 2}])
    merged = merge_rows(path, [{"id": "b", "value": 3}], lambda row: row["id"])
    assert merged == [{"id": "a", "value": "1"}, {"id": "b", "value": 3}]


def test_reused_stage_obeys_stop_boundary(tmp_path, monkeypatch):
    from vvcstudy import results, cli
    from vvcstudy.runner import stop
    monkeypatch.setattr(results, "completed_study", lambda *args: True)
    stop(tmp_path / "runs/full")
    assert cli.execute_study(tmp_path, "full", "div2k", "source", 1, reset_stop=False)["status"] == "stopped"


def test_analysis_reuses_checked_outputs_without_resampling(monkeypatch):
    from vvcstudy import cli, jpeg_analysis

    def forbidden(*args, **kwargs):
        pytest.fail("A valid analysis cache must not resample")

    monkeypatch.setattr(jpeg_analysis, "analyze_jpeg", forbidden)
    assert cli.analyze(ROOT, "full", "all")["reused"]


def test_compact_analysis_is_reusable_without_any_generated_figures(tmp_path):
    from vvcstudy.analysis_cache import completed_analysis
    shutil.copytree(ROOT / "results/full", tmp_path / "results/full",
                    ignore=shutil.ignore_patterns("*.png", "*.npz", "bootstrap_cache"))
    shutil.copytree(ROOT / "manifests", tmp_path / "manifests")
    assert not list(tmp_path.rglob("*.png"))
    assert completed_analysis(tmp_path, "full", 99_999)


def test_completed_full_workflow_does_not_encode_or_repeat_timing(monkeypatch):
    from vvcstudy import cli, timing, jpeg_analysis

    def forbidden(*args, **kwargs):
        pytest.fail("Verified results must be reused without encoding, timing or resampling")

    monkeypatch.setattr(cli, "_prepared", forbidden)
    monkeypatch.setattr(timing, "run_timing", forbidden)
    monkeypatch.setattr(jpeg_analysis, "analyze_jpeg", forbidden)
    assert cli.main(["--root", str(ROOT), "reproduce", "--profile", "full"]) == 0


def test_reused_timing_obeys_stop_boundary(tmp_path, monkeypatch):
    from vvcstudy import cli, timing
    from vvcstudy.runner import stop
    monkeypatch.setattr(timing, "verify_timing", lambda *args: {"accepted": True})
    stop(tmp_path / "runs/full")
    assert cli.execute_timing(tmp_path, "source", reset_stop=False)["status"] == "stopped"


def test_mixed_quality_protocol_cannot_be_sealed(tmp_path):
    (tmp_path / "configs").mkdir()
    shutil.copyfile(ROOT / "configs/full.toml", tmp_path / "configs/full.toml")
    rows = read_csv(ROOT / "results/full/tables/encodings.csv")
    next(row for row in rows if row["dataset"] == "div2k")["quality_protocol"] = "opencv_rgb_v1"
    write_csv(tmp_path / "results/full/tables/encodings.csv", rows)
    with pytest.raises(ValueError, match="metric protocol"):
        seal_results(tmp_path, studies=["div2k"])


def test_full_workflow_with_completed_tables_reaches_only_timing(tmp_path, monkeypatch):
    from vvcstudy import cli, results, timing
    prepared = []
    measured = []
    monkeypatch.setattr(results, "completed_study", lambda *args: True)
    monkeypatch.setattr(results, "verify_results", lambda *args: {"accepted": True})

    def codecs(root, profile, study, mode):
        prepared.append(study)
        return {"vtm23-baseline-trace": {"encoder": "fixture"}}

    def measure(root, matched, codec, **kwargs):
        measured.append(matched)
        return {"status": "stopped"}

    monkeypatch.setattr(cli, "_prepared", codecs)
    monkeypatch.setattr(timing, "run_timing", measure)
    assert cli.main(["--root", str(tmp_path), "reproduce", "--profile", "full"]) == 2
    assert prepared == ["timing"]
    assert measured == [tmp_path / "results/full/tables/actual_by_qp.csv"]
