import pytest

from vvcstudy.artifacts import read_json, write_json
from vvcstudy.cli import _parser
from vvcstudy.replication import compare_replication, replication_plan


def fixture_rows():
    return [{"source": f"kodim{i:02d}", "qp": qp, "bpp": qp/100, "psnr_rgb_db": 60-qp}
            for i in range(1, 25) for qp in (22, 27, 32, 37)]


def fixture_published():
    return [{"anchor": "duan_vtm18", "qp": qp, "bpp": qp/100, "psnr_rgb_db": 60-qp}
            for qp in (22, 27, 32, 37)]


def test_same_qp_comparison_does_not_choose_nearest_rate():
    rows = fixture_rows()
    # Move QP22 nearer to the published QP27 rate; it must still compare with QP22.
    for row in rows:
        if row["qp"] == 22:
            row["bpp"] = 0.27
    result = compare_replication(rows, fixture_published())
    assert result[0]["published_bpp"] == 0.22
    assert result[0]["bpp_gap"] == pytest.approx(0.05)


def test_replication_plots_need_only_verified_compact_measurements(tmp_path):
    from pathlib import Path
    import shutil
    from vvcstudy.replication import plot_replication
    root = Path(__file__).resolve().parents[1]
    output = tmp_path / "results/validation/lossy-vae/full"
    output.mkdir(parents=True)
    for name in ("measurements.csv", "context.json"):
        shutil.copyfile(root / "results/validation/lossy-vae/full" / name, output / name)
    shutil.copytree(root / "manifests", tmp_path / "manifests")
    assert not (tmp_path / "runs").exists()
    assert plot_replication(tmp_path)["measured_points"] == 96
    assert len(list(output.glob("*.png"))) == 2
    with (output / "measurements.csv").open("a") as stream:
        stream.write("tampered")
    with pytest.raises(ValueError, match="integrity"):
        plot_replication(tmp_path)


@pytest.mark.parametrize("bad", ["missing", "duplicate", "wrong_source"])
def test_incomplete_or_ambiguous_grid_is_not_a_replication(bad):
    rows = fixture_rows()
    if bad == "missing":
        rows.pop()
    elif bad == "duplicate":
        rows[-1] = rows[0].copy()
    else:
        rows[0]["source"] = "unknown"
    with pytest.raises(ValueError, match="exactly all 24"):
        compare_replication(rows, fixture_published())


def test_replication_cli_is_separate_from_existing_study_grid(tmp_path):
    assert replication_plan(tmp_path)["expected"] == 96
    for command in ("plan", "status", "stop", "analyze", "run"):
        args = _parser().parse_args([command, "--study", "lossy-vae"])
        assert args.study == "lossy-vae"


def test_status_counts_only_verified_jobs_in_current_plan(tmp_path):
    from vvcstudy.artifacts import file_record, identity
    from vvcstudy.replication import replication_status
    folder = tmp_path / "runs/full/lossy-vae"
    current = {"kind": "lossy-vae", "qp": 22}
    previous = {"kind": "lossy-vae", "qp": 22, "old": True}
    for spec in (current, previous):
        job_dir = folder / "jobs" / identity(spec)
        job_dir.mkdir(parents=True)
        payload = job_dir / "data"
        payload.write_bytes(b"result")
        write_json(job_dir / "result.json", {"spec": spec, "job_id": identity(spec),
            "status": "complete", "artifacts": [file_record(payload, job_dir)]})
    write_json(folder / "plan.json", {"jobs": [{"spec": current}]})
    assert replication_status(tmp_path)["completed"] == 1
    (folder / "jobs" / identity(current) / "data").write_bytes(b"broken")
    assert replication_status(tmp_path)["completed"] == 0


def test_partial_grid_writes_no_published_comparison(tmp_path):
    from vvcstudy.replication import analyze_replication
    plan = tmp_path / "runs/full/lossy-vae/plan.json"
    write_json(plan, {"jobs": [{"spec": {"kind": "lossy-vae", "qp": 22}}]})
    output = tmp_path / "results/validation/lossy-vae/full"
    output.mkdir(parents=True)
    (output / "same_qp.csv").write_text("stale means")
    result = analyze_replication(tmp_path)
    assert result["accepted"] is False
    assert result["completed"] == 0
    assert not (tmp_path / "results/validation/lossy-vae/full/same_qp.csv").exists()
    assert read_json(tmp_path / "results/validation/lossy-vae/full/context.json") == result


def test_retained_streams_are_optional_but_identity_and_hashes_are_required(tmp_path, monkeypatch):
    from vvcstudy.replication import replication_jobs
    from vvcstudy.artifacts import sha256
    from vvcstudy import datasets
    config = tmp_path / "configs/codecs/vtm18_duan.cfg"
    config.parent.mkdir(parents=True)
    config.write_text("QP: 32\n")
    write_json(tmp_path / "manifests/validation.json", {"lossy_vae": {
        "configuration": "configs/codecs/vtm18_duan.cfg", "configuration_sha256": sha256(config),
        "codec": {"source_commit": "pinned"}}})
    item = {"id": "kodim01", "sha256": "input", "width": 768, "height": 512, "path": "input.png"}
    monkeypatch.setattr(datasets, "prepare_datasets", lambda *_: {"kodak": [item]})
    codec = {"encoder": "encoder", "decoder": "decoder", "encoder_sha256": "enc", "decoder_sha256": "dec"}
    assert len(replication_jobs(tmp_path, "full", codec)) == 4
    retained = tmp_path / "runs/full/lossy-vae/retained"
    retained.mkdir(parents=True)
    (retained / "stream.vvc").write_bytes(b"stream")
    (retained / "encoder.log").write_text("log")
    row = {"source": "kodim01", "qp": 22, "input_sha256": "input", "config_sha256": sha256(config),
           "stream": {"path": "stream.vvc", "sha256": sha256(retained / "stream.vvc")},
           "encoder_log": {"path": "encoder.log", "sha256": sha256(retained / "encoder.log")}}
    write_json(retained / "manifest.json", {"schema_version": 1, "codec_version": "VTM 18.0", "streams": [row]})
    jobs = replication_jobs(tmp_path, "full", codec)
    assert "retained" in jobs[0] and all("retained" not in j for j in jobs[1:])
    (retained / "stream.vvc").write_bytes(b"tamper")
    with pytest.raises(ValueError, match="checksum/path"):
        replication_jobs(tmp_path, "full", codec)
