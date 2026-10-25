import pytest
from threading import Barrier

from vvcstudy.artifacts import file_record, identity, verified_result, write_json
from vvcstudy.cli import result_exit_code, verify_run
from vvcstudy.runner import exclusive_run, run_jobs, stop


def jobs(count=3):
    return [{"spec": {"kind": "test", "index": i}, "label": str(i)} for i in range(count)]


def write_result(job, folder):
    path = folder / "value.txt"
    path.write_text(str(job["spec"]["index"]))
    return {"values": {"index": job["spec"]["index"]}, "artifacts": [file_record(path, folder)]}


def test_graceful_stop_commits_active_job_and_resume_skips_it(tmp_path):
    executed = []

    def first(job, folder):
        executed.append(job["spec"]["index"])
        value = write_result(job, folder)
        stop(tmp_path)
        return value

    result = run_jobs(jobs(), first, tmp_path)
    assert result["status"] == "stopped"
    assert executed == [0]
    assert verified_result(tmp_path / "jobs" / identity(jobs()[0]["spec"]), jobs()[0]["spec"])

    def resumed(job, folder):
        executed.append(job["spec"]["index"])
        return write_result(job, folder)

    result = run_jobs(jobs(), resumed, tmp_path)
    assert result["status"] == "complete" and result["reused"] == 1
    assert executed == [0, 1, 2]
    assert run_jobs(jobs(), resumed, tmp_path)["reused"] == 3
    assert executed == [0, 1, 2]


def test_corruption_recomputes_only_damaged_job(tmp_path):
    run_jobs(jobs(), write_result, tmp_path)
    damaged = tmp_path / "jobs" / identity(jobs()[1]["spec"]) / "value.txt"
    damaged.write_text("9")  # Same length; a size-only check would miss this.
    done = []

    def execute(job, folder):
        done.append(job["spec"]["index"])
        return write_result(job, folder)

    assert run_jobs(jobs(), execute, tmp_path)["reused"] == 2
    assert done == [1]


def test_failed_job_is_not_accepted_or_followed_by_more_jobs(tmp_path):
    def fail(job, folder):
        (folder / "unfinished.txt").write_text("partial")
        raise RuntimeError("simulated interrupted codec")

    result = run_jobs(jobs(), fail, tmp_path)
    assert result["status"] == "failed" and result["completed"] == 0
    assert len(result["errors"]) == 1
    assert not list(tmp_path.rglob("result.json"))


def test_parallel_stop_saves_both_active_jobs_without_dispatching_more(tmp_path):
    barrier = Barrier(2)

    def execute(job, folder):
        barrier.wait(timeout=5)
        stop(tmp_path)
        return write_result(job, folder)

    result = run_jobs(jobs(5), execute, tmp_path, workers=2)
    assert result["status"] == "stopped" and result["completed"] == 2
    assert run_jobs(jobs(5), write_result, tmp_path, workers=2)["reused"] == 2


def test_stop_at_stage_boundary_does_not_start_the_next_stage(tmp_path):
    def last(job, folder):
        stop(tmp_path)
        return write_result(job, folder)

    assert run_jobs(jobs(1), last, tmp_path)["status"] == "stopped"
    assert run_jobs(jobs(3), write_result, tmp_path, reset_stop=False)["completed"] == 1
    assert run_jobs(jobs(3), write_result, tmp_path)["status"] == "complete"


def test_exclusive_run_rejects_second_writer(tmp_path):
    with exclusive_run(tmp_path):
        with pytest.raises(RuntimeError, match="Another experiment"):
            with exclusive_run(tmp_path):
                pass


def test_artifact_path_cannot_escape_job_folder(tmp_path):
    outside = tmp_path / "outside.txt"
    outside.write_text("content")
    folder = tmp_path / "job"
    spec = {"kind": "test"}
    artifact = file_record(outside, tmp_path)
    artifact["path"] = "../outside.txt"
    write_json(folder / "result.json", {"job_id": identity(spec), "spec": spec,
                                       "status": "complete", "artifacts": [artifact]})
    assert verified_result(folder, spec) is None


def test_verification_detects_missing_planned_jobs(tmp_path):
    folder = tmp_path / "runs" / "smoke"
    run_jobs(jobs(1), write_result, folder)
    write_json(folder / "plans" / "test.json", {"jobs": [
        {"job_id": identity(job["spec"]), "spec": job["spec"]} for job in jobs(2)]})
    result = verify_run(tmp_path, "smoke")
    assert not result["accepted"]
    assert result["planned_jobs"] == 2 and len(result["missing"]) == 1


def test_verification_checks_saved_plan_identity(tmp_path):
    folder = tmp_path / "runs" / "smoke"
    job = jobs(1)[0]
    run_jobs([job], write_result, folder)
    write_json(folder / "plans/test.json", {"jobs": [
        {"job_id": identity(job["spec"]), "spec": {**job["spec"], "index": 99}}]})
    result = verify_run(tmp_path, "smoke")
    assert not result["accepted"] and result["errors"]


def test_verification_does_not_infer_complete_grid_without_plan(tmp_path):
    run_jobs(jobs(1), write_result, tmp_path / "runs/smoke")
    assert not verify_run(tmp_path, "smoke")["accepted"]


@pytest.mark.parametrize("result,code", [
    ({"partitions": {"status": "failed"}}, 1),
    ({"timing": {"status": "stopped"}}, 2),
    ({"verification": {"accepted": False}}, 1),
    ({"status": "complete", "values": [1, 2]}, 0),
])
def test_nested_command_status_is_not_silently_successful(result, code):
    assert result_exit_code(result) == code
