"""Synthetic regression controls for encoder, decoder and partition output."""

from pathlib import Path

from .artifacts import read_json, sha256, write_json
from .encoding import execute_encode
from .runner import run_jobs
from .studies import config, encode_jobs, prepare_inputs


CODECS = ("vtm23-baseline-trace", "vtm23-csf-trace", "vvenc-baseline-trace", "vvenc-csf-trace")


def control_jobs(root: Path, codecs: dict) -> list[dict]:
    # The second odd-sized smoke image is tested separately in the VTM padding pipeline.
    inputs = prepare_inputs(root, "synthetic", [("clean", 0, None)], "smoke")[:1]
    settings = {**config(root, "smoke"), "qps": [22, 32]}
    jobs = encode_jobs(root, inputs, settings, codecs)
    if set(CODECS).issubset(codecs):
        jobs += encode_jobs(root, inputs, settings, codecs, matrices=True)
    return jobs


def control_summary(folder: Path, job_ids: set[str] | None = None) -> list[dict]:
    rows = []
    for path in (folder / "jobs").glob("*/result.json"):
        record = read_json(path)
        if job_ids is not None and record["job_id"] not in job_ids:
            continue
        spec = record["spec"]
        values = record["values"]
        rows.append({"codec": spec["codec"], "qp": spec["qp"], "conversion": spec["conversion"],
                     "input_sha256": spec["input_sha256"],
                     "stream_sha256": sha256(path.parent / "stream.vvc"),
                     "decoded_sha256": sha256(path.parent / "decoded.yuv"),
                     "input_yuv_sha256": sha256(path.parent / "input.yuv"),
                     "bitstream_bytes": values["bitstream_bytes"],
                     "cu_count": values.get("cu_count"),
                     "partitions_sha256": (sha256(path.parent / "partitions.csv")
                                           if spec["trace"] else None)})
    return sorted(rows, key=lambda r: (r["codec"], r["conversion"], r["qp"]))


def check_codecs(root: Path, codecs: dict) -> dict:
    expected = read_json(root / "tests/fixtures/codecs/expected.json")
    # Build hashes isolate results from different executables.
    from .artifacts import identity
    key = identity({name: {kind: codec[kind + "_sha256"] for kind in ("encoder", "decoder")}
                    for name, codec in codecs.items()})[:16]
    folder = root / "runs" / "codec_checks" / key
    jobs = control_jobs(root, codecs)
    keys = {(job["spec"]["codec"], job["spec"]["conversion"], job["spec"]["qp"]) for job in jobs}
    controls = [row for row in expected["controls"] if (row["codec"], row["conversion"], row["qp"]) in keys]
    outcome = run_jobs(jobs, execute_encode, folder, workers=1)
    if outcome["status"] != "complete":
        return outcome
    actual = control_summary(folder, {identity(job["spec"]) for job in jobs})
    errors = []
    if len(actual) != len(controls) or len(controls) != len(jobs):
        errors.append("Control grid size differs.")
    else:
        for found, reference in zip(actual, controls, strict=True):
            differences = [field for field in reference if found.get(field) != reference[field]]
            if differences:
                errors.append({"codec": reference["codec"], "conversion": reference["conversion"],
                               "qp": reference["qp"], "different_fields": differences})
    result = {"accepted": not errors, "controls": len(actual), "errors": errors,
              "scope": "Bitstream, decoded pixels and partition checks on synthetic inputs",
              "expected_fixture_sha256": sha256(root / "tests/fixtures/codecs/expected.json"),
              "config_sha256": sha256(root / "configs/codecs/vtm_intra.cfg"),
              "decoder_hashes": {name: codec["decoder_sha256"] for name, codec in codecs.items()},
              "runtime_hashes": {name: codec.get("runtime_hashes", {}) for name, codec in codecs.items()},
              "codec_hashes": {name: codec["encoder_sha256"] for name, codec in codecs.items()}}
    write_json(folder / "comparison.json", result)
    return result
