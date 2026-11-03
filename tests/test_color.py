from pathlib import Path
import shutil

import cv2
import numpy as np
import pytest

from vvcstudy.color import reconstruction_rgb
from vvcstudy.metrics import opencv_yuv_to_rgb8, psnr


def test_opencv_reconstruction_uses_original_rgb_as_quality_reference(tmp_path):
    original = np.full((8, 8, 3), [255, 0, 0], dtype=np.uint8)
    yuv = cv2.cvtColor(original, cv2.COLOR_RGB2YUV).transpose(2, 0, 1)
    frame = tmp_path / "decoded.yuv"
    frame.write_bytes((yuv.astype("<u2") * 4).tobytes())
    reconstructed = reconstruction_rgb(frame, 8, 8, "opencv_444")
    np.testing.assert_array_equal(reconstructed, opencv_yuv_to_rgb8(yuv, 8))
    # Even a lossless YUV round trip can distort saturated RGB. Using the
    # YUV-derived reference would hide that error and incorrectly give infinity.
    assert np.isfinite(psnr(original, reconstructed))
    assert psnr(reconstructed, reconstructed) == float("inf")


@pytest.mark.parametrize("conversion", ["ffmpeg_444", "ffmpeg_420"])
@pytest.mark.parametrize("bit_depth", [8, 10])
def test_ffmpeg_inverse_preserves_limited_range_black_and_white(tmp_path, conversion, bit_depth):
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        pytest.skip("Native FFmpeg is optional for unit-only environments")
    y = np.tile([16, 16, 235, 235], (4, 1))
    uv_count = 16 if conversion.endswith("444") else 4
    samples = np.concatenate([y.ravel(), np.full(2 * uv_count, 128)]) * (1 << (bit_depth - 8))
    frame = tmp_path / "frame.yuv"
    frame.write_bytes(samples.astype(np.uint8 if bit_depth == 8 else "<u2").tobytes())
    actual = reconstruction_rgb(frame, 4, 4, conversion, bit_depth=bit_depth, ffmpeg=ffmpeg)
    np.testing.assert_array_equal(actual, np.repeat(np.where(y == 16, 0, 255)[..., None], 3, axis=2))


def test_wrong_frame_size_or_bit_depth_is_rejected(tmp_path):
    frame = tmp_path / "bad.yuv"
    frame.write_bytes(bytes(3))
    with pytest.raises(ValueError, match="exactly one"):
        reconstruction_rgb(frame, 2, 2, "opencv_444")
    frame.write_bytes(np.full(12, 1024, dtype="<u2").tobytes())
    with pytest.raises(ValueError, match="bit depth"):
        reconstruction_rgb(frame, 2, 2, "opencv_444")


def test_matrix_opencv_baseline_jobs_are_shared_with_partitions(tmp_path, monkeypatch):
    from vvcstudy.studies import encode_jobs
    (tmp_path / "configs/codecs").mkdir(parents=True)
    (tmp_path / "configs/codecs/vtm_intra.cfg").write_text("fixture")
    binary = tmp_path / "fixture.exe"
    binary.write_bytes(b"fixture")
    monkeypatch.setattr(shutil, "which", lambda name: str(binary))
    codecs = {name: {"encoder": str(binary), "decoder": str(binary)} for name in
              ("vtm23-baseline-trace", "vtm23-csf-trace", "vvenc-baseline-trace", "vvenc-csf-trace")}
    inputs = [dict(dataset="kodak", source="kodim01.png", distortion="clean", level=0,
                   seed=None, input_sha256="fixture", width=768, height=512, path="input.png", features={})]
    settings = {"qps": [22, 27, 32, 37]}
    primary = encode_jobs(tmp_path, inputs, settings, codecs)
    matrix = encode_jobs(tmp_path, inputs, settings, codecs, matrices=True)
    assert len(matrix) == 24
    shared = [j for j in matrix if j["spec"]["conversion"] == "opencv_444" and j["spec"]["mode"] == "baseline"]
    assert shared == primary


def test_modified_opencv_points_cannot_enter_primary_analyses():
    from vvcstudy.artifacts import read_csv
    from vvcstudy.results import stage_rows
    root = Path(__file__).resolve().parents[1]
    rows = read_csv(root / "results/full/tables/encodings.csv")
    for study in ("partitions", "div2k", "jpeg-vvc"):
        assert all(r["mode"] == "baseline" for r in stage_rows(rows, study))
