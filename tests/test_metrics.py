import numpy as np
import pytest

from vvcstudy.metrics import (
    _validated_ms_ssim_image,
    _wang_ms_ssim_image,
    calculate_color_metrics,
    expected_yuv_size,
    msssim_rgb,
    opencv_yuv_to_rgb8,
    psnr,
    psnr_rgb,
    quality,
    read_yuv,
)


def test_rgb_quality_preserves_registered_numeric_definition():
    y, x = np.mgrid[0:64, 0:80]
    bgr = np.stack([(3*x+5*y) % 256, (7*x+y) % 256, (x+11*y) % 256], axis=-1).astype(np.uint8)
    reference = np.flip(bgr, axis=-1).copy()
    distorted = np.clip(reference.astype(np.int16) + ((x % 9)-4)[..., None], 0, 255).astype(np.uint8)

    assert quality(reference, distorted) == pytest.approx({
        "psnr_rgb_db": 40.00985557217338,
        "msssim_rgb": 0.9971848267790113,
    }, abs=1e-12)


def test_two_msssim_definitions_keep_distinct_regression_values():
    y, x = np.mgrid[0:256, 0:256]
    reference = ((3*x + 5*y + (x*y) % 37) % 256).astype(np.uint8)
    distorted = np.clip(reference.astype(np.int16) + ((x % 9)-4), 0, 255).astype(np.uint8)

    assert _wang_ms_ssim_image(reference/255, distorted/255) == pytest.approx(0.9980908180234471, abs=1e-12)
    assert _validated_ms_ssim_image(reference/255, distorted/255) == pytest.approx(0.9980556308053101, abs=1e-12)


def test_identity_and_capped_psnr_cap_are_explicit():
    image = np.full((8, 8, 3), 128, dtype=np.uint8)
    assert quality(image, image) == {"psnr_rgb_db": float("inf"), "msssim_rgb": 1.0}
    assert psnr(image, image) == float("inf")
    channel = [0.0, 0.5, 1.0]
    assert psnr_rgb(channel, channel, channel, channel, channel, channel) == 99.0


def test_rgb_msssim_is_mean_of_channel_scores():
    rng = np.random.default_rng(73)
    ref = rng.uniform(size=(3, 32, 40))
    rec = np.clip(ref + np.array([0.01, 0.04, 0.1])[:, None, None], 0, 1)
    expected = np.mean([_validated_ms_ssim_image(a, b) for a, b in zip(ref, rec)])
    assert msssim_rgb(*ref, *rec, 40, 32) == pytest.approx(expected, abs=1e-15)


def test_opencv_inverse_preserves_neutral_samples_and_8_to_10_bit_shift():
    gray = np.arange(256, dtype=np.uint16)
    planes = np.stack((gray, np.full(256, 128), np.full(256, 128)))[:, None, :]
    expected = np.repeat(gray[None, :, None], 3, axis=2).astype(np.uint8)
    np.testing.assert_array_equal(opencv_yuv_to_rgb8(planes, 8), expected)
    np.testing.assert_array_equal(opencv_yuv_to_rgb8(planes*4, 10), expected)


def test_opencv_inverse_rounding_and_clipping_regression():
    planes = np.array([[[80, 110, 200]], [[100, 135, 128]], [[150, 115, 128]]])
    expected = np.array([[[105, 78, 23], [95, 115, 124], [200, 200, 200]]], dtype=np.uint8)
    np.testing.assert_array_equal(opencv_yuv_to_rgb8(planes*4, 10), expected)


@pytest.mark.parametrize("chroma,samples", [("444", 48), ("420", 24)])
@pytest.mark.parametrize("bit_depth", [8, 10])
def test_yuv_plane_sizes_and_bit_depth_normalization(tmp_path, chroma, samples, bit_depth):
    dtype = np.uint8 if bit_depth == 8 else np.dtype("<u2")
    values = np.arange(samples, dtype=dtype)
    path = tmp_path / "planes.yuv"
    path.write_bytes(values.tobytes())
    planes = read_yuv(path, 4, 4, bit_depth, chroma)
    assert path.stat().st_size == expected_yuv_size(4, 4, bit_depth, chroma)
    np.testing.assert_array_equal(np.concatenate(planes), values / ((1 << bit_depth)-1))


def test_yuv_incomplete_frame_is_rejected(tmp_path):
    path = tmp_path / "short.yuv"
    path.write_bytes(bytes(10))
    with pytest.raises(ValueError, match="too small"):
        read_yuv(path, 4, 4, 10, "444")


def test_yuv_color_metrics_identity(tmp_path):
    path = tmp_path / "frame.yuv"
    path.write_bytes(np.full((3, 8, 8), 128, dtype=np.uint8).tobytes())
    assert calculate_color_metrics(path, path, 8, 8, 8, 8, "444") == {"psnr_rgb": 99.0, "msssim_rgb": 1.0}


def test_wang_requires_five_valid_scales_and_matching_shapes():
    with pytest.raises(ValueError, match="176"):
        _wang_ms_ssim_image(np.zeros((64, 64)), np.zeros((64, 64)))
    with pytest.raises(ValueError, match="different shapes"):
        _validated_ms_ssim_image(np.zeros((32, 32)), np.zeros((32, 31)))
