import numpy as np
import pytest

from vvcstudy.metrics import expected_yuv_size, opencv_yuv_to_rgb8, psnr, quality


def test_identity_has_infinite_psnr_and_unit_msssim():
    image = np.full((192, 192, 3), 128, dtype=np.uint8)
    assert quality(image, image) == {"psnr_rgb_db": float("inf"), "msssim_rgb": 1.0}
    assert psnr(image, image) == float("inf")


def test_psnr_uses_global_rgb_mse_not_mean_channel_psnr():
    reference = np.zeros((2, 2, 3), dtype=np.uint8)
    reconstructed = reference + np.array([0, 1, 2], dtype=np.uint8)
    assert psnr(reference, reconstructed) == pytest.approx(10 * np.log10(255**2 / (5/3)))


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
def test_yuv_frame_size(chroma, samples, bit_depth):
    assert expected_yuv_size(4, 4, bit_depth, chroma) == samples * (1 if bit_depth == 8 else 2)
