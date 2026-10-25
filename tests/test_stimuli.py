from pathlib import Path
import hashlib

import numpy as np
import pytest

from vvcstudy.stimuli import (
    add_achromatic_awgn, add_sinusoidal_interference, complexity_metrics,
    derive_image_seed, read_png, write_png,
)


def test_awgn_is_deterministic_and_achromatic() -> None:
    image = np.full((128, 128, 3), (96, 128, 160), dtype=np.uint8)

    first = add_achromatic_awgn(image, sigma=10.0, seed=42)
    repeated = add_achromatic_awgn(image, sigma=10.0, seed=42)
    different = add_achromatic_awgn(image, sigma=10.0, seed=43)

    assert np.array_equal(first, repeated)
    assert not np.array_equal(first, different)
    assert np.array_equal(first[:, :, 1].astype(int) - first[:, :, 0], np.full((128, 128), 32))
    assert np.array_equal(first[:, :, 2].astype(int) - first[:, :, 1], np.full((128, 128), 32))
    assert np.std(first[:, :, 0].astype(float) - image[:, :, 0]) == pytest.approx(10.0, abs=0.2)


def test_sinusoidal_interference_has_fixed_period_and_preserves_alpha() -> None:
    image = np.full((32, 24, 4), 128, dtype=np.uint8)
    image[:, :, 3] = np.arange(32, dtype=np.uint8)[:, None]

    result = add_sinusoidal_interference(image, amplitude=20.0)

    assert np.array_equal(result[0, :, :3], result[16, :, :3])
    assert np.all(result[4, :, :3] == 148)
    assert np.all(result[12, :, :3] == 108)
    assert np.array_equal(result[:, :, 3], image[:, :, 3])


def test_complexity_metrics_distinguish_flat_and_edge_rich_images() -> None:
    flat = np.full((64, 64), 128, dtype=np.uint8)
    stripes = np.tile(np.repeat([0, 255], 4), (64, 8)).astype(np.uint8)

    flat_metrics = complexity_metrics(flat)
    stripe_metrics = complexity_metrics(stripes)

    assert flat_metrics == {"sobel_si": 0.0}
    assert stripe_metrics["sobel_si"] > 0


def test_image_seed_is_stable_and_image_specific() -> None:
    assert derive_image_seed(20260811, 0) == derive_image_seed(20260811, 0)
    assert derive_image_seed(20260811, 0) != derive_image_seed(20260811, 1)


def test_registered_noise_seed_rounding_and_clipping_regression():
    y, x = np.mgrid[0:64, 0:80]
    image = np.stack([(3*x+5*y) % 256, (7*x+y) % 256, (x+11*y) % 256], axis=-1).astype(np.uint8)
    seed = derive_image_seed(20260811, 17)
    assert seed == 5131668710055686531
    result = add_achromatic_awgn(image, sigma=30, seed=seed)
    assert hashlib.sha256(result.tobytes()).hexdigest() == "1a664683e3117bc0f1b284f76bad270f9993b476edce30132ed8af12c852ec63"


def test_png_roundtrip_handles_unicode_path_and_keeps_bgr_pixels(tmp_path: Path):
    path = tmp_path / "зображення.png"
    image = np.random.default_rng(61).integers(0, 256, (13, 17, 3), dtype=np.uint8)
    write_png(path, image)
    np.testing.assert_array_equal(read_png(path), image)
