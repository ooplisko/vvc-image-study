from __future__ import annotations

import numpy as np
import pytest

from vvcstudy.descriptors import glcm_features, glcm_matrices, spatial_complexity_metrics
from vvcstudy.stimuli import complexity_metrics


def test_flat_image_has_known_descriptors() -> None:
    values = spatial_complexity_metrics(np.full((8, 9), 128, dtype=np.uint8))
    assert values == {"sobel_si": 0, "luma_sd": 0, "edge_fraction": 0,
                      "glcm_contrast": 0, "glcm_entropy": 0, "glcm_homogeneity": 1}


def test_glcm_counts_valid_horizontal_pairs_and_symmetrizes() -> None:
    image = np.array([[0, 0, 32], [0, 32, 32]], dtype=np.uint8)
    matrix = glcm_matrices(image, offsets=((0, 1),))[0]
    expected = np.zeros((8, 8))
    expected[:2, :2] = 0.25
    np.testing.assert_array_equal(matrix, expected)
    assert glcm_features(image, offsets=((0, 1),)) == pytest.approx({
        "glcm_contrast": 0.5, "glcm_entropy": np.log(4), "glcm_homogeneity": 0.75})


def test_uniform_quantizer_handles_top_value_without_overflow() -> None:
    image = np.array([[0, 255], [255, 0]], dtype=np.uint8)
    for levels in (8, 32):
        matrix = glcm_matrices(image, levels, ((0, 1),))[0]
        assert matrix[0, levels - 1] == matrix[levels - 1, 0] == 0.5
        assert glcm_features(image, levels, ((0, 1),))["glcm_contrast"] == (levels - 1) ** 2


def test_four_axis_glcm_is_invariant_to_quarter_turn_and_reflection() -> None:
    image = np.random.default_rng(12).integers(0, 256, (7, 9), dtype=np.uint8)
    original = glcm_features(image)
    assert glcm_features(np.rot90(image)) == pytest.approx(original)
    assert glcm_features(np.fliplr(image)) == pytest.approx(original)
    assert np.allclose(glcm_matrices(image).sum(axis=(1, 2)), 1)


def test_entropy_is_mean_of_direction_entropies_not_entropy_of_mixture() -> None:
    image = np.tile(np.array([0, 255, 0, 255], dtype=np.uint8), (4, 1))
    matrices = glcm_matrices(image)
    mixture = matrices.mean(axis=0)
    mixture_entropy = -np.sum(mixture[mixture > 0] * np.log(mixture[mixture > 0]))
    assert glcm_features(image)["glcm_entropy"] == pytest.approx(np.log(2))
    assert mixture_entropy > glcm_features(image)["glcm_entropy"]


def test_equal_histograms_can_have_different_spatial_descriptors() -> None:
    bands = np.tile(np.array([0, 0, 0, 0, 255, 255, 255, 255], dtype=np.uint8), (8, 1))
    stripes = bands[:, [0, 4, 1, 5, 2, 6, 3, 7]]
    first, second = spatial_complexity_metrics(bands), spatial_complexity_metrics(stripes)
    assert first["luma_sd"] == second["luma_sd"]
    assert first["glcm_contrast"] < second["glcm_contrast"]


def test_edge_fraction_uses_four_masks_and_strict_published_threshold() -> None:
    # A bottom-minus-top difference of 15 gives responses 60+45+0+45=150.
    ramp = np.tile(np.array([0, 8, 15], dtype=np.uint8)[:, None], (1, 3))
    assert spatial_complexity_metrics(ramp)["edge_fraction"] == 0
    stronger = np.tile(np.array([0, 8, 16], dtype=np.uint8)[:, None], (1, 3))
    assert spatial_complexity_metrics(stronger)["edge_fraction"] == 1


def test_color_baseline_compatibility_and_determinism() -> None:
    image = np.random.default_rng(19).integers(0, 256, (11, 13, 3), dtype=np.uint8)
    first = spatial_complexity_metrics(image)
    assert first == spatial_complexity_metrics(image)
    assert first["sobel_si"] == complexity_metrics(image)["sobel_si"]


def test_invalid_glcm_and_too_small_images_fail() -> None:
    with pytest.raises(ValueError, match="uint8"):
        glcm_matrices(np.zeros((3, 3), dtype=float))
    with pytest.raises(ValueError, match="at least"):
        spatial_complexity_metrics(np.zeros((2, 2), dtype=np.uint8))


def test_all_six_descriptors_preserve_registered_color_fixture():
    y, x = np.mgrid[0:64, 0:80]
    image = np.stack([(3*x+5*y) % 256, (7*x+y) % 256, (x+11*y) % 256], axis=-1).astype(np.uint8)
    assert spatial_complexity_metrics(image) == pytest.approx({
        "sobel_si": 122.9703569350242,
        "luma_sd": 49.81261376534254,
        "edge_fraction": 0.14536807278742764,
        "glcm_contrast": 0.6967629219409284,
        "glcm_entropy": 2.524874090041326,
        "glcm_homogeneity": 0.8868389764011739,
    }, abs=1e-12)
