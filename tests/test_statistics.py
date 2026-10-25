from __future__ import annotations

import itertools

import numpy as np
import pytest
from scipy.stats import rankdata, spearmanr

from vvcstudy.statistics import (
    holm_adjust,
    paired_spearman_bootstrap,
    percentile_intervals,
    simultaneous_basic_intervals,
    spearman_independence_permutation,
    spearman_last_axis,
)


def test_spearman_matches_scipy_with_ties_and_broadcast_axes() -> None:
    x = np.array([[1, 1, 4, 2, 8], [2, 7, 7, 0, 3]], dtype=float)
    y = np.array([3, 9, 0, 3, 2], dtype=float)
    expected = np.array([spearmanr(row, y).statistic for row in x])

    np.testing.assert_allclose(spearman_last_axis(x, y), expected, atol=1e-14)
    np.testing.assert_allclose(spearman_last_axis(x[None], y), expected[None], atol=1e-14)
    np.testing.assert_allclose(spearman_last_axis(x, -x), [-1, -1], atol=1e-14)
    order = [4, 0, 3, 1, 2]
    np.testing.assert_allclose(spearman_last_axis(x[:, order], y[order]), expected, atol=1e-14)


def test_spearman_preserves_constant_and_nonfinite_cells_as_nan() -> None:
    x = np.array([[1, 1, 1], [0, np.nan, 2], [0, np.inf, 2], [0, 1, 2]])
    result = spearman_last_axis(x, np.array([2, 0, 1]))

    assert np.isnan(result[:3]).all()
    assert np.isfinite(result[3])
    assert np.isnan(spearman_last_axis([0, 1, 2], [1, 1, 1]))


def test_bootstrap_preserves_paired_cells_and_reranks_duplicate_images() -> None:
    x = np.array([[0, 1, 2, 3, 4], [7, 7, 2, 3, 9]], dtype=float)
    y = np.array([[8, 4, 3, 7, 0], [5, 1, 8, 3, 6]], dtype=float)
    seed, count = 1729, 17
    observed, distribution = paired_spearman_bootstrap(x, y, n_resamples=count, seed=seed, batch_size=4)
    indices = np.random.Generator(np.random.PCG64(seed)).integers(0, 5, size=(count, 5))
    expected = np.array([
        [spearmanr(x[cell, index], y[cell, index]).statistic for cell in range(2)]
        for index in indices
    ])

    np.testing.assert_allclose(observed, [spearmanr(a, b).statistic for a, b in zip(x, y)])
    np.testing.assert_allclose(distribution, expected, atol=1e-14)
    global_x, global_y = rankdata(x[0]), rankdata(y[0])
    wrong = [np.corrcoef(global_x[index], global_y[index])[0, 1] for index in indices]
    assert not np.allclose(distribution[:, 0], wrong)


@pytest.mark.parametrize("batch_size", [1, 4, 17, 100])
def test_bootstrap_random_stream_is_independent_of_batch_size(batch_size: int) -> None:
    x = np.array([[0, 1, 2, 3, 4], [0, 2, 2, 2, 4]])
    y = x[:, [4, 0, 3, 1, 2]]
    reference = paired_spearman_bootstrap(x, y, n_resamples=43, seed=28, batch_size=7)
    result = paired_spearman_bootstrap(x, y, n_resamples=43, seed=28, batch_size=batch_size)

    np.testing.assert_array_equal(result[0], reference[0])
    np.testing.assert_array_equal(result[1], reference[1])


def test_identical_correlations_have_zero_paired_contrasts() -> None:
    x = np.tile(np.arange(12), (2, 1))
    y = np.tile([5, 0, 2, 1, 3, 9, 7, 6, 4, 11, 10, 8], (2, 1))
    observed, distribution = paired_spearman_bootstrap(x, y, n_resamples=97, batch_size=11)
    contrast = observed[:1] - observed[1:]
    contrast_distribution = distribution[:, :1] - distribution[:, 1:]

    np.testing.assert_array_equal(contrast_distribution, np.zeros((97, 1)))
    result = simultaneous_basic_intervals(contrast, contrast_distribution)
    np.testing.assert_array_equal(result["low"], [0])
    np.testing.assert_array_equal(result["high"], [0])
    assert result["degenerate"].tolist() == [True]


def test_interval_rules_count_invalid_resamples_without_dropping_them() -> None:
    observed = np.array([0.3, 0.2, np.nan, 0.5])
    distribution = np.array([[0, 0.2, 0.2, 0.5], [0.5, np.nan, 0.3, 0.5], [1, 0.4, 0.4, 0.5]])
    result = percentile_intervals(observed, distribution)

    np.testing.assert_allclose(result["low"], [0.025, np.nan, np.nan, 0.5], equal_nan=True)
    np.testing.assert_allclose(result["high"], [0.975, np.nan, np.nan, 0.5], equal_nan=True)
    assert result["invalid_resamples"].tolist() == [0, 1, 0, 0]
    assert result["evaluable"].tolist() == [True, False, False, True]
    assert result["degenerate"].tolist() == [False, False, False, True]


def test_simultaneous_intervals_use_joint_max_errors_and_clip_support() -> None:
    observed = np.array([1.9, -1.9, 0.2])
    distribution = np.array([[1.8, -1.7, 0.2], [1.6, -1.9, np.nan], [1.9, -1.5, 0.1]])
    result = simultaneous_basic_intervals(observed, distribution, alpha=0.25)

    # Joint errors by replicate are 0.2, 0.3 and 0.4; their 0.75 quantile is 0.35.
    assert result["critical_value"] == pytest.approx(0.35)
    np.testing.assert_allclose(result["low"], [1.55, -2, np.nan], equal_nan=True)
    np.testing.assert_allclose(result["high"], [2, -1.55, np.nan], equal_nan=True)
    empty = simultaneous_basic_intervals(np.array([np.nan]), np.full((3, 1), np.nan))
    assert np.isnan(empty["critical_value"])
    assert empty["invalid_resamples"].tolist() == [3]


def test_difference_of_changes_and_mixed_families_keep_their_support() -> None:
    observed = np.array([-1.5, -3.8])
    draws = np.array([[-1.0, -3.3], [-0.5, -2.8]])
    result = simultaneous_basic_intervals(observed, draws, alpha=0.5, bounds=(-4, 4))
    assert result["critical_value"] == pytest.approx(0.75)
    np.testing.assert_allclose(result["low"], [-2.25, -4])
    np.testing.assert_allclose(result["high"], [-0.75, -3.05])
    mixed = simultaneous_basic_intervals(
        observed, draws, alpha=0.5, bounds=(np.array([-2, -4]), np.array([2, 4])))
    np.testing.assert_allclose(mixed["low"], [-2, -4])
    np.testing.assert_array_equal(mixed["high"], result["high"])
    with pytest.raises(ValueError, match="bounds"):
        simultaneous_basic_intervals(observed, draws, bounds=(2, -2))


def test_holm_keeps_missing_hypotheses_in_family_size_and_original_order() -> None:
    pvalues = np.array([0.04, np.nan, 0.01, 0.03, 0.8])
    np.testing.assert_allclose(holm_adjust(pvalues), [0.12, np.nan, 0.05, 0.12, 1.0], equal_nan=True)
    np.testing.assert_array_equal(holm_adjust([0, 0, 1]), [0, 0, 1])
    assert np.isnan(holm_adjust([np.nan])).all()


@pytest.mark.parametrize("batch_size", [1, 4, 17, 100])
def test_permutation_stream_is_independent_of_batch_size(batch_size: int) -> None:
    x = np.array([[1, 1, 4, 2, 8], [2, 7, 7, 0, 3]], dtype=float)
    y = x[:, [3, 1, 0, 4, 2]]
    reference = spearman_independence_permutation(x, y, n_resamples=43, seed=72, batch_size=7)
    result = spearman_independence_permutation(x, y, n_resamples=43, seed=72, batch_size=batch_size)

    for key in reference:
        np.testing.assert_array_equal(result[key], reference[key])


def test_monte_carlo_p_matches_tiny_exhaustive_permutation_fixture() -> None:
    x, y = np.array([0, 1, 2]), np.array([0, 1, 2])
    null = np.array([spearmanr(x, y[list(order)]).statistic for order in itertools.permutations(range(3))])
    exact = np.mean(np.abs(null) >= 1 - 1e-12)
    result = spearman_independence_permutation(x[None], y[None], n_resamples=5_999, seed=31)

    assert exact == pytest.approx(1 / 3)
    assert result["pvalue"][0] == pytest.approx(exact, abs=0.02)
    assert result["pvalue"][0] == (result["extreme_count"][0] + 1) / 6_000


def test_permutation_plus_one_ties_and_constant_inputs() -> None:
    x = np.array([[0, 1, 2, 3, 4, 5], [0, 0, 1, 1, 2, 2], [1, 1, 1, 1, 1, 1]])
    result = spearman_independence_permutation(x, x, n_resamples=7, seed=17, batch_size=3)

    assert np.all(result["pvalue"][:2] >= 1 / 8)
    assert np.all(result["pvalue"][:2] <= 1)
    assert np.isnan(result["pvalue"][2])
    assert result["invalid_resamples"].tolist() == [0, 0, 7]


@pytest.mark.parametrize("function", [paired_spearman_bootstrap, spearman_independence_permutation])
def test_resampling_rejects_unpaired_shapes(function) -> None:
    with pytest.raises(ValueError, match="identical"):
        function(np.zeros((2, 5)), np.zeros((2, 4)), n_resamples=2)
    with pytest.raises(ValueError, match="positive integer"):
        function(np.zeros((2, 5)), np.zeros((2, 5)), n_resamples=0)


def test_interval_and_pvalue_validation() -> None:
    with pytest.raises(ValueError, match="distribution"):
        percentile_intervals(np.array([0]), np.zeros((2, 3)))
    with pytest.raises(ValueError, match="confidence"):
        percentile_intervals(np.array([0]), np.zeros((2, 1)), confidence_level=1)
    with pytest.raises(ValueError, match="alpha"):
        simultaneous_basic_intervals(np.array([0]), np.zeros((2, 1)), alpha=0)
    with pytest.raises(ValueError, match="p_values"):
        holm_adjust([0.2, 1.01])
