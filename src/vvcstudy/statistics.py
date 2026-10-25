"""Paired image-level Spearman inference for the VTM secondary analysis.

Bootstrap intervals are approximate. Permutation p-values test independence,
not equality of two dependent correlations or a general zero-correlation null.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import rankdata


def _normalized_ranks(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    ranks = rankdata(values, method="average", axis=-1, nan_policy="propagate")
    centered = ranks - ranks.mean(axis=-1, keepdims=True)
    norm = np.sqrt(np.sum(centered * centered, axis=-1, keepdims=True))
    valid = np.isfinite(values).all(axis=-1, keepdims=True) & (norm > 0)
    return np.divide(centered, norm, out=np.full_like(centered, np.nan), where=valid)


def spearman_last_axis(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Correlate average ranks along the last axis, propagating invalid cells.

    Leading axes broadcast. Constant or non-finite inputs produce NaN; ties
    receive their average rank. Each call ranks its supplied observations anew.
    """

    x, y = np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64)
    if x.ndim == 0 or y.ndim == 0 or x.shape[-1] != y.shape[-1] or x.shape[-1] < 2:
        raise ValueError("x and y must have the same last-axis length of at least two")
    result = np.sum(_normalized_ranks(x) * _normalized_ranks(y), axis=-1)
    return np.clip(result, -1.0, 1.0)


def _paired_cells(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    x, y = np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64)
    if x.ndim != 2 or y.shape != x.shape or x.shape[0] == 0 or x.shape[1] < 2:
        raise ValueError("x and y must have identical nonempty (cells, images) shapes")
    return x, y


def _resampling_sizes(n_resamples: int, batch_size: int) -> None:
    if not isinstance(n_resamples, (int, np.integer)) or n_resamples < 1:
        raise ValueError("n_resamples must be a positive integer")
    if not isinstance(batch_size, (int, np.integer)) or batch_size < 1:
        raise ValueError("batch_size must be a positive integer")


def paired_spearman_bootstrap(
    x: np.ndarray,
    y: np.ndarray,
    *,
    n_resamples: int = 99_999,
    seed: int = 20260905,
    batch_size: int = 512,
) -> tuple[np.ndarray, np.ndarray]:
    """Return observed correlations and a (resamples, cells) distribution.

    Every replicate uses one shared image-index vector for all cells and both
    arrays. Ranks are recomputed after sampling. Invalid replicates are retained
    as NaN, never redrawn. Changing batch_size preserves the random draws.
    """

    x, y = _paired_cells(x, y)
    _resampling_sizes(n_resamples, batch_size)
    rng = np.random.Generator(np.random.PCG64(seed))
    distribution = np.empty((n_resamples, x.shape[0]), dtype=np.float64)
    for start in range(0, n_resamples, batch_size):
        stop = min(start + batch_size, n_resamples)
        indices = rng.integers(0, x.shape[1], size=(stop - start, x.shape[1]))
        # Advanced indexing yields (cells, resamples, images).
        distribution[start:stop] = spearman_last_axis(x[:, indices], y[:, indices]).T
    return spearman_last_axis(x, y), distribution


def _interval_inputs(
    observed: np.ndarray, distribution: np.ndarray
) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    observed = np.asarray(observed, dtype=np.float64)
    distribution = np.asarray(distribution, dtype=np.float64)
    if (
        observed.ndim != 1
        or distribution.ndim != 2
        or distribution.shape[1] != observed.size
        or distribution.shape[0] == 0
    ):
        raise ValueError("expected observed (cells,) and distribution (resamples, cells)")
    invalid = np.sum(~np.isfinite(distribution), axis=0)
    evaluable = np.isfinite(observed) & (invalid == 0)
    degenerate = evaluable & np.all(distribution == distribution[0], axis=0)
    result = {
        "low": np.full(observed.shape, np.nan),
        "high": np.full(observed.shape, np.nan),
        "invalid_resamples": invalid,
        "evaluable": evaluable,
        "degenerate": degenerate,
    }
    return observed, distribution, result


def percentile_intervals(
    observed: np.ndarray,
    distribution: np.ndarray,
    *,
    confidence_level: float = 0.95,
) -> dict[str, np.ndarray]:
    """Pointwise percentile intervals; any invalid replicate makes its CI NaN."""

    if not 0 < confidence_level < 1:
        raise ValueError("confidence_level must be between zero and one")
    _, distribution, result = _interval_inputs(observed, distribution)
    valid = result["evaluable"]
    tail = (1.0 - confidence_level) / 2.0
    if valid.any():
        bounds = np.quantile(distribution[:, valid], [tail, 1.0 - tail], axis=0, method="linear")
        result["low"][valid], result["high"][valid] = bounds
    return result


def simultaneous_basic_intervals(
    observed: np.ndarray,
    distribution: np.ndarray,
    *,
    alpha: float = 0.025,
    bounds: tuple[float | np.ndarray, float | np.ndarray] = (-2.0, 2.0),
) -> dict[str, np.ndarray | float]:
    """Max-absolute-error basic bootstrap CIs for one supplied contrast family.

    Bounds default to [-2, 2] for correlation differences; differences of changes
    need [-4, 4]. Arrays permit coordinate-specific support in a mixed family.
    Only fully evaluable contrasts
    enter the maximum; others retain NaN intervals and explicit status/counts.
    The caller constructs paired contrasts before calling this function.
    """

    if not 0 < alpha < 1:
        raise ValueError("alpha must be between zero and one")
    observed, distribution, result = _interval_inputs(observed, distribution)
    lower, upper = (np.broadcast_to(bound, observed.shape) for bound in bounds)
    if not (np.isfinite(lower).all() and np.isfinite(upper).all() and (lower < upper).all()):
        raise ValueError("Interval bounds must be finite and increasing")
    valid = result["evaluable"]
    critical = np.nan
    if valid.any():
        errors = np.max(np.abs(distribution[:, valid] - observed[valid]), axis=1)
        critical = float(np.quantile(errors, 1.0 - alpha, method="linear"))
        result["low"][valid] = np.maximum(lower[valid], observed[valid] - critical)
        result["high"][valid] = np.minimum(upper[valid], observed[valid] + critical)
    return {**result, "critical_value": critical}


def holm_adjust(p_values: np.ndarray) -> np.ndarray:
    """Holm adjustment, retaining NaN hypotheses in the original family size."""

    p_values = np.asarray(p_values, dtype=np.float64)
    if p_values.ndim != 1 or np.any((p_values < 0) | (p_values > 1)):
        raise ValueError("p_values must be a one-dimensional array in [0, 1] or NaN")
    valid = np.flatnonzero(np.isfinite(p_values))
    order = valid[np.argsort(p_values[valid], kind="stable")]
    adjusted = np.full(p_values.shape, np.nan)
    factors = p_values.size - np.arange(order.size)
    adjusted[order] = np.minimum(1.0, np.maximum.accumulate(factors * p_values[order]))
    return adjusted


def spearman_independence_permutation(
    x: np.ndarray,
    y: np.ndarray,
    *,
    n_resamples: int = 99_999,
    seed: int = 20260906,
    batch_size: int = 512,
) -> dict[str, np.ndarray]:
    """Monte Carlo independence p-values using |rho| and the plus-one rule.

    Uniform permutations of image labels are shared across cells. Marginal
    ranks can be reused because permutation, unlike bootstrap, preserves all
    observations. This is not a test of equality of dependent correlations.
    """

    x, y = _paired_cells(x, y)
    _resampling_sizes(n_resamples, batch_size)
    x_ranks, y_ranks = _normalized_ranks(x), _normalized_ranks(y)
    observed = np.clip(np.sum(x_ranks * y_ranks, axis=-1), -1.0, 1.0)
    extreme = np.zeros(x.shape[0], dtype=np.int64)
    invalid = np.zeros(x.shape[0], dtype=np.int64)
    rng = np.random.Generator(np.random.PCG64(seed))
    for start in range(0, n_resamples, batch_size):
        size = min(batch_size, n_resamples - start)
        indices = rng.permuted(np.broadcast_to(np.arange(x.shape[1]), (size, x.shape[1])), axis=1)
        permuted = np.clip(np.sum(x_ranks[:, None, :] * y_ranks[:, indices], axis=-1), -1.0, 1.0)
        invalid += np.sum(~np.isfinite(permuted), axis=1)
        extreme += np.sum(np.abs(permuted) >= np.abs(observed[:, None]) - 1e-12, axis=1)
    pvalue = (extreme + 1.0) / (n_resamples + 1.0)
    pvalue[(invalid > 0) | ~np.isfinite(observed)] = np.nan
    return {
        "observed": observed,
        "pvalue": pvalue,
        "extreme_count": extreme,
        "invalid_resamples": invalid,
    }
