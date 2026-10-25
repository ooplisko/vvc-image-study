"""Fixed image-level descriptors for the VTM secondary-analysis protocol."""

from __future__ import annotations

from collections.abc import Sequence

import cv2
import numpy as np

from vvcstudy.stimuli import _to_luma, complexity_metrics


OFFSETS = ((0, 1), (1, 1), (1, 0), (1, -1))
FEATURES = ("sobel_si", "luma_sd", "edge_fraction", "glcm_contrast", "glcm_entropy", "glcm_homogeneity")
EDGE_KERNELS = np.asarray([
    [[-1, -2, -1], [0, 0, 0], [1, 2, 1]],
    [[-2, -1, 0], [-1, 0, 1], [0, 1, 2]],
    [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
    [[0, -1, -2], [1, 0, -1], [2, 1, 0]],
], dtype=np.float64)


def glcm_matrices(
    luma: np.ndarray, levels: int = 8, offsets: Sequence[tuple[int, int]] = OFFSETS
) -> np.ndarray:
    """Symmetric probability matrices from valid pairs, one per (row, column) offset."""

    if luma.ndim != 2 or luma.dtype != np.uint8:
        raise ValueError("GLCM input must be a two-dimensional uint8 luma plane")
    if levels not in (8, 32) or not offsets:
        raise ValueError("The fixed protocol supports 8 or 32 levels and nonempty offsets")
    quantized = luma.astype(np.int64) * levels // 256
    height, width = luma.shape
    matrices = []
    for dr, dc in offsets:
        if (dr, dc) not in OFFSETS:
            raise ValueError("Offset is outside the registered nearest-neighbor axes")
        r0, r1 = max(0, -dr), min(height, height - dr)
        c0, c1 = max(0, -dc), min(width, width - dc)
        first = quantized[r0:r1, c0:c1]
        second = quantized[r0 + dr:r1 + dr, c0 + dc:c1 + dc]
        if first.size == 0:
            raise ValueError("Image is too small for the registered GLCM offset")
        counts = np.bincount((first * levels + second).ravel(), minlength=levels**2).reshape(levels, levels)
        counts = counts + counts.T
        matrices.append(counts / counts.sum())
    return np.asarray(matrices)


def glcm_features(
    luma: np.ndarray, levels: int = 8, offsets: Sequence[tuple[int, int]] = OFFSETS
) -> dict[str, float]:
    """Average properties of normalized direction matrices, never of their mixture."""

    probabilities = glcm_matrices(luma, levels, offsets)
    differences_squared = (np.arange(levels)[:, None] - np.arange(levels)[None, :]) ** 2
    logs = np.zeros_like(probabilities)
    np.log(probabilities, out=logs, where=probabilities > 0)
    return {
        "glcm_contrast": float(np.sum(probabilities * differences_squared, axis=(1, 2)).mean()),
        "glcm_entropy": float(-np.sum(probabilities * logs, axis=(1, 2)).mean()),
        "glcm_homogeneity": float(np.sum(probabilities / (1 + differences_squared), axis=(1, 2)).mean()),
    }


def spatial_complexity_metrics(image: np.ndarray) -> dict[str, float]:
    """The six fixed primary descriptors on the actual 8-bit PNG stimulus."""

    luma = _to_luma(image)
    if min(luma.shape) < 3 or luma.dtype != np.uint8:
        raise ValueError("Descriptors require uint8 images at least 3 by 3 pixels")
    values = luma.astype(np.float64)
    edge_sum = sum(np.abs(cv2.filter2D(values, cv2.CV_64F, kernel)) for kernel in EDGE_KERNELS)
    return {
        **complexity_metrics(image),
        "luma_sd": float(np.std(values)),
        "edge_fraction": float(np.mean(edge_sum[1:-1, 1:-1] > 150)),
        **glcm_features(luma),
    }
