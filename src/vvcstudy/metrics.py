"""Image quality metrics with explicit RGB8 and normalized-plane interfaces.

``quality`` compares original/reconstructed RGB8 arrays for JPEG/VVC. It uses
OpenCV-border MS-SSIM with averaged RGB channel scores. ``msssim_rgb`` retains
the normalized-plane interface used by matrix studies; ``msssim_luma`` is the
separate five-scale Wang implementation. These definitions are not interchangeable.
"""

from __future__ import annotations

import math
from pathlib import Path

import cv2
import numpy as np
from scipy import ndimage, signal


def read_luma(path: Path, width: int, height: int, bit_depth: int) -> list[float]:
    """Read the luma plane from a raw planar YUV file as normalized samples."""

    sample_count = width * height
    data = path.read_bytes()
    bytes_per_sample = 1 if bit_depth <= 8 else 2
    expected_luma_bytes = sample_count * bytes_per_sample
    if len(data) < expected_luma_bytes:
        raise ValueError(f"{path} is too small for a {width}x{height} {bit_depth}-bit luma plane")

    if bit_depth <= 8:
        values = data[:sample_count]
        scale = 255.0
        return [value / scale for value in values]

    values: list[float] = []
    scale = float((1 << bit_depth) - 1)
    for index in range(sample_count):
        offset = index * 2
        values.append(int.from_bytes(data[offset : offset + 2], "little") / scale)
    return values


def msssim_luma(reference: list[float], distorted: list[float], width: int, height: int) -> float:
    """Compute Wang et al. MS-SSIM on normalized luma samples."""

    reference_image = _as_image(reference, width, height)
    distorted_image = _as_image(distorted, width, height)
    return _wang_ms_ssim_image(reference_image, distorted_image)


def read_yuv(path: Path, width: int, height: int, bit_depth: int, chroma_format: str = "420") -> tuple[list[float], list[float], list[float]]:
    """Read planar YUV as normalized Y, U, and V planes."""

    data = path.read_bytes()
    y_count = width * height
    uv_width = width if chroma_format == "444" else width // 2
    uv_height = height if chroma_format == "444" else height // 2
    uv_count = uv_width * uv_height
    expected_size = expected_yuv_size(width, height, bit_depth, chroma_format)
    if len(data) < expected_size:
        raise ValueError(f"{path} is too small for a {width}x{height} {bit_depth}-bit {chroma_format} YUV frame")

    if bit_depth <= 8:
        scale = 255.0
        y = [data[i] / scale for i in range(y_count)]
        u_start = y_count
        u = [data[u_start + i] / scale for i in range(uv_count)]
        v_start = u_start + uv_count
        v = [data[v_start + i] / scale for i in range(uv_count)]
    else:
        scale = float((1 << bit_depth) - 1)
        y = []
        for i in range(y_count):
            offset = i * 2
            y.append(int.from_bytes(data[offset : offset + 2], "little") / scale)
        u_byte_start = y_count * 2
        u = []
        for i in range(uv_count):
            offset = u_byte_start + i * 2
            u.append(int.from_bytes(data[offset : offset + 2], "little") / scale)
        v_byte_start = u_byte_start + uv_count * 2
        v = []
        for i in range(uv_count):
            offset = v_byte_start + i * 2
            v.append(int.from_bytes(data[offset : offset + 2], "little") / scale)
    return y, u, v


def expected_yuv_size(width: int, height: int, bit_depth: int, chroma_format: str = "420") -> int:
    """Return the expected byte size for one planar YUV frame."""

    y_count = width * height
    uv_width = width if chroma_format == "444" else width // 2
    uv_height = height if chroma_format == "444" else height // 2
    samples = y_count + 2 * uv_width * uv_height
    bytes_per_sample = 1 if bit_depth <= 8 else 2
    return samples * bytes_per_sample


def _upsample_chroma(plane: list[float], uv_width: int, uv_height: int) -> list[float]:
    width = uv_width * 2
    out = [0.0] * (width * uv_height * 2)
    for row in range(uv_height):
        for col in range(uv_width):
            val = plane[row * uv_width + col]
            dst_row = row * 2
            dst_col = col * 2
            out[dst_row * width + dst_col] = val
            out[dst_row * width + dst_col + 1] = val
            out[(dst_row + 1) * width + dst_col] = val
            out[(dst_row + 1) * width + dst_col + 1] = val
    return out


def _yuv_to_rgb(y: list[float], u: list[float], v: list[float], width: int, height: int, chroma_format: str = "420") -> tuple[list[float], list[float], list[float]]:
    if chroma_format == "444":
        u_full = u
        v_full = v
    else:
        uv_width = width // 2
        uv_height = height // 2
        u_full = _upsample_chroma(u, uv_width, uv_height)
        v_full = _upsample_chroma(v, uv_width, uv_height)
    n = width * height
    r = [0.0] * n
    g = [0.0] * n
    b = [0.0] * n
    for i in range(n):
        y_val = y[i]
        cb = u_full[i] - 0.5
        cr = v_full[i] - 0.5
        r[i] = max(0.0, min(1.0, y_val + 1.402 * cr))
        g[i] = max(0.0, min(1.0, y_val - 0.344136 * cb - 0.714136 * cr))
        b[i] = max(0.0, min(1.0, y_val + 1.772 * cb))
    return r, g, b


def psnr_rgb(
    ref_r: list[float], ref_g: list[float], ref_b: list[float],
    dis_r: list[float], dis_g: list[float], dis_b: list[float],
) -> float:
    """Compute RGB PSNR from normalized RGB channel samples."""

    n = len(ref_r)
    mse_r = sum((a - b) ** 2 for a, b in zip(ref_r, dis_r)) / n
    mse_g = sum((a - b) ** 2 for a, b in zip(ref_g, dis_g)) / n
    mse_b = sum((a - b) ** 2 for a, b in zip(ref_b, dis_b)) / n
    mse = (mse_r + mse_g + mse_b) / 3.0
    if mse <= 0.0:
        return 99.0
    return 10.0 * math.log10(1.0 / mse)


def msssim_rgb(
    ref_r: list[float], ref_g: list[float], ref_b: list[float],
    dis_r: list[float], dis_g: list[float], dis_b: list[float],
    width: int, height: int,
) -> float:
    """Compute standard Gaussian-window MS-SSIM averaged across RGB channels."""

    score_r = _validated_ms_ssim_image(_as_image(ref_r, width, height), _as_image(dis_r, width, height))
    score_g = _validated_ms_ssim_image(_as_image(ref_g, width, height), _as_image(dis_g, width, height))
    score_b = _validated_ms_ssim_image(_as_image(ref_b, width, height), _as_image(dis_b, width, height))
    return (score_r + score_g + score_b) / 3.0


def _as_image(values: list[float], width: int, height: int) -> np.ndarray:
    return np.asarray(values, dtype=np.float64).reshape((height, width))


def _wang_ms_ssim_image(reference: np.ndarray, distorted: np.ndarray) -> float:
    if reference.shape != distorted.shape:
        raise ValueError(f"MS-SSIM inputs have different shapes: {reference.shape} != {distorted.shape}")

    weights = np.asarray([0.0448, 0.2856, 0.3001, 0.2363, 0.1333], dtype=np.float64)
    if min(reference.shape) / (2 ** (len(weights) - 1)) < 11:
        raise ValueError("Wang et al. MS-SSIM requires at least 176 pixels in each dimension for five scales")

    ref = np.clip(reference.astype(np.float64, copy=False), 0.0, 1.0) * 255.0
    dis = np.clip(distorted.astype(np.float64, copy=False), 0.0, 1.0) * 255.0
    ssim_scores = []
    contrast_scores = []
    for _level in range(len(weights)):
        ssim_value, contrast_value = _wang_ssim_components(ref, dis)
        ssim_scores.append(max(ssim_value, 0.0))
        contrast_scores.append(max(contrast_value, 0.0))
        ref = ndimage.uniform_filter(ref, size=2, mode="reflect")[::2, ::2]
        dis = ndimage.uniform_filter(dis, size=2, mode="reflect")[::2, ::2]

    value = float(np.prod(np.power(contrast_scores[:-1], weights[:-1])))
    value *= float(ssim_scores[-1] ** weights[-1])
    return max(0.0, min(1.0, value))


def _wang_ssim_components(reference: np.ndarray, distorted: np.ndarray) -> tuple[float, float]:
    axis = np.arange(11, dtype=np.float64) - 5.0
    gaussian = np.exp(-(axis**2) / (2.0 * 1.5**2))
    window = np.outer(gaussian, gaussian)
    window /= window.sum()

    mu_ref = signal.convolve2d(reference, window, mode="valid")
    mu_dis = signal.convolve2d(distorted, window, mode="valid")
    mu_ref_sq = mu_ref * mu_ref
    mu_dis_sq = mu_dis * mu_dis
    mu_ref_dis = mu_ref * mu_dis
    sigma_ref_sq = signal.convolve2d(reference * reference, window, mode="valid") - mu_ref_sq
    sigma_dis_sq = signal.convolve2d(distorted * distorted, window, mode="valid") - mu_dis_sq
    sigma_ref_dis = signal.convolve2d(reference * distorted, window, mode="valid") - mu_ref_dis
    c1 = (0.01 * 255.0) ** 2
    c2 = (0.03 * 255.0) ** 2

    contrast_map = (2.0 * sigma_ref_dis + c2) / (sigma_ref_sq + sigma_dis_sq + c2)
    ssim_map = ((2.0 * mu_ref_dis + c1) * (2.0 * sigma_ref_dis + c2)) / (
        (mu_ref_sq + mu_dis_sq + c1) * (sigma_ref_sq + sigma_dis_sq + c2)
    )
    return float(np.mean(ssim_map)), float(np.mean(contrast_map))


def _validated_ms_ssim_image(reference: np.ndarray, distorted: np.ndarray) -> float:
    if reference.shape != distorted.shape:
        raise ValueError(f"MS-SSIM inputs have different shapes: {reference.shape} != {distorted.shape}")

    weights = np.asarray([0.0448, 0.2856, 0.3001, 0.2363, 0.1333], dtype=np.float64)
    levels = _msssim_levels(reference.shape[0], reference.shape[1], len(weights))
    weights = weights[:levels]
    weights = weights / weights.sum()

    ref = np.clip(reference.astype(np.float64, copy=False), 0.0, 1.0)
    dis = np.clip(distorted.astype(np.float64, copy=False), 0.0, 1.0)
    scores = []
    contrast_scores = []

    for level in range(levels):
        ssim_value, contrast_value = _ssim_components(ref, dis)
        scores.append(max(ssim_value, 1e-12))
        contrast_scores.append(max(contrast_value, 1e-12))
        if level < levels - 1:
            ref = _downsample2_image(ref)
            dis = _downsample2_image(dis)

    value = 1.0
    for contrast_value, weight in zip(contrast_scores[:-1], weights[:-1]):
        value *= contrast_value ** weight
    value *= scores[-1] ** weights[-1]
    return float(max(0.0, min(1.0, value)))


def _msssim_levels(height: int, width: int, max_levels: int) -> int:
    levels = 1
    h = height
    w = width
    while levels < max_levels and h >= 22 and w >= 22:
        h //= 2
        w //= 2
        levels += 1
    return levels


def _ssim_components(reference: np.ndarray, distorted: np.ndarray) -> tuple[float, float]:
    c1 = 0.01 ** 2
    c2 = 0.03 ** 2
    kernel_size = _gaussian_kernel_size(reference.shape)

    mu_ref = cv2.GaussianBlur(reference, (kernel_size, kernel_size), 1.5, borderType=cv2.BORDER_REFLECT)
    mu_dis = cv2.GaussianBlur(distorted, (kernel_size, kernel_size), 1.5, borderType=cv2.BORDER_REFLECT)
    mu_ref_sq = mu_ref * mu_ref
    mu_dis_sq = mu_dis * mu_dis
    mu_ref_dis = mu_ref * mu_dis

    sigma_ref_sq = cv2.GaussianBlur(reference * reference, (kernel_size, kernel_size), 1.5, borderType=cv2.BORDER_REFLECT) - mu_ref_sq
    sigma_dis_sq = cv2.GaussianBlur(distorted * distorted, (kernel_size, kernel_size), 1.5, borderType=cv2.BORDER_REFLECT) - mu_dis_sq
    sigma_ref_dis = cv2.GaussianBlur(reference * distorted, (kernel_size, kernel_size), 1.5, borderType=cv2.BORDER_REFLECT) - mu_ref_dis

    contrast_map = (2.0 * sigma_ref_dis + c2) / (sigma_ref_sq + sigma_dis_sq + c2)
    ssim_map = ((2.0 * mu_ref_dis + c1) * (2.0 * sigma_ref_dis + c2)) / ((mu_ref_sq + mu_dis_sq + c1) * (sigma_ref_sq + sigma_dis_sq + c2))
    return float(np.mean(ssim_map)), float(np.mean(contrast_map))


def _gaussian_kernel_size(shape: tuple[int, int]) -> int:
    max_size = min(11, shape[0] if shape[0] % 2 else shape[0] - 1, shape[1] if shape[1] % 2 else shape[1] - 1)
    return max(3, max_size)


def _downsample2_image(values: np.ndarray) -> np.ndarray:
    height = values.shape[0] - values.shape[0] % 2
    width = values.shape[1] - values.shape[1] % 2
    cropped = values[:height, :width]
    return (
        cropped[0::2, 0::2]
        + cropped[1::2, 0::2]
        + cropped[0::2, 1::2]
        + cropped[1::2, 1::2]
    ) * 0.25


def calculate_color_metrics(
    reference_yuv: Path,
    distorted_yuv: Path,
    width: int,
    height: int,
    reference_bit_depth: int = 8,
    distorted_bit_depth: int = 10,
    chroma_format: str = "420",
) -> dict[str, float]:
    """Calculate RGB-derived metrics for reference and reconstructed YUV files."""

    ref_y, ref_u, ref_v = read_yuv(reference_yuv, width, height, reference_bit_depth, chroma_format)
    dis_y, dis_u, dis_v = read_yuv(distorted_yuv, width, height, distorted_bit_depth, chroma_format)
    ref_r, ref_g, ref_b = _yuv_to_rgb(ref_y, ref_u, ref_v, width, height, chroma_format)
    dis_r, dis_g, dis_b = _yuv_to_rgb(dis_y, dis_u, dis_v, width, height, chroma_format)
    return {
        "psnr_rgb": psnr_rgb(ref_r, ref_g, ref_b, dis_r, dis_g, dis_b),
        "msssim_rgb": msssim_rgb(ref_r, ref_g, ref_b, dis_r, dis_g, dis_b, width, height),
    }


def opencv_yuv_to_rgb8(planes: np.ndarray, bit_depth: int) -> np.ndarray:
    """Invert OpenCV YUV, preserving the 8-to-10-bit left-shift convention.

    Input is planar Y,U,V. Chroma neutral is 128 at 8 bits, 512 at 10 bits.
    The inverse acts before clipping/rounding the RGB output to 8 bits.
    """
    yuv = planes.astype(np.float64) / (1 << (bit_depth - 8))
    y, u, v = yuv[0], yuv[1] - 128.0, yuv[2] - 128.0
    rgb = np.stack((y + 1.140 * v, y - 0.395 * u - 0.581 * v,
                    y + 2.032 * u), axis=-1)
    return np.floor(np.clip(rgb, 0.0, 255.0) + 0.5).astype(np.uint8)


def psnr(a: np.ndarray, b: np.ndarray, peak: float = 255.0) -> float:
    mse = np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2)
    return float(10 * np.log10(peak * peak / mse)) if mse else float('inf')


def quality(reference: np.ndarray, reconstruction: np.ndarray) -> dict:
    h, w = reference.shape[:2]
    ref = np.moveaxis(reference.astype(np.float64) / 255, -1, 0)
    rec = np.moveaxis(reconstruction.astype(np.float64) / 255, -1, 0)
    return dict(psnr_rgb_db=psnr(reference, reconstruction),
                msssim_rgb=msssim_rgb(*ref, *rec, w, h))
