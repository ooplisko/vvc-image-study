"""Shared original-RGB8 quality metrics and the OpenCV reconstruction inverse."""

from __future__ import annotations

from functools import lru_cache

import numpy as np

CURRENT_RGB_PROTOCOL = "pytorch_rgb_v1"


def expected_yuv_size(width: int, height: int, bit_depth: int, chroma_format: str = "420") -> int:
    """Return the expected byte size for one planar YUV frame."""

    y_count = width * height
    uv_width = width if chroma_format == "444" else width // 2
    uv_height = height if chroma_format == "444" else height // 2
    samples = y_count + 2 * uv_width * uv_height
    bytes_per_sample = 1 if bit_depth <= 8 else 2
    return samples * bytes_per_sample


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


@lru_cache(maxsize=1)
def _reference_backend():
    from importlib.metadata import version
    try:
        import torch
        from pytorch_msssim import ms_ssim
    except ImportError as exc:
        raise RuntimeError("Install the locked dependencies: uv sync --locked") from exc
    if version("pytorch-msssim") != "1.0.0" or version("torch").split("+")[0] != "2.8.0":
        raise ValueError("The reference metric protocol requires torch 2.8.0 and pytorch-msssim 1.0.0.")
    torch.set_num_threads(1)
    return torch, ms_ssim


def quality(reference: np.ndarray, reconstruction: np.ndarray) -> dict:
    """Global float64 RGB PSNR and pinned five-scale RGB MS-SSIM."""
    if (reference.shape != reconstruction.shape or reference.ndim != 3
            or reference.shape[2] != 3 or reference.dtype != np.uint8
            or reconstruction.dtype != np.uint8):
        raise ValueError("The reference protocol requires equally sized RGB8 arrays.")
    if min(reference.shape[:2]) <= 160:
        raise ValueError("Five-scale reference MS-SSIM requires dimensions greater than 160.")
    torch, ms_ssim = _reference_backend()
    a, b = (torch.from_numpy(x.copy()).float().unsqueeze(0).permute(0, 3, 1, 2)
            for x in (reference, reconstruction))
    with torch.no_grad():
        value = float(ms_ssim(a, b, data_range=255.0).item())
    return {"psnr_rgb_db": psnr(reference, reconstruction), "msssim_rgb": value}
