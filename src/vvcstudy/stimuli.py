"""Deterministic image stimuli; three-channel arrays use OpenCV BGR order."""

from __future__ import annotations

import hashlib
import math
import re
from pathlib import Path

import cv2
import numpy as np


SINUSOIDAL_PERIOD_PIXELS = 16.0


def read_png(path: Path) -> np.ndarray:
    image = cv2.imdecode(np.frombuffer(path.read_bytes(), dtype=np.uint8), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise RuntimeError(f"Could not read PNG image: {path}")
    if image.dtype != np.uint8:
        raise ValueError(f"Only 8-bit PNG images are supported: {path}")
    if image.ndim not in (2, 3) or (image.ndim == 3 and image.shape[2] not in (3, 4)):
        raise ValueError(f"Unsupported PNG channel layout: {path}")
    return image


def write_png(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ok, encoded = cv2.imencode(".png", image, [cv2.IMWRITE_PNG_COMPRESSION, 3])
    if not ok:
        raise RuntimeError(f"Could not write PNG image: {path}")
    path.write_bytes(encoded.tobytes())


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def complexity_metrics(image: np.ndarray) -> dict[str, float]:
    """Return the pre-registered Sobel spatial-information measure on 8-bit luma."""

    luma = _to_luma(image)
    luma_float = luma.astype(np.float64)
    sobel_x = cv2.Sobel(luma_float, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(luma_float, cv2.CV_64F, 0, 1, ksize=3)
    sobel_magnitude = np.hypot(sobel_x, sobel_y)
    interior = sobel_magnitude[1:-1, 1:-1]
    if interior.size == 0:
        raise ValueError("images must be at least 3x3 pixels for Sobel SI")
    return {"sobel_si": float(np.std(interior))}


def add_achromatic_awgn(image: np.ndarray, sigma: float, seed: int) -> np.ndarray:
    if sigma < 0:
        raise ValueError("AWGN sigma must be non-negative")
    rng = np.random.Generator(np.random.PCG64(seed))
    delta = rng.normal(0.0, sigma, image.shape[:2])
    return _add_achromatic_delta(image, delta)


def add_sinusoidal_interference(image: np.ndarray, amplitude: float) -> np.ndarray:
    """Add deterministic horizontal sinusoidal bands with a fixed 16-pixel period."""

    if amplitude < 0:
        raise ValueError("Sinusoidal amplitude must be non-negative")
    y = np.arange(image.shape[0], dtype=np.float64)[:, None]
    delta = amplitude * np.sin(2.0 * math.pi * y / SINUSOIDAL_PERIOD_PIXELS)
    return _add_achromatic_delta(image, np.broadcast_to(delta, image.shape[:2]))


def derive_image_seed(base_seed: int, source_index: int) -> int:
    """Derive a reproducible image-specific PCG64 seed from a registered base seed."""

    if source_index < 0:
        raise ValueError("source_index must be non-negative")
    state = np.random.SeedSequence([base_seed, source_index]).generate_state(1, dtype=np.uint64)
    return int(state[0])


def luma_rms_difference(reference: np.ndarray, distorted: np.ndarray) -> float:
    if reference.shape != distorted.shape:
        raise ValueError("reference and distorted images must have identical shapes")
    difference = _to_luma(distorted).astype(np.float64) - _to_luma(reference).astype(np.float64)
    return float(np.sqrt(np.mean(difference * difference)))


def stable_stimulus_name(source: Path, source_sha256: str, distortion: str, level: float, seed: int | None) -> str:
    source_name = re.sub(r"[^A-Za-z0-9_-]+", "-", source.stem).strip("-") or "image"
    level_token = _number_token(level)
    suffix = f"{distortion}_{level_token}"
    if seed is not None:
        suffix += f"_seed-{seed}"
    return f"{source_name}_{source_sha256[:12]}__{suffix}.png"


def _to_luma(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        return image
    rgb = cv2.cvtColor(image[:, :, :3], cv2.COLOR_BGR2RGB)
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2YUV)[:, :, 0]


def _add_achromatic_delta(image: np.ndarray, delta: np.ndarray) -> np.ndarray:
    result = image.copy()
    if image.ndim == 2:
        result[:] = np.clip(np.rint(image.astype(np.float64) + delta), 0, 255).astype(np.uint8)
        return result

    channels = image[:, :, :3].astype(np.float64) + delta[:, :, None]
    result[:, :, :3] = np.clip(np.rint(channels), 0, 255).astype(np.uint8)
    return result


def _number_token(value: float) -> str:
    text = format(float(value), ".12g")
    return text.replace("-", "m").replace(".", "p").replace("+", "")
