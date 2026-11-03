"""Compare RGB metrics on identical pixels with pinned independent implementations."""

import hashlib
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import cv2
import numpy as np

from .artifacts import read_json, sha256, write_csv, write_json
from .encoding import decode_jpeg, encode_jpeg, read_rgb
from .metrics import CURRENT_RGB_PROTOCOL, quality


def reference_quality(reference: np.ndarray, reconstruction: np.ndarray) -> dict:
    """CompressAI's float32 torch PSNR and pytorch-msssim call, on RGB8 arrays."""
    try:
        import torch
        from pytorch_msssim import ms_ssim
    except ImportError as exc:
        raise RuntimeError("Install the pinned dependencies: uv sync --locked") from exc
    if reference.shape != reconstruction.shape or reference.ndim != 3 or reference.shape[2] != 3:
        raise ValueError("Expected equally sized RGB images.")
    if reference.dtype != np.uint8 or reconstruction.dtype != np.uint8:
        raise ValueError("Metric comparison requires RGB8 arrays.")
    if min(reference.shape[:2]) <= 160:
        raise ValueError("The reference MS-SSIM requires both dimensions greater than 160.")
    torch.set_num_threads(1)
    a, b = (torch.from_numpy(x.copy()).permute(2, 0, 1).unsqueeze(0).float()
            for x in (reference, reconstruction))
    with torch.no_grad():
        mse = torch.mean((a - b) ** 2).item()
        psnr = float(20 * np.log10(255.0) - 10 * np.log10(mse)) if mse else float("inf")
        score = float(ms_ssim(a, b, data_range=255.0).item())
    # Duan's evaluation uses normalized float64 arrays and a global RGB MSE.
    error = reconstruction.astype(np.float64) / 255 - reference.astype(np.float64) / 255
    mse64 = float(np.mean(error ** 2))
    return {"psnr_rgb_db": psnr, "msssim_rgb": score,
            "duan_psnr_rgb_db": float(-10 * np.log10(mse64)) if mse64 else float("inf")}


def synthetic_pairs():
    """Fixed fixtures exercise boundaries, odd sizes, channels and negative covariance."""
    for height, width in ((192, 208), (193, 211)):
        y, x = np.indices((height, width))
        a = np.stack(((3*x + y) % 256, (x + 5*y) % 256,
                      (x*y + 17) % 256), axis=-1).astype(np.uint8)
        yield f"identity_{height}", a, a.copy()
        b = np.clip(a.astype(np.int16) + ((x % 9) - 4)[..., None], 0, 255).astype(np.uint8)
        yield f"perturbed_{height}", a, b
        b = a.copy()
        b[:10] = 255 - b[:10]
        yield f"boundary_{height}", a, b
        b = a.copy()
        b[:, :, 1] = 255 - b[:, :, 1]
        yield f"channel_{height}", a, b


def _difference(a, b):
    return 0.0 if a == b else float(a - b)


def compare_pair(name, reference, reconstruction, tolerances):
    project = quality(reference, reconstruction)
    external = reference_quality(reference, reconstruction)
    return {"case": name, "height": reference.shape[0], "width": reference.shape[1],
            "reference_rgb_sha256": hashlib.sha256(reference.tobytes()).hexdigest(),
            "reconstruction_rgb_sha256": hashlib.sha256(reconstruction.tobytes()).hexdigest(),
            "project_psnr_rgb_db": project["psnr_rgb_db"],
            "compressai_psnr_rgb_db": external["psnr_rgb_db"],
            "duan_psnr_rgb_db": external["duan_psnr_rgb_db"],
            "psnr_gap_db": _difference(project["psnr_rgb_db"], external["psnr_rgb_db"]),
            "duan_psnr_gap_db": _difference(project["psnr_rgb_db"], external["duan_psnr_rgb_db"]),
            "psnr_agrees": abs(_difference(project["psnr_rgb_db"], external["psnr_rgb_db"])) <= tolerances["psnr_absolute_tolerance_db"]
                           and abs(_difference(project["psnr_rgb_db"], external["duan_psnr_rgb_db"])) <= tolerances["psnr_absolute_tolerance_db"],
            "project_msssim_rgb": project["msssim_rgb"],
            "reference_msssim_rgb": external["msssim_rgb"],
            "msssim_gap": project["msssim_rgb"] - external["msssim_rgb"],
            "msssim_agrees": abs(project["msssim_rgb"] - external["msssim_rgb"]) <= tolerances["msssim_absolute_tolerance"]}


def check_metrics(root: Path, *, kodak: bool = False) -> dict:
    manifest = read_json(root / "manifests/validation.json")
    tolerance = manifest["metric_reference"]
    for package in ("torch", "pytorch-msssim"):
        try:
            actual = version(package)
        except PackageNotFoundError as exc:
            raise RuntimeError("Install the pinned dependencies: uv sync --locked") from exc
        if actual.split("+")[0] != tolerance[package.replace("-", "_")]:
            raise ValueError(f"Unexpected {package} version: {actual}")
    rows = [compare_pair(name, a, b, tolerance) for name, a, b in synthetic_pairs()]
    input_hashes = {}
    if kodak:
        from .datasets import prepare_datasets
        for item in prepare_datasets(root, ["kodak"])["kodak"]:
            a = read_rgb(Path(item["path"]))
            input_hashes[item["id"]] = item["sha256"]
            for qf in (30, 80):
                b, _ = decode_jpeg(encode_jpeg(a, qf))
                rows.append(compare_pair(f"{item['id']}_jpeg_qf{qf}", a, b, tolerance))
    output = root / "results/validation/metrics" / ("kodak" if kodak else "synthetic")
    write_csv(output / "comparison.csv", rows)
    agreement = {"psnr_rgb": all(r["psnr_agrees"] for r in rows),
                 "msssim_rgb": all(r["msssim_agrees"] for r in rows)}
    groups = {}
    for name, selected in (("synthetic", [r for r in rows if not r["case"].startswith("kodim")]),
                           ("kodak_jpeg_qf30_qf80", [r for r in rows if r["case"].startswith("kodim")])):
        if selected:
            groups[name] = {"cases": len(selected),
                            "max_abs_msssim_gap": max(abs(r["msssim_gap"]) for r in selected)}
    result = {"accepted": all(agreement.values()), "checks_executed": True,
              "cases": len(rows), "agreement": agreement,
              "max_abs_psnr_gap_db": max(abs(r["psnr_gap_db"]) for r in rows),
              "max_abs_msssim_gap": max(abs(r["msssim_gap"]) for r in rows),
              "groups": groups,
              "scope": tolerance["scope"], "output": str(output),
              "protocol": CURRENT_RGB_PROTOCOL,
              "interpretation": "Agreement on identical RGB arrays does not establish reproduction of a codec or its color conversion."}
    write_json(output / "context.json", {
        **result, "tolerances": tolerance,
        "sources": manifest["sources"], "numpy": np.__version__, "opencv": cv2.__version__,
        "torch": version("torch"), "pytorch_msssim": version("pytorch-msssim"),
        "input_png_hashes": input_hashes,
        "implementation_sha256": sha256(Path(__file__)),
        "metrics_sha256": sha256(Path(__file__).with_name("metrics.py")),
        "table_sha256": sha256(output / "comparison.csv")})
    return result
