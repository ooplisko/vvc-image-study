"""Reconstruct RGB using the inverse associated with each encoder input path."""

from pathlib import Path
import shutil
import subprocess

import numpy as np

from .metrics import CURRENT_RGB_PROTOCOL, expected_yuv_size, opencv_yuv_to_rgb8, quality


RGB_PROTOCOL = "original_rgb8_v1"


def reconstruction_rgb(path: Path, width: int, height: int, conversion: str,
                       *, bit_depth: int = 10, ffmpeg: str = "ffmpeg") -> np.ndarray:
    """Read one visible YUV frame and return RGB8; never infer its color range."""
    if bit_depth not in (8, 10) or conversion not in {"opencv_444", "ffmpeg_444", "ffmpeg_420"}:
        raise ValueError("Unsupported RGB reconstruction protocol.")
    chroma = conversion[-3:]
    if chroma == "420" and (width % 2 or height % 2):
        raise ValueError("4:2:0 reconstruction requires even dimensions.")
    if path.stat().st_size != expected_yuv_size(width, height, bit_depth, chroma):
        raise ValueError("Expected exactly one visible YUV frame.")
    samples = np.fromfile(path, dtype=np.uint8 if bit_depth == 8 else "<u2")
    if samples.max() >= 1 << bit_depth:
        raise ValueError("YUV sample exceeds the declared bit depth.")
    if conversion == "opencv_444":
        return opencv_yuv_to_rgb8(samples.reshape(3, height, width), bit_depth)
    pixel_format = "yuv" + chroma + ("p" if bit_depth == 8 else "p10le")
    # FFmpeg PNG -> yuv444p/yuv420p uses limited-range BT.601. Specify the
    # inverse, scaling flags and RGB output explicitly, including 4:2:0 upsampling.
    command = [ffmpeg, "-v", "error", "-f", "rawvideo", "-pixel_format", pixel_format,
               "-video_size", f"{width}x{height}", "-i", str(path.resolve()),
               "-vf", "scale=in_range=tv:out_range=pc:in_color_matrix=bt601:flags=bicubic+accurate_rnd+full_chroma_int+bitexact",
               "-frames:v", "1", "-threads", "1", "-pix_fmt", "rgb24", "-f", "rawvideo", "pipe:1"]
    result = subprocess.run(command, capture_output=True, check=True)
    if len(result.stdout) != width * height * 3:
        raise ValueError("FFmpeg returned an incomplete RGB frame.")
    return np.frombuffer(result.stdout, dtype=np.uint8).reshape(height, width, 3)


def refresh_matrix_quality(root: Path, profile: str = "full") -> dict:
    """Recalculate matrix RGB quality from retained frames, without encoding.

    Validate all inputs first. Replace the compact table only after every point
    succeeds. The caller explicitly seals the changed result stage afterwards.
    """
    from .artifacts import read_csv, read_json, sha256, write_csv, write_json
    from .encoding import read_rgb
    from .results import stage_rows

    base = root / "results" / profile
    path = base / "tables/encodings.csv"
    rows = read_csv(path)
    selected = stage_rows(rows, "matrices")
    if not selected:
        raise ValueError("No matrix measurements available.")
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise ValueError("FFmpeg is required to reconstruct matrix RGB images.")
    originals = {r["id"] + ".png": r for r in read_json(root / "manifests/datasets/kodak.json")["images"]}
    bundle = root / "runs" / profile / "artifacts"
    retained = ({(bundle / r["path"]).resolve(): r for r in read_json(bundle / "manifest.json")["files"]}
                if (bundle / "manifest.json").exists() else {})
    inputs, checksums = [], {}

    def check(file: Path, expected: str | None = None) -> str:
        file = file.resolve()
        if not file.is_relative_to(root.resolve()):
            raise ValueError("Measurement file lies outside the repository.")
        if file not in checksums:
            checksums[file] = sha256(file)
        actual = checksums[file]
        registered = retained.get(file)
        if (expected and expected != actual) or (registered and registered["sha256"] != actual):
            raise ValueError(f"Measurement file checksum differs: {file}")
        return actual

    for row in selected:
        if not row.get("stream"):
            raise ValueError("Raw stream paths are needed; compact tables alone cannot be remeasured.")
        stream = root / row["stream"]
        folder = stream.parent
        source = root / "data/originals/kodak" / row["source"]
        source_hash = check(source, originals[row["source"]]["sha256"])
        stream_hash = check(stream, row.get("stream_sha256"))
        if stream.stat().st_size != int(row["bitstream_bytes"]):
            raise ValueError("Bitstream size differs from the measured table.")
        decoded = folder / "decoded.yuv"
        decoded_hash = check(decoded, row.get("decoded_sha256"))
        if check(folder / "reconstruction.yuv", row.get("reconstruction_sha256")) != decoded_hash:
            raise ValueError("Decoder output differs from encoder reconstruction.")
        inputs.append((row, source, decoded, source_hash, stream_hash, decoded_hash))

    records = []
    for row, source, decoded, source_hash, stream_hash, decoded_hash in inputs:
        rgb = read_rgb(source)
        height, width = rgb.shape[:2]
        if (width, height) != (int(row["width"]), int(row["height"])):
            raise ValueError("Original dimensions differ from the measured table.")
        measured = quality(rgb, reconstruction_rgb(decoded, width, height, row["conversion"], ffmpeg=ffmpeg))
        # Preserve already registered main-series rows byte-for-byte. Check their
        # RGB metrics rather than inserting matrix-only metadata into them.
        if row["conversion"] == "opencv_444" and row["mode"] == "baseline":
            for name, value in measured.items():
                if not np.isclose(float(row[name]), value, atol=1e-12, rtol=0):
                    raise ValueError(f"Main-series RGB metric changed: {row['source']} {name}")
        else:
            row.update(measured, rgb_protocol=RGB_PROTOCOL, input_sha256=source_hash,
                       stream_sha256=stream_hash, decoded_sha256=decoded_hash,
                       reconstruction_sha256=decoded_hash, reconstruction_verified=True)
        row["quality_protocol"] = CURRENT_RGB_PROTOCOL
        records.append({key: row[key] for key in ("source", "qp", "codec", "conversion", "mode")} | {
            "quality_protocol": CURRENT_RGB_PROTOCOL,
            "rgb_protocol": RGB_PROTOCOL, "input_sha256": source_hash,
            "stream_sha256": stream_hash, "decoded_sha256": decoded_hash, **measured})
    write_csv(path, rows)
    write_csv(base / "tables/matrix_rgb_checks.csv", records)
    write_json(base / "tables/matrix_rgb_context.json", {
        "rgb_protocol": RGB_PROTOCOL, "quality_protocol": CURRENT_RGB_PROTOCOL,
        "points": len(records), "reencoded_points": 0,
        "reference": "original Kodak RGB8 PNG", "reconstruction_bit_depth": 10,
        "ffmpeg_inverse": "limited BT.601 to RGB24; bicubic+accurate_rnd+full_chroma_int+bitexact",
        "ffmpeg_sha256": sha256(Path(ffmpeg)),
        "ffmpeg_version": subprocess.run([ffmpeg, "-version"], capture_output=True,
                                          text=True, check=True).stdout.splitlines()[0],
        "color_code_sha256": sha256(Path(__file__)),
        "metrics_code_sha256": sha256(Path(__file__).with_name("metrics.py"))})
    return {"status": "complete", "points": len(records), "reencoded_points": 0}
