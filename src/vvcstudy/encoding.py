"""Explicit codec commands, color conversions and quality measurements."""

from __future__ import annotations

from io import BytesIO
from pathlib import Path
import re
import shutil
import subprocess
import time

import cv2
import numpy as np
from PIL import Image, ImageFile, JpegImagePlugin, features

from .artifacts import file_record, sha256, write_csv
from .metrics import calculate_color_metrics, opencv_yuv_to_rgb8, psnr, quality
from .partitions import parse_trace, summarize_partitions


def jpeg_environment() -> dict:
    return {"pillow": Image.__version__, "libjpeg_turbo": features.version("libjpeg_turbo"),
            "optimize": True, "progressive": False, "subsampling": 0,
            "tables": "Pillow/libjpeg default tables scaled by QF", "maxblock": 32 * 1024 * 1024}


def encode_jpeg(rgb: np.ndarray, qf: int) -> bytes:
    if not 1 <= qf <= 100:
        raise ValueError("JPEG QF must be in 1..100.")
    ImageFile.MAXBLOCK = 32 * 1024 * 1024
    buffer = BytesIO()
    Image.fromarray(rgb).save(buffer, format="JPEG", quality=qf, subsampling=0,
                              optimize=True, progressive=False)
    return buffer.getvalue()


def decode_jpeg(payload: bytes) -> tuple[np.ndarray, dict]:
    with Image.open(BytesIO(payload)) as image:
        if image.format != "JPEG" or JpegImagePlugin.get_sampling(image) != 0:
            raise ValueError("Expected JPEG 4:4:4.")
        if image.info.get("progressive"):
            raise ValueError("Unexpected progressive JPEG.")
        return np.array(image.convert("RGB")), image.quantization


def read_rgb(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        if image.mode != "RGB":
            raise ValueError("Expected unmodified RGB input.")
        return np.array(image)


def execute_command(command: list[str], folder: Path, log: str) -> str:
    with (folder / log).open("wb") as stream:
        result = subprocess.run(command, cwd=folder, stdout=stream, stderr=subprocess.STDOUT)
    text = (folder / log).read_text(encoding="utf-8", errors="replace")
    if result.returncode:
        raise RuntimeError(f"Codec failed ({result.returncode}); see {folder / log}: {text[-1200:]}")
    return text


def prepare_yuv(rgb: np.ndarray, output: Path, conversion: str, *, source: Path | None = None,
                ffmpeg: str = "ffmpeg") -> None:
    if conversion == "opencv_444":
        planes = np.moveaxis(cv2.cvtColor(rgb, cv2.COLOR_RGB2YUV), -1, 0)
        output.write_bytes(planes.tobytes())
    elif conversion in {"ffmpeg_444", "ffmpeg_420"}:
        if source is None:
            raise ValueError("FFmpeg matrix conversion requires its original PNG input.")
        local_source = output.parent / "source.png"
        shutil.copyfile(source, local_source)
        execute_command([ffmpeg, "-y", "-v", "error", "-i", local_source.name,
                         "-pix_fmt", "yuv" + conversion[-3:] + "p", "-frames:v", "1",
                         output.name], output.parent, "conversion.log")
    else:
        raise ValueError(f"Unsupported conversion: {conversion}")


def encoder_command(spec: dict, encoder: str, *, timing: bool = False) -> list[str]:
    width, height, qp = spec["width"], spec["height"], spec["qp"]
    if spec["codec"].startswith("vtm"):
        command = [encoder, "-c", "encoder.cfg", "-i", "input.yuv", "-wdt", str(width),
                   "-hgt", str(height), "-fr", "1", "-f", "1", "-q", str(qp),
                   "-b", "stream.vvc", "--InputChromaFormat=444", "--InputBitDepth=8"]
        if not timing:
            command += ["-o", "reconstruction.yuv"]
        if spec.get("padding"):
            command += ["--ConformanceWindowMode=1"]
        if spec["mode"] == "csf":
            command += ["--CSFScalingList=1"]
        if spec.get("trace") and not timing:
            command += ["--TraceFile=trace.txt", "--TraceRule=D_QP:poc==0"]
    else:
        command = [encoder, "--InputFile", "input.yuv", "--SourceWidth", str(width),
                   "--SourceHeight", str(height), "--FrameRate", "1", "--FramesToBeEncoded", "1",
                   "--QP", str(qp), "--preset", "medium", "--BitstreamFile", "stream.vvc"]
        if not timing:
            command += ["--ReconFile", "reconstruction.yuv"]
        if spec["mode"] == "csf":
            command += ["--CSFScalingList", "1"]
        if spec.get("trace") and not timing:
            command += ["--TraceFile", "trace.txt", "--TraceRule", "D_QP:poc==0"]
    return command


def execute_encode(job: dict, folder: Path) -> dict:
    spec = job["spec"]
    source = Path(job["input"])
    if sha256(source) != spec["input_sha256"]:
        raise ValueError("Input changed after the job was prepared.")
    rgb = read_rgb(source)
    if rgb.shape != (spec["height"], spec["width"], 3):
        raise ValueError("Input dimensions changed.")
    if spec["codec"].startswith("vtm"):
        shutil.copyfile(job["config"], folder / "encoder.cfg")
    prepare_yuv(rgb, folder / "input.yuv", spec["conversion"], source=source,
                ffmpeg=job.get("ffmpeg", "ffmpeg"))
    command = encoder_command(spec, job["encoder"])
    start = time.perf_counter()
    log = execute_command(command, folder, "encoder.log")
    elapsed = time.perf_counter() - start
    execute_command([job["decoder"], "-b", "stream.vvc", "-o", "decoded.yuv"],
                    folder, "decoder.log")
    if sha256(folder / "reconstruction.yuv") != sha256(folder / "decoded.yuv"):
        raise ValueError("Decoder output differs from encoder reconstruction.")
    width, height = spec["width"], spec["height"]
    values = {**{key: spec[key] for key in ("dataset", "source", "distortion", "level", "seed",
                                            "qp", "mode", "codec", "conversion")},
              "width": width, "height": height, "stimulus": source.stem,
              "bitstream_bytes": (folder / "stream.vvc").stat().st_size,
              "file_bpp": (folder / "stream.vvc").stat().st_size * 8 / (width * height),
              "process_elapsed_seconds": elapsed,
              "timing_protocol": "parallel_job_log_not_serial_benchmark", **job.get("features", {})}
    if spec["conversion"] == "opencv_444":
        planes = np.fromfile(folder / "decoded.yuv", dtype="<u2")
        if planes.size != width * height * 3 or planes.max() > 1023:
            raise ValueError("Invalid visible 10-bit 4:4:4 reconstruction.")
        planes = planes.reshape(3, height, width)
        values.update(quality(rgb, opencv_yuv_to_rgb8(planes, 10)))
        ref_y = np.fromfile(folder / "input.yuv", dtype=np.uint8, count=width * height)
        values["psnr_y_db"] = psnr(ref_y.astype(np.float64) * 4, planes[0].ravel(), peak=1020)
    else:
        values.update(calculate_color_metrics(folder / "input.yuv", folder / "decoded.yuv",
                                              width, height, chroma_format=spec["conversion"][-3:]))
        # Matrix series reports the encoder's PSNR-Y alongside YUV-derived RGB metrics.
        match = re.search(r"\bY\s+([\d.]+)\s+dB", log)
        if match:
            values["psnr_y_db"] = float(match.group(1))
        else:
            # Both encoders summarize one intra frame as '1 a bitrate Y U V ...'.
            match = re.search(r"\s1\s+a\s+[\d.]+\s+([\d.]+|inf)", log, re.I)
            if match:
                values["psnr_y_db"] = float(match.group(1))
            else:
                raise ValueError("Could not read matrix experiment PSNR-Y from encoder log.")
    if spec.get("trace"):
        partitions = parse_trace(folder / "trace.txt", frame=0, mode=spec["mode"])
        write_csv(folder / "partitions.csv", partitions)
        values.update(summarize_partitions(partitions, width, height))
    names = ["stream.vvc", "reconstruction.yuv", "decoded.yuv", "encoder.log", "decoder.log", "input.yuv"]
    if spec.get("trace"):
        names += ["trace.txt", "partitions.csv"]
    return {"values": values, "command": command,
            "artifacts": [file_record(folder / name, folder) for name in names]}


def execute_jpeg(job: dict, folder: Path) -> dict:
    from .artifacts import write_json
    spec = job["spec"]
    source = Path(job["input"])
    if sha256(source) != spec["input_sha256"]:
        raise ValueError("JPEG source changed.")
    if jpeg_environment() != spec["implementation"]:
        raise ValueError("JPEG implementation differs from the job specification.")
    rgb = read_rgb(source)
    height, width = rgb.shape[:2]
    rows, artifacts, tables = [], [], {}
    for qf in spec["qfs"]:
        payload = encode_jpeg(rgb, qf)
        reconstructed, qtables = decode_jpeg(payload)
        path = folder / f"QF{qf:03d}.jpg"
        path.write_bytes(payload)
        item = file_record(path, folder)
        artifacts.append(item)
        tables[str(qf)] = qtables
        rows.append({"codec": "JPEG", "parameter": qf, "bytes": len(payload),
                     "bpp": len(payload) * 8 / (width * height),
                     "file": path.name, "sha256": item["sha256"], **quality(rgb, reconstructed)})
    write_csv(folder / "measurements.csv", rows)
    write_json(folder / "quantization_tables.json", tables)
    artifacts += [file_record(folder / name, folder)
                  for name in ("measurements.csv", "quantization_tables.json")]
    return {"values": {key: spec[key] for key in ("dataset", "source", "distortion", "level", "seed")},
            "measurements": rows, "features": job["features"], "artifacts": artifacts}
