"""Statistics and validation for final luma CU partitions."""

from __future__ import annotations

import json
import math
import re
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any


_INTEGER_RE = re.compile(r"^-?\d+$")


def _integer(value: object, name: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be an integer")
    if isinstance(value, int):
        return value
    if isinstance(value, str) and _INTEGER_RE.fullmatch(value.strip()):
        return int(value)
    raise ValueError(f"{name} must be an integer")


def _percentile(sorted_values: Sequence[int], fraction: float) -> float:
    position = (len(sorted_values) - 1) * fraction
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return float(sorted_values[lower])
    weight = position - lower
    return sorted_values[lower] * (1.0 - weight) + sorted_values[upper] * weight


def _validate_coverage(rectangles: Sequence[tuple[int, int, int, int]], coded_area: int) -> None:
    events: list[tuple[int, int, int, int, int]] = []
    total_area = 0
    for index, (x, y, width, height) in enumerate(rectangles):
        events.append((x, 1, index, y, y + height))
        events.append((x + width, 0, index, y, y + height))
        total_area += width * height

    active: dict[int, tuple[int, int]] = {}
    for _, event_type, index, y0, y1 in sorted(events):
        if event_type == 0:
            active.pop(index)
            continue
        for active_y0, active_y1 in active.values():
            if max(y0, active_y0) < min(y1, active_y1):
                raise ValueError("CU rectangles overlap in the coded plane")
        active[index] = (y0, y1)

    if total_area != coded_area:
        raise ValueError("CU rectangles do not completely cover the coded plane")


def summarize_partitions(
    rows: Sequence[Mapping[str, Any]],
    source_width: int,
    source_height: int,
    dominant_limit: int = 6,
) -> dict[str, int | float | str]:
    """Validate parsed ``D_QP`` rows and summarize the final luma CU partition.

    The coded extent is inferred from the largest CU right and bottom edges. Area
    statistics are count-weighted over CUs; shares are fractions in the range
    ``[0, 1]``. Percentiles use linear interpolation between adjacent ranks.
    """

    source_width = _integer(source_width, "source_width")
    source_height = _integer(source_height, "source_height")
    dominant_limit = _integer(dominant_limit, "dominant_limit")
    if source_width <= 0 or source_height <= 0:
        raise ValueError("source dimensions must be positive")
    if dominant_limit <= 0:
        raise ValueError("dominant_limit must be positive")
    if not rows:
        raise ValueError("at least one CU row is required")

    rectangles: list[tuple[int, int, int, int]] = []
    for index, row in enumerate(rows):
        try:
            x = _integer(row["x"], f"rows[{index}].x")
            y = _integer(row["y"], f"rows[{index}].y")
            width = _integer(row["width"], f"rows[{index}].width")
            height = _integer(row["height"], f"rows[{index}].height")
        except KeyError as error:
            raise ValueError(f"rows[{index}] is missing {error.args[0]!r}") from error
        if x < 0 or y < 0:
            raise ValueError("CU coordinates must be non-negative")
        if width <= 0 or height <= 0:
            raise ValueError("CU dimensions must be positive")
        rectangles.append((x, y, width, height))

    coded_width = max(x + width for x, _, width, _ in rectangles)
    coded_height = max(y + height for _, y, _, height in rectangles)
    if source_width > coded_width or source_height > coded_height:
        raise ValueError("source dimensions exceed the inferred coded extent")

    source_area = source_width * source_height
    coded_area = coded_width * coded_height
    _validate_coverage(rectangles, coded_area)

    areas = sorted(width * height for _, _, width, height in rectangles)
    count = len(areas)
    size_counts = Counter((width, height) for _, _, width, height in rectangles)
    histogram = {f"{width}x{height}": size_counts[(width, height)] for width, height in sorted(size_counts)}
    dominant = sorted(size_counts.items(), key=lambda item: (-item[1], item[0]))[:dominant_limit]

    return {
        "cu_count": count,
        "source_area": source_area,
        "coded_width": coded_width,
        "coded_height": coded_height,
        "coded_area": coded_area,
        "padding_area": coded_area - source_area,
        "cu_density_per_mpixel": count * 1_000_000.0 / coded_area,
        "min_area": areas[0],
        "max_area": areas[-1],
        "mean_area": sum(areas) / count,
        "median_area": _percentile(areas, 0.5),
        "geometric_mean_area": math.exp(sum(math.log(area) for area in areas) / count),
        "q1_area": _percentile(areas, 0.25),
        "q3_area": _percentile(areas, 0.75),
        "p10_area": _percentile(areas, 0.10),
        "p90_area": _percentile(areas, 0.90),
        "mean_equivalent_side": sum(math.sqrt(area) for area in areas) / count,
        "share_area_le_64": sum(area <= 64 for area in areas) / count,
        "share_area_le_256": sum(area <= 256 for area in areas) / count,
        "size_histogram_json": json.dumps(histogram, separators=(",", ":"), ensure_ascii=True),
        "dominant_sizes": "; ".join(f"{width}x{height}:{value}" for (width, height), value in dominant),
    }


LUMA_CU_RE = re.compile(
    r"^x=(?P<x>\d+),\s*y=(?P<y>\d+),\s*w=(?P<width>\d+),\s*h=(?P<height>\d+),\s*qp=(?P<qp>-?\d+)"
)


def parse_trace(trace_path: Path, frame: int = 0, mode: str = "unknown") -> list[dict[str, int | str]]:
    rows: list[dict[str, int | str]] = []
    with trace_path.open("r", encoding="utf-8", errors="replace") as stream:
        for line in stream:
            match = LUMA_CU_RE.match(line.strip())
            if not match:
                continue

            x = int(match.group("x"))
            y = int(match.group("y"))
            width = int(match.group("width"))
            height = int(match.group("height"))
            rows.append(
                {
                    "poc": frame,
                    "type": "cu",
                    "x": x,
                    "y": y,
                    "width": width,
                    "height": height,
                    "depth": _depth(width, height),
                    "mode": mode,
                    "qp": int(match.group("qp")),
                }
            )
    return rows


def _depth(width: int, height: int) -> int:
    max_side = max(width, height)
    if max_side >= 128:
        return 0
    if max_side >= 64:
        return 1
    if max_side >= 32:
        return 2
    if max_side >= 16:
        return 3
    if max_side >= 8:
        return 4
    return 5
