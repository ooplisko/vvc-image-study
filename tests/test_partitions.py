from __future__ import annotations

import json

import pytest

from vvcstudy.partitions import parse_trace, summarize_partitions


def test_summarize_partitions_handles_rectangular_cus_and_padding() -> None:
    rows = [
        {"x": "0", "y": "0", "width": "4", "height": "4"},
        {"x": "4", "y": "0", "width": "4", "height": "2"},
        {"x": "4", "y": "2", "width": "2", "height": "2"},
        {"x": "6", "y": "2", "width": "2", "height": "2"},
    ]

    summary = summarize_partitions(rows, source_width=5, source_height=4)

    assert summary["cu_count"] == 4
    assert summary["source_area"] == 20
    assert summary["coded_width"] == 8
    assert summary["coded_height"] == 4
    assert summary["coded_area"] == 32
    assert summary["padding_area"] == 12
    assert summary["cu_density_per_mpixel"] == pytest.approx(125_000.0)
    assert summary["min_area"] == 4
    assert summary["max_area"] == 16
    assert summary["mean_area"] == pytest.approx(8.0)
    assert summary["median_area"] == pytest.approx(6.0)
    assert summary["geometric_mean_area"] == pytest.approx(6.727171322)
    assert summary["q1_area"] == pytest.approx(4.0)
    assert summary["q3_area"] == pytest.approx(10.0)
    assert summary["p10_area"] == pytest.approx(4.0)
    assert summary["p90_area"] == pytest.approx(13.6)
    assert summary["mean_equivalent_side"] == pytest.approx(2.707106781)
    assert summary["share_area_le_64"] == 1.0
    assert summary["share_area_le_256"] == 1.0
    assert summary["size_histogram_json"] == '{"2x2":2,"4x2":1,"4x4":1}'
    assert json.loads(str(summary["size_histogram_json"])) == {"2x2": 2, "4x2": 1, "4x4": 1}
    assert summary["dominant_sizes"] == "2x2:2; 4x2:1; 4x4:1"


def test_summarize_partitions_supports_500_pixel_source_with_504_pixel_coded_width() -> None:
    rows = [
        {"x": 0, "y": 0, "width": 500, "height": 480},
        {"x": 500, "y": 0, "width": 4, "height": 480},
    ]

    summary = summarize_partitions(rows, source_width=500, source_height=480)

    assert summary["coded_width"] == 504
    assert summary["coded_height"] == 480
    assert summary["padding_area"] == 4 * 480


def test_summarize_partitions_rejects_gap() -> None:
    rows = [
        {"x": 0, "y": 0, "width": 2, "height": 2},
        {"x": 2, "y": 0, "width": 2, "height": 1},
    ]

    with pytest.raises(ValueError, match="completely cover"):
        summarize_partitions(rows, source_width=4, source_height=2)


def test_summarize_partitions_rejects_overlap_even_when_total_area_matches_extent() -> None:
    rows = [
        {"x": 0, "y": 0, "width": 3, "height": 2},
        {"x": 2, "y": 0, "width": 2, "height": 1},
    ]

    with pytest.raises(ValueError, match="overlap"):
        summarize_partitions(rows, source_width=4, source_height=2)


@pytest.mark.parametrize(
    ("rows", "source_width", "source_height", "message"),
    [
        ([], 4, 4, "at least one"),
        ([{"x": 0, "y": 0, "width": 0, "height": 4}], 4, 4, "dimensions must be positive"),
        ([{"x": -1, "y": 0, "width": 4, "height": 4}], 4, 4, "coordinates must be non-negative"),
        ([{"x": 0, "y": 0, "width": 4, "height": 4}], 5, 4, "exceed"),
    ],
)
def test_summarize_partitions_rejects_invalid_input(
    rows: list[dict[str, int]], source_width: int, source_height: int, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        summarize_partitions(rows, source_width, source_height)


def test_trace_parser_uses_luma_cus_and_preserves_rectangular_geometry(tmp_path):
    trace = tmp_path / "trace.txt"
    trace.write_text("header\nchroma x=0,y=0,w=4,h=4,qp=22\n"
                     "x=0, y=0, w=8, h=4, qp=27\n"
                     " x=8, y=0, w=8, h=4, qp=-2\n", encoding="utf-8")
    rows = parse_trace(trace, frame=0, mode="baseline")
    assert rows == [
        dict(poc=0, type="cu", x=0, y=0, width=8, height=4, depth=4, mode="baseline", qp=27),
        dict(poc=0, type="cu", x=8, y=0, width=8, height=4, depth=4, mode="baseline", qp=-2),
    ]
    assert summarize_partitions(rows, 16, 4)["cu_count"] == 2
