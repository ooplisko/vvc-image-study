import pytest

from vvcstudy.anchors import anchor_points, compare_points


def test_unknown_anchor_qps_are_not_invented():
    points = anchor_points({"sources": [{"id": "reference", "url": "https://example.org",
                                        "results": {"bpp": [1, 2], "psnr-rgb": [30, 35]}}]})
    assert [p["qp"] for p in points] == ["", ""]


def test_nearest_anchor_preserves_both_gaps_and_excludes_extrapolation():
    published = [{"anchor": "a", "point": 1, "qp": "", "bpp": 1.0, "psnr_rgb_db": 30.0},
                 {"anchor": "a", "point": 2, "qp": "", "bpp": 2.0, "psnr_rgb_db": 35.0}]
    local = [{"bpp": b, "psnr_rgb_db": 33.0} for b in (0.5, 1.2, 3.0)]
    rows = compare_points(local, published)
    assert rows[0]["status"] == rows[2]["status"] == "outside_published_bpp_range"
    assert "psnr_gap_db" not in rows[0]
    assert rows[1]["bpp_gap"] == pytest.approx(0.2)
    assert rows[1]["psnr_gap_db"] == 3.0


def test_anchor_rejects_mismatched_vectors():
    with pytest.raises(ValueError, match="grid"):
        anchor_points({"sources": [{"results": {"bpp": [1, 2], "psnr": [30]}}]})


def test_only_published_msssim_values_are_exported():
    manifest = {"sources": [
        {"id": "psnr_only", "url": "https://example.org/one",
         "results": {"bpp": [1], "psnr": [30]}},
        {"id": "both", "url": "https://example.org/two",
         "results": {"bpp": [2], "psnr-rgb": [35], "ms-ssim-rgb": [0.975]}}]}
    points = anchor_points(manifest)
    assert "msssim_rgb" not in points[0]
    assert points[1]["msssim_rgb"] == 0.975
    manifest["sources"][1]["results"]["ms-ssim-rgb"] = [0.975, 0.98]
    with pytest.raises(ValueError, match="MS-SSIM"):
        anchor_points(manifest)
