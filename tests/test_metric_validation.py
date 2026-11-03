import numpy as np
import pytest

from vvcstudy import color, encoding, metrics
from vvcstudy.artifacts import read_json
from vvcstudy.metric_validation import compare_pair, reference_quality, synthetic_pairs


def test_all_active_rgb_encoding_paths_share_the_quality_function():
    assert encoding.quality is metrics.quality
    assert color.quality is metrics.quality


@pytest.mark.parametrize("case", ["identity_192", "perturbed_193", "boundary_192", "channel_193"])
def test_reference_protocol_agrees_on_identical_rgb_arrays(case):
    pytest.importorskip("torch")
    pytest.importorskip("pytorch_msssim")
    name, a, b = next(p for p in synthetic_pairs() if p[0] == case)
    row = compare_pair(name, a, b, {"psnr_absolute_tolerance_db": 1e-5,
                                   "msssim_absolute_tolerance": 1e-5})
    assert row["psnr_agrees"] and row["msssim_agrees"]
    assert metrics.quality(a, b)["psnr_rgb_db"] == metrics.psnr(a, b)


def test_protocol_selection_is_explicit_and_does_not_fall_back():
    a = np.zeros((8, 8, 3), dtype=np.uint8)
    with pytest.raises(ValueError, match="greater than 160"):
        metrics.quality(a, a)


def test_reference_rejects_ambiguous_input_domain():
    pytest.importorskip("torch")
    pytest.importorskip("pytorch_msssim")
    a = np.zeros((192, 192, 3), dtype=np.float64)
    with pytest.raises(ValueError, match="RGB8"):
        reference_quality(a, a)
    a = np.zeros((64, 64, 3), dtype=np.uint8)
    with pytest.raises(ValueError, match="greater than 160"):
        reference_quality(a, a)


def test_validation_manifest_pins_author_config():
    from pathlib import Path
    from vvcstudy.artifacts import sha256
    root = Path(__file__).resolve().parents[1]
    manifest = read_json(root / "manifests/validation.json")
    item = manifest["lossy_vae"]
    assert sha256(root / item["configuration"]) == item["configuration_sha256"]
    assert item["decoder_output_bits"] == 8
