from io import BytesIO

import numpy as np
from PIL import Image, ImageFile

from vvcstudy.encoding import decode_jpeg, encode_jpeg, encoder_command, prepare_yuv


def test_explicit_buffer_preserves_jpeg_bitstream_and_tables():
    rgb = np.random.default_rng(32).integers(0, 256, (64, 64, 3), dtype=np.uint8)
    ImageFile.MAXBLOCK = 32 * 1024 * 1024
    old = BytesIO()
    Image.fromarray(rgb).save(old, "JPEG", quality=83, subsampling=0, optimize=True, progressive=False)
    ImageFile.MAXBLOCK = 65536
    new = encode_jpeg(rgb, 83)
    assert new == old.getvalue()
    decoded, tables = decode_jpeg(new)
    assert decoded.shape == rgb.shape and set(tables) == {0, 1}
    assert all(len(table) == 64 for table in tables.values())


def test_vtm_padding_and_timing_commands_have_different_output_scope():
    spec = {"codec": "vtm23-baseline-trace", "mode": "baseline", "qp": 32,
            "width": 97, "height": 99, "trace": True, "padding": True}
    normal = encoder_command(spec, "encoder.exe")
    timed = encoder_command(spec, "encoder.exe", timing=True)
    assert "--ConformanceWindowMode=1" in normal
    assert "--TraceRule=D_QP:poc==0" in normal and "reconstruction.yuv" in normal
    assert not any("Trace" in arg for arg in timed)
    assert "reconstruction.yuv" not in timed


def test_opencv_conversion_is_planar_not_interleaved(tmp_path):
    import cv2
    rgb = np.array([[[0, 255, 0], [255, 0, 0]], [[0, 0, 255], [127, 127, 127]]], dtype=np.uint8)
    path = tmp_path / "input.yuv"
    prepare_yuv(rgb, path, "opencv_444")
    actual = np.frombuffer(path.read_bytes(), dtype=np.uint8).reshape(3, 2, 2)
    expected = cv2.cvtColor(rgb, cv2.COLOR_RGB2YUV)
    assert np.array_equal(actual, expected.transpose(2, 0, 1))
