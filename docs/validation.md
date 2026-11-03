# Metric checks and VTM reproduction

Two independent checks answer different questions. Metric checks use identical
RGB8 arrays to isolate the calculation. VTM reproduction also includes color
conversion, encoding and decoding. A completed run is not, by itself, numerical
agreement with a publication.

## RGB metric implementation

The metric dependencies are installed by `uv sync --locked`. Run:

```powershell
uv run --locked vvc-study check-metrics
uv run --locked vvc-study check-metrics --kodak
```

All main studies share `metrics.quality()` with protocol `pytorch_rgb_v1`.
PSNR uses global RGB mean squared error in float64. MS-SSIM calls
`pytorch-msssim==1.0.0` with float32 RGB8 values and `data_range=255`;
PyTorch 2.8.0 runs on CPU with one thread. JPEG, VTM/VVenC and matrix
remeasurement use this same entry point. The reference is the actual input
RGB8, including noise where present. PSNR-Y is a separate measurement.

The check compares these outputs with the pinned CompressAI evaluation calls
and, for PSNR, Duan's normalized float64 calculation. Source revisions and
absolute tolerances are in `manifests/validation.json`: 0.00001 dB for PSNR
and 0.00001 for MS-SSIM. The dependency lock fixes the local versions; it does
not establish the package versions that generated the published CompressAI JSON.

Eight synthetic cases cover even/odd dimensions, identity, boundary distortion
and distortion confined to one channel. `--kodak` adds actual JPEG files at
QF30 and QF80 for all 24 Kodak images. All 56 cases pass. Maximum absolute
differences are 7.34e-7 dB for PSNR and 1.20e-7 for MS-SSIM. These 48 JPEG
pairs test the implementation, not the complete JPEG/VVC experiment.

Tables and source/version records are written to
`results/validation/metrics/{synthetic,kodak}/`. The command returns
exit code 1 if either metric fails its tolerance. The public interface exposes
only the shared metric definition used by the main experiments.

Agreement on identical arrays validates the metric calculation. It does not
establish reproduction of a codec, its color conversion or a published RD curve.
The completed VTM18 experiment below provides a separate full-path check for
bpp and PSNR. The CompressAI curves remain a cross-version comparison.

## VTM 18.0 reproduction

This optional study builds the official VTM 18.0 source revision pinned in
`manifests/validation.json`. It requires the same Windows compiler prerequisites
as the other source builds. The `codecs-v1` binary bundle does not
contain VTM18. The compiler force-includes `<cstdint>` for
GCC 13 compatibility; no coding tools or algorithms are patched.

```powershell
uv run --locked vvc-study plan --study lossy-vae
uv run --locked vvc-study run --study lossy-vae --profile smoke --workers 1
uv run --locked vvc-study run --study lossy-vae --workers 4
```

The full grid is 24 Kodak images times four QPs: 96 encodes. The author settings
are used directly: OpenCV RGB-to-YUV 4:4:4, one frame, the supplied intra
configuration and decoder `--OutputBitDepth=8`. Decoded planes are converted by
OpenCV `COLOR_YUV2RGB`; PSNR uses the original RGB8 PNG. This decoder conversion
is deliberately separate from the project's VTM23 reconstruction protocol.

Jobs save commands, input/configuration/program hashes, bitstreams, decoded
planes and per-image measurements. Already completed jobs are reused only after
integrity verification.

Results are written to `results/validation/lossy-vae/full/`. Only a complete
96-point grid produces `same_qp.csv`, `psnr_rgb_db_by_qp.png` and `bpp_by_qp.png`.
Both plotted series use the same QPs. Means are computed per QP over 24 images;
partial means are never compared with a full-dataset reference. Actual signed
differences are reported without interpolation, extrapolation or a promise of
exact agreement. Synthetic checks do not count toward Kodak coverage.

```powershell
# Use a second console while replication is running.
uv run --locked vvc-study stop --study lossy-vae
uv run --locked vvc-study status --study lossy-vae

# After the first process exits, resume with the same run command.
uv run --locked vvc-study run --study lossy-vae --workers 4

# Recreate both QP figures from compact measurements, without encoding.
uv run --locked vvc-study plot --study lossy-vae

# Rebuild measurements and comparisons from retained raw job records.
uv run --locked vvc-study analyze --study lossy-vae
```

The stop command finishes active jobs and prevents dispatching additional ones.
Wait until the running command exits before shutting down. A stop requested
during compilation is honored before the first encoding job. Validation results
are separate from `results/full/`; the existing studies and their result
manifests are not rewritten by these commands.
