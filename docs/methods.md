# Implemented methods

## Inputs and disturbances

Kodak consists of 24 original RGB PNGs. DIV2K consists of 100 full-resolution
validation PNGs (0801–0900). Dataset manifests fix their identities and geometry.
AWGN uses NumPy PCG64, a pixel-wise Gaussian delta shared by the three channels,
round-to-nearest and clipping to [0,255]. Image-specific seeds come from
`SeedSequence([base_seed, source_index])` in fixed manifest order. The three
Kodak base seeds are 20260811, 20260812 and 20260813; DIV2K uses 20260811.
Sinusoidal interference has horizontal bands and a 16-pixel period. Channel
clipping means nominal sigma/amplitude and measured luma RMS are distinct.

The six descriptors are Sobel-magnitude standard deviation, luma standard
deviation, edge-pixel fraction, and GLCM contrast, entropy and homogeneity.
The edge rule uses the registered four kernels and threshold 150, with image
borders excluded. GLCMs use eight levels, symmetric normalized matrices at four
nearest-neighbor orientations, and averaged properties. Kodak sensitivity checks
use 32 levels and the horizontal orientation separately.

## Partition experiments

VTM 23.0 uses the fixed All-Intra configuration, QP 22/27/32/37, one frame,
8-bit 4:4:4 input and 10-bit reconstruction. OpenCV RGB-to-YUV supplies the input.
The `D_QP` trace provides final luma CUs. The decoder output must equal the
encoder reconstruction; CU rectangles must cover the coded extent without gaps
or overlaps. DIV2K conformance padding stays internal to VTM. File bpp uses
original visible pixels; CU density uses coded area, including codec padding.
PSNR-Y compares the actual YUV input with its reconstruction. The supplied
clean Kodak OpenCV series has median PSNR-Y 42.2502, 38.5711, 34.9805 and
31.67125 dB at QP 22/27/32/37. These are medians, not the full observed range
or thresholds of perceptual invisibility. Keep them separate from the FFmpeg
matrix series and from PSNR-RGB used for JPEG/VVC matching.

Kodak analysis preserves paired-image bootstrap and permutation procedures,
the fixed multiple-comparison families, leave-one-out checks and GLCM sensitivity.
DIV2K CU/descriptor analysis uses 99,999 paired source-image bootstrap samples
(PCG64 seed 20260905), reranking observations in each sample. All six descriptors,
four QPs and both conditions share each sampled image-index vector. Correlations
and AWGN-minus-clean correlation changes have pointwise 95% percentile intervals.
Changes also have simultaneous 95% basic bootstrap intervals based on the maximum
absolute error over all 24 descriptor/QP contrasts. Their metadata and tables are
in `results/full/analysis/partitions/div2k/`. DIV2K bpp correlations remain descriptive.
An interval containing zero does not establish equivalence or noise invariance.

## Quantization matrices

The matrix series is Kodak24 × QP4 × baseline/CSF × VTM/VVenC: 384 points.
QP32 provides 96 CU maps/summaries. Its conversion is FFmpeg 4:4:4 for VTM and
FFmpeg 4:2:0 for VVenC. Do not merge it with the OpenCV 4:4:4 comparison protocol.
The CSF implementation defines frequency-dependent weights; VTM
and VVenC do not apply the small-transform matrix identically.

Matrix RGB metrics use the YUV-derived reference/reconstruction
procedure; JPEG/VVC RGB metrics use original/stimulus RGB. These are different
references. Comparisons use measured operating points; a smaller CSF file at
the same QP alone does not establish improved rate-distortion efficiency.

## JPEG/VVC and quality matching

Pillow 12.3.0 / libjpeg-turbo 3.1.4.1 encodes JPEG QF 1–100 with 4:4:4,
`optimize=True`, `progressive=False`, and default quantization tables scaled by
QF. Actual tables are saved. A 32-MiB Pillow buffer is set explicitly inside the
JPEG module, without relying on global `sitecustomize.py` behavior.

For each QP, matching is performed separately by PSNR-RGB and MS-SSIM-RGB.
The reference is the actual clean or noisy input RGB, not a clean image used to
evaluate denoising. Dominated operating points are removed first; nearest,
lower and upper quality neighbors are selected among actual files on that
size/quality frontier. Consequently, the selected nearest point need not be
the closest quality among all 100 QFs before frontier filtering. No rate or
quality value is interpolated or extrapolated.

R = JPEG bpp / VVC bpp includes the full file sizes. Every comparison retains
the signed quality gap. Correlations use all six descriptors separately for
each QP, metric, dataset and clean/noisy condition. Image indices are resampled
together for paired comparisons. JPEG/VVC intervals are pointwise 95% intervals;
they do not establish simultaneous confidence for all reported correlations.
PSNR differences within 0.1 dB form a sensitivity subset, not a visibility or
JND threshold. Results include neighbor-choice and quality-gap checks.

## Timing and interpretation

The serial timing protocol and exact timer boundaries are documented in
[reproduce.md](reproduce.md). Parallel worker logs are retained as execution
metadata and are not pooled with the serial benchmark on 24 clean images.
Report absolute times, R and quality differences for the same pairs. A universal
codec-selection threshold is not implied by a descriptor correlation.

For each VTM image/QP point, use the median of three elapsed-time measurements.
For each selected JPEG image/QF point, divide each batch's elapsed time by its
32 encodes, then take the median of three batch estimates. The per-QP summaries
aggregate those per-image medians over all 24 images. A JPEG operating point
selected by multiple comparisons is timed once per repeat and reused in those
comparisons; it is not counted as a new independent measurement.

All repeats are retained in `results/full/timing/repeats.csv`. In the supplied
measurements, the largest VTM maximum/minimum repeat ratio is 1.0275. For 19 of
183 JPEG operating points, one batch estimate is over 20% longer than the shortest
estimate. At every JPEG point, the median is within 3.86% of the shortest estimate.
This describes observed repeat variation; the shortest estimate is not assumed
to be the true encoding time. The medians reduce the effect of individual long
repeats without deleting measurements.
