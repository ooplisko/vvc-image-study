# vvc-image-study

<p align="center">
  <a href="LICENSE">
    <img src="https://img.shields.io/badge/license-MIT-green" alt="MIT license">
  </a>
  <a href="https://www.python.org/downloads/release/python-3119/">
    <img src="https://img.shields.io/badge/python-3.11.9-blue" alt="Python 3.11.9">
  </a>
  <a href="docs/reproduce.md">
    <img src="https://img.shields.io/badge/platform-Windows_x64-blue" alt="Windows x64">
  </a>
  <a href="https://github.com/ooplisko/vvc-image-study/actions/workflows/checks.yml">
    <img src="https://github.com/ooplisko/vvc-image-study/actions/workflows/checks.yml/badge.svg?branch=main" alt="Windows checks">
  </a>
</p>

**O. O. Plisko** - [Department of Information and Communication Technologies](https://dict.khai.edu/), National Aerospace University "Kharkiv Aviation Institute"

Experiments on VVC still-image compression: block partitions, image complexity,
noise, quantization matrices, and comparisons with JPEG. Runs on Windows x64.

## Quick start

Install [uv](https://docs.astral.sh/uv/getting-started/installation/) **0.11.19**
and Git, then open PowerShell in this directory. The lock file selects Python
3.11.9 and the Python dependencies.

```powershell
uv sync --locked
uv run --locked vvc-study verify
uv run --locked vvc-study analyze --check-only
uv run --locked vvc-study plan
```

Measurement tables and statistical outputs are in `results/full/`. These commands
check their completeness and integrity without encoding images. `analyze`
reuses verified analysis outputs; `analyze --recompute` recalculates them.

## Run experiments

Building the codecs requires CMake, MinGW-w64 GCC **13.2.0** and
`mingw32-make` on `PATH`. Matrix experiments also require FFmpeg. Run
`uv run --locked vvc-study doctor` to check available tools.

```powershell
# Tiny synthetic inputs: four VTM encodes and twelve JPEG files.
.\reproduce.ps1 -Profile smoke

# Complete workflow; downloads inputs and builds codecs as needed.
.\reproduce.ps1 -Profile full -Workers 4
```

The complete workflow reuses verified result tables and completed jobs. It
encodes only missing work, then measures encoding time and analyzes the results.
From an empty results directory this is a long computation. The timing stage
runs sequentially and requires an idle PC.

Modules can be run independently:

```powershell
uv run --locked vvc-study run --study partitions --workers 4
uv run --locked vvc-study run --study div2k --workers 4
uv run --locked vvc-study run --study matrices --workers 4
uv run --locked vvc-study run --study jpeg-vvc --workers 4
uv run --locked vvc-study run --study timing --workers 1
uv run --locked vvc-study analyze
```

`run --recompute` regenerates a stage from its job records, encoding absent or
invalid jobs. A verified portable codec bundle can replace a local compiler;
see [codec setup](docs/codecs.md).

## Experiments

- **Kodak:** 24 original images; clean, Gaussian noise with sigma 5/15/30 and
  three seeds, and sinusoidal interference with amplitudes 8/16/32.
- **DIV2K:** all 100 full validation images; clean and Gaussian noise with
  sigma 30. Images are neither cropped nor resized.
- **Partitions:** VTM, QP 22/27/32/37; final luma coding units and six image
  descriptors, with paired statistical and sensitivity analyses.
- **Matrices:** baseline and CSF configurations for VTM and VVenC; four QPs,
  with partition traces at QP 32.
- **JPEG/VVC:** JPEG QF 1–100 on 248 clean/noisy inputs; measured pairs selected
  separately by PSNR-RGB and MS-SSIM-RGB, without interpolation.
- **Timing:** all 24 clean Kodak images; three VTM repeats per QP and repeated
  JPEG measurements at the selected quality factors. Completed measurements,
  paired comparisons and summaries are in [results/full/timing](results/full/timing).

## Stop and resume

Request a stop from another terminal:

```powershell
uv run --locked vvc-study stop --profile full
uv run --locked vvc-study status --profile full
```

Wait for the active command to return `status: stopped` before shutting down.
It finishes and saves active work before stopping. Repeat the original command
to resume. Completed jobs are verified before reuse; incomplete jobs are retried.
Timing uses one worker and warms up each resumed session.

## Layout

```text
configs/       Experiment profiles and codec settings
manifests/     Dataset checksums and pinned codec sources
patches/       Codec modifications
src/vvcstudy/  Preparation, execution, measurements and analysis
tests/         Numerical, integrity and resume tests
results/full/  Measurement tables, statistical outputs and figures
docs/          Setup, methods and result formats
```

Downloaded data, builds, raw job files and environments are ignored by Git.
Compact result tables and figures can be versioned with the code. See
[reproduction](docs/reproduce.md), [methods](docs/methods.md),
[result files](docs/artifacts.md) and [contributing](CONTRIBUTING.md).

The project code uses the [MIT license](LICENSE); external materials have their
own [terms](THIRD_PARTY_NOTICES.md). Citation metadata is in [CITATION.cff](CITATION.cff).
