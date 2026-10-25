# Running on Windows

## Environment

Use Windows x64, Git and uv 0.11.19. Run `uv sync --locked` to install Python
3.11.9 and the pinned Python packages. `vvc-study doctor` lists available system
tools. Codec builds require CMake, MinGW-w64 GCC 13.2.0 and `mingw32-make`;
FFmpeg is needed for matrix experiments. See [codecs](codecs.md) for builds
and portable bundles.

## Profiles and modules

`smoke` uses deterministic 96×96 and 97×99 synthetic inputs. It checks encoding,
decoding, padding, metrics, JPEG matching and job reuse without downloading a
dataset. `full` uses the experiment settings in `configs/full.toml`.

```powershell
.\reproduce.ps1 -Profile smoke
.\reproduce.ps1 -Profile full -Workers 4
```

The full sequence runs `partitions`, `div2k`, `matrices`, `jpeg-vvc`, `timing`
and analysis. A complete verified result stage is reused. To regenerate a stage
from job files, use `vvc-study run --study <name> --recompute`; valid jobs remain
reusable. Starting without results or jobs performs all required computations.

The encoding grid contains 1,248 Kodak partition points, 800 DIV2K partition
points, 384 matrix points, and 24,800 JPEG files. JPEG/VVC matching reuses the
relevant partition encodes. `vvc-study plan` prints the grid and a lower bound
on storage. Raw YUV and trace files require substantially more space than CSVs.

`prepare --profile full` downloads and verifies original data and prepares
codecs without encoding the dataset. Kodak has 24 PNGs; DIV2K validation has
100 PNGs, IDs 0801–0900. Existing originals are checked by SHA256 and geometry.
No resizing or cropping is applied. VTM performs required edge padding.

## Results and analysis

```powershell
uv run --locked vvc-study verify
uv run --locked vvc-study analyze --check-only
uv run --locked vvc-study analyze
```

Results are under `results/full/`. Verification checks experiment keys,
completeness, numerical relationships and checksums. `verify --jobs` checks
available raw job artifacts under `runs/full/`; compact tables alone do not
supply those files. `analyze` reuses complete analysis outputs whose inputs and
checksums match. Use `--recompute` to run the statistical calculations again.
The default analysis uses 99,999 resamples and can take appreciably longer than
validation. Smaller `--resamples` values are for software testing.

If an optional raw-output bundle is available under `runs/full/artifacts/`, check
its files with `vvc-study verify --artifacts`. This reads every listed file and
checks its size and SHA256; it does not encode images. See [result files](artifacts.md)
for the difference between compact tables, raw bundles and resumable job records.

## Encoding-time measurements

```powershell
uv run --locked vvc-study timing-plan
uv run --locked vvc-study run --study timing --workers 1
```

The input is `results/full/tables/actual_by_qp.csv`; `--matched <path>` selects
another complete matching table. The protocol uses all 24 clean Kodak images,
QP 22/27/32/37 and three VTM repeats: 288 measurements. JPEG uses each distinct
selected image/QF once per repeat, measured in batches of 32 encodes. Quality
search and metric calculation are outside the timer.

Keep the PC idle. Only one encoding operation runs at a time. `--cpu 0` selects
one logical CPU; the machine, CPU model, affinity and software are recorded.
The timer starts with an already loaded RGB array and ends after writing the
compressed file. It includes color conversion and the native VTM process start,
and excludes PNG loading, Python imports, metric calculations and hash checks.
Each timed file must match its quality-measured compressed-file checksum.

Raw measurements and resume state are in `runs/full/timing/`. Completed summaries
are written to `results/full/timing/`. A changed hardware/software context is
rejected to prevent combining incompatible measurements.

## Stop and resume

```powershell
uv run --locked vvc-study stop --profile full
uv run --locked vvc-study status --profile full
```

The stop request finishes active jobs, saves them, and prevents further dispatch.
Wait until the running command exits before shutting down. Repeat that same
command to resume. Job parameters and artifact hashes are verified before reuse;
damaged or unfinished jobs are repeated. Timing warms up every resumed session.
Use the stop command rather than Ctrl+C for timing.

Exit codes are 0 for completion, 1 for failure and 2 for a graceful stop.
