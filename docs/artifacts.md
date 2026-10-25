# Result files

`results/full/` contains compact, portable outputs:

- `tables/encodings.csv`: partition, descriptor and quality measurements for
  VTM/VVenC operating points. Unmeasured fields are empty.
- `tables/measurements.csv`: JPEG QF 1–100 and VTM QP 22/27/32/37 measurements
  for the clean/noisy Kodak and DIV2K comparison.
- `tables/actual_by_qp.csv`: actual nearest, lower and upper JPEG quality
  neighbors, file-size ratios, signed quality gaps and compressed-file hashes.
- `analysis/`: statistical tables and figures, grouped by experiment.
- `timing/`: completed serial time measurements and summaries for 24 clean Kodak images.
- `manifest.json`: experiment configuration and checked result-stage identities.

The full set has 2,432 encoding points, 25,792 JPEG/VTM measurements and 1,984
quality matches. A file-location field may be empty when only the measured table
is distributed. A checksum identifies the measured file; it does not mean the
raw file is included. Fresh runs record raw paths relative to the repository.

`verify` checks compact results. The separate `verify --jobs` checks saved plans
and raw artifacts under `runs/full/`. Statistical output hashes are checked by
`analyze`; `analyze --recompute` recalculates the analysis.

## Timing outputs

- `timing/repeats.csv`: 837 measurements: 288 VTM repeats and 549 JPEG batches.
  Each JPEG batch contains 32 encodes; `elapsed_seconds` is the batch time divided
  by 32. Warmups are excluded.
- `timing/paired.csv`: 192 image/QP/quality-metric comparisons, with median encoding
  times, additional VTM time, file-size ratio R and the actual quality difference.
- `timing/summary.csv`: eight summaries (four QPs × two quality metrics). Mean,
  sample standard deviation and median are calculated over the 24 per-image values.
- `timing/context.json`: CPU, affinity, software versions and protocol identities.
- `timing/manifest.json`: summary checksums and the matching-table identity.

`verify` checks the summary files and their matching-table identity. `verify --jobs`
also checks the individual measured compressed files and job records. The timing
data cover clean Kodak images on one logical CPU of an Intel Core i7-12700F.
They do not measure noisy inputs, DIV2K or VVenC execution time.

## Raw outputs

New encoding jobs retain compressed files, reconstructions, encoder/decoder logs,
and CU traces where requested under `runs/<profile>/jobs/`. Each completed job
has a `result.json` with its parameters and artifact checksums. JPEG sweeps retain
the generated JPEG files and their quantization tables. Files are kept on disk;
they are not deleted after exporting the compact result tables.

Raw files distributed alongside compact results can be placed under
`runs/full/artifacts/`. Table file-location fields point to available files using
repository-relative paths. Its `manifest.json` lists file sizes and SHA256 values.
Check this optional bundle with `vvc-study verify --artifacts`; use
`verify --jobs` for job execution records. Bundle verification covers listed
files, not every measured JPEG QF. A blank file-location field means that the
measurement is available but its raw file is not included.

Reusing complete result tables does not recreate absent raw files. Keep a separate
backup of expensive encoding outputs. To distribute them, archive the artifact
directory with its manifest; recipients extract it to the same relative location.

## Generated files and Git

One root `.gitignore` excludes `data/`, `runs/`, `.cache/`, `.build/`, `.venv/`
and `dist/`. It also excludes Python caches and local editor files. CSV, JSON
and PNG are not ignored globally because configurations and result tables use
those extensions.

Keep compact result tables and selected figures in Git. Large raw YUV files,
traces, downloaded originals and executables stay outside ordinary commits.
Use separate assets with SHA256 manifests when distributing large outputs.
Codec bundles include their licenses; dataset redistribution follows each
dataset provider's terms.

Use a version tag to identify a stable software/result set. Corrections belong
in a new version rather than replacing a previously distributed archive.
