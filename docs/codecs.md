# Codecs

Source URLs, commits and build options are pinned in `manifests/codecs.lock.json`.
The pipeline uses VTM 23.0 and VVenC baseline/CSF encoder variants, with VTM and
VVdeC decoders. `D_QP` tracing records final luma coding units.

## Build from source

Install CMake, MinGW-w64 GCC **13.2.0** and GNU Make (`mingw32-make`) on `PATH`.

```powershell
uv run --locked vvc-study build-codecs --parallel 4
uv run --locked vvc-study check-codecs
```

Source checkouts are stored in `.cache/sources/`; builds and their logs are in
`.build/`. Completed builds are verified and reused. Compilation uses Release
and tracing-enabled encoder variants. The VTM CSF modification is applied from
`patches/vtm-csf.patch`; CSF streams use a baseline decoder. VVenC/VVdeC build
with link-time optimization disabled. Build manifests record flags, versions
and executable checksums.

The synthetic checks compare compressed streams, decoded pixels and available
partition traces with fixed expected outputs at QP 22 and 32. A mismatch is an
error to investigate before running larger experiments.

FFmpeg converts matrix inputs to 4:4:4 for VTM and 4:2:0 for VVenC. Its executable
hash is recorded. The partition and JPEG/VTM comparison uses OpenCV 4:4:4.

## Portable Windows bundle

Download the qualified [codecs-v1 archive](https://github.com/ooplisko/vvc-image-study/releases/tag/codecs-v1)
from the repository root in PowerShell:

```powershell
uv sync --locked
New-Item -ItemType Directory -Force .cache/downloads | Out-Null
Invoke-WebRequest -UseBasicParsing -Uri https://github.com/ooplisko/vvc-image-study/releases/download/codecs-v1/vvc-codecs-windows-x64.zip -OutFile .cache/downloads/vvc-codecs-windows-x64.zip
uv run --locked vvc-study import-bundle --archive .cache/downloads/vvc-codecs-windows-x64.zip --sha256 7f60af60a93be540b0b6a1ba6dc2ad8071dd64c9db6bf567c628fb12501c6ea3
.\reproduce.ps1 -Profile smoke -CodecMode bundle
```

Stop if any command reports an error. Import checks the archive against the
SHA256 above and verifies its manifest and payload files. The same checksum is
also supplied as `vvc-codecs-windows-x64.zip.sha256` in the release.

The archive includes encoders, decoders, MinGW runtime DLLs, licenses, build
metadata, synthetic-check results and file checksums. It does not include
Python, FFmpeg, datasets or Windows system libraries. The small `smoke` profile
needs no dataset or FFmpeg; `check-codecs` includes matrix controls and requires
FFmpeg on `PATH`. Full experiments also require FFmpeg for the matrix stage.
After installing FFmpeg, run the complete codec controls:

```powershell
uv run --locked vvc-study check-codecs --codec-mode bundle
```

Imported bundles are checked and cached under `.cache/bundles/`. Use
`--codec-mode bundle` for CLI experiments or `-CodecMode bundle` with
`reproduce.ps1`. For example:

```powershell
.\reproduce.ps1 -Profile full -CodecMode bundle -Workers 4
```

See [reproduction](reproduce.md) for result reuse, full recalculation and timing
requirements. The companion `gcc-runtime-sources-13.2.0.zip` contains GCC source
materials and is not needed to run the codecs. See
[binary distribution](binary-distribution.md) for component sources and licenses.

## Package a local build

After building and checking the codecs:

```powershell
uv run --locked vvc-study package-codecs
```

This creates `dist/vvc-codecs-windows-x64.zip` and a `.zip.sha256` sidecar.
It does not publish a release. A new build can have a different checksum; use
its own SHA256 when importing it. Preserve all bundled notices and provide
the runtime source materials appropriate to that build when redistributing it.
