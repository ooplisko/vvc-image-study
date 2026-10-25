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

After building and checking the codecs:

```powershell
uv run --locked vvc-study package-codecs
```

This creates `dist/vvc-codecs-windows-x64.zip` and a `.zip.sha256` sidecar.
The archive includes encoders, decoders, MinGW runtime DLLs, licenses, build
metadata, synthetic-check results and file checksums. It does not include
Python, FFmpeg, datasets or Windows system libraries.

Import the archive using its independently supplied SHA256:

```powershell
uv run --locked vvc-study import-bundle --archive C:/downloads/vvc-codecs-windows-x64.zip --sha256 <trusted-sha256>
uv run --locked vvc-study check-codecs --codec-mode bundle
.\reproduce.ps1 -Profile smoke -CodecMode bundle
```

Replace `<trusted-sha256>` with the actual 64-character checksum. Imported
bundles are checked and cached under `.cache/bundles/`. Use `--codec-mode bundle`
for CLI experiments or `-CodecMode bundle` with `reproduce.ps1` to run without
a compiler. Bundle checks detect missing or changed payload files.

Preserve all bundled notices and comply with the exact runtime distribution's
redistribution requirements when sharing binaries. See [notices](../THIRD_PARTY_NOTICES.md).
