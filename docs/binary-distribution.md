# Windows codec distribution

The [codecs-v1 release](https://github.com/ooplisko/vvc-image-study/releases/tag/codecs-v1)
provides the qualified Windows x64 encoders, decoders and runtime libraries.
The source lock, codec configuration and control fixtures are included in the
archive. `bundle.json` records build settings and hashes for every payload file.

## Codec sources and notices

- VTM 23.0 uses commit `8fc9675016f85fa7965e96a32661ca89833e2a0d` from
  [VVCSoftware_VTM_CSF](https://github.com/ooplisko/VVCSoftware_VTM_CSF/tree/8fc9675016f85fa7965e96a32661ca89833e2a0d).
  The CSF encoder also applies the included `patches/vtm-csf.patch`.
  The upstream BSD notice is preserved under `licenses/vtm23-*/`.
- VVenC uses commits
  [`6f767486ba2199d9f3f8a75b7739995bd3ed9034`](https://github.com/ooplisko/vvenc/tree/6f767486ba2199d9f3f8a75b7739995bd3ed9034)
  and [`408b42d6847c03403a0bceed1a0ce54f6d914b8e`](https://github.com/ooplisko/vvenc/tree/408b42d6847c03403a0bceed1a0ce54f6d914b8e)
  for the baseline and CSF encoders.
- VVdeC uses commit
  [`4046482c5693d613aead9bc33a6f60630e20cca6`](https://github.com/fraunhoferhhi/vvdec/tree/4046482c5693d613aead9bc33a6f60630e20cca6).
  VVenC/VVdeC Clear BSD notices and author lists are preserved in `licenses/`.
  These licenses do not grant patent rights.

## Runtime libraries

The three DLLs are unmodified files from the MinGW-Builds
[13.2.0-rt_v11-rev0 distribution](https://github.com/niXman/mingw-builds-binaries/releases/tag/13.2.0-rt_v11-rev0):
`x86_64-13.2.0-release-posix-seh-ucrt-rt_v11-rev0.7z`.
Its SHA256 is
`a05578fab9068c678ce761e7b2e284d23fa7416a2f4f36349e05e506c944e2c7`.
Bundled DLLs and notices were compared with that upstream archive.

- `libgcc_s_seh-1.dll` and `libstdc++-6.dll`: GCC 13.2.0, GPLv3 with the
  GCC Runtime Library Exception 3.1. The GPL and exception texts are included
  under `licenses/runtime/gcc/`. The exception applies to eligible compiled
  combinations; it does not remove the library's own license.
- `libwinpthread-1.dll`: MinGW-w64 winpthreads, with the MIT and BSD notices
  retained under `licenses/runtime/winpthreads/`.
- MinGW-w64 runtime and incorporated-component notices are retained under
  `licenses/runtime/mingw-w64/`.

The companion `gcc-runtime-sources-13.2.0.zip` in the same release provides the
GCC source archive, the ten GCC patches listed in the compiler's build metadata,
MinGW-Builds scripts, the provider's build information and license texts.
It is available alongside the binaries without requiring a separate request.
Its source manifest records download URLs and checksums. It is not needed to
run the codecs.

The GCC archive is the official
[13.2.0 source release](https://gcc.gnu.org/pub/gcc/releases/gcc-13.2.0/), verified
against its published SHA512. The build-script snapshot is
[`b876df909372d7fb0ab21aa58ce1b7c0df813dbe`](https://github.com/niXman/mingw-builds/tree/b876df909372d7fb0ab21aa58ce1b7c0df813dbe);
its GCC patch list matches the provider's build information. See the
[GCC exception FAQ](https://www.gnu.org/licenses/gcc-exception-3.1-faq.html.en)
for the distinction between distributing a compiled program and the runtime
library itself.

## Scope

Python, FFmpeg, datasets and Windows system libraries are external prerequisites.
Keep the notices with redistributed binaries. The repository's MIT license
does not replace any of the upstream licenses.
