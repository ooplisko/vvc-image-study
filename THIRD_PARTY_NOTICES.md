# Third-party materials

The MIT license at the repository root covers this project's code. It does not
relicense datasets, codec sources, codec binaries or external Python packages.

- **VTM**: reference software from JVET, pinned through the source URL and commit
  in `manifests/codecs.lock.json`. Preserve the upstream license and copyright
  notices when redistributing source or binaries; the CSF patch does not remove
  them. The build fetches the original source including its license.
- **VVenC / VVdeC**: Fraunhofer HHI codec implementations, with upstream
  `LICENSE.txt` and associated notices. Source-build recipes retain copies of
  those license files. A binary package must carry the applicable notices.
- **FFmpeg**: an external program, not included in this source repository.
  Licensing depends on the distributed build and its enabled components.
- **MinGW runtime libraries**: the optional local codec archive includes
  `libgcc_s_seh-1.dll`, `libstdc++-6.dll` and `libwinpthread-1.dll` from its
  recorded compiler installation. GCC, GCC Runtime Library Exception,
  MinGW-w64 and winpthreads notices are copied from that distribution.
  These libraries are not covered by this project's MIT license. Before a
  public binary release, confirm the redistribution requirements for the exact
  runtime distribution, including corresponding source availability where required.
- **Kodak**: [Kodak Lossless True Color Image Suite](https://r0k.us/graphics/kodak/).
  Originals are obtained from their source and are not included in Git.
- **DIV2K**: [official dataset](https://data.vision.ee.ethz.ch/cvl/DIV2K/), used for
  academic research. Cite E. Agustsson and R. Timofte, *NTIRE 2017 Challenge on
  Single Image Super-Resolution: Dataset and Study*, CVPR Workshops, 2017.
  Original images are downloaded from the dataset provider.
- **Python dependencies**: versions and distributions are recorded in `uv.lock`;
  their respective package licenses remain applicable.

Scientific figures under `results/` may contain dataset image content;
the image owners' rights remain applicable. Do not interpret the root MIT
license as granting independent rights to that image content.
