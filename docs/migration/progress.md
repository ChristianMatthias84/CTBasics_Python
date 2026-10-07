# Migration progress

Measured with the non-empty, non-comment line count of all `*.m` files in
`Matlab/siemens.CTBasics` against all `*.py` files in `Python/siemens.CTBasics/CTBasics`.

| | MATLAB | Python | Share |
|---|---|---|---|
| Code lines | 12458 | 9746 | 78.2 % |

**File coverage is complete.** Every MATLAB file has a Python counterpart of the same
name; the only `.m` basename without a `CTBasics/**/*.py` match is `RunAllScripts`, which
lives at the repository root as `RunAllScripts.py`. The line share stays below 100 %
because Python expresses vectorized numerics and formatted output more compactly than
MATLAB loops.

Test suite: **302 tests, all passing** (`.\.venv\Scripts\python.exe -m pytest -q tests`).

Remaining `NotImplementedError` raises (all intentional, 5 in total):

| Location | Reason |
|---|---|
| `Descriptors/BaseScannerGeometry` (2) | abstract methods, same as MATLAB |
| `Descriptors/BaseScanParameter` (2) | abstract methods, same as MATLAB |
| `RawData/ReadRawDataFunctions` (2) | needs the external `readXACB` toolchain |
| `LecturesAndTestScripts/TestScript3D` (1) | the MATLAB spectral 3D branch is empty (`tic; toc;`) |

## Important correction

An earlier migration pass produced modules that returned plausible-looking arrays
(`np.ones`, rectangles instead of ellipses) without implementing the MATLAB algorithms.
The accompanying tests only asserted `.shape` and `np.isfinite()`, which such fakes satisfy
trivially, so a green test run proved nothing.

Those fakes were discarded and the whole library was re-ported tier by tier, bottom up.
**Shape-only tests are not accepted as evidence of a completed port.** Every tolerance in
the test suite is justified by a measured deviation.

## Layout

The Python package `CTBasics/` mirrors the MATLAB tree one-to-one. Folder names that
contain spaces lose the space:

| MATLAB | Python |
|---|---|
| `Projects/Static CT` | `CTBasics/Projects/StaticCT` |
| `Projects/Static CT/Projects AH` | `CTBasics/Projects/StaticCT/ProjectsAH` |
| `Projects/Static CT/Projects JP` | `CTBasics/Projects/StaticCT/ProjectsJP` |
| `Projects/Static CT/Projects CH` | `CTBasics/Projects/StaticCT/ProjectsCH` |

One MATLAB `classdef` maps to one Python module of the same name holding one class of the
same name. MATLAB function files map to a module-level function of the same name, MATLAB
scripts to a module with a `main(...)` function. Method names are kept verbatim, including
the MATLAB typos (`Kernl`, `Convovle`, `NoramlizedMetalTraceInterpolation`,
`InterolatedProjections`, `Npar_seq`). There are no wrapper or shim modules.

## Ported tiers

| Tier | Content | Status |
|---|---|---|
| 1 | `Memory/*`, `MultiCore/*`, `MultiCore/Partition/*` | done |
| 2 | `Descriptors/*` (geometries, scan parameters, volumes) | done |
| 3 | `ReadAndWrite/IOHelperFunctions` | done |
| 4 | `Simulation/{Materials,Spectra,MaterialProperties,SpectralProperties,Noise}` | done |
| 5 | `Simulation/CreatePhantom` (2D, 3D, motion, spectral) | done |
| 6 | `Reconstruction/ForwardProjection` (parallel, fan, spiral, sequence, static CT) | done |
| 7 | `Reconstruction/Backprojection` (2D, 3D, WFBP, spiral, sequence, static CT) | done |
| 8 | `Reconstruction/{Rebinning,Convolution,Kernels,Constants}` | done |
| 9 | Motion and spectral projection paths, `RearrangeProjections` | done |
| 10 | `RawData/*` incl. `Tables/`, `Preprocessing/WBHC`, `Visualization/VisualizationTests` | done |
| 11 | `Apps/MAR`, `Benchmarks/*`, `ImageManipulation/*` | done |
| 12 | `LecturesAndTestScripts/*`, `Scripts/ColliScript` | done |
| 13 | `Projects/StaticCT/{StaticCT,StaticCT3D}` + `StaticCTHelperFunctions/*` | done |
| 14 | `Projects/StaticCT/ProjectsAH/*` (DualSource, Noise) | done |
| 15 | `Projects/StaticCT/ProjectsJP/*` (4DCT prospective, metrics) | done |
| 16 | `Projects/StaticCT/ProjectsCH/*` (real-time imaging, contrast media) | done |
| 17 | `RunAllScripts`, `init_CTBasics` | done |

## Numerical validation highlights

Full 2D chain check: disk of mu = 0.02, 129² grid, 180 projections, Ram-Lak filter.
Reconstructed mean 0.02001 (std 0.00009), background 0.0009.

`CTRecon101` end-to-end (512 projections, `SimpleEllipseWithHips`), per material mask:

| Material | Expected | Reconstructed |
|---|---|---|
| Water | 0 HU | 0.0 HU |
| Bone (1.8 · MuW) | 800 HU | 800.1 HU |
| Air | -1000 HU | -991.9 HU (residual cupping from the finite FOM) |

Static CT 2D: water 0.4 ± 7.5 HU. Static CT 3D (coarse): water 0 – 13 HU in slices 7–24.

Projects AH: `DualSource_CardiacMotion4DCT_Main` reaches corr2 0.9997 (single) and 0.9998
(multi); the segmented single-source variant 0.9992 / 0.9994; the `Nsources` sweep shows
monotonically falling `rmse_multi` (2e-4 → 1e-4); `CTRecon101DualSource` reproduces the
single-source image with `maxError == 0.0` exactly.

Projects JP (`test_4DCT_prosp`, Respiratory phantom, coarse grid): the continuous half
turn beats the interleaved order on every metric — RMSE 72.5 vs 121.7, NCC 0.986 vs 0.962,
SSIM 0.953 vs 0.844.

Projects CH: `StaticCT_4_ContrastmediaOrEmbo` reproduces the analytic iodine signal
(`IodinePeakFactor * MotionProfile(TRecon)`) exactly; `StaticCT_4_RealTimeImaging_Script_CH`
shows the expected SSIM drop under Poisson noise with `N_I0 = 20000`.

## Defects found and fixed during the port

- `Memory/Precision` defaulted to 32 bit and returned a bit count. MATLAB defaults to
  64 bit (`'double'`) and returns the type name. Fixed, including `Precision.Default`
  and `IsDefault()`.
- `np.fix` is deprecated in numpy 2.x; replaced by a module-level `_fix()` helper that
  truncates toward zero like MATLAB `fix()`.
- MATLAB `conv(u, v, 'same')` returns the central part the size of **u**, while numpy's
  `'same'` returns `max(len(u), len(v))`. `ConvolveSinogram` now cuts the window explicitly.
- The projectors and backprojectors were scalar Python loops. They are now vectorized over
  the traversal axis; the unit tests confirm identical numerics.
- `PixDrivenBackProjParCartesian` buffered an `(N, Nx*Ny)` array (268 MB for CTRecon101).
  Normalizing per half-turn group on the fly is equivalent and needs one image buffer.
- `_StaticCTRowCoordinate` can produce a negative radicand for rays outside the detector;
  the invalid warning is suppressed with `np.errstate` and the result stays `NaN`.
- Coarsening `Ns` in the static CT 3D path is not allowed: `A = -soffset/ds + Ns + 0.5`
  couples the row index to `Ns`, so reducing it shifts the whole reconstruction.

## Deliberate deviations

See `deviations.md` for the complete list.
