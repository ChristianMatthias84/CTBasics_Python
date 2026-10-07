# Deliberate deviations from the MATLAB source

Every entry below is intentional. Behaviour that is simply missing is not listed here;
it raises `NotImplementedError` instead (see `progress.md` for the five remaining cases).

## Naming and layout

- `Projects/Static CT` becomes `CTBasics/Projects/StaticCT`; `Projects AH`, `Projects JP`
  and `Projects CH` become `ProjectsAH`, `ProjectsJP`, `ProjectsCH`. Python packages
  cannot contain spaces.
- MATLAB scripts become modules with a `main(...)` function. MATLAB script names that are
  unpythonic are kept verbatim anyway (`test_4DCT_prosp.py`,
  `StaticCT_4_RealTimeImaging_Script_CH.py`).
- MATLAB typos in method names are preserved: `Kernl`, `Convovle`,
  `NoramlizedMetalTraceInterpolation`, `InterolatedProjections`, `Npar_seq`.
- `Scripts/__init__.py` re-exports the function `ColliScript`, which shadows the module of
  the same name. Import individual functions from `CTBasics.Scripts.ColliScript`.

## MATLAB bugs reproduced verbatim

- `CalculateIntersectionPointsCircleAndLineIn2D_StaticCTGeo` returns `NaN` when the line
  misses the collimator circle. MATLAB produces complex values there; callers reject the ray.
- `StaticCTCollimation*SourceSampled` accumulates samples **without** dividing by their
  count, matching MATLAB despite the `Averaged...` name.
- `CreateVoxelPhantom2D('SimpleEllipseWithHips3D')` indexes `Nz`, which is not a parameter
  of the 2D entry point, so the MATLAB call always fails. Python raises a `ValueError`
  pointing at `CreateVoxelPhantom3D`.
- `CreateVoxelPhantomCardiac2DWithMotion` references an undefined `ContrastFlag` in MATLAB.
  Python uses the parsed `ContrastMedium` argument.
- `WaterWith4Arteries` artery 4: MATLAB negates the quarter condition. Ported verbatim.
- `_TraceRay3D` keeps `nz_real = zsource/dz + (B - x*s + y*s)*A/dz - z0/dz` verbatim.
- `_StaticCTRowCoordinate` keeps MATLAB's `l = sqrt((RF^2 - xi^2) - x*ss + y*cc)`;
  a negative radicand yields `NaN` (warning suppressed via `np.errstate`).
- `SpiralWFBP3DFullyPixelDrivenParallelized_StaticCT` weights the cone with `RFD` and sizes
  the tail window with `zStacksize`.
- `SequenceWFBP3DFullyPixelDrivenParallelized_StaticCT` calls the **fan** variant
  `SequenceWFBP3DFullyPixelDriven` for the tail stack.
- `..._Dual_linear` allocates but never fills `PhantomTruth`; `..._Dual_ParallelSimple`
  sizes `PhantomTruth` as `(N, M)`.
- `ImageManipulation.Resample2D` keeps MATLAB's `1:sz/N:sz` query grid.
- `MAR.Apply` never removes the +1000 input shift; NMAR normalization adds 1000 to the
  line integrals.
- `MAR.MetalTraceInterpolation` raises `IndexError` at the detector border, where MATLAB
  aborts with an index error too.
- `StaticCTSystemHelperFunctions.GreedyGapSamplingSubset(0, ...)` returns one angle,
  replicating MATLAB's empty-array growth.
- `CTRecon101DualSource.m` passes 12 of 13 arguments to the projector, so MATLAB aborts
  with "Not enough input arguments". Python raises `TypeError`; the keyword
  `UseArbitraryOrder=True` provides a working escape hatch.
- `StaticCT_4_ContrastmediaOrEmbo` recreates the ground truth inside the backprojection
  loop even though it is loop invariant, and leaves `ProjectionsGT` empty (`[]` in MATLAB).
- `NsourcesSweep` passes no `SelectedPhantom` to the projector (MATLAB does not either),
  so the projector default `"Cardiac"` applies, and it discards the Moire filtered multi
  sinogram when `SimNoise` is on.
- `TestNoiseCardiac4DCT` hard-codes `Heartrate = 80` in the projector calls.
- `Spiral.SetScanRange` uses `max(self.Geometry.beta)` and therefore crashes on
  `StaticCTBeam`. MATLAB has the identical bug.
- `ReadRawDataFunctions.GetNoOfScans` reads the full tag value; MATLAB reads a single
  character and therefore caps at 9.

## Environment and interactivity

- MATLAB `parfor` loops are ported as serial loops. `MultiCore/Parallelization` still
  reports the available workers.
- Debug `figure()`, `fprintf`, `waitbar`, `tic`/`toc` and `imshow(mat2gray(...))` calls
  inside numerical routines are omitted or moved behind an optional `ShowFigures` flag.
- `CTRecon101` takes the output directory as an argument and only writes when one is
  given, instead of requiring a `CTBasicsConfig.txt`.
- `TestFuseFFS.main(filename)` and `TestScriptScannerData3D.main(filename)` replace
  MATLAB's `uigetfile` dialog.
- `TestMAR.main(UsePatientData=False)` and `TestScript2D.main(PatientData=False)` default
  to the simulated branch so the scripts run without raw data.
- `TestCardiacPhantom.main(outputDir=None)` replaces a hardcoded personal output
  directory; PNGs are only written when a directory is given.
- `init_CTBasics` keeps MATLAB's config-file presence check and the TFS directory
  validation, but has no `addpath(genpath(...))` equivalent: the package is importable
  after `pip install -e .`. The TFS subfolders are reported instead of added to a path.
- `RunAllScripts.main` runs the same twelve scripts in the same order as
  `RunAllScripts.m`, with an opt-in `StopOnError=False` to continue after a failure.

## Signature changes (test affordances)

- `TestScript3D.main` and `StaticCT3D.main` accept optional `Descriptor`, `SimVol` and
  `ReconVol` overrides so the tests can run on coarse grids.
- `TestCardiac4DCT.main`, `TestNoiseCardiac4DCT.main` and the sweep scripts accept a
  `Seed` argument to make the randomized view order reproducible. The Poisson noise still
  draws from the global generator, matching MATLAB.
- `test_4DCT_prosp.main` gains `UseLinearCache` so the `.mat` cache can be bypassed, and
  `SimNoise` / `N_I0` (default **off**, so the JP path stays MATLAB identical).
- `StaticCT_4_RealTimeImaging_Script_CH.main` delegates to `test_4DCT_prosp.main` with the
  CH defaults (`SimNoise=True`, `NScansPerRecon=5`, `TRot=0.1`,
  `SelectedPhantom="WaterWith4Arteries"`). The MATLAB file is a copy of
  `test_4DCT_prosp.m` with a Poisson-noise stage added; sharing one implementation avoids
  roughly 400 duplicated lines.
- `DualSource_showAndSaveImage` drops MATLAB's unused trailing `cx, cy` arguments and
  takes a `ShowFigures` flag.

## Output format changes

- The `DualSource_*` sweep writes `.npz` instead of a MATLAB `.mat` file.
- `StaticCT_computeReconstructionMetrics` returns `roi_mask` and `center_mask` as Python
  dicts keyed `(phaseIndex, roiIndex)` instead of MATLAB cell arrays.
- Report files written with `fprintf(fid, '...\r\n')` are reproduced byte-identically via
  `open(path, "w", newline="")` plus explicit `\r\n`.

## Local reimplementations of MATLAB toolbox functions

To avoid adding dependencies, a few Image Processing Toolbox functions are reimplemented:

- `mat2gray` → private `_MatToGray` (linear rescale to `[0, 1]`; a constant image maps to
  all zeros).
- `corr2` / `rmse` → module-level `corr2` / `rmse` in
  `DualSource_CardiacMotion4DCT_Main.py`, private `_Corr2` / `_RMSE` in
  `StaticCT_computeReconstructionMetrics.py`.
- `ssim(A, B, 'DynamicRange', dr)` → private `_SSIM` with an 11x11 Gaussian window
  (sigma 1.5), `K = [0.01, 0.03]` and symmetric padding; the window shrinks for crops
  smaller than 11 pixels. Returns `NaN` for a zero dynamic range or an empty ROI.
