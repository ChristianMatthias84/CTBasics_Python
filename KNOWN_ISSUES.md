# Known Issues: Python CTBasics

Known bugs, limitations, and questionable behaviour of the library. **Living list, subject to
change.** Add an entry when you find an issue you do not fix right away; remove it in the
change that fixes it and note the fix in [CHANGELOG.md](CHANGELOG.md).

One bullet per issue: where (module, class, or script), what happens, and the workaround if
there is one.

## Limitations

- Scanner raw data (`RawData/`, `TestScriptScannerData3D`, `FanBeam.Load`) depends on the
  MATLAB-only TFS toolbox (`readXACB`, ...) and does not work in Python. Planned redesign:
  see section 9 of [LibraryConcept.md](LibraryConcept.md).
- The spectral 3D forward projection is not implemented (it was empty in MATLAB as well);
  `TestScript3D` raises `NotImplementedError` for it.

## Bugs inherited from MATLAB

Ported verbatim to keep parity with the MATLAB original. Each is a candidate for a fix.

Reconstruction and simulation:

- `ForwardProjection.StaticCTCollimation*SourceSampled` sum the source samples without
  dividing by their count.
- `ForwardProjection.CalculateIntersectionPointsCircleAndLineIn2D_StaticCTGeo` returns `NaN`
  when the line misses the collimator circle; callers reject such rays.
- `Backprojection._WFBP3DVoxels` computes `l = sqrt((RF^2 - xi^2) - x*ss + y*cc)`; a negative
  radicand yields `NaN`.
- `Backprojection.SpiralWFBP3DFullyPixelDrivenParallelized_StaticCT` weights the cone angle
  with `RFD` (exact only for the centre beam) and sizes the last z window with `zStacksize`
  instead of `zStacksizeLast`.
- `Backprojection.SequenceWFBP3DFullyPixelDrivenParallelized_StaticCT` reconstructs the last
  z stack with the fan-beam `SequenceWFBP3DFullyPixelDriven`.
- `Spiral.SetScanRange` uses `max(Geometry.beta)` and therefore fails for `StaticCTBeam`.
- `ForwardProjection.RayDrivenForwProjParCartesianMultipleWithMotionDual_linear` allocates but
  never fills `PhantomTruth`; `..._Dual_ParallelSimple` sizes it `(N, M)`.
- `CreatePhantom`: `WaterWith4Arteries` negates the quarter condition for artery 4;
  `CreateVoxelPhantom2D('SimpleEllipseWithHips3D')` cannot work (raises `ValueError`).
- `ImageManipulation.Resample2D` uses the query grid `1:sz/N:sz`, which is not centred.

Applications and projects:

- `MAR.Apply` never removes the +1000 input shift; NMAR normalization adds 1000 to the line
  integrals. `MAR.MetalTraceInterpolation` raises `IndexError` at the detector border.
- `StaticCTSystemHelperFunctions.GreedyGapSamplingSubset(0, ...)` returns one angle.
- `CTRecon101DualSource` calls the projector with one argument missing (`TypeError`);
  `UseArbitraryOrder=True` is the working path.
- `StaticCT_4_ContrastmediaOrEmbo` recreates the loop-invariant ground truth inside the
  backprojection loop and leaves `ProjectionsGT` empty.
- `test_4DCT_prosp` always uses 2 subsets for the arbitrary angle order (`NSubsets` only
  labels the output).
- `NsourcesSweep` passes no `SelectedPhantom` (projector default `"Cardiac"`) and discards
  the Moire-filtered multi-source sinogram when `SimNoise` is on.
- `TestNoiseCardiac4DCT` hard-codes `Heartrate = 80` in the projector calls.
