# Changelog: Python CTBasics

Short history of the project, newest first. Add one line per notable change (feature,
behaviour or numerical change, bug fix, restructuring, documentation concept) in the same
commit, under the current date. Details belong in the commit message, not here.

## 2026-10-22

- `LibraryConcept.md`: responsibility of every package defined.
- Added `KNOWN_ISSUES.md` and `CHANGELOG.md`; the migration folder with MATLAB parity tools
  and references was removed.
- Added `LibraryConcept.md` and the repository `ONBOARDING.md`. MATLAB is no longer the
  numerical reference; the library is developed on its own from here on.
- 3D accelerated with numba: cone-beam forward projection (fan beam, Static CT), WFBP 3D,
  and rebinning; convolution as a Toeplitz matrix product (BLAS). `StaticCT3D` 70 s
  (MATLAB 413 s), `TestScript3D` 30 s (MATLAB 217 s).
- Shared Static CT helpers consolidated in `StaticCTHelperFunctions/`; coding guidelines and
  shared `Data/` moved to the repository root.

## 2026-10-21

- 2D accelerated with numba: forward projection (parallel, fan, Static CT) and
  backprojection (parallel, WFBP 2D); 6–20× faster than MATLAB.
- MATLAB `parfor` ported as `Parallelization.Init`/`ParFor`/`Chunks`.
- Parity with MATLAB reached: about 1e-13 on deterministic paths; noisy outputs differ only
  by the random generator.

## 2026-10-07

- Library re-ported from MATLAB tier by tier (descriptors, simulation, reconstruction,
  applications, projects) with a pytest suite.

## 2026-10-01

- Initial Python port of the MATLAB CTBasics framework.
