# Progress

## Current milestone

M0 is in progress.

## Completed so far

- Confirmed the MATLAB repository baseline commit: `52e4374ed22298dcacf71a132de8c2d7a47f18ed`.
- Confirmed the MATLAB working tree is clean at the start of migration.
- Counted 485 source and resource files in the MATLAB tree.
- Identified unresolved raw-data dependencies in `RawData/ReadRawDataFunctions.m`.
- Created the initial Python project scaffold and migration docs.
- Confirmed the local Python runtime is available: Python 3.12.10 with pip 25/26 on this machine.
- Created and populated the isolated virtual environment in `Python/siemens.CTBasics/.venv`.
- Installed the editable package plus baseline dev dependencies from `pyproject.toml`.
- Added and passed a first pytest smoke test: `tests/unit/test_package_import.py`.
- Generated `docs/migration/original_manifest.csv` with 485 file hashes from the MATLAB tree.
- Generated `docs/migration/inventory.csv` as a first seed inventory over all MATLAB `.m` files and external raw-data dependencies.
- Added a reusable PowerShell generator at `tools/generate_migration_artifacts.ps1`.
- Added `validation/matlab_reference/README.md` as the isolated MATLAB reference area marker.
- Implemented the first Python descriptor slice: `ctbasics.descriptors` with enums, sampling, geometry classes, scan parameters, and `ScanDescriptor`.
- Added focused unit tests for descriptor factory creation, copy semantics, and the package smoke import.
- Verified the new descriptor slice with `pytest` on `tests/unit/test_package_import.py` and `tests/unit/test_descriptors.py`.
- Implemented a minimal IO helper slice with config parsing, folder listing, and DICOM slice reading support.
- Implemented a minimal raw-data helper slice with pure metadata functions and explicit `NotImplementedError` boundaries for binary-dependent readers.
- Implemented the memory helper slice with precision selection, allocation, and type casting.
- Added focused tests for IO, raw-data, and memory helpers.
- Verified the IO/raw-data/memory slices with `pytest` on the new unit tests.
- Implemented the first simulation slice: 2D phantom generation for the most-used test phantoms, including spectral material-code output.
- Added focused tests for representative non-spectral and spectral phantom cases.
- Verified the new simulation slice with `pytest` on the phantom and existing unit tests.
- Extended the simulation layer with motion phantoms, motion-profile support, spectral/material table loading, and Poisson-noise helpers.
- Copied the small spectral and attenuation coefficient tables into the Python package so the spectral helpers work without MATLAB paths.
- Added focused tests for motion-profile, motion phantom, and spectral material interpolation behavior.
- Verified the expanded simulation slice with `pytest` on the new motion/spectral tests and the previous unit tests.
- Implemented a focused 2D reconstruction slice with fan forward projection, fan-to-parallel rebinning, ramp/filter kernels, and pixel-driven backprojection.
- Implemented preprocessing helpers for beam-hardening correction and Moiré filtering.
- Implemented image-resampling helpers used by the lecture/test scripts.
- Implemented the 3D simple-ellipse phantom slice with z-varying hips, spine, and insert geometry.
- Added a legacy cardiac phantom compatibility wrapper that maps the older MATLAB entry point onto the current cardiac motion helper.
- Implemented StaticCT focal-spot sampling and the StaticCT visualization helper class, including ring/geometry/collimation plots.
- Implemented the StaticCT ring and leaf collimation forward-projection helpers plus the legacy `ColliScript` entry point.
- Implemented the MAR helper class and its core parallel-cartesian projection/interpolation/reconstruction pipeline.
- Implemented the motion-aware parallel forward-projection wrapper used by the respiratory/cardiac synthetic pipeline.
- Implemented the motion arbitrary-order forward-projection wrapper and the legacy motion-dual compatibility entry points.
- Implemented the 2D StaticCT forward-projection helpers used by the StaticCT geometry scripts.
- Added focused tests for the reconstruction, preprocessing, and image-manipulation slices.
- Added a focused regression test for the 3D phantom slice.
- Added a focused regression test for the legacy cardiac wrapper.
- Added focused tests for the StaticCT visualization, ColliScript, MAR, and motion-projection slices.
- Added focused tests for the motion arbitrary-order, motion-dual, and StaticCT forward-projection slices.
- Verified the new reconstruction/preprocessing/image-manipulation slices with `pytest` on the extended unit test set.
- Verified the 3D phantom slice with a focused `pytest` run.
- Verified the legacy cardiac wrapper with a focused `pytest` run.
- Verified the StaticCT visualization, ColliScript, MAR, and motion-projection slices with focused `pytest` runs.
- Verified the motion arbitrary-order, motion-dual, and StaticCT forward-projection slices with focused `pytest` runs.

## Open blockers

- MATLAB reference execution is not yet attempted.
- External TFS / raw-data toolbox availability still needs to be verified in an isolated reference setup.
- The inventory is still seed-level and needs method-by-method refinement.
- The Python descriptor layer is intentionally partial and still missing many MATLAB behaviors, including loading from raw descriptors, parallel/static-CT variants beyond initialization, and the rest of the pipeline.
- The raw-data reader and several higher-level script/project paths remain blocked by external TFS tooling and/or still-unported dependencies.
- The simulation layer is still partial and currently covers the core 2D phantom slice only; motion phantoms, spectra, material-property tables, and noise are still pending.
- The simulation layer is still partial and now covers the core 2D phantom slice plus the most-used motion, spectral-table, and noise helpers; the remaining 4D and edge-case variants are still pending.
- The reconstruction layer is still partial and currently covers the 2D fan/parallel path; the 3D, Static CT, and remaining variant-specific branches are still pending.
- The preprocessing layer is still partial and currently covers the core WBHC and Moiré helpers used by the synthetic pipeline.

## Next work package

- Refine the inventory for descriptors and raw-data I/O first, because those control the later pipeline setup.
- Expand the descriptor layer with the next highest-value behavior slice, likely `RawData` or `IOHelperFunctions` support depending on dependency proximity.
- Continue with the simulation layer or higher-level reconstruction entry points, depending on the next tightest dependency boundary.
- Continue the simulation layer with motion phantoms, spectra, and noise before moving on to reconstruction.
- Continue the simulation layer with the remaining 4D and edge-case variants before moving on to reconstruction.
- Expand the isolated MATLAB reference metadata once the first parity candidate is identified.
