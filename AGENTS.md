# Repository guidance

## Read first

- Read the migration guide in `COPILOT_MIGRATION_GUIDE.md` before changing code.
- Keep the Python port aligned with the MATLAB source tree and the documented milestones.
- Preserve existing user changes and keep edits focused on the requested task.

## Project and layout

- `src/ctbasics/` mirrors the MATLAB domains: descriptors, simulation, reconstruction, preprocessing, io, rawdata, parallel, memory, image_manipulation, visualization, and apps.
- `docs/migration/` holds the manifest, inventory, dependency notes, contract, validation plan, progress, and deviations.
- `validation/matlab_reference/` is reserved for isolated MATLAB reference material and generated comparison artifacts.
- `tests/unit/`, `tests/parity/`, and `tests/integration/` separate algorithmic checks from MATLAB parity and end-to-end coverage.

## Implementation conventions

- Search for existing functionality before adding new code.
- Prefer one implementation per behavior and avoid duplicate helpers.
- Keep the public API explicit and document non-obvious numerical choices, axis order, dtype handling, and boundary behavior.
- Use `pathlib` and configurable resource locations rather than hardcoded paths.

## Validation

- Add the smallest useful test for each ported slice before expanding scope.
- Record any MATLAB-only or blocked behavior explicitly in the migration docs.
