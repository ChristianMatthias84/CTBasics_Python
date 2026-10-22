# Python-Specific Coding Guidelines

Supplements the general [CodingGuidelinesAndGeneralPrinciples.md](../../CodingGuidelinesAndGeneralPrinciples.md).

## Structure and naming

- The code lives in the package `CTBasics/`. Its names originate from the MATLAB CTBasics port
  and may be changed deliberately as the design evolves (see `LibraryConcept.md`).
- One class per module, named like the module. Function collections are classes with
  `@staticmethod` methods.
- Scripts are modules with a `main(...)` function. The script's control flags are keyword
  arguments with the same names and defaults, `main` returns the results as a `dict`, and the
  module ends with an `if __name__ == "__main__":` block so it can be run directly.
- Use absolute imports only (`from CTBasics.Reconstruction.Backprojection import Backprojection`);
  relative imports break direct execution of a script.

## Numerics

- Use NumPy/SciPy. Document axis order, dtype, and boundary behavior where they are not obvious.
- MATLAB is no longer the reference. Validate numerical changes against the previous Python
  results and against physical or analytical expectations. Existing comments on MATLAB
  differences (index bases, memory order, rounding) explain the current code; keep them
  accurate when that code changes.

## Paths

- Use `pathlib`. Obtain data and output folders through `IOHelperFunctions`, which reads
  `CTBasicsConfig.txt` from the MATLAB user folder (`Documents/MATLAB`) or from the folder set
  in the environment variable `CTBASICS_CONFIG`.

## Environment and tests

- Use the project's `.venv` (`python -m pip install -e .` installs the package in editable mode).
- Add or extend a focused pytest test in `tests/` for every change; tests use small sizes and
  a temporary `CTBasicsConfig.txt`.
- Run the full suite before closing a larger change: `python -m pytest -q tests`.
