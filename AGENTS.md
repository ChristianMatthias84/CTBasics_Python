# Repository guidance

## Read first

- Read `../../CodingGuidelinesAndGeneralPrinciples.md` (general principles for all projects)
  and `PythonSpecificCodingGuidelines.md` before changing code. Both are referenced by the
  central `../../.github/copilot-instructions.md`.
- Read `LibraryConcept.md` for the architecture: packages, data model, pipeline, conventions,
  and where new code goes. Keep it current when the architecture or a convention changes.
- Preserve existing user changes and keep edits focused on the requested task.

## Project and layout

CTBasics is a Python framework for CT simulation and reconstruction (port of the MATLAB
CTBasics). The package lives in `CTBasics/`, the pytest suite in `tests/` (`unit/`,
`integration/`), and `RunAllScripts.py` runs all scripts (slow, opens figures). The package
structure is described in `LibraryConcept.md`.

## Setup

- Python 3.11+ with the local virtual environment `.venv`; install with
  `python -m pip install -e .`.
- The active configuration is `CTBasicsConfig.txt` in the MATLAB user folder
  (`Documents/MATLAB`, OneDrive redirection is followed) or in the folder given by the
  environment variable `CTBASICS_CONFIG`. Line 2 is the AI_Coding_Master root holding the
  shared `Data/` folder,
  line 4 the output directory, line 6 the external TFS toolbox path. Data and output
  directories must exist.

## Validation

- Run the focused tests for the changed code, then the full suite before closing a larger
  change: `.\.venv\Scripts\python.exe -m pytest -q tests`.
- Scripts can be run directly (`python CTBasics/LecturesAndTestScripts/CTRecon101.py`) or as
  `main(...)` with small sizes for quick checks.
- For numerical changes, check dimensions, finite values, and relevant image or quantitative
  results against a suitable baseline; successful execution alone does not establish
  reconstruction correctness.
