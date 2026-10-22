# CTBasics (Python)

Python framework for CT physics: phantom simulation, forward projection, fan-to-parallel
rebinning, filtering, and weighted filtered backprojection for 2D, 3D sequence, and 3D spiral
scans, plus stationary multi-source (Static CT) experiments and metal artifact reduction.

Architecture and conventions: [LibraryConcept.md](LibraryConcept.md).

## Installation

```bash
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e .[dev]
```

## Configuration

`CTBasicsConfig.txt` in `Documents/MATLAB` (or in the folder set by the environment variable
`CTBASICS_CONFIG`) defines the paths:

```text
%global folder of the repository
<AI_Coding_Master folder, contains the shared Data/>
%Output directory
<existing output folder>
%path to TFS matlab functions for CT
<TFS toolbox folder, only needed for scanner raw data>
```

## Usage

Run a script directly, e.g. `python CTBasics/LecturesAndTestScripts/CTRecon101.py`, or call its
`main(...)` with the desired parameters:

```python
from CTBasics.LecturesAndTestScripts.CTRecon101 import main

Results = main(ShowFigures=True)
```

`python RunAllScripts.py` runs the complete script suite.

## Tests

```bash
.\.venv\Scripts\python.exe -m pytest -q tests
```
