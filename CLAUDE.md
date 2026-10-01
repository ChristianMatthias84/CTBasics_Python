# CLAUDE.md

This file gives implementation guidance for the Python port of the CTBasics MATLAB framework.

## What this is

The repository is the Python migration target for CTBasics, covering phantom simulation, forward projection, fan-to-parallel rebinning, filtering, weighted filtered backprojection, and application logic for Static CT and MAR. The port keeps the original MATLAB domain structure while using Python package names and idioms.

## Setup and running

The project is configured as a regular editable Python package:

```bash
cd Python/siemens.CTBasics
python -m pip install -e .
```

The project uses the local virtual environment in `.venv` and a standard pytest validation flow.

```bash
.\.venv\Scripts\python.exe -m pytest -q tests/unit
```

No MATLAB runtime is required for the Python validation path. The external raw-data/toolbox readers are intentionally deferred and should remain out of scope until explicitly requested.

## Architecture

The Python package mirrors the MATLAB domain structure under `src/ctbasics`:

1. `Descriptors/`: scan geometry, scan parameters, and descriptor-based setup.
2. `Simulation/`: phantom creation, motion, spectra, materials, and noise.
3. `Reconstruction/`: projection, rebinning, convolution, kernels, and backprojection.
4. `Preprocessing/`: beam-hardening and filtering helpers.
5. `ReadAndWrite/`: configuration and file-access helpers.
6. `Visualization/`: geometry and imaging plots.
7. `Apps/`: higher-level user-facing entry points such as `ColliScript` and `MAR`.

Compatibility wrappers are intentionally thin. The goal is to keep the original MATLAB naming while delegating to the Python implementation.

## Coding guidelines

Follow these principles:

- Search for existing functionality before implementing new behavior.
- Prefer extension of existing modules over duplicate helper logic.
- Use real Python names and package conventions, but keep MATLAB-style module names available through wrappers.
- Document non-obvious numerical assumptions and geometric conventions.
- Keep the structure aligned to the MATLAB repository to reduce migration overhead.

## Validation checklist

- Run the focused unit tests for the changed slice.
- Validate the affected reconstruction path with the smallest relevant tests.
- Run the full unit suite before closing a broader change set.
- Document exactly which checks were run and what passed.
