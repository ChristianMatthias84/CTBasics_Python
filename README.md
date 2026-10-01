# CTBasics Python Port

This repository contains the Python implementation of the CTBasics simulation and reconstruction framework for the non-external raw-data portions of the MATLAB codebase.

## Final status

The core Python port is functionally complete for the non-raw-data domain layers, including:

- scan descriptors and geometry configuration
- phantom creation and motion/spectral helpers
- forward projection and fan-to-parallel rebinning
- convolution, weighting, and backprojection
- preprocessing and MAR-style helpers
- visualization and project-level helper modules

The only intentionally deferred area is the Siemens scanner raw-data reader path, which depends on external toolbox data and remains out of scope for this port.

## Project layout

The real implementation lives in one package tree under `src/ctbasics/`. This follows the MATLAB domain structure logically while keeping a standard Python package layout and avoiding duplicate wrapper trees.

## Installation

```bash
python -m pip install -e .
```

## Validation

The project is validated with the unit suite:

```bash
python -m pytest -q tests/unit
```

The current verified result is: `36 passed`.
