# CTBasics Python Port

This repository contains the Python implementation of the CTBasics simulation and reconstruction framework for the non-external raw-data portions of the MATLAB codebase.

## Status

The repository layout mirrors the MATLAB library one-to-one, but the **functional port is still in progress**: roughly 15 % of the MATLAB code lines are implemented. Everything that is not ported yet raises `NotImplementedError` instead of returning a plausible-looking wrong result, so unported paths fail loudly.

Ported and numerically validated:

- reconstruction kernels (Ram-Lak, Shepp-Logan, Hann, Moire, sinc-square, frequency-domain filtering)
- 2D ray-driven forward projection for parallel, fan, and static CT geometries
- fan-to-parallel rebinning
- 2D pixel-driven backprojection and weighted FBP
- static CT z-collimation including the leaf collimator variant
- scan descriptors, geometry configuration, and memory/precision helpers

Validation is numerical, not structural: the forward projector is checked against the analytic chord length of a disk, and the full filter-backproject chain recovers the disk attenuation to within 2 %.

Not ported yet (main gaps):

- real phantom geometry in `CreatePhantom` (ellipse-based phantoms, motion and cardiac variants)
- `Convolution`
- all 3D, spiral, and sequence variants of projection, rebinning, and backprojection
- spectral and motion forward projection paths
- `LecturesAndTestScripts/`, `Scripts/`, and `Projects/` content

The Siemens scanner raw-data reader path is intentionally deferred because it depends on external toolbox data.

## Project layout

The implementation lives in a single package tree under `CTBasics/` whose folder and file names mirror the MATLAB library one-to-one. Every MATLAB `classdef` file becomes one Python module of the same name containing one class of the same name, and the MATLAB method names are kept verbatim.

```
CTBasics/
  Apps/            Benchmarks/        Data/
  Descriptors/     ImageManipulation/ LecturesAndTestScripts/
  Memory/          MultiCore/         Preprocessing/
  Projects/        RawData/           ReadAndWrite/
  Reconstruction/  Scripts/           Simulation/
  Visualization/
```

Imports therefore read like the MATLAB calls:

```python
from CTBasics.Reconstruction import Backprojection, ForwardProjection
from CTBasics.Simulation import CreatePhantom
```

The only structural deviation is `Projects/StaticCT/` (MATLAB: `Projects/Static CT/`), because Python package names cannot contain spaces.

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
