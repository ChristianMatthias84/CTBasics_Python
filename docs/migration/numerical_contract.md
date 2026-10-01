# Numerical Contract

This document starts from the MATLAB conventions used throughout CTBasics and will be refined as each ported slice is validated.

## Baseline assumptions

- MATLAB uses 1-based indexing; Python ports must document every index conversion.
- MATLAB arrays are column-major; reshape, linear indexing, and transpose behavior must be matched explicitly.
- Image, sinogram, and volume axis order must be recorded per function rather than assumed globally.
- Dtype conversions must be explicit, especially for `float32`, `float64`, integer detector data, and any complex-valued operations.

## To be validated per algorithm

- Coordinate origin and voxel/detector center definitions.
- Angle direction and angular periodicity.
- Interpolation, padding, extrapolation, and boundary handling.
- FFT scaling and filter normalization.
- `squeeze`, singleton dimensions, and row/column vector behavior.
- Missing-data behavior, NaN/Inf handling, and zero-intensity edge cases.
