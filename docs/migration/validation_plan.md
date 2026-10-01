# Validation Plan

## M0

- Capture the MATLAB repository commit and working-tree state.
- Generate a SHA-256 manifest of the MATLAB source tree outside git metadata.
- Record the discovered external raw-data dependencies.
- Establish the isolated MATLAB reference strategy before any parity run.

## M1 and beyond

- Start with small synthetic reference cases before patient or scanner data.
- Validate the 2D pipeline stepwise: phantom -> projections -> preprocessing -> rebinning -> filtering -> reconstruction.
- Compare shapes, dtypes, finite-value behavior, and numerical error metrics for each stage.
- Store reference inputs, outputs, and generation metadata under `validation/matlab_reference/`.
