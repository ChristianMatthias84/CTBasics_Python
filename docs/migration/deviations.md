# Deviations

## Observed MATLAB-side dependencies and gaps

- `readXACB`, `readTable`, and `xml2struct` are not implemented in the repository tree and appear to come from external tooling.
- `init_CTBasics.m` hard-depends on the external TFS path when raw-data projects are enabled.

## Pending decisions

- Whether each external dependency will be reimplemented natively in Python, wrapped through a transitional bridge, or marked as blocked.
- Which scripts require dedicated reference data before any parity comparison is meaningful.
