# Dependencies

## Internal MATLAB dependencies

- `ReadRawDataFunctions.m` depends on `readXACB`, `readTable`, and `xml2struct`.
- `init_CTBasics.m` depends on `CTBasicsConfig.txt` in the MATLAB user path and on the external TFS CT MATLAB toolbox path in line 6 of that config file.
- `RunAllScripts.m` chains lecture scripts and Static CT project scripts.

## External dependencies observed so far

- `readXACB`: unresolved in the repository, likely provided by the external TFS/CT MATLAB toolbox or a MEX wrapper.
- `readTable`: unresolved in the repository, likely provided externally with the raw-data toolchain.
- `xml2struct`: unresolved in the repository, likely provided by the external MATLAB toolbox.
- TFS `general`, `mex`, `mexSources`, and `MatlabDicom` folders are added by `init_CTBasics.m` when raw-data projects are enabled.

## Open decisions

- Determine which external dependencies can be replaced in Python and which remain blocked by proprietary binary interfaces or missing data.
- Confirm the supported Python package versions and lock strategy before the first functional slice is ported.
