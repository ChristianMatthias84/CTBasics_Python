"""Location of the JP subset sweep results, shared by the sweep and its plot wrappers."""
from pathlib import Path

import numpy as np

from CTBasics.ReadAndWrite.IOHelperFunctions import IOHelperFunctions

SweepResultFileName = "JPHeartRateVsNSubsets_AllMetrics"


def SweepSearchRoot():
    """Folder below which StaticCT_JP_HeartRateVsNSubsets_Sweep writes its results."""
    return Path(IOHelperFunctions().GlobalOutputFolderPath) / "cardiacPhantom" / \
        "parameterSweeps" / "JPHeartRateVsNSubsets"


def SweepResultsNewestFirst():
    """All ``.npz`` sweep results (MATLAB: ``.mat``), newest first."""
    searchRoot = SweepSearchRoot()
    candidates = list(searchRoot.rglob(f"{SweepResultFileName}.npz")) \
        if searchRoot.is_dir() else []
    if not candidates:
        raise FileNotFoundError(f"No JP subset sweep result found below {searchRoot}.")
    return sorted(candidates, key=lambda candidate: candidate.stat().st_mtime, reverse=True)


def SweepVariesBoth(sweepPath, xParameterName, yParameterName):
    """True if the stored sweepTable has two distinct finite values for both parameters."""
    with np.load(sweepPath, allow_pickle=False) as candidateData:
        if "sweepTable" not in candidateData.files or \
                candidateData["sweepTable"].dtype.names is None:
            return False
        candidateTable = candidateData["sweepTable"]
    names = candidateTable.dtype.names
    if xParameterName not in names or yParameterName not in names:
        return False
    for name in (xParameterName, yParameterName):
        values = np.asarray(candidateTable[name], dtype=float)
        if np.unique(values[np.isfinite(values)]).size < 2:
            return False
    return True
