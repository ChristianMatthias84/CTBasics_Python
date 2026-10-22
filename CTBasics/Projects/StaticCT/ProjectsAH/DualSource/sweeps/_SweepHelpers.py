"""Shared table, naming and label helpers of the DualSource sweep scripts.

MATLAB duplicates iConstantNumber, iConstantText and iAxisLabel as identical local functions
in DualSource_ParameterVsMetric_Plot.m and DualSource_TwoParameterSurface_Plot.m; the port
keeps one copy here under the same names.

MATLAB tables are represented as dicts of equally long numpy column arrays (insertion order
is the column order). ``writetable`` / ``readtable`` mimic MATLAB's CSV round trip:
doubles with 15 significant digits, logicals as 1/0, NaN as ``NaN``; reading turns every
fully numeric column into float64 and keeps the others as text.
"""
import csv
import math
import re
from pathlib import Path

import numpy as np

from CTBasics.ReadAndWrite.IOHelperFunctions import IOHelperFunctions

SweepResultFileName = "DualSourceHeartRateVsTRot_AllMetrics"


def SweepSearchRoot():
    """Folder below which the HeartRate-vs-TRot sweep writes its results."""
    return Path(IOHelperFunctions().GlobalOutputFolderPath) / "cardiacPhantom" / \
        "parameterSweeps" / "DualSourceHeartRateVsTRot"


def MatlabColon(a, d, b):
    """MATLAB ``a:d:b`` with its symmetric evaluation (exact end point, no drift)."""
    n = math.floor((b - a) / d + 1e-10)
    values = np.empty(n + 1)
    half = n // 2
    k = np.arange(half + 1)
    values[k] = a + k * d
    values[n - k] = (a + n * d) - k * d
    if n % 2 == 0:
        values[half] = (a + (a + n * d)) / 2
    return values


def makeValidName(name):
    """matlab.lang.makeValidName for plain labels (keywords are not special-cased)."""
    name = re.sub(r"\s+([a-z])", lambda match: match.group(1).upper(), str(name))
    name = re.sub(r"\s", "", name)
    name = re.sub(r"[^A-Za-z0-9_]", "_", name)
    if not re.match(r"[A-Za-z]", name):
        name = "x" + name
    return name[:63]


def NumRows(table):
    return len(next(iter(table.values()))) if table else 0


def SelectRows(table, rows):
    return {name: np.asarray(column)[rows] for name, column in table.items()}


def IsNumericColumn(column):
    """MATLAB isnumeric: logical and text columns are not numeric."""
    return np.issubdtype(np.asarray(column).dtype, np.number)


def TableToRecords(table):
    """Table -> structured array, so .npz files load without pickle."""
    columns = {name: np.asarray(column) for name, column in table.items()}
    records = np.empty(NumRows(table), dtype=[(name, column.dtype)
                                              for name, column in columns.items()])
    for name, column in columns.items():
        records[name] = column
    return records


def RecordsToTable(records):
    return {name: np.asarray(records[name]) for name in records.dtype.names}


def _CsvValue(value):
    if isinstance(value, (bool, np.bool_)):
        return "1" if value else "0"
    if isinstance(value, (str, np.str_)):
        return str(value)
    value = float(value)
    if math.isnan(value):
        return "NaN"
    if math.isinf(value):
        return "Inf" if value > 0 else "-Inf"
    return "%.15g" % value


def writetable(table, path):
    with open(path, "w", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(list(table))
        for row in zip(*table.values()):
            writer.writerow([_CsvValue(value) for value in row])


def readtable(path):
    with open(path, newline="") as handle:
        rows = list(csv.reader(handle))
    header, body = rows[0], rows[1:]
    table = {}
    for column, name in enumerate(header):
        values = [row[column] if column < len(row) else "" for row in body]
        try:
            table[name] = np.array([float(value) if value.strip() else math.nan
                                    for value in values])
        except ValueError:
            table[name] = np.array(values, dtype=str)
    return table


def _ToDouble(value):
    """MATLAB double(string): numeric text converts, everything else becomes NaN."""
    try:
        return float(value)
    except ValueError:
        return math.nan


def iConstantNumber(sourceTable, variableName, fallback):
    if variableName in sourceTable:
        column = np.asarray(sourceTable[variableName])
        if column.dtype.kind in "US":
            column = np.array([_ToDouble(value) for value in column])
        values = np.unique(column.astype(float), equal_nan=False)
        if values.size == 1:
            return float(values[0])
    return fallback


def iConstantText(sourceTable, variableName, fallback):
    if variableName in sourceTable:
        values = np.unique([str(value) for value in np.asarray(sourceTable[variableName])])
        if values.size == 1:
            return str(values[0])
    return fallback


def iAxisLabel(variableName):
    """Readable axis labels; MATLAB's TeX ``T_{Rot}`` becomes matplotlib mathtext."""
    labels = {
        "HeartRateBPM": "Heart rate [bpm]",
        "TRotSeconds": "Gantry rotation time $T_{Rot}$ [s]",
        "Nsources": "Number of sources",
        "NSubsets": "Number of subsets",
        "PCA_Error_DualSource": "PCA combined dual-source error [0 = best, 1 = worst]",
        "PCA_Error_Linear": "PCA combined linear error [0 = best, 1 = worst]",
        "PCA_Error_Arbitrary": "PCA combined arbitrary error [0 = best, 1 = worst]",
        "PCA_Error_ArbitraryMinusLinear":
            "PCA error difference [positive = arbitrary worse]",
    }
    variableName = str(variableName)
    if variableName in labels:
        return labels[variableName]
    readableName = variableName.replace("_", " ")
    if variableName.startswith("RMSE_"):
        return f"{readableName} [HU; lower = better]"
    if variableName.startswith(("NCC_", "SSIM_")):
        return f"{readableName} [higher = better]"
    return readableName


def NewFigure(name, visible):
    """MATLAB figure('Name', name, 'Color', 'w', 'Visible', ...)."""
    import matplotlib.pyplot as plt

    figure = plt.figure(facecolor="w")
    figure.set_label(name)
    if visible and figure.canvas.manager is not None:
        figure.canvas.manager.set_window_title(name)
    return figure
