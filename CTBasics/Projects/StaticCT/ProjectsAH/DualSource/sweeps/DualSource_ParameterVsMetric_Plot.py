"""Exact one-dimensional slice of DualSource sweep data.

Port of Projects/Static CT/Projects AH/DualSource/sweeps/DualSource_ParameterVsMetric_Plot.m.

``DualSource_ParameterVsMetric_Plot(sweepData, parameterName, metricName, **options)``
plots ``metricName`` over ``parameterName`` at exactly ``FixedParameterName =
FixedParameterValue``; no averaging is performed. ``sweepData`` is a sweep table (dict of
columns), a ``.csv`` / ``.npz`` result path, or ``None`` for the newest
``DualSourceHeartRateVsTRot_AllMetrics.npz`` (MATLAB: ``.mat``) below the sweep folder.

Writes ``<base>/Plots2D/<metric>_vs_<parameter>/<config>/<stem>_Data.csv`` and, with
SaveFigure, ``<stem>.png``. Returns the plotted rows as a table.
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from CTBasics.Projects.StaticCT.StaticCTHelperFunctions import StaticCTVisualizationAndEvaluationHelpers
from CTBasics.Projects.StaticCT.ProjectsAH.DualSource.sweeps._SweepHelpers import (IsNumericColumn, NewFigure, NumRows, RecordsToTable,
                            SelectRows, SweepResultFileName, SweepSearchRoot, iAxisLabel,
                            iConstantNumber, iConstantText, makeValidName, readtable,
                            writetable)


def DualSource_ParameterVsMetric_Plot(sweepData=None, parameterName="TRotSeconds",
                                      metricName="RMSE_DualSource", *,
                                      FixedParameterName="HeartRateBPM",
                                      FixedParameterValue=90, MatchTolerance=1e-10,
                                      ShowFigure=True, SaveFigure=True,
                                      OutputDirectory=""):
    parameterName = str(parameterName)
    metricName = str(metricName)
    FixedParameterName = str(FixedParameterName)
    FixedParameterValue = float(FixedParameterValue)
    if MatchTolerance < 0:
        raise ValueError("MatchTolerance must be nonnegative.")

    sourceTable, sourceDirectory = iLoadSweepTable(sweepData)
    selectedNames = [parameterName, metricName, FixedParameterName]
    missingNames = sorted(set(selectedNames) - set(sourceTable))
    if missingNames:
        raise ValueError(f"Unknown sweep table column(s): {', '.join(missingNames)}")
    if parameterName == FixedParameterName:
        raise ValueError("The plotted and fixed parameters must be different columns.")

    parameterValues = np.asarray(sourceTable[parameterName])
    metricValues = np.asarray(sourceTable[metricName])
    fixedValues = np.asarray(sourceTable[FixedParameterName])
    if not (IsNumericColumn(parameterValues) and IsNumericColumn(metricValues)
            and IsNumericColumn(fixedValues)):
        raise ValueError(
            "The plotted parameter, fixed parameter, and metric must be numeric.")

    fixedRows = np.abs(fixedValues - FixedParameterValue) <= MatchTolerance
    plotTable = SelectRows(sourceTable, fixedRows)
    if NumRows(plotTable) == 0:
        availableValues = np.unique(fixedValues)
        raise ValueError("No data for %s = %g. Available values: %s" % (
            FixedParameterName, FixedParameterValue,
            ", ".join("%g" % value for value in availableValues)))

    parameterValues = np.asarray(plotTable[parameterName], dtype=float)
    metricValues = np.asarray(plotTable[metricName], dtype=float)
    validRows = np.isfinite(parameterValues) & np.isfinite(metricValues)
    plotTable = SelectRows(plotTable, validRows)
    parameterValues = parameterValues[validRows]
    metricValues = metricValues[validRows]
    sortOrder = np.argsort(parameterValues, kind="stable")
    parameterValues = parameterValues[sortOrder]
    metricValues = metricValues[sortOrder]
    plotTable = SelectRows(plotTable, sortOrder)

    # Every x value must identify one simulation after fixing the second sweep parameter.
    # Multiple rows would imply another uncontrolled variable or repeated simulations and
    # must not be silently averaged.
    if np.unique(parameterValues).size != parameterValues.size:
        raise ValueError(
            f"The fixed slice still contains duplicate {parameterName} values. Filter the "
            "remaining varying parameter explicitly before plotting.")

    plotFigure = NewFigure("%s vs. %s at %s = %g" % (
        parameterName, metricName, FixedParameterName, FixedParameterValue), ShowFigure)
    axes = plotFigure.add_subplot(1, 1, 1)
    lineColor = (0.00, 0.32, 0.67)
    axes.plot(parameterValues, metricValues, "-o", color=lineColor,
              markerfacecolor=lineColor, linewidth=1.8)
    axes.set_xlabel(iAxisLabel(parameterName))
    axes.set_ylabel(iAxisLabel(metricName))
    axes.set_title("%s vs. %s at %s = %g" % (
        metricName.replace("_", " "), parameterName.replace("_", " "),
        FixedParameterName.replace("_", " "), FixedParameterValue))
    axes.grid(True)

    baseOutputDirectory = str(OutputDirectory)
    if len(baseOutputDirectory) == 0:
        baseOutputDirectory = sourceDirectory
    configurationLabel = iConfigurationLabel(plotTable, FixedParameterName,
                                             FixedParameterValue)
    diagramLabel = "%s_vs_%s" % (makeValidName(metricName), makeValidName(parameterName))
    outputDirectory = Path(baseOutputDirectory) / "Plots2D" / diagramLabel / configurationLabel
    outputDirectory.mkdir(parents=True, exist_ok=True)

    fileStem = "%s_vs_%s_at_%s_%g" % (
        makeValidName(metricName), makeValidName(parameterName),
        makeValidName(FixedParameterName), FixedParameterValue)
    writetable(plotTable, outputDirectory / f"{fileStem}_Data.csv")
    if SaveFigure:
        # MATLAB additionally writes <stem>.fig (savefig); there is no matplotlib equivalent.
        StaticCTVisualizationAndEvaluationHelpers.StaticCT_exportFigure(
            plotFigure, str(outputDirectory / f"{fileStem}.png"), 300)
    if not ShowFigure:
        plt.close(plotFigure)
    return plotTable


def iLoadSweepTable(sweepData):
    if isinstance(sweepData, dict):
        return sweepData, str(SweepSearchRoot())
    if sweepData is None:
        sweepData = iFindLatestSweepResult()
    if not isinstance(sweepData, (str, Path)):
        raise TypeError("sweepData must be a table or a CSV/NPZ file path.")
    sweepPath = Path(sweepData)
    if not sweepPath.is_file():
        raise FileNotFoundError(f"Sweep result file not found: {sweepPath}")
    sourceDirectory = str(sweepPath.parent)
    extension = sweepPath.suffix.lower()
    if extension == ".csv":
        sweepTable = readtable(sweepPath)
    elif extension == ".npz":
        with np.load(sweepPath, allow_pickle=False) as loadedData:
            if "sweepTable" not in loadedData.files or \
                    loadedData["sweepTable"].dtype.names is None:
                raise ValueError("The NPZ file does not contain sweepTable.")
            sweepTable = RecordsToTable(loadedData["sweepTable"])
    else:
        raise ValueError("Only CSV and NPZ sweep result files are supported.")
    return sweepTable, sourceDirectory


def iFindLatestSweepResult():
    searchRoot = SweepSearchRoot()
    candidates = list(searchRoot.rglob(f"{SweepResultFileName}.npz")) \
        if searchRoot.is_dir() else []
    if not candidates:
        raise FileNotFoundError(f"No dual-source sweep result found below {searchRoot}.")
    return max(candidates, key=lambda candidate: candidate.stat().st_mtime)


def iConfigurationLabel(sourceTable, fixedName, fixedValue):
    phantom = iConstantText(sourceTable, "SelectedPhantom", "UnknownPhantom")
    label = "Ph_%s_%s_%g" % (makeValidName(phantom), makeValidName(fixedName), fixedValue)
    metadataColumns = ["Nsources", "NSubsets", "Npar", "N2pi",
                       "DetectorElements", "ImageSize", "NRecons", "NScansPerRecon",
                       "TRotSeconds", "PhotonCount", "SimulateNoise", "RandomSeed"]
    metadataLabels = ["NS", "NSub", "Npar", "N2pi", "M", "Nx",
                      "NRecon", "NScans", "TRot", "I0", "Noise", "Seed"]
    for column, columnLabel in zip(metadataColumns, metadataLabels):
        value = iConstantNumber(sourceTable, column, np.nan)
        if np.isfinite(value) and column != fixedName:
            label = "%s_%s_%g" % (label, columnLabel, value)
    return makeValidName(label)


main = DualSource_ParameterVsMetric_Plot
