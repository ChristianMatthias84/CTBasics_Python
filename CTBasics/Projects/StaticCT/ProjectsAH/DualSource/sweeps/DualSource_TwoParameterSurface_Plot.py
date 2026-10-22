"""Two sweep parameters against one metric as 3D surface and 2D heatmap.

Port of Projects/Static CT/Projects AH/DualSource/sweeps/DualSource_TwoParameterSurface_Plot.m.

``DualSource_TwoParameterSurface_Plot(sweepData, xParameterName, yParameterName,
metricName, **options)``; ``sweepData`` is a sweep table (dict of columns), a ``.csv`` /
``.npz`` result path, or ``None`` for the newest two-dimensional
``DualSourceHeartRateVsTRot_AllMetrics.npz`` (MATLAB: ``.mat``) below the sweep folder.

Duplicate parameter combinations are averaged, missing combinations stay NaN unless
InterpolateMissing is set. Writes ``<base>/Plots3D/<diagram>/<config>/<stem>_Surface*``
and ``<base>/PlotsHeatmap/<diagram>/<config>/<stem>_Heatmap*`` (``_Data.csv``, ``.png``)
and returns the plotted grid in long (column-major) format.
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import griddata

from CTBasics.Projects.StaticCT.StaticCTHelperFunctions import StaticCTVisualizationAndEvaluationHelpers
from CTBasics.Projects.StaticCT.ProjectsAH.DualSource.sweeps._SweepHelpers import (IsNumericColumn, NewFigure, RecordsToTable, SweepResultFileName,
                            SweepSearchRoot, iAxisLabel, iConstantNumber, iConstantText,
                            makeValidName, readtable, writetable)

ColorMap = "viridis"  # MATLAB parula is not available in matplotlib


def DualSource_TwoParameterSurface_Plot(sweepData=None, xParameterName="TRotSeconds",
                                        yParameterName="HeartRateBPM",
                                        metricName="PCA_Error_DualSource", *,
                                        InterpolateMissing=False, ShowDataPoints=True,
                                        CreateHeatmap=True, ShowHeatmapValues=True,
                                        ShowFigure=True, SaveFigure=True,
                                        ViewAzimuth=-37.5, ViewElevation=30,
                                        OutputDirectory=""):
    xParameterName = str(xParameterName)
    yParameterName = str(yParameterName)
    metricName = str(metricName)

    sourceTable, sourceDirectory = iLoadSweepTable(sweepData, xParameterName, yParameterName)
    selectedNames = [xParameterName, yParameterName, metricName]
    missingNames = sorted(set(selectedNames) - set(sourceTable))
    if missingNames:
        raise ValueError(f"Unknown sweep table column(s): {', '.join(missingNames)}")

    xValues = np.asarray(sourceTable[xParameterName])
    yValues = np.asarray(sourceTable[yParameterName])
    metricValues = np.asarray(sourceTable[metricName])
    if not (IsNumericColumn(xValues) and IsNumericColumn(yValues)
            and IsNumericColumn(metricValues)) or \
            xValues.ndim != 1 or yValues.ndim != 1 or metricValues.ndim != 1:
        raise ValueError("Both parameters and the metric must be numeric vectors.")

    xValues = xValues.astype(float)
    yValues = yValues.astype(float)
    metricValues = metricValues.astype(float)
    validRows = np.isfinite(xValues) & np.isfinite(yValues) & np.isfinite(metricValues)
    xValues = xValues[validRows]
    yValues = yValues[validRows]
    metricValues = metricValues[validRows]
    if metricValues.size == 0:
        raise ValueError("The selected columns contain no finite value triplets.")

    uniqueX = np.unique(xValues)
    uniqueY = np.unique(yValues)
    if uniqueX.size < 2 or uniqueY.size < 2:
        raise ValueError(
            "A surface plot requires at least two distinct values per parameter.")

    XGrid, YGrid = np.meshgrid(uniqueX, uniqueY)
    metricGrid = np.full(XGrid.shape, np.nan)
    sampleCountGrid = np.zeros(XGrid.shape)
    for iY in range(uniqueY.size):
        for iX in range(uniqueX.size):
            combinationRows = (xValues == uniqueX[iX]) & (yValues == uniqueY[iY])
            if np.any(combinationRows):
                metricGrid[iY, iX] = np.nanmean(metricValues[combinationRows])
                sampleCountGrid[iY, iX] = np.count_nonzero(combinationRows)

    if InterpolateMissing and np.any(np.isnan(metricGrid)):
        knownGridPoints = ~np.isnan(metricGrid)
        if np.count_nonzero(knownGridPoints) < 3:
            raise ValueError(
                "At least three valid grid points are required for interpolation.")
        # MATLAB uses griddata(..., 'natural'); scipy offers no natural neighbour method.
        metricGrid = griddata(
            (XGrid[knownGridPoints], YGrid[knownGridPoints]),
            metricGrid[knownGridPoints], (XGrid, YGrid), method="linear")

    # Return the exact plotted grid in long (column-major) format for reproducibility.
    surfaceTable = {
        xParameterName: XGrid.ravel(order="F"),
        yParameterName: YGrid.ravel(order="F"),
        metricName: metricGrid.ravel(order="F"),
        "NumberOfSamples": sampleCountGrid.ravel(order="F"),
    }

    surfaceFigure = NewFigure("%s and %s vs. %s" % (
        xParameterName, yParameterName, metricName), ShowFigure)
    axes = surfaceFigure.add_subplot(1, 1, 1, projection="3d")
    surfaceHandle = axes.plot_surface(
        XGrid, YGrid, metricGrid, cmap=ColorMap,
        edgecolor=(0.25, 0.25, 0.25), linewidth=0.6, label="Metric surface")
    if ShowDataPoints:
        axes.scatter(xValues, yValues, metricValues, s=42, c=metricValues, cmap=ColorMap,
                     norm=surfaceHandle.norm, edgecolors="k", depthshade=False,
                     label="Simulation results")
    metricColorbar = surfaceFigure.colorbar(surfaceHandle, ax=axes)
    metricColorbar.set_label(iAxisLabel(metricName))
    axes.set_xlabel(iAxisLabel(xParameterName))
    axes.set_ylabel(iAxisLabel(yParameterName))
    axes.set_zlabel(iAxisLabel(metricName))
    axes.set_title("%s versus %s and %s" % (
        metricName.replace("_", " "), xParameterName.replace("_", " "),
        yParameterName.replace("_", " ")))
    axes.grid(True)
    # MATLAB view(az, el) measures az from the negative y axis, matplotlib from x.
    axes.view_init(elev=ViewElevation, azim=ViewAzimuth - 90)
    if ShowDataPoints:
        axes.legend(loc="best")

    baseOutputDirectory = str(OutputDirectory)
    if len(baseOutputDirectory) == 0:
        baseOutputDirectory = sourceDirectory
    if len(baseOutputDirectory) == 0:
        baseOutputDirectory = str(SweepSearchRoot().parent / "DualSourceSurfacePlots")
    diagramLabel = "%s_vs_%s_and_%s" % (
        makeValidName(metricName), makeValidName(xParameterName),
        makeValidName(yParameterName))
    configurationLabel = iConfigurationLabel(
        sourceTable, xParameterName, uniqueX, yParameterName, uniqueY)
    outputDirectory = Path(baseOutputDirectory) / "Plots3D" / diagramLabel / configurationLabel
    outputDirectory.mkdir(parents=True, exist_ok=True)

    fileStem = "%s_%s_vs_%s_Surface" % (
        makeValidName(xParameterName), makeValidName(yParameterName),
        makeValidName(metricName))
    writetable(surfaceTable, outputDirectory / f"{fileStem}_Data.csv")
    if SaveFigure:
        # MATLAB additionally writes <stem>.fig (savefig); there is no matplotlib equivalent.
        StaticCTVisualizationAndEvaluationHelpers.StaticCT_exportFigure(
            surfaceFigure, str(outputDirectory / f"{fileStem}.png"), 300)
    if not ShowFigure:
        plt.close(surfaceFigure)

    if CreateHeatmap:
        heatmapFigure = iCreateHeatmapFigure(
            uniqueX, uniqueY, metricGrid, xParameterName, yParameterName,
            metricName, ShowHeatmapValues, ShowFigure)
        heatmapOutputDirectory = Path(baseOutputDirectory) / "PlotsHeatmap" / \
            diagramLabel / configurationLabel
        heatmapOutputDirectory.mkdir(parents=True, exist_ok=True)
        heatmapFileStem = "%s_%s_vs_%s_Heatmap" % (
            makeValidName(xParameterName), makeValidName(yParameterName),
            makeValidName(metricName))
        writetable(surfaceTable, heatmapOutputDirectory / f"{heatmapFileStem}_Data.csv")
        if SaveFigure:
            StaticCTVisualizationAndEvaluationHelpers.StaticCT_exportFigure(
                heatmapFigure, str(heatmapOutputDirectory / f"{heatmapFileStem}.png"), 300)
        if not ShowFigure:
            plt.close(heatmapFigure)
    return surfaceTable


def iCreateHeatmapFigure(uniqueX, uniqueY, metricGrid, xName, yName, metricName,
                         showValues, figureVisibility):
    # Display the simulated grid directly; NaN cells remain visibly empty. Every simulated
    # combination receives one equally sized cell; the (potentially non-equidistant)
    # parameter values are labels, not image coordinates.
    heatmapFigure = NewFigure("%s and %s heatmap: %s" % (xName, yName, metricName),
                              figureVisibility)
    axes = heatmapFigure.add_subplot(1, 1, 1)
    xPositions = np.arange(1, uniqueX.size + 1)
    yPositions = np.arange(1, uniqueY.size + 1)
    heatmapImage = axes.imshow(
        np.ma.masked_invalid(metricGrid), cmap=ColorMap, origin="lower", aspect="auto",
        interpolation="nearest",
        extent=(0.5, uniqueX.size + 0.5, 0.5, uniqueY.size + 0.5))
    axes.set_facecolor((0.85, 0.85, 0.85))
    axes.set_xticks(xPositions)
    axes.set_xticklabels(["%g" % value for value in uniqueX])
    axes.set_yticks(yPositions)
    axes.set_yticklabels(["%g" % value for value in uniqueY])
    axes.set_xlim(0.5, uniqueX.size + 0.5)
    axes.set_ylim(0.5, uniqueY.size + 0.5)
    metricColorbar = heatmapFigure.colorbar(heatmapImage, ax=axes)
    metricColorbar.set_label(iAxisLabel(metricName))
    axes.set_xlabel(iAxisLabel(xName))
    axes.set_ylabel(iAxisLabel(yName))
    axes.set_title("%s versus %s and %s" % (
        metricName.replace("_", " "), xName.replace("_", " "), yName.replace("_", " ")))

    if showValues:
        finiteValues = metricGrid[np.isfinite(metricGrid)]
        colorThreshold = finiteValues.min() + (finiteValues.max() - finiteValues.min()) / 2
        for iY in range(uniqueY.size):
            for iX in range(uniqueX.size):
                value = metricGrid[iY, iX]
                if not np.isfinite(value):
                    continue
                textColor = "w" if value < colorThreshold else "k"
                axes.text(xPositions[iX], yPositions[iY], "%.4g" % value,
                          horizontalalignment="center", verticalalignment="center",
                          color=textColor, fontweight="bold")
    return heatmapFigure


def iLoadSweepTable(sweepData, xParameterName, yParameterName):
    # Load table or default dual-source result and retain its source directory.
    sourceDirectory = ""
    if isinstance(sweepData, dict):
        return sweepData, sourceDirectory

    if sweepData is None:
        sweepData = iFindLatestSweepResult(xParameterName, yParameterName)
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
            fieldNames = loadedData.files
            if "sweepTable" in fieldNames:
                fieldNames = ["sweepTable"] + [name for name in fieldNames
                                               if name != "sweepTable"]
            sweepTable = None
            for fieldName in fieldNames:
                candidate = loadedData[fieldName]
                if candidate.dtype.names is not None:
                    sweepTable = RecordsToTable(candidate)
                    break
        if sweepTable is None:
            raise ValueError("The NPZ file does not contain a sweep result table.")
    else:
        raise ValueError("Only CSV and NPZ sweep result files are supported.")
    return sweepTable, sourceDirectory


def iFindLatestSweepResult(xParameterName, yParameterName):
    # Find the newest result that actually contains a two-dimensional grid. The latest
    # sweep may be a one-parameter study and is unsuitable for a surface even though it
    # contains both parameter columns as metadata.
    searchRoot = SweepSearchRoot()
    candidates = list(searchRoot.rglob(f"{SweepResultFileName}.npz")) \
        if searchRoot.is_dir() else []
    if not candidates:
        raise FileNotFoundError(f"No dual-source sweep result found below {searchRoot}.")

    candidates.sort(key=lambda candidate: candidate.stat().st_mtime, reverse=True)
    for candidatePath in candidates:
        with np.load(candidatePath, allow_pickle=False) as candidateData:
            if "sweepTable" not in candidateData.files or \
                    candidateData["sweepTable"].dtype.names is None:
                continue
            candidateTable = candidateData["sweepTable"]

        candidateNames = candidateTable.dtype.names
        if xParameterName not in candidateNames or yParameterName not in candidateNames:
            continue

        xValues = np.asarray(candidateTable[xParameterName], dtype=float)
        yValues = np.asarray(candidateTable[yParameterName], dtype=float)
        if np.unique(xValues[np.isfinite(xValues)]).size >= 2 and \
                np.unique(yValues[np.isfinite(yValues)]).size >= 2:
            return candidatePath

    raise FileNotFoundError(
        f"No sweep with at least two distinct values for both {xParameterName} and "
        f"{yParameterName} was found below {searchRoot}. Run a two-parameter sweep first.")


def iConfigurationLabel(sourceTable, xName, xValues, yName, yValues):
    phantom = iConstantText(sourceTable, "SelectedPhantom", "UnknownPhantom")
    label = "Ph_%s_%s_%g-%g_n%d_%s_%g-%g_n%d" % (
        makeValidName(phantom),
        makeValidName(xName), np.min(xValues), np.max(xValues), np.size(xValues),
        makeValidName(yName), np.min(yValues), np.max(yValues), np.size(yValues))
    metadataColumns = ["Nsources", "NSubsets", "Npar", "N2pi",
                       "DetectorElements", "ImageSize", "NRecons", "NScansPerRecon",
                       "TRotSeconds", "PhotonCount", "SimulateNoise", "RandomSeed"]
    metadataLabels = ["NS", "NSub", "Npar", "N2pi", "M", "Nx",
                      "NRecon", "NScans", "TRot", "I0", "Noise", "Seed"]
    for column, columnLabel in zip(metadataColumns, metadataLabels):
        value = iConstantNumber(sourceTable, column, np.nan)
        isSurfaceParameter = column == xName or column == yName
        if np.isfinite(value) and not isSurfaceParameter:
            label = "%s_%s_%g" % (label, columnLabel, value)
    return makeValidName(label)


main = DualSource_TwoParameterSurface_Plot
