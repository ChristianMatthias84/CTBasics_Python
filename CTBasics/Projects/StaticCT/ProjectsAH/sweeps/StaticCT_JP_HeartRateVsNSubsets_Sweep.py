"""Heart rate, subset count and rotation time sweep of the JP subset acquisition.

Port of Projects/Static CT/Projects AH/sweeps/StaticCT_JP_HeartRateVsNSubsets_Sweep.m.

MATLAB is a function ``sweepTable = StaticCT_JP_HeartRateVsNSubsets_Sweep(heartRatesBPM,
nSubsetsValues, options)``; the name-value options become keyword-only arguments with the
same names and defaults. Only three two-parameter planes are simulated, the third
parameter of each plane is held at its ``Fixed*`` value. The table is returned as a dict of
column arrays (see DualSource/sweeps/_SweepHelpers).

Outputs (below ``<Output>/cardiacPhantom/parameterSweeps/JPHeartRateVsNSubsets/<config>/
<grid>``): ``JPHeartRateVsNSubsets_AllMetrics.csv``, ``..._PhaseROIMetrics.csv``,
``..._LinearVsArbitrary.csv``, ``JPHeartRateVsNSubsets_AllMetrics.npz`` in place of MATLAB's
``.mat``, ``PairwiseMatrices/``, ``PairwisePlots/`` and ``SimulationOverview/`` (PNG only,
matplotlib has no ``.fig``).
"""
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from CTBasics.Simulation.CreatePhantom import CreatePhantom
from CTBasics.Projects.StaticCT.StaticCTHelperFunctions import StaticCTVisualizationAndEvaluationHelpers
from CTBasics.Projects.StaticCT.ProjectsAH.DualSource.sweeps.DualSource_HeartRateVsTRot_Sweep import pca
from CTBasics.Projects.StaticCT.ProjectsAH.DualSource.sweeps._SweepHelpers import (NewFigure, SelectRows, TableToRecords,
                                                                                   makeValidName, writetable)
from CTBasics.Projects.StaticCT.ProjectsAH.sweeps.StaticCT_4_Cardiac_JP import StaticCT_4_Cardiac_JP
from CTBasics.Projects.StaticCT.ProjectsAH.sweeps.StaticCT_JP_ParameterVsMetric_Plot import StaticCT_JP_ParameterVsMetric_Plot
from CTBasics.Projects.StaticCT.ProjectsAH.sweeps.StaticCT_JP_TwoParameterSurface_Plot import StaticCT_JP_TwoParameterSurface_Plot
from CTBasics.Projects.StaticCT.ProjectsAH.sweeps._JPSweepHelpers import SweepResultFileName, SweepSearchRoot

metricNames = ["RMSE_Linear", "RMSE_Arbitrary", "NCC_Linear", "NCC_Arbitrary",
               "SSIM_Linear", "SSIM_Arbitrary"]

DefaultMetricsToPlot = ("PCA_Error_Linear", "PCA_Error_Arbitrary",
                        "PCA_Error_ArbitraryMinusLinear", "RMSE_Arbitrary",
                        "NCC_Arbitrary", "SSIM_Arbitrary")


def StaticCT_JP_HeartRateVsNSubsets_Sweep(
        heartRatesBPM=(60, 120), nSubsetsValues=(8, 128), *, TRot=(0.1, 0.5),
        FixedTRotForHeartRateVsNSubsets=0.2, FixedNSubsetsForHeartRateVsTRot=32,
        FixedHeartRateForNSubsetsVsTRot=80, NRecons=1, NScansPerRecon=1, N2pi=1024,
        DetectorElements=500, ImageSize=512, PhotonCount=1e7, RandomSeed=1,
        SelectedPhantom="Cardiac", RestFraction=0.5, ReconstructionPhaseFraction=0.25,
        ApplyMoireFilter=False, SimulateNoise=False, SavePerRunResults=False,
        ShowReconstructions=False, UseLinearCache=True, ShowPlots=True, SavePlots=True,
        CreateHeatmaps=True, CreateSimulationOverview=True,
        OverviewHeartRateBPM=math.nan, OverviewNSubsets=math.nan,
        OverviewTRot=math.nan, OverviewReconstructionType="Linear",
        MetricsToPlot=DefaultMetricsToPlot):
    """Runs the three requested parameter planes and exports metrics and plots."""
    heartRatesBPM = np.atleast_1d(np.asarray(heartRatesBPM, dtype=float))
    nSubsetsValues = np.atleast_1d(np.asarray(nSubsetsValues, dtype=float))
    rotationTimes = np.atleast_1d(np.asarray(TRot, dtype=float))
    if np.any(heartRatesBPM <= 0) or np.any(nSubsetsValues <= 0) or \
            np.any(rotationTimes <= 0):
        raise ValueError("heartRatesBPM, nSubsetsValues and TRot must be positive.")
    if RestFraction >= 1:
        raise ValueError(
            "RestFraction must be greater than or equal to 0 and smaller than 1.")
    if ReconstructionPhaseFraction >= 1:
        raise ValueError("ReconstructionPhaseFraction must be greater than or equal to "
                         "0 and smaller than 1.")
    if np.any(nSubsetsValues > N2pi):
        raise ValueError("Every NSubsets value must be less than or equal to N2pi.")
    SelectedPhantom = str(SelectedPhantom)
    MetricsToPlot = [str(name) for name in MetricsToPlot]

    # Fixed values are independent scan settings and need not occur in the varied vectors.
    fixedTRot = float(FixedTRotForHeartRateVsNSubsets)
    fixedNSubsets = float(FixedNSubsetsForHeartRateVsTRot)
    fixedHeartRate = float(FixedHeartRateForNSubsetsVsTRot)

    overviewHeartRate = iSelectSweepValue(heartRatesBPM, OverviewHeartRateBPM,
                                          "OverviewHeartRateBPM", fixedHeartRate)
    overviewNSubsets = iSelectSweepValue(nSubsetsValues, OverviewNSubsets,
                                         "OverviewNSubsets", fixedNSubsets)
    overviewTRot = iSelectSweepValue(rotationTimes, OverviewTRot, "OverviewTRot",
                                     fixedTRot)

    runGrid = iBuildPairwiseRunGrid(heartRatesBPM, nSubsetsValues, rotationTimes,
                                    fixedTRot, fixedNSubsets, fixedHeartRate)
    if CreateSimulationOverview:
        runGrid = iUniqueRowsStable(np.vstack((
            runGrid, [overviewHeartRate, overviewNSubsets, overviewTRot])))
    heartRateColumn = runGrid[:, 0]
    subsetColumn = runGrid[:, 1]
    rotationTimeColumn = runGrid[:, 2]
    nRuns = runGrid.shape[0]
    if nRuns < 2:
        raise ValueError("At least two parameter combinations are required for PCA.")

    metricValues = np.full((nRuns, len(metricNames)), np.nan)
    detailedTables = []
    overviewRun = np.flatnonzero((heartRateColumn == overviewHeartRate)
                                 & (subsetColumn == overviewNSubsets)
                                 & (rotationTimeColumn == overviewTRot))
    overviewRun = overviewRun[0] if overviewRun.size else None
    overviewResult = None

    print(f"Starting JP HeartRate-vs-NSubsets-vs-TRot sweep with {nRuns} simulations.")
    for iRun in range(nRuns):
        heartRate = heartRateColumn[iRun]
        nSubsets = int(subsetColumn[iRun])
        rotationTime = rotationTimeColumn[iRun]
        captureOverview = bool(CreateSimulationOverview) and iRun == overviewRun
        print(f"[{iRun + 1}/{nRuns}] Heart rate = {heartRate:.0f} bpm, "
              f"NSubsets = {nSubsets}, TRot = {rotationTime:.3f} s")

        result = StaticCT_4_Cardiac_JP(
            heartRate, rotationTime, nSubsets,
            NRecons=NRecons, NScansPerRecon=NScansPerRecon, N2pi=N2pi,
            DetectorElements=DetectorElements, ImageSize=ImageSize,
            PhotonCount=PhotonCount, RandomSeed=RandomSeed,
            SelectedPhantom=SelectedPhantom, RestFraction=RestFraction,
            ReconstructionPhaseFraction=ReconstructionPhaseFraction,
            ApplyMoireFilter=ApplyMoireFilter, SimulateNoise=SimulateNoise,
            ShowPlots=ShowReconstructions, WriteData=SavePerRunResults,
            UseLinearCache=UseLinearCache, IncludeImageData=captureOverview)

        if captureOverview:
            overviewResult = result
        for iMetric, metricName in enumerate(metricNames):
            metricValues[iRun, iMetric] = result["Metrics"][metricName]
        detailedTables.append(iBuildDetailedMetricTable(result, heartRate, rotationTime,
                                                        nSubsets))

    restFraction = np.full(nRuns, float(RestFraction))
    phaseFraction = np.full(nRuns, float(ReconstructionPhaseFraction))
    sweepTable = {
        "HeartRateBPM": heartRateColumn,
        "NSubsets": subsetColumn,
        "TRotSeconds": rotationTimeColumn,
        "NRecons": np.full(nRuns, float(NRecons)),
        "NScansPerRecon": np.full(nRuns, float(NScansPerRecon)),
        "N2pi": np.full(nRuns, float(N2pi)),
        "DetectorElements": np.full(nRuns, float(DetectorElements)),
        "ImageSize": np.full(nRuns, float(ImageSize)),
        "PhotonCount": np.full(nRuns, float(PhotonCount)),
        "RandomSeed": np.full(nRuns, float(RandomSeed)),
        "SelectedPhantom": np.full(nRuns, SelectedPhantom),
        "RestFraction": restFraction,
        "RestDurationSeconds": restFraction * (60 / heartRateColumn),
        "ReconstructionPhaseFraction": phaseFraction,
        "FirstReconstructionTimeSeconds": phaseFraction * (60 / heartRateColumn),
        "SimulateNoise": np.full(nRuns, bool(SimulateNoise)),
        "ApplyMoireFilter": np.full(nRuns, bool(ApplyMoireFilter)),
    }
    for iMetric, metricName in enumerate(metricNames):
        sweepTable[metricName] = metricValues[:, iMetric]
    phaseRoiMetricTable = {name: np.concatenate([table[name] for table in detailedTables])
                           for name in detailedTables[0]}

    # One PCA for the paired linear and arbitrary results: both acquisition types share
    # standardization, PC1 direction and 0-1 normalization and are directly comparable.
    (sweepTable["PCA_Error_Linear"], sweepTable["PCA_Error_Arbitrary"], pcaInfo,
     acquisitionComparisonTable) = iComputeJointPcaErrors(sweepTable)
    sweepTable["PCA_Error_ArbitraryMinusLinear"] = \
        sweepTable["PCA_Error_Arbitrary"] - sweepTable["PCA_Error_Linear"]

    # Self-describing folders keep different scan configurations and grids apart.
    scanConfigurationFolder = (
        "Ph_%s_N2pi_%d_M_%d_Nx_%d_NRecon_%d_NScans_%d_I0_%g_RestFrac_%g_ReconPhase_%g"
        "_Noise_%d_Moire_%d_Seed_%d" % (
            makeValidName(SelectedPhantom), N2pi, DetectorElements, ImageSize, NRecons,
            NScansPerRecon, PhotonCount, RestFraction, ReconstructionPhaseFraction,
            SimulateNoise, ApplyMoireFilter, RandomSeed))
    sweepGridFolder = (
        "HR_%g-%g_n%d_NSub_%g-%g_n%d_TRotms_%g-%g_n%d_"
        "FixedTRotms_%g_FixedNSub_%g_FixedHR_%g" % (
            heartRatesBPM.min(), heartRatesBPM.max(), heartRatesBPM.size,
            nSubsetsValues.min(), nSubsetsValues.max(), nSubsetsValues.size,
            1000 * rotationTimes.min(), 1000 * rotationTimes.max(), rotationTimes.size,
            1000 * fixedTRot, fixedNSubsets, fixedHeartRate))
    outputDirectory = SweepSearchRoot() / scanConfigurationFolder / sweepGridFolder
    outputDirectory.mkdir(parents=True, exist_ok=True)

    if CreateSimulationOverview:
        overviewDirectory = outputDirectory / "SimulationOverview"
        overviewDirectory.mkdir(parents=True, exist_ok=True)
        iExportSimulationOverview(overviewResult, overviewDirectory,
                                  OverviewReconstructionType, ShowPlots, SavePlots)

    options = {
        "TRot": rotationTimes.tolist(),
        "FixedTRotForHeartRateVsNSubsets": fixedTRot,
        "FixedNSubsetsForHeartRateVsTRot": fixedNSubsets,
        "FixedHeartRateForNSubsetsVsTRot": fixedHeartRate,
        "NRecons": NRecons, "NScansPerRecon": NScansPerRecon, "N2pi": N2pi,
        "DetectorElements": DetectorElements, "ImageSize": ImageSize,
        "PhotonCount": PhotonCount, "RandomSeed": RandomSeed,
        "SelectedPhantom": SelectedPhantom, "RestFraction": RestFraction,
        "ReconstructionPhaseFraction": ReconstructionPhaseFraction,
        "ApplyMoireFilter": bool(ApplyMoireFilter), "SimulateNoise": bool(SimulateNoise),
        "SavePerRunResults": bool(SavePerRunResults),
        "ShowReconstructions": bool(ShowReconstructions),
        "UseLinearCache": bool(UseLinearCache), "ShowPlots": bool(ShowPlots),
        "SavePlots": bool(SavePlots), "CreateHeatmaps": bool(CreateHeatmaps),
        "CreateSimulationOverview": bool(CreateSimulationOverview),
        "OverviewHeartRateBPM": OverviewHeartRateBPM,
        "OverviewNSubsets": OverviewNSubsets, "OverviewTRot": OverviewTRot,
        "OverviewReconstructionType": str(OverviewReconstructionType),
        "MetricsToPlot": MetricsToPlot,
    }

    csvPath = outputDirectory / f"{SweepResultFileName}.csv"
    phaseCsvPath = outputDirectory / "JPHeartRateVsNSubsets_PhaseROIMetrics.csv"
    comparisonCsvPath = outputDirectory / "JPHeartRateVsNSubsets_LinearVsArbitrary.csv"
    npzPath = outputDirectory / f"{SweepResultFileName}.npz"
    writetable(sweepTable, csvPath)
    writetable(phaseRoiMetricTable, phaseCsvPath)
    writetable(acquisitionComparisonTable, comparisonCsvPath)
    # Structs are stored as JSON text so the .npz loads without pickle.
    np.savez(npzPath, sweepTable=TableToRecords(sweepTable),
             phaseRoiMetricTable=TableToRecords(phaseRoiMetricTable),
             acquisitionComparisonTable=TableToRecords(acquisitionComparisonTable),
             heartRatesBPM=heartRatesBPM, nSubsetsValues=nSubsetsValues,
             pcaInfo=np.array(json.dumps(pcaInfo)), options=np.array(json.dumps(options)))

    allExportMetricNames = metricNames + ["PCA_Error_Linear", "PCA_Error_Arbitrary",
                                          "PCA_Error_ArbitraryMinusLinear"]
    pairDefinitions = iPairwiseParameterDefinitions(
        heartRatesBPM, nSubsetsValues, rotationTimes, fixedTRot, fixedNSubsets,
        fixedHeartRate)
    for metricName in allExportMetricNames:
        iExportPairwiseMetricMatrices(sweepTable, metricName, outputDirectory,
                                      pairDefinitions)

    invalidPlotMetrics = sorted(set(MetricsToPlot) - set(allExportMetricNames))
    if invalidPlotMetrics:
        raise ValueError(f"Unknown plot metric(s): {', '.join(invalidPlotMetrics)}")

    if ShowPlots or SavePlots:
        for metricName in MetricsToPlot:
            iPlotPairwiseMetricSlices(sweepTable, metricName, outputDirectory,
                                      CreateHeatmaps, ShowPlots, SavePlots,
                                      pairDefinitions)

    print(f"JP subset sweep completed. Summary written to:\n{csvPath}\n{phaseCsvPath}\n"
          f"{comparisonCsvPath}\n{npzPath}")
    return sweepTable


def iPairwiseParameterDefinitions(heartRates, nSubsetsValues, rotationTimes,
                                  fixedTRot, fixedNSubsets, fixedHeartRate):
    """(x axis, y axis, fixed parameter, fixed value, x values, y values) per plane."""
    return [
        ("NSubsets", "HeartRateBPM", "TRotSeconds", fixedTRot, nSubsetsValues, heartRates),
        ("TRotSeconds", "HeartRateBPM", "NSubsets", fixedNSubsets, rotationTimes,
         heartRates),
        ("TRotSeconds", "NSubsets", "HeartRateBPM", fixedHeartRate, rotationTimes,
         nSubsetsValues),
    ]


def iSliceTable(sweepTable, xName, yName, fixedName, fixedValue, xValues, yValues):
    sliceMask = (sweepTable[fixedName] == fixedValue) \
        & np.isin(sweepTable[xName], xValues) & np.isin(sweepTable[yName], yValues)
    return SelectRows(sweepTable, sliceMask)


def iExportPairwiseMetricMatrices(sweepTable, metricName, outputDirectory, pairDefinitions):
    """One exact two-parameter slice per selected comparison as rows=y, columns=x CSV."""
    matrixDirectory = Path(outputDirectory) / "PairwiseMatrices"
    matrixDirectory.mkdir(parents=True, exist_ok=True)
    for xName, yName, fixedName, fixedValue, xValues, yValues in pairDefinitions:
        if xValues.size < 2 or yValues.size < 2:
            continue
        sliceTable = iSliceTable(sweepTable, xName, yName, fixedName, fixedValue,
                                 xValues, yValues)
        metricMatrix = iBuildMetricMatrix(sliceTable, xName, yName, metricName,
                                          xValues, yValues)
        matrixTable = {yName: yValues}
        for iColumn, xValue in enumerate(xValues):
            matrixTable[makeValidName("%s_%g" % (xName, xValue))] = metricMatrix[:, iColumn]
        fileName = makeValidName("%s_%sVs%s_Fixed%s_%g_Matrix" % (
            metricName, yName, xName, fixedName, fixedValue))
        writetable(matrixTable, matrixDirectory / f"{fileName}.csv")


def iPlotPairwiseMetricSlices(sweepTable, metricName, outputDirectory, createHeatmaps,
                              showPlots, savePlots, pairDefinitions):
    """Each parameter pair only at its selected fixed value."""
    createdSurface = False
    for xName, yName, fixedName, fixedValue, xValues, yValues in pairDefinitions:
        sliceTable = iSliceTable(sweepTable, xName, yName, fixedName, fixedValue,
                                 xValues, yValues)
        if np.unique(sliceTable[xName]).size < 2 or np.unique(sliceTable[yName]).size < 2:
            continue
        sliceDirectory = Path(outputDirectory) / "PairwisePlots" / f"{yName}Vs{xName}" / \
            makeValidName("Fixed_%s_%g" % (fixedName, fixedValue))
        sliceDirectory.mkdir(parents=True, exist_ok=True)
        StaticCT_JP_TwoParameterSurface_Plot(
            sliceTable, xName, yName, metricName, CreateHeatmap=createHeatmaps,
            ShowFigure=showPlots, SaveFigure=savePlots, OutputDirectory=str(sliceDirectory))
        createdSurface = True

    # Keep useful output when only one of the three parameters is varied.
    if not createdSurface:
        parameterNames = ["HeartRateBPM", "NSubsets", "TRotSeconds"]
        valueCounts = [np.unique(sweepTable[name]).size for name in parameterNames]
        variedNames = [name for name, count in zip(parameterNames, valueCounts) if count > 1]
        if variedNames:
            fixedName = [name for name, count in zip(parameterNames, valueCounts)
                         if count == 1][0]
            StaticCT_JP_ParameterVsMetric_Plot(
                sweepTable, variedNames[0], metricName,
                FixedParameterName=fixedName,
                FixedParameterValue=sweepTable[fixedName][0],
                ShowFigure=showPlots, SaveFigure=savePlots,
                OutputDirectory=str(outputDirectory))


def iBuildMetricMatrix(sliceTable, xName, yName, metricName, xValues, yValues):
    """One complete exact slice as rows=y and columns=x."""
    metricMatrix = np.full((yValues.size, xValues.size), np.nan)
    for iY, yValue in enumerate(yValues):
        for iX, xValue in enumerate(xValues):
            rowMask = (sliceTable[xName] == xValue) & (sliceTable[yName] == yValue)
            if np.count_nonzero(rowMask) != 1:
                raise ValueError("Expected exactly one row for %s=%g and %s=%g, but found "
                                 "%d." % (xName, xValue, yName, yValue,
                                          np.count_nonzero(rowMask)))
            metricMatrix[iY, iX] = sliceTable[metricName][rowMask][0]
    return metricMatrix


def iBuildPairwiseRunGrid(heartRates, nSubsetsValues, rotationTimes, fixedTRot,
                          fixedNSubsets, fixedHeartRate):
    """Only the three requested planes instead of the complete 3D cube."""
    def Plane(first, second):  # ndgrid(first, second)(:): first varies fastest
        return np.tile(first, second.size), np.repeat(second, first.size)

    heartRate, nSubsets = Plane(heartRates, nSubsetsValues)
    heartRateVsSubsetPlane = np.column_stack(
        (heartRate, nSubsets, np.full(heartRate.size, fixedTRot)))
    heartRate, rotationTime = Plane(heartRates, rotationTimes)
    heartRateVsTRotPlane = np.column_stack(
        (heartRate, np.full(heartRate.size, fixedNSubsets), rotationTime))
    nSubsets, rotationTime = Plane(nSubsetsValues, rotationTimes)
    subsetVsTRotPlane = np.column_stack(
        (np.full(nSubsets.size, fixedHeartRate), nSubsets, rotationTime))
    # The planes intersect; simulate each repeated combination once in stable order.
    return iUniqueRowsStable(np.vstack((heartRateVsSubsetPlane, heartRateVsTRotPlane,
                                        subsetVsTRotPlane)))


def iUniqueRowsStable(rows):
    """MATLAB unique(rows, 'rows', 'stable')."""
    _, firstIndex = np.unique(rows, axis=0, return_index=True)
    return rows[np.sort(firstIndex)]


def iBuildDetailedMetricTable(result, heartRate, tRot, nSubsets):
    """Phase-by-ROI matrices as long-format table (phase index varies fastest)."""
    detailed = result["DetailedMetrics"]
    nRecons, nRois = detailed["RMSE_Arbitrary"].shape
    phaseIndex = np.tile(np.arange(1, nRecons + 1), nRois)
    nRows = phaseIndex.size
    table = {
        "HeartRateBPM": np.full(nRows, float(heartRate)),
        "TRotSeconds": np.full(nRows, float(tRot)),
        "NSubsets": np.full(nRows, float(nSubsets)),
        "PhaseIndex": phaseIndex.astype(float),
        "ROIIndex": np.repeat(np.arange(1, nRois + 1), nRecons).astype(float),
        "TReconSeconds": np.asarray(detailed["TReconSeconds"])[phaseIndex - 1],
    }
    for name in metricNames:
        table[name] = np.asarray(detailed[name]).ravel(order="F")
    return table


def iSelectSweepValue(values, requestedValue, optionName, defaultValue):
    """Selects and validates one exact value from a supplied sweep vector."""
    if math.isnan(requestedValue):
        return float(defaultValue)
    allowedValues = list(dict.fromkeys([*values.tolist(), float(defaultValue)]))
    if requestedValue not in allowedValues:
        raise ValueError("%s must be one of the corresponding sweep values: %s" % (
            optionName, ", ".join("%g" % value for value in allowedValues)))
    return float(requestedValue)


def iComputeJointPcaErrors(sweepTable):
    """One PCA model fitted to both members of every acquisition pair."""
    nRuns = sweepTable["HeartRateBPM"].size
    linearFeatures = np.column_stack((sweepTable["RMSE_Linear"],
                                      1 - sweepTable["NCC_Linear"],
                                      1 - sweepTable["SSIM_Linear"]))
    arbitraryFeatures = np.column_stack((sweepTable["RMSE_Arbitrary"],
                                         1 - sweepTable["NCC_Arbitrary"],
                                         1 - sweepTable["SSIM_Arbitrary"]))
    jointErrorFeatures = np.vstack((linearFeatures, arbitraryFeatures))
    if not np.all(np.isfinite(jointErrorFeatures)):
        raise ValueError("Joint PCA input contains non-finite linear or arbitrary RMSE, "
                         "NCC or SSIM values.")

    featureMean = jointErrorFeatures.mean(axis=0)
    featureStd = jointErrorFeatures.std(axis=0, ddof=1)
    featureStd[featureStd == 0] = 1
    standardizedErrors = (jointErrorFeatures - featureMean) / featureStd
    coefficients, scores, explainedVariance = pca(standardizedErrors)
    jointPcaError = scores[:, 0].copy()
    if np.dot(jointPcaError, standardizedErrors.mean(axis=1)) < 0:
        jointPcaError = -jointPcaError
        coefficients[:, 0] = -coefficients[:, 0]
    rawPcaScore = jointPcaError.copy()
    pcaRange = jointPcaError.max() - jointPcaError.min()
    if pcaRange > 0:
        jointPcaError = (jointPcaError - jointPcaError.min()) / pcaRange
    else:
        jointPcaError[:] = 0
    linearError = jointPcaError[:nRuns]
    arbitraryError = jointPcaError[nRuns:]

    pcaDetails = {
        "InputMetrics": ["RMSE", "1-NCC", "1-SSIM"],
        "AcquisitionRowOrder":
            "Rows 1:nRuns = Linear; rows nRuns+1:2*nRuns = Arbitrary",
        "FeatureMean": featureMean.tolist(),
        "FeatureStd": featureStd.tolist(),
        "Coefficients": coefficients.tolist(),
        "RawPC1ScoreJoint": rawPcaScore.tolist(),
        "RawPC1ScoreLinear": rawPcaScore[:nRuns].tolist(),
        "RawPC1ScoreArbitrary": rawPcaScore[nRuns:].tolist(),
        "ExplainedVariancePercent": explainedVariance.tolist(),
        "PC1ExplainedVariancePercent": float(explainedVariance[0]),
        "ScoreConvention":
            "Shared scale: 0 = best, 1 = worst across both acquisition types",
        "NumberOfPairedSimulations": nRuns,
    }

    def Interleaved(linear, arbitrary):  # linear/arbitrary row pairs per simulation
        return np.column_stack((linear, arbitrary)).ravel()

    comparisonTable = {
        "PairIndex": np.repeat(np.arange(1, nRuns + 1), 2).astype(float),
        "HeartRateBPM": np.repeat(sweepTable["HeartRateBPM"], 2),
        "NSubsets": np.repeat(sweepTable["NSubsets"], 2),
        "TRotSeconds": np.repeat(sweepTable["TRotSeconds"], 2),
        "AcquisitionType": np.tile(np.array(["Linear", "Arbitrary"]), nRuns),
        "RMSE": Interleaved(sweepTable["RMSE_Linear"], sweepTable["RMSE_Arbitrary"]),
        "NCC": Interleaved(sweepTable["NCC_Linear"], sweepTable["NCC_Arbitrary"]),
        "SSIM": Interleaved(sweepTable["SSIM_Linear"], sweepTable["SSIM_Arbitrary"]),
        "PCA_Error_Shared": Interleaved(linearError, arbitraryError),
    }
    return linearError, arbitraryError, pcaDetails, comparisonTable


def iExportSimulationOverview(result, outputDirectory, reconstructionType, showFigures,
                              saveFigures):
    """One representative reconstruction and its cardiac sampling curve."""
    if not result or "Images" not in result or "MotionSampling" not in result:
        raise ValueError("The representative simulation did not return image or motion "
                         "data.")
    reconstructionType = iValidateReconstructionType(reconstructionType)
    imageField = reconstructionType + "HU"

    parameters = result["Parameters"]
    images = result["Images"]
    sampling = result["MotionSampling"]
    imageSize = parameters["ImageSize"]
    nPhases = images[imageField].shape[0]
    phaseIndex = math.ceil(nPhases / 2)

    def PhaseImage(name):  # MATLAB reshape(..., [n, n])'
        return np.asarray(images[name][phaseIndex - 1, :]).reshape(imageSize, imageSize)

    phantomTruth = PhaseImage("PhantomHU")
    referenceRecon = PhaseImage("GroundTruthHU")
    reconstruction = PhaseImage(imageField)
    absoluteError = np.abs(reconstruction - referenceRecon)

    reconstructionFigure = NewFigure("Representative JP reconstruction", showFigures)
    axesList = reconstructionFigure.subplots(1, 4)
    reconstructionFigure.suptitle(
        "Representative simulation: HR = %g bpm, NSubsets = %d, %s, phase %d/%d at "
        "t = %.4g s" % (parameters["HeartRateBPM"], parameters["NSubsets"],
                        reconstructionType, phaseIndex, nPhases,
                        sampling["TReconSeconds"][phaseIndex - 1]))
    for axes, image, title, colorMap, limits in (
            (axesList[0], phantomTruth, "Simulated phantom [HU]", "gray", (-250, 350)),
            (axesList[1], referenceRecon, "Near-static reference recon [HU]", "gray",
             (-250, 350)),
            (axesList[2], reconstruction, f"{reconstructionType} acquisition recon [HU]",
             "gray", (-250, 350)),
            (axesList[3], absoluteError, "|Recon - reference recon| [HU]", "hot",
             (None, None))):
        imageHandle = axes.imshow(image, cmap=colorMap, vmin=limits[0], vmax=limits[1])
        axes.axis("off")
        reconstructionFigure.colorbar(imageHandle, ax=axes)
        axes.set_title(title)

    heartRate = parameters["HeartRateBPM"]
    # Always one complete cycle; windows around t = 0 are wrapped to the right edge.
    curveTime = np.linspace(0, sampling["HeartCycleSeconds"], 2000)
    normalizedPosition = np.array([CreatePhantom.MotionProfile(
        time, heartRate, sampling["MotionEnabled"], sampling["MotionType"],
        sampling["RestDurationSeconds"]) for time in curveTime])
    curvePosition = sampling["MotionAmplitudeMm"] * normalizedPosition
    directionVector = np.array([math.sin(sampling["MotionDirectionRadians"]),
                                math.cos(sampling["MotionDirectionRadians"])])
    phasePosition = directionVector @ np.asarray(sampling["PhaseDisplacementXYMm"])

    motionFigure = NewFigure("Cardiac motion sampling", showFigures)
    motionAxes = motionFigure.add_subplot(1, 1, 1)
    motionLimits = 1.1 * sampling["MotionAmplitudeMm"] * np.array([-1, 1])
    isFirstIntervalPatch = True
    for iPhase in range(np.size(sampling["TReconSeconds"])):
        for segmentLeft, segmentRight in iWrapIntervalToCycle(
                sampling["TLeftSeconds"][iPhase], sampling["TRightSeconds"][iPhase],
                sampling["HeartCycleSeconds"]):
            motionAxes.fill(
                [segmentLeft, segmentRight, segmentRight, segmentLeft],
                [motionLimits[0], motionLimits[0], motionLimits[1], motionLimits[1]],
                color=(0.45, 0.85, 0.45), alpha=0.22, edgecolor="none",
                label="Acquisition interval" if isFirstIntervalPatch else None)
            isFirstIntervalPatch = False
    motionAxes.plot(curveTime, curvePosition, "b-", linewidth=1.8, label="Cardiac motion")
    motionAxes.plot(sampling["TReconSeconds"], phasePosition, "ro", markerfacecolor="r",
                    markersize=7, label="Reconstructed phase")
    motionAxes.grid(True)
    motionAxes.set_xlabel("Time [s]")
    motionAxes.set_ylabel("Displacement along motion direction [mm]")
    motionAxes.set_title("Cardiac motion sampling: HR = %g bpm, NSubsets = %d, "
                         "$T_{Rot}$ = %g s" % (heartRate, parameters["NSubsets"],
                                               parameters["TRotSeconds"]))
    motionAxes.legend(loc="best")
    motionAxes.set_xlim(0, sampling["HeartCycleSeconds"])
    motionAxes.set_xticks(np.linspace(0, sampling["HeartCycleSeconds"], 5))
    motionAxes.set_ylim(*motionLimits)

    reconstructionFileTag = "%s_HR_%g_NSubsets_%d" % (
        reconstructionType, heartRate, parameters["NSubsets"])
    motionFileTag = "HR_%g_NSubsets_%d" % (heartRate, parameters["NSubsets"])
    if saveFigures:
        # MATLAB additionally writes .fig files (savefig); matplotlib has no equivalent.
        exportFigure = StaticCTVisualizationAndEvaluationHelpers.StaticCT_exportFigure
        exportFigure(reconstructionFigure, str(Path(outputDirectory) / (
            f"RepresentativeReconstruction_{reconstructionFileTag}.png")), 300)
        exportFigure(motionFigure, str(Path(outputDirectory) / (
            f"CardiacMotionSampling_{motionFileTag}.png")), 300)
    if not showFigures:
        plt.close(reconstructionFigure)
        plt.close(motionFigure)


def iValidateReconstructionType(reconstructionType):
    """MATLAB validatestring: case-insensitive prefix match on Linear/Arbitrary."""
    text = str(reconstructionType).lower()
    matches = [name for name in ("Linear", "Arbitrary")
               if text and name.lower().startswith(text)]
    if len(matches) != 1:
        raise ValueError("OverviewReconstructionType must be 'Linear' or 'Arbitrary'.")
    return matches[0]


def iWrapIntervalToCycle(leftTime, rightTime, cycleDuration):
    """Periodic acquisition interval mapped onto one displayed cardiac cycle."""
    if rightTime - leftTime >= cycleDuration:
        return [(0.0, cycleDuration)]
    wrappedLeft = leftTime % cycleDuration
    wrappedRight = rightTime % cycleDuration
    if wrappedLeft <= wrappedRight:
        return [(wrappedLeft, wrappedRight)]
    return [(0.0, wrappedRight), (wrappedLeft, cycleDuration)]


main = StaticCT_JP_HeartRateVsNSubsets_Sweep


if __name__ == "__main__":
    main()
    plt.show()
