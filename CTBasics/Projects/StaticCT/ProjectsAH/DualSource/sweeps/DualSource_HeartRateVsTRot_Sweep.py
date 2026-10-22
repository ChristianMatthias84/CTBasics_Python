"""Heart rate versus rotation time sweep of the normal/dual-source cardiac scan.

Port of Projects/Static CT/Projects AH/DualSource/sweeps/DualSource_HeartRateVsTRot_Sweep.m.

MATLAB is a function ``sweepTable = DualSource_HeartRateVsTRot_Sweep(heartRatesBPM,
rotationTimes, options)``. The name-value options become keyword-only arguments with the
same names and defaults; ``None`` for the grids selects MATLAB's ``50:5:90`` and
``0.1:0.05:0.5``. The table is returned as a dict of column arrays (see _SweepHelpers).

Outputs (folder and file names as in MATLAB, below
``<Output>/cardiacPhantom/parameterSweeps/DualSourceHeartRateVsTRot/<config>/<grid>``):
``DualSourceHeartRateVsTRot_AllMetrics.csv``, ``<metric>_Matrix.csv`` per metric and
``DualSourceHeartRateVsTRot_AllMetrics.npz`` in place of MATLAB's ``.mat``.
"""
import json

import numpy as np

from CTBasics.Projects.StaticCT.ProjectsAH.DualSource.DualSource_CardiacMotion4DCT_Main import main as DualSource_CardiacMotion4DCT_Main
from CTBasics.Projects.StaticCT.ProjectsAH.DualSource.sweeps._SweepHelpers import (MatlabColon, SweepResultFileName, SweepSearchRoot,
                            TableToRecords, makeValidName, writetable)
from CTBasics.Projects.StaticCT.ProjectsAH.DualSource.sweeps.DualSource_ParameterVsMetric_Plot import DualSource_ParameterVsMetric_Plot
from CTBasics.Projects.StaticCT.ProjectsAH.DualSource.sweeps.DualSource_TwoParameterSurface_Plot import DualSource_TwoParameterSurface_Plot

metricNames = [
    "RMSE_Normal", "RMSE_DualSource",
    "NCC_Normal", "NCC_DualSource",
    "SSIM_Normal", "SSIM_DualSource",
    "RMSE_Normal_ROI", "RMSE_DualSource_ROI",
    "NCC_Normal_ROI", "NCC_DualSource_ROI",
    "SSIM_Normal_ROI", "SSIM_DualSource_ROI"]

DefaultMetricsToPlot = (
    "PCA_Error_DualSource",
    "RMSE_DualSource", "NCC_DualSource", "SSIM_DualSource",
    "RMSE_DualSource_ROI", "NCC_DualSource_ROI", "SSIM_DualSource_ROI")


def DualSource_HeartRateVsTRot_Sweep(heartRatesBPM=None, rotationTimes=None, *,
                                     Nsources=2, Npar=512, DetectorElements=500,
                                     ImageSize=512, PhotonCount=2.25e5, RandomSeed=1,
                                     SelectedPhantom="WaterWith4Arteries",
                                     ApplyMoireFilter=False, SimulateNoise=True,
                                     SavePerRunResults=False, ShowReconstructions=False,
                                     ShowPlots=True, SavePlots=True,
                                     MetricsToPlot=DefaultMetricsToPlot):
    """Evaluates the Cartesian product of heart rates and rotation times."""
    if heartRatesBPM is None:
        heartRatesBPM = MatlabColon(50, 5, 90)
    if rotationTimes is None:
        rotationTimes = MatlabColon(0.1, 0.05, 0.5)
    heartRatesBPM = np.atleast_1d(np.asarray(heartRatesBPM, dtype=float))
    rotationTimes = np.atleast_1d(np.asarray(rotationTimes, dtype=float))
    if np.any(heartRatesBPM <= 0) or np.any(rotationTimes <= 0):
        raise ValueError("heartRatesBPM and rotationTimes must be positive.")
    MetricsToPlot = [str(name) for name in MetricsToPlot]
    SelectedPhantom = str(SelectedPhantom)

    # ndgrid + (:): the heart rate varies fastest.
    heartRateColumn = np.tile(heartRatesBPM, rotationTimes.size)
    rotationTimeColumn = np.repeat(rotationTimes, heartRatesBPM.size)
    nRuns = heartRateColumn.size

    if nRuns < 2:
        raise ValueError("At least two parameter combinations are required for PCA.")

    metricValues = np.full((nRuns, len(metricNames)), np.nan)

    print(f"Starting normal/dual-source HeartRate-vs-TRot sweep with {nRuns} simulations.")
    for iRun in range(nRuns):
        heartRate = heartRateColumn[iRun]
        rotationTime = rotationTimeColumn[iRun]
        print(f"[{iRun + 1}/{nRuns}] Heart rate = {heartRate:.0f} bpm, "
              f"TRot = {rotationTime:.3f} s, sources = {Nsources}")

        result = DualSource_CardiacMotion4DCT_Main(
            heartRate, rotationTime,
            Nsources=Nsources,
            Npar=Npar,
            DetectorElements=DetectorElements,
            ImageSize=ImageSize,
            PhotonCount=PhotonCount,
            RandomSeed=RandomSeed,
            SelectedPhantom=SelectedPhantom,
            ApplyMoireFilter=ApplyMoireFilter,
            SimulateNoise=SimulateNoise,
            ShowPictures=ShowReconstructions,
            ShowTimingPlot=False,
            CreateGroundTruthGif=False,
            WriteData=SavePerRunResults,
            IncludeImageData=False)

        for iMetric, metricName in enumerate(metricNames):
            metricValues[iRun, iMetric] = result["Metrics"][metricName]

    sweepTable = {
        "HeartRateBPM": heartRateColumn,
        "TRotSeconds": rotationTimeColumn,
        "Nsources": np.full(nRuns, float(Nsources)),
        "Npar": np.full(nRuns, float(Npar)),
        "DetectorElements": np.full(nRuns, float(DetectorElements)),
        "ImageSize": np.full(nRuns, float(ImageSize)),
        "PhotonCount": np.full(nRuns, float(PhotonCount)),
        "RandomSeed": np.full(nRuns, float(RandomSeed)),
        "SelectedPhantom": np.full(nRuns, SelectedPhantom),
        "SimulateNoise": np.full(nRuns, bool(SimulateNoise)),
        "ApplyMoireFilter": np.full(nRuns, bool(ApplyMoireFilter)),
    }
    for iMetric, metricName in enumerate(metricNames):
        sweepTable[metricName] = metricValues[:, iMetric]

    # Combine the dual-source ROI metrics into a standardized motion-error score. Full-image
    # NCC and SSIM are dominated by the static water and background and can hide changes
    # in the moving structure. NCC and SSIM are inverted so larger always means worse; the
    # normalized PCA score is 0 for the best and 1 for the worst result of this sweep.
    errorFeatures = np.column_stack((
        sweepTable["RMSE_DualSource_ROI"],
        1 - sweepTable["NCC_DualSource_ROI"],
        1 - sweepTable["SSIM_DualSource_ROI"]))
    if not np.all(np.isfinite(errorFeatures)):
        raise ValueError("PCA input contains non-finite RMSE, NCC or SSIM values.")

    featureMean = errorFeatures.mean(axis=0)
    featureStd = errorFeatures.std(axis=0, ddof=1)
    featureStd[featureStd == 0] = 1
    standardizedErrors = (errorFeatures - featureMean) / featureStd
    pcaCoefficients, pcaScores, explainedVariance = pca(standardizedErrors)
    pcaError = pcaScores[:, 0].copy()

    # Orient PC1 so that a larger score means a larger mean standardized error.
    if np.dot(pcaError, standardizedErrors.mean(axis=1)) < 0:
        pcaError = -pcaError
        pcaCoefficients[:, 0] = -pcaCoefficients[:, 0]
    rawPcaScore = pcaError.copy()
    pcaScoreRange = pcaError.max() - pcaError.min()
    if pcaScoreRange > 0:
        pcaError = (pcaError - pcaError.min()) / pcaScoreRange
    else:
        pcaError[:] = 0
    sweepTable["PCA_Error_DualSource"] = pcaError

    pcaInfo = {
        "InputMetrics": ["RMSE_DualSource_ROI", "1-NCC_DualSource_ROI",
                         "1-SSIM_DualSource_ROI"],
        "FeatureMean": featureMean.tolist(),
        "FeatureStd": featureStd.tolist(),
        "Coefficients": pcaCoefficients.tolist(),
        "RawPC1Score": rawPcaScore.tolist(),
        "ExplainedVariancePercent": explainedVariance.tolist(),
        "PC1ExplainedVariancePercent": float(explainedVariance[0]),
        "ScoreConvention": "0 = best, 1 = worst within this sweep",
    }

    options = {
        "Nsources": Nsources, "Npar": Npar, "DetectorElements": DetectorElements,
        "ImageSize": ImageSize, "PhotonCount": PhotonCount, "RandomSeed": RandomSeed,
        "SelectedPhantom": SelectedPhantom, "ApplyMoireFilter": bool(ApplyMoireFilter),
        "SimulateNoise": bool(SimulateNoise), "SavePerRunResults": bool(SavePerRunResults),
        "ShowReconstructions": bool(ShowReconstructions), "ShowPlots": bool(ShowPlots),
        "SavePlots": bool(SavePlots), "MetricsToPlot": MetricsToPlot,
    }

    phantomFolder = makeValidName(SelectedPhantom)
    scanConfigurationFolder = (
        "Ph_%s_NS_%d_Npar_%d_M_%d_Nx_%d_I0_%g_Noise_%d_Moire_%d_Seed_%d" % (
            phantomFolder, Nsources, Npar, DetectorElements, ImageSize, PhotonCount,
            SimulateNoise, ApplyMoireFilter, RandomSeed))
    sweepGridFolder = "HR_%g-%g_n%d_TRot_%g-%gms_n%d" % (
        heartRatesBPM.min(), heartRatesBPM.max(), heartRatesBPM.size,
        1000 * rotationTimes.min(), 1000 * rotationTimes.max(), rotationTimes.size)
    outputDirectory = SweepSearchRoot() / scanConfigurationFolder / sweepGridFolder
    outputDirectory.mkdir(parents=True, exist_ok=True)

    csvPath = outputDirectory / f"{SweepResultFileName}.csv"
    npzPath = outputDirectory / f"{SweepResultFileName}.npz"
    writetable(sweepTable, csvPath)
    # MATLAB saves sweepTable, heartRatesBPM, rotationTimes, pcaInfo and options to a .mat
    # file; the structs are stored as JSON text so the .npz loads without pickle.
    np.savez(npzPath, sweepTable=TableToRecords(sweepTable), heartRatesBPM=heartRatesBPM,
             rotationTimes=rotationTimes, pcaInfo=np.array(json.dumps(pcaInfo)),
             options=np.array(json.dumps(options)))

    # Every metric additionally as a rectangular CSV: heart rates as rows, rotation
    # times as columns.
    allExportMetricNames = metricNames + ["PCA_Error_DualSource"]
    rotationVariableNames = [makeValidName("TRot_%g_s" % value) for value in rotationTimes]
    for metricName in allExportMetricNames:
        metricMatrix = sweepTable[metricName].reshape(
            rotationTimes.size, heartRatesBPM.size).T
        matrixTable = {"HeartRateBPM": heartRatesBPM}
        for iColumn, columnName in enumerate(rotationVariableNames):
            matrixTable[columnName] = metricMatrix[:, iColumn]
        writetable(matrixTable, outputDirectory / f"{metricName}_Matrix.csv")

    invalidPlotMetrics = sorted(set(MetricsToPlot) - set(allExportMetricNames))
    if invalidPlotMetrics:
        raise ValueError(f"Unknown plot metric(s): {', '.join(invalidPlotMetrics)}")

    if ShowPlots or SavePlots:
        useSurfacePlot = heartRatesBPM.size > 1 and rotationTimes.size > 1
        if rotationTimes.size > 1:
            variedParameterName = "TRotSeconds"
            fixedParameterName = "HeartRateBPM"
            fixedParameterValue = heartRatesBPM[0]
        else:
            variedParameterName = "HeartRateBPM"
            fixedParameterName = "TRotSeconds"
            fixedParameterValue = rotationTimes[0]

        for metricName in MetricsToPlot:
            if useSurfacePlot:
                DualSource_TwoParameterSurface_Plot(
                    sweepTable, "TRotSeconds", "HeartRateBPM", metricName,
                    ShowFigure=ShowPlots,
                    SaveFigure=SavePlots,
                    OutputDirectory=str(outputDirectory))
            else:
                DualSource_ParameterVsMetric_Plot(
                    sweepTable, variedParameterName, metricName,
                    FixedParameterName=fixedParameterName,
                    FixedParameterValue=fixedParameterValue,
                    ShowFigure=ShowPlots,
                    SaveFigure=SavePlots,
                    OutputDirectory=str(outputDirectory))

    print(f"Sweep completed. Summary written to:\n{csvPath}\n{npzPath}")
    return sweepTable


def pca(X):
    """MATLAB [coeff, score, ~, ~, explained] = pca(X) (SVD, economy size).

    Each coefficient column is signed so that its largest-magnitude entry is positive,
    as in MATLAB; only min(n - 1, p) components are kept.
    """
    X = np.asarray(X, dtype=float)
    n, p = X.shape
    Xc = X - X.mean(axis=0)
    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    nComponents = min(n - 1, p)
    coeff = Vt[:nComponents].T
    score = U[:, :nComponents] * S[:nComponents]
    latent = S[:nComponents] ** 2 / (n - 1)

    largest = np.argmax(np.abs(coeff), axis=0)
    signs = np.sign(coeff[largest, np.arange(nComponents)])
    signs[signs == 0] = 1
    coeff = coeff * signs
    score = score * signs

    explained = 100 * latent / latent.sum()
    return coeff, score, explained


main = DualSource_HeartRateVsTRot_Sweep
