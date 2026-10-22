"""JP subset sweep as 3D surface and heatmap of two parameters against one metric.

Port of Projects/Static CT/Projects AH/sweeps/StaticCT_JP_TwoParameterSurface_Plot.m, a thin
wrapper around DualSource_TwoParameterSurface_Plot. By default NSubsets and heart rate are
plotted against the combined arbitrary PCA error; ``sweepData=None`` uses the newest
``JPHeartRateVsNSubsets_AllMetrics.npz`` (MATLAB: ``.mat``) that varies both parameters.
MATLAB's path check for the shared plotter is not needed, the import resolves it.
"""
from CTBasics.Projects.StaticCT.ProjectsAH.DualSource.sweeps.DualSource_TwoParameterSurface_Plot import DualSource_TwoParameterSurface_Plot
from CTBasics.Projects.StaticCT.ProjectsAH.sweeps._JPSweepHelpers import (SweepResultsNewestFirst, SweepSearchRoot,
                                                                          SweepVariesBoth)


def StaticCT_JP_TwoParameterSurface_Plot(sweepData=None, xParameterName="NSubsets",
                                         yParameterName="HeartRateBPM",
                                         metricName="PCA_Error_Arbitrary", *,
                                         InterpolateMissing=False, ShowDataPoints=True,
                                         CreateHeatmap=True, ShowHeatmapValues=True,
                                         ShowFigure=True, SaveFigure=True,
                                         OutputDirectory=""):
    if sweepData is None:
        sweepData = iFindLatestTwoDimensionalJPSweep(str(xParameterName),
                                                     str(yParameterName))
    return DualSource_TwoParameterSurface_Plot(
        sweepData, xParameterName, yParameterName, metricName,
        InterpolateMissing=InterpolateMissing, ShowDataPoints=ShowDataPoints,
        CreateHeatmap=CreateHeatmap, ShowHeatmapValues=ShowHeatmapValues,
        ShowFigure=ShowFigure, SaveFigure=SaveFigure, OutputDirectory=OutputDirectory)


def iFindLatestTwoDimensionalJPSweep(xParameterName, yParameterName):
    for candidatePath in SweepResultsNewestFirst():
        if SweepVariesBoth(candidatePath, xParameterName, yParameterName):
            return candidatePath
    raise FileNotFoundError(
        f"No JP sweep with at least two distinct values for both {xParameterName} and "
        f"{yParameterName} was found below {SweepSearchRoot()}.")


main = StaticCT_JP_TwoParameterSurface_Plot


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    main()
    plt.show()
