"""Exact one-dimensional slice of a JP subset sweep.

Port of Projects/Static CT/Projects AH/sweeps/StaticCT_JP_ParameterVsMetric_Plot.m, a thin
wrapper around DualSource_ParameterVsMetric_Plot. By default the heart rate is plotted
against the arbitrary RMSE at exactly ``NSubsets = 64``; ``sweepData=None`` uses the newest
``JPHeartRateVsNSubsets_AllMetrics.npz`` (MATLAB: ``.mat``).
"""
from CTBasics.Projects.StaticCT.ProjectsAH.DualSource.sweeps.DualSource_ParameterVsMetric_Plot import DualSource_ParameterVsMetric_Plot
from CTBasics.Projects.StaticCT.ProjectsAH.sweeps._JPSweepHelpers import SweepResultsNewestFirst


def StaticCT_JP_ParameterVsMetric_Plot(sweepData=None, parameterName="HeartRateBPM",
                                       metricName="RMSE_Arbitrary", *,
                                       FixedParameterName="NSubsets",
                                       FixedParameterValue=64, ShowFigure=True,
                                       SaveFigure=True, OutputDirectory=""):
    if sweepData is None:
        sweepData = SweepResultsNewestFirst()[0]
    return DualSource_ParameterVsMetric_Plot(
        sweepData, parameterName, metricName,
        FixedParameterName=FixedParameterName,
        FixedParameterValue=FixedParameterValue,
        ShowFigure=ShowFigure, SaveFigure=SaveFigure, OutputDirectory=OutputDirectory)


main = StaticCT_JP_ParameterVsMetric_Plot


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    main()
    plt.show()
