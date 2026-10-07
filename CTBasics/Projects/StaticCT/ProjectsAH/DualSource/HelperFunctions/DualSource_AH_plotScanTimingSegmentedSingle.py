"""Timing plot of a continuous single source scan versus a segmented single source scan.

Port of Projects/Static CT/Projects AH/DualSource/HelperFunctions/
DualSource_AH_plotScanTimingSegmentedSingle.m.
"""
import numpy as np

from ......Simulation.CreatePhantom import CreatePhantom


def DualSource_AH_plotScanTimingSegmentedSingle(
        Heartrate, MotionType, TRest, Shift_GT, TRot, Nsources, THeartCycle, Shift_Single,
        TRot_segmented, isSynchronizedSegments=False):
    import matplotlib.pyplot as plt

    figure = plt.figure()
    axes = figure.gca()
    axes.grid(True)

    amp = 10

    if isSynchronizedSegments:
        segmentCenter = Shift_GT
        segmentHalfWindow = TRot_segmented / 4
        segmentStarts = np.full(Nsources, segmentCenter - segmentHalfWindow)
        segmentEnds = np.full(Nsources, segmentCenter + segmentHalfWindow)
        segmentCenters = np.full(Nsources, segmentCenter)
    else:
        singleReadStartTime = Shift_GT - Shift_Single
        singleReadEndTime = Shift_GT + Shift_Single
        segmentPhaseEdges = np.linspace(singleReadStartTime, singleReadEndTime, Nsources + 1)
        segmentStarts = segmentPhaseEdges[:-1]
        segmentEnds = segmentPhaseEdges[1:]
        segmentCenters = (segmentStarts + segmentEnds) / 2

    tMin = min(segmentStarts.min(), Shift_GT - Shift_Single) - 0.2 * THeartCycle
    tMax = max(segmentEnds.max(), Shift_GT + Shift_Single) + 0.2 * THeartCycle

    t_cont = np.linspace(tMin, tMax, 1600)
    y_sin = amp * np.array([
        CreatePhantom.MotionProfile(tt, Heartrate, True, MotionType, TRest) for tt in t_cont
    ])
    axes.plot(t_cont, y_sin, "b", linewidth=2)

    t1Single = Shift_GT - Shift_Single
    t2Single = Shift_GT + Shift_Single
    ySingle = amp * CreatePhantom.MotionProfile(Shift_GT, Heartrate, True, MotionType, TRest)
    axes.plot([t1Single, t2Single], [ySingle, ySingle], "r", linewidth=4)

    if isSynchronizedSegments:
        ySegBase = amp * CreatePhantom.MotionProfile(
            segmentCenters[0], Heartrate, True, MotionType, TRest
        )
        if Nsources > 1:
            offsetScale = np.linspace(-0.35, 0.35, Nsources)
        else:
            offsetScale = np.zeros(1)

        for sourceIdx in range(Nsources):
            ySeg = ySegBase + offsetScale[sourceIdx]
            axes.plot([segmentStarts[sourceIdx], segmentEnds[sourceIdx]], [ySeg, ySeg],
                      "g", linewidth=2)
            axes.plot(segmentCenters[sourceIdx], ySeg, "ko", markerfacecolor="k", markersize=3)

        titleStr = ("Timing: continuous single vs synchronized segmented single "
                    f"(Nsegments = {Nsources})")
        segmentedLegend = "Segmented (same time)"
    else:
        for sourceIdx in range(Nsources):
            ySeg = amp * CreatePhantom.MotionProfile(
                segmentCenters[sourceIdx], Heartrate, True, MotionType, TRest
            )
            axes.plot([segmentStarts[sourceIdx], segmentEnds[sourceIdx]], [ySeg, ySeg],
                      "g", linewidth=3)
            axes.plot(segmentCenters[sourceIdx], ySeg, "ko", markerfacecolor="k", markersize=4)

        titleStr = f"Timing: continuous single vs segmented single (Nsegments = {Nsources})"
        segmentedLegend = "Segmented bins"

    axes.set_xlabel("Time [s]")
    axes.set_ylabel("Displacement [mm]")
    axes.set_title(titleStr)
    axes.legend(["Motion", "Single continuous", segmentedLegend, "Segment centers"])

    txt = (f"TRot_segmented = {TRot_segmented:.3f}s "
           f"({TRot_segmented / TRot:.1f}x TRot)")
    axes.text(tMin + 0.02 * (tMax - tMin), amp * 0.9, txt, color=(0.1, 0.1, 0.1))
    axes.set_xlim(tMin, tMax)

    return figure
