"""Plots the single and multi source scan windows on the cardiac motion curve.

Port of Projects/Static CT/Projects AH/DualSource/Shared/DualSource_plotScanTiming.m.
"""
import numpy as np

from ......Simulation.CreatePhantom import CreatePhantom


def DualSource_plotScanTiming(Heartrate, MotionType, TRest, Shift_GT, TRot, Nsources,
                              THeartCycle, Shift_Single):
    import matplotlib.pyplot as plt

    figure = plt.figure()
    axes = figure.gca()
    axes.grid(True)

    t_cont = np.linspace(-THeartCycle, 2 * THeartCycle, 1000)
    amp = 10
    y_sin = amp * np.array([
        CreatePhantom.MotionProfile(tt, Heartrate, True, MotionType, TRest) for tt in t_cont
    ])
    axes.plot(t_cont, y_sin, "b", linewidth=2)

    y_center_single = amp * CreatePhantom.MotionProfile(
        Shift_GT, Heartrate, True, MotionType, TRest
    )
    window_single = TRot / 2
    t1_single = Shift_GT - window_single / 2
    t2_single = Shift_GT + window_single / 2
    axes.plot([t1_single, t2_single], [y_center_single, y_center_single], "r", linewidth=4)

    y_center_multi = amp * CreatePhantom.MotionProfile(
        Shift_GT, Heartrate, True, MotionType, TRest
    )
    window_multi = TRot / (2 * Nsources)
    t1_multi = Shift_GT - window_multi / 2
    t2_multi = Shift_GT + window_multi / 2
    axes.plot([t1_multi, t2_multi], [y_center_multi, y_center_multi], "g", linewidth=2)

    axes.plot(Shift_GT, y_center_single, "o", color=(0.3, 0.3, 0.3),
              markersize=4, linewidth=2)

    axes.set_xlabel("Time [s]")
    axes.set_ylabel("Displacement [mm]")
    axes.set_title("Scan windows centered on motion")
    axes.legend(["Motion", "Single", "Multi", "GroundTruth"])
    axes.set_xlim(0 - Shift_Single, 2 * THeartCycle)

    return figure
