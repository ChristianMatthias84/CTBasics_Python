"""Renders the ground truth motion as an animated GIF.

Port of Projects/Static CT/Projects AH/DualSource/Shared/DualSource_createGroundTruthGif.m.
MATLAB captures the figure frame by frame; here matplotlib's PillowWriter does the same.
"""
from pathlib import Path

import numpy as np

from ......Simulation.CreatePhantom import CreatePhantom


def DualSource_createGroundTruthGif(
        Outputdirectory, Nx, Ny, dx, dy, x0, y0,
        Heartrate, CM, Stent, Plaque, Rotation, Direction, MotionType, TRest,
        Shift_GT, TRot, Nsources, Npar, SelectedPhantom):
    import matplotlib.pyplot as plt
    from matplotlib.animation import PillowWriter
    from matplotlib.patches import Ellipse

    ampGif = 10

    window_single = TRot / 2
    t1_single = Shift_GT - window_single / 2
    t2_single = Shift_GT + window_single / 2

    window_multi = TRot / (2 * Nsources)
    t1_multi = Shift_GT - window_multi / 2
    t2_multi = Shift_GT + window_multi / 2

    tStart = t1_single
    tEnd = t2_single

    NframesGif = 100
    dT_gif = (tEnd - tStart) / (NframesGif - 1)
    gifSpeedFactor = 0.2
    gifDelay = max(dT_gif / gifSpeedFactor, 0.02)

    xMargin = window_single / 2
    tCurve = np.linspace(tStart - xMargin, tEnd + xMargin, 500)
    yCurve = ampGif * np.array([
        CreatePhantom.MotionProfile(tt, Heartrate, True, MotionType, TRest) for tt in tCurve
    ])

    gifFile = Path(Outputdirectory) / (
        f"GroundTruth_Motion_HR{int(Heartrate)}_TRot{round(TRot * 1000)}ms_N{int(Npar)}.gif"
    )
    gifFile.parent.mkdir(parents=True, exist_ok=True)

    figGif = plt.figure(figsize=(13, 4.5), facecolor="w")
    figGif.canvas.manager.set_window_title("Ground Truth Motion") if figGif.canvas.manager else None
    axCurve = figGif.add_subplot(1, 3, 1)
    axImg = figGif.add_subplot(1, 3, 2)
    axLed = figGif.add_subplot(1, 3, 3)

    ySingle = ampGif * CreatePhantom.MotionProfile(
        (t1_single + t2_single) / 2, Heartrate, True, MotionType, TRest
    )
    yMulti = ampGif * CreatePhantom.MotionProfile(
        (t1_multi + t2_multi) / 2, Heartrate, True, MotionType, TRest
    )
    yGT = ampGif * CreatePhantom.MotionProfile(Shift_GT, Heartrate, True, MotionType, TRest)

    writer = PillowWriter(fps=max(1.0 / gifDelay, 1.0))

    with writer.saving(figGif, str(gifFile), dpi=80):
        for f in range(1, NframesGif + 1):
            tNow = tStart + (f - 1) * dT_gif
            nT_gif = (tNow - Shift_GT) / dT_gif

            Image_gt, _, _, _ = CreatePhantom.CreateVoxelPhantom2DWithMotionUnified(
                Nx, dx, x0, Ny, dy, y0,
                Shift_GT, dT_gif, nT_gif, SelectedPhantom,
                Heartrate=Heartrate, CM=CM, Stent=Stent, Plaque=Plaque,
                Motion=True, Rotation=Rotation, Direction=Direction,
                MotionType=MotionType, TRest=TRest,
            )

            axCurve.clear()
            axCurve.grid(True)
            axCurve.plot(tCurve, yCurve, "b", linewidth=1.5)
            axCurve.plot([t1_single, t2_single], [ySingle, ySingle], "r", linewidth=4)
            axCurve.plot([t1_multi, t2_multi], [yMulti, yMulti], "g", linewidth=4)
            axCurve.plot(Shift_GT, yGT, "o", color=(0.3, 0.3, 0.3),
                         markerfacecolor=(0.8, 0.8, 0.8), markersize=6)
            yNow = ampGif * CreatePhantom.MotionProfile(
                tNow, Heartrate, True, MotionType, TRest
            )
            axCurve.plot(tNow, yNow, "ko", markerfacecolor="w", markersize=8)
            axCurve.axvline(tNow, linestyle="--", color="k")
            axCurve.set_xlabel("Time [s]")
            axCurve.set_ylabel("Displacement [mm]")
            axCurve.set_title("Scan position on motion")
            axCurve.legend(["Motion", "Single window", "Multi window", "GroundTruth", "Current"],
                           loc="lower center")
            axCurve.set_xlim(tStart - xMargin, tEnd + xMargin)

            axImg.clear()
            axImg.imshow(np.asarray(Image_gt).reshape(Ny, Nx), cmap="gray")
            axImg.set_title(f"Ground Truth   t = {tNow:.3f} s")
            axImg.axis("off")

            singleActive = t1_single <= tNow <= t2_single
            multiActive = t1_multi <= tNow <= t2_multi

            axLed.clear()
            colSingle = _PickLedColor(singleActive, (1.0, 0.2, 0.2), (0.35, 0.15, 0.15))
            colMulti = _PickLedColor(multiActive, (0.2, 0.9, 0.2), (0.15, 0.30, 0.15))
            axLed.add_patch(Ellipse((0.3, 0.70), 0.2, 0.2, facecolor=colSingle, edgecolor="k"))
            axLed.text(0.5, 0.70, f"Single Source  [{_OnOff(singleActive)}]",
                       va="center", fontweight="bold")
            axLed.add_patch(Ellipse((0.3, 0.35), 0.2, 0.2, facecolor=colMulti, edgecolor="k"))
            axLed.text(0.5, 0.35, f"Multi Source   [{_OnOff(multiActive)}]",
                       va="center", fontweight="bold")
            axLed.set_title("Scan status")
            axLed.set_xlim(0, 1.6)
            axLed.set_ylim(0, 1)
            axLed.axis("off")

            writer.grab_frame()

    plt.close(figGif)
    print(f"Ground truth motion GIF saved: {gifFile}")
    return str(gifFile)


def _PickLedColor(isActive, onColor, offColor):
    return onColor if isActive else offColor


def _OnOff(isActive):
    return "ON" if isActive else "OFF"
