"""Animates the scanned angles on the unit circle and writes the result as a GIF.

Port of Projects/Static CT/Projects JP/StaticCT_createSubsetAnimationGif.m.
"""
import math
from pathlib import Path

import numpy as np


def StaticCT_createSubsetAnimationGif(ThetaArray, reconType, NSubsets, Outputdirectory,
                                      MaxAngles=100, MaxFrames=30, DelayTime=0.2,
                                      Visible=False):
    import matplotlib.pyplot as plt
    from matplotlib.animation import PillowWriter

    ThetaArray = np.asarray(ThetaArray, dtype=float).ravel()

    maxAngles = round(MaxAngles)
    maxFrames = round(MaxFrames)

    NGifAngles = min(maxAngles, ThetaArray.size)
    if NGifAngles < 1:
        raise ValueError("ThetaArray is empty.")

    nFrames = min(maxFrames, NGifAngles)
    frameIndices = np.unique(np.round(np.linspace(1, NGifAngles, nFrames)).astype(int))

    t = np.linspace(0, 2 * math.pi, 500)

    fig = plt.figure(facecolor="w")
    axes = fig.gca()
    axes.plot(np.cos(t), np.sin(t), "k", linewidth=1.5)
    axes.set_aspect("equal")
    axes.set_xlim(-1.2, 1.2)
    axes.set_ylim(-1.2, 1.2)
    axes.grid(True)
    axes.set_title("Animation of angles on unit circle")
    axes.set_xlabel(r"cos($\theta$)")
    axes.set_ylabel(r"sin($\theta$)")

    if reconType == "180":
        subset_angles = np.linspace(0, math.pi, NSubsets + 1)
    else:
        subset_angles = np.linspace(0, 2 * math.pi, NSubsets + 1)

    r_outer = 1.0
    r_inner = 0.92
    for angle in subset_angles:
        axes.plot([r_inner * math.cos(angle), r_outer * math.cos(angle)],
                  [r_inner * math.sin(angle), r_outer * math.sin(angle)],
                  "b", linewidth=2)

    hPts, = axes.plot([], [], "ro", markerfacecolor="r", markersize=6)

    filename = Path(Outputdirectory) / "winkel_animation.gif"
    filename.parent.mkdir(parents=True, exist_ok=True)

    writer = PillowWriter(fps=max(1.0 / DelayTime, 1.0))
    with writer.saving(fig, str(filename), dpi=80):
        for idx in frameIndices:
            hPts.set_data(np.cos(ThetaArray[:idx]), np.sin(ThetaArray[:idx]))
            writer.grab_frame()

    if Visible:
        plt.show()
    else:
        plt.close(fig)

    print(f"GIF saved as: {filename}")
    return str(filename)
