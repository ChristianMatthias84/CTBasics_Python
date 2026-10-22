"""Renders the cardiac phantom motion over half a second as a PNG series.

Port of Projects/Static CT/Projects JP/TestCardiacPhantom.m. The MATLAB script hardcodes a
personal output directory; here the caller supplies it and nothing is written when it is
omitted.
"""
from pathlib import Path

import numpy as np

from CTBasics.Memory.Allocation import Allocation
from CTBasics.Reconstruction.Constants import Constants
from CTBasics.Simulation.CreatePhantom import CreatePhantom


def main(outputDir=None, ShowFigures=False, Nx=512, Ny=512, dx=1, dy=1,
         x0=-250, y0=-250, T0=0, dT=0.01, duration=0.5,
         window_center=0, window_width=1000,
         SelectedPhantom="CardiacPhantom"):
    NT = int(duration / dT)  # in seconds => 10 pictures at the end

    lower_window = window_center - (window_width / 2)
    upper_window = window_center + (window_width / 2)

    xpoints = Allocation.zeros(1, NT).ravel()
    ypoints = Allocation.zeros(1, NT).ravel()

    Frames = np.zeros((NT, Ny, Nx), dtype=np.uint8)

    if outputDir is not None:
        outputDir = Path(outputDir)
        outputDir.mkdir(parents=True, exist_ok=True)

    for nT in range(1, NT + 1):
        Phantom, _, xpoint, ypoint = \
            CreatePhantom.CreateVoxelPhantom2DWithMotionRespiratory(
                Nx, dx, x0, Ny, dy, y0, T0, dT, nT, SelectedPhantom)

        aImage = np.asarray(Phantom).reshape(Ny, Nx)
        huImage = (aImage - Constants.MuW) / Constants.MuW * 1000

        scaled = 255 * (huImage - lower_window) / (upper_window - lower_window)
        scaledImage = np.clip(scaled, 0, 255).astype(np.uint8)
        Frames[nT - 1] = scaledImage

        xpoints[nT - 1] = xpoint
        ypoints[nT - 1] = ypoint

        if outputDir is not None:
            import matplotlib.pyplot as plt
            plt.imsave(outputDir / f"cardiac_phantom_{nT:03d}.png",
                       scaledImage, cmap="gray", vmin=0, vmax=255)

    if ShowFigures:
        import matplotlib.pyplot as plt

        plt.figure()
        plt.imshow(Frames[-1], cmap="gray", vmin=0, vmax=255)
        plt.figtext(0.5, 0.01,
                    "Blood = 55 HU, Artery Wall = 40 HU, "
                    "Plaque (calcified) = 100 HU, Stent = 350 HU",
                    ha="center", fontsize=12, color="black")

        plt.figure()
        plt.plot(xpoints, ypoints, "ro", markersize=3)
        plt.title("cardiac phantom movement")
        plt.xlabel("X Time [sec]")
        plt.ylabel("Y Position")
        plt.show()

    return {"Frames": Frames, "xpoints": xpoints, "ypoints": ypoints,
            "NT": NT, "lower_window": lower_window, "upper_window": upper_window}

if __name__ == "__main__":
    import matplotlib.pyplot as plt

    main(ShowFigures=True)
    plt.show()
