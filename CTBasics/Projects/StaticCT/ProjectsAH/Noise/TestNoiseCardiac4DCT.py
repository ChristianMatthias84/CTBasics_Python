"""Noise study for the cardiac 4D CT scan with randomized view order.

Port of Projects/Static CT/Projects AH/Noise/TestNoiseCardiac4DCT.m.
The script reconstructs the same scan twice, once in angular order and once with the two
half turns randomly permuted, to compare the noise and motion artefacts of both orders.
"""
import math

import numpy as np

from CTBasics.LecturesAndTestScripts.CTRecon101 import _Save
from CTBasics.Memory.Allocation import Allocation
from CTBasics.ReadAndWrite.IOHelperFunctions import IOHelperFunctions
from CTBasics.Reconstruction.Backprojection import Backprojection
from CTBasics.Reconstruction.Constants import Constants
from CTBasics.Reconstruction.Convolution import Convolution
from CTBasics.Reconstruction.ForwardProjection import ForwardProjection
from CTBasics.Reconstruction.Kernels import Kernels
from CTBasics.Reconstruction.MoireFilter import MoireFilter
from CTBasics.Simulation.Noise import Noise


def main(Outputdirectory=None, ShowFigures=False, Reconkernel=Kernels.RamLak,
         SimNoise=True, ApplyMoireFilter=False, N_I0=100000, TRot=0.05,
         RM=250, M=500, thetaN=8 * math.pi, N2pi=1024, Nx=512, Ny=512,
         SelectedPhantom="CardiacPhantom", Heartrate=60, CM=False, Stent=False, Plaque=0,
         Seed=None):
    if Outputdirectory is None:
        Outputdirectory = str(IOHelperFunctions().GlobalOutputFolderPath) + "cardiacPhantom\\"

    N = int(N2pi * thetaN / (2 * math.pi))

    dxi = 2 * RM / (M - 1)
    xi0 = -RM - dxi
    theta0 = 0
    dtheta = thetaN / N

    dx = 2 * RM / Nx
    dy = 2 * RM / Ny
    x0 = -RM - dx / 2.0
    y0 = -RM - dy / 2.0

    # 0 to 2*pi-dtheta, so every theta appears exactly once and 2*pi is not duplicated.
    ThetaArray = np.linspace(0, 2 * math.pi - dtheta, N2pi)

    rng = np.random.default_rng(Seed)
    idx = rng.permutation(N2pi // 2)
    ThetaArray_rand = np.concatenate((ThetaArray[idx], ThetaArray[idx + N2pi // 2]))

    # MATLAB overrides the Heartrate variable with a literal 80 bpm in both calls.
    Projections, Angles_start, center_points_1, _, _, _, _ = \
        ForwardProjection.RayDrivenForwProjParCartesianMultipleWithMotionArbitraryOrder(
            x0, dx, Nx, y0, dy, Ny, N, xi0, dxi, M, N2pi, TRot, ThetaArray,
            Heartrate=80, CM=CM, Plaque=Plaque, Stent=Stent)

    Projections_arbitrary, Angles_rearr, center_points_2, _, _, _, _ = \
        ForwardProjection.RayDrivenForwProjParCartesianMultipleWithMotionArbitraryOrder(
            x0, dx, Nx, y0, dy, Ny, N, xi0, dxi, M, N2pi, TRot, ThetaArray_rand,
            Heartrate=80, CM=CM, Plaque=Plaque, Stent=Stent)

    Projections = np.asarray(Projections).ravel()
    Projections_arbitrary = np.asarray(Projections_arbitrary).ravel()

    _Save(Outputdirectory, "Projections2DParallelBeam", Projections)

    if SimNoise:
        Projections = Noise.ApplyPoissonNoise(Projections, M * N, N_I0)
        _Save(Outputdirectory, "Projections2D", Projections)

    if ApplyMoireFilter:
        Projections = MoireFilter.ApplyMoireFilterOnProjections(
            Projections, N, M, dxi, 0.0 * (1 / dxi), 0.5 * (1 / dxi))
        Projections_arbitrary = MoireFilter.ApplyMoireFilterOnProjections(
            Projections_arbitrary, N, M, dxi, 0.0 * (1 / dxi), 0.5 * (1 / dxi))

    Kernel = None
    if Reconkernel == Kernels.SheppLogan:
        Kernel = Kernels.SheppLoganKernelSD(M, dxi)
    elif Reconkernel == Kernels.RamLak:
        Kernel = Kernels.RamLakKernelSD(M, dxi)

    Projections = Convolution.ConvolveSinogram(Projections, Kernel, M, N, dxi)
    Projections_arbitrary = Convolution.ConvolveSinogram(
        Projections_arbitrary, Kernel, M, N, dxi)

    _Save(Outputdirectory, "Projections2DFiltered", Projections)

    Npi = N2pi // 2
    Nphases = N // Npi
    MultiphaseRecon = Allocation.zeros(Nphases, Nx * Ny)
    MultiphaseRecon_rand = Allocation.zeros(Nphases, Nx * Ny)

    for i in range(1, Nphases + 1):
        theta0_i = theta0 + (i - 1) * Npi * dtheta
        ReconstructedImage = Backprojection.PixDrivenBackProjParCartesian(
            theta0_i, dtheta, Npi, N2pi, xi0, dxi, M,
            Projections[(i - 1) * Npi * M:i * Npi * M], x0, dx, Nx, y0, dy, Ny)
        ReconstructedImage_arbitrary = Backprojection.PixDrivenBackProjParCartesian(
            theta0_i, dtheta, Npi, N2pi, xi0, dxi, M,
            Projections_arbitrary[(i - 1) * Npi * M:i * Npi * M], x0, dx, Nx, y0, dy, Ny)

        MultiphaseRecon[i - 1, :] = ReconstructedImage
        MultiphaseRecon_rand[i - 1, :] = ReconstructedImage_arbitrary

    MultiphaseReconHU = (MultiphaseRecon - Constants.MuW) / Constants.MuW * 1000
    MultiphaseReconHU_rand = (MultiphaseRecon_rand - Constants.MuW) / Constants.MuW * 1000

    # MATLAB transposes before flattening, so the files are stored phase-interleaved.
    _Save(Outputdirectory, "MultiphaseRecon", MultiphaseRecon.T.ravel())
    _Save(Outputdirectory, f"MultiphaseReconHU_{Heartrate:.0f}_{TRot:.2f}s",
          MultiphaseReconHU.T.ravel())
    _Save(Outputdirectory, f"MultiphaseReconHU_rand_{Heartrate:.0f}_{TRot:.2f}s",
          MultiphaseReconHU_rand.T.ravel())

    if ShowFigures:
        import matplotlib.pyplot as plt

        plt.figure()
        plt.imshow(Projections.reshape(N, M), cmap="gray")
        plt.title("Filtered projections")
        for i in range(Nphases):
            plt.figure()
            plt.imshow(MultiphaseRecon[i].reshape(Ny, Nx), cmap="gray")
            plt.title(f"Phase {i + 1}")
        plt.show()

    return {
        "Projections": Projections,
        "Projections_arbitrary": Projections_arbitrary,
        "Angles_start": Angles_start,
        "Angles_rearr": Angles_rearr,
        "center_points_1": center_points_1,
        "center_points_2": center_points_2,
        "Kernel": Kernel,
        "MultiphaseRecon": MultiphaseRecon,
        "MultiphaseRecon_rand": MultiphaseRecon_rand,
        "MultiphaseReconHU": MultiphaseReconHU,
        "MultiphaseReconHU_rand": MultiphaseReconHU_rand,
        "Geometry": {"RM": RM, "M": M, "N": N, "N2pi": N2pi, "Npi": Npi,
                     "Nphases": Nphases, "thetaN": thetaN,
                     "theta0": theta0, "dtheta": dtheta, "xi0": xi0, "dxi": dxi,
                     "x0": x0, "dx": dx, "Nx": Nx, "y0": y0, "dy": dy, "Ny": Ny,
                     "ThetaArray": ThetaArray, "ThetaArray_rand": ThetaArray_rand},
    }

if __name__ == "__main__":
    import matplotlib.pyplot as plt

    main(ShowFigures=True)
    plt.show()
