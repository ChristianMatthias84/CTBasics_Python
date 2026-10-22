"""Single versus dual source parallel beam reconstruction of a static phantom.

Port of Projects/Static CT/Projects AH/DualSource/CTRecon101DualSource.m.
The dual source scan splits the half turn into two angular halves that are projected
separately and concatenated, so the resulting sinogram must match the single source one.

DEVIATION / MATLAB bug: the MATLAB script passes an angle *array* plus only 12 of the 13
arguments of RayDrivenForwProjParCartesianMultiple, so it aborts with "Not enough input
arguments" before any projection is computed. The port keeps the call verbatim; it raises
TypeError at the same place. Pass UseArbitraryOrder=True to run the experiment with the
angle-array projector that the script obviously meant to call.
"""
import math

import numpy as np

from CTBasics.LecturesAndTestScripts.CTRecon101 import _Save
from CTBasics.ReadAndWrite.IOHelperFunctions import IOHelperFunctions
from CTBasics.Reconstruction.Backprojection import Backprojection
from CTBasics.Reconstruction.Constants import Constants
from CTBasics.Reconstruction.Convolution import Convolution
from CTBasics.Reconstruction.ForwardProjection import ForwardProjection
from CTBasics.Reconstruction.Kernels import Kernels
from CTBasics.Reconstruction.MoireFilter import MoireFilter
from CTBasics.Simulation.CreatePhantom import CreatePhantom


def main(Outputdirectory=None, ShowFigures=False, Reconkernel=Kernels.RamLak,
         ApplyMoireFilter=False, RM=250, M=500, Npar=512,
         Nxsim=256, Nysim=256, Nx=256, Ny=256,
         SelectedPhantom="SimpleEllipseWithHips", UseArbitraryOrder=False):
    if Outputdirectory is None:
        Outputdirectory = IOHelperFunctions().GlobalOutputFolderPath

    thetaN = math.pi
    N2pi = 2 * Npar

    dxi = 2 * RM / (M - 1)
    xi0 = -RM - dxi

    # MATLAB drops the duplicated pi entry so the half turn is sampled Npar times.
    theta_all = np.linspace(0, math.pi, Npar + 1)[:-1]

    NparSingle = Npar // 2
    theta_A = theta_all[:NparSingle]
    theta_B = theta_all[NparSingle:]

    theta0 = 0
    dtheta = thetaN / Npar

    dxsim = 2 * RM / Nxsim
    dysim = 2 * RM / Nysim
    x0sim = -RM - dxsim / 2.0
    y0sim = -RM - dysim / 2.0

    dx = 2 * RM / Nx
    dy = 2 * RM / Ny
    x0 = -RM - dx / 2.0
    y0 = -RM - dy / 2.0

    t_per_proj = 0.0005  # 0.5 ms per proj (just imaginary)

    results = {}

    for mode in (1, 2):
        UseDualSource = mode == 2
        print("Running DUAL-SOURCE..." if UseDualSource else "Running SINGLE-SOURCE...")

        Voxelphantom, isSpectral = CreatePhantom.CreateVoxelPhantom2D(
            Nxsim, dxsim, x0sim, Nysim, dysim, y0sim, SelectedPhantom)

        _Save(Outputdirectory, "Phantom", Voxelphantom)

        if UseDualSource:
            Proj_A = _Project(x0sim, dxsim, Nxsim, y0sim, dysim, Nysim, Voxelphantom,
                              theta_A, NparSingle, xi0, dxi, M, UseArbitraryOrder)
            Proj_B = _Project(x0sim, dxsim, Nxsim, y0sim, dysim, Nysim, Voxelphantom,
                              theta_B, NparSingle, xi0, dxi, M, UseArbitraryOrder)
            Projections = np.concatenate((Proj_A, Proj_B))
            results["Dual"] = Projections
        else:
            Projections = _Project(x0sim, dxsim, Nxsim, y0sim, dysim, Nysim, Voxelphantom,
                                   theta_all, Npar, xi0, dxi, M, UseArbitraryOrder)
            results["Single"] = Projections

        _Save(Outputdirectory, "Projections2DParallelBeam", Projections)

        if ApplyMoireFilter:
            Projections = MoireFilter.ApplyMoireFilterOnProjections(
                Projections, Npar, M, dxi, 0.0 * (1 / dxi), 0.5 * (1 / dxi))

        Kernel = None
        if Reconkernel == Kernels.SheppLogan:
            Kernel = Kernels.SheppLoganKernelSD(M, dxi)
        elif Reconkernel == Kernels.RamLak:
            Kernel = Kernels.RamLakKernelSD(M, dxi)

        Projections = Convolution.ConvolveSinogram(Projections, Kernel, M, Npar, dxi)
        _Save(Outputdirectory, "Projections2DFiltered", Projections)

        ReconstructedImage = Backprojection.PixDrivenBackProjParCartesian(
            theta0, dtheta, Npar, N2pi, xi0, dxi, M, Projections,
            x0, dx, Nx, y0, dy, Ny)

        ReconstructedImageHU = (ReconstructedImage - Constants.MuW) / Constants.MuW * 1000
        _Save(Outputdirectory, "Reconstruction", ReconstructedImage)
        _Save(Outputdirectory, "ReconstructionHU", ReconstructedImageHU)

        results["ReconstructedImageHU" + ("Dual" if UseDualSource else "Single")] = \
            ReconstructedImageHU
        results["FilteredProjections" + ("Dual" if UseDualSource else "Single")] = Projections
        results["Kernel"] = Kernel
        results["Voxelphantom"] = Voxelphantom
        results["isSpectral"] = isSpectral

    Difference = results["Single"] - results["Dual"]
    maxError = float(np.max(np.abs(Difference)))
    print(f"Max difference: {maxError:e}")

    if ShowFigures:
        _ShowFigures(results, Difference, M, Npar, Nx, Ny, theta0, thetaN, t_per_proj)

    results["Difference"] = Difference
    results["maxError"] = maxError
    results["Geometry"] = {"RM": RM, "M": M, "Npar": Npar, "N2pi": N2pi,
                           "theta0": theta0, "dtheta": dtheta, "thetaN": thetaN,
                           "xi0": xi0, "dxi": dxi,
                           "x0": x0, "dx": dx, "Nx": Nx, "y0": y0, "dy": dy, "Ny": Ny,
                           "theta_all": theta_all, "theta_A": theta_A, "theta_B": theta_B}
    return results


def _Project(x0sim, dxsim, Nxsim, y0sim, dysim, Nysim, Voxelphantom,
             ThetaArray, N, xi0, dxi, M, UseArbitraryOrder):
    if not UseArbitraryOrder:
        # Verbatim MATLAB call: an angle array is passed as theta0 and M is missing.
        return ForwardProjection.RayDrivenForwProjParCartesianMultiple(
            x0sim, dxsim, Nxsim, y0sim, dysim, Nysim, Voxelphantom,
            ThetaArray, N, xi0, dxi, M)

    Projections2D = np.empty(N * M)
    for n in range(1, N + 1):
        Projections2D[(n - 1) * M:n * M] = \
            ForwardProjection.RayDrivenForwProjParCartesianSingle(
                x0sim, dxsim, Nxsim, y0sim, dysim, Nysim, Voxelphantom,
                float(ThetaArray[n - 1]), xi0, dxi, M)
    return Projections2D


def _ShowFigures(results, Difference, M, Npar, Nx, Ny, theta0, thetaN, t_per_proj):
    import matplotlib.pyplot as plt

    plt.figure()
    plt.plot(results["Kernel"])
    plt.title("Reconstruction kernel")

    for key, title in (("Single", "Single Source Scan"), ("Dual", "Dual Source Scan")):
        plt.figure()
        plt.imshow(results[key].reshape(Npar, M), cmap="gray")
        plt.title(title)

    plt.figure()
    plt.imshow(Difference.reshape(Npar, M), cmap="gray")
    plt.colorbar()
    plt.title("Difference Image (Single - Dual)")

    plt.figure()
    plt.imshow(results["ReconstructedImageHUSingle"].reshape(Ny, Nx), cmap="gray")
    plt.title("Reconstruction single source")

    plt.figure()
    angles_all = np.linspace(theta0, thetaN, Npar)
    plt.plot(angles_all * 180 / math.pi, np.arange(Npar) * t_per_proj * 1000,
             "b", linewidth=2)
    plt.legend(["Single Source"])
    plt.xlabel("Angle")
    plt.ylabel("Time (ms)")
    plt.title("Angle coverage and irradiation time")
    plt.grid(True)

    plt.show()

if __name__ == "__main__":
    import matplotlib.pyplot as plt

    main(ShowFigures=True)
    plt.show()
