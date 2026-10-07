"""Multiphase (4D) parallel beam reconstruction of a breathing lung phantom.

Port of LecturesAndTestScripts/Test4DCT.m. The scan covers four turns; the full-scan
reconstruction is blurred by the motion, while the per-half-turn reconstructions resolve
the individual breathing phases.
"""
import numpy as np

from ..Memory.Allocation import Allocation
from ..Reconstruction.Backprojection import Backprojection
from ..Reconstruction.Constants import Constants
from ..Reconstruction.Convolution import Convolution
from ..Reconstruction.ForwardProjection import ForwardProjection
from ..Reconstruction.Kernels import Kernels
from ..Reconstruction.MoireFilter import MoireFilter
from .CTRecon101 import _Save


def main(Outputdirectory=None, ShowFigures=False, Reconkernel=Kernels.RamLak,
         ApplyMoireFilter=False, SelectedPhantom="SimpleLungWithTumor4D",
         RM=250, M=500, thetaN=8 * np.pi, N2pi=1024, TRot=0.5,
         Nxsim=256, Nysim=256, Nx=256, Ny=256):
    # =========================================================
    # Geometry
    # =========================================================

    N = int(N2pi / (2 * np.pi) * thetaN)
    dxi = 2 * RM / (M - 1)
    xi0 = -RM - dxi
    theta0 = 0
    dtheta = thetaN / N

    dxsim = 2 * RM / Nxsim
    dysim = 2 * RM / Nysim
    xcsim = 0
    ycsim = 0
    x0sim = -RM - dxsim / 2.0 + xcsim
    y0sim = -RM - dysim / 2.0 + ycsim

    dx = 2 * RM / Nx
    dy = 2 * RM / Ny
    xc = 0
    yc = 0
    x0 = -RM - dx / 2.0 + xc
    y0 = -RM - dy / 2.0 + yc

    # =========================================================
    # Rawdata simulation of the moving phantom
    # =========================================================

    # MATLAB passes the reconstruction raster here, not the simulation raster.
    Projections = ForwardProjection.RayDrivenForwProjParCartesianMultipleWithMotion(
        x0, dx, Nx, y0, dy, Ny, SelectedPhantom,
        theta0, dtheta, N, xi0, dxi, M, N2pi, TRot
    )

    if ApplyMoireFilter:
        Projections = MoireFilter.ApplyMoireFilterOnProjections(
            Projections, N, M, dxi, 0.0 * (1 / dxi), 0.5 * (1 / dxi)
        )

    # =========================================================
    # Filtering
    # =========================================================

    if Reconkernel == Kernels.SheppLogan:
        Kernel = Kernels.SheppLoganKernelSD(M, dxi)
    elif Reconkernel == Kernels.RamLak:
        Kernel = Kernels.RamLakKernelSD(M, dxi)
    else:
        raise ValueError(f"Unknown reconstruction kernel: {Reconkernel}")

    FilteredProjections = Convolution.ConvolveSinogram(Projections, Kernel, M, N, dxi)

    # =========================================================
    # Reconstruction over the full scan
    # =========================================================

    ReconstructedImage = Backprojection.PixDrivenBackProjParCartesian(
        theta0, dtheta, N, N2pi, xi0, dxi, M, FilteredProjections, x0, dx, Nx, y0, dy, Ny
    )
    ReconstructedImageHU = (ReconstructedImage - Constants.MuW) / Constants.MuW * 1000

    # =========================================================
    # Phase resolved reconstruction, one image per half turn
    # =========================================================

    Npi = int(N2pi // 2)
    Nphases = int(N // Npi)
    MultiphaseRecon = Allocation.zeros(Nphases, Nx * Ny)

    for i in range(1, Nphases + 1):
        theta0_i = theta0 + (i - 1) * Npi * dtheta
        MultiphaseRecon[i - 1, :] = Backprojection.PixDrivenBackProjParCartesian(
            theta0_i, dtheta, Npi, N2pi, xi0, dxi, M,
            FilteredProjections[(i - 1) * Npi * M:i * Npi * M],
            x0, dx, Nx, y0, dy, Ny
        )

    MultiphaseReconHU = (MultiphaseRecon - Constants.MuW) / Constants.MuW * 1000

    if Outputdirectory is not None:
        _Save(Outputdirectory, "Projections2DParallelBeam", Projections)
        _Save(Outputdirectory, "Projections2DFiltered", FilteredProjections)
        _Save(Outputdirectory, "Reconstruction", ReconstructedImage)
        _Save(Outputdirectory, "ReconstructionHU", ReconstructedImageHU)
        # MATLAB transposes the phase stack before writing, i.e. phases vary fastest.
        _Save(Outputdirectory, "MultiphaseRecon", MultiphaseRecon.T.ravel())
        _Save(Outputdirectory, "MultiphaseReconHU", MultiphaseReconHU.T.ravel())

    if ShowFigures:
        _ShowFigures(Projections, FilteredProjections, Kernel, M, N,
                     ReconstructedImageHU, MultiphaseReconHU, Nx, Ny)

    return {
        "Projections": Projections,
        "Kernel": Kernel,
        "FilteredProjections": FilteredProjections,
        "ReconstructedImage": ReconstructedImage,
        "ReconstructedImageHU": ReconstructedImageHU,
        "MultiphaseRecon": MultiphaseRecon,
        "MultiphaseReconHU": MultiphaseReconHU,
        "Geometry": {
            "RM": RM, "M": M, "N": N, "N2pi": N2pi, "Npi": Npi, "Nphases": Nphases,
            "theta0": theta0, "dtheta": dtheta, "thetaN": thetaN,
            "xi0": xi0, "dxi": dxi, "TRot": TRot,
            "Nx": Nx, "Ny": Ny, "dx": dx, "dy": dy, "x0": x0, "y0": y0,
            "Nxsim": Nxsim, "Nysim": Nysim, "dxsim": dxsim, "dysim": dysim,
            "x0sim": x0sim, "y0sim": y0sim,
        },
    }


def _ShowFigures(Projections, FilteredProjections, Kernel, M, N,
                 ReconstructedImageHU, MultiphaseReconHU, Nx, Ny):
    import matplotlib.pyplot as plt

    plt.figure()
    plt.imshow(np.asarray(Projections).reshape(N, M), cmap="gray", aspect="auto")
    plt.title("Parallel beam sinogram with motion")

    plt.figure()
    plt.plot(Kernel)
    plt.title("Reconstruction kernel")

    plt.figure()
    plt.imshow(np.asarray(FilteredProjections).reshape(N, M), cmap="gray", aspect="auto")
    plt.title("Filtered sinogram")

    plt.figure()
    plt.imshow(ReconstructedImageHU.reshape(Ny, Nx), cmap="gray", vmin=-1000, vmax=200)
    plt.title("Full scan reconstruction [HU]")
    plt.colorbar()

    for i in range(MultiphaseReconHU.shape[0]):
        plt.figure()
        plt.imshow(MultiphaseReconHU[i].reshape(Ny, Nx), cmap="gray", vmin=-1000, vmax=200)
        plt.title(f"Phase {i + 1} [HU]")
        plt.colorbar()

    plt.show()


if __name__ == "__main__":
    result = main()
    print(f"{result['Geometry']['Nphases']} phases reconstructed")
