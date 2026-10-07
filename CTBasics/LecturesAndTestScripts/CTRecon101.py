"""Simulate & reconstruct in 2D (parallel beam geometry).

Port of LecturesAndTestScripts/CTRecon101.m. The MATLAB script shows figures and writes
raw binaries to the configured output folder; here both are optional so the script runs
without a CTBasicsConfig.txt.
"""
import numpy as np

from ..Reconstruction.Backprojection import Backprojection
from ..Reconstruction.Constants import Constants
from ..Reconstruction.Convolution import Convolution
from ..Reconstruction.ForwardProjection import ForwardProjection
from ..Reconstruction.Kernels import Kernels
from ..Reconstruction.MoireFilter import MoireFilter
from ..Simulation.CreatePhantom import CreatePhantom


def main(Outputdirectory=None, ShowFigures=False, Reconkernel=Kernels.RamLak, ApplyMoireFilter=False,
         SelectedPhantom="SimpleEllipseWithHips", Npar=512):
    # =========================================================
    # Parameter Definition
    # =========================================================

    # Parallel beam scanner geometry, independent variables.
    RM = 250.0          # Radius of the field of measurement (FOM).
    M = 500             # Number of detector elements.
    thetaN = np.pi
    N2pi = 2 * Npar

    # Parallel beam scanner geometry, dependent variables.
    dxi = 2 * RM / (M - 1)
    xi0 = -RM - dxi
    theta0 = 0.0
    dtheta = thetaN / Npar

    # Simulation geometry.
    Nxsim = 256
    Nysim = 256
    dxsim = 2 * RM / Nxsim
    dysim = 2 * RM / Nysim
    xcsim = 0.0
    ycsim = 0.0
    x0sim = -RM - dxsim / 2.0 + xcsim
    y0sim = -RM - dysim / 2.0 + ycsim

    # Reconstruction geometry.
    Nx = 256
    Ny = 256
    dx = 2 * RM / Nx
    dy = 2 * RM / Ny
    xc = 0.0
    yc = 0.0
    x0 = -RM - dx / 2.0 + xc
    y0 = -RM - dy / 2.0 + yc

    # =========================================================
    # Phantom Creation
    # =========================================================

    Voxelphantom, isSpectral = CreatePhantom.CreateVoxelPhantom2D(
        Nxsim, dxsim, x0sim, Nysim, dysim, y0sim, SelectedPhantom
    )

    # =========================================================
    # Rawdata Simulation
    # =========================================================

    Projections = ForwardProjection.RayDrivenForwProjParCartesianMultiple(
        x0sim, dxsim, Nxsim, y0sim, dysim, Nysim, Voxelphantom,
        theta0, dtheta, Npar, xi0, dxi, M,
    )
    RawProjections = Projections.copy()

    # =========================================================
    # Apply Moire filter
    # =========================================================

    if ApplyMoireFilter:
        Projections = MoireFilter.ApplyMoireFilterOnProjections(
            Projections, Npar, M, dxi, 0.0 * (1 / dxi), 0.5 * (1 / dxi)
        )

    # =========================================================
    # Convolution Kernel definition & Projection convolution
    # =========================================================

    if Reconkernel == Kernels.SheppLogan:
        Kernel = Kernels.SheppLoganKernelSD(M, dxi)
    elif Reconkernel == Kernels.RamLak:
        Kernel = Kernels.RamLakKernelSD(M, dxi)
    else:
        raise ValueError(f"Unknown reconstruction kernel: {Reconkernel}")

    FilteredProjections = Convolution.ConvolveSinogram(Projections, Kernel, M, Npar, dxi)

    # =========================================================
    # Reconstruction
    # =========================================================

    ReconstructedImage = Backprojection.PixDrivenBackProjParCartesian(
        theta0, dtheta, Npar, N2pi, xi0, dxi, M, FilteredProjections,
        x0, dx, Nx, y0, dy, Ny,
    )

    ReconstructedImageHU = (ReconstructedImage - Constants.MuW) / Constants.MuW * 1000.0

    if Outputdirectory is not None:
        _Save(Outputdirectory, "Phantom", Voxelphantom)
        _Save(Outputdirectory, "Projections2DParallelBeam", RawProjections)
        _Save(Outputdirectory, "Projections2DFiltered", FilteredProjections)
        _Save(Outputdirectory, "Reconstruction", ReconstructedImage)
        _Save(Outputdirectory, "ReconstructionHU", ReconstructedImageHU)

    if ShowFigures:
        _ShowFigures(Voxelphantom, Nxsim, Nysim, RawProjections, FilteredProjections,
                     M, Npar, Kernel, ReconstructedImageHU, Nx, Ny)

    return {
        "Voxelphantom": Voxelphantom,
        "isSpectral": isSpectral,
        "Projections": RawProjections,
        "Kernel": Kernel,
        "FilteredProjections": FilteredProjections,
        "ReconstructedImage": ReconstructedImage,
        "ReconstructedImageHU": ReconstructedImageHU,
        "Geometry": {
            "RM": RM, "M": M, "Npar": Npar, "N2pi": N2pi,
            "dxi": dxi, "xi0": xi0, "theta0": theta0, "dtheta": dtheta,
            "Nx": Nx, "Ny": Ny, "dx": dx, "dy": dy, "x0": x0, "y0": y0,
        },
    }


def _Save(Outputdirectory, name, array):
    from pathlib import Path

    path = Path(Outputdirectory)
    path.mkdir(parents=True, exist_ok=True)
    # MATLAB writes with fwrite(..., 'double'), so keep little-endian float64 on disk.
    np.asarray(array, dtype=np.float64).tofile(path / name)


def _ShowFigures(Voxelphantom, Nxsim, Nysim, Projections, FilteredProjections,
                 M, Npar, Kernel, ReconstructedImageHU, Nx, Ny):
    import matplotlib.pyplot as plt

    figures = [
        ("Phantom", Voxelphantom.reshape(Nysim, Nxsim), None),
        ("Sinogram", Projections.reshape(Npar, M), None),
        ("Filtered sinogram", FilteredProjections.reshape(Npar, M), None),
        ("Reconstruction [HU]", ReconstructedImageHU.reshape(Ny, Nx), (-200, 400)),
    ]
    for title, image, clim in figures:
        plt.figure()
        plt.imshow(image, cmap="gray")
        if clim is not None:
            plt.clim(*clim)
        plt.title(title)
        plt.colorbar()

    plt.figure()
    plt.plot(Kernel)
    plt.title("Reconstruction kernel")
    plt.show()


if __name__ == "__main__":
    result = main(ShowFigures=True)
    recon = result["ReconstructedImageHU"]
    print(f"Reconstruction: min {recon.min():.1f} HU, max {recon.max():.1f} HU")
