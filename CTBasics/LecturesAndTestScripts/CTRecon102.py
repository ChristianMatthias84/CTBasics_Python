"""Simulate and reconstruct in 2D fan beam geometry.

Port of LecturesAndTestScripts/CTRecon102.m. The MATLAB script shows figures and writes
raw binaries to the configured output folder; here both are optional so the script runs
without a CTBasicsConfig.txt.
"""
import numpy as np

from ..Memory.Allocation import Allocation
from ..Reconstruction.Backprojection import Backprojection
from ..Reconstruction.Constants import Constants
from ..Reconstruction.Convolution import Convolution
from ..Reconstruction.ForwardProjection import ForwardProjection
from ..Reconstruction.Kernels import Kernels
from ..Reconstruction.Rebinning import Rebinning
from ..Simulation.CreatePhantom import CreatePhantom
from .CTRecon101 import _Save


def main(Outputdirectory=None, ShowFigures=False, Reconkernel=Kernels.RamLak,
         FFTConvolution=False, SelectedPhantom="SimpleEllipseWithHips",
         N2pi=512, MFan=500, Nxsim=256, Nysim=256, Nx=256, Ny=256):
    # =========================================================
    # Parameter Definition
    # =========================================================

    # Fan beam geometry.
    RF = 595.0
    RD = 490.6
    betaMax = 24.8 / 180 * np.pi
    RM = np.sin(betaMax) * RF           # Radius of the field of measurement (FOM).
    dbeta = -2 * betaMax / (MFan - 1)
    beta = betaMax + dbeta * np.arange(MFan)

    alphaN = np.pi + 2 * betaMax        # Short scan: half turn plus the fan angle.
    NFan = int(N2pi / (2 * np.pi) * alphaN)
    alpha0 = -betaMax
    dalpha = alphaN / NFan

    # Parallel beam scanner geometry, independent variables.
    M = MFan
    thetaN = alphaN - 2 * betaMax
    Npar = int(N2pi / (2 * np.pi) * thetaN)

    # Parallel beam scanner geometry, dependent variables.
    dxi = 2 * RM / (M - 1)
    xi0 = -RM - dxi
    theta0 = alpha0 + betaMax
    dtheta = thetaN / Npar

    # Simulation geometry.
    dxsim = 2 * RM / Nxsim
    dysim = 2 * RM / Nysim
    xcsim = 0.0
    ycsim = 0.0
    x0sim = -RM - dxsim / 2.0 + xcsim
    y0sim = -RM - dysim / 2.0 + ycsim

    # Reconstruction geometry.
    dx = 2 * RM / Nx
    dy = 2 * RM / Ny
    xc = 0.0
    yc = 0.0
    x0 = -RM - dx / 2.0 + xc
    y0 = -RM - dy / 2.0 + yc

    # =========================================================
    # Phantom Definition & Phantom Creation
    # =========================================================

    Voxelphantom, isSpectral = CreatePhantom.CreateVoxelPhantom2D(
        Nxsim, dxsim, x0sim, Nysim, dysim, y0sim, SelectedPhantom
    )

    # =========================================================
    # Rawdata Simulation
    # =========================================================

    FanProjections = ForwardProjection.RayDrivenForwardProj2DFanMultiple(
        RF, alpha0, dalpha, beta, x0sim, dxsim, Nxsim, y0sim, dysim, Nysim,
        Voxelphantom, MFan, NFan,
    )

    # =========================================================
    # Rebinning
    # =========================================================

    Projections = Rebinning.RebinningFanToParallel(
        FanProjections, RF, MFan, NFan, alpha0, dalpha, beta,
        M, Npar, theta0, dtheta, xi0, dxi,
    )
    RebinnedProjections = Projections.copy()

    # =========================================================
    # Convolution Kernel definition & Projection convolution
    # =========================================================

    Kernel = None
    if not FFTConvolution:
        if Reconkernel == Kernels.SheppLogan:
            Kernel = Kernels.SheppLoganKernelSD(M, dxi)
        elif Reconkernel == Kernels.RamLak:
            Kernel = Kernels.RamLakKernelSD(M, dxi)
        else:
            raise ValueError(f"Unknown reconstruction kernel: {Reconkernel}")

        Projections = Convolution.ConvolveSinogram(Projections, Kernel, M, Npar, dxi)
    else:
        NFFT = 2 * M
        # The space domain kernel is defined symmetrically on 2M+1 samples, so it is cut
        # to NFFT and rotated into the FFT ordering before transforming.
        RamlakKernelSD = Kernels.RamLakKernelSD(M, dxi)[:NFFT]
        RamlakKernelSD = np.fft.fftshift(RamlakKernelSD)
        RamlakKernelFD = np.fft.fft(RamlakKernelSD)
        # Every discretized Fourier transform carries an extra factor dxi over numpy's FFT.
        RamlakKernelFD = RamlakKernelFD[:NFFT] * dxi
        Kernel = RamlakKernelFD

        Projections = Kernels.PerformFilteringInFrequencyDomain(
            Projections, Npar, M, NFFT, RamlakKernelFD
        )

    FilteredProjections = Projections

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
        _Save(Outputdirectory, "Projection2DFanBeam", FanProjections)
        _Save(Outputdirectory, "ProjectiondataParallelBeamRebinned", RebinnedProjections)
        _Save(Outputdirectory, "Projections2DFiltered", FilteredProjections)
        _Save(Outputdirectory, "Reconstruction", ReconstructedImage)
        _Save(Outputdirectory, "ReconstructionHU", ReconstructedImageHU)

    if ShowFigures:
        _ShowFigures(Voxelphantom, Nxsim, Nysim, FanProjections, MFan, NFan,
                     RebinnedProjections, FilteredProjections, M, Npar,
                     Kernel, ReconstructedImageHU, Nx, Ny)

    return {
        "Voxelphantom": Voxelphantom,
        "isSpectral": isSpectral,
        "FanProjections": FanProjections,
        "RebinnedProjections": RebinnedProjections,
        "Kernel": Kernel,
        "FilteredProjections": FilteredProjections,
        "ReconstructedImage": ReconstructedImage,
        "ReconstructedImageHU": ReconstructedImageHU,
        "Geometry": {
            "RF": RF, "RD": RD, "RM": RM, "betaMax": betaMax, "beta": beta,
            "MFan": MFan, "NFan": NFan, "alpha0": alpha0, "dalpha": dalpha,
            "M": M, "Npar": Npar, "N2pi": N2pi,
            "dxi": dxi, "xi0": xi0, "theta0": theta0, "dtheta": dtheta,
            "Nx": Nx, "Ny": Ny, "dx": dx, "dy": dy, "x0": x0, "y0": y0,
        },
    }


def _ShowFigures(Voxelphantom, Nxsim, Nysim, FanProjections, MFan, NFan,
                 RebinnedProjections, FilteredProjections, M, Npar,
                 Kernel, ReconstructedImageHU, Nx, Ny):
    import matplotlib.pyplot as plt

    figures = [
        ("Phantom", Voxelphantom.reshape(Nysim, Nxsim), None),
        ("Fan beam sinogram", FanProjections.reshape(NFan, MFan), None),
        ("Rebinned parallel sinogram", RebinnedProjections.reshape(Npar, M), None),
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

    if Kernel is not None:
        plt.figure()
        plt.plot(np.abs(Kernel))
        plt.title("Reconstruction kernel")

    plt.show()


if __name__ == "__main__":
    result = main(ShowFigures=True)
    recon = result["ReconstructedImageHU"]
    print(f"Reconstruction: min {recon.min():.1f} HU, max {recon.max():.1f} HU")
