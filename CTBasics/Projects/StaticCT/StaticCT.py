"""2D static CT simulation and reconstruction.

Port of Projects/Static CT/StaticCT.m. The source ring and the detector ring are
sampled independently, rebinned to parallel geometry and reconstructed with a standard
filtered backprojection.
"""
import numpy as np

from ...LecturesAndTestScripts.CTRecon101 import _Save
from ...Reconstruction.Backprojection import Backprojection
from ...Reconstruction.Constants import Constants
from ...Reconstruction.Convolution import Convolution
from ...Reconstruction.ForwardProjection import ForwardProjection
from ...Reconstruction.Kernels import Kernels
from ...Reconstruction.Rebinning import Rebinning
from ...Simulation.CreatePhantom import CreatePhantom


def main(Outputdirectory=None, ShowFigures=False, Reconkernel=Kernels.RamLak,
         FFTConvolution=False, SelectedPhantom="SimpleEllipseWithHips",
         MDelta=2048, MbetaDash=2048, N2pi=1024 // 2,
         Nxsim=256, Nysim=256, Nx=256, Ny=256, Nz=10):
    # =========================================================
    # Static CT ring geometry
    # =========================================================

    RF = 1150 / 2
    RD = 850 / 2

    dDelta = 2 * np.pi / MDelta
    delta0 = 0

    dbetaDash = 2 * np.pi / MbetaDash
    betaDash0 = -np.pi

    # The fan is limited by an arbitrary cut-off ray rather than by the detector width.
    arbitraryLimitangle = np.pi / 2
    betaMax = np.arctan(np.sin(arbitraryLimitangle) * RD
                        / (np.cos(arbitraryLimitangle) * RD + RF))
    RM = np.sin(betaMax) * RF

    alphaN = np.pi + 2 * betaMax
    alpha0 = -betaMax
    N = int(N2pi / (2 * np.pi) * alphaN)
    dalpha = alphaN / N

    dxi = dDelta * RF
    xi0 = -RD - dxi
    MPar = int(2 * RD / dxi + 1)

    thetaN = alphaN - 2 * betaMax
    NPar = int(N2pi / (2 * np.pi) * thetaN)
    theta0 = alpha0 + betaMax
    dtheta = thetaN / NPar

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
    # Phantom and rawdata simulation
    # =========================================================

    Voxelphantom, isSpectral = CreatePhantom.CreateVoxelPhantom2D(
        Nxsim, dxsim, x0sim, Nysim, dysim, y0sim, SelectedPhantom
    )

    Projections = ForwardProjection.RayDrivenForwardProj2DStaticCTMultiple(
        RF, RD, alpha0, dalpha, delta0, dDelta, x0sim, dxsim, Nxsim,
        y0sim, dysim, Nysim, Voxelphantom, MDelta, N
    )
    StaticCTProjections = Projections

    # =========================================================
    # Rebinning to parallel geometry
    # =========================================================

    Projections = Rebinning.RebinningStaticCTToParallel(
        Projections, RF, RD, MDelta, N, alpha0, dalpha, delta0, dDelta,
        MPar, NPar, theta0, dtheta, xi0, dxi, dbetaDash, betaDash0, MbetaDash
    )
    RebinnedProjections = np.array(Projections, copy=True)

    # =========================================================
    # Filtering
    # =========================================================

    if not FFTConvolution:
        if Reconkernel == Kernels.SheppLogan:
            Kernel = Kernels.SheppLoganKernelSD(MPar, dxi)
        elif Reconkernel == Kernels.RamLak:
            Kernel = Kernels.RamLakKernelSD(MPar, dxi)
        else:
            raise ValueError(f"Unknown reconstruction kernel: {Reconkernel}")

        Projections = Convolution.ConvolveSinogram(Projections, Kernel, MPar, NPar, dxi)
    else:
        NFFT = 2 * MPar
        Kernel = Kernels.RamLakKernelSD(MPar, dxi)[:NFFT]
        # The real space kernel is symmetric around its centre; fftshift moves the centre
        # to index 0, which is the representation the FFT expects.
        RamlakKernelFD = np.fft.fft(np.fft.fftshift(Kernel))[:NFFT] * dxi
        Projections = Kernels.PerformFilteringInFrequencyDomain(
            Projections, NPar, MPar, NFFT, RamlakKernelFD
        )

    FilteredProjections = Projections

    # =========================================================
    # Reconstruction
    # =========================================================

    ReconstructedImage = Backprojection.PixDrivenBackProjParCartesian(
        theta0, dtheta, NPar, N2pi, xi0, dxi, MPar, FilteredProjections,
        x0, dx, Nx, y0, dy, Ny
    )
    ReconstructedImageHU = (ReconstructedImage - Constants.MuW) / Constants.MuW * 1000

    if Outputdirectory is not None:
        _Save(Outputdirectory, "Phantom", Voxelphantom)
        _Save(Outputdirectory, "ProjectionStaticCT", StaticCTProjections)
        _Save(Outputdirectory, "ProjectiondataParallelBeamRebinned", RebinnedProjections)
        _Save(Outputdirectory, "Projections2DFiltered", FilteredProjections)
        _Save(Outputdirectory, "Reconstruction", ReconstructedImage)
        _Save(Outputdirectory, "ReconstructionHU", ReconstructedImageHU)

    Geometry = {
        "RF": RF, "RD": RD, "RM": RM, "betaMax": betaMax,
        "MDelta": MDelta, "dDelta": dDelta, "delta0": delta0,
        "MbetaDash": MbetaDash, "dbetaDash": dbetaDash, "betaDash0": betaDash0,
        "N": N, "N2pi": N2pi, "alpha0": alpha0, "dalpha": dalpha, "alphaN": alphaN,
        "MPar": MPar, "NPar": NPar, "theta0": theta0, "dtheta": dtheta,
        "thetaN": thetaN, "xi0": xi0, "dxi": dxi,
        "Nxsim": Nxsim, "Nysim": Nysim, "dxsim": dxsim, "dysim": dysim,
        "x0sim": x0sim, "y0sim": y0sim,
        "Nx": Nx, "Ny": Ny, "Nz": Nz, "dx": dx, "dy": dy, "x0": x0, "y0": y0,
    }

    if ShowFigures:
        _ShowFigures(Voxelphantom, StaticCTProjections, RebinnedProjections, Kernel,
                     FilteredProjections, ReconstructedImageHU, Geometry)

    return {
        "Voxelphantom": Voxelphantom,
        "isSpectral": isSpectral,
        "StaticCTProjections": StaticCTProjections,
        "RebinnedProjections": RebinnedProjections,
        "Kernel": Kernel,
        "FilteredProjections": FilteredProjections,
        "ReconstructedImage": ReconstructedImage,
        "ReconstructedImageHU": ReconstructedImageHU,
        "Geometry": Geometry,
    }


def _ShowFigures(Voxelphantom, StaticCTProjections, RebinnedProjections, Kernel,
                 FilteredProjections, ReconstructedImageHU, Geometry):
    import matplotlib.pyplot as plt

    def Show(image, title, **kwargs):
        plt.figure()
        plt.imshow(image, cmap="gray", aspect="auto", **kwargs)
        plt.title(title)

    Show(Voxelphantom.reshape(Geometry["Nysim"], Geometry["Nxsim"]), "Phantom")
    Show(np.asarray(StaticCTProjections).reshape(Geometry["N"], Geometry["MDelta"]),
         "Static CT sinogram")
    Show(np.asarray(RebinnedProjections).reshape(Geometry["NPar"], Geometry["MPar"]),
         "Rebinned parallel sinogram")

    plt.figure()
    plt.plot(Kernel)
    plt.title("Reconstruction kernel")

    Show(np.asarray(FilteredProjections).reshape(Geometry["NPar"], Geometry["MPar"]),
         "Filtered sinogram")
    Show(ReconstructedImageHU.reshape(Geometry["Ny"], Geometry["Nx"]),
         "Reconstruction [HU]", vmin=-200, vmax=400)
    plt.show()


if __name__ == "__main__":
    result = main()
    print(f"Reconstruction range {result['ReconstructedImageHU'].min():.0f} .. "
          f"{result['ReconstructedImageHU'].max():.0f} HU")
