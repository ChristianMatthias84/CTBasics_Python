"""Simulate and reconstruct 2D in fan beam geometry with noise and a polychromatic spectrum.

Port of LecturesAndTestScripts/CTRecon103.m. Figures and the raw binary dumps are
optional so the script runs without a CTBasicsConfig.txt.
"""
import numpy as np

from CTBasics.Preprocessing.WBHC import WBHC
from CTBasics.Reconstruction.Backprojection import Backprojection
from CTBasics.Reconstruction.Constants import Constants
from CTBasics.Reconstruction.Convolution import Convolution
from CTBasics.Reconstruction.ForwardProjection import ForwardProjection
from CTBasics.Reconstruction.Kernels import Kernels
from CTBasics.Reconstruction.Rebinning import Rebinning
from CTBasics.Simulation.CreatePhantom import CreatePhantom
from CTBasics.Simulation.Noise import Noise
from CTBasics.Simulation.Spectra import Spectra
from CTBasics.LecturesAndTestScripts.CTRecon101 import _Save


def main(Outputdirectory=None, ShowFigures=False, Reconkernel=Kernels.RamLak,
         FFTConvolution=False, SimNoise=True, WBHCorr=True, N_I0=1000000,
         SelectedPhantom="WaterPhantomSpectral20_CBinsert",
         spectrum=Spectra.kV140, E0=20, deltaE=10, N_Ebins=12,
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

    alphaN = np.pi + 2 * betaMax
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
    x0sim = -RM - dxsim / 2.0
    y0sim = -RM - dysim / 2.0

    # Reconstruction geometry.
    dx = 2 * RM / Nx
    dy = 2 * RM / Ny
    x0 = -RM - dx / 2.0
    y0 = -RM - dy / 2.0

    # =========================================================
    # Phantom Definition & Phantom Creation
    # =========================================================

    Voxelphantom, isSpectral = CreatePhantom.CreateVoxelPhantom2D(
        Nxsim, dxsim, x0sim, Nysim, dysim, y0sim, SelectedPhantom
    )

    # =========================================================
    # Rawdata Simulation
    # =========================================================

    if not isSpectral:
        FanProjections = ForwardProjection.RayDrivenForwardProj2DFanMultiple(
            RF, alpha0, dalpha, beta, x0sim, dxsim, Nxsim, y0sim, dysim, Nysim,
            Voxelphantom, MFan, NFan,
        )
    else:
        FanProjections = ForwardProjection.SpectralRayDrivenForwardProj2DFanMultiple(
            RF, alpha0, dalpha, beta, x0sim, dxsim, Nxsim, y0sim, dysim, Nysim,
            Voxelphantom, MFan, NFan, spectrum, E0, N_Ebins, deltaE,
        )

    Projections = FanProjections

    # =========================================================
    # WBHC correction
    # =========================================================

    # MATLAB always builds the LUT so EffectiveMuWater is defined, even without correction.
    _, EffectiveMuWater = WBHC.LUTBasedWBHCCorrection(
        spectrum, E0, N_Ebins, deltaE, Projections
    )
    if isSpectral and WBHCorr:
        Projections, EffectiveMuWater = WBHC.LUTBasedWBHCCorrection(
            spectrum, E0, N_Ebins, deltaE, Projections
        )

    CorrectedProjections = Projections

    # =========================================================
    # Rebinning
    # =========================================================

    Projections = Rebinning.RebinningFanToParallel(
        Projections, RF, MFan, NFan, alpha0, dalpha, beta,
        M, Npar, theta0, dtheta, xi0, dxi,
    )
    RebinnedProjections = Projections.copy()

    # =========================================================
    # Noise Simulation
    # =========================================================

    if SimNoise:
        Projections = Noise.ApplyPoissonNoise(Projections, M * Npar, N_I0)

    NoisyProjections = Projections

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
        RamlakKernelSD = Kernels.RamLakKernelSD(M, dxi)[:NFFT]
        RamlakKernelSD = np.fft.fftshift(RamlakKernelSD)
        RamlakKernelFD = np.fft.fft(RamlakKernelSD)[:NFFT] * dxi
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

    if not isSpectral:
        ReconstructedImageHU = (ReconstructedImage - Constants.MuW) / Constants.MuW * 1000
    else:
        # The polychromatic scan has a different water level, so HU scales on it.
        ReconstructedImageHU = (ReconstructedImage - EffectiveMuWater) / EffectiveMuWater * 1000

    if Outputdirectory is not None:
        _Save(Outputdirectory, "Phantom", Voxelphantom)
        _Save(Outputdirectory, "Projection2DFanBeam", FanProjections)
        _Save(Outputdirectory, "ProjectiondataParallelBeamRebinned", RebinnedProjections)
        if SimNoise:
            _Save(Outputdirectory, "Projections2D", NoisyProjections)
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
        "CorrectedProjections": CorrectedProjections,
        "RebinnedProjections": RebinnedProjections,
        "NoisyProjections": NoisyProjections,
        "EffectiveMuWater": EffectiveMuWater,
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
    from CTBasics.LecturesAndTestScripts.CTRecon102 import _ShowFigures as _Show

    _Show(Voxelphantom, Nxsim, Nysim, FanProjections, MFan, NFan,
          RebinnedProjections, FilteredProjections, M, Npar,
          Kernel, ReconstructedImageHU, Nx, Ny)


if __name__ == "__main__":
    result = main(ShowFigures=True)
    recon = result["ReconstructedImageHU"]
    print(f"EffectiveMuWater: {result['EffectiveMuWater']:.5f} /mm")
    print(f"Reconstruction: min {recon.min():.1f} HU, max {recon.max():.1f} HU")
