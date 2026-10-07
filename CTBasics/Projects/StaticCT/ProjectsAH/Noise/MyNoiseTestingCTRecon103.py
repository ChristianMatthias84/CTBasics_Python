"""Noise study based on the spectral fan beam reconstruction of CTRecon103.

Port of Projects/Static CT/Projects AH/Noise/MyNoiseTestingCTRecon103.m.
"""
import math

import numpy as np

from .....LecturesAndTestScripts.CTRecon101 import _Save
from .....Preprocessing.WBHC import WBHC
from .....ReadAndWrite.IOHelperFunctions import IOHelperFunctions
from .....Reconstruction.Backprojection import Backprojection
from .....Reconstruction.Constants import Constants
from .....Reconstruction.Convolution import Convolution
from .....Reconstruction.ForwardProjection import ForwardProjection
from .....Reconstruction.Kernels import Kernels
from .....Reconstruction.Rebinning import Rebinning
from .....Simulation.CreatePhantom import CreatePhantom
from .....Simulation.Noise import Noise
from .....Simulation.Spectra import Spectra


def main(Outputdirectory=None, ShowFigures=False, Reconkernel=Kernels.RamLak,
         FFTConvolution=False, SimNoise=True, WBHCorr=True, N_I0=1000000,
         MFan=500, N2pi=512, Nxsim=256, Nysim=256, Nx=256, Ny=256,
         spectrum=Spectra.kV140, E0=20, deltaE=10, N_Ebins=12,
         SelectedPhantom="WaterPhantomSpectral20_CBinsert"):
    if Outputdirectory is None:
        Outputdirectory = IOHelperFunctions().GlobalOutputFolderPath

    RF = 595.0
    RD = 490.6

    betaMax = 24.8 / 180 * math.pi
    RM = math.sin(betaMax) * RF  # Radius of the field of measurement (FOM).

    dbeta = -2 * betaMax / (MFan - 1)
    beta = betaMax + dbeta * np.arange(MFan)

    alphaN = math.pi + 2 * betaMax
    NFan = int(N2pi / (2 * math.pi) * alphaN)
    alpha0 = -betaMax
    dalpha = alphaN / NFan

    M = MFan
    thetaN = alphaN - 2 * betaMax
    Npar = int(N2pi / (2 * math.pi) * thetaN)
    dxi = 2 * RM / (M - 1)
    xi0 = -RM - dxi
    theta0 = alpha0 + betaMax
    dtheta = thetaN / Npar

    dxsim = 2 * RM / Nxsim
    dysim = 2 * RM / Nysim
    x0sim = -RM - dxsim / 2.0
    y0sim = -RM - dysim / 2.0

    dx = 2 * RM / Nx
    dy = 2 * RM / Ny
    x0 = -RM - dx / 2.0
    y0 = -RM - dy / 2.0

    Voxelphantom, isSpectral = CreatePhantom.CreateVoxelPhantom2D(
        Nxsim, dxsim, x0sim, Nysim, dysim, y0sim, SelectedPhantom)
    _Save(Outputdirectory, "Phantom", Voxelphantom)

    if not isSpectral:
        Projections = ForwardProjection.RayDrivenForwardProj2DFanMultiple(
            RF, alpha0, dalpha, beta, x0sim, dxsim, Nxsim, y0sim, dysim, Nysim,
            Voxelphantom, MFan, NFan)
    else:
        Projections = ForwardProjection.SpectralRayDrivenForwardProj2DFanMultiple(
            RF, alpha0, dalpha, beta, x0sim, dxsim, Nxsim, y0sim, dysim, Nysim,
            Voxelphantom, MFan, NFan, spectrum, E0, N_Ebins, deltaE)

    _Save(Outputdirectory, "Projection2DFanBeam", Projections)
    FanProjections = np.array(Projections, copy=True)

    # MATLAB always evaluates the LUT once to obtain EffectiveMuWater, even without WBHC.
    _, EffectiveMuWater = WBHC.LUTBasedWBHCCorrection(
        spectrum, E0, N_Ebins, deltaE, Projections)

    if isSpectral and WBHCorr:
        Projections, EffectiveMuWater = WBHC.LUTBasedWBHCCorrection(
            spectrum, E0, N_Ebins, deltaE, Projections)

    Projections = Rebinning.RebinningFanToParallel(
        Projections, RF, MFan, NFan, alpha0, dalpha, beta,
        M, Npar, theta0, dtheta, xi0, dxi)
    _Save(Outputdirectory, "ProjectiondataParallelBeamRebinned", Projections)
    RebinnedProjections = np.array(Projections, copy=True)

    if SimNoise:
        Projections = Noise.ApplyPoissonNoise(Projections, M * Npar, N_I0)
        _Save(Outputdirectory, "Projections2D", Projections)

    NoisyProjections = np.array(Projections, copy=True)

    if not FFTConvolution:
        Kernel = None
        if Reconkernel == Kernels.SheppLogan:
            Kernel = Kernels.SheppLoganKernelSD(M, dxi)
        elif Reconkernel == Kernels.RamLak:
            Kernel = Kernels.RamLakKernelSD(M, dxi)
        Projections = Convolution.ConvolveSinogram(Projections, Kernel, M, Npar, dxi)
    else:
        NFFT = 2 * M
        # The kernel is defined symmetrically around zero, so only the first NFFT samples
        # are used before the shift into the FFT representation.
        RamlakKernelSD = Kernels.RamLakKernelSD(M, dxi)[:NFFT]
        RamlakKernelSD = np.fft.fftshift(RamlakKernelSD)
        RamlakKernelFD = np.fft.fft(RamlakKernelSD)
        # Every discretized Fourier transform carries an extra dxi compared to the FFT.
        Kernel = RamlakKernelFD[:NFFT] * dxi
        Projections = Kernels.PerformFilteringInFrequencyDomain(
            Projections, Npar, M, NFFT, Kernel)

    _Save(Outputdirectory, "Projections2DFiltered", Projections)

    ReconstructedImage = Backprojection.PixDrivenBackProjParCartesian(
        theta0, dtheta, Npar, N2pi, xi0, dxi, M, Projections,
        x0, dx, Nx, y0, dy, Ny)

    if not isSpectral:
        ReconstructedImageHU = (ReconstructedImage - Constants.MuW) / Constants.MuW * 1000
    else:
        ReconstructedImageHU = (ReconstructedImage - EffectiveMuWater) / EffectiveMuWater * 1000

    _Save(Outputdirectory, "Reconstruction", ReconstructedImage)
    _Save(Outputdirectory, "ReconstructionHU", ReconstructedImageHU)

    if ShowFigures:
        from .....LecturesAndTestScripts.CTRecon102 import _ShowFigures
        _ShowFigures(Voxelphantom, Nxsim, Nysim, FanProjections, MFan, NFan,
                     RebinnedProjections, M, Npar, Kernel, Projections,
                     ReconstructedImage, Nx, Ny)

    return {
        "Voxelphantom": Voxelphantom,
        "isSpectral": isSpectral,
        "FanProjections": FanProjections,
        "RebinnedProjections": RebinnedProjections,
        "NoisyProjections": NoisyProjections,
        "EffectiveMuWater": EffectiveMuWater,
        "Kernel": Kernel,
        "FilteredProjections": Projections,
        "ReconstructedImage": ReconstructedImage,
        "ReconstructedImageHU": ReconstructedImageHU,
        "Geometry": {"RF": RF, "RD": RD, "RM": RM, "betaMax": betaMax, "beta": beta,
                     "MFan": MFan, "NFan": NFan, "alpha0": alpha0, "dalpha": dalpha,
                     "alphaN": alphaN, "M": M, "Npar": Npar, "N2pi": N2pi,
                     "theta0": theta0, "dtheta": dtheta, "thetaN": thetaN,
                     "xi0": xi0, "dxi": dxi,
                     "x0": x0, "dx": dx, "Nx": Nx, "y0": y0, "dy": dy, "Ny": Ny},
    }
