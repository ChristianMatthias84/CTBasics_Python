"""Over-scan fan beam simulation and reconstruction with redundancy weighting.

Port of LecturesAndTestScripts/TestScript2D.m. In contrast to CTRecon102 this scans more
than a full turn and uses the weighted FBP that handles the redundant views. MATLAB loads
a DICOM slice by default; because that data is not part of the repository the simulated
phantom is the default here.
"""
import numpy as np

from CTBasics.ImageManipulation.ImageManipulation import ImageManipulation
from CTBasics.Preprocessing.WBHC import WBHC
from CTBasics.ReadAndWrite.IOHelperFunctions import IOHelperFunctions
from CTBasics.Reconstruction.Backprojection import Backprojection
from CTBasics.Reconstruction.Constants import Constants
from CTBasics.Reconstruction.Convolution import Convolution
from CTBasics.Reconstruction.ForwardProjection import ForwardProjection
from CTBasics.Reconstruction.Kernels import Kernels
from CTBasics.Reconstruction.MoireFilter import MoireFilter
from CTBasics.Reconstruction.Rebinning import Rebinning
from CTBasics.Simulation.CreatePhantom import CreatePhantom
from CTBasics.Simulation.Noise import Noise
from CTBasics.Simulation.Spectra import Spectra
from CTBasics.LecturesAndTestScripts.CTRecon101 import _Save


def main(Outputdirectory=None, ShowFigures=False, Reconkernel=Kernels.RamLak,
         FFTConvolution=False, KerneldefinitionInFD=False, SimNoise=False,
         ApplyMoireFilter=False, PatientData=True, WBHCCorrection=False,
         N_I0=1000000, DataDirectory=None,
         SelectedPhantom="SpectralHipWithMetal",
         spectrum=Spectra.kV120, E0=20, deltaE=10, N_Ebins=12,
         N2pi=512, MFan=500, Nxsim=256, Nysim=256, Nx=256, Ny=256):
    # =========================================================
    # Parameter Definition
    # =========================================================

    RF = 595.0
    RD = 490.6
    betaMax = 24.8 / 180 * np.pi
    RM = np.sin(betaMax) * RF
    dbeta = -2 * betaMax / (MFan - 1)
    beta = betaMax + dbeta * np.arange(MFan)

    # Over-scan: a full turn plus the fan angle plus another quarter turn.
    alphaN = 2.0 * np.pi + 2 * betaMax + np.pi / 2
    alphaN = max(np.pi + 2 * betaMax, alphaN)
    NFan = int(N2pi / (2 * np.pi) * alphaN)
    alpha0 = -betaMax
    dalpha = alphaN / NFan

    M = MFan
    thetaN = alphaN - 2 * betaMax
    Npar = int(N2pi / (2 * np.pi) * thetaN)

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

    # =========================================================
    # Phantom Definition & Phantom Creation
    # =========================================================

    if not PatientData:
        Voxelphantom, isSpectral = CreatePhantom.CreateVoxelPhantom2D(
            Nxsim, dxsim, x0sim, Nysim, dysim, y0sim, SelectedPhantom
        )
    else:
        from pathlib import Path

        if DataDirectory is None:
            DataDirectory = IOHelperFunctions().GlobalDataPath

        isSpectral = False
        V_HU, V_info, V_mapped = IOHelperFunctions.read_Dicom_Slice(
            str(Path(DataDirectory) / "MAR2D" / "hip2.ima")
        )
        V_HU = ImageManipulation.Resample2D(np.asarray(V_HU, dtype=float), Nxsim, Nysim)
        # MATLAB `X = V'; X(:)'` is the row-major flattening of V.
        Voxelphantom = V_HU.ravel()
        # HU back to attenuation.
        Voxelphantom = (Voxelphantom + 1000) * Constants.MuW / 1000

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

    EffectiveMuWater = Constants.MuW
    if isSpectral and WBHCCorrection:
        Projections, EffectiveMuWater = WBHC.LUTBasedWBHCCorrection(
            spectrum, E0, N_Ebins, deltaE, Projections
        )

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

    if ApplyMoireFilter:
        Projections = MoireFilter.ApplyMoireFilterOnProjections(
            Projections, Npar, M, dxi, 0.1 * (1 / dxi), 0.5 * (1 / dxi)
        )

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
        # NFFT = 2*M because the detector signal is zero padded to avoid circular convolution.
        NFFT = 2 * M

        if KerneldefinitionInFD:
            from CTBasics.Memory.Allocation import Allocation

            RamlakKernelFD = Kernels.RamLakKernelFD(NFFT, dxi, Allocation.ones(NFFT))
            RamlakKernelSD = np.fft.ifft(RamlakKernelFD)
            # Limit the kernel support in image space; beyond 2*M the signal is zero anyway.
            zeroPadding = int(0.5 * NFFT)
            RamlakKernelSD[NFFT // 2 - zeroPadding // 2:NFFT // 2 + zeroPadding // 2 + 1] = 0
            # No dxi factor here because ifft and fft cancel out.
            RamlakKernelFD = np.fft.fft(RamlakKernelSD)
        else:
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

    # Weighted FBP because the over-scan measures some rays more than once.
    ReconstructedImage = Backprojection.WFBP2DFullyPixelDriven(
        theta0, dtheta, Npar, N2pi, xi0, dxi, M, FilteredProjections,
        x0, dx, Nx, y0, dy, Ny,
    )

    if not isSpectral:
        ReconstructedImageHU = (ReconstructedImage - Constants.MuW) / Constants.MuW * 1000
    else:
        ReconstructedImageHU = (ReconstructedImage - EffectiveMuWater) / EffectiveMuWater * 1000

    if Outputdirectory is not None:
        _Save(Outputdirectory, "Patientdata" if PatientData else "Phantom", Voxelphantom)
        _Save(Outputdirectory, "Projection2DFanBeam", FanProjections)
        _Save(Outputdirectory, "ProjectiondataParallelBeamRebinned", RebinnedProjections)
        _Save(Outputdirectory, "Projections2DFiltered", FilteredProjections)
        _Save(Outputdirectory, "Reconstruction", ReconstructedImage)
        _Save(Outputdirectory, "ReconstructionHU", ReconstructedImageHU)

    if ShowFigures:
        from CTBasics.LecturesAndTestScripts.CTRecon102 import _ShowFigures

        _ShowFigures(Voxelphantom, Nxsim, Nysim, FanProjections, MFan, NFan,
                     RebinnedProjections, FilteredProjections, M, Npar,
                     Kernel, ReconstructedImageHU, Nx, Ny)

    return {
        "Voxelphantom": Voxelphantom,
        "isSpectral": isSpectral,
        "FanProjections": FanProjections,
        "RebinnedProjections": RebinnedProjections,
        "EffectiveMuWater": EffectiveMuWater,
        "Kernel": Kernel,
        "FilteredProjections": FilteredProjections,
        "ReconstructedImage": ReconstructedImage,
        "ReconstructedImageHU": ReconstructedImageHU,
        "Geometry": {
            "RF": RF, "RD": RD, "RM": RM, "betaMax": betaMax, "beta": beta,
            "MFan": MFan, "NFan": NFan, "alpha0": alpha0, "dalpha": dalpha, "alphaN": alphaN,
            "M": M, "Npar": Npar, "N2pi": N2pi,
            "dxi": dxi, "xi0": xi0, "theta0": theta0, "dtheta": dtheta,
            "Nx": Nx, "Ny": Ny, "dx": dx, "dy": dy, "x0": x0, "y0": y0,
        },
    }


if __name__ == "__main__":
    result = main(ShowFigures=True)
    recon = result["ReconstructedImageHU"]
    print(f"Reconstruction: min {recon.min():.1f} HU, max {recon.max():.1f} HU")
