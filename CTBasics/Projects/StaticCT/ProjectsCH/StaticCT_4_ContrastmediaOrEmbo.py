"""Multi phase simulation of a temporally varying iodine distribution.

Port of Projects/Static CT/Projects CH/StaticCT_4_ContrastmediaOrEmbo.m.
"""
import math
from pathlib import Path

import numpy as np

from ....Memory.Allocation import Allocation
from ....MultiCore.Parallelization import Parallelization
from ....RawData.RawDataHelperFunctions import RawDataHelperFunctions
from ....ReadAndWrite.IOHelperFunctions import IOHelperFunctions
from ....Reconstruction.Backprojection import Backprojection
from ....Reconstruction.Constants import Constants
from ....Reconstruction.Convolution import Convolution
from ....Reconstruction.ForwardProjection import ForwardProjection
from ....Reconstruction.Kernels import Kernels
from ....Reconstruction.MoireFilter import MoireFilter
from ....Simulation.CreatePhantom import CreatePhantom
from ....Simulation.Noise import Noise
from ..ProjectsJP.StaticCT_createSubsetAnimationGif import \
    StaticCT_createSubsetAnimationGif
from ..StaticCTHelperFunctions.StaticCTSystemHelperFunctions import \
    StaticCTSystemHelperFunctions


def main(OutputBaseDir=None, ReconType="180", ReconKernel=Kernels.RamLak,
         ApplyMoireFilter=False, SimulateNoise=False, WriteData=True,
         ShowFinalRecons=True, isGIF=False, SaveRawRecon=True, SaveReconGifs=True,
         NSubsets=64, NRecons=8, NScansPerRecon=1, TRot=0.5,
         TemporalPhantomName="TemporalIodinePhantom", HeartRateBPM=60,
         MotionEnabled=True, MotionDirectionRad=0, MotionType="full", TRest=0,
         IodinePeakFactor=3.0, IodineSigmaMin=0.25, IodineSigmaMax=4.5,
         N_I0=100000 / 5, RM=60, M=500, N2pi=1024, Nx=512, Ny=512):
    if OutputBaseDir is None:
        OutputBaseDir = Path(IOHelperFunctions().GlobalOutputFolderPath) \
            / "temporalIodinePhantom" / "data"
    OutputBaseDir = Path(OutputBaseDir)

    Parallelization.Init()

    if ReconType == "180":
        thetaN = math.pi
        TReconInterval = NScansPerRecon * TRot / 2
    elif ReconType == "360":
        thetaN = 2 * math.pi
        TReconInterval = NScansPerRecon * TRot
    else:
        raise ValueError(f'Unknown ReconType "{ReconType}". Use "180" or "360".')

    Npi = N2pi // 2
    N = int(N2pi * thetaN / (2 * math.pi))

    dxi = 2 * RM / (M - 1)
    xi0 = -RM - dxi
    theta0 = 0
    dtheta = thetaN / N
    dT = TRot / N2pi

    dx = 2 * RM / Nx
    dy = 2 * RM / Ny
    x0 = -RM - dx / 2.0
    y0 = -RM - dy / 2.0

    THeart = 60 / HeartRateBPM
    TRecon = np.linspace(0, THeart, NRecons + 1)[:-1]
    TLeft = TRecon - TReconInterval / 2

    ThetaArrayArbitrary = StaticCTSystemHelperFunctions.SubsetInterleaving(
        N2pi, NSubsets, ReconType, 0)
    ThetaArrayLinear = np.linspace(0, 2 * math.pi - dtheta, N2pi)

    OutputDir = OutputBaseDir / (
        f"TemporalIodine_HR{HeartRateBPM}_TRot{round(TRot * 1000)}ms_NSub{NSubsets}"
        f"_NRecons{NRecons}_NScans{NScansPerRecon}_{ReconType}"
        f"_Ipf{IodinePeakFactor:.2f}_Sig{IodineSigmaMin:.2f}-{IodineSigmaMax:.2f}")
    OutputDir.mkdir(parents=True, exist_ok=True)

    ProjectionsArbitrary = [Allocation.zeros(NScansPerRecon, N * M)
                            for _ in range(NRecons)]
    ProjectionsLinear = [Allocation.zeros(NScansPerRecon, N * M) for _ in range(NRecons)]

    ReconGT = Allocation.zeros(NRecons, Nx * Ny)
    for k in range(1, NRecons + 1):
        ReconGT[k - 1, :] = _GroundTruthPhase(
            Nx, dx, x0, Ny, dy, y0, TRecon[k - 1], dT, TemporalPhantomName,
            HeartRateBPM, MotionEnabled, MotionDirectionRad, MotionType, TRest,
            IodinePeakFactor, IodineSigmaMin, IodineSigmaMax)

    if WriteData and SaveRawRecon:
        IOHelperFunctions.SaveDoubleToFile(
            OutputDir / (f"GroundTruthPhantomMu_HR{HeartRateBPM:.0f}_TRot{TRot:.2f}s"
                         f"_NSub{NSubsets}_NScans{NScansPerRecon}"),
            RawDataHelperFunctions.SimpleLinearizeArray(ReconGT))

    print("Forward projection for temporal iodine phases running...")
    for k in range(1, NRecons + 1):
        T0Phase = TLeft[k - 1]
        for j in range(1, NScansPerRecon + 1):
            if ReconType == "180":
                ThetaIdx = slice(Npi, N2pi) if j % 2 == 0 else slice(0, Npi)
                T0Scan = T0Phase + (j - 1) * (TRot / 2)
            else:
                ThetaIdx = slice(0, N2pi)
                T0Scan = T0Phase + (j - 1) * TRot

            Options = dict(Heartrate=HeartRateBPM, Motion=MotionEnabled,
                           Direction=MotionDirectionRad, T0=T0Scan,
                           SelectedPhantom=TemporalPhantomName, MotionType=MotionType,
                           TRest=TRest, IodinePeakFactor=IodinePeakFactor,
                           IodineSigmaMin=IodineSigmaMin, IodineSigmaMax=IodineSigmaMax)

            ProjArb, _, _, _ = ForwardProjection.\
                RayDrivenForwProjParCartesianMultipleWithMotionArbitraryOrderParallelized(
                    x0, dx, Nx, y0, dy, Ny, N, xi0, dxi, M, N2pi, TRot,
                    ThetaArrayArbitrary[ThetaIdx], **Options)
            ProjectionsArbitrary[k - 1][j - 1, :] = np.asarray(ProjArb).ravel()

            ProjLin, _, _, _ = ForwardProjection.\
                RayDrivenForwProjParCartesianMultipleWithMotionArbitraryOrderParallelized(
                    x0, dx, Nx, y0, dy, Ny, N, xi0, dxi, M, N2pi, TRot,
                    ThetaArrayLinear[ThetaIdx], **Options)
            ProjectionsLinear[k - 1][j - 1, :] = np.asarray(ProjLin).ravel()

    if SimulateNoise:
        print("Applying Poisson noise...")
        for k in range(NRecons):
            for j in range(NScansPerRecon):
                ProjectionsArbitrary[k][j, :] = Noise.ApplyPoissonNoise(
                    ProjectionsArbitrary[k][j, :], M * N, N_I0)
                ProjectionsLinear[k][j, :] = Noise.ApplyPoissonNoise(
                    ProjectionsLinear[k][j, :], M * N, N_I0)

    if ApplyMoireFilter:
        for k in range(NRecons):
            for j in range(NScansPerRecon):
                ProjectionsLinear[k][j, :] = MoireFilter.ApplyMoireFilterOnProjections(
                    ProjectionsLinear[k][j, :], N, M, dxi, 0.0 * (1 / dxi),
                    0.5 * (1 / dxi))
                ProjectionsArbitrary[k][j, :] = \
                    MoireFilter.ApplyMoireFilterOnProjections(
                        ProjectionsArbitrary[k][j, :], N, M, dxi, 0.0 * (1 / dxi),
                        0.5 * (1 / dxi))

    if ReconKernel == Kernels.SheppLogan:
        Kernel = Kernels.SheppLoganKernelSD(M, dxi)
    elif ReconKernel == Kernels.RamLak:
        Kernel = Kernels.RamLakKernelSD(M, dxi)
    else:
        raise ValueError("Unknown reconstruction kernel.")

    for k in range(NRecons):
        for j in range(NScansPerRecon):
            ProjectionsLinear[k][j, :] = Convolution.ConvolveSinogram(
                ProjectionsLinear[k][j, :], Kernel, M, N, dxi)
            ProjectionsArbitrary[k][j, :] = Convolution.ConvolveSinogram(
                ProjectionsArbitrary[k][j, :], Kernel, M, N, dxi)

    print("Backprojection for all phases running...")
    ReconLinear = Allocation.zeros(NRecons, Nx * Ny)
    ReconArbitrary = Allocation.zeros(NRecons, Nx * Ny)

    for k in range(1, NRecons + 1):
        ProjLin_k = RawDataHelperFunctions.SimpleLinearizeArray(ProjectionsLinear[k - 1])
        ProjArb_k = RawDataHelperFunctions.SimpleLinearizeArray(
            ProjectionsArbitrary[k - 1])

        ReconLinear[k - 1, :] = Backprojection.PixDrivenBackProjParCartesian(
            theta0, dtheta, NScansPerRecon * N, N2pi, xi0, dxi, M, ProjLin_k,
            x0, dx, Nx, y0, dy, Ny)
        ReconArbitrary[k - 1, :] = Backprojection.PixDrivenBackProjParCartesian(
            theta0, dtheta, NScansPerRecon * N, N2pi, xi0, dxi, M, ProjArb_k,
            x0, dx, Nx, y0, dy, Ny)

        # MATLAB recreates the ground truth here even though it was already filled
        # above; kept because it keeps the two code paths in sync.
        ReconGT[k - 1, :] = _GroundTruthPhase(
            Nx, dx, x0, Ny, dy, y0, TRecon[k - 1], dT, TemporalPhantomName,
            HeartRateBPM, MotionEnabled, MotionDirectionRad, MotionType, TRest,
            IodinePeakFactor, IodineSigmaMin, IodineSigmaMax)

    ReconHU_Linear = 1000 * (ReconLinear - Constants.MuW) / Constants.MuW
    ReconHU_Arbitrary = 1000 * (ReconArbitrary - Constants.MuW) / Constants.MuW
    ReconHU_GT = 1000 * (ReconGT - Constants.MuW) / Constants.MuW

    ReconHU_LinearMinusGT = ReconHU_Linear - ReconHU_GT
    ReconHU_ArbitraryMinusGT = ReconHU_Arbitrary - ReconHU_GT

    if ShowFinalRecons:
        _ShowFinalRecons(ReconHU_GT, ReconHU_Linear, ReconHU_Arbitrary, NRecons, Nx, Ny)

    t = np.linspace(0, max(TRecon.max(), THeart), 1000)
    Tcycle = 60 / HeartRateBPM
    TMotion = Tcycle - TRest
    if TMotion <= 0:
        raise ValueError(f"TRest ({TRest:.3f} s) must be smaller than cycle duration "
                         f"({Tcycle:.3f} s).")

    iodineTemporalSignal = IodinePeakFactor * _TemporalSignal(t, Tcycle, TMotion,
                                                              MotionType)

    iodineTemporalDiscrete = np.zeros(NRecons)
    scanMidSignal = np.zeros((NRecons, NScansPerRecon))
    TRight = TRecon + TReconInterval / 2

    for k in range(NRecons):
        iodineTemporalDiscrete[k] = IodinePeakFactor * CreatePhantom.MotionProfile(
            TRecon[k], HeartRateBPM, True, MotionType, TRest)
        for j in range(1, NScansPerRecon + 1):
            if ReconType == "180":
                tScanMid = TLeft[k] + (j - 0.5) * (TRot / 2)
            else:
                tScanMid = TLeft[k] + (j - 0.5) * TRot
            scanMidSignal[k, j - 1] = IodinePeakFactor * CreatePhantom.MotionProfile(
                tScanMid, HeartRateBPM, True, MotionType, TRest)

    figSignal = _PlotTemporalSignal(
        t, iodineTemporalSignal, TRecon, TLeft, TRight, iodineTemporalDiscrete,
        scanMidSignal, NScansPerRecon, ReconType, TRot, THeart, Tcycle, TMotion,
        MotionType, IodinePeakFactor, show=ShowFinalRecons)

    if WriteData:
        Results = {
            "ReconType": ReconType, "NSubsets": NSubsets, "NRecons": NRecons,
            "NScansPerRecon": NScansPerRecon, "TRecon": TRecon,
            "HeartRateBPM": HeartRateBPM, "MotionEnabled": MotionEnabled,
            "MotionDirectionRad": MotionDirectionRad, "MotionType": MotionType,
            "TRest": TRest, "IodinePeakFactor": IodinePeakFactor,
            "IodineSigmaMin": IodineSigmaMin, "IodineSigmaMax": IodineSigmaMax,
            "ReconHU_GT": ReconHU_GT, "ReconHU_Linear": ReconHU_Linear,
            "ReconHU_Arbitrary": ReconHU_Arbitrary,
            "ReconHU_LinearMinusGT": ReconHU_LinearMinusGT,
            "ReconHU_ArbitraryMinusGT": ReconHU_ArbitraryMinusGT,
            "ProjectionsLinear": np.array(ProjectionsLinear),
            "ProjectionsArbitrary": np.array(ProjectionsArbitrary),
            # MATLAB leaves the ground truth projections empty.
            "ProjectionsGT": np.array([]),
        }
        IOHelperFunctions.SaveStructToMat(
            OutputDir / "TemporalIodine_MultiphaseResults.mat", Results)

        if SaveReconGifs:
            print("Saving reconstructions as animated GIFs...")
            gif_fps = NRecons / THeart
            gifWindows = ((0, 400), (0, 1000), (0, 4000))

            for gif_WC, gif_WW in gifWindows:
                winTag = f"WC{gif_WC:.0f}_WW{gif_WW:.0f}"
                stem = (f"HR{HeartRateBPM:.0f}_TRot{TRot:.2f}s_NSub{NSubsets}"
                        f"_NScans{NScansPerRecon}_{winTag}")
                for Data, name, label in (
                        (ReconHU_Linear, "LinearReconHU", "Linear Recon"),
                        (ReconHU_LinearMinusGT, "LinearReconHU_DiffToGT", "Linear Diff"),
                        (ReconHU_Arbitrary, "ArbitraryReconHU", "Arbitrary Recon"),
                        (ReconHU_ArbitraryMinusGT, "ArbitraryReconHU_DiffToGT",
                         "Arbitrary Diff"),
                        (ReconHU_GT, "GroundTruthHU", "Ground Truth")):
                    IOHelperFunctions.SaveReconAsGif(
                        Data, OutputDir / f"{name}_{stem}.gif", Nx, Ny, NRecons,
                        label, gif_WC, gif_WW, gif_fps)

        if SaveRawRecon:
            stem = (f"HR{HeartRateBPM:.0f}_TRot{TRot:.2f}s_NSub{NSubsets}"
                    f"_NScans{NScansPerRecon}")
            for Data, name in ((ReconHU_Linear, "LinearReconHU"),
                               (ReconHU_LinearMinusGT, "LinearReconHU_DiffToGT"),
                               (ReconHU_Arbitrary, "ArbitraryReconHU"),
                               (ReconHU_ArbitraryMinusGT, "ArbitraryReconHU_DiffToGT"),
                               (ReconHU_GT, "GroundTruthHU")):
                IOHelperFunctions.SaveDoubleToFile(
                    OutputDir / f"{name}_{stem}",
                    RawDataHelperFunctions.SimpleLinearizeArray(Data))

        IOHelperFunctions.SaveStructToMat(
            OutputDir / "additionalInfo.mat",
            {"TRecon": TRecon, "ReconType": ReconType, "NSubsets": NSubsets,
             "NRecons": NRecons, "NScansPerRecon": NScansPerRecon,
             "ThetaArrayArbitrary": ThetaArrayArbitrary,
             "HeartRateBPM": HeartRateBPM, "MotionEnabled": MotionEnabled,
             "MotionDirectionRad": MotionDirectionRad, "MotionType": MotionType,
             "TRest": TRest, "IodinePeakFactor": IodinePeakFactor,
             "IodineSigmaMin": IodineSigmaMin, "IodineSigmaMax": IodineSigmaMax})

        if isGIF:
            StaticCT_createSubsetAnimationGif(
                ThetaArrayArbitrary, ReconType, NSubsets, OutputDir,
                MaxAngles=100, MaxFrames=100, DelayTime=0.25, Visible=False)

    print("Temporal iodine multi-phase simulation completed.")

    return {
        "OutputDir": str(OutputDir),
        "TRecon": TRecon, "TLeft": TLeft, "TRight": TRight,
        "ThetaArrayArbitrary": ThetaArrayArbitrary,
        "ThetaArrayLinear": ThetaArrayLinear,
        "ProjectionsLinear": ProjectionsLinear,
        "ProjectionsArbitrary": ProjectionsArbitrary,
        "Kernel": Kernel,
        "ReconLinear": ReconLinear, "ReconArbitrary": ReconArbitrary,
        "ReconGT": ReconGT,
        "ReconHU_Linear": ReconHU_Linear, "ReconHU_Arbitrary": ReconHU_Arbitrary,
        "ReconHU_GT": ReconHU_GT,
        "ReconHU_LinearMinusGT": ReconHU_LinearMinusGT,
        "ReconHU_ArbitraryMinusGT": ReconHU_ArbitraryMinusGT,
        "iodineTemporalSignal": iodineTemporalSignal,
        "iodineTemporalDiscrete": iodineTemporalDiscrete,
        "scanMidSignal": scanMidSignal,
        "figSignal": figSignal,
        "Geometry": {"RM": RM, "M": M, "N": N, "N2pi": N2pi, "Npi": Npi,
                     "thetaN": thetaN, "theta0": theta0, "dtheta": dtheta, "dT": dT,
                     "xi0": xi0, "dxi": dxi, "x0": x0, "dx": dx, "Nx": Nx,
                     "y0": y0, "dy": dy, "Ny": Ny, "THeart": THeart,
                     "TReconInterval": TReconInterval, "NRecons": NRecons,
                     "NScansPerRecon": NScansPerRecon, "NSubsets": NSubsets},
    }


def _GroundTruthPhase(Nx, dx, x0, Ny, dy, y0, TRecon_k, dT, TemporalPhantomName,
                      HeartRateBPM, MotionEnabled, MotionDirectionRad, MotionType,
                      TRest, IodinePeakFactor, IodineSigmaMin, IodineSigmaMax):
    Phantom, _, _, _, _ = CreatePhantom.CreateVoxelPhantom2DWithMotionUnified(
        Nx, dx, x0, Ny, dy, y0, TRecon_k, dT, 0, TemporalPhantomName,
        Heartrate=HeartRateBPM, Motion=MotionEnabled, Direction=MotionDirectionRad,
        MotionType=MotionType, TRest=TRest, IodinePeakFactor=IodinePeakFactor,
        IodineSigmaMin=IodineSigmaMin, IodineSigmaMax=IodineSigmaMax)
    return np.asarray(Phantom).ravel()


def _TemporalSignal(t, Tcycle, TMotion, MotionType):
    """Normalized iodine signal, identical in shape to the phantom motion profile."""
    tLocal = np.mod(np.asarray(t, dtype=float), Tcycle)
    Signal = np.zeros_like(tLocal)
    activeMask = tLocal < TMotion

    if MotionType == "full":
        Signal[activeMask] = np.sin(2 * math.pi * tLocal[activeMask] / TMotion)
    elif MotionType == "half":
        Signal[activeMask] = np.sin(math.pi * tLocal[activeMask] / TMotion)
    else:
        raise ValueError(f'Unknown MotionType "{MotionType}". Use "full" or "half".')

    return Signal


def _ShowFinalRecons(ReconHU_GT, ReconHU_Linear, ReconHU_Arbitrary, NRecons, Nx, Ny):
    import matplotlib.pyplot as plt

    for k in range(1, NRecons + 1):
        fig, axes = plt.subplots(1, 3, num=f"Temporal Iodine Phase {k}",
                                 facecolor="w")
        for ax, Data, title in ((axes[0], ReconHU_GT, "GT"),
                                (axes[1], ReconHU_Linear, "Linear"),
                                (axes[2], ReconHU_Arbitrary, "Arbitrary")):
            image = ax.imshow(Data[k - 1, :].reshape(Ny, Nx), cmap="gray",
                              vmin=-200, vmax=600)
            ax.axis("off")
            ax.set_title(f"{title} Phase {k}")
            fig.colorbar(image, ax=ax)

    plt.show()


def _PlotTemporalSignal(t, iodineTemporalSignal, TRecon, TLeft, TRight,
                        iodineTemporalDiscrete, scanMidSignal, NScansPerRecon,
                        ReconType, TRot, THeart, Tcycle, TMotion, MotionType,
                        IodinePeakFactor, show):
    import matplotlib.pyplot as plt

    SegmentDuration = TRot / 2 if ReconType == "180" else TRot

    fig, axes = plt.subplots(1, 2, num="Temporal iodine signal and reconstruction timing",
                             facecolor="w")

    ax = axes[0]
    ax.plot(t, iodineTemporalSignal, "-", color="blue", label="temporal iodine signal")
    ax.plot(TRecon, iodineTemporalDiscrete, ".", color="red", markersize=10,
            label="reconstructed point")
    for i in range(TRecon.size):
        ax.plot([TLeft[i], TRight[i]],
                [iodineTemporalDiscrete[i], iodineTemporalDiscrete[i]], "-",
                linewidth=1.5, color="g", marker="|", markersize=12,
                label="reconstruction interval" if i == 0 else None)
        for j in range(NScansPerRecon):
            ax.plot([TLeft[i] + j * SegmentDuration,
                     TLeft[i] + (j + 1) * SegmentDuration],
                    [scanMidSignal[i, j], scanMidSignal[i, j]], "-", linewidth=1,
                    color=(0.85, 0.33, 0.10), marker="|", markersize=8,
                    label="single scan segment" if i == 0 and j == 0 else None)

    ax.legend()
    ax.set_xlabel("time in s")
    ax.set_ylabel("relative iodine signal")
    ax.set_title("Temporal iodine signal with reconstructed points "
                 "and reconstruction windows")
    ax.axis([-0.2, max(TRecon.max(), THeart) + 0.2,
             -1.2 * IodinePeakFactor, 1.2 * IodinePeakFactor])
    ax.grid(True)

    ax = axes[1]
    phaseToZoom = 0
    zoomMargin = 0.15
    zoomLeft = max(0, TLeft[phaseToZoom] - zoomMargin)
    zoomRight = min(max(TRecon.max(), THeart), TRight[phaseToZoom] + zoomMargin)
    tZoom = np.linspace(zoomLeft, zoomRight, 300)
    iodineTemporalZoom = IodinePeakFactor * _TemporalSignal(tZoom, Tcycle, TMotion,
                                                            MotionType)

    ax.plot(tZoom, iodineTemporalZoom, "-", color="blue",
            label="temporal iodine signal")
    ax.plot(TRecon[phaseToZoom], iodineTemporalDiscrete[phaseToZoom], ".", color="red",
            markersize=10, label="reconstructed point")
    ax.plot([TLeft[phaseToZoom], TRight[phaseToZoom]],
            [iodineTemporalDiscrete[phaseToZoom], iodineTemporalDiscrete[phaseToZoom]],
            "-", linewidth=1.5, color="g", marker="|", markersize=12,
            label="reconstruction interval")
    for j in range(NScansPerRecon):
        ax.plot([TLeft[phaseToZoom] + j * SegmentDuration,
                 TLeft[phaseToZoom] + (j + 1) * SegmentDuration],
                [scanMidSignal[phaseToZoom, j], scanMidSignal[phaseToZoom, j]], "-",
                linewidth=1, color=(0.85, 0.33, 0.10), marker="|", markersize=8,
                label="single scan segment" if j == 0 else None)

    ax.legend()
    ax.set_xlabel("time in s")
    ax.set_ylabel("relative iodine signal")
    ax.set_title("Zoom of the first reconstruction window")
    ax.axis([zoomLeft, zoomRight, -1.2 * IodinePeakFactor, 1.2 * IodinePeakFactor])
    ax.grid(True)

    if show:
        plt.show()
    else:
        plt.close(fig)

    return fig
