"""Prospectively gated 4D CT: linear versus subset interleaved scan order.

Port of Projects/Static CT/Projects JP/test_4DCT_prosp.m. The MATLAB script caches the
linear and ground truth reconstructions in a .mat file so that only the arbitrary order
scan has to be recomputed when sweeping NSubsets; the port keeps that cache (scipy .mat).

SimNoise/N_I0 are off by default so that the JP behaviour is bit identical to MATLAB.
They exist because Projects CH/StaticCT_4_RealTimeImaging_Script_CH.m is the same script
with a Poisson noise stage added, and the port shares one implementation instead of
duplicating 400 lines.
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
from .StaticCT_appendReconstructionMetrics import StaticCT_appendReconstructionMetrics
from .StaticCT_computeReconstructionMetrics import StaticCT_computeReconstructionMetrics
from .StaticCT_createSubsetAnimationGif import StaticCT_createSubsetAnimationGif
from .StaticCT_plotReconstructionResults import StaticCT_plotReconstructionResults
from ..StaticCTHelperFunctions.StaticCTSystemHelperFunctions import \
    StaticCTSystemHelperFunctions

# Array for the consecutive simulation of different subset counts (MATLAB keeps it as a
# convenience variable for manual sweeps).
NSubsets_array = [2, 4, 8, 16, 32, 64, 128]


def main(Outputdirectory_base=None, reconType="180",
         Reconkernel=Kernels.RamLak, ApplyMoireFilter=False,
         NSubsets=64, NRecons=8, SelectedPhantom="Respiratory",
         Heartrate=60, Motion=True, Direction=0, MotionType="full", TRest=0,
         CM=False, Stent=False, Plaque=1, Rotation=0,
         isplot=True, phasesToPlot=(1, 3), isGIF=False, writeData=True,
         NScansPerRecon=1, TRot=0.5, RM=None, M=500, N2pi=1024, Nx=512, Ny=512,
         UseLinearCache=True, SimNoise=False, N_I0=100000 / 5):
    if Outputdirectory_base is None:
        Outputdirectory_base = Path(IOHelperFunctions().GlobalOutputFolderPath) \
            / "cardiacPhantom" / "data"
    Outputdirectory_base = Path(Outputdirectory_base)

    Parallelization.Init()

    THeart = 60 / Heartrate  # duration of one heart cycle

    if RM is None:
        RM = 150 if SelectedPhantom == "Respiratory" else 60

    if reconType == "180":
        TReconInterval = NScansPerRecon * TRot / 2  # time interval of one reconstruction
        thetaN = math.pi
    elif reconType == "360":
        thetaN = 2 * math.pi
        TReconInterval = NScansPerRecon * TRot
    else:
        raise ValueError("reconType must be '180' or '360'.")

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

    ThetaArray_arbitrary = StaticCTSystemHelperFunctions.SubsetInterleaving(
        N2pi, NSubsets, reconType, 0)
    ThetaArray = np.linspace(0, 2 * math.pi - dtheta, N2pi)

    Outputdirectory_base.mkdir(parents=True, exist_ok=True)
    linearScan_fullfile = Outputdirectory_base / (
        f"linearRecon_HR_{Heartrate}_TRot_{round(TRot * 1000)}ms_NRecons{NRecons}"
        f"_NScans{NScansPerRecon}_{int(reconType)}Recon"
        f"_{round(math.degrees(Direction))}deg_{SelectedPhantom}.mat")

    cache = None
    if UseLinearCache and linearScan_fullfile.is_file():
        from scipy.io import loadmat
        cache = loadmat(str(linearScan_fullfile))
        linearRecon = False  # linear recon for these parameters already exists
    else:
        linearRecon = True

    Outputdirectory = Outputdirectory_base / (
        f"HR_{Heartrate}_TRot_{round(TRot * 1000)}ms_NSub_{NSubsets}"
        f"_NRecons{NRecons}_NScans{NScansPerRecon}_{int(reconType)}Recon"
        f"_{round(math.degrees(Direction))}deg_{SelectedPhantom}")
    Outputdirectory.mkdir(parents=True, exist_ok=True)

    # Equally spaced phases over one heart cycle; the trailing step is dropped so that a
    # GIF does not repeat the starting point (t = 0 equals t = THeart).
    TRecon = np.linspace(0, THeart, NRecons + 1)[:-1]

    if not linearRecon:
        TRecon_save = np.asarray(cache["TRecon_save"], dtype=float).ravel()
        if not np.array_equal(TRecon, TRecon_save):
            raise ValueError("selected and saved reconstrution times are different. "
                             "linear reconstruction can not be used!")

    TLeft = TRecon - TReconInterval / 2   # start time
    TRight = TRecon + TReconInterval / 2  # end time

    Projections_arbitrary = [Allocation.zeros(NScansPerRecon, N * M)
                             for _ in range(NRecons)]
    Projections_linear = [Allocation.zeros(NScansPerRecon, N * M)
                          for _ in range(NRecons)]
    Projections_GT = [Allocation.zeros(NScansPerRecon, N * M) for _ in range(NRecons)]

    center_points_1 = {}

    print("Forward Projection for specified recons running...")
    for k in range(1, NRecons + 1):
        T0 = TLeft[k - 1]  # start time of the current phase
        for j in range(1, NScansPerRecon + 1):
            if j % 2 == 0:
                ThetaIdx = slice(Npi, N2pi)  # 180-360 degrees
            else:
                ThetaIdx = slice(0, Npi)     # 0-180 degrees

            T0_rot = T0 + (j - 1) * (TRot / 2)  # start time of each half rotation

            Proj, Angles_start, points, _ = \
                ForwardProjection.\
                RayDrivenForwProjParCartesianMultipleWithMotionArbitraryOrderParallelized(
                    x0, dx, Nx, y0, dy, Ny, N, xi0, dxi, M, N2pi, TRot,
                    ThetaArray_arbitrary[ThetaIdx],
                    Heartrate=Heartrate, CM=CM, Plaque=Plaque, Stent=Stent,
                    Motion=Motion, T0=T0_rot, Direction=Direction,
                    SelectedPhantom=SelectedPhantom, Rotation=Rotation,
                    MotionType=MotionType, TRest=TRest)
            Projections_arbitrary[k - 1][j - 1, :] = np.asarray(Proj).ravel()
            center_points_1[(k - 1, j - 1)] = points

            if linearRecon:
                ProjLin, _, _, _ = \
                    ForwardProjection.\
                    RayDrivenForwProjParCartesianMultipleWithMotionArbitraryOrderParallelized(
                        x0, dx, Nx, y0, dy, Ny, N, xi0, dxi, M, N2pi, TRot,
                        ThetaArray[ThetaIdx],
                        Heartrate=Heartrate, CM=CM, Plaque=Plaque, Stent=Stent,
                        Motion=Motion, T0=T0_rot, Direction=Direction,
                        SelectedPhantom=SelectedPhantom, Rotation=Rotation,
                        MotionType=MotionType, TRest=TRest)
                Projections_linear[k - 1][j - 1, :] = np.asarray(ProjLin).ravel()

                # TRot = 0.0001 freezes the phantom, giving the motion free reference.
                ProjGT, _, _, _ = \
                    ForwardProjection.\
                    RayDrivenForwProjParCartesianMultipleWithMotionArbitraryOrderParallelized(
                        x0, dx, Nx, y0, dy, Ny, N, xi0, dxi, M, N2pi, 0.0001,
                        ThetaArray[ThetaIdx],
                        Heartrate=Heartrate, CM=CM, Plaque=Plaque, Stent=Stent,
                        Motion=Motion, T0=TRecon[k - 1], Direction=Direction,
                        SelectedPhantom=SelectedPhantom, Rotation=Rotation,
                        MotionType=MotionType, TRest=TRest)
                Projections_GT[k - 1][j - 1, :] = np.asarray(ProjGT).ravel()

    if not linearRecon:
        Projections_linear = [np.asarray(x, dtype=float)
                              for x in _UnpackCell(cache["Projections_linear_save"])]
        Projections_GT = [np.asarray(x, dtype=float)
                          for x in _UnpackCell(cache["Projections_GT_save"])]

    Projections_diff = [Projections_arbitrary[k] - Projections_linear[k]
                        for k in range(NRecons)]

    PhantomTruth = Allocation.zeros(NRecons, Nx * Ny)
    displacements = Allocation.zeros(2, NRecons)
    RoI_all = []  # RoI per phase

    for k in range(1, NRecons + 1):
        Phantom, _, xDisp, yDisp, RoI = \
            CreatePhantom.CreateVoxelPhantom2DWithMotionUnified(
                Nx, dx, x0, Ny, dy, y0, TRecon[k - 1], dT, 0, SelectedPhantom,
                CM=CM, Stent=Stent, Plaque=Plaque, Motion=Motion,
                Direction=Direction, Rotation=Rotation, MotionType=MotionType,
                TRest=TRest)
        PhantomTruth[k - 1, :] = np.asarray(Phantom).ravel()
        displacements[0, k - 1] = xDisp
        displacements[1, k - 1] = yDisp
        RoI_all.append(np.asarray(RoI, dtype=float))

    PhantomTruthHU = 1000 * (PhantomTruth - Constants.MuW) / Constants.MuW

    if SimNoise:
        print("Applying Poisson Noise to projections...")
        for k in range(NRecons):
            for j in range(NScansPerRecon):
                Projections_arbitrary[k][j, :] = Noise.ApplyPoissonNoise(
                    Projections_arbitrary[k][j, :], M * N, N_I0)
                Projections_linear[k][j, :] = Noise.ApplyPoissonNoise(
                    Projections_linear[k][j, :], M * N, N_I0)
                Projections_GT[k][j, :] = Noise.ApplyPoissonNoise(
                    Projections_GT[k][j, :], M * N, N_I0)

    if ApplyMoireFilter:
        for k in range(NRecons):
            for j in range(NScansPerRecon):
                Projections_linear[k][j, :] = MoireFilter.ApplyMoireFilterOnProjections(
                    Projections_linear[k][j, :], N, M, dxi, 0.0 * (1 / dxi),
                    0.5 * (1 / dxi))
                Projections_arbitrary[k][j, :] = \
                    MoireFilter.ApplyMoireFilterOnProjections(
                        Projections_arbitrary[k][j, :], N, M, dxi, 0.0 * (1 / dxi),
                        0.5 * (1 / dxi))

    Kernel = None
    if Reconkernel == Kernels.SheppLogan:
        Kernel = Kernels.SheppLoganKernelSD(M, dxi)
    elif Reconkernel == Kernels.RamLak:
        Kernel = Kernels.RamLakKernelSD(M, dxi)

    for k in range(NRecons):
        for j in range(NScansPerRecon):
            Projections_linear[k][j, :] = Convolution.ConvolveSinogram(
                Projections_linear[k][j, :], Kernel, M, N, dxi)
            Projections_arbitrary[k][j, :] = Convolution.ConvolveSinogram(
                Projections_arbitrary[k][j, :], Kernel, M, N, dxi)
            Projections_GT[k][j, :] = Convolution.ConvolveSinogram(
                Projections_GT[k][j, :], Kernel, M, N, dxi)

    Projections_linear_save = [x.copy() for x in Projections_linear]
    Projections_GT_save = [x.copy() for x in Projections_GT]

    MultiphaseRecon_linear = Allocation.zeros(NRecons, Nx * Ny)
    MultiphaseRecon_arbitrary = Allocation.zeros(NRecons, Nx * Ny)
    MultiphaseRecon_GT = Allocation.zeros(NRecons, Nx * Ny)

    print("Backprojection for each Time-point running...")
    FlatLinear = [RawDataHelperFunctions.SimpleLinearizeArray(x)
                  for x in Projections_linear]
    FlatGT = [RawDataHelperFunctions.SimpleLinearizeArray(x) for x in Projections_GT]
    FlatArbitrary = [RawDataHelperFunctions.SimpleLinearizeArray(x)
                     for x in Projections_arbitrary]

    for k in range(1, NRecons + 1):
        if linearRecon:
            MultiphaseRecon_linear[k - 1, :] = \
                Backprojection.PixDrivenBackProjParCartesian(
                    theta0, dtheta, NScansPerRecon * N, N2pi, xi0, dxi, M,
                    FlatLinear[k - 1], x0, dx, Nx, y0, dy, Ny)
            MultiphaseRecon_GT[k - 1, :] = \
                Backprojection.PixDrivenBackProjParCartesian(
                    theta0, dtheta, NScansPerRecon * N, N2pi, xi0, dxi, M,
                    FlatGT[k - 1], x0, dx, Nx, y0, dy, Ny)
        MultiphaseRecon_arbitrary[k - 1, :] = \
            Backprojection.PixDrivenBackProjParCartesian(
                theta0, dtheta, NScansPerRecon * N, N2pi, xi0, dxi, M,
                FlatArbitrary[k - 1], x0, dx, Nx, y0, dy, Ny)

    if not linearRecon:
        MultiphaseRecon_linear = np.asarray(cache["MultiphaseRecon_linear_save"],
                                            dtype=float)
        MultiphaseRecon_GT = np.asarray(cache["MultiphaseRecon_GT_save"], dtype=float)

    MultiphaseReconHU_linear = \
        1000 * (MultiphaseRecon_linear - Constants.MuW) / Constants.MuW
    MultiphaseReconHU_GT = 1000 * (MultiphaseRecon_GT - Constants.MuW) / Constants.MuW
    MultiphaseReconHU_arbitrary = \
        1000 * (MultiphaseRecon_arbitrary - Constants.MuW) / Constants.MuW

    MultiphaseRecon_arbitrary_GT = MultiphaseReconHU_arbitrary - MultiphaseReconHU_GT
    MultiphaseRecon_linear_GT = MultiphaseReconHU_linear - MultiphaseReconHU_GT
    MultiphaseRecon_diff = MultiphaseReconHU_arbitrary - MultiphaseReconHU_linear

    MultiphaseRecon_linear_save = MultiphaseRecon_linear.copy()
    MultiphaseRecon_GT_save = MultiphaseRecon_GT.copy()

    roi_radius = 60 if SelectedPhantom == "Respiratory" else 15

    (RMSE_linear, RMSE_arbitrary, NCC_linear, NCC_arbitrary,
     ssim_linear, ssim_arbitrary, nRoI, roi_mask, center_mask) = \
        StaticCT_computeReconstructionMetrics(
            MultiphaseReconHU_linear, MultiphaseReconHU_arbitrary,
            MultiphaseReconHU_GT, RoI_all, Nx, Ny, x0, y0, dx, dy, roi_radius)

    if SelectedPhantom == "Respiratory":
        WC, WW = -600, 1500
    else:
        WC, WW = 0, 500

    if isplot:
        StaticCT_plotReconstructionResults(
            MultiphaseReconHU_GT, MultiphaseReconHU_linear, MultiphaseReconHU_arbitrary,
            MultiphaseRecon_linear_GT, MultiphaseRecon_arbitrary_GT,
            RMSE_linear, RMSE_arbitrary, NCC_linear, NCC_arbitrary,
            ssim_linear, ssim_arbitrary, roi_mask, center_mask, Nx, Ny, nRoI,
            WC, WW, NSubsets, phasesToPlot)

    t = np.linspace(0, max(TRecon.max(), THeart), 1000)
    norm_displacement = _NormalizedMotion(t, Motion, Heartrate, TRest, MotionType)

    r = 5
    yDisplacement = 2 * r * norm_displacement * math.cos(Direction)
    xDisplacement = 2 * r * norm_displacement * math.sin(Direction)

    yDisplacement_discrete = displacements[1, :].copy()
    xDisplacement_discrete = displacements[0, :].copy()

    yDisplacement_rot = np.zeros((NRecons, NScansPerRecon))
    xDisplacement_rot = np.zeros((NRecons, NScansPerRecon))
    for k in range(NRecons):
        for j in range(NScansPerRecon):
            Points = np.asarray(center_points_1[(k, j)], dtype=float)
            yDisplacement_rot[k, j] = Points[1, N // 2 - 1]
            xDisplacement_rot[k, j] = Points[0, N // 2 - 1]

    figx = None
    if isplot or writeData:
        figx = _PlotSampling(t, xDisplacement, yDisplacement, TRecon, TLeft, TRight,
                             displacements, xDisplacement_discrete,
                             yDisplacement_discrete, xDisplacement_rot,
                             yDisplacement_rot, NScansPerRecon, TRot, show=isplot)

    summary = None
    if writeData:
        summary = StaticCT_appendReconstructionMetrics(
            Outputdirectory, SelectedPhantom, TRecon,
            RMSE_linear, RMSE_arbitrary, NCC_linear, NCC_arbitrary,
            ssim_linear, ssim_arbitrary,
            RoIAll=RoI_all, RoIRadiusMm=roi_radius,
            Tag=f"HR_{Heartrate}_TRot_{round(TRot * 1000)}ms_NSub_{NSubsets}"
                f"_NScans{NScansPerRecon}_{reconType}")

        print("Saving reconstructions as animated GIFs...")
        gif_WW_diff = max(np.percentile(MultiphaseRecon_linear_GT, 90),
                          np.percentile(MultiphaseRecon_arbitrary_GT, 90))
        gif_fps = NRecons / THeart  # desired GIF frame rate [frames/s]
        suffix = (f"HR{Heartrate:.0f}_TRot{TRot:.2f}s_{NSubsets:.0f}subsets"
                  f"_{NScansPerRecon:.0f}scans")

        IOHelperFunctions.SaveReconAsGif(
            MultiphaseReconHU_linear, Outputdirectory / f"LinearReconHU_{suffix}.gif",
            Nx, Ny, NRecons, "Linear Recon", WC, WW, gif_fps)
        IOHelperFunctions.SaveReconAsGif(
            MultiphaseRecon_linear_GT,
            Outputdirectory / f"LinearReconHU_difference_{suffix}.gif",
            Nx, Ny, NRecons, "Linear Diff", 0, gif_WW_diff, gif_fps)
        IOHelperFunctions.SaveReconAsGif(
            MultiphaseReconHU_arbitrary,
            Outputdirectory / f"ArbitraryReconHU_{suffix}.gif",
            Nx, Ny, NRecons, "Arbitrary Recon", WC, WW, gif_fps)
        IOHelperFunctions.SaveReconAsGif(
            MultiphaseRecon_arbitrary_GT,
            Outputdirectory / f"ArbitraryReconHU_difference_{suffix}.gif",
            Nx, Ny, NRecons, "Arbitrary Diff", 0, gif_WW_diff, gif_fps)
        IOHelperFunctions.SaveReconAsGif(
            MultiphaseReconHU_GT, Outputdirectory / f"GroundTruthHU_{suffix}.gif",
            Nx, Ny, NRecons, "Ground Truth", WC, WW, gif_fps)

        rawSuffix = (f"{Heartrate:.0f}_{TRot:.2f}s_{NSubsets:.0f}subsets"
                     f"_{NScansPerRecon:.0f}scans")
        IOHelperFunctions.SaveDoubleToFile(
            Outputdirectory / f"LinearReconHU{rawSuffix}",
            RawDataHelperFunctions.SimpleLinearizeArray(MultiphaseReconHU_linear))
        IOHelperFunctions.SaveDoubleToFile(
            Outputdirectory / f"LinearReconHU_difference{rawSuffix}",
            RawDataHelperFunctions.SimpleLinearizeArray(MultiphaseRecon_linear_GT))
        IOHelperFunctions.SaveDoubleToFile(
            Outputdirectory / f"ArbitraryReconHU{rawSuffix}",
            RawDataHelperFunctions.SimpleLinearizeArray(MultiphaseReconHU_arbitrary))
        IOHelperFunctions.SaveDoubleToFile(
            Outputdirectory / f"ArbitraryReconHU_difference{rawSuffix}",
            RawDataHelperFunctions.SimpleLinearizeArray(MultiphaseRecon_arbitrary_GT))
        IOHelperFunctions.SaveDoubleToFile(
            Outputdirectory / f"GroundTruthHU{rawSuffix}",
            RawDataHelperFunctions.SimpleLinearizeArray(MultiphaseReconHU_GT))

        if figx is not None:
            figx.savefig(Outputdirectory /
                         f"Sampling_arbitrary_{Heartrate:.0f}_{TRot:.2f}s"
                         f"_{NScansPerRecon:.0f}scans.jpg", dpi=300)

        IOHelperFunctions.SaveStructToMat(
            Outputdirectory / "additionalInfo.mat",
            {"TRecon": TRecon, "roi_radius": roi_radius,
             "NCC_linear": NCC_linear, "NCC_arbitrary": NCC_arbitrary,
             "RMSE_linear": RMSE_linear, "RMSE_arbitrary": RMSE_arbitrary,
             "reconType": reconType, "NSubsets": NSubsets,
             "ThetaArray_arbitrary": ThetaArray_arbitrary, "Heartrate": Heartrate,
             "ssim_arbitrary": ssim_arbitrary, "ssim_linear": ssim_linear})

    if UseLinearCache:
        IOHelperFunctions.SaveStructToMat(
            linearScan_fullfile,
            {"TRecon_save": TRecon,
             "Projections_linear_save": np.array(Projections_linear_save),
             "MultiphaseRecon_linear_save": MultiphaseRecon_linear_save,
             "Projections_GT_save": np.array(Projections_GT_save),
             "MultiphaseRecon_GT_save": MultiphaseRecon_GT_save})

    if isGIF:
        StaticCT_createSubsetAnimationGif(
            ThetaArray_arbitrary, reconType, NSubsets, Outputdirectory,
            MaxAngles=100, MaxFrames=100, DelayTime=0.25, Visible=False)

    if figx is not None and not isplot:
        import matplotlib.pyplot as plt
        plt.close(figx)

    return {
        "Outputdirectory": str(Outputdirectory),
        "linearScan_fullfile": str(linearScan_fullfile),
        "linearRecon": linearRecon,
        "TRecon": TRecon, "TLeft": TLeft, "TRight": TRight,
        "ThetaArray": ThetaArray, "ThetaArray_arbitrary": ThetaArray_arbitrary,
        "Projections_linear": Projections_linear,
        "Projections_arbitrary": Projections_arbitrary,
        "Projections_GT": Projections_GT,
        "Projections_diff": Projections_diff,
        "PhantomTruth": PhantomTruth, "PhantomTruthHU": PhantomTruthHU,
        "RoI_all": RoI_all, "displacements": displacements,
        "center_points_1": center_points_1,
        "Kernel": Kernel,
        "MultiphaseRecon_linear": MultiphaseRecon_linear,
        "MultiphaseRecon_arbitrary": MultiphaseRecon_arbitrary,
        "MultiphaseRecon_GT": MultiphaseRecon_GT,
        "MultiphaseReconHU_linear": MultiphaseReconHU_linear,
        "MultiphaseReconHU_arbitrary": MultiphaseReconHU_arbitrary,
        "MultiphaseReconHU_GT": MultiphaseReconHU_GT,
        "MultiphaseRecon_linear_GT": MultiphaseRecon_linear_GT,
        "MultiphaseRecon_arbitrary_GT": MultiphaseRecon_arbitrary_GT,
        "MultiphaseRecon_diff": MultiphaseRecon_diff,
        "RMSE_linear": RMSE_linear, "RMSE_arbitrary": RMSE_arbitrary,
        "NCC_linear": NCC_linear, "NCC_arbitrary": NCC_arbitrary,
        "ssim_linear": ssim_linear, "ssim_arbitrary": ssim_arbitrary,
        "nRoI": nRoI, "roi_mask": roi_mask, "center_mask": center_mask,
        "roi_radius": roi_radius, "WC": WC, "WW": WW,
        "Summary": summary,
        "Geometry": {"RM": RM, "M": M, "N": N, "N2pi": N2pi, "Npi": Npi,
                     "thetaN": thetaN, "theta0": theta0, "dtheta": dtheta, "dT": dT,
                     "xi0": xi0, "dxi": dxi, "x0": x0, "dx": dx, "Nx": Nx,
                     "y0": y0, "dy": dy, "Ny": Ny, "THeart": THeart,
                     "TReconInterval": TReconInterval, "NRecons": NRecons,
                     "NScansPerRecon": NScansPerRecon, "NSubsets": NSubsets},
    }


def _UnpackCell(value):
    """Restores a list of 2D arrays from a MATLAB cell array loaded by scipy."""
    Array = np.asarray(value)
    if Array.dtype == object:
        return [np.asarray(x, dtype=float) for x in Array.ravel()]
    return [Array[k] for k in range(Array.shape[0])]


def _NormalizedMotion(t, Motion, Heartrate, TRest, MotionType):
    """Continuous reference curve matching CreatePhantom.MotionProfile."""
    t = np.asarray(t, dtype=float)
    if not Motion:
        return np.zeros_like(t)

    Tcycle = 60 / Heartrate
    TMotion = Tcycle - TRest
    if TMotion <= 0:
        raise ValueError(f"TRest ({TRest:.3f} s) must be smaller than cycle duration "
                         f"({Tcycle:.3f} s).")

    tLocal = np.mod(t, Tcycle)
    norm_displacement = np.zeros_like(tLocal)
    activeMask = tLocal < TMotion

    if MotionType == "full":
        norm_displacement[activeMask] = np.sin(2 * math.pi * tLocal[activeMask] / TMotion)
    elif MotionType == "half":
        norm_displacement[activeMask] = np.sin(math.pi * tLocal[activeMask] / TMotion)
    else:
        raise ValueError(f'Unknown MotionType "{MotionType}". Use "full" or "half".')

    return norm_displacement


def _PlotSampling(t, xDisplacement, yDisplacement, TRecon, TLeft, TRight, displacements,
                  xDisplacement_discrete, yDisplacement_discrete, xDisplacement_rot,
                  yDisplacement_rot, NScansPerRecon, TRot, show):
    import matplotlib.pyplot as plt

    figx, axes = plt.subplots(1, 2)
    figx.suptitle("Continous motion (blue) with reconstructed points (red) "
                  "and corresponding interval")

    Panels = ((axes[0], yDisplacement, yDisplacement_discrete, yDisplacement_rot,
               displacements[1, :], "y-displacement in mm"),
              (axes[1], xDisplacement, xDisplacement_discrete, xDisplacement_rot,
               displacements[0, :], "x-displacement in mm"))

    for ax, Continuous, Discrete, Rotational, Interval, label in Panels:
        ax.plot(t, Continuous, "-", color="blue", label="theoretical motion")
        ax.plot(TRecon, Discrete, ".", color="red", markersize=10,
                label="reconstructed point")

        for i in range(TRecon.size):
            ax.plot([TLeft[i], TRight[i]], [Interval[i], Interval[i]], "-",
                    linewidth=1.5, color="g", marker="|", markersize=12,
                    label="reconstruction interval" if i == 0 else None)
            for j in range(NScansPerRecon):
                ax.plot([TLeft[i] + j * (TRot / 2), TLeft[i] + (j + 1) * (TRot / 2)],
                        [Rotational[i, j], Rotational[i, j]], "-", linewidth=1,
                        color=(0.85, 0.33, 0.10), marker="|", markersize=8,
                        label="single rotation" if i == 0 and j == 0 else None)

        ax.legend()
        ax.set_xlabel("time in s")
        ax.set_ylabel(label)
        ax.axis([-0.2, 1.2, -11, 11])
        ax.grid(True)

    if show:
        plt.show()

    return figx
