"""Dual source cardiac 4D CT experiment.

Port of Projects/Static CT/Projects AH/DualSource/DualSource_CardiacMotion4DCT_Main.m.
MATLAB is a function ``Results = DualSource_CardiacMotion4DCT_Main(Heartrate, TRot,
options)``; the name-value options become keyword-only arguments with the same names and
defaults, and ``Results`` is returned as nested dicts with the MATLAB field names.
"""
import math
import time
from pathlib import Path

import numpy as np

from CTBasics.Memory.Allocation import Allocation
from CTBasics.MultiCore.Parallelization import Parallelization
from CTBasics.ReadAndWrite.IOHelperFunctions import IOHelperFunctions
from CTBasics.Reconstruction.Backprojection import Backprojection
from CTBasics.Reconstruction.Constants import Constants
from CTBasics.Reconstruction.Convolution import Convolution
from CTBasics.Reconstruction.ForwardProjection import ForwardProjection
from CTBasics.Reconstruction.Kernels import Kernels
from CTBasics.Reconstruction.MoireFilter import MoireFilter
from CTBasics.Simulation.Noise import Noise, rng
from CTBasics.Projects.StaticCT.StaticCTHelperFunctions import StaticCTIOHelpers, StaticCTVisualizationAndEvaluationHelpers
from CTBasics.Projects.StaticCT.StaticCTHelperFunctions.StaticCTVisualizationAndEvaluationHelpers import _SSIMMap


def corr2(A, B):
    """MATLAB corr2: Pearson correlation of two equally sized arrays."""
    A = np.asarray(A, dtype=float).ravel()
    B = np.asarray(B, dtype=float).ravel()
    A = A - A.mean()
    B = B - B.mean()
    denominator = math.sqrt(float(np.sum(A * A)) * float(np.sum(B * B)))
    if denominator == 0.0:
        return float("nan")
    return float(np.sum(A * B)) / denominator


def rmse(A, B):
    """MATLAB rmse: root mean squared error of two equally sized arrays."""
    A = np.asarray(A, dtype=float).ravel()
    B = np.asarray(B, dtype=float).ravel()
    return float(np.sqrt(np.mean((A - B) ** 2)))


def ssim(A, Ref, DynamicRange):
    """MATLAB [ssimval, ssimmap] = ssim(A, Ref, 'DynamicRange', DynamicRange)."""
    ssimmap = _SSIMMap(A, Ref, DynamicRange)
    return float(np.mean(ssimmap)), ssimmap


def main(Heartrate=60, TRot=0.5, *, Nsources=2, Npar=512, DetectorElements=500,
         ImageSize=512, PhotonCount=2.25e5, RandomSeed=1, SelectedPhantom="Cardiac",
         ApplyMoireFilter=False, SimulateNoise=True, ShowPictures=True, ShowTimingPlot=True,
         CreateGroundTruthGif=False, WriteData=True, IncludeImageData=True,
         ShowFigures=False):
    """Simulates one normal/dual-source scan and returns the compact ``Results`` dict.

    ShowFigures is Python only: ShowPictures selects the MATLAB branch (which also writes
    the raw images), ShowFigures additionally opens the matplotlib windows.
    """
    if Heartrate <= 0 or TRot <= 0 or PhotonCount <= 0:
        raise ValueError("Heartrate, TRot and PhotonCount must be positive.")
    for name, value in (("Nsources", Nsources), ("Npar", Npar),
                        ("DetectorElements", DetectorElements), ("ImageSize", ImageSize)):
        if value != int(value) or value <= 0:
            raise ValueError(f"{name} must be a positive integer.")
    if RandomSeed != int(RandomSeed) or RandomSeed < 0:
        raise ValueError("RandomSeed must be a nonnegative integer.")
    Nsources, Npar = int(Nsources), int(Npar)

    tStart = time.perf_counter()

    # Same seed for each sweep run, so noise differences are not mistaken for an effect.
    rng(int(RandomSeed), "twister")

    aIOHelper = IOHelperFunctions()
    Outputdirectory = Path(aIOHelper.GlobalOutputFolderPath) / "cardiacPhantom" / "data"

    Reconkernel = Kernels.RamLak
    SimNoise = SimulateNoise

    showPics = ShowPictures
    VisualiseTime = ShowTimingPlot

    N_I0 = PhotonCount

    RM = 60
    M = int(DetectorElements)
    thetaN = 2 * math.pi

    N2pi = 2 * Npar
    N = int(N2pi * thetaN / (2 * math.pi))

    dxi = 2 * RM / (M - 1)
    xi0 = -RM - dxi
    theta0 = 0
    dtheta = thetaN / N

    Nx = int(ImageSize)
    Ny = int(ImageSize)

    dx = 2 * RM / Nx
    dy = 2 * RM / Ny

    xc = 0
    yc = 0

    x0 = -RM - dx / 2.0 + xc
    y0 = -RM - dy / 2.0 + yc

    dT = TRot / N2pi

    Parallelization.Init()

    PhantomOld = False
    SelectedPhantom = str(SelectedPhantom)

    CM = False
    Stent = False
    Plaque = 1

    Shift_Single = TRot / 4
    Shift_Multi = TRot / (4 * Nsources)

    Direction = 0
    Rotation = 0

    MotionType = "full"
    TRest = 0

    THeartCycle = 60 / Heartrate
    Shift_GT_Rest = THeartCycle - TRest / 2  # stationary reference at rest

    Shift_GT = 0

    naming = StaticCTIOHelpers.DualSource_buildOutputNaming(SelectedPhantom, Plaque, CM, Stent)

    # MATLAB formats with %d; integral values print without decimals.
    Outputdirectory = Outputdirectory / naming["RootFolderLabel"] / (
        f"aRot_{_d(Rotation)}"
        f"_HR_{_d(Heartrate)}"
        f"_TRot_{_MatlabRound(TRot * 1000)}ms"
        f"_N_{Npar}"
        f"_PS_{_MatlabRound(Shift_GT * 1000)}"
        f"_NSources_{Nsources}"
        f"_Ph_{naming['PhantomCode']}"
    )

    if WriteData:
        Outputdirectory.mkdir(parents=True, exist_ok=True)

        StaticCTIOHelpers.DualSource_writeSimulationParameters(
            Outputdirectory, Nsources, TRot, Heartrate, dT,
            RM, M, Npar, N2pi, dtheta, Nx, Ny, dx, dy,
            Shift_GT, Shift_GT_Rest, Rotation, Direction, MotionType, TRest,
            ApplyMoireFilter, SimNoise, N_I0, naming)

    # MATLAB drops the duplicated 2*pi entry of linspace, so dtheta is recomputed here.
    ThetaArray = np.linspace(0, 2 * math.pi, N2pi + 1)[:-1]
    dtheta = ThetaArray[1] - ThetaArray[0]

    tProjection = time.perf_counter()

    Npi = N2pi // 2
    ThetaArrayMulti = np.linspace(0, math.pi, Npi + 1)[:-1]

    NprojPerSourceBase = Npi // Nsources
    NprojRemainder = Npi % Nsources
    NprojPerSource = np.full(Nsources, NprojPerSourceBase, dtype=np.int64)
    NprojPerSource[:NprojRemainder] += 1
    if int(NprojPerSource.sum()) != Npi:
        raise ValueError("Projection distribution over sources does not sum to Npi.")

    PhantomOptions = dict(SelectedPhantom=SelectedPhantom, Heartrate=Heartrate, CM=CM,
                          Plaque=Plaque, Stent=Stent, PhantomOld=PhantomOld,
                          Direction=Direction, Rotation=Rotation,
                          MotionType=MotionType, TRest=TRest)

    Projections_multi_sources = []
    points_multi_sources = []
    sourceStartIdx = 0
    for sourceIdx in range(Nsources):
        NprojSource = int(NprojPerSource[sourceIdx])
        ThetaArraySource = ThetaArrayMulti[sourceStartIdx:sourceStartIdx + NprojSource]

        ProjSource, _, pointsSource, _, _ = \
            ForwardProjection.RayDrivenForwProjParCartesianMultipleWithMotionDual(
                x0, dx, Nx, y0, dy, Ny,
                NprojSource, xi0, dxi, M, N2pi, TRot, ThetaArraySource,
                Shift_GT=Shift_GT - Shift_Multi, **PhantomOptions)

        Projections_multi_sources.append(np.asarray(ProjSource).ravel())
        points_multi_sources.append(pointsSource)
        sourceStartIdx += NprojSource

    Projections_multi = np.concatenate(Projections_multi_sources)

    ThetaArraysingle = np.linspace(0, math.pi, N2pi // 2 + 1)[:-1]
    Projections, _, _, _, _ = \
        ForwardProjection.RayDrivenForwProjParCartesianMultipleWithMotionDual(
            x0, dx, Nx, y0, dy, Ny,
            N // 2, xi0, dxi, M, N2pi, TRot, ThetaArraysingle,
            Shift_GT=Shift_GT - Shift_Single, **PhantomOptions)
    Projections = np.asarray(Projections).ravel()

    # TRot = 1e-4 freezes the phantom, so these two runs are the static references.
    Projections_groundtruth, _, points_groundtruth, _, _ = \
        ForwardProjection.RayDrivenForwProjParCartesianMultipleWithMotionDual(
            x0, dx, Nx, y0, dy, Ny,
            N, xi0, dxi, M, N2pi, 0.0001, ThetaArray,
            Shift_GT=Shift_GT, **PhantomOptions)
    Projections_groundtruth = np.asarray(Projections_groundtruth).ravel()

    Projections_groundtruth_rest, _, _, _, _ = \
        ForwardProjection.RayDrivenForwProjParCartesianMultipleWithMotionDual(
            x0, dx, Nx, y0, dy, Ny,
            N, xi0, dxi, M, N2pi, 0.0001, ThetaArray,
            Motion=False, Shift_GT=Shift_GT_Rest, **PhantomOptions)
    Projections_groundtruth_rest = np.asarray(Projections_groundtruth_rest).ravel()

    _toc(tProjection)

    if CreateGroundTruthGif and WriteData:
        StaticCTIOHelpers.DualSource_createGroundTruthGif(
            Outputdirectory, Nx, Ny, dx, dy, x0, y0,
            Heartrate, CM, Stent, Plaque, Rotation, Direction, MotionType, TRest,
            Shift_GT, TRot, Nsources, Npar, SelectedPhantom)

    if VisualiseTime:
        timingFigure = StaticCTVisualizationAndEvaluationHelpers.DualSource_plotScanTiming(
            Heartrate, MotionType, TRest, Shift_GT, TRot, Nsources, THeartCycle, Shift_Single)
        _showOrClose(timingFigure, ShowFigures)

    NprojSingle = Projections.size // M
    NprojMulti = Projections_multi.size // M
    NprojGround = Projections_groundtruth.size // M
    NprojGroundRest = Projections_groundtruth_rest.size // M

    for name, Data in (("Projections", Projections),
                       ("Projections_multi", Projections_multi),
                       ("Projections_groundtruth", Projections_groundtruth),
                       ("Projections_groundtruth_rest", Projections_groundtruth_rest)):
        if Data.size % M != 0:
            raise ValueError(f"{name} length must be divisible by M.")

    if ApplyMoireFilter:
        Projections = MoireFilter.ApplyMoireFilterOnProjections(
            Projections, NprojSingle, M, dxi, 0.0 * (1 / dxi), 0.5 * (1 / dxi))
        Projections_multi = MoireFilter.ApplyMoireFilterOnProjections(
            Projections_multi, NprojMulti, M, dxi, 0.0 * (1 / dxi), 0.5 * (1 / dxi))
        Projections_groundtruth = MoireFilter.ApplyMoireFilterOnProjections(
            Projections_groundtruth, NprojGround, M, dxi, 0.0 * (1 / dxi), 0.5 * (1 / dxi))
        Projections_groundtruth_rest = MoireFilter.ApplyMoireFilterOnProjections(
            Projections_groundtruth_rest, NprojGroundRest, M, dxi,
            0.0 * (1 / dxi), 0.5 * (1 / dxi))

    tNoise = time.perf_counter()
    if SimNoise:
        Projections = Noise.ApplyPoissonNoise(Projections, M * NprojSingle, N_I0)

        # Total photon budget as in single source: each source gets 1/Nsources of it.
        N_I0_multi_per_source = N_I0 / Nsources
        for sourceIdx in range(Nsources):
            Projections_multi_sources[sourceIdx] = Noise.ApplyPoissonNoise(
                Projections_multi_sources[sourceIdx],
                Projections_multi_sources[sourceIdx].size,
                N_I0_multi_per_source)
        Projections_multi = np.concatenate(Projections_multi_sources)

        if showPics and ShowFigures:
            import matplotlib.pyplot as plt

            for Data, Nproj in ((Projections, NprojSingle), (Projections_multi, NprojMulti)):
                plt.figure()
                plt.imshow(Data.reshape(Nproj, M), cmap="gray")
                plt.show()
    _toc(tNoise)

    Kernel = None
    if Reconkernel == Kernels.SheppLogan:
        Kernel = Kernels.SheppLoganKernelSD(M, dxi)
    elif Reconkernel == Kernels.RamLak:
        Kernel = Kernels.RamLakKernelSD(M, dxi)

    Projections = Convolution.ConvolveSinogram(Projections, Kernel, M, NprojSingle, dxi)
    Projections_multi = Convolution.ConvolveSinogram(
        Projections_multi, Kernel, M, NprojMulti, dxi)
    Projections_ground = Convolution.ConvolveSinogram(
        Projections_groundtruth, Kernel, M, N, dxi)
    Projections_ground_rest = Convolution.ConvolveSinogram(
        Projections_groundtruth_rest, Kernel, M, N, dxi)

    Nphases = 1
    MultiphaseRecon = Allocation.zeros(Nphases, Nx * Ny)
    MultiphaseRecon_multi = Allocation.zeros(Nphases, Nx * Ny)
    MultiphaseRecon_groundtruth = Allocation.zeros(Nphases, Nx * Ny)
    MultiphaseRecon_groundtruth_rest = Allocation.zeros(Nphases, Nx * Ny)

    params = {"Heartrate": Heartrate, "TRot": TRot, "Npar": Npar,
              "Shift_GT": Shift_GT, "PhantomTag": naming["PhantomTag"]}

    for i in range(1, Nphases + 1):
        idx = slice((i - 1) * Npi * M, i * Npi * M)

        # ROI center from the exact ground-truth phase (not the middle of the scan window).
        cx_phys = points_groundtruth[0, 0]
        cy_phys = points_groundtruth[1, 0]

        cx = _MatlabRound((cx_phys - x0) / dx)
        cy = _MatlabRound((cy_phys - y0) / dy)

        cx = max(1, min(Nx, cx))
        cy = max(1, min(Ny, cy))

        roiMask, _, _ = StaticCTVisualizationAndEvaluationHelpers.DualSource_buildPhantomRoiMask(
            SelectedPhantom, [cx_phys, cy_phys], Nx, Ny, x0, y0, dx, dy, [cx, cy])

        if showPics:
            for name, Data in (
                    ("Proj-single", Projections[idx]),
                    ("Proj-multi", Projections_multi[idx]),
                    ("Proj-ground", Projections_ground[idx]),
                    ("Proj-ground-rest", Projections_ground_rest[idx]),
                    ("Diff-single-multi", Projections[idx] - Projections_multi[idx]),
                    ("Diff-ground-multi", Projections_ground[idx] - Projections_multi[idx]),
                    ("Diff-ground-single", Projections_ground[idx] - Projections[idx]),
                    ("Diff-ground-rest-single",
                     Projections_ground_rest[idx] - Projections[idx]),
                    ("Diff-ground-rest-multi",
                     Projections_ground_rest[idx] - Projections_multi[idx])):
                StaticCTIOHelpers.DualSource_showAndSaveImage(
                    Data, (M, Npi), Outputdirectory, name, WriteData, params,
                    ShowFigures=ShowFigures)

        theta0_i = theta0 + (i - 1) * Npi * dtheta

        ReconstructedImage = Backprojection.PixDrivenBackProjParCartesian(
            theta0_i, dtheta, Npi, N2pi, xi0, dxi, M,
            Projections[idx], x0, dx, Nx, y0, dy, Ny)
        ReconstructedImage_multi = Backprojection.PixDrivenBackProjParCartesian(
            theta0_i, dtheta, Npi, N2pi, xi0, dxi, M,
            Projections_multi[idx], x0, dx, Nx, y0, dy, Ny)
        ReconstructedImage_groundtruth = Backprojection.PixDrivenBackProjParCartesian(
            theta0_i, dtheta, Npi, N2pi, xi0, dxi, M,
            Projections_ground[idx], x0, dx, Nx, y0, dy, Ny)
        ReconstructedImage_groundtruth_rest = Backprojection.PixDrivenBackProjParCartesian(
            theta0_i, dtheta, Npi, N2pi, xi0, dxi, M,
            Projections_ground_rest[idx], x0, dx, Nx, y0, dy, Ny)

        if showPics:
            for name, Data in (
                    ("Recon-single", ReconstructedImage),
                    ("Recon-multi", ReconstructedImage_multi),
                    ("Recon-ground", ReconstructedImage_groundtruth),
                    ("Recon-ground-rest", ReconstructedImage_groundtruth_rest),
                    ("Diff-recon", ReconstructedImage_multi - ReconstructedImage),
                    ("Diff-gt-single", ReconstructedImage_groundtruth - ReconstructedImage),
                    ("Diff-gt-multi",
                     ReconstructedImage_groundtruth - ReconstructedImage_multi),
                    ("Diff-gt-rest-single",
                     ReconstructedImage_groundtruth_rest - ReconstructedImage),
                    ("Diff-gt-rest-multi",
                     ReconstructedImage_groundtruth_rest - ReconstructedImage_multi)):
                StaticCTIOHelpers.DualSource_showAndSaveImage(
                    Data, (Nx, Ny), Outputdirectory, name, WriteData, params, roiMask,
                    ShowFigures=ShowFigures)

        # Errors in HU so RMSE is interpretable; NCC and SSIM are unaffected by the
        # common affine conversion apart from numerical scaling.
        def toHU(image):
            return (image - Constants.MuW) / Constants.MuW * 1000

        GT = toHU(ReconstructedImage_groundtruth.reshape(Ny, Nx))
        GTRest = toHU(ReconstructedImage_groundtruth_rest.reshape(Ny, Nx))
        Single = toHU(ReconstructedImage.reshape(Ny, Nx))
        Multi = toHU(ReconstructedImage_multi.reshape(Ny, Nx))

        GT_roi = GT[roiMask]
        GTRest_roi = GTRest[roiMask]
        Single_roi = Single[roiMask]
        Multi_roi = Multi[roiMask]

        corr2_single = corr2(GT, Single)
        corr2_multi = corr2(GT, Multi)
        corr2_single_rest = corr2(GTRest, Single)
        corr2_multi_rest = corr2(GTRest, Multi)
        rmse_single = rmse(GT, Single)
        rmse_multi = rmse(GT, Multi)
        rmse_single_rest = rmse(GTRest, Single)
        rmse_multi_rest = rmse(GTRest, Multi)

        corr2_single_roi = corr2(GT_roi, Single_roi)
        corr2_multi_roi = corr2(GT_roi, Multi_roi)
        corr2_single_rest_roi = corr2(GTRest_roi, Single_roi)
        corr2_multi_rest_roi = corr2(GTRest_roi, Multi_roi)
        rmse_single_roi = rmse(GT_roi, Single_roi)
        rmse_multi_roi = rmse(GT_roi, Multi_roi)
        rmse_single_rest_roi = rmse(GTRest_roi, Single_roi)
        rmse_multi_rest_roi = rmse(GTRest_roi, Multi_roi)

        # Shared dynamic range keeps normal and dual-source SSIM comparable.
        dynamicRange = float(GT.max() - GT.min())
        if dynamicRange == 0:
            dynamicRange = 1
        ssim_single, ssimMapSingle = ssim(Single, GT, dynamicRange)
        ssim_multi, ssimMapMulti = ssim(Multi, GT, dynamicRange)

        # ROI SSIM averages the full-image map inside the ROI (no zero-padded crop).
        ssim_single_roi = float(np.nanmean(ssimMapSingle[roiMask]))
        ssim_multi_roi = float(np.nanmean(ssimMapMulti[roiMask]))

        print(f"Correlation single = {corr2_single:.4f}")
        print(f"Correlation multi   = {corr2_multi:.4f}")
        print(f"Correlation single vs rest GT = {corr2_single_rest:.4f}")
        print(f"Correlation multi   vs rest GT = {corr2_multi_rest:.4f}")
        print(f"RMSE single = {rmse_single:.4f}")
        print(f"RMSE multi   = {rmse_multi:.4f}")
        print(f"RMSE single vs rest GT = {rmse_single_rest:.4f}")
        print(f"RMSE multi   vs rest GT = {rmse_multi_rest:.4f}")

        print(f"ROI correlation single = {corr2_single_roi:.4f}")
        print(f"ROI correlation multi   = {corr2_multi_roi:.4f}")
        print(f"ROI correlation single vs rest GT = {corr2_single_rest_roi:.4f}")
        print(f"ROI correlation multi   vs rest GT = {corr2_multi_rest_roi:.4f}")
        print(f"ROI RMSE single = {rmse_single_roi:.4f}")
        print(f"ROI RMSE multi   = {rmse_multi_roi:.4f}")
        print(f"ROI RMSE single vs rest GT = {rmse_single_rest_roi:.4f}")
        print(f"ROI RMSE multi   vs rest GT = {rmse_multi_rest_roi:.4f}")
        print(f"SSIM single = {ssim_single:.4f}, multi = {ssim_multi:.4f}")
        print(f"ROI SSIM single = {ssim_single_roi:.4f}, multi = {ssim_multi_roi:.4f}")

        if WriteData:
            StaticCTIOHelpers.DualSource_appendReconstructionMetrics(
                Outputdirectory,
                corr2_single, corr2_multi, corr2_single_rest, corr2_multi_rest,
                rmse_single, rmse_multi, rmse_single_rest, rmse_multi_rest,
                corr2_single_roi, corr2_multi_roi, corr2_single_rest_roi, corr2_multi_rest_roi,
                rmse_single_roi, rmse_multi_roi, rmse_single_rest_roi, rmse_multi_rest_roi)

        MultiphaseRecon[i - 1, :] = ReconstructedImage
        MultiphaseRecon_multi[i - 1, :] = ReconstructedImage_multi
        MultiphaseRecon_groundtruth[i - 1, :] = ReconstructedImage_groundtruth
        MultiphaseRecon_groundtruth_rest[i - 1, :] = ReconstructedImage_groundtruth_rest

    # Compact scalar results for sweeps; images are optional because they are expensive.
    Results = {
        "Parameters": {
            "HeartRateBPM": Heartrate,
            "TRotSeconds": TRot,
            "Nsources": Nsources,
            "Npar": Npar,
            "DetectorElements": M,
            "ImageSize": Nx,
            "PhotonCount": N_I0,
            "RandomSeed": RandomSeed,
            "SelectedPhantom": SelectedPhantom,
        },
        "Metrics": {
            "NCC_Normal": corr2_single,
            "NCC_DualSource": corr2_multi,
            "RMSE_Normal": rmse_single,
            "RMSE_DualSource": rmse_multi,
            "SSIM_Normal": ssim_single,
            "SSIM_DualSource": ssim_multi,
            "NCC_Normal_ROI": corr2_single_roi,
            "NCC_DualSource_ROI": corr2_multi_roi,
            "RMSE_Normal_ROI": rmse_single_roi,
            "RMSE_DualSource_ROI": rmse_multi_roi,
            "SSIM_Normal_ROI": ssim_single_roi,
            "SSIM_DualSource_ROI": ssim_multi_roi,
        },
        "OutputDirectory": str(Outputdirectory),
    }

    if IncludeImageData:
        Results["Images"] = {
            "Normal": Single,
            "DualSource": Multi,
            "GroundTruth": GT,
            "GroundTruthRest": GTRest,
            "ROIMask": roiMask,
        }

    _toc(tNoise)
    return Results


def _MatlabRound(x):
    """MATLAB round: halves away from zero (Python's round is banker's rounding)."""
    return int(np.sign(x) * np.floor(np.abs(x) + 0.5))


def _d(value):
    """MATLAB %d: integral values print as integers, others fall back to %e."""
    return str(int(value)) if float(value).is_integer() else f"{value:e}"


def _toc(tStart):
    print(f"Elapsed time is {time.perf_counter() - tStart:.6f} seconds.")


def _showOrClose(figure, ShowFigures):
    import matplotlib.pyplot as plt

    if ShowFigures:
        plt.show()
    else:
        plt.close(figure)

if __name__ == "__main__":
    import matplotlib.pyplot as plt

    main(ShowFigures=True)
    plt.show()
