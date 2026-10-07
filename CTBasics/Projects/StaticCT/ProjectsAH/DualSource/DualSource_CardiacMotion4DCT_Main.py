"""Dual source cardiac 4D CT experiment.

Port of Projects/Static CT/Projects AH/DualSource/DualSource_CardiacMotion4DCT_Main.m.
MATLAB runs this as a script with hard coded settings; the port exposes the same values as
keyword defaults so the experiment can also be driven from tests.
"""
import math
from pathlib import Path

import numpy as np

from .....Memory.Allocation import Allocation
from .....MultiCore.Parallelization import Parallelization
from .....ReadAndWrite.IOHelperFunctions import IOHelperFunctions
from .....Reconstruction.Backprojection import Backprojection
from .....Reconstruction.Convolution import Convolution
from .....Reconstruction.ForwardProjection import ForwardProjection
from .....Reconstruction.Kernels import Kernels
from .....Reconstruction.MoireFilter import MoireFilter
from .....Simulation.Noise import Noise
from .Shared.DualSource_appendReconstructionMetrics import DualSource_appendReconstructionMetrics
from .Shared.DualSource_buildOutputNaming import DualSource_buildOutputNaming
from .Shared.DualSource_createGroundTruthGif import DualSource_createGroundTruthGif
from .Shared.DualSource_plotScanTiming import DualSource_plotScanTiming
from .Shared.DualSource_showAndSaveImage import DualSource_showAndSaveImage


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


def main(Outputdirectory=None, Reconkernel=Kernels.RamLak, ApplyMoireFilter=False,
         SimNoise=False, showPics=False, VisualiseTime=False, CreateGroundTruthGif=False,
         Nsources=2, N_I0=1e7, TRot=0.5,
         RM=60, M=500, Npar=512, Nx=512, Ny=512, xc=0, yc=0,
         SelectedPhantom="LinePattern", PhantomOld=False,
         Heartrate=60, CM=False, Stent=False, Plaque=1,
         Direction=0, Rotation=0, MotionType="full", TRest=0, Shift_GT=0):
    if Outputdirectory is None:
        aIOHelper = IOHelperFunctions()
        Outputdirectory = Path(aIOHelper.GlobalOutputFolderPath) / "cardiacPhantom" / "data"

    thetaN = 2 * math.pi
    N2pi = 2 * Npar
    N = int(N2pi * thetaN / (2 * math.pi))

    dxi = 2 * RM / (M - 1)
    xi0 = -RM - dxi
    theta0 = 0
    dtheta = thetaN / N

    dx = 2 * RM / Nx
    dy = 2 * RM / Ny
    x0 = -RM - dx / 2.0 + xc
    y0 = -RM - dy / 2.0 + yc

    dT = TRot / N2pi

    Parallelization.Init()

    Shift_Single = TRot / 4
    Shift_Multi = TRot / (4 * Nsources)

    THeartCycle = 60 / Heartrate
    Shift_GT_Rest = THeartCycle - TRest / 2  # stationary reference at rest

    naming = DualSource_buildOutputNaming(SelectedPhantom, Plaque, CM, Stent)

    Outputdirectory = Path(Outputdirectory) / naming["RootFolderLabel"] / (
        f"aRot_{int(Rotation)}"
        f"_HR_{int(Heartrate)}"
        f"_TRot_{round(TRot * 1000)}ms"
        f"_N_{int(Npar)}"
        f"_PS_{round(Shift_GT * 1000)}"
        f"_NSources_{int(Nsources)}"
        f"_Ph_{naming['PhantomCode']}"
    )
    Outputdirectory.mkdir(parents=True, exist_ok=True)

    from .Shared.DualSource_writeSimulationParameters import DualSource_writeSimulationParameters
    DualSource_writeSimulationParameters(
        Outputdirectory, Nsources, TRot, Heartrate, dT,
        RM, M, Npar, N2pi, dtheta, Nx, Ny, dx, dy,
        Shift_GT, Shift_GT_Rest, Rotation, Direction, MotionType, TRest,
        ApplyMoireFilter, SimNoise, N_I0, naming)

    # MATLAB drops the duplicated 2*pi entry of linspace, so dtheta is recomputed here.
    ThetaArray = np.linspace(0, 2 * math.pi, N2pi + 1)[:-1]
    dtheta = ThetaArray[1] - ThetaArray[0]

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
    Projections, _, points_single, _, _ = \
        ForwardProjection.RayDrivenForwProjParCartesianMultipleWithMotionDual(
            x0, dx, Nx, y0, dy, Ny,
            N // 2, xi0, dxi, M, N2pi, TRot, ThetaArraysingle,
            Shift_GT=Shift_GT - Shift_Single, **PhantomOptions)
    Projections = np.asarray(Projections).ravel()

    # TRot = 1e-4 freezes the phantom, so these two runs are the static references.
    Projections_groundtruth, _, _, _, _ = \
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

    if CreateGroundTruthGif:
        DualSource_createGroundTruthGif(
            Outputdirectory, Nx, Ny, dx, dy, x0, y0,
            Heartrate, CM, Stent, Plaque, Rotation, Direction, MotionType, TRest,
            Shift_GT, TRot, Nsources, Npar, SelectedPhantom)

    if VisualiseTime:
        DualSource_plotScanTiming(
            Heartrate, MotionType, TRest, Shift_GT, TRot, Nsources, THeartCycle, Shift_Single)

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

    if SimNoise:
        Projections = Noise.ApplyPoissonNoise(Projections, M * NprojSingle, N_I0)
        N_I0_multi_per_source = N_I0 / Nsources
        for sourceIdx in range(Nsources):
            Projections_multi_sources[sourceIdx] = Noise.ApplyPoissonNoise(
                Projections_multi_sources[sourceIdx],
                Projections_multi_sources[sourceIdx].size,
                N_I0_multi_per_source)
        Projections_multi = np.concatenate(Projections_multi_sources)

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

    Metrics = []

    for i in range(1, Nphases + 1):
        idx = slice((i - 1) * Npi * M, i * Npi * M)

        projStart = max(1, math.ceil(((i - 1) * Npi * M + 1) / M))
        projEnd = min(np.shape(points_single)[1], math.ceil((i * Npi * M) / M))
        midProj = round((projStart + projEnd) / 2)

        cx_phys = points_single[0, midProj - 1]
        cy_phys = points_single[1, midProj - 1]
        cx = max(1, min(Nx, round((cx_phys - x0) / dx)))
        cy = max(1, min(Ny, round((cy_phys - y0) / dy)))

        r = 80
        X, Y = np.meshgrid(np.arange(1, Nx + 1), np.arange(1, Ny + 1))
        roiMask = ((X - cx) ** 2 + (Y - cy) ** 2) <= r ** 2

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
                DualSource_showAndSaveImage(Data, (M, Npi), Outputdirectory, name,
                                            True, params)

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
                DualSource_showAndSaveImage(Data, (Nx, Ny), Outputdirectory, name,
                                            True, params, roiMask)

        GT = ReconstructedImage_groundtruth.reshape(Ny, Nx)
        GTRest = ReconstructedImage_groundtruth_rest.reshape(Ny, Nx)
        Single = ReconstructedImage.reshape(Ny, Nx)
        Multi = ReconstructedImage_multi.reshape(Ny, Nx)

        GT_roi = GT[roiMask]
        GTRest_roi = GTRest[roiMask]
        Single_roi = Single[roiMask]
        Multi_roi = Multi[roiMask]

        Phase = {
            "corr2_single": corr2(GT, Single),
            "corr2_multi": corr2(GT, Multi),
            "corr2_single_rest": corr2(GTRest, Single),
            "corr2_multi_rest": corr2(GTRest, Multi),
            "rmse_single": rmse(GT, Single),
            "rmse_multi": rmse(GT, Multi),
            "rmse_single_rest": rmse(GTRest, Single),
            "rmse_multi_rest": rmse(GTRest, Multi),
            "corr2_single_roi": corr2(GT_roi, Single_roi),
            "corr2_multi_roi": corr2(GT_roi, Multi_roi),
            "corr2_single_rest_roi": corr2(GTRest_roi, Single_roi),
            "corr2_multi_rest_roi": corr2(GTRest_roi, Multi_roi),
            "rmse_single_roi": rmse(GT_roi, Single_roi),
            "rmse_multi_roi": rmse(GT_roi, Multi_roi),
            "rmse_single_rest_roi": rmse(GTRest_roi, Single_roi),
            "rmse_multi_rest_roi": rmse(GTRest_roi, Multi_roi),
        }
        Metrics.append(Phase)

        DualSource_appendReconstructionMetrics(
            Outputdirectory,
            Phase["corr2_single"], Phase["corr2_multi"],
            Phase["corr2_single_rest"], Phase["corr2_multi_rest"],
            Phase["rmse_single"], Phase["rmse_multi"],
            Phase["rmse_single_rest"], Phase["rmse_multi_rest"],
            Phase["corr2_single_roi"], Phase["corr2_multi_roi"],
            Phase["corr2_single_rest_roi"], Phase["corr2_multi_rest_roi"],
            Phase["rmse_single_roi"], Phase["rmse_multi_roi"],
            Phase["rmse_single_rest_roi"], Phase["rmse_multi_rest_roi"])

        MultiphaseRecon[i - 1, :] = ReconstructedImage
        MultiphaseRecon_multi[i - 1, :] = ReconstructedImage_multi
        MultiphaseRecon_groundtruth[i - 1, :] = ReconstructedImage_groundtruth
        MultiphaseRecon_groundtruth_rest[i - 1, :] = ReconstructedImage_groundtruth_rest

    return {
        "Outputdirectory": str(Outputdirectory),
        "Naming": naming,
        "Projections": Projections,
        "Projections_multi": Projections_multi,
        "Projections_ground": Projections_ground,
        "Projections_ground_rest": Projections_ground_rest,
        "Kernel": Kernel,
        "MultiphaseRecon": MultiphaseRecon,
        "MultiphaseRecon_multi": MultiphaseRecon_multi,
        "MultiphaseRecon_groundtruth": MultiphaseRecon_groundtruth,
        "MultiphaseRecon_groundtruth_rest": MultiphaseRecon_groundtruth_rest,
        "Metrics": Metrics,
        "Geometry": {"RM": RM, "M": M, "N": N, "N2pi": N2pi, "Npi": Npi,
                     "theta0": theta0, "dtheta": dtheta, "xi0": xi0, "dxi": dxi,
                     "x0": x0, "dx": dx, "Nx": Nx, "y0": y0, "dy": dy, "Ny": Ny,
                     "dT": dT, "Shift_Single": Shift_Single, "Shift_Multi": Shift_Multi,
                     "Shift_GT_Rest": Shift_GT_Rest,
                     "NprojPerSource": NprojPerSource},
    }
