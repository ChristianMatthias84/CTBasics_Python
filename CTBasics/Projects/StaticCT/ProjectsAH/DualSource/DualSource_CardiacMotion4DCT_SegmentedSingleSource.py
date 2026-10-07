"""Segmented single source variant of the cardiac 4D CT experiment.

Port of Projects/Static CT/Projects AH/DualSource/
DualSource_CardiacMotion4DCT_SegmentedSingleSource.m.

Every segment scans the complete half turn with a shortened rotation time, centred on its
own slot of the single source read window. Even numbered segments are mirrored along the
detector axis (xi -> -xi), so the concatenated sinogram holds Nsources * Npi views.
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
from .DualSource_CardiacMotion4DCT_Main import corr2, rmse
from .HelperFunctions.DualSource_AH_plotScanTimingSegmentedSingle import (
    DualSource_AH_plotScanTimingSegmentedSingle,
)
from .Shared.DualSource_appendReconstructionMetrics import DualSource_appendReconstructionMetrics
from .Shared.DualSource_buildOutputNaming import DualSource_buildOutputNaming
from .Shared.DualSource_createGroundTruthGif import DualSource_createGroundTruthGif
from .Shared.DualSource_showAndSaveImage import DualSource_showAndSaveImage
from .Shared.DualSource_writeSimulationParameters import DualSource_writeSimulationParameters


def main(Outputdirectory=None, Reconkernel=Kernels.RamLak, ApplyMoireFilter=False,
         SimNoise=True, showPics=False, VisualiseTime=False, CreateGroundTruthGif=False,
         Nsources=2, N_I0=1e6, TRot=0.5,
         RM=60, M=500, Npar=512, Nx=512, Ny=512, xc=0, yc=0,
         SelectedPhantom="WaterWith4Arteries", PhantomOld=False,
         Heartrate=60, CM=False, Stent=False, Plaque=1,
         Direction=0, Rotation=0, MotionType="full", TRest=0, Shift_GT=0):
    if Outputdirectory is None:
        Outputdirectory = Path(IOHelperFunctions().GlobalOutputFolderPath) \
            / "cardiacPhantom" / "data"

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
    Shift_GT_Rest = THeartCycle - TRest / 2

    naming = DualSource_buildOutputNaming(SelectedPhantom, Plaque, CM, Stent)

    Outputdirectory = Path(Outputdirectory) / naming["RootFolderLabel"] / (
        f"SegmentedSingle_aRot_{int(Rotation)}"
        f"_HR_{int(Heartrate)}"
        f"_TRot_{round(TRot * 1000)}ms"
        f"_N_{int(Npar)}"
        f"_PS_{round(Shift_GT * 1000)}"
        f"_NSeg_{int(Nsources)}"
        f"_Ph_{naming['PhantomCode']}"
    )
    Outputdirectory.mkdir(parents=True, exist_ok=True)

    DualSource_writeSimulationParameters(
        Outputdirectory, Nsources, TRot, Heartrate, dT,
        RM, M, Npar, N2pi, dtheta, Nx, Ny, dx, dy,
        Shift_GT, Shift_GT_Rest, Rotation, Direction, MotionType, TRest,
        ApplyMoireFilter, SimNoise, N_I0, naming)

    ThetaArray = np.linspace(0, 2 * math.pi, N2pi + 1)[:-1]
    dtheta = ThetaArray[1] - ThetaArray[0]

    Npi = N2pi // 2

    PhantomOptions = dict(SelectedPhantom=SelectedPhantom, Heartrate=Heartrate, CM=CM,
                          Plaque=Plaque, Stent=Stent, PhantomOld=PhantomOld,
                          Direction=Direction, Rotation=Rotation,
                          MotionType=MotionType, TRest=TRest)

    TRot_segmented = TRot / Nsources
    ThetaArraysingle = np.linspace(0, math.pi, N2pi // 2 + 1)[:-1]

    Projections_multi = np.empty(0)
    for sourceIdx in range(1, Nsources + 1):
        segmentStartTime = Shift_GT - Shift_Single + (sourceIdx - 1) * (2 * Shift_Multi)
        segmentEndTime = Shift_GT - Shift_Single + sourceIdx * (2 * Shift_Multi)
        segmentCenterTime = (segmentStartTime + segmentEndTime) / 2

        ProjSegment, _, _, _, _ = \
            ForwardProjection.RayDrivenForwProjParCartesianMultipleWithMotionDual(
                x0, dx, Nx, y0, dy, Ny,
                N // 2, xi0, dxi, M, N2pi, TRot_segmented, ThetaArraysingle,
                Shift_GT=segmentCenterTime, **PhantomOptions)
        ProjSegment = np.asarray(ProjSegment).ravel()

        if sourceIdx % 2 == 0:
            seg = ProjSegment.reshape(-1, M)
            seg = seg[:, ::-1]  # mirror detector elements (xi -> -xi)
            ProjSegment = seg.ravel()

        Projections_multi = np.concatenate((Projections_multi, ProjSegment))

    Projections, _, points_single, _, _ = \
        ForwardProjection.RayDrivenForwProjParCartesianMultipleWithMotionDual(
            x0, dx, Nx, y0, dy, Ny,
            N // 2, xi0, dxi, M, N2pi, TRot, ThetaArraysingle,
            Shift_GT=Shift_GT - Shift_Single, **PhantomOptions)
    Projections = np.asarray(Projections).ravel()

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
        DualSource_AH_plotScanTimingSegmentedSingle(
            Heartrate, MotionType, TRest, Shift_GT, TRot, Nsources, THeartCycle,
            Shift_Single, TRot_segmented)

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
        N_I0_per_view_multi = N_I0 / Nsources
        Projections_multi = Noise.ApplyPoissonNoise(
            Projections_multi, Projections_multi.size, N_I0_per_view_multi)

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

        cx = max(1, min(Nx, round((points_single[0, midProj - 1] - x0) / dx)))
        cy = max(1, min(Ny, round((points_single[1, midProj - 1] - y0) / dy)))

        r = 80
        X, Y = np.meshgrid(np.arange(1, Nx + 1), np.arange(1, Ny + 1))
        roiMask = ((X - cx) ** 2 + (Y - cy) ** 2) <= r ** 2

        if showPics:
            DualSource_showAndSaveImage(Projections[idx], (M, Npi), Outputdirectory,
                                        "Proj-single", True, params)
            DualSource_showAndSaveImage(Projections_multi, (M, NprojMulti), Outputdirectory,
                                        "Proj-multi", True, params)
            for name, Data in (
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
        # The segmented scan keeps all Nsources half turns, so it backprojects NprojMulti
        # views over the same angular range.
        ReconstructedImage_multi = Backprojection.PixDrivenBackProjParCartesian(
            theta0_i, dtheta, NprojMulti, N2pi, xi0, dxi, M,
            Projections_multi, x0, dx, Nx, y0, dy, Ny)
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
                     "NprojMulti": NprojMulti, "NprojSingle": NprojSingle,
                     "theta0": theta0, "dtheta": dtheta, "xi0": xi0, "dxi": dxi,
                     "x0": x0, "dx": dx, "Nx": Nx, "y0": y0, "dy": dy, "Ny": Ny,
                     "dT": dT, "TRot_segmented": TRot_segmented,
                     "Shift_Single": Shift_Single, "Shift_Multi": Shift_Multi,
                     "Shift_GT_Rest": Shift_GT_Rest},
    }
