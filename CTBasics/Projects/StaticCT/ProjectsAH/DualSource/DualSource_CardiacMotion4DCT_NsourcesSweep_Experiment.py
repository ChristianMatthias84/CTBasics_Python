"""Sweep of the number of sources in the cardiac 4D CT experiment.

Port of Projects/Static CT/Projects AH/DualSource/
DualSource_CardiacMotion4DCT_NsourcesSweep_Experiment.m.

The single source and ground truth reconstructions are computed once; only the multi source
scan is repeated for every entry of NsourcesList.
"""
import csv
import math
from pathlib import Path

import numpy as np

from .....MultiCore.Parallelization import Parallelization
from .....ReadAndWrite.IOHelperFunctions import IOHelperFunctions
from .....Reconstruction.Backprojection import Backprojection
from .....Reconstruction.Convolution import Convolution
from .....Reconstruction.ForwardProjection import ForwardProjection
from .....Reconstruction.Kernels import Kernels
from .....Reconstruction.MoireFilter import MoireFilter
from .....Simulation.Noise import Noise
from .DualSource_CardiacMotion4DCT_Main import corr2, rmse
from .Shared.DualSource_appendReconstructionMetrics import DualSource_appendReconstructionMetrics
from .Shared.DualSource_buildOutputNaming import DualSource_buildOutputNaming
from .Shared.DualSource_writeSimulationParameters import DualSource_writeSimulationParameters

SummaryVarNames = [
    "Nsources",
    "corr2_single", "corr2_multi",
    "corr2_single_rest", "corr2_multi_rest",
    "rmse_single", "rmse_multi",
    "rmse_single_rest", "rmse_multi_rest",
    "corr2_single_roi", "corr2_multi_roi",
    "corr2_single_rest_roi", "corr2_multi_rest_roi",
    "rmse_single_roi", "rmse_multi_roi",
    "rmse_single_rest_roi", "rmse_multi_rest_roi",
]


def main(OutputRoot=None, ShowFigures=False, Reconkernel=Kernels.RamLak,
         ApplyMoireFilter=True, SimNoise=True, NsourcesList=(1, 2, 3, 4, 5),
         N_I0=1e7, TRot=0.5, RM=60, M=500, Npar=512, Nx=512, Ny=512,
         SelectedPhantom="CardiacPhantom", PhantomOld=False,
         Heartrate=60, CM=False, Stent=False, Plaque=1,
         Direction=0, Rotation=0, MotionType="full", TRest=0, Shift_GT=0):
    if OutputRoot is None:
        OutputRoot = Path(IOHelperFunctions().GlobalOutputFolderPath) \
            / "cardiacPhantom" / "data"

    NsourcesList = list(NsourcesList)

    thetaN = 2 * math.pi
    N2pi = 2 * Npar
    N = int(N2pi * thetaN / (2 * math.pi))

    dxi = 2 * RM / (M - 1)
    xi0 = -RM - dxi
    theta0 = 0

    dx = 2 * RM / Nx
    dy = 2 * RM / Ny
    x0 = -RM - dx / 2.0
    y0 = -RM - dy / 2.0

    dT = TRot / N2pi

    Shift_Single = TRot / 4
    THeartCycle = 60 / Heartrate
    Shift_GT_Rest = THeartCycle - TRest / 2

    naming = DualSource_buildOutputNaming(SelectedPhantom, Plaque, CM, Stent)
    OutputRoot = Path(OutputRoot) / naming["RootFolderLabel"] / \
        f"NsourcesSweepNoise_Ph_{naming['PhantomCode']}"

    Parallelization.Init()
    OutputRoot.mkdir(parents=True, exist_ok=True)

    ThetaArray = np.linspace(0, 2 * math.pi, N2pi + 1)[:-1]
    dtheta = ThetaArray[1] - ThetaArray[0]
    Npi = N2pi // 2

    # MATLAB does not pass SelectedPhantom here, so the projector default 'Cardiac' applies.
    PhantomOptions = dict(Heartrate=Heartrate, CM=CM, Plaque=Plaque, Stent=Stent,
                          PhantomOld=PhantomOld, Direction=Direction, Rotation=Rotation,
                          MotionType=MotionType, TRest=TRest)

    print("--- Simulating single-source projections ...")
    ThetaArraySingle = np.linspace(0, math.pi, Npi + 1)[:-1]
    Projections_single, _, points_single, _, _ = \
        ForwardProjection.RayDrivenForwProjParCartesianMultipleWithMotionDual(
            x0, dx, Nx, y0, dy, Ny,
            N // 2, xi0, dxi, M, N2pi, TRot, ThetaArraySingle,
            Shift_GT=Shift_GT - Shift_Single, **PhantomOptions)
    Projections_single = np.asarray(Projections_single).ravel()

    print("--- Simulating ground-truth projections ...")
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

    NprojSingle = Projections_single.size // M
    NprojGround = Projections_groundtruth.size // M
    NprojGroundRest = Projections_groundtruth_rest.size // M

    for Data in (Projections_single, Projections_groundtruth, Projections_groundtruth_rest):
        if Data.size % M != 0:
            raise ValueError("Projection length must be divisible by M.")

    if ApplyMoireFilter:
        Projections_single = MoireFilter.ApplyMoireFilterOnProjections(
            Projections_single, NprojSingle, M, dxi, 0.0 * (1 / dxi), 0.5 * (1 / dxi))
        Projections_groundtruth = MoireFilter.ApplyMoireFilterOnProjections(
            Projections_groundtruth, NprojGround, M, dxi, 0.0 * (1 / dxi), 0.5 * (1 / dxi))
        Projections_groundtruth_rest = MoireFilter.ApplyMoireFilterOnProjections(
            Projections_groundtruth_rest, NprojGroundRest, M, dxi,
            0.0 * (1 / dxi), 0.5 * (1 / dxi))

    if SimNoise:
        Projections_single = Noise.ApplyPoissonNoise(
            Projections_single, M * NprojSingle, N_I0)

    Kernel = None
    if Reconkernel == Kernels.SheppLogan:
        Kernel = Kernels.SheppLoganKernelSD(M, dxi)
    elif Reconkernel == Kernels.RamLak:
        Kernel = Kernels.RamLakKernelSD(M, dxi)

    Projections_single_conv = Convolution.ConvolveSinogram(
        Projections_single, Kernel, M, NprojSingle, dxi)
    Projections_ground_conv = Convolution.ConvolveSinogram(
        Projections_groundtruth, Kernel, M, N, dxi)
    Projections_ground_rest_conv = Convolution.ConvolveSinogram(
        Projections_groundtruth_rest, Kernel, M, N, dxi)

    idx = slice(0, Npi * M)
    theta0_i = theta0

    ReconSingle = Backprojection.PixDrivenBackProjParCartesian(
        theta0_i, dtheta, Npi, N2pi, xi0, dxi, M,
        Projections_single_conv[idx], x0, dx, Nx, y0, dy, Ny)
    ReconGT = Backprojection.PixDrivenBackProjParCartesian(
        theta0_i, dtheta, Npi, N2pi, xi0, dxi, M,
        Projections_ground_conv[idx], x0, dx, Nx, y0, dy, Ny)
    ReconGTRest = Backprojection.PixDrivenBackProjParCartesian(
        theta0_i, dtheta, Npi, N2pi, xi0, dxi, M,
        Projections_ground_rest_conv[idx], x0, dx, Nx, y0, dy, Ny)

    projStart = max(1, math.ceil(1 / M))
    projEnd = min(np.shape(points_single)[1], math.ceil((Npi * M) / M))
    midProj = round((projStart + projEnd) / 2)

    cx = max(1, min(Nx, round((points_single[0, midProj - 1] - x0) / dx)))
    cy = max(1, min(Ny, round((points_single[1, midProj - 1] - y0) / dy)))

    r = 80
    X, Y = np.meshgrid(np.arange(1, Nx + 1), np.arange(1, Ny + 1))
    roiMask = ((X - cx) ** 2 + (Y - cy) ** 2) <= r ** 2

    GT = ReconGT.reshape(Ny, Nx)
    GTRest = ReconGTRest.reshape(Ny, Nx)
    Single = ReconSingle.reshape(Ny, Nx)

    GT_roi = GT[roiMask]
    GTRest_roi = GTRest[roiMask]
    Single_roi = Single[roiMask]

    corr2_single = corr2(GT, Single)
    corr2_single_rest = corr2(GTRest, Single)
    rmse_single = rmse(GT, Single)
    rmse_single_rest = rmse(GTRest, Single)
    corr2_single_roi = corr2(GT_roi, Single_roi)
    corr2_single_rest_roi = corr2(GTRest_roi, Single_roi)
    rmse_single_roi = rmse(GT_roi, Single_roi)
    rmse_single_rest_roi = rmse(GTRest_roi, Single_roi)

    print("\nSingle-source reference:")
    print(f"  corr2 = {corr2_single:.4f}   rmse = {rmse_single:.4f}")
    print(f"  corr2 (rest GT) = {corr2_single_rest:.4f}   rmse = {rmse_single_rest:.4f}")

    if ShowFigures:
        _ShowReferenceFigure(GT, Single, corr2_single, rmse_single, SelectedPhantom)

    summaryMetrics = []
    ReconsMulti = {}

    print(f"\n--- Starting sweep over Nsources = {NsourcesList[0]} .. "
          f"{NsourcesList[-1]} ---\n")

    for Nsources in NsourcesList:
        Shift_Multi = TRot / (4 * Nsources)
        print(f"Nsources = {Nsources:2d}  (Shift_Multi = {Shift_Multi:.4f} s) ...")

        ThetaArrayMulti = np.linspace(0, math.pi, Npi + 1)[:-1]

        NprojPerSourceBase = Npi // Nsources
        NprojRemainder = Npi % Nsources
        NprojPerSource = np.full(Nsources, NprojPerSourceBase, dtype=np.int64)
        NprojPerSource[:NprojRemainder] += 1
        if int(NprojPerSource.sum()) != Npi:
            raise ValueError("Projection distribution does not sum to Npi.")

        Projections_multi_sources = []
        sourceStartIdx = 0
        for sourceIdx in range(Nsources):
            NprojSource = int(NprojPerSource[sourceIdx])
            ProjSource, _, _, _, _ = \
                ForwardProjection.RayDrivenForwProjParCartesianMultipleWithMotionDual(
                    x0, dx, Nx, y0, dy, Ny,
                    NprojSource, xi0, dxi, M, N2pi, TRot,
                    ThetaArrayMulti[sourceStartIdx:sourceStartIdx + NprojSource],
                    Shift_GT=Shift_GT - Shift_Multi, **PhantomOptions)
            Projections_multi_sources.append(np.asarray(ProjSource).ravel())
            sourceStartIdx += NprojSource

        Projections_multi = np.concatenate(Projections_multi_sources)
        if Projections_multi.size % M != 0:
            raise ValueError("Projections_multi length must be divisible by M.")
        NprojMulti = Projections_multi.size // M

        if ApplyMoireFilter:
            Projections_multi = MoireFilter.ApplyMoireFilterOnProjections(
                Projections_multi, NprojMulti, M, dxi, 0.0 * (1 / dxi), 0.5 * (1 / dxi))

        if SimNoise:
            # MATLAB re-concatenates the unfiltered per-source data here, so the Moire
            # filter result above is discarded when noise is enabled.
            N_I0_multi_per_source = N_I0 / Nsources
            for sourceIdx in range(Nsources):
                Projections_multi_sources[sourceIdx] = Noise.ApplyPoissonNoise(
                    Projections_multi_sources[sourceIdx],
                    Projections_multi_sources[sourceIdx].size,
                    N_I0_multi_per_source)
            Projections_multi = np.concatenate(Projections_multi_sources)

        Projections_multi_conv = Convolution.ConvolveSinogram(
            Projections_multi, Kernel, M, NprojMulti, dxi)

        ReconMulti = Backprojection.PixDrivenBackProjParCartesian(
            theta0_i, dtheta, Npi, N2pi, xi0, dxi, M,
            Projections_multi_conv[idx], x0, dx, Nx, y0, dy, Ny)

        Multi = ReconMulti.reshape(Ny, Nx)
        Multi_roi = Multi[roiMask]
        ReconsMulti[Nsources] = ReconMulti

        corr2_multi = corr2(GT, Multi)
        corr2_multi_rest = corr2(GTRest, Multi)
        rmse_multi = rmse(GT, Multi)
        rmse_multi_rest = rmse(GTRest, Multi)
        corr2_multi_roi = corr2(GT_roi, Multi_roi)
        corr2_multi_rest_roi = corr2(GTRest_roi, Multi_roi)
        rmse_multi_roi = rmse(GT_roi, Multi_roi)
        rmse_multi_rest_roi = rmse(GTRest_roi, Multi_roi)

        print(f"  corr2_multi = {corr2_multi:.4f}   rmse_multi = {rmse_multi:.4f}   "
              f"corr2_multi_rest = {corr2_multi_rest:.4f}   "
              f"rmse_multi_rest = {rmse_multi_rest:.4f}")

        summaryMetrics.append([
            Nsources,
            corr2_single, corr2_multi,
            corr2_single_rest, corr2_multi_rest,
            rmse_single, rmse_multi,
            rmse_single_rest, rmse_multi_rest,
            corr2_single_roi, corr2_multi_roi,
            corr2_single_rest_roi, corr2_multi_rest_roi,
            rmse_single_roi, rmse_multi_roi,
            rmse_single_rest_roi, rmse_multi_rest_roi,
        ])

        nsDir = OutputRoot / f"Nsources_{Nsources:02d}"
        nsDir.mkdir(parents=True, exist_ok=True)

        DualSource_writeSimulationParameters(
            nsDir, Nsources, TRot, Heartrate, dT,
            RM, M, Npar, N2pi, dtheta, Nx, Ny, dx, dy,
            Shift_GT, Shift_GT_Rest, Rotation, Direction, MotionType, TRest,
            ApplyMoireFilter, SimNoise, N_I0, naming)

        DualSource_appendReconstructionMetrics(
            nsDir,
            corr2_single, corr2_multi, corr2_single_rest, corr2_multi_rest,
            rmse_single, rmse_multi, rmse_single_rest, rmse_multi_rest,
            corr2_single_roi, corr2_multi_roi, corr2_single_rest_roi, corr2_multi_rest_roi,
            rmse_single_roi, rmse_multi_roi, rmse_single_rest_roi, rmse_multi_rest_roi)

    summaryCsv = OutputRoot / "reconstruction_metrics_nsources_sweep.csv"
    with open(summaryCsv, "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(SummaryVarNames)
        writer.writerows(summaryMetrics)

    # MATLAB additionally stores a .mat file; numpy's .npz is the equivalent here.
    summaryNpz = OutputRoot / "reconstruction_metrics_nsources_sweep.npz"
    np.savez(summaryNpz, summaryMetrics=np.asarray(summaryMetrics, dtype=float),
             NsourcesList=np.asarray(NsourcesList, dtype=np.int64),
             summaryVarNames=np.asarray(SummaryVarNames))

    print(f"\nSummary saved:\n  {summaryCsv}\n  {summaryNpz}")

    return {
        "OutputRoot": str(OutputRoot),
        "SummaryVarNames": SummaryVarNames,
        "summaryMetrics": np.asarray(summaryMetrics, dtype=float),
        "NsourcesList": NsourcesList,
        "ReconSingle": ReconSingle,
        "ReconGT": ReconGT,
        "ReconGTRest": ReconGTRest,
        "ReconsMulti": ReconsMulti,
        "roiMask": roiMask,
        "Kernel": Kernel,
        "Geometry": {"RM": RM, "M": M, "N": N, "N2pi": N2pi, "Npi": Npi,
                     "theta0": theta0, "dtheta": dtheta, "xi0": xi0, "dxi": dxi,
                     "x0": x0, "dx": dx, "Nx": Nx, "y0": y0, "dy": dy, "Ny": Ny,
                     "dT": dT, "Shift_Single": Shift_Single,
                     "Shift_GT_Rest": Shift_GT_Rest},
    }


def _ShowReferenceFigure(GT, Single, corr2_single, rmse_single, SelectedPhantom):
    import matplotlib.pyplot as plt

    refMin = float(min(GT.min(), Single.min()))
    refMax = float(max(GT.max(), Single.max()))

    fig, axes = plt.subplots(1, 2, facecolor="w")
    for ax, Data, title in (
            (axes[0], GT, "Ground Truth Recon"),
            (axes[1], Single,
             f"Single Recon (corr2={corr2_single:.4f}, rmse={rmse_single:.4f})")):
        handle = ax.imshow(Data, cmap="gray", vmin=refMin, vmax=refMax)
        ax.axis("off")
        ax.set_title(title)
        fig.colorbar(handle, ax=ax)

    fig.suptitle(f"Reference comparison - Phantom: {SelectedPhantom}")
    plt.show()
