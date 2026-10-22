"""Cardiac 4D CT: linear versus subset interleaved (``SubsetInterleaving_idx``) scan order.

Port of Projects/Static CT/Projects JP/StaticCT_4_Cardiac_JP.m (a script). The script
variables are keyword arguments of ``main``. Like test_4DCT_prosp, the linear and ground
truth reconstructions are cached in a .mat file; MATLAB forces a fresh linear
reconstruction (``linearRecon = true``), mirrored by ``ForceLinearRecon=True``.

The private helpers below are shared with StaticCT_4_Respiratory_JP and with
ProjectsAH/sweeps/StaticCT_4_Cardiac_JP, whose MATLAB sources repeat the same blocks.
"""
import math
from pathlib import Path

import numpy as np

from CTBasics.Memory.Allocation import Allocation
from CTBasics.MultiCore.Parallelization import Parallelization
from CTBasics.RawData.RawDataHelperFunctions import RawDataHelperFunctions
from CTBasics.ReadAndWrite.IOHelperFunctions import IOHelperFunctions
from CTBasics.Reconstruction.Backprojection import Backprojection
from CTBasics.Reconstruction.Constants import Constants
from CTBasics.Reconstruction.Convolution import Convolution
from CTBasics.Reconstruction.ForwardProjection import ForwardProjection
from CTBasics.Reconstruction.Kernels import Kernels
from CTBasics.Reconstruction.MoireFilter import MoireFilter
from CTBasics.Simulation.CreatePhantom import CreatePhantom
from CTBasics.Simulation.Noise import Noise
from CTBasics.Projects.StaticCT.ProjectsAH.DualSource.DualSource_CardiacMotion4DCT_Main import _MatlabRound, _d
from CTBasics.Projects.StaticCT.StaticCTHelperFunctions import (StaticCTIOHelpers, StaticCTSystemHelperFunctions,
                                       StaticCTVisualizationAndEvaluationHelpers)
from CTBasics.Projects.StaticCT.ProjectsJP.test_4DCT_prosp import _CacheMatches, _NormalizedMotion, _PlotSampling, _UnpackCell

_CardiacPhantoms = ("Cardiac", "WaterWith4Arteries", "LinePattern")


def main(Outputdirectory_base=None, reconType="180", Reconkernel=Kernels.RamLak,
         ApplyMoireFilter=False, SimNoise=False, NSubsets=64, NRecons=8,
         NScansPerRecon=1, isplot=True, phasesToPlot=(1, 3, 4), isGIF=False,
         writeData=True, TRot=0.5, N_I0=2.25e5, SelectedPhantom="WaterWith4Arteries",
         Heartrate=60, Motion=True, Direction=0, MotionType="half", TRest=None,
         CM=False, Stent=False, Plaque=1, Rotation=math.pi / 4,
         RM=60, M=500, N2pi=1024, Nx=512, Ny=512,
         UseLinearCache=True, ForceLinearRecon=True):
    """Runs the script; ``TRest=None`` means the MATLAB default ``0.6 * 60 / Heartrate``."""
    if Outputdirectory_base is None:
        Outputdirectory_base = Path(IOHelperFunctions().GlobalOutputFolderPath) \
            / "cardiacPhantom" / "data"
    Outputdirectory_base = Path(Outputdirectory_base)

    Parallelization.Init()

    if SelectedPhantom not in _CardiacPhantoms:
        raise ValueError("SelectedPhantom must be one of: " + ", ".join(_CardiacPhantoms))

    if TRest is None:
        TRest = 0.6 * 60 / Heartrate  # MATLAB: TRest = 0.6 * THeart

    PhantomFlags = [SelectedPhantom, Heartrate, Motion, Direction, MotionType, TRest,
                    CM, Stent, Plaque, Rotation]

    THeart = 60 / Heartrate  # duration of one heart cycle
    G = _Geometry(reconType, NScansPerRecon, TRot, RM, M, N2pi, Nx, Ny)
    SimParams = [TRot, NRecons, NScansPerRecon, RM, M, N2pi, reconType, SimNoise, N_I0,
                 Reconkernel, ApplyMoireFilter]

    TRecon = np.linspace(0, THeart, NRecons + 1)[:-1]  # drop t = THeart (equals t = 0)
    TLeft = TRecon - G["TReconInterval"] / 2
    TRight = TRecon + G["TReconInterval"] / 2

    ThetaArray = np.linspace(0, 2 * math.pi, N2pi + 1)[:-1]
    ThetaArray_arbitrary = StaticCTSystemHelperFunctions.SubsetInterleaving_idx(
        N2pi, NSubsets, reconType, 0, ThetaArray)

    Names = _Names("HR", Heartrate, TRot, N2pi, NSubsets, NRecons, NScansPerRecon,
                   reconType, Direction, SelectedPhantom)
    linearScan_fullfile = Outputdirectory_base / Names["CacheFile"]
    Outputdirectory = Outputdirectory_base / Names["Directory"]
    Outputdirectory.mkdir(parents=True, exist_ok=True)

    linearScanData, linearRecon = _LoadCacheIfPresent(linearScan_fullfile, UseLinearCache)
    if ForceLinearRecon:
        linearRecon = True  # MATLAB: overwrite the cache after changes in linear/GT
    linearRecon = _CheckCache(linearRecon, linearScanData, TRecon, PhantomFlags, SimParams)

    ProjectionOptions = dict(Heartrate=Heartrate, CM=CM, Plaque=Plaque, Stent=Stent,
                             Motion=Motion, Direction=Direction,
                             SelectedPhantom=SelectedPhantom, Rotation=Rotation,
                             MotionType=MotionType, TRest=TRest)
    Projections_arbitrary, Projections_linear, Projections_GT, center_points_1 = \
        _ForwardProject(G, NRecons, NScansPerRecon, TLeft, TRecon, ThetaArray,
                        ThetaArray_arbitrary, linearRecon, ProjectionOptions)

    # MATLAB omits 'Heartrate' here, so the truth phantom uses the default 60 bpm.
    PhantomTruth, displacements, RoI_all = _GroundTruth(
        G, TRecon, SelectedPhantom,
        dict(CM=CM, Stent=Stent, Plaque=Plaque, Motion=Motion, Direction=Direction,
             Rotation=Rotation, MotionType=MotionType, TRest=TRest))
    PhantomTruthHU = _HU(PhantomTruth)

    Kernel, N_I0_per_scan = _FilterProjections(
        G, Projections_arbitrary, Projections_linear, Projections_GT, ApplyMoireFilter,
        SimNoise, N_I0, NScansPerRecon, Reconkernel, IncludeLinear=True,
        ArbitraryFirst=False)

    Projections_linear_save = [x.copy() for x in Projections_linear]
    Projections_GT_save = [x.copy() for x in Projections_GT]
    if not linearRecon:
        Projections_linear = _UnpackCell(linearScanData["Projections_linear_save"])
        Projections_GT = _UnpackCell(linearScanData["Projections_GT_save"])
        # MATLAB would re-save the never computed linear/GT projections here.
        Projections_linear_save = [x.copy() for x in Projections_linear]
        Projections_GT_save = [x.copy() for x in Projections_GT]

    # MATLAB has the per-scan reconstruction and the averaging commented out.
    MultiphaseRecon_linear, MultiphaseRecon_arbitrary, MultiphaseRecon_GT = \
        _BackprojectLongSinogram(G, Projections_arbitrary, Projections_linear,
                                 Projections_GT, NScansPerRecon, linearRecon)
    if not linearRecon:
        MultiphaseRecon_linear = np.asarray(
            linearScanData["MultiphaseRecon_linear_save"], dtype=float)
        MultiphaseRecon_GT = np.asarray(linearScanData["MultiphaseRecon_GT_save"],
                                        dtype=float)

    MultiphaseReconHU_linear = _HU(MultiphaseRecon_linear)
    MultiphaseReconHU_GT = _HU(MultiphaseRecon_GT)
    MultiphaseReconHU_arbitrary = _HU(MultiphaseRecon_arbitrary)

    MultiphaseRecon_arbitrary_GT = MultiphaseReconHU_arbitrary - MultiphaseReconHU_GT
    MultiphaseRecon_linear_GT = MultiphaseReconHU_linear - MultiphaseReconHU_GT
    MultiphaseRecon_diff = MultiphaseReconHU_arbitrary - MultiphaseReconHU_linear

    MultiphaseRecon_linear_save = MultiphaseRecon_linear.copy()
    MultiphaseRecon_GT_save = MultiphaseRecon_GT.copy()

    roi_radius = 15  # RoI radius in mm (small moving structures)
    (RMSE_linear, RMSE_arbitrary, NCC_linear, NCC_arbitrary,
     ssim_linear, ssim_arbitrary, nRoI, roi_mask, center_mask,
     RMSE_linear_bg, RMSE_arbitrary_bg, NCC_linear_bg, NCC_arbitrary_bg,
     ssim_linear_bg, ssim_arbitrary_bg, bg_mask) = \
        StaticCTVisualizationAndEvaluationHelpers.StaticCT_computeReconstructionMetrics(
            MultiphaseReconHU_linear, MultiphaseReconHU_arbitrary,
            MultiphaseReconHU_GT, RoI_all, G["Nx"], G["Ny"], G["x0"], G["y0"],
            G["dx"], G["dy"], roi_radius)

    WC, WW = 0, 500
    if isplot:
        StaticCTVisualizationAndEvaluationHelpers.StaticCT_plotReconstructionResults(
            MultiphaseReconHU_GT, MultiphaseReconHU_linear, MultiphaseReconHU_arbitrary,
            MultiphaseRecon_linear_GT, MultiphaseRecon_arbitrary_GT,
            RMSE_linear, RMSE_arbitrary, NCC_linear, NCC_arbitrary,
            ssim_linear, ssim_arbitrary, roi_mask, center_mask, Nx, Ny, nRoI,
            WC, WW, NSubsets, phasesToPlot)

    figx = None
    if isplot or writeData:
        figx = _SamplingFigure(G, TRecon, TLeft, TRight, THeart, displacements,
                               center_points_1, Motion, Heartrate, TRest, MotionType,
                               Direction, NScansPerRecon, TRot, isplot)

    summary = None
    if writeData:
        summary = StaticCTIOHelpers.StaticCT_appendReconstructionMetrics(
            Outputdirectory, SelectedPhantom, TRecon,
            RMSE_linear, RMSE_arbitrary, NCC_linear, NCC_arbitrary,
            ssim_linear, ssim_arbitrary,
            RoIAll=RoI_all, RoIRadiusMm=roi_radius, Tag=Names["Tag"])
        _WriteReconstructions(Outputdirectory, "HR", Heartrate, TRot, N2pi, NSubsets,
                              NScansPerRecon, NRecons, THeart, Nx, Ny, WC, WW,
                              MultiphaseReconHU_linear, MultiphaseRecon_linear_GT,
                              MultiphaseReconHU_arbitrary, MultiphaseRecon_arbitrary_GT,
                              MultiphaseReconHU_GT, figx)
        IOHelperFunctions.SaveStructToMat(
            Outputdirectory / "additionalInfo.mat",
            {"TRecon": TRecon, "roi_radius": roi_radius,
             "NCC_linear": NCC_linear, "NCC_arbitrary": NCC_arbitrary,
             "RMSE_linear": RMSE_linear, "RMSE_arbitrary": RMSE_arbitrary,
             "reconType": reconType, "N2pi": N2pi, "NSubsets": NSubsets,
             "ThetaArray_arbitrary": ThetaArray_arbitrary, "Heartrate": Heartrate,
             "ssim_arbitrary": ssim_arbitrary, "ssim_linear": ssim_linear})

    if UseLinearCache:
        _SaveCache(linearScan_fullfile, TRecon, MultiphaseRecon_linear_save,
                   MultiphaseRecon_GT_save, PhantomFlags, SimParams,
                   Projections_linear_save, Projections_GT_save)

    if isGIF:
        StaticCTIOHelpers.StaticCT_createSubsetAnimationGif(
            ThetaArray_arbitrary, reconType, NSubsets, Outputdirectory,
            MaxAngles=100, MaxFrames=100, DelayTime=0.25, Visible=False)

    _CloseUnlessShown(figx, isplot)

    return {
        "Outputdirectory": str(Outputdirectory),
        "linearScan_fullfile": str(linearScan_fullfile),
        "linearRecon": linearRecon,
        "PhantomFlags": PhantomFlags, "SimParams": SimParams,
        "N_I0_per_scan": N_I0_per_scan,
        "TRecon": TRecon, "TLeft": TLeft, "TRight": TRight,
        "ThetaArray": ThetaArray, "ThetaArray_arbitrary": ThetaArray_arbitrary,
        "Projections_linear": Projections_linear,
        "Projections_arbitrary": Projections_arbitrary,
        "Projections_GT": Projections_GT,
        "PhantomTruth": PhantomTruth, "PhantomTruthHU": PhantomTruthHU,
        "RoI_all": RoI_all, "displacements": displacements,
        "center_points_1": center_points_1, "Kernel": Kernel,
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
        "RMSE_linear_bg": RMSE_linear_bg, "RMSE_arbitrary_bg": RMSE_arbitrary_bg,
        "NCC_linear_bg": NCC_linear_bg, "NCC_arbitrary_bg": NCC_arbitrary_bg,
        "ssim_linear_bg": ssim_linear_bg, "ssim_arbitrary_bg": ssim_arbitrary_bg,
        "bg_mask": bg_mask,
        "nRoI": nRoI, "roi_mask": roi_mask, "center_mask": center_mask,
        "roi_radius": roi_radius, "WC": WC, "WW": WW,
        "Summary": summary,
        "Geometry": dict(G, THeart=THeart, NRecons=NRecons,
                         NScansPerRecon=NScansPerRecon, NSubsets=NSubsets),
    }


# ---------------------------------------------------------------------------------------
# Blocks shared by StaticCT_4_Cardiac_JP, StaticCT_4_Respiratory_JP and the AH sweep copy.
# ---------------------------------------------------------------------------------------

def _Geometry(reconType, NScansPerRecon, TRot, RM, M, N2pi, Nx, Ny):
    """Scan and image geometry of the JP scripts (theta0 = -dtheta)."""
    if reconType == "180":
        TReconInterval = NScansPerRecon * TRot / 2  # time interval of one reconstruction
        thetaN = math.pi
    elif reconType == "360":
        thetaN = 2 * math.pi
        TReconInterval = NScansPerRecon * TRot
    else:
        raise ValueError("reconType must be '180' or '360'.")

    N = _MatlabRound(N2pi * thetaN / (2 * math.pi))
    dxi = 2 * RM / (M - 1)
    dtheta = thetaN / N
    dx = 2 * RM / Nx
    dy = 2 * RM / Ny
    return {"RM": RM, "M": M, "N": N, "N2pi": N2pi, "Npi": N2pi // 2, "thetaN": thetaN,
            "theta0": -dtheta, "dtheta": dtheta, "dT": TRot / N2pi, "TRot": TRot,
            "xi0": -RM - dxi, "dxi": dxi, "x0": -RM - dx / 2.0, "dx": dx, "Nx": Nx,
            "y0": -RM - dy / 2.0, "dy": dy, "Ny": Ny, "TReconInterval": TReconInterval}


def _Names(RateLabel, Rate, TRot, N2pi, NSubsets, NRecons, NScansPerRecon, reconType,
           Direction, SelectedPhantom, Phase=""):
    """Cache file, output directory and metrics tag (MATLAB %d formatting)."""
    TRotMs = _d(_MatlabRound(TRot * 1000))
    Tail = f"_{_d(int(reconType))}Recon_{_d(math.degrees(Direction))}deg_{SelectedPhantom}"
    Head = f"{RateLabel}_{_d(Rate)}_TRot_{TRotMs}ms_N2pi_{_d(N2pi)}"
    Scans = f"_NRecons{_d(NRecons)}_NScans{_d(NScansPerRecon)}{Phase}"
    return {"CacheFile": f"linearRecon_{Head}{Scans}{Tail}.mat",
            "Directory": f"{Head}_NSub_{_d(NSubsets)}{Scans}{Tail}",
            "Tag": f"{Head}_NSub_{_d(NSubsets)}_NScans{_d(NScansPerRecon)}_{reconType}"}


def _LoadCacheIfPresent(linearScan_fullfile, UseLinearCache):
    """Loads ``linearScanData`` (MATLAB struct) into a dict; returns (data, linearRecon)."""
    if not (UseLinearCache and Path(linearScan_fullfile).is_file()):
        return None, True
    from scipy.io import loadmat

    Data = loadmat(str(linearScan_fullfile))
    if "linearScanData" not in Data:
        return {}, False  # not a JP cache: _CacheMatches rejects it
    S = Data["linearScanData"][0, 0]
    return {Name: S[Name] for Name in S.dtype.names}, False


def _CheckCache(linearRecon, linearScanData, TRecon, PhantomFlags, SimParams):
    if linearRecon:
        return True
    if not _CacheMatches(linearScanData, TRecon, PhantomFlags, SimParams):
        print("selected and saved reconstruction times are different. linear "
              "reconstruction can not be used! linear Recon must be done again.")
        return True
    print("Previous linear recon will be used...")
    return False


def _CellOfArrays(values):
    """Object array of arrays (scipy writes a MATLAB cell), also for equal shapes."""
    Cell = np.empty(len(values), dtype=object)
    for k, Value in enumerate(values):
        Cell[k] = Value
    return Cell


def _SaveCache(linearScan_fullfile, TRecon, MultiphaseRecon_linear_save,
               MultiphaseRecon_GT_save, PhantomFlags, SimParams,
               Projections_linear_save=None, Projections_GT_save=None):
    """MATLAB ``save(file, "linearScanData")`` with the struct fields in MATLAB order."""
    linearScanData = {"TRecon_save": TRecon}
    if Projections_linear_save is not None:
        linearScanData["Projections_linear_save"] = _CellOfArrays(Projections_linear_save)
    linearScanData["MultiphaseRecon_linear_save"] = MultiphaseRecon_linear_save
    if Projections_GT_save is not None:
        linearScanData["Projections_GT_save"] = _CellOfArrays(Projections_GT_save)
    linearScanData["MultiphaseRecon_GT_save"] = MultiphaseRecon_GT_save
    linearScanData["PhantomFlags"] = _CellOfArrays(PhantomFlags)
    linearScanData["SimParams"] = _CellOfArrays(SimParams)
    Path(linearScan_fullfile).parent.mkdir(parents=True, exist_ok=True)
    IOHelperFunctions.SaveStructToMat(linearScan_fullfile,
                                      {"linearScanData": linearScanData})


def _ForwardProject(G, NRecons, NScansPerRecon, TLeft, TRecon, ThetaArray,
                    ThetaArray_arbitrary, linearRecon, ProjectionOptions,
                    AllocateLinear=True):
    """Arbitrary, linear and motion free (TRot = 0.0001) projections per phase and scan."""
    N, M, N2pi, Npi = G["N"], G["M"], G["N2pi"], G["Npi"]
    Geometry = (G["x0"], G["dx"], G["Nx"], G["y0"], G["dy"], G["Ny"], N, G["xi0"],
                G["dxi"], M, N2pi)
    Project = ForwardProjection.\
        RayDrivenForwProjParCartesianMultipleWithMotionArbitraryOrderParallelized

    Projections_arbitrary = [Allocation.zeros(NScansPerRecon, N * M)
                             for _ in range(NRecons)]
    Projections_linear = None
    Projections_GT = None
    if AllocateLinear:
        Projections_linear = [Allocation.zeros(NScansPerRecon, N * M)
                              for _ in range(NRecons)]
        Projections_GT = [Allocation.zeros(NScansPerRecon, N * M)
                          for _ in range(NRecons)]
    center_points_1 = {}

    print("Forward Projection for specified recons running...")
    for k in range(NRecons):
        for j in range(NScansPerRecon):
            ThetaIdx = slice(Npi, N2pi) if (j + 1) % 2 == 0 else slice(0, Npi)
            T0_rot = TLeft[k] + j * (G["TRot"] / 2)  # start time of each half rotation

            Proj, _, points, _ = Project(*Geometry, G["TRot"],
                                         ThetaArray_arbitrary[ThetaIdx], T0=T0_rot,
                                         **ProjectionOptions)
            Projections_arbitrary[k][j, :] = np.asarray(Proj).ravel()
            center_points_1[(k, j)] = points

            if linearRecon:
                ProjLin, _, _, _ = Project(*Geometry, G["TRot"], ThetaArray[ThetaIdx],
                                           T0=T0_rot, **ProjectionOptions)
                Projections_linear[k][j, :] = np.asarray(ProjLin).ravel()
                # TRot = 0.0001 freezes the phantom, giving the motion free reference.
                ProjGT, _, _, _ = Project(*Geometry, 0.0001, ThetaArray[ThetaIdx],
                                          T0=TRecon[k], **ProjectionOptions)
                Projections_GT[k][j, :] = np.asarray(ProjGT).ravel()

    return Projections_arbitrary, Projections_linear, Projections_GT, center_points_1


def _GroundTruth(G, TRecon, SelectedPhantom, PhantomOptions):
    """Phantom, displacement and RoI per reconstructed phase."""
    NRecons = TRecon.size
    PhantomTruth = Allocation.zeros(NRecons, G["Nx"] * G["Ny"])
    displacements = Allocation.zeros(2, NRecons)
    RoI_all = []
    for k in range(NRecons):
        Phantom, _, xDisp, yDisp, RoI = \
            CreatePhantom.CreateVoxelPhantom2DWithMotionUnified(
                G["Nx"], G["dx"], G["x0"], G["Ny"], G["dy"], G["y0"], TRecon[k],
                G["dT"], 0, SelectedPhantom, **PhantomOptions)
        PhantomTruth[k, :] = np.asarray(Phantom).ravel()
        displacements[0, k] = xDisp
        displacements[1, k] = yDisp
        RoI_all.append(np.asarray(RoI, dtype=float))
    return PhantomTruth, displacements, RoI_all


def _FilterProjections(G, Projections_arbitrary, Projections_linear, Projections_GT,
                       ApplyMoireFilter, SimNoise, N_I0, NScansPerRecon, Reconkernel,
                       IncludeLinear, ArbitraryFirst):
    """Moire filter, Poisson noise and convolution (in place); returns (Kernel, I0)."""
    N, M, dxi = G["N"], G["M"], G["dxi"]
    Measured = [Projections_arbitrary]
    if IncludeLinear:
        Measured = [Projections_arbitrary, Projections_linear] if ArbitraryFirst \
            else [Projections_linear, Projections_arbitrary]
    NRecons = len(Projections_arbitrary)

    if ApplyMoireFilter:
        for k in range(NRecons):
            for j in range(NScansPerRecon):
                for P in Measured:
                    P[k][j, :] = MoireFilter.ApplyMoireFilterOnProjections(
                        P[k][j, :], N, M, dxi, 0.0 * (1 / dxi), 0.5 * (1 / dxi))

    N_I0_per_scan = None
    if SimNoise:
        # The photon budget of one reconstruction is shared by its scan segments.
        N_I0_per_scan = N_I0 / NScansPerRecon
        for k in range(NRecons):
            for j in range(NScansPerRecon):
                for P in Measured:
                    P[k][j, :] = Noise.ApplyPoissonNoise(P[k][j, :], P[k][j, :].size,
                                                         N_I0_per_scan)

    Kernel = None
    if Reconkernel == Kernels.SheppLogan:
        Kernel = Kernels.SheppLoganKernelSD(M, dxi)
    elif Reconkernel == Kernels.RamLak:
        Kernel = Kernels.RamLakKernelSD(M, dxi)

    Convolved = Measured + ([Projections_GT] if IncludeLinear else [])
    for k in range(NRecons):
        for j in range(NScansPerRecon):
            for P in Convolved:
                P[k][j, :] = Convolution.ConvolveSinogram(P[k][j, :], Kernel, M, N, dxi)
    return Kernel, N_I0_per_scan


def _BackprojectSingleScans(G, Projections_arbitrary, Projections_linear, reconType,
                            NScansPerRecon, IncludeLinear):
    """Per scan segment reconstruction (second half turn starts at theta0 + pi) and the
    average over the segments of each phase."""
    NRecons = len(Projections_arbitrary)
    K = G["Nx"] * G["Ny"]
    Single_arbitrary = np.zeros((NRecons, NScansPerRecon, K))
    Single_linear = np.zeros((NRecons, NScansPerRecon, K))
    for k in range(NRecons):
        for j in range(NScansPerRecon):
            theta0_scan = G["theta0"]
            if reconType == "180" and (j + 1) % 2 == 0:
                theta0_scan = G["theta0"] + math.pi
            Single_arbitrary[k, j, :] = _Backproject(G, theta0_scan, G["N"],
                                                     Projections_arbitrary[k][j, :])
            if IncludeLinear:
                Single_linear[k, j, :] = _Backproject(G, theta0_scan, G["N"],
                                                      Projections_linear[k][j, :])

    Average_linear = Allocation.zeros(NRecons, K)
    Average_arbitrary = Allocation.zeros(NRecons, K)
    for k in range(NRecons):
        sum_arbitrary = np.zeros(K)
        sum_linear = np.zeros(K)
        for j in range(NScansPerRecon):
            sum_arbitrary = sum_arbitrary + Single_arbitrary[k, j, :]
            if IncludeLinear:
                sum_linear = sum_linear + Single_linear[k, j, :]
        Average_arbitrary[k, :] = sum_arbitrary / NScansPerRecon
        if IncludeLinear:
            Average_linear[k, :] = sum_linear / NScansPerRecon
    return Single_arbitrary, Single_linear, Average_linear, Average_arbitrary


def _BackprojectLongSinogram(G, Projections_arbitrary, Projections_linear,
                             Projections_GT, NScansPerRecon, linearRecon):
    """Reconstruction of the concatenated scan segments of each phase."""
    NRecons = len(Projections_arbitrary)
    K = G["Nx"] * G["Ny"]
    MultiphaseRecon_linear = Allocation.zeros(NRecons, K)
    MultiphaseRecon_arbitrary = Allocation.zeros(NRecons, K)
    MultiphaseRecon_GT = Allocation.zeros(NRecons, K)
    Long = RawDataHelperFunctions.SimpleLinearizeArray
    for k in range(NRecons):
        if linearRecon:
            MultiphaseRecon_linear[k, :] = _Backproject(
                G, G["theta0"], NScansPerRecon * G["N"], Long(Projections_linear[k]))
            MultiphaseRecon_GT[k, :] = _Backproject(
                G, G["theta0"], NScansPerRecon * G["N"], Long(Projections_GT[k]))
        MultiphaseRecon_arbitrary[k, :] = _Backproject(
            G, G["theta0"], NScansPerRecon * G["N"], Long(Projections_arbitrary[k]))
    return MultiphaseRecon_linear, MultiphaseRecon_arbitrary, MultiphaseRecon_GT


def _Backproject(G, theta0, NProj, Projection):
    return Backprojection.PixDrivenBackProjParCartesian(
        theta0, G["dtheta"], NProj, G["N2pi"], G["xi0"], G["dxi"], G["M"], Projection,
        G["x0"], G["dx"], G["Nx"], G["y0"], G["dy"], G["Ny"])


def _HU(Image):
    return 1000 * (Image - Constants.MuW) / Constants.MuW


def _CompareLongVsAverage(HU_linear, HU_linear_avg, HU_arbitrary, HU_arbitrary_avg,
                          IncludeLinear=True):
    """Long sinogram versus averaged single scan reconstructions (printed)."""
    eps = np.finfo(float).eps
    Result = {}
    print("Comparison long-sinogram vs averaged single-scan reconstructions:")
    for Name, HU, Avg in (("linear", HU_linear, HU_linear_avg),
                          ("arbitrary", HU_arbitrary, HU_arbitrary_avg)):
        Label = Name.capitalize()
        if Name == "linear" and not IncludeLinear:
            print("  Linear: reused cached reconstruction (NSubsets-independent).")
            continue
        Delta = HU - Avg
        maxAbsDelta = float(np.max(np.abs(Delta)))
        relDiff = float(np.linalg.norm(Delta.ravel())
                        / max(np.linalg.norm(HU.ravel()), eps))
        print(f"  {Label}: max abs delta = {maxAbsDelta:.6e} HU, "
              f"relative norm diff = {relDiff:.6e}")
        Result.update({f"MultiphaseRecon_{Name}_delta": Delta,
                       f"maxAbsDelta_{Name}": maxAbsDelta, f"relDiff_{Name}": relDiff})
    return Result


def _SamplingFigure(G, TRecon, TLeft, TRight, TCycle, displacements, center_points_1,
                    Motion, Rate, TRest, MotionType, Direction, NScansPerRecon, TRot,
                    show, axisLimits=(-0.2, 1.2, -11, 11)):
    """Continuous motion, reconstructed points and acquisition intervals."""
    t = np.linspace(0, max(TRecon.max(), TCycle), 1000)
    norm_displacement = _NormalizedMotion(t, Motion, Rate, TRest, MotionType)
    r = 5
    yDisplacement = 2 * r * norm_displacement * math.cos(Direction)
    xDisplacement = 2 * r * norm_displacement * math.sin(Direction)

    NRecons = TRecon.size
    yDisplacement_rot = np.zeros((NRecons, NScansPerRecon))
    xDisplacement_rot = np.zeros((NRecons, NScansPerRecon))
    Column = _MatlabRound(G["N"] / 2) - 1
    for k in range(NRecons):
        for j in range(NScansPerRecon):
            Points = np.asarray(center_points_1[(k, j)], dtype=float)
            yDisplacement_rot[k, j] = Points[1, Column]
            xDisplacement_rot[k, j] = Points[0, Column]

    return _PlotSampling(t, xDisplacement, yDisplacement, TRecon, TLeft, TRight,
                         displacements, displacements[0, :].copy(),
                         displacements[1, :].copy(), xDisplacement_rot,
                         yDisplacement_rot, NScansPerRecon, TRot, show=show,
                         axisLimits=axisLimits)


def _WriteReconstructions(Outputdirectory, RateLabel, Rate, TRot, N2pi, NSubsets,
                          NScansPerRecon, NRecons, TCycle, Nx, Ny, WC, WW,
                          HU_linear, linear_GT, HU_arbitrary, arbitrary_GT, HU_GT, figx):
    """Animated GIFs, raw double files and the sampling figure of one run."""
    print("Saving reconstructions as animated GIFs...")
    gif_WW_diff = max(np.percentile(linear_GT, 90), np.percentile(arbitrary_GT, 90))
    gif_fps = NRecons / TCycle  # one cycle per GIF loop
    Common = (f"TRot{TRot:.2f}s_N2pi{_d(N2pi)}_{NSubsets:.0f}subsets"
              f"_{NScansPerRecon:.0f}scans")
    suffix = f"{RateLabel}{Rate:.0f}_{Common}"

    for Data, Name, Label, Center, Width in (
            (HU_linear, "LinearReconHU_", "Linear Recon", WC, WW),
            (linear_GT, "LinearReconHU_difference_", "Linear Diff", 0, gif_WW_diff),
            (HU_arbitrary, "ArbitraryReconHU_", "Arbitrary Recon", WC, WW),
            (arbitrary_GT, "ArbitraryReconHU_difference_", "Arbitrary Diff", 0,
             gif_WW_diff),
            (HU_GT, "GroundTruthHU_", "Ground Truth", WC, WW)):
        IOHelperFunctions.SaveReconAsGif(Data, Outputdirectory / f"{Name}{suffix}.gif",
                                         Nx, Ny, NRecons, Label, Center, Width, gif_fps)

    rawSuffix = (f"{Rate:.0f}_{TRot:.2f}s_N2pi{_d(N2pi)}_{NSubsets:.0f}subsets"
                 f"_{NScansPerRecon:.0f}scans")
    for Data, Name in ((HU_linear, "LinearReconHU"),
                       (linear_GT, "LinearReconHU_difference"),
                       (HU_arbitrary, "ArbitraryReconHU"),
                       (arbitrary_GT, "ArbitraryReconHU_difference"),
                       (HU_GT, "GroundTruthHU")):
        IOHelperFunctions.SaveDoubleToFile(
            Outputdirectory / f"{Name}{rawSuffix}",
            RawDataHelperFunctions.SimpleLinearizeArray(Data))

    if figx is not None:
        figx.savefig(Outputdirectory / f"Sampling_arbitrary_{Rate:.0f}_{TRot:.2f}s"
                     f"_N2pi{_d(N2pi)}_{NScansPerRecon:.0f}scans.jpg", dpi=300)


def _CloseUnlessShown(figx, isplot):
    if figx is not None and not isplot:
        import matplotlib.pyplot as plt
        plt.close(figx)


if __name__ == "__main__":
    main()
