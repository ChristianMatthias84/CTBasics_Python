"""Respiratory 4D CT: linear versus subset interleaved (``SubsetInterleaving_idx``) scan order.

Port of Projects/Static CT/Projects JP/StaticCT_4_Respiratory_JP.m (a script). The script
variables are keyword arguments of ``main``; caching follows StaticCT_4_Cardiac_JP.
Every phase is reconstructed twice: per scan segment (then averaged) and from the
concatenated long sinogram; the metrics use the long sinogram reconstruction.
"""
import math
from pathlib import Path

import numpy as np

from CTBasics.MultiCore.Parallelization import Parallelization
from CTBasics.ReadAndWrite.IOHelperFunctions import IOHelperFunctions
from CTBasics.Reconstruction.Kernels import Kernels
from CTBasics.Projects.StaticCT.StaticCTHelperFunctions import (StaticCTIOHelpers, StaticCTSystemHelperFunctions,
                                       StaticCTVisualizationAndEvaluationHelpers)
from CTBasics.Projects.StaticCT.ProjectsJP.StaticCT_4_Cardiac_JP import (_BackprojectLongSinogram, _BackprojectSingleScans,
                                    _CheckCache, _CloseUnlessShown, _CompareLongVsAverage,
                                    _FilterProjections, _ForwardProject, _Geometry,
                                    _GroundTruth, _HU, _LoadCacheIfPresent, _Names,
                                    _SamplingFigure, _SaveCache, _WriteReconstructions)
from CTBasics.Projects.StaticCT.ProjectsJP.test_4DCT_prosp import _PlotSingleScans, _UnpackCell


def main(Outputdirectory_base=None, reconType="180", Reconkernel=Kernels.RamLak,
         ApplyMoireFilter=False, SimNoise=False, NSubsets=64, NRecons=8,
         NScansPerRecon=1, isplot=True, phasesToPlot=(1, 3, 4), isGIF=False,
         writeData=True, TRot=0.5, N_I0=1e7, BreathRate=15, Motion=True, Direction=0,
         MotionType="full", TRest=0, RM=150, M=500, N2pi=1024, Nx=512, Ny=512,
         UseLinearCache=True, ForceLinearRecon=True):
    if Outputdirectory_base is None:
        Outputdirectory_base = Path(IOHelperFunctions().GlobalOutputFolderPath) \
            / "respiratoryPhantom" / "data"
    Outputdirectory_base = Path(Outputdirectory_base)

    Parallelization.Init()

    SelectedPhantom = "Respiratory"
    PhantomFlags = [SelectedPhantom, BreathRate, Motion, Direction, MotionType, TRest]

    TBreath = 60 / BreathRate  # duration of one breathing cycle
    G = _Geometry(reconType, NScansPerRecon, TRot, RM, M, N2pi, Nx, Ny)
    SimParams = [TRot, NRecons, NScansPerRecon, RM, M, N2pi, reconType, SimNoise, N_I0,
                 Reconkernel, ApplyMoireFilter]

    TRecon = np.linspace(0, TBreath, NRecons + 1)[:-1]  # drop t = TBreath (equals t = 0)
    TLeft = TRecon - G["TReconInterval"] / 2
    TRight = TRecon + G["TReconInterval"] / 2

    ThetaArray = np.linspace(0, 2 * math.pi, N2pi + 1)[:-1]
    ThetaArray_arbitrary = StaticCTSystemHelperFunctions.SubsetInterleaving_idx(
        N2pi, NSubsets, reconType, 0, ThetaArray)

    Names = _Names("BR", BreathRate, TRot, N2pi, NSubsets, NRecons, NScansPerRecon,
                   reconType, Direction, SelectedPhantom)
    linearScan_fullfile = Outputdirectory_base / Names["CacheFile"]
    Outputdirectory = Outputdirectory_base / Names["Directory"]
    Outputdirectory.mkdir(parents=True, exist_ok=True)

    linearScanData, linearRecon = _LoadCacheIfPresent(linearScan_fullfile, UseLinearCache)
    if ForceLinearRecon:
        linearRecon = True  # MATLAB: overwrite the cache after changes in linear/GT
    linearRecon = _CheckCache(linearRecon, linearScanData, TRecon, PhantomFlags, SimParams)

    # The breathing rate is passed as 'Heartrate' (rate of the motion cycle).
    ProjectionOptions = dict(Heartrate=BreathRate, Motion=Motion, Direction=Direction,
                             SelectedPhantom=SelectedPhantom, MotionType=MotionType,
                             TRest=TRest)
    Projections_arbitrary, Projections_linear, Projections_GT, center_points_1 = \
        _ForwardProject(G, NRecons, NScansPerRecon, TLeft, TRecon, ThetaArray,
                        ThetaArray_arbitrary, linearRecon, ProjectionOptions)

    PhantomTruth, displacements, RoI_all = _GroundTruth(
        G, TRecon, SelectedPhantom,
        dict(Heartrate=BreathRate, Motion=Motion, Direction=Direction,
             MotionType=MotionType, TRest=TRest))
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

    (ReconstructedImage_arbitrary_single, ReconstructedImage_linear_single,
     MultiphaseRecon_linear_avg, MultiphaseRecon_arbitrary_avg) = \
        _BackprojectSingleScans(G, Projections_arbitrary, Projections_linear, reconType,
                                NScansPerRecon, IncludeLinear=True)
    MultiphaseReconHU_linear_avg = _HU(MultiphaseRecon_linear_avg)
    MultiphaseReconHU_arbitrary_avg = _HU(MultiphaseRecon_arbitrary_avg)
    ReconstructedImage_arbitrary_single_HU = _HU(ReconstructedImage_arbitrary_single)

    phaseIdx = min(2, NRecons)  # phase to inspect (1..NRecons)
    if isplot:
        _PlotSingleScans(ReconstructedImage_arbitrary_single_HU,
                         MultiphaseReconHU_arbitrary_avg, phaseIdx, NScansPerRecon,
                         Nx, Ny)

    print("Backprojection for each Time-point running...")
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

    Comparison = _CompareLongVsAverage(MultiphaseReconHU_linear,
                                       MultiphaseReconHU_linear_avg,
                                       MultiphaseReconHU_arbitrary,
                                       MultiphaseReconHU_arbitrary_avg)

    roi_radius = 60  # RoI radius in mm (large lung/liver structures)
    (RMSE_linear, RMSE_arbitrary, NCC_linear, NCC_arbitrary,
     ssim_linear, ssim_arbitrary, nRoI, roi_mask, center_mask) = \
        StaticCTVisualizationAndEvaluationHelpers.StaticCT_computeReconstructionMetrics(
            MultiphaseReconHU_linear, MultiphaseReconHU_arbitrary,
            MultiphaseReconHU_GT, RoI_all, G["Nx"], G["Ny"], G["x0"], G["y0"],
            G["dx"], G["dy"], roi_radius)[:9]

    WC, WW = -600, 1500  # lung window
    if isplot:
        StaticCTVisualizationAndEvaluationHelpers.StaticCT_plotReconstructionResults(
            MultiphaseReconHU_GT, MultiphaseReconHU_linear, MultiphaseReconHU_arbitrary,
            MultiphaseRecon_linear_GT, MultiphaseRecon_arbitrary_GT,
            RMSE_linear, RMSE_arbitrary, NCC_linear, NCC_arbitrary,
            ssim_linear, ssim_arbitrary, roi_mask, center_mask, Nx, Ny, nRoI,
            WC, WW, NSubsets, phasesToPlot)

    figx = None
    if isplot or writeData:
        # MATLAB has the fixed axis limits commented out for the breathing cycle.
        figx = _SamplingFigure(G, TRecon, TLeft, TRight, TBreath, displacements,
                               center_points_1, Motion, BreathRate, TRest, MotionType,
                               Direction, NScansPerRecon, TRot, isplot, axisLimits=None)

    summary = None
    if writeData:
        summary = StaticCTIOHelpers.StaticCT_appendReconstructionMetrics(
            Outputdirectory, SelectedPhantom, TRecon,
            RMSE_linear, RMSE_arbitrary, NCC_linear, NCC_arbitrary,
            ssim_linear, ssim_arbitrary,
            RoIAll=RoI_all, RoIRadiusMm=roi_radius, Tag=Names["Tag"])
        _WriteReconstructions(Outputdirectory, "BR", BreathRate, TRot, N2pi, NSubsets,
                              NScansPerRecon, NRecons, TBreath, Nx, Ny, WC, WW,
                              MultiphaseReconHU_linear, MultiphaseRecon_linear_GT,
                              MultiphaseReconHU_arbitrary, MultiphaseRecon_arbitrary_GT,
                              MultiphaseReconHU_GT, figx)
        IOHelperFunctions.SaveStructToMat(
            Outputdirectory / "additionalInfo.mat",
            {"TRecon": TRecon, "roi_radius": roi_radius,
             "NCC_linear": NCC_linear, "NCC_arbitrary": NCC_arbitrary,
             "RMSE_linear": RMSE_linear, "RMSE_arbitrary": RMSE_arbitrary,
             "reconType": reconType, "N2pi": N2pi, "NSubsets": NSubsets,
             "ThetaArray_arbitrary": ThetaArray_arbitrary, "BreathRate": BreathRate,
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

    return dict({
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
        "ReconstructedImage_arbitrary_single": ReconstructedImage_arbitrary_single,
        "ReconstructedImage_linear_single": ReconstructedImage_linear_single,
        "ReconstructedImage_arbitrary_single_HU": ReconstructedImage_arbitrary_single_HU,
        "MultiphaseRecon_linear_avg": MultiphaseRecon_linear_avg,
        "MultiphaseRecon_arbitrary_avg": MultiphaseRecon_arbitrary_avg,
        "MultiphaseReconHU_linear_avg": MultiphaseReconHU_linear_avg,
        "MultiphaseReconHU_arbitrary_avg": MultiphaseReconHU_arbitrary_avg,
        "phaseIdx": phaseIdx,
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
        "Geometry": dict(G, TBreath=TBreath, NRecons=NRecons,
                         NScansPerRecon=NScansPerRecon, NSubsets=NSubsets),
    }, **Comparison)


if __name__ == "__main__":
    main()
