"""One prospective subset reconstruction study (sweep building block).

Port of Projects/Static CT/Projects AH/sweeps/StaticCT_4_Cardiac_JP.m. MATLAB is a function
``Results = StaticCT_4_Cardiac_JP(Heartrate, TRot, NSubsets, options)``; the name-value
options become keyword-only arguments with the same names and defaults. ``Results`` is a
dict with the MATLAB struct fields.

Compared with the Projects JP script the function scales the resting phase with the heart
cycle, shifts the reconstruction phases by ``ReconstructionPhaseFraction``, caches only the
finished linear/ground truth reconstructions and always compares the long sinogram
reconstruction with the average of the single scan reconstructions. The simulation blocks
are shared with ProjectsJP/StaticCT_4_Cardiac_JP.

The sweeps folder also contains StaticCT_4_Respiratory_JP.m, which is identical to the
Projects JP script and is therefore ported only once (ProjectsJP/StaticCT_4_Respiratory_JP).
"""
import math
import time
from pathlib import Path

import numpy as np

from CTBasics.MultiCore.Parallelization import Parallelization
from CTBasics.ReadAndWrite.IOHelperFunctions import IOHelperFunctions
from CTBasics.Reconstruction.Kernels import Kernels
from CTBasics.Simulation.Noise import rng
from CTBasics.Projects.StaticCT.StaticCTHelperFunctions import (StaticCTIOHelpers, StaticCTSystemHelperFunctions,
                                                                StaticCTVisualizationAndEvaluationHelpers)
from CTBasics.Projects.StaticCT.ProjectsJP.StaticCT_4_Cardiac_JP import (
    _BackprojectLongSinogram, _BackprojectSingleScans, _CardiacPhantoms, _CheckCache,
    _CloseUnlessShown, _CompareLongVsAverage, _FilterProjections, _ForwardProject,
    _Geometry, _GroundTruth, _HU, _LoadCacheIfPresent, _Names, _SamplingFigure, _SaveCache,
    _WriteReconstructions)
from CTBasics.Projects.StaticCT.ProjectsJP.test_4DCT_prosp import _PlotSingleScans


def StaticCT_4_Cardiac_JP(Heartrate=60, TRot=0.5, NSubsets=64, *, NRecons=2,
                          NScansPerRecon=1, N2pi=1024, DetectorElements=500,
                          ImageSize=512, PhotonCount=1e7, RandomSeed=1,
                          SelectedPhantom="Cardiac", RestFraction=0.5,
                          ReconstructionPhaseFraction=0, ApplyMoireFilter=False,
                          SimulateNoise=False, ShowPlots=True, WriteData=True,
                          UseLinearCache=True, IncludeImageData=True):
    """Compares interleaved subset and linear acquisition over several cardiac phases."""
    tOuter = time.perf_counter()
    rng(int(RandomSeed), "twister")
    if Heartrate <= 0 or TRot <= 0 or NSubsets <= 0:
        raise ValueError("Heartrate, TRot and NSubsets must be positive.")
    if RestFraction >= 1:
        raise ValueError(
            "RestFraction must be greater than or equal to 0 and smaller than 1.")
    if ReconstructionPhaseFraction >= 1:
        raise ValueError("ReconstructionPhaseFraction must be greater than or equal to "
                         "0 and smaller than 1.")

    Outputdirectory_base = Path(IOHelperFunctions().GlobalOutputFolderPath) \
        / "cardiacPhantom" / "data"
    Parallelization.Init()

    reconType = "180"  # 360 degree reconstruction is not working yet
    Reconkernel = Kernels.RamLak
    SimNoise = bool(SimulateNoise)
    isplot = bool(ShowPlots)
    phasesToPlot = (1, 3, 4)
    isGIF = False
    writeData = bool(WriteData)
    N_I0 = PhotonCount  # photons per detection interval

    SelectedPhantom = str(SelectedPhantom)
    if SelectedPhantom not in _CardiacPhantoms:
        raise ValueError("SelectedPhantom must be one of: " + ", ".join(_CardiacPhantoms))

    Motion = True
    Direction = 0  # 0 -> up/down, pi/2 -> left/right
    MotionType = "half"  # half sinus swing with a resting phase (systole/diastole-like)
    THeart = 60 / Heartrate
    # Scaling the rest with the cycle keeps the motion/rest ratio constant across a heart
    # rate sweep; a fixed 0.5 s rest would consume the complete cycle at 120 bpm.
    TRest = RestFraction * THeart
    CM = False
    Stent = False
    Plaque = 1
    Rotation = math.pi / 4
    PhantomFlags = [SelectedPhantom, Heartrate, Motion, Direction, MotionType, TRest,
                    CM, Stent, Plaque, Rotation]

    RM = 60
    M = DetectorElements
    if N2pi % 2 != 0:
        raise ValueError("N2pi must be even for 180-degree reconstruction.")
    Nx = ImageSize
    Ny = ImageSize
    G = _Geometry(reconType, NScansPerRecon, TRot, RM, M, N2pi, Nx, Ny)
    SimParams = [TRot, NRecons, NScansPerRecon, RM, M, N2pi, Nx, Ny, reconType, SimNoise,
                 N_I0, RandomSeed, Reconkernel, bool(ApplyMoireFilter),
                 ReconstructionPhaseFraction]

    # ReconstructionPhaseFraction is the midpoint of the first reconstruction relative to
    # one heart cycle; further phases stay equally spaced and wrap periodically.
    phaseFractions = np.mod(ReconstructionPhaseFraction + np.arange(NRecons) / NRecons, 1)
    TRecon = phaseFractions * THeart
    TLeft = TRecon - G["TReconInterval"] / 2
    TRight = TRecon + G["TReconInterval"] / 2

    ThetaArray = np.linspace(0, 2 * math.pi, N2pi + 1)[:-1]
    if NSubsets > N2pi:
        raise ValueError(f"NSubsets ({NSubsets}) must not exceed N2pi ({N2pi}).")
    ThetaArray_arbitrary = StaticCTSystemHelperFunctions.SubsetInterleaving_idx(
        N2pi, NSubsets, reconType, 0, ThetaArray)

    Names = _Names("HR", Heartrate, TRot, N2pi, NSubsets, NRecons, NScansPerRecon,
                   reconType, Direction, SelectedPhantom,
                   Phase="_Phase_%g" % ReconstructionPhaseFraction)
    linearScan_fullfile = Outputdirectory_base / Names["CacheFile"]

    # The linear/ground truth reference does not depend on NSubsets and is reused.
    linearScanData, linearRecon = _LoadCacheIfPresent(linearScan_fullfile, UseLinearCache)
    linearRecon = _CheckCache(linearRecon, linearScanData, TRecon, PhantomFlags, SimParams)
    if not linearRecon and ("Projections_linear_save" in linearScanData
                            or "Projections_GT_save" in linearScanData):
        # Older caches also hold the large filtered projections, although only the
        # finished reconstructions are reused. Drop them once to save load time and disk.
        _SaveCache(linearScan_fullfile, TRecon,
                   np.asarray(linearScanData["MultiphaseRecon_linear_save"], dtype=float),
                   np.asarray(linearScanData["MultiphaseRecon_GT_save"], dtype=float),
                   PhantomFlags, SimParams)

    Outputdirectory = Outputdirectory_base / Names["Directory"]
    if writeData:
        Outputdirectory.mkdir(parents=True, exist_ok=True)

    ProjectionOptions = dict(Heartrate=Heartrate, CM=CM, Plaque=Plaque, Stent=Stent,
                             Motion=Motion, Direction=Direction,
                             SelectedPhantom=SelectedPhantom, Rotation=Rotation,
                             MotionType=MotionType, TRest=TRest)
    Projections_arbitrary, Projections_linear, Projections_GT, center_points_1 = \
        _ForwardProject(G, NRecons, NScansPerRecon, TLeft, TRecon, ThetaArray,
                        ThetaArray_arbitrary, linearRecon, ProjectionOptions,
                        AllocateLinear=linearRecon)

    PhantomTruth, displacements, RoI_all = _GroundTruth(
        G, TRecon, SelectedPhantom,
        dict(CM=CM, Stent=Stent, Plaque=Plaque, Motion=Motion, Heartrate=Heartrate,
             Direction=Direction, Rotation=Rotation, MotionType=MotionType, TRest=TRest))
    PhantomTruthHU = _HU(PhantomTruth)

    _FilterProjections(G, Projections_arbitrary, Projections_linear, Projections_GT,
                       ApplyMoireFilter, SimNoise, N_I0, NScansPerRecon, Reconkernel,
                       IncludeLinear=linearRecon, ArbitraryFirst=True)

    # Each scan is reconstructed on its own angular interval and the scans of a phase are
    # averaged, which matches the long sinogram reconstruction closely.
    Single_arbitrary, _, Average_linear, Average_arbitrary = _BackprojectSingleScans(
        G, Projections_arbitrary, Projections_linear, reconType, NScansPerRecon,
        IncludeLinear=linearRecon)
    MultiphaseReconHU_linear_avg = _HU(Average_linear)
    MultiphaseReconHU_arbitrary_avg = _HU(Average_arbitrary)
    ReconstructedImage_arbitrary_single_HU = _HU(Single_arbitrary)

    if isplot:
        phaseIdx = min(2, NRecons)
        _PlotSingleScans(ReconstructedImage_arbitrary_single_HU,
                         MultiphaseReconHU_arbitrary_avg, phaseIdx, NScansPerRecon, Nx, Ny)

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

    _CompareLongVsAverage(MultiphaseReconHU_linear, MultiphaseReconHU_linear_avg,
                          MultiphaseReconHU_arbitrary, MultiphaseReconHU_arbitrary_avg,
                          IncludeLinear=linearRecon)

    roi_radius = 15  # RoI radius in mm (small moving structures)
    (RMSE_linear, RMSE_arbitrary, NCC_linear, NCC_arbitrary, ssim_linear, ssim_arbitrary,
     nRoI, roi_mask, center_mask) = \
        StaticCTVisualizationAndEvaluationHelpers.StaticCT_computeReconstructionMetrics(
            MultiphaseReconHU_linear, MultiphaseReconHU_arbitrary, MultiphaseReconHU_GT,
            RoI_all, Nx, Ny, G["x0"], G["y0"], G["dx"], G["dy"], roi_radius)[:9]

    # Scalar values summarize only phases and ROIs; sweep combinations are never averaged.
    Results = {
        "Parameters": {
            "HeartRateBPM": Heartrate, "TRotSeconds": TRot, "NSubsets": NSubsets,
            "NRecons": NRecons, "NScansPerRecon": NScansPerRecon, "N2pi": N2pi,
            "DetectorElements": M, "ImageSize": Nx, "PhotonCount": N_I0,
            "RandomSeed": RandomSeed, "SelectedPhantom": SelectedPhantom,
            "RestFraction": RestFraction, "RestDurationSeconds": TRest,
            "ReconstructionPhaseFraction": ReconstructionPhaseFraction},
        "Metrics": {
            "RMSE_Linear": float(np.mean(RMSE_linear)),
            "RMSE_Arbitrary": float(np.mean(RMSE_arbitrary)),
            "NCC_Linear": float(np.mean(NCC_linear)),
            "NCC_Arbitrary": float(np.mean(NCC_arbitrary)),
            "SSIM_Linear": float(np.mean(ssim_linear)),
            "SSIM_Arbitrary": float(np.mean(ssim_arbitrary))},
        "DetailedMetrics": {
            "RMSE_Linear": RMSE_linear, "RMSE_Arbitrary": RMSE_arbitrary,
            "NCC_Linear": NCC_linear, "NCC_Arbitrary": NCC_arbitrary,
            "SSIM_Linear": ssim_linear, "SSIM_Arbitrary": ssim_arbitrary,
            "TReconSeconds": TRecon, "NumberOfROIs": nRoI},
        # Timing and motion model of the run, so that sweeps can document the simulated
        # cardiac positions and acquisition windows without repeating the logic.
        "MotionSampling": {
            "TReconSeconds": TRecon, "TLeftSeconds": TLeft, "TRightSeconds": TRight,
            "HeartCycleSeconds": THeart,
            "ReconstructionIntervalSeconds": G["TReconInterval"],
            "PhaseDisplacementXYMm": displacements, "MotionEnabled": Motion,
            "MotionDirectionRadians": Direction, "MotionType": MotionType,
            "RestDurationSeconds": TRest, "MotionAmplitudeMm": 10},
        "OutputDirectory": str(Outputdirectory),
    }
    if IncludeImageData:
        Results["Images"] = {
            "PhantomHU": PhantomTruthHU, "LinearHU": MultiphaseReconHU_linear,
            "ArbitraryHU": MultiphaseReconHU_arbitrary,
            "GroundTruthHU": MultiphaseReconHU_GT, "ROIMasks": roi_mask}

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

    if writeData:
        StaticCTIOHelpers.StaticCT_appendReconstructionMetrics(
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

    if UseLinearCache and linearRecon:
        _SaveCache(linearScan_fullfile, TRecon, MultiphaseRecon_linear, MultiphaseRecon_GT,
                   PhantomFlags, SimParams)

    if isGIF:
        StaticCTIOHelpers.StaticCT_createSubsetAnimationGif(
            ThetaArray_arbitrary, reconType, NSubsets, Outputdirectory,
            MaxAngles=100, MaxFrames=100, DelayTime=0.25, Visible=False)

    _CloseUnlessShown(figx, isplot)
    print(f"JP subset simulation completed in {time.perf_counter() - tOuter:.3f} s.")
    return Results


main = StaticCT_4_Cardiac_JP


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    main()
    plt.show()
