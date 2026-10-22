"""Unit tests for the ported Projects JP and Projects CH scripts."""
import math
from pathlib import Path

import matplotlib
import numpy as np
import pytest

matplotlib.use("Agg")

from CTBasics.Projects.StaticCT.ProjectsCH import (  # noqa: E402
    StaticCT_4_ContrastmediaOrEmbo,
    StaticCT_4_RealTimeImaging_Script_CH,
)
from CTBasics.Projects.StaticCT.ProjectsJP import (  # noqa: E402
    TestCardiac4DCT,
    TestCardiacPhantom,
    test_4DCT_prosp,
)
from CTBasics.Projects.StaticCT.StaticCTHelperFunctions import (  # noqa: E402
    StaticCTIOHelpers,
    StaticCTVisualizationAndEvaluationHelpers,
)

StaticCT_appendReconstructionMetrics = StaticCTIOHelpers.StaticCT_appendReconstructionMetrics
StaticCT_computeReconstructionMetrics = (
    StaticCTVisualizationAndEvaluationHelpers.StaticCT_computeReconstructionMetrics)
StaticCT_createSubsetAnimationGif = StaticCTIOHelpers.StaticCT_createSubsetAnimationGif
StaticCT_plotReconstructionResults = (
    StaticCTVisualizationAndEvaluationHelpers.StaticCT_plotReconstructionResults)

COARSE = dict(M=97, N2pi=64, Nx=64, Ny=64)


@pytest.fixture(scope="module")
def prosp(tmp_path_factory):
    directory = tmp_path_factory.mktemp("prosp")
    return test_4DCT_prosp.main(
        Outputdirectory_base=str(directory), NSubsets=8, NRecons=3, isplot=False,
        SelectedPhantom="Respiratory", MotionType="full", TRest=0,
        UseLinearCache=False, **COARSE)


def test_compute_metrics_is_perfect_for_identical_images():
    Nx = Ny = 32
    dx = dy = 1.0
    x0 = y0 = -16.0

    rng = np.random.default_rng(0)
    Image = rng.random((2, Nx * Ny)) * 100
    RoI_all = [np.array([[0.0], [0.0]]), np.array([[2.0], [-2.0]])]

    (RMSE_linear, RMSE_arbitrary, NCC_linear, NCC_arbitrary,
     ssim_linear, ssim_arbitrary, nRoI, roi_mask, center_mask,
     RMSE_linear_bg, _, NCC_linear_bg, _, ssim_linear_bg, _, background_mask) = \
        StaticCT_computeReconstructionMetrics(Image, Image, Image, RoI_all,
                                              Nx, Ny, x0, y0, dx, dy, 6.0)

    assert nRoI == 1
    assert np.allclose(RMSE_linear, 0.0)
    assert np.allclose(RMSE_arbitrary, 0.0)
    assert np.allclose(NCC_linear, 1.0)
    assert np.allclose(ssim_linear, 1.0)
    assert np.allclose(ssim_arbitrary, 1.0)
    assert np.allclose(RMSE_linear_bg, 0.0)
    assert np.allclose(NCC_linear_bg, 1.0)
    assert np.allclose(ssim_linear_bg, 1.0)
    assert not np.any(background_mask[0] & roi_mask[(0, 0)])

    # The ROI center in pixels follows (center - origin)/spacing + 1.
    assert np.allclose(center_mask[(0, 0)], [17.0, 17.0])
    assert roi_mask[(0, 0)].shape == (Nx, Ny)


def test_compute_metrics_degrades_with_noise():
    Nx = Ny = 32
    rng = np.random.default_rng(1)
    Reference = rng.random((1, Nx * Ny)) * 100
    Noisy = Reference + rng.normal(0, 10, Reference.shape)
    RoI_all = [np.array([[0.0], [0.0]])]

    Metrics = StaticCT_computeReconstructionMetrics(
        Noisy, Reference, Reference, RoI_all, Nx, Ny, -16.0, -16.0, 1.0, 1.0, 8.0)

    RMSE_linear, RMSE_arbitrary = Metrics[0], Metrics[1]
    ssim_linear, ssim_arbitrary = Metrics[4], Metrics[5]

    assert RMSE_linear[0, 0] > RMSE_arbitrary[0, 0]
    assert ssim_linear[0, 0] < ssim_arbitrary[0, 0]
    assert ssim_arbitrary[0, 0] == pytest.approx(1.0)


def test_ssim_of_constant_images_reduces_to_the_luminance_term():
    from CTBasics.Projects.StaticCT.StaticCTHelperFunctions.\
        StaticCTVisualizationAndEvaluationHelpers import _SSIM

    A = np.arange(16.0).reshape(4, 4) % 4

    assert _SSIM(A, A, 3.0) == pytest.approx(1.0)
    assert _SSIM(A, A.T.copy(), 3.0) < 1.0

    # Replicate padding keeps constant images constant, so only the luminance term remains.
    C1 = (0.01 * 3.0) ** 2
    Constant = np.full((4, 4), 2.0)
    assert _SSIM(Constant, Constant + 1.0, 3.0) == pytest.approx((12 + C1) / (13 + C1))


def test_phantom_roi_mask_places_four_arteries():
    roiMask, centers, radii = StaticCTVisualizationAndEvaluationHelpers.DualSource_buildPhantomRoiMask(
        "Cardiac4Arteries", [3, -2], 128, 128, -64, -64, 1, 1, [60, 60])

    assert roiMask.shape == (128, 128)
    assert centers.shape == (2, 4)
    assert np.allclose(radii, 8.0)
    # MATLAB rounds the centers to whole pixels: round(25*cos(pi/4) + 3 + 64) = 85.
    assert centers[:, 0].tolist() == [85.0, 80.0]
    assert roiMask[79, 84]


def test_phantom_roi_mask_falls_back_to_a_legacy_circle():
    roiMask, centers, radii = StaticCTVisualizationAndEvaluationHelpers.DualSource_buildPhantomRoiMask(
        "Unknown", [0, 0], 32, 32, -16, -16, 1, 1, [10, 40])

    assert centers.ravel().tolist() == [10, 32]
    assert radii.tolist() == [80.0]
    assert roiMask.all()


def test_export_figure_writes_the_file(tmp_path):
    import matplotlib.pyplot as plt

    figure = plt.figure()
    target = tmp_path / "figure.png"

    StaticCTVisualizationAndEvaluationHelpers.StaticCT_exportFigure(figure, str(target), 50)
    plt.close(figure)

    assert target.stat().st_size > 0

def test_append_metrics_pads_and_averages(tmp_path):
    RMSE_linear = np.array([[10.0, 20.0], [30.0, np.nan]])
    RMSE_arbitrary = np.array([[5.0, 10.0], [15.0, np.nan]])
    Ones = np.ones((2, 2))

    summary = StaticCT_appendReconstructionMetrics(
        str(tmp_path), "WaterWith4Arteries", [0.0, 0.5],
        RMSE_linear, RMSE_arbitrary, Ones, Ones, Ones, Ones,
        RoIAll=[np.array([[1.0, 2.0], [3.0, 4.0]])], RoIRadiusMm=15.0, Tag="unit")

    assert summary["NRecons"] == 2
    assert summary["NRoI"] == 2
    # NaN entries are ignored, so ROI 2 averages over a single phase.
    assert summary["RMSE_linear_mean_roi"][1] == pytest.approx(20.0)
    assert summary["RMSE_global_improvement"] == pytest.approx(20.0 - 10.0)
    assert summary["NCC_global_improvement"] == pytest.approx(0.0)

    content = Path(summary["ResultFile"]).read_text()
    assert "ROI 1 (Stent)" in content
    assert "Tag: unit" in content


def test_append_metrics_rejects_empty_input(tmp_path):
    with pytest.raises(ValueError):
        StaticCT_appendReconstructionMetrics(
            str(tmp_path), "Respiratory", None, [], [], [], [], [], [])


def test_create_subset_animation_gif(tmp_path):
    path = StaticCT_createSubsetAnimationGif(
        np.linspace(0, math.pi, 20), "180", 4, str(tmp_path),
        MaxAngles=20, MaxFrames=5, DelayTime=0.25)

    assert Path(path).name == "winkel_animation.gif"
    assert Path(path).stat().st_size > 0


def test_create_subset_animation_gif_rejects_empty_angles(tmp_path):
    with pytest.raises(ValueError):
        StaticCT_createSubsetAnimationGif([], "180", 4, str(tmp_path))


def test_plot_reconstruction_results_skips_invalid_phases():
    Empty = np.zeros((2, 16))

    with pytest.warns(UserWarning):
        figures = StaticCT_plotReconstructionResults(
            Empty, Empty, Empty, Empty, Empty, Empty, Empty, Empty, Empty,
            Empty, Empty, {}, {}, 4, 4, 0, 0, 500, 4, phasesToPlot=[99])

    assert figures == []


def test_test_cardiac_phantom_renders_frames():
    # 0.75/0.25 is exact in binary floating point, so NT is unambiguously 3.
    result = TestCardiacPhantom.main(Nx=64, Ny=64, dx=4, dy=4, x0=-128, y0=-128,
                                     dT=0.25, duration=0.75)

    assert result["NT"] == 3
    assert result["Frames"].shape == (3, 64, 64)
    assert result["Frames"].dtype == np.uint8
    assert result["xpoints"].size == 3


def test_test_cardiac_phantom_truncates_inexact_frame_counts():
    # MATLAB evaluates 0.3/0.1 as 2.9999999999999996, so "for nT = 1:NT" runs twice.
    result = TestCardiacPhantom.main(Nx=32, Ny=32, dx=8, dy=8, x0=-128, y0=-128,
                                     dT=0.1, duration=0.3)

    assert result["NT"] == 2


def test_test_cardiac_4dct_subset_order_is_a_permutation(tmp_path):
    result = TestCardiac4DCT.main(Outputdirectory=str(tmp_path), NSubsets=8, Seed=3,
                                  **COARSE)
    geometry = result["Geometry"]

    assert geometry["Nphases"] == geometry["N"] // geometry["Npi"]
    # The interleaved order must reuse every angle of the half turn exactly once, plus
    # the same order shifted by pi.
    Sub = result["ThetaArray_sub"]
    assert Sub.size == geometry["N2pi"]
    assert np.unique(Sub).size == geometry["N2pi"]
    assert np.allclose(np.sort(Sub[:geometry["Npi"]] + math.pi),
                       np.sort(Sub[geometry["Npi"]:]))


def test_test_cardiac_4dct_rejects_other_recon_types(tmp_path):
    with pytest.raises(ValueError):
        TestCardiac4DCT.main(Outputdirectory=str(tmp_path), reconType="180")


def test_prosp_geometry_and_outputs(prosp):
    geometry = prosp["Geometry"]

    assert geometry["RM"] == 150  # Respiratory phantom enlarges the FOM
    assert geometry["N"] == geometry["N2pi"] // 2  # 180 degree reconstruction
    assert geometry["thetaN"] == pytest.approx(math.pi)
    assert prosp["nRoI"] == 2
    assert prosp["TRecon"].size == 3
    assert Path(prosp["Outputdirectory"], "reconstruction_metrics.txt").is_file()
    assert Path(prosp["Outputdirectory"], "additionalInfo.mat").is_file()


def test_prosp_linear_order_beats_interleaved_order(prosp):
    # A continuous half turn samples one motion state; interleaving spreads the views
    # over the whole window and therefore blurs more.
    assert np.nanmean(prosp["RMSE_linear"]) < np.nanmean(prosp["RMSE_arbitrary"])
    assert np.nanmean(prosp["NCC_linear"]) > np.nanmean(prosp["NCC_arbitrary"])
    assert np.nanmean(prosp["ssim_linear"]) > np.nanmean(prosp["ssim_arbitrary"])


def test_prosp_rejects_unknown_recon_type(tmp_path):
    with pytest.raises(ValueError):
        test_4DCT_prosp.main(Outputdirectory_base=str(tmp_path), reconType="90")


def test_prosp_linear_cache_round_trip(tmp_path):
    arguments = dict(Outputdirectory_base=str(tmp_path), NSubsets=4, NRecons=2,
                     isplot=False, writeData=False, ForceLinearRecon=False, **COARSE)

    first = test_4DCT_prosp.main(**arguments)
    assert first["linearRecon"] is True
    assert Path(first["linearScan_fullfile"]).is_file()

    second = test_4DCT_prosp.main(**arguments)
    assert second["linearRecon"] is False
    assert np.allclose(first["MultiphaseRecon_linear"],
                       second["MultiphaseRecon_linear"])
    assert np.allclose(first["MultiphaseRecon_linear_avg"],
                       second["MultiphaseRecon_linear_avg"])

    # The cache file name ignores TRest and N_I0, but PhantomFlags/SimParams do not.
    changedPhantom = test_4DCT_prosp.main(**{**arguments, "TRest": 0.25})
    assert changedPhantom["linearRecon"] is True
    changedSim = test_4DCT_prosp.main(**{**arguments, "TRest": 0.25, "N_I0": 1e6})
    assert changedSim["linearRecon"] is True


def test_prosp_force_linear_recon_ignores_cache(tmp_path):
    arguments = dict(Outputdirectory_base=str(tmp_path), NSubsets=4, NRecons=2,
                     isplot=False, writeData=False, **COARSE)

    test_4DCT_prosp.main(**arguments)
    # MATLAB currently sets linearRecon = true unconditionally.
    assert test_4DCT_prosp.main(**arguments)["linearRecon"] is True


def test_prosp_cache_without_parameters_is_rejected():
    TRecon = np.array([0.0, 0.5])
    PhantomFlags = ["Cardiac", 60, True, 0, "half", 0.5, False, False, 1, math.pi / 4]
    Saved = {"TRecon_save": TRecon.reshape(1, -1),
             "PhantomFlags": test_4DCT_prosp._ToCell(PhantomFlags),
             "SimParams": test_4DCT_prosp._ToCell([0.5, "180", False])}

    assert test_4DCT_prosp._CacheMatches(Saved, TRecon, PhantomFlags,
                                         [0.5, "180", False])
    assert not test_4DCT_prosp._CacheMatches(Saved, TRecon, PhantomFlags,
                                             [0.5, "360", False])
    assert not test_4DCT_prosp._CacheMatches(Saved, TRecon, PhantomFlags,
                                             [0.5, "180", True])
    # Caches written before PhantomFlags/SimParams existed are never reused.
    assert not test_4DCT_prosp._CacheMatches({"TRecon_save": TRecon}, TRecon,
                                             PhantomFlags, [0.5, "180", False])


def test_prosp_angle_arrays(prosp):
    expected = np.linspace(0, 2 * math.pi, 64 + 1)[:-1]

    assert np.allclose(prosp["ThetaArray"], expected)
    # MATLAB hardcodes 2 subsets, so NSubsets=8 does not change the arbitrary order.
    arbitrary = np.asarray(prosp["ThetaArray_arbitrary"]).ravel()
    from CTBasics.Projects.StaticCT.StaticCTHelperFunctions import (
        StaticCTSystemHelperFunctions)
    assert np.allclose(arbitrary,
                       StaticCTSystemHelperFunctions.SubsetInterleaving(64, 2, "180", 0))
    assert np.unique(arbitrary).size == 64
    assert not np.allclose(arbitrary, expected)


def test_prosp_average_of_single_scans_matches_long_sinogram(tmp_path):
    result = test_4DCT_prosp.main(
        Outputdirectory_base=str(tmp_path), NSubsets=4, NRecons=2, NScansPerRecon=2,
        isplot=False, writeData=False, UseLinearCache=False, **COARSE)

    assert result["ReconstructedImage_arbitrary_single"].shape == (2, 2, 64 * 64)
    assert result["phaseIdx"] == 2
    # The second half turn is reconstructed from theta0 = pi, so the per-scan average
    # equals the long-sinogram reconstruction up to rounding.
    assert result["relDiff_linear"] < 1e-10
    assert result["relDiff_arbitrary"] < 1e-10
    assert np.allclose(result["MultiphaseReconHU_arbitrary_avg"],
                       result["MultiphaseReconHU_arbitrary"], atol=1e-8)
    assert result["N_I0_per_scan"] is None


def test_prosp_noise_budget_is_split_over_scans(tmp_path, monkeypatch):
    from CTBasics.Simulation.Noise import Noise

    calls = []

    def RecordNoise(Projections, N, I0):
        calls.append((np.asarray(Projections).size, N, I0))
        return np.asarray(Projections, dtype=float).ravel()

    monkeypatch.setattr(Noise, "ApplyPoissonNoise", staticmethod(RecordNoise))

    result = test_4DCT_prosp.main(
        Outputdirectory_base=str(tmp_path), NSubsets=4, NRecons=2, NScansPerRecon=4,
        SimNoise=True, N_I0=1000, isplot=False, writeData=False, UseLinearCache=False,
        **COARSE)

    assert result["N_I0_per_scan"] == pytest.approx(250)
    # Linear and arbitrary projections only (no noise on the ground truth).
    assert len(calls) == 2 * 2 * 4
    rowSize = 97 * 32
    assert all(size == rowSize and N == rowSize and I0 == pytest.approx(250)
               for size, N, I0 in calls)


def test_real_time_imaging_adds_poisson_noise(tmp_path):
    result = StaticCT_4_RealTimeImaging_Script_CH.main(
        Outputdirectory_base=str(tmp_path), NSubsets=8, NRecons=2, NScansPerRecon=3,
        isplot=False, UseLinearCache=False, **COARSE)
    geometry = result["Geometry"]

    assert geometry["NScansPerRecon"] == 3
    assert geometry["RM"] == 60  # WaterWith4Arteries keeps the small FOM
    assert result["nRoI"] == 4
    # Noise raises the error level far above the noise free JP configuration.
    assert np.nanmean(result["RMSE_linear"]) > 20
    # The CH script keeps the original JP angle arrays.
    assert result["ThetaArray"][-1] == pytest.approx(2 * math.pi - geometry["dtheta"])


def test_contrastmedia_angle_order_and_n2pi_tag(tmp_path):
    result = StaticCT_4_ContrastmediaOrEmbo.main(
        OutputBaseDir=str(tmp_path), NSubsets=8, NRecons=1, M=97, N2pi=64,
        Nx=32, Ny=32, ShowFinalRecons=False, WriteData=False)
    geometry = result["Geometry"]

    assert "_N2pi64_NSub8_" in Path(result["OutputDir"]).name
    assert geometry["theta0"] == pytest.approx(-geometry["dtheta"])
    # Index-domain interleaving reorders the linear angles without repeating any.
    assert np.allclose(np.sort(result["ThetaArrayArbitrary"]),
                       np.sort(result["ThetaArrayLinear"]))
    assert not np.allclose(result["ThetaArrayArbitrary"], result["ThetaArrayLinear"])


def test_contrastmedia_signal_follows_motion_profile(tmp_path):
    result = StaticCT_4_ContrastmediaOrEmbo.main(
        OutputBaseDir=str(tmp_path), NSubsets=8, NRecons=3, M=97, N2pi=64,
        Nx=64, Ny=64, ShowFinalRecons=False, SaveReconGifs=False)

    discrete = result["iodineTemporalDiscrete"]
    # IodinePeakFactor 3 and a full sine over 1 s give 0, +3*sin(120deg), -3*sin(120deg).
    assert discrete[0] == pytest.approx(0.0, abs=1e-12)
    assert discrete[1] == pytest.approx(3.0 * math.sin(2 * math.pi / 3))
    assert discrete[2] == pytest.approx(-3.0 * math.sin(2 * math.pi / 3))
    assert result["iodineTemporalSignal"].max() == pytest.approx(3.0, abs=1e-3)

    # The iodine insert is far brighter than water in every phase.
    assert result["ReconHU_GT"].max() > 2000
    assert Path(result["OutputDir"], "TemporalIodine_MultiphaseResults.mat").is_file()


def test_contrastmedia_rejects_unknown_recon_type(tmp_path):
    with pytest.raises(ValueError):
        StaticCT_4_ContrastmediaOrEmbo.main(OutputBaseDir=str(tmp_path),
                                            ReconType="90")
