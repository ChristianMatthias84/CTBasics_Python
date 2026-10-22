"""Unit tests for the Projects AH sweeps package (JP subset sweep)."""
import math

import matplotlib
import numpy as np
import pytest

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402

from CTBasics.Projects.StaticCT.ProjectsAH.sweeps import StaticCT_JP_HeartRateVsNSubsets_Sweep as SweepModule  # noqa: E402,E501
from CTBasics.Projects.StaticCT.ProjectsAH.sweeps.StaticCT_4_Cardiac_JP import StaticCT_4_Cardiac_JP  # noqa: E402,E501
from CTBasics.Projects.StaticCT.ProjectsAH.sweeps.StaticCT_JP_ParameterVsMetric_Plot import StaticCT_JP_ParameterVsMetric_Plot  # noqa: E402,E501
from CTBasics.Projects.StaticCT.ProjectsAH.sweeps.StaticCT_JP_TwoParameterSurface_Plot import StaticCT_JP_TwoParameterSurface_Plot  # noqa: E402,E501

TINY = dict(N2pi=32, DetectorElements=41, ImageSize=32)


@pytest.fixture(scope="module")
def ctbasicsconfig(tmp_path_factory):
    """Temporary CTBasicsConfig.txt whose output folder lives in a temp directory."""
    directory = tmp_path_factory.mktemp("ctbasics_config_jp_sweeps")
    output = directory / "out"
    output.mkdir()
    (directory / "Data").mkdir()
    (directory / "CTBasicsConfig.txt").write_text(
        f"#\n{directory}\n#\n{output}\n#\n\n")
    with pytest.MonkeyPatch.context() as patch:
        patch.setenv("CTBASICS_CONFIG", str(directory))
        yield output


@pytest.fixture(scope="module")
def sweepresult(ctbasicsconfig):
    table = SweepModule.StaticCT_JP_HeartRateVsNSubsets_Sweep(
        [60, 120], [4, 8], TRot=[0.2, 0.4], FixedTRotForHeartRateVsNSubsets=0.3,
        FixedNSubsetsForHeartRateVsTRot=16, FixedHeartRateForNSubsetsVsTRot=80,
        ShowPlots=False, MetricsToPlot=["PCA_Error_Arbitrary"], **TINY)
    plt.close("all")
    root = ctbasicsconfig / "cardiacPhantom" / "parameterSweeps" / "JPHeartRateVsNSubsets"
    return table, next(root.rglob("JPHeartRateVsNSubsets_AllMetrics.npz")).parent


# --- StaticCT_4_Cardiac_JP --------------------------------------------------------------

def test_cardiac_jp_reuses_linear_cache(ctbasicsconfig, capsys):
    options = dict(NRecons=2, ShowPlots=False, WriteData=False, **TINY)
    first = StaticCT_4_Cardiac_JP(70, 0.4, 8, **options)
    second = StaticCT_4_Cardiac_JP(70, 0.4, 4, **options)

    assert "Previous linear recon will be used" in capsys.readouterr().out
    assert first["Metrics"]["RMSE_Linear"] == second["Metrics"]["RMSE_Linear"]
    assert first["Metrics"]["RMSE_Arbitrary"] != second["Metrics"]["RMSE_Arbitrary"]
    assert first["Images"]["LinearHU"].shape == (2, 32 * 32)
    sampling = first["MotionSampling"]
    np.testing.assert_allclose(sampling["TReconSeconds"], [0, 60 / 70 / 2])
    assert sampling["RestDurationSeconds"] == pytest.approx(0.5 * 60 / 70)


def test_cardiac_jp_writes_run_outputs(ctbasicsconfig):
    result = StaticCT_4_Cardiac_JP(60, 0.4, 4, NRecons=1, ShowPlots=False,
                                   UseLinearCache=False, IncludeImageData=False,
                                   ReconstructionPhaseFraction=0.25, **TINY)
    plt.close("all")
    outputDirectory = ctbasicsconfig / "cardiacPhantom" / "data" / \
        "HR_60_TRot_400ms_N2pi_32_NSub_4_NRecons1_NScans1_Phase_0.25_180Recon_0deg_Cardiac"

    assert result["OutputDirectory"] == str(outputDirectory)
    assert "Images" not in result
    assert (outputDirectory / "additionalInfo.mat").is_file()
    assert any(outputDirectory.glob("ArbitraryReconHU_HR60_*.gif"))


def test_cardiac_jp_rejects_invalid_options(ctbasicsconfig):
    with pytest.raises(ValueError, match="RestFraction"):
        StaticCT_4_Cardiac_JP(RestFraction=1)
    with pytest.raises(ValueError, match="N2pi must be even"):
        StaticCT_4_Cardiac_JP(N2pi=31, ShowPlots=False, WriteData=False)


# --- sweep helpers ----------------------------------------------------------------------

def test_run_grid_contains_three_planes_once():
    grid = SweepModule.iBuildPairwiseRunGrid(
        np.array([60.0, 120.0]), np.array([4.0, 8.0]), np.array([0.2, 0.4]),
        0.4, 8.0, 60.0)

    np.testing.assert_array_equal(grid[:4], [[60, 4, 0.4], [120, 4, 0.4],
                                             [60, 8, 0.4], [120, 8, 0.4]])
    # The planes share four of their twelve points.
    assert grid.shape == (np.unique(grid, axis=0).shape[0], 3)
    assert grid.shape[0] == 7


def test_wrap_interval_to_cycle():
    assert SweepModule.iWrapIntervalToCycle(-0.1, 0.1, 1.0) == [(0.0, 0.1), (0.9, 1.0)]
    assert SweepModule.iWrapIntervalToCycle(0.2, 0.4, 1.0) == [(0.2, 0.4)]
    assert SweepModule.iWrapIntervalToCycle(0.0, 2.0, 1.0) == [(0.0, 1.0)]


def test_select_sweep_value_validates_membership():
    values = np.array([60.0, 120.0])
    assert SweepModule.iSelectSweepValue(values, math.nan, "X", 80) == 80
    assert SweepModule.iSelectSweepValue(values, 80, "X", 80) == 80
    with pytest.raises(ValueError, match="X must be one of"):
        SweepModule.iSelectSweepValue(values, 90, "X", 80)


def test_reconstruction_type_matches_like_validatestring():
    assert SweepModule.iValidateReconstructionType("arb") == "Arbitrary"
    with pytest.raises(ValueError):
        SweepModule.iValidateReconstructionType("x")


# --- complete sweep ---------------------------------------------------------------------

def test_sweep_table_and_joint_pca(sweepresult):
    table, _ = sweepresult

    assert table["HeartRateBPM"].size == 13  # 3 x 4 plane points + overview point
    for name in ("PCA_Error_Linear", "PCA_Error_Arbitrary"):
        assert np.all((table[name] >= 0) & (table[name] <= 1))
    joint = np.concatenate((table["PCA_Error_Linear"], table["PCA_Error_Arbitrary"]))
    assert joint.min() == 0 and joint.max() == 1
    np.testing.assert_allclose(table["PCA_Error_ArbitraryMinusLinear"],
                               table["PCA_Error_Arbitrary"] - table["PCA_Error_Linear"])


def test_sweep_writes_all_outputs(sweepresult):
    _, directory = sweepresult

    assert directory.name == ("HR_60-120_n2_NSub_4-8_n2_TRotms_200-400_n2_"
                              "FixedTRotms_300_FixedNSub_16_FixedHR_80")
    for name in ("AllMetrics.csv", "PhaseROIMetrics.csv", "LinearVsArbitrary.csv"):
        assert (directory / f"JPHeartRateVsNSubsets_{name}").is_file()
    # makeValidName truncates to MATLAB's namelengthmax of 63 characters.
    assert (directory / "PairwiseMatrices" /
            "RMSE_Arbitrary_HeartRateBPMVsNSubsets_FixedTRotSeconds_0_3_Matr.csv").is_file()
    assert len(list((directory / "PairwiseMatrices").glob("*.csv"))) == 27
    assert len(list((directory / "PairwisePlots").rglob("*.png"))) == 6
    assert len(list((directory / "SimulationOverview").glob("*.png"))) == 2


def test_plot_wrappers_find_latest_sweep(sweepresult):
    surface = StaticCT_JP_TwoParameterSurface_Plot(ShowFigure=False, SaveFigure=False,
                                                   CreateHeatmap=False)
    plt.close("all")
    assert surface["NumberOfSamples"].sum() == 13

    # Every fixed value of this sweep also contains the overview point, so the loaded
    # slice is ambiguous and must not be averaged.
    with pytest.raises(ValueError, match="duplicate HeartRateBPM"):
        StaticCT_JP_ParameterVsMetric_Plot(FixedParameterName="TRotSeconds",
                                           FixedParameterValue=0.3,
                                           ShowFigure=False, SaveFigure=False)
    plt.close("all")
