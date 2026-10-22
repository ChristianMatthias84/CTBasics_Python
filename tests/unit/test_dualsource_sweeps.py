"""Unit tests for the DualSource Experiment_to_dual script and the sweeps package."""
import csv
import json
import math
from pathlib import Path

import matplotlib
import numpy as np
import pytest

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402

from CTBasics.Projects.StaticCT.ProjectsAH.DualSource import (  # noqa: E402
    DualSource_CardiacMotion4DCT_SegmentedSingleSource,
    DualSource_CardiacMotion4DCT_SegmentedSingleSource_Experiment_to_dual,
)
from CTBasics.Projects.StaticCT.ProjectsAH.DualSource.sweeps import (  # noqa: E402
    DualSource_HeartRateVsTRot_Sweep as SweepModule,
    _SweepHelpers,
)
from CTBasics.Projects.StaticCT.ProjectsAH.DualSource.sweeps.DualSource_HeartRateVsTRot_Sweep import (  # noqa: E402,E501
    DualSource_HeartRateVsTRot_Sweep,
    metricNames,
)
from CTBasics.Projects.StaticCT.ProjectsAH.DualSource.sweeps.DualSource_ParameterVsMetric_Plot import (  # noqa: E402,E501
    DualSource_ParameterVsMetric_Plot,
)
from CTBasics.Projects.StaticCT.ProjectsAH.DualSource.sweeps.DualSource_TwoParameterSurface_Plot import (  # noqa: E402,E501
    DualSource_TwoParameterSurface_Plot,
)

COARSE = dict(M=97, Npar=48, Nx=64, Ny=64)
SWEEP_COARSE = dict(Npar=48, DetectorElements=97, ImageSize=64, SimulateNoise=False)
SWEEP_COLUMNS = ["HeartRateBPM", "TRotSeconds", "Nsources", "Npar", "DetectorElements",
                 "ImageSize", "PhotonCount", "RandomSeed", "SelectedPhantom",
                 "SimulateNoise", "ApplyMoireFilter"] + metricNames + ["PCA_Error_DualSource"]
SWEEP_CONFIG = "Ph_WaterWith4Arteries_NS_2_Npar_48_M_97_Nx_64_I0_225000_Noise_0_Moire_0_Seed_1"


@pytest.fixture(scope="module")
def ctbasicsconfig(tmp_path_factory):
    """Temporary CTBasicsConfig.txt whose output folder lives in a temp directory."""
    directory = tmp_path_factory.mktemp("ctbasics_config_sweeps")
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
    """Two point TRot sweep at one heart rate (exercises the 2D plot branch)."""
    table = DualSource_HeartRateVsTRot_Sweep(
        [60], [0.3, 0.5], ShowPlots=False,
        MetricsToPlot=["PCA_Error_DualSource", "RMSE_DualSource_ROI"], **SWEEP_COARSE)
    directory = (ctbasicsconfig / "cardiacPhantom" / "parameterSweeps" /
                 "DualSourceHeartRateVsTRot" / SWEEP_CONFIG / "HR_60-60_n1_TRot_300-500ms_n2")
    return table, directory


def _SyntheticTable():
    """2 x 3 grid (TRot x HR) with a known metric, ordered like the sweep (HR fastest)."""
    heartRates = np.array([60.0, 70.0, 80.0])
    rotationTimes = np.array([0.3, 0.5])
    HR = np.tile(heartRates, rotationTimes.size)
    TRot = np.repeat(rotationTimes, heartRates.size)
    n = HR.size
    return {
        "HeartRateBPM": HR, "TRotSeconds": TRot,
        "Nsources": np.full(n, 2.0), "Npar": np.full(n, 48.0),
        "SelectedPhantom": np.full(n, "WaterWith4Arteries"),
        "SimulateNoise": np.full(n, False),
        "PCA_Error_DualSource": HR / 100 + TRot,
    }


# --- Experiment_to_dual -----------------------------------------------------------------

def test_experiment_to_dual_synchronizes_segments(tmp_path):
    result = DualSource_CardiacMotion4DCT_SegmentedSingleSource_Experiment_to_dual.main(
        Outputdirectory=str(tmp_path), Nsources=3, SimNoise=False,
        SelectedPhantom="WaterWith4Arteries", **COARSE)
    geometry = result["Geometry"]
    outputDirectory = Path(result["Outputdirectory"])

    assert outputDirectory.name.startswith("SegmentedSingle_")
    assert "_NSeg_3_Ph_" in outputDirectory.name
    assert (outputDirectory / "simulation_parameters.txt").is_file()
    assert (outputDirectory / "reconstruction_metrics.txt").is_file()
    assert geometry["NprojMulti"] == 3 * geometry["Npi"]

    # All segments share the time window around Shift_GT: odd segments are identical.
    segments = result["Projections_multi"].reshape(3, -1)
    np.testing.assert_allclose(segments[0], segments[2], rtol=0, atol=1e-12)

    sequential = DualSource_CardiacMotion4DCT_SegmentedSingleSource.main(
        Outputdirectory=str(tmp_path / "sequential"), Nsources=3, SimNoise=False,
        SelectedPhantom="WaterWith4Arteries", **COARSE)
    sequentialSegments = sequential["Projections_multi"].reshape(3, -1)
    assert not np.allclose(sequentialSegments[0], sequentialSegments[2])
    plt.close("all")


def test_experiment_to_dual_defaults_to_twenty_segments():
    import inspect
    signature = inspect.signature(
        DualSource_CardiacMotion4DCT_SegmentedSingleSource_Experiment_to_dual.main)
    assert signature.parameters["Nsources"].default == 20


# --- helpers ----------------------------------------------------------------------------

def test_make_valid_name_matches_matlab():
    assert _SweepHelpers.makeValidName("TRot_0.1_s") == "TRot_0_1_s"
    assert _SweepHelpers.makeValidName("Water with arteries") == "WaterWithArteries"
    assert _SweepHelpers.makeValidName("1abc") == "x1abc"
    assert _SweepHelpers.makeValidName("Ph_a_I0_1e+06") == "Ph_a_I0_1e_06"


def test_matlab_colon_default_rotation_times():
    values = _SweepHelpers.MatlabColon(0.1, 0.05, 0.5)
    assert values.size == 9
    assert values[0] == 0.1 and values[-1] == 0.5
    assert ["%g" % value for value in values] == [
        "0.1", "0.15", "0.2", "0.25", "0.3", "0.35", "0.4", "0.45", "0.5"]
    np.testing.assert_array_equal(_SweepHelpers.MatlabColon(50, 5, 90), np.arange(50, 95, 5))


def test_pca_sign_convention_and_explained():
    rng = np.random.default_rng(0)
    X = rng.standard_normal((6, 3))
    coeff, score, explained = SweepModule.pca(X)
    Xc = X - X.mean(axis=0)

    np.testing.assert_allclose(score, Xc @ coeff, atol=1e-12)
    largest = np.argmax(np.abs(coeff), axis=0)
    assert np.all(coeff[largest, np.arange(coeff.shape[1])] > 0)
    assert explained.sum() == pytest.approx(100)
    assert np.all(np.diff(explained) <= 0)


def test_table_csv_round_trip(tmp_path):
    table = _SyntheticTable()
    table["PCA_Error_DualSource"][1] = np.nan
    path = tmp_path / "table.csv"
    _SweepHelpers.writetable(table, path)

    lines = path.read_text().splitlines()
    assert lines[0] == ",".join(table)
    assert lines[1].endswith(",WaterWith4Arteries,0,0.9")
    loaded = _SweepHelpers.readtable(path)
    assert list(loaded) == list(table)
    assert loaded["SelectedPhantom"].dtype.kind == "U"
    np.testing.assert_array_equal(loaded["SimulateNoise"], np.zeros(6))
    np.testing.assert_allclose(loaded["PCA_Error_DualSource"],
                               table["PCA_Error_DualSource"], equal_nan=True)


# --- sweep ------------------------------------------------------------------------------

def test_sweep_writes_tables(sweepresult):
    table, directory = sweepresult

    assert list(table) == SWEEP_COLUMNS
    np.testing.assert_array_equal(table["HeartRateBPM"], [60, 60])
    np.testing.assert_array_equal(table["TRotSeconds"], [0.3, 0.5])
    assert sorted(table["PCA_Error_DualSource"]) == [0.0, 1.0]

    csvPath = directory / "DualSourceHeartRateVsTRot_AllMetrics.csv"
    with open(csvPath, newline="") as handle:
        rows = list(csv.reader(handle))
    assert rows[0] == SWEEP_COLUMNS
    assert len(rows) == 3

    with np.load(directory / "DualSourceHeartRateVsTRot_AllMetrics.npz") as data:
        assert list(data["sweepTable"].dtype.names) == SWEEP_COLUMNS
        np.testing.assert_array_equal(data["rotationTimes"], [0.3, 0.5])
        pcaInfo = json.loads(str(data["pcaInfo"]))
        options = json.loads(str(data["options"]))
    assert len(pcaInfo["Coefficients"]) == 3
    assert options["Npar"] == 48

    for metricName in metricNames + ["PCA_Error_DualSource"]:
        with open(directory / f"{metricName}_Matrix.csv", newline="") as handle:
            matrixRows = list(csv.reader(handle))
        assert matrixRows[0] == ["HeartRateBPM", "TRot_0_3_s", "TRot_0_5_s"]
        assert len(matrixRows) == 2
    rmseMatrix = _SweepHelpers.readtable(directory / "RMSE_DualSource_Matrix.csv")
    assert rmseMatrix["TRot_0_5_s"][0] == pytest.approx(table["RMSE_DualSource"][1],
                                                        rel=1e-14)

    # makeValidName truncates to namelengthmax (63) like MATLAB.
    configuration = ("Ph_WaterWith4Arteries_HeartRateBPM_60_NS_2_Npar_48_M_97_Nx_64_I0_225000"
                     "_Noise_0_Seed_1")[:63]
    plotDirectory = directory / "Plots2D" / "PCA_Error_DualSource_vs_TRotSeconds" / configuration
    stem = "PCA_Error_DualSource_vs_TRotSeconds_at_HeartRateBPM_60"
    assert (plotDirectory / f"{stem}.png").is_file()
    assert (plotDirectory / f"{stem}_Data.csv").is_file()
    assert (directory / "Plots2D" / "RMSE_DualSource_ROI_vs_TRotSeconds").is_dir()


def test_sweep_rejects_single_run_and_unknown_metric(ctbasicsconfig):
    with pytest.raises(ValueError, match="At least two"):
        DualSource_HeartRateVsTRot_Sweep([60], [0.5], ShowPlots=False, **SWEEP_COARSE)
    with pytest.raises(ValueError, match="positive"):
        DualSource_HeartRateVsTRot_Sweep([60, -1], [0.5], ShowPlots=False, **SWEEP_COARSE)


# --- ParameterVsMetric ------------------------------------------------------------------

def test_parameter_vs_metric_from_csv(sweepresult, tmp_path):
    table, directory = sweepresult
    plotTable = DualSource_ParameterVsMetric_Plot(
        str(directory / "DualSourceHeartRateVsTRot_AllMetrics.csv"), "TRotSeconds",
        "RMSE_DualSource", FixedParameterValue=60, ShowFigure=False,
        OutputDirectory=str(tmp_path))

    np.testing.assert_array_equal(plotTable["TRotSeconds"], [0.3, 0.5])
    np.testing.assert_allclose(plotTable["RMSE_DualSource"], table["RMSE_DualSource"],
                               rtol=1e-14)
    files = sorted(path.name for path in (tmp_path / "Plots2D").rglob("*.*"))
    assert files == ["RMSE_DualSource_vs_TRotSeconds_at_HeartRateBPM_60.png",
                     "RMSE_DualSource_vs_TRotSeconds_at_HeartRateBPM_60_Data.csv"]
    assert plt.get_fignums() == []


def test_parameter_vs_metric_finds_latest_npz(sweepresult):
    _, directory = sweepresult
    plotTable = DualSource_ParameterVsMetric_Plot(
        None, "TRotSeconds", "SSIM_DualSource", FixedParameterValue=60,
        ShowFigure=False, SaveFigure=False)

    assert plotTable["SelectedPhantom"][0] == "WaterWith4Arteries"
    # Without OutputDirectory the plots land next to the loaded sweep result.
    stem = "SSIM_DualSource_vs_TRotSeconds_at_HeartRateBPM_60"
    found = list((directory / "Plots2D").rglob(f"{stem}_Data.csv"))
    assert len(found) == 1
    assert not list((directory / "Plots2D").rglob(f"{stem}.png"))


def test_parameter_vs_metric_errors(sweepresult):
    table, _ = sweepresult
    with pytest.raises(ValueError, match="No data for HeartRateBPM = 90. Available values: 60"):
        DualSource_ParameterVsMetric_Plot(table, ShowFigure=False, SaveFigure=False)
    with pytest.raises(ValueError, match="Unknown sweep table column"):
        DualSource_ParameterVsMetric_Plot(table, "Nope", ShowFigure=False)
    with pytest.raises(ValueError, match="must be different"):
        DualSource_ParameterVsMetric_Plot(table, "HeartRateBPM", ShowFigure=False)
    with pytest.raises(ValueError, match="numeric"):
        DualSource_ParameterVsMetric_Plot(table, "TRotSeconds", "SelectedPhantom",
                                          FixedParameterValue=60, ShowFigure=False)
    with pytest.raises(ValueError, match="duplicate"):
        DualSource_ParameterVsMetric_Plot(_SyntheticTable(), "TRotSeconds",
                                          "PCA_Error_DualSource",
                                          FixedParameterName="Npar",
                                          FixedParameterValue=48, ShowFigure=False)


# --- TwoParameterSurface ----------------------------------------------------------------

def test_two_parameter_surface_from_table(ctbasicsconfig, tmp_path):
    table = _SyntheticTable()
    surfaceTable = DualSource_TwoParameterSurface_Plot(
        table, ShowFigure=False, OutputDirectory=str(tmp_path))

    assert list(surfaceTable) == ["TRotSeconds", "HeartRateBPM", "PCA_Error_DualSource",
                                  "NumberOfSamples"]
    # Column-major ravel of the (HR x TRot) grid: heart rate varies fastest.
    np.testing.assert_array_equal(surfaceTable["TRotSeconds"], [0.3] * 3 + [0.5] * 3)
    np.testing.assert_array_equal(surfaceTable["HeartRateBPM"], [60, 70, 80] * 2)
    np.testing.assert_allclose(surfaceTable["PCA_Error_DualSource"],
                               table["PCA_Error_DualSource"])
    np.testing.assert_array_equal(surfaceTable["NumberOfSamples"], np.ones(6))

    configuration = ("Ph_WaterWith4Arteries_TRotSeconds_0_3_0_5_n2_HeartRateBPM_60_80_n3"
                     "_NS_2_Npar_48_Noise_0")[:63]
    diagram = "PCA_Error_DualSource_vs_TRotSeconds_and_HeartRateBPM"
    stem = "TRotSeconds_HeartRateBPM_vs_PCA_Error_DualSource"
    for folder, suffix in (("Plots3D", "Surface"), ("PlotsHeatmap", "Heatmap")):
        outputDirectory = tmp_path / folder / diagram / configuration
        assert (outputDirectory / f"{stem}_{suffix}.png").is_file()
        assert (outputDirectory / f"{stem}_{suffix}_Data.csv").is_file()
    assert plt.get_fignums() == []


def test_two_parameter_surface_averages_duplicates_and_interpolates(tmp_path):
    table = _SyntheticTable()
    # Duplicate one combination and drop another one.
    keep = np.array([True, True, True, True, False, True])
    table = {name: column[keep] for name, column in table.items()}
    table = {name: np.concatenate((column, column[:1])) for name, column in table.items()}
    table["PCA_Error_DualSource"][-1] += 0.2

    surfaceTable = DualSource_TwoParameterSurface_Plot(
        table, ShowFigure=False, SaveFigure=False, OutputDirectory=str(tmp_path))
    assert surfaceTable["NumberOfSamples"][0] == 2
    assert surfaceTable["PCA_Error_DualSource"][0] == pytest.approx(0.9 + 0.1)
    assert math.isnan(surfaceTable["PCA_Error_DualSource"][4])
    assert surfaceTable["NumberOfSamples"][4] == 0

    interpolated = DualSource_TwoParameterSurface_Plot(
        table, ShowFigure=False, SaveFigure=False, InterpolateMissing=True,
        CreateHeatmap=False, OutputDirectory=str(tmp_path))
    assert np.isfinite(interpolated["PCA_Error_DualSource"][4])


def test_two_parameter_surface_finds_two_dimensional_npz(ctbasicsconfig):
    # A dedicated 2 x 2 sweep result; the 1D sweep of the other tests must be skipped.
    table = _SyntheticTable()
    directory = (ctbasicsconfig / "cardiacPhantom" / "parameterSweeps" /
                 "DualSourceHeartRateVsTRot" / "synthetic2D")
    directory.mkdir(parents=True, exist_ok=True)
    np.savez(directory / "DualSourceHeartRateVsTRot_AllMetrics.npz",
             sweepTable=_SweepHelpers.TableToRecords(table))

    surfaceTable = DualSource_TwoParameterSurface_Plot(
        None, ShowFigure=False, SaveFigure=False, CreateHeatmap=False)
    assert surfaceTable["HeartRateBPM"].size == 6
    assert list((directory / "Plots3D").rglob("*_Surface_Data.csv"))


def test_two_parameter_surface_errors(sweepresult, tmp_path):
    table, _ = sweepresult
    with pytest.raises(ValueError, match="two distinct values"):
        DualSource_TwoParameterSurface_Plot(table, ShowFigure=False, SaveFigure=False)
    matPath = tmp_path / "DualSourceHeartRateVsTRot_AllMetrics.mat"
    matPath.write_text("")
    with pytest.raises(ValueError, match="Only CSV and NPZ"):
        DualSource_TwoParameterSurface_Plot(str(matPath), ShowFigure=False)
    with pytest.raises(ValueError, match="Unknown sweep table column"):
        DualSource_TwoParameterSurface_Plot(table, "TRotSeconds", "Nope",
                                            ShowFigure=False)
