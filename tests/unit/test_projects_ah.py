"""Unit tests for the ported Projects AH scripts (DualSource and Noise)."""
import math
from pathlib import Path

import matplotlib
import numpy as np
import pytest

matplotlib.use("Agg")

from CTBasics.Projects.StaticCT.ProjectsAH.DualSource import (  # noqa: E402
    CTRecon101DualSource,
    DualSource_CardiacMotion4DCT_Main,
    DualSource_CardiacMotion4DCT_NsourcesSweep_Experiment,
    DualSource_CardiacMotion4DCT_SegmentedSingleSource,
)
from CTBasics.Projects.StaticCT.StaticCTHelperFunctions import StaticCTIOHelpers  # noqa: E402
from CTBasics.Projects.StaticCT.ProjectsAH.Noise import (  # noqa: E402
    MyNoiseTestingCTRecon103,
    TestNoiseCardiac4DCT,
    WaterNoiseCalibration,
)
from CTBasics.Reconstruction.Constants import Constants  # noqa: E402

DualSource_appendReconstructionMetrics = StaticCTIOHelpers.DualSource_appendReconstructionMetrics
DualSource_buildOutputNaming = StaticCTIOHelpers.DualSource_buildOutputNaming
DualSource_showAndSaveImage = StaticCTIOHelpers.DualSource_showAndSaveImage
DualSource_writeSimulationParameters = StaticCTIOHelpers.DualSource_writeSimulationParameters

COARSE = dict(M=97, Npar=48, Nx=64, Ny=64)


MAIN_COARSE = dict(Npar=48, DetectorElements=97, ImageSize=64, ShowTimingPlot=False)


@pytest.fixture(scope="module")
def ctbasicsconfig(tmp_path_factory):
    """Temporary CTBasicsConfig.txt whose output folder lives in a temp directory."""
    directory = tmp_path_factory.mktemp("ctbasics_config")
    output = directory / "out"
    output.mkdir()
    (directory / "Data").mkdir()
    (directory / "CTBasicsConfig.txt").write_text(
        f"#\n{directory}\n#\n{output}\n#\n\n")
    with pytest.MonkeyPatch.context() as patch:
        patch.setenv("CTBASICS_CONFIG", str(directory))
        yield output


@pytest.fixture(scope="module")
def dualsourcemain(ctbasicsconfig):
    return DualSource_CardiacMotion4DCT_Main.main(
        Nsources=2, SelectedPhantom="LinePattern", SimulateNoise=False, **MAIN_COARSE)


def test_build_output_naming_line_pattern():
    naming = DualSource_buildOutputNaming("LinePattern", 1, False, False)

    assert naming["RootFolderLabel"] == "linePattern"
    assert naming["PhantomCode"] == "P1C0S0"
    assert naming["PhantomTag"] == "LinePattern_P1C0S0"


def test_build_output_naming_sanitizes_and_encodes_flags():
    naming = DualSource_buildOutputNaming("Water With 4 Arteries!", 3, True, True)

    # Everything outside [A-Za-z0-9_-] is stripped, so the label stays filename safe.
    assert naming["PhantomLabel"] == "WaterWith4Arteries"
    assert naming["RootFolderLabel"] == "WaterWith4Arteries"
    assert naming["PhantomCode"] == "P3C1S1"


def test_write_simulation_parameters_contains_all_values(tmp_path):
    path = DualSource_writeSimulationParameters(
        str(tmp_path), 2, 0.5, 60, 0.001, 60, 500, 512, 1024, 2 * math.pi / 1024,
        512, 512, 0.234, 0.234, 0.125, 0.0, 0.0, 0.0, "full", 0.0,
        False, True, 1e7)

    content = Path(path).read_text()
    assert "N Sources: 2" in content
    assert "0.5000 s" in content
    # MATLAB writes CRLF for the body lines; the port must not normalize them.
    with open(path, newline="") as handle:
        assert "\r\n" in handle.read()


def test_append_reconstruction_metrics_is_cumulative(tmp_path):
    arguments = [0.99, 0.98, 0.97, 0.96, 0.1, 0.2, 0.3, 0.4,
                 0.95, 0.94, 0.93, 0.92, 0.5, 0.6, 0.7, 0.8]

    path = DualSource_appendReconstructionMetrics(str(tmp_path), *arguments)
    first = Path(path).read_text()
    DualSource_appendReconstructionMetrics(str(tmp_path), *arguments)
    second = Path(path).read_text()

    assert len(second) > len(first)


def test_show_and_save_image_writes_hu_values(tmp_path):
    values = np.full(6 * 4, Constants.MuW)
    params = {"Heartrate": 60, "TRot": 0.5, "Npar": 48, "Shift_GT": 0.0,
              "PhantomTag": "LinePattern_P1C0S0"}

    image, filename = DualSource_showAndSaveImage(
        values, (6, 4), str(tmp_path), "Recon", True, params)

    assert image.shape == (4, 6)
    stored = np.fromfile(filename, dtype=np.float64)
    # Water maps exactly to 0 HU.
    assert np.allclose(stored, 0.0)
    assert "Ph_LinePattern_P1C0S0" in Path(filename).name


def test_dualsource_main_returns_matlab_results_fields(dualsourcemain, ctbasicsconfig):
    assert set(dualsourcemain) == {"Parameters", "Metrics", "OutputDirectory", "Images"}
    assert dualsourcemain["Parameters"] == {
        "HeartRateBPM": 60, "TRotSeconds": 0.5, "Nsources": 2, "Npar": 48,
        "DetectorElements": 97, "ImageSize": 64, "PhotonCount": 2.25e5,
        "RandomSeed": 1, "SelectedPhantom": "LinePattern"}
    assert set(dualsourcemain["Metrics"]) == {
        f"{metric}_{variant}{roi}" for metric in ("NCC", "RMSE", "SSIM")
        for variant in ("Normal", "DualSource") for roi in ("", "_ROI")}

    images = dualsourcemain["Images"]
    assert {key: value.shape for key, value in images.items()} == dict.fromkeys(
        ("Normal", "DualSource", "GroundTruth", "GroundTruthRest", "ROIMask"), (64, 64))
    assert images["ROIMask"].dtype == bool and images["ROIMask"].any()

    output = Path(dualsourcemain["OutputDirectory"])
    assert output.parent == ctbasicsconfig / "cardiacPhantom" / "data" / "linePattern"
    assert output.name == "aRot_0_HR_60_TRot_500ms_N_48_PS_0_NSources_2_Ph_P1C0S0"
    assert (output / "simulation_parameters.txt").is_file()
    assert (output / "reconstruction_metrics.txt").is_file()
    assert (output / "Recon-multi_HR60_TRot500ms_N48_PS0_Ph_LinePattern_P1C0S0").is_file()


def test_dualsource_main_reaches_high_correlation(dualsourcemain):
    metrics = dualsourcemain["Metrics"]

    assert metrics["NCC_Normal"] > 0.99
    assert metrics["NCC_DualSource"] > 0.99
    assert 0 < metrics["SSIM_DualSource"] <= 1
    # Metrics are in HU now: the noise-free coarse run stays within a few HU.
    assert metrics["RMSE_DualSource"] < 20


def test_dualsource_main_is_reproducible_and_writes_nothing_without_writedata(
        ctbasicsconfig):
    arguments = dict(Nsources=3, SelectedPhantom="Cardiac", PhotonCount=1e5,
                     ShowPictures=False, WriteData=False, IncludeImageData=False,
                     Npar=24, DetectorElements=49, ImageSize=32, ShowTimingPlot=False)

    first = DualSource_CardiacMotion4DCT_Main.main(72, 0.3, **arguments)
    second = DualSource_CardiacMotion4DCT_Main.main(72, 0.3, **arguments)

    assert "Images" not in first
    # rng(RandomSeed) reseeds the shared Poisson generator, so noisy runs repeat.
    assert first["Metrics"] == second["Metrics"]
    assert first["Parameters"]["HeartRateBPM"] == 72
    assert not Path(first["OutputDirectory"]).exists()
    assert Path(first["OutputDirectory"]).name == \
        "aRot_0_HR_72_TRot_300ms_N_24_PS_0_NSources_3_Ph_P1C0S0"


def test_dualsource_main_rejects_invalid_options(ctbasicsconfig):
    with pytest.raises(ValueError):
        DualSource_CardiacMotion4DCT_Main.main(Nsources=0)
    with pytest.raises(ValueError):
        DualSource_CardiacMotion4DCT_Main.main(-60)


def test_water_noise_calibration_follows_inverse_sqrt_law():
    result = WaterNoiseCalibration.main(
        N_I0_sweep=(1e4, 4e4, 1.6e5), M=65, Npar=32, Nx=32, Ny=32)

    sigma = result["measuredSigmaHU"]
    assert result["waterRoiRadiusPx"] == 7
    assert result["waterMask"].sum() > 100
    # Quadrupling the photons roughly halves the water noise.
    assert sigma[0] > sigma[1] > sigma[2]
    assert sigma[0] / sigma[1] == pytest.approx(2.0, rel=0.35)
    assert np.allclose(result["cEstimatePerRun"], sigma * np.sqrt(result["N_I0_sweep"]))
    assert result["recommendedN_I0"] == pytest.approx(
        (result["cEstimate"] / 35) ** 2)
    assert np.all(np.abs(result["measuredMeanHU"]) < 100)


def test_ctrecon101_dualsource_reproduces_matlab_argument_bug(tmp_path):
    # The MATLAB script passes 12 of 13 arguments, so MATLAB aborts with
    # "Not enough input arguments"; the verbatim port raises TypeError.
    with pytest.raises(TypeError):
        CTRecon101DualSource.main(Outputdirectory=str(tmp_path), M=65, Npar=32,
                                  Nxsim=64, Nysim=64, Nx=64, Ny=64)


def test_ctrecon101_dualsource_single_and_dual_are_identical(tmp_path):
    result = CTRecon101DualSource.main(
        Outputdirectory=str(tmp_path), M=129, Npar=64, Nxsim=96, Nysim=96,
        Nx=96, Ny=96, UseArbitraryOrder=True)

    # The point of the script: splitting the half turn over two sources changes nothing.
    assert result["maxError"] == 0.0


def test_segmented_single_source_uses_all_segment_views(tmp_path):
    result = DualSource_CardiacMotion4DCT_SegmentedSingleSource.main(
        Outputdirectory=str(tmp_path), Nsources=2, SimNoise=False,
        SelectedPhantom="WaterWith4Arteries", **COARSE)
    geometry = result["Geometry"]

    assert geometry["NprojMulti"] == 2 * geometry["Npi"]
    assert result["Metrics"][0]["corr2_multi"] > 0.99


def test_nsources_sweep_improves_with_more_sources(tmp_path):
    result = DualSource_CardiacMotion4DCT_NsourcesSweep_Experiment.main(
        OutputRoot=str(tmp_path), NsourcesList=(1, 2, 3), SimNoise=False,
        ApplyMoireFilter=False, **COARSE)

    summary = result["summaryMetrics"]
    assert summary.shape == (3, 17)
    assert list(summary[:, 0]) == [1, 2, 3]

    rmse_multi = summary[:, 6]
    assert rmse_multi[1] <= rmse_multi[0]
    assert rmse_multi[2] <= rmse_multi[1]
    assert (Path(result["OutputRoot"]) /
            "reconstruction_metrics_nsources_sweep.csv").is_file()


def test_my_noise_testing_ctrecon103_geometry(tmp_path):
    result = MyNoiseTestingCTRecon103.main(
        Outputdirectory=str(tmp_path) + "/", MFan=129, N2pi=128,
        Nxsim=96, Nysim=96, Nx=96, Ny=96)
    geometry = result["Geometry"]

    assert result["isSpectral"] is True
    # The effective water attenuation of the 140 kV spectrum.
    assert result["EffectiveMuWater"] == pytest.approx(0.0206, abs=5e-4)
    assert geometry["NFan"] == 81
    assert geometry["Npar"] == 64


def test_test_noise_cardiac_4dct_is_reproducible(tmp_path):
    # Seed only controls the view permutation; the Poisson noise draws from the global
    # generator, so the reconstruction is only reproducible with SimNoise off.
    arguments = dict(M=97, thetaN=4 * math.pi, N2pi=64, Nx=64, Ny=64, Seed=7,
                     SimNoise=False)

    first = TestNoiseCardiac4DCT.main(Outputdirectory=str(tmp_path) + "/", **arguments)
    second = TestNoiseCardiac4DCT.main(Outputdirectory=str(tmp_path) + "/", **arguments)

    assert first["Geometry"]["Nphases"] == 4
    assert np.array_equal(first["Angles_rearr"], second["Angles_rearr"])
    assert np.allclose(first["MultiphaseReconHU"], second["MultiphaseReconHU"])
