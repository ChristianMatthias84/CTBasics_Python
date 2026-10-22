"""Lecture scripts TestScript2D (over-scan WFBP) and TestMAR."""
import numpy as np
import pytest

from CTBasics.LecturesAndTestScripts import TestMAR, TestScript2D

SMALL = dict(Nxsim=96, Nysim=96, Nx=96, Ny=96, N2pi=192, MFan=193, PatientData=False,
             SelectedPhantom="SimpleEllipseWithHips")


@pytest.fixture(scope="module")
def script2d():
    return TestScript2D.main(**SMALL)


def test_over_scan_covers_more_than_a_full_turn(script2d):
    Geometry = script2d["Geometry"]

    # 2*pi + fan + a quarter turn, i.e. about 1.39 rotations.
    assert Geometry["alphaN"] == pytest.approx(2 * np.pi + 2 * Geometry["betaMax"] + np.pi / 2)
    assert Geometry["alphaN"] / (2 * np.pi) > 1.0
    # The rebinned parallel scan keeps the redundant views.
    assert Geometry["Npar"] > Geometry["N2pi"] / 2


def test_weighted_fbp_handles_the_redundant_views(script2d):
    Geometry = script2d["Geometry"]
    ax = Geometry["x0"] + np.arange(1, Geometry["Nx"] + 1) * Geometry["dx"]
    X, Y = np.meshgrid(ax, ax)

    HU = script2d["ReconstructedImageHU"].reshape(Geometry["Ny"], Geometry["Nx"])
    water = X ** 2 + Y ** 2 <= (0.3 * Geometry["RM"]) ** 2

    # Without the redundancy weighting the doubly measured sector would be twice as bright.
    # Measured: 0.2 HU mean, 7.7 HU std.
    assert HU[water].mean() == pytest.approx(0.0, abs=5.0)
    assert HU[water].std() < 15.0
    assert HU.min() < -900.0


def test_moire_filter_does_not_change_the_water_level(script2d):
    filtered = TestScript2D.main(ApplyMoireFilter=True, **SMALL)

    Geometry = script2d["Geometry"]
    ax = Geometry["x0"] + np.arange(1, Geometry["Nx"] + 1) * Geometry["dx"]
    X, Y = np.meshgrid(ax, ax)
    water = X ** 2 + Y ** 2 <= (0.3 * Geometry["RM"]) ** 2

    plain = script2d["ReconstructedImageHU"].reshape(Geometry["Ny"], -1)
    moire = filtered["ReconstructedImageHU"].reshape(Geometry["Ny"], -1)

    assert moire[water].mean() == pytest.approx(plain[water].mean(), abs=10.0)
    assert not np.array_equal(moire, plain)


def test_poisson_noise_increases_the_image_noise(script2d):
    noisy = TestScript2D.main(SimNoise=True, N_I0=50000, **SMALL)

    Geometry = script2d["Geometry"]
    ax = Geometry["x0"] + np.arange(1, Geometry["Nx"] + 1) * Geometry["dx"]
    X, Y = np.meshgrid(ax, ax)
    water = X ** 2 + Y ** 2 <= (0.3 * Geometry["RM"]) ** 2

    quiet_std = script2d["ReconstructedImageHU"].reshape(Geometry["Ny"], -1)[water].std()
    noisy_std = noisy["ReconstructedImageHU"].reshape(Geometry["Ny"], -1)[water].std()

    assert noisy_std > quiet_std


def test_test_mar_reduces_the_streaks_of_a_metal_hip(tmp_path, monkeypatch):
    output = tmp_path / "out"
    output.mkdir()
    (tmp_path / "Data").mkdir()
    (tmp_path / "CTBasicsConfig.txt").write_text(f"#\n{tmp_path}\n#\n{output}\n#\n\n")
    monkeypatch.setenv("CTBASICS_CONFIG", str(tmp_path))

    result = TestMAR.main(UsePatientData=False, Nx=96, Ny=96, Npar=128, M=129)

    Phantom = result["Voxelphantom"].reshape(96, 96)
    # HipWithMetal carries a dense implant well above the metal threshold.
    assert Phantom.max() == pytest.approx(4000.0)
    assert np.count_nonzero(Phantom > 3000) > 0
    assert Phantom.min() == pytest.approx(-1000.0)

    Geometry = result["Geometry"]
    ax = Geometry["x0"] + np.arange(1, 97) * Geometry["dx"]
    X, Y = np.meshgrid(ax, ax)
    soft = np.hypot(X, Y) < 60.0

    # MATLAB never removes the +1000 shift it applies to the input, so soft tissue
    # lands near 1000 instead of 0 HU. Measured: 995.7 HU.
    assert result["Result"].reshape(96, 96)[soft].mean() == pytest.approx(1000.0, abs=20.0)
    assert np.isfinite(result["Result"]).all()


def test_test_mar_patient_branch_needs_the_dicom_data():
    with pytest.raises((FileNotFoundError, OSError)):
        TestMAR.main(UsePatientData=True, DataDirectory="does/not/exist",
                     Nx=32, Ny=32, Npar=16, M=17)
