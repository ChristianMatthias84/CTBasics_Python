"""Fan beam lecture scripts CTRecon102 (monochromatic) and CTRecon103 (polychromatic).

The geometries are coarsened so the scripts stay affordable; the physics checks are the
same ones the MATLAB lecture demonstrates.
"""
import numpy as np
import pytest

from CTBasics.LecturesAndTestScripts import CTRecon102, CTRecon103

SMALL = dict(Nxsim=96, Nysim=96, Nx=96, Ny=96, N2pi=192, MFan=193)
# Fewer energy bins keep the polychromatic forward projection affordable while the beam
# still hardens measurably.
SPECTRAL = dict(N_Ebins=6, deltaE=20, **SMALL)


def _RadialProfile(result, radii, width=5.0):
    Geometry = result["Geometry"]
    Nx, Ny = Geometry["Nx"], Geometry["Ny"]
    ax = Geometry["x0"] + np.arange(1, Nx + 1) * Geometry["dx"]
    ay = Geometry["y0"] + np.arange(1, Ny + 1) * Geometry["dy"]
    X, Y = np.meshgrid(ax, ay)
    R = np.hypot(X, Y)

    HU = result["ReconstructedImageHU"].reshape(Ny, Nx)
    return np.array([HU[(R > r - width) & (R <= r + width)].mean() for r in radii])


@pytest.fixture(scope="module")
def recon102():
    return CTRecon102.main(**SMALL)


@pytest.fixture(scope="module")
def recon103_uncorrected():
    return CTRecon103.main(SimNoise=False, WBHCorr=False, **SPECTRAL)


@pytest.fixture(scope="module")
def recon103_corrected():
    return CTRecon103.main(SimNoise=False, WBHCorr=True, **SPECTRAL)


# ------------------------------------------------------------------- CTRecon102

def test_short_scan_geometry_is_consistent(recon102):
    Geometry = recon102["Geometry"]

    betaMax = Geometry["betaMax"]
    # Short scan: half a turn plus the full fan angle.
    assert Geometry["NFan"] == int(Geometry["N2pi"] / (2 * np.pi) * (np.pi + 2 * betaMax))
    # After rebinning only the half turn remains.
    assert Geometry["Npar"] == int(Geometry["N2pi"] / (2 * np.pi) * np.pi)
    assert Geometry["RM"] == pytest.approx(np.sin(betaMax) * Geometry["RF"])
    assert Geometry["beta"][0] == pytest.approx(betaMax)
    assert Geometry["beta"][-1] == pytest.approx(-betaMax)


def test_fan_reconstruction_puts_water_at_zero_hu(recon102):
    Geometry = recon102["Geometry"]
    Nx, Ny = Geometry["Nx"], Geometry["Ny"]
    ax = Geometry["x0"] + np.arange(1, Nx + 1) * Geometry["dx"]
    X, Y = np.meshgrid(ax, ax)

    HU = recon102["ReconstructedImageHU"].reshape(Ny, Nx)
    water = X ** 2 + Y ** 2 <= (0.3 * Geometry["RM"]) ** 2

    # Measured on this grid: mean 0.4 HU, std 6.2 HU.
    assert HU[water].mean() == pytest.approx(0.0, abs=5.0)
    assert HU[water].std() < 15.0
    # Air outside the phantom.
    assert HU.min() < -900.0


def test_rebinning_halves_the_number_of_views(recon102):
    Geometry = recon102["Geometry"]

    assert recon102["FanProjections"].size == Geometry["MFan"] * Geometry["NFan"]
    assert recon102["RebinnedProjections"].size == Geometry["M"] * Geometry["Npar"]
    assert recon102["isSpectral"] is False


def test_frequency_domain_filtering_matches_the_spatial_convolution():
    spatial = CTRecon102.main(FFTConvolution=False, **SMALL)
    frequency = CTRecon102.main(FFTConvolution=True, **SMALL)

    a = spatial["ReconstructedImageHU"]
    b = frequency["ReconstructedImageHU"]

    # Both paths implement the same Ram-Lak filter, so the images must agree closely.
    assert np.abs(a - b).max() < 50.0
    assert np.corrcoef(a, b)[0, 1] > 0.999


def test_shepp_logan_kernel_smooths_more_than_ram_lak(recon102):
    from CTBasics.Reconstruction.Kernels import Kernels

    smooth = CTRecon102.main(Reconkernel=Kernels.SheppLogan, **SMALL)

    Geometry = recon102["Geometry"]
    ax = Geometry["x0"] + np.arange(1, Geometry["Nx"] + 1) * Geometry["dx"]
    X, Y = np.meshgrid(ax, ax)
    water = X ** 2 + Y ** 2 <= (0.3 * Geometry["RM"]) ** 2

    sharp_std = recon102["ReconstructedImageHU"].reshape(Geometry["Ny"], -1)[water].std()
    smooth_std = smooth["ReconstructedImageHU"].reshape(Geometry["Ny"], -1)[water].std()

    assert smooth_std < sharp_std


def test_unknown_kernel_is_rejected():
    with pytest.raises(ValueError, match="Unknown reconstruction kernel"):
        CTRecon102.main(Reconkernel="NoSuchKernel", **SMALL)


# ------------------------------------------------------------------- CTRecon103

def test_spectral_phantom_is_detected_and_water_mu_is_physical(recon103_corrected):
    assert recon103_corrected["isSpectral"] is True
    # 140 kV effective water attenuation, physically around 0.02 /mm.
    assert 0.018 < recon103_corrected["EffectiveMuWater"] < 0.023


def test_water_beam_hardening_correction_removes_the_cupping(
    recon103_uncorrected, recon103_corrected
):
    radii = [30.0, 50.0, 70.0, 90.0]

    uncorrected = _RadialProfile(recon103_uncorrected, radii)
    corrected = _RadialProfile(recon103_corrected, radii)

    # Measured on this 6-bin spectrum: cupping drops from 14.7 HU to below 5 HU, and the
    # whole water region moves from about -30 HU back to 0 HU.
    assert uncorrected.max() - uncorrected.min() > 10.0
    assert np.abs(uncorrected).max() > 15.0
    assert corrected.max() - corrected.min() < 5.0
    assert np.abs(corrected).max() < 5.0


def test_poisson_noise_raises_the_image_noise_without_shifting_the_mean():
    quiet = CTRecon103.main(SimNoise=False, **SPECTRAL)
    noisy = CTRecon103.main(SimNoise=True, N_I0=100000, **SPECTRAL)

    radii = [50.0]
    assert _RadialProfile(noisy, radii)[0] == pytest.approx(
        _RadialProfile(quiet, radii)[0], abs=20.0
    )

    Geometry = quiet["Geometry"]
    ax = Geometry["x0"] + np.arange(1, Geometry["Nx"] + 1) * Geometry["dx"]
    X, Y = np.meshgrid(ax, ax)
    water = (np.hypot(X, Y) > 30.0) & (np.hypot(X, Y) < 90.0)

    quiet_std = quiet["ReconstructedImageHU"].reshape(Geometry["Ny"], -1)[water].std()
    noisy_std = noisy["ReconstructedImageHU"].reshape(Geometry["Ny"], -1)[water].std()

    assert noisy_std > quiet_std
