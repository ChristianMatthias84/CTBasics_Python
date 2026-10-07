"""Numerical checks of the tabulated spectral data and the Poisson noise model."""
import numpy as np
import pytest

from CTBasics.Reconstruction.Constants import Constants
from CTBasics.Simulation.MaterialProperties import MaterialProperties
from CTBasics.Simulation.Materials import Materials
from CTBasics.Simulation.Noise import Noise
from CTBasics.Simulation.Spectra import Spectra
from CTBasics.Simulation.SpectralProperties import SpectralProperties


@pytest.fixture(scope="module")
def properties():
    return MaterialProperties()


def test_water_attenuation_matches_the_muw_constant(properties):
    # Constants.MuW is documented as water at the ~70 keV mean energy of the spectrum.
    mu = properties.GetSpectralAttenuationCoefficient(Materials.Nyst_water, 70.0)

    assert abs(mu - Constants.MuW) / Constants.MuW < 0.02


def test_attenuation_decreases_with_energy(properties):
    values = [properties.GetSpectralAttenuationCoefficient(Materials.Nyst_water, E)
              for E in (40.0, 70.0, 100.0, 140.0)]

    assert all(a > b for a, b in zip(values, values[1:]))


def test_materials_are_ordered_by_attenuation(properties):
    air = properties.GetSpectralAttenuationCoefficient(Materials.Air, 70.0)
    water = properties.GetSpectralAttenuationCoefficient(Materials.Nyst_water, 70.0)
    bone = properties.GetSpectralAttenuationCoefficient(Materials.Nyst_CorticalBone, 70.0)
    iron = properties.GetSpectralAttenuationCoefficient(Materials.Nyst_Iron, 70.0)

    assert air == 0.0
    assert water < bone < iron

    # Cortical bone is roughly 2.5x water at 70 keV.
    assert 2.0 < bone / water < 3.0


def test_interpolation_is_linear_between_tabulated_points(properties):
    table = properties.SpectralAttenuationCoefficients[Materials.Nyst_water]
    lower, upper = table[10], table[11]
    middle = (lower[0] + upper[0]) / 2

    value = properties.GetSpectralAttenuationCoefficient(Materials.Nyst_water, middle)

    assert value == pytest.approx((lower[1] + upper[1]) / 2)


def test_tabulated_energies_are_in_kev_and_coefficients_in_per_mm(properties):
    table = properties.MassAttenuationCoefficients[Materials.Nyst_water]

    # The NIST file is in MeV and cm^2/g; the port converts to keV and mm^2/g.
    assert table[0, 0] == pytest.approx(1.0)
    assert table[-1, 0] > 1000.0


def test_material_image_conversion_maps_every_code(properties):
    MaterialImage = np.array([
        Materials.Air, Materials.Nyst_water,
        Materials.Nyst_CorticalBone, Materials.Nyst_Iron,
    ], dtype=int)

    values = properties.MaterialToSpectralAttenuationCoefficientConversion(MaterialImage, 2, 2, 70.0)

    assert values.shape == (4,)
    assert values[0] == 0.0
    for code, value in zip(MaterialImage, values):
        assert value == pytest.approx(properties.GetSpectralAttenuationCoefficient(code, 70.0))


def test_spectra_tables_are_loaded_for_every_tube_voltage():
    spectral = SpectralProperties()

    assert set(spectral.Spectra) == set(Spectra)

    for spectrum, table in spectral.Spectra.items():
        assert table.shape[1] == 2
        assert table[:, 1].min() >= 0.0
        assert table[:, 1].sum() > 0.0


def test_spectrum_mean_energy_grows_with_tube_voltage():
    spectral = SpectralProperties()

    def mean_energy(spectrum):
        table = spectral.Spectra[spectrum]
        return (table[:, 0] * table[:, 1]).sum() / table[:, 1].sum()

    means = [mean_energy(s) for s in (Spectra.kV70, Spectra.kV80, Spectra.kV100,
                                      Spectra.kV120, Spectra.kV140)]

    assert all(a < b for a, b in zip(means, means[1:]))


def test_spectrum_has_no_fluence_above_the_tube_voltage():
    spectral = SpectralProperties()
    table = spectral.Spectra[Spectra.kV70]

    assert table[table[:, 0] > 70.0, 1].sum() == 0.0


def test_poisson_noise_is_unbiased_and_matches_the_double_process_variance():
    rng = np.random.default_rng(0)
    I0 = 1.0e5
    p = 2.0
    projections = np.full(200000, p)

    noisy = Noise.ApplyPoissonNoise(projections, projections.size, I0)

    assert abs(noisy.mean() - p) < 0.01

    # Emission and detection are both Poisson, so Var(Y) = I0*e^-p*(1 + e^-p).
    expected = np.sqrt(I0 * np.exp(-p) * (1 + np.exp(-p))) / (I0 * np.exp(-p))
    assert abs(noisy.std() - expected) / expected < 0.1


def test_detection_only_noise_has_the_single_poisson_variance():
    I0 = 1.0e5
    p = 2.0
    projections = np.full(200000, p)

    noisy = Noise.ApplyingPoissonNoisePhotonAfterAbsorption(projections, I0)

    expected = 1.0 / np.sqrt(I0 * np.exp(-p))
    assert abs(noisy.mean() - p) < 0.01
    assert abs(noisy.std() - expected) / expected < 0.1


def test_noise_grows_with_attenuation():
    I0 = 1.0e5

    deviations = [
        Noise.ApplyPoissonNoise(np.full(50000, p), 50000, I0).std()
        for p in (1.0, 2.0, 3.0)
    ]

    assert deviations[0] < deviations[1] < deviations[2]


def test_block_parallel_noise_keeps_the_projection_length():
    I0 = 1.0e5
    projections = np.full(1000, 1.5)

    noisy = Noise.ApplyPoissonNoiseParallel(projections, projections.size, I0)

    assert noisy.shape == projections.shape
    assert abs(noisy.mean() - 1.5) < 0.05
