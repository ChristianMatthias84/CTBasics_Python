"""Water beam hardening correction."""
import numpy as np
import pytest

from CTBasics.Preprocessing.WBHC import WBHC

SPECTRUM = "kV120"
E0, N_EBINS, DELTA_E = 20.0, 10, 10.0


@pytest.fixture(scope="module")
def lut():
    return WBHC.GenerateLUTWithWedgeSimulation(SPECTRUM, E0, N_EBINS, DELTA_E)


def test_effective_water_attenuation_is_physical(lut):
    _, EffectiveMuWater = lut

    # Water at diagnostic CT energies attenuates with roughly 0.02 /mm.
    assert 0.015 < EffectiveMuWater < 0.030


def test_lut_covers_the_simulated_wedge(lut):
    LUT, EffectiveMuWater = lut

    assert LUT.shape == (2, 81)
    assert LUT[0, 0] == pytest.approx(0.0, abs=1e-12)
    # Row 2 is the ideal monochromatic line through the effective attenuation.
    Diameters = np.arange(0, 810, 10, dtype=float)
    assert np.allclose(LUT[1], EffectiveMuWater * Diameters)


def test_beam_hardening_lowers_the_measured_line_integral(lut):
    LUT, _ = lut

    assert np.all(LUT[0] <= LUT[1] + 1e-12)
    assert np.all(np.diff(LUT[0]) > 0)
    # The further the beam hardens, the larger the gap grows.
    assert np.all(np.diff(LUT[1] - LUT[0]) >= -1e-12)


def test_correction_maps_the_lut_nodes_onto_the_ideal_line(lut):
    LUT, _ = lut

    Corrected, EffectiveMuWater = WBHC.LUTBasedWBHCCorrection(
        SPECTRUM, E0, N_EBINS, DELTA_E, LUT[0]
    )

    assert np.allclose(Corrected, LUT[1])
    assert EffectiveMuWater == pytest.approx(_ := EffectiveMuWater)


def test_correction_interpolates_between_lut_nodes(lut):
    LUT, EffectiveMuWater = lut

    Hardened = 0.5 * (LUT[0, 10] + LUT[0, 11])
    Corrected, _ = WBHC.LUTBasedWBHCCorrection(SPECTRUM, E0, N_EBINS, DELTA_E, [Hardened])

    assert LUT[1, 10] < Corrected[0] < LUT[1, 11]


def test_correction_saturates_beyond_the_wedge(lut):
    LUT, _ = lut

    Beyond = LUT[0, -1] + 1.0
    Corrected, _ = WBHC.LUTBasedWBHCCorrection(SPECTRUM, E0, N_EBINS, DELTA_E, [Beyond])

    assert Corrected[0] == pytest.approx(Beyond + (LUT[1, -1] - LUT[0, -1]))


def test_correction_leaves_an_empty_ray_untouched():
    Corrected, _ = WBHC.LUTBasedWBHCCorrection(SPECTRUM, E0, N_EBINS, DELTA_E, [0.0])

    assert Corrected[0] == pytest.approx(0.0, abs=1e-12)
