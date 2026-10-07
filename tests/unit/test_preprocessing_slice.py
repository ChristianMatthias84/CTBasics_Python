import numpy as np

from CTBasics.Preprocessing import WBHC
from CTBasics.Reconstruction import MoireFilter
from CTBasics.Simulation import Spectra


def test_wbhc_lut_is_monotonic_and_returns_effective_mu_water():
    lut, effective_mu_water = WBHC.GenerateLUTWithWedgeSimulation(Spectra.kV70, 20, 5, 10)

    assert lut.shape[0] == 2
    assert lut.shape[1] > 1
    assert effective_mu_water > 0.0
    assert np.all(np.diff(lut[0]) >= 0)


def test_moire_filter_preserves_input_shape():
    signal = np.arange(16, dtype=float)
    filtered = MoireFilter.ApplyMoireFilterOnProjections(signal, 1, 16, 1.0, 0.1, 0.2)

    assert filtered.shape == (16,)
