import numpy as np

from ctbasics.preprocessing import MoireFilter, WBHC
from ctbasics.simulation import Spectra


def test_wbhc_lut_is_monotonic_and_returns_effective_mu_water():
    lut, effective_mu_water = WBHC.generate_lut_with_wedge_simulation(Spectra.kV70, 20, 5, 10)

    assert lut.shape[0] == 2
    assert lut.shape[1] > 1
    assert effective_mu_water > 0.0
    assert np.all(np.diff(lut[0]) >= 0)


def test_moire_filter_preserves_input_shape():
    signal = np.arange(16, dtype=float)
    filtered = MoireFilter.apply_moire_filter_on_projections(signal, 1, 16, 1.0, 0.1, 0.2)

    assert filtered.shape == (16,)