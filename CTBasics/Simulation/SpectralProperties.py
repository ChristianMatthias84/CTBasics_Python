from pathlib import Path

import numpy as np

from CTBasics.Simulation.Spectra import Spectra

_SPECTRA = Path(__file__).parent / "spectra"

_FILES = {
    Spectra.kV70: "70kV",
    Spectra.kV80: "80kV",
    Spectra.kV100: "100kV",
    Spectra.kV120: "120kV",
    Spectra.kV140: "140kV",
}


class SpectralProperties:
    """Tabulated tube spectra; column 1 is the energy in keV, column 2 the relative fluence."""

    def __init__(self):
        self.Spectra = {
            spectrum: np.loadtxt(_SPECTRA / f"{name}.txt")[:, [0, 1]].astype(float)
            for spectrum, name in _FILES.items()
        }


def _SpectralBins(spectrum, E0, N_Ebins, deltaE):
    """Samples the tube spectrum at the energies of N_Ebins bins.

    MATLAB walks the table once with a moving start index and takes the first row whose
    energy reaches the bin energy, so bins beyond the table end keep a fluence of zero.
    Returns (I0Values, SumOverI0).
    """
    # MATLAB passes char(spectrum); accept both the enum and its name.
    key = spectrum if isinstance(spectrum, Spectra) else Spectra[str(spectrum)]
    aspectrum = np.asarray(SpectralProperties().Spectra[key], dtype=float)
    aspectrumLength = len(aspectrum)

    I0Values = np.zeros(N_Ebins)
    SumOverI0 = 0.0

    searchstart = 1
    for nbin in range(1, N_Ebins + 1):
        energy = int(E0 + nbin * deltaE)
        for i in range(searchstart, aspectrumLength + 1):
            if aspectrum[i - 1, 0] >= energy:
                searchstart = i
                I0Values[nbin - 1] = aspectrum[i - 1, 1]
                SumOverI0 += I0Values[nbin - 1]
                break

    return I0Values, SumOverI0
