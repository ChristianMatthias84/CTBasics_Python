from enum import IntEnum


class Spectra(IntEnum):
    """Available tube-voltage spectra, matching the files in Simulation/spectra."""

    kV70 = 1
    kV80 = 2
    kV100 = 3
    kV120 = 4
    kV140 = 5
