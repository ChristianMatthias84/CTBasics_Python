from enum import IntEnum


class Materials(IntEnum):
    """Material codes used by the spectral phantom and attenuation tables."""

    Air = 0
    Nyst_aluminum = 1
    Nyst_CorticalBone = 2
    Nyst_water = 3
    Nyst_Iron = 4
