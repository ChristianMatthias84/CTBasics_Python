from pathlib import Path

import numpy as np

from CTBasics.Memory.Allocation import Allocation
from CTBasics.Reconstruction.Constants import Constants
from CTBasics.Simulation.Materials import Materials

_COEFFS = Path(__file__).parent / "attenuationcoeffs"

_DENSITIES = {
    Materials.Air: Constants.DensityAir,
    Materials.Nyst_aluminum: Constants.DensityAluminum,
    Materials.Nyst_CorticalBone: Constants.DensityCorticalBone,
    Materials.Nyst_water: Constants.DensityWater,
    Materials.Nyst_Iron: Constants.DensityIron,
}


def _LoadTable(name, scale_units):
    """Reads an NIST table; columns are energy and mass attenuation coefficient."""
    data = np.loadtxt(_COEFFS / f"{name}.txt")[:, [0, 1]].astype(float)

    if scale_units:
        data[:, 0] *= 1000  # MeV -> keV
        data[:, 1] *= 100   # cm^2/g -> mm^2/g

    return data


class MaterialProperties:
    """Tabulated mass and spectral attenuation coefficients for the simulated materials."""

    def __init__(self):
        # The Air table is all zeros and is therefore used unscaled, like in MATLAB.
        tables = {
            Materials.Air: _LoadTable("Air", scale_units=False),
            Materials.Nyst_aluminum: _LoadTable("Nyst_aluminum", scale_units=True),
            Materials.Nyst_CorticalBone: _LoadTable("Nyst_CorticalBone", scale_units=True),
            Materials.Nyst_water: _LoadTable("Nyst_water", scale_units=True),
            Materials.Nyst_Iron: _LoadTable("Nyst_Iron", scale_units=True),
        }

        self.MassAttenuationCoefficients = {material: table for material, table in tables.items()}
        self.MassDensities = dict(_DENSITIES)

        self.SpectralAttenuationCoefficients = {}
        for material, table in tables.items():
            spectral = table.copy()
            if material != Materials.Air:
                spectral[:, 1] *= _DENSITIES[material]
            self.SpectralAttenuationCoefficients[material] = spectral

    def GetSpectralAttenuationCoefficient(self, Material, energy):
        """Linearly interpolates mu(E) between the two tabulated energies bracketing `energy`."""
        data = self.SpectralAttenuationCoefficients[Materials(Material)]

        energies = data[:, 0]
        SpectralAttenuationCoefficient = data[:, 1]

        index = len(energies)
        for i in range(1, len(energies) + 1):
            index = i
            if energies[i - 1] >= energy:
                break

        if energies[index - 1] - energies[index - 2] > 0:
            w = (energy - energies[index - 2]) / (energies[index - 1] - energies[index - 2])
            return w * SpectralAttenuationCoefficient[index - 1] + (1 - w) * SpectralAttenuationCoefficient[index - 2]

        return SpectralAttenuationCoefficient[index - 1]

    def GetSpectralAttenuationCoefficients(self, energy):
        return {int(Material): self.GetSpectralAttenuationCoefficient(Material, energy)
                for Material in Materials}

    def MaterialToSpectralAttenuationCoefficientConversion(self, MaterialImage, Nx, Ny, energy):
        """Replaces every material code of the phantom by its attenuation at the given energy."""
        MaterialImage = np.asarray(MaterialImage).ravel()

        SpecAttCoeffAtEnergy = self.GetSpectralAttenuationCoefficients(energy)

        SpectralAttenuationCoeffecients = Allocation.zeros(Nx * Ny)
        for code, value in SpecAttCoeffAtEnergy.items():
            SpectralAttenuationCoeffecients[MaterialImage[:Nx * Ny] == code] = value

        return SpectralAttenuationCoeffecients
