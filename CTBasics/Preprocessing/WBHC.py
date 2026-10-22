import numpy as np

from CTBasics.Memory.Allocation import Allocation
from CTBasics.Simulation.Materials import Materials
from CTBasics.Simulation.MaterialProperties import MaterialProperties
from CTBasics.Simulation.SpectralProperties import _SpectralBins


class WBHC:
    """Water beam hardening correction based on a simulated water wedge."""

    WBHC_LUTs = {}

    @staticmethod
    def LUTBasedWBHCCorrection(spectrum, E0, N_Ebins, deltaE, UncorrectedProjection):
        """Linearizes polychromatic line integrals onto the monochromatic water line.

        Returns (WHBCCorrectedProjections, EffectiveMuWater).
        """
        LUT, EffectiveMuWater = WBHC.GenerateLUTWithWedgeSimulation(spectrum, E0, N_Ebins, deltaE)

        UncorrectedProjection = np.asarray(UncorrectedProjection, dtype=float).ravel()
        NrDetectorValues = len(UncorrectedProjection)
        sizeLUT = LUT.shape[1]

        WHBCCorrectedProjections = Allocation.zeros(NrDetectorValues)
        for i in range(1, NrDetectorValues + 1):
            unccorectedValue = UncorrectedProjection[i - 1]
            WHBCCorrectedProjections[i - 1] = unccorectedValue
            for j in range(1, sizeLUT + 1):
                if LUT[0, j - 1] > unccorectedValue:
                    w = (LUT[0, j - 1] - unccorectedValue) / (LUT[0, j - 1] - LUT[0, j - 2])
                    delta = (1 - w) * LUT[1, j - 1] + w * LUT[1, j - 2] - unccorectedValue
                    WHBCCorrectedProjections[i - 1] = unccorectedValue + delta
                    break
                if j == sizeLUT:
                    # Beyond the wedge the correction saturates at the last LUT offset.
                    delta = LUT[1, sizeLUT - 1] - LUT[0, sizeLUT - 1]
                    WHBCCorrectedProjections[i - 1] = unccorectedValue + delta

        return WHBCCorrectedProjections, EffectiveMuWater

    @staticmethod
    def GenerateLUTWithWedgeSimulation(spectrum, E0, N_Ebins, deltaE):
        """Simulates a water wedge to map polychromatic onto monochromatic line integrals.

        Row 1 of the LUT holds the hardened values, row 2 the ideal ones.
        Returns (LUT, EffectiveMuWater).
        """
        I0Values, SumOverI0 = _SpectralBins(spectrum, E0, N_Ebins, deltaE)

        aMaterialProperties = MaterialProperties()
        mu_Water = np.array([
            aMaterialProperties.GetSpectralAttenuationCoefficient(
                Materials.Nyst_water, int(E0 + nbin * deltaE))
            for nbin in range(1, N_Ebins + 1)
        ])

        mean_mu_Water = float(np.sum(mu_Water * I0Values) / SumOverI0)

        maxDiameter = 800  # in mm
        SamplingWidth = 10  # in mm
        Diameters = np.arange(0, maxDiameter + SamplingWidth, SamplingWidth, dtype=float)

        Intensity = (I0Values[:, None] * np.exp(-mu_Water[:, None] * Diameters[None, :])).sum(axis=0)
        ProjectionsWithBeamHardening = -np.log(Intensity / SumOverI0)
        ProjectionsWithoutBeamHardening = mean_mu_Water * Diameters

        LUT = np.vstack([ProjectionsWithBeamHardening, ProjectionsWithoutBeamHardening])

        return LUT, mean_mu_Water
