import numpy as np


class FocalSpotSimulationStaticCT:
    """Sampling positions across the extended focal spot of the static CT source."""

    @staticmethod
    def CalculateFocalSpotSamplingPositions(focalSpotWidth=None, focalSpotLength=None,
                                            anodeAngle=None, nrOfWidthSamples=None,
                                            nrOfLengthSamples=None):
        """Returns an N x 3 matrix of [dzF, drF, dxyF] offsets relative to the nominal focus.

        The length direction runs along the anode slope, so a sample at distance l
        contributes both an axial (dzF) and a radial (drF) offset.
        """
        if focalSpotWidth is None:
            focalSpotWidth = 1.7
        if focalSpotLength is None:
            focalSpotLength = 12.2
        if anodeAngle is None:
            anodeAngle = -8 / 180 * np.pi
        if nrOfWidthSamples is None:
            nrOfWidthSamples = 3
        if nrOfLengthSamples is None:
            nrOfLengthSamples = 35

        nrOfWidthSamples = int(nrOfWidthSamples)
        nrOfLengthSamples = int(nrOfLengthSamples)

        dw = focalSpotWidth / (nrOfWidthSamples - 1) if nrOfWidthSamples > 1 else 0.0
        dl = focalSpotLength / (nrOfLengthSamples - 1) if nrOfLengthSamples > 1 else 0.0

        matrixOfSamplingPositions = np.zeros((nrOfWidthSamples * nrOfLengthSamples, 3))
        ctr = 0

        for iw in range(nrOfWidthSamples):
            w = (-(nrOfWidthSamples - 1) / 2 + iw) * dw
            for il in range(nrOfLengthSamples):
                l = (-(nrOfLengthSamples - 1) / 2 + il) * dl

                matrixOfSamplingPositions[ctr, :] = [l * np.sin(anodeAngle), l * np.cos(anodeAngle), w]
                ctr += 1

        return matrixOfSamplingPositions
