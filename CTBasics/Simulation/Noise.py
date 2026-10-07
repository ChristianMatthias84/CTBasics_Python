import numpy as np

from ..Memory.Allocation import Allocation

_Generator = np.random.default_rng()


def _Poisson(expectation):
    return _Generator.poisson(np.asarray(expectation, dtype=float)).astype(float)


class Noise:
    """Poisson statistics of x-ray generation and detection."""

    @staticmethod
    def ApplyingPoissonNoiseBeforeAbsorption(Projections, M, N, I0):
        """Only the emission process is random; the absorption stays deterministic."""
        Projections = np.asarray(Projections, dtype=float).ravel()
        emitted = _Poisson(I0 * Allocation.ones(M * N))
        return -np.log((emitted * np.exp(-Projections)) / I0)

    @staticmethod
    def ApplyingPoissonNoisePhotonAfterAbsorption(Projections, I0):
        """Only the detection process is random."""
        Projections = np.asarray(Projections, dtype=float).ravel()
        return -np.log(_Poisson(I0 * np.exp(-Projections)) / I0)

    @staticmethod
    def ApplyPoissonNoise(Projections, N, I0):
        """Combined emission and detection statistics; zero counts are clamped to one photon."""
        Projections = np.asarray(Projections, dtype=float).ravel()

        NumberPhotons = _Poisson(_Poisson(I0 * Allocation.ones(N)) * np.exp(-Projections))

        if np.any(NumberPhotons < 0):
            raise ValueError("The number of photons must not be less then zero")

        return -np.log(np.where(NumberPhotons > 0, NumberPhotons, 1.0) / I0)

    @staticmethod
    def ApplyPoissonNoiseParallel(Projections, N, I0):
        """Block-wise variant of ApplyPoissonNoise; MATLAB runs the blocks in a parfor."""
        Projections = np.asarray(Projections, dtype=float).ravel()

        NrBlocks = 12
        Blocksize = N // NrBlocks
        Rest = N - NrBlocks * Blocksize

        SubProjections = Allocation.zeros(NrBlocks, Blocksize)

        for nr in range(1, NrBlocks + 1):
            SubProjections[nr - 1, :] = Noise.ApplyPoissonNoise(
                Projections[(nr - 1) * Blocksize:nr * Blocksize], Blocksize, I0
            )

        NoisyProjections = SubProjections.ravel()

        if Rest > 0:
            SubProjectionRest = Noise.ApplyPoissonNoise(
                Projections[NrBlocks * Blocksize:NrBlocks * Blocksize + Rest], Rest, I0
            )
            NoisyProjections = np.concatenate([NoisyProjections, SubProjectionRest])

        return NoisyProjections
