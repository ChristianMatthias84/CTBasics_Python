import numpy as np

from CTBasics.Memory.Allocation import Allocation
from CTBasics.Reconstruction.Kernels import Kernels


class MoireFilter:
    @staticmethod
    def ApplyMoireFilterOnProjections(UnfilteredProjections, Nprojections, NFFT, ds, Startfrequency, Stopfrequency):
        """Suppress Moire artefacts by damping a frequency band of each projection.

        The 1/NFFT scaling of the forward transform and the NFFT of the inverse cancel;
        both are kept so the intermediate spectrum matches the MATLAB amplitudes.
        """
        UnfilteredProjections = np.asarray(UnfilteredProjections, dtype=float).ravel()

        Filter = Kernels.GetMoireFilter(NFFT, ds, Startfrequency, Stopfrequency)

        FilteredProjections = Allocation.zeros(Nprojections * NFFT)
        for n in range(1, Nprojections + 1):
            Projection = UnfilteredProjections[(n - 1) * NFFT:n * NFFT]
            Spectrum = np.fft.fft(Projection) / NFFT
            FilteredProjections[(n - 1) * NFFT:n * NFFT] = np.real(np.fft.ifft(Spectrum * Filter) * NFFT)

        return FilteredProjections

    @staticmethod
    def ApplyMoireFilterOnProjections3D(Projections, ParGeo_in, FanGeo_in):
        """Applies the Moire filter row by row on a 3D parallel sinogram.

        Each detector row is filtered as one long 1D signal, so the regrouping extracts
        the (n, m) block of a single row before filtering and interleaves it back after.
        """
        from CTBasics.Descriptors.ScanTypes import ScanTypes

        fstart = 0.2 * (1 / ParGeo_in.dxi)
        fstop = 0.5 * (1 / ParGeo_in.dxi)

        M = ParGeo_in.M
        Ns = FanGeo_in.Ns
        dxi = ParGeo_in.dxi

        if FanGeo_in.ScanType == ScanTypes.Sequence:
            Ntotal = ParGeo_in.Npar_seq * FanGeo_in.NzScans_seq
        else:
            Ntotal = ParGeo_in.Npar_spi

        RegroupedProjections = np.asarray(Projections, dtype=float).reshape(Ns * Ntotal, M)

        MoireFilteredProjections = Allocation.zeros(Ns, M * Ntotal)
        for ns in range(1, Ns + 1):
            subgroup = RegroupedProjections[ns - 1::Ns, :].ravel()
            MoireFilteredProjections[ns - 1, :] = MoireFilter.ApplyMoireFilterOnProjections(
                subgroup, Ntotal, M, dxi, fstart, fstop
            )

        return MoireFilteredProjections.reshape(Ns, Ntotal, M).transpose(1, 0, 2).ravel()
