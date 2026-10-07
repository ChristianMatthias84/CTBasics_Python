import numpy as np

from ..Memory.Allocation import Allocation


class Kernels:
    SheppLogan = "SheppLogan"
    RamLak = "RamLak"
    RamLakHann = "RamLakHann"

    @staticmethod
    def GetRamLakKernelValueSD(dxi, m):
        if m == 0:
            return 1.0 / (4.0 * dxi * dxi)
        if m % 2 == 0:
            return 0.0
        return -1.0 / (np.pi * np.pi * dxi * dxi * m * m)

    @staticmethod
    def RamLakKernelSD(M, dxi):
        Kernel = Allocation.zeros(2 * M + 1)
        for m in range(-M, M + 1):
            Kernel[m + M] = Kernels.GetRamLakKernelValueSD(dxi, m)
        return Kernel

    @staticmethod
    def GetSheppLoganKernlValueSD(dxi, m):
        return -2.0 / (np.pi * np.pi * dxi * dxi) * 1.0 / (4.0 * m * m - 1.0)

    @staticmethod
    def SheppLoganKernelSD(M, dxi):
        Kernel = Allocation.zeros(2 * M + 1)
        for m in range(-M, M + 1):
            Kernel[m + M] = Kernels.GetSheppLoganKernlValueSD(dxi, m)
        return Kernel

    @staticmethod
    def RamLakHannKernelSD(M, dxi):
        """Ram-Lak tapered by a Hann window, which suppresses Gibbs ringing at sharp edges."""
        Kernel = Allocation.zeros(2 * M + 1)
        for m in range(-M, M + 1):
            w = 0.5 * (1.0 + np.cos(np.pi * m / M))
            Kernel[m + M] = Kernels.GetRamLakKernelValueSD(dxi, m) * w
        return Kernel

    @staticmethod
    def _MirrorFilter(Filter, NFFT):
        """MATLAB mirrors the upper half because fft stores the signal with 1/ds periodicity."""
        endOfMirror = NFFT // 2
        for i in range(NFFT, endOfMirror - 1, -1):
            Filter[i - 1] = Filter[NFFT - i + 1]
        return Filter

    @staticmethod
    def RamLakKernelFD(NFFT, ds, Wq):
        Kernel = Allocation.zeros(NFFT)
        du = 1.0 / (ds * NFFT)
        uNyquist = 0.5 * 1.0 / ds
        for i in range(1, NFFT + 1):
            u = (i - 1) * du
            if u < uNyquist:
                Kernel[i - 1] = u * Wq[i - 1]
        return Kernels._MirrorFilter(Kernel, NFFT)

    @staticmethod
    def GetMoireFilter(NFFT, ds, Startfrequency, Stoppfrequency):
        MoireFilter = Allocation.zeros(NFFT)
        MoireStart = Startfrequency
        MoireStopp = Stoppfrequency
        df = 1.0 / (NFFT * ds)
        FadingWidth = MoireStopp - MoireStart
        for i in range(1, NFFT + 1):
            f = (i - 1) * df
            if f < MoireStart:
                MoireFilter[i - 1] = 1.0
            elif f <= MoireStopp:
                value = np.cos(np.pi / 2.0 * (f - MoireStart) / FadingWidth)
                MoireFilter[i - 1] = value * value
            else:
                MoireFilter[i - 1] = 0.0
        return Kernels._MirrorFilter(MoireFilter, NFFT)

    @staticmethod
    def GetSincSquareValue(n, deltax, NFFT, dxi):
        arg = (n * deltax * np.pi) / (dxi * NFFT)
        Sinc = np.sin(arg) / arg
        return Sinc * Sinc

    @staticmethod
    def SincSquareFD(deltax, NFFT, deltam):
        SincSquare = Allocation.zeros(NFFT)
        for n in range(1, NFFT + 1):
            SincSquare[n - 1] = Kernels.GetSincSquareValue(n, deltax, NFFT, deltam)
        return SincSquare

    @staticmethod
    def AssembleKernel(Kernel, Moire, Sinc, Length):
        CompleteKernel = Allocation.zeros(Length)
        for i in range(Length):
            CompleteKernel[i] = Kernel[i] * Moire[i] / Sinc[i]
        return CompleteKernel

    @staticmethod
    def PerformFilteringInFrequencyDomain(UnfilteredProjections, N180, M, NFFT, Kernel):
        # The MATLAB original opens debug figures between the steps; they are omitted here.
        UnfilteredProjectionsZeroPadded = Allocation.zeros(N180 * NFFT)
        for n in range(N180):
            UnfilteredProjectionsZeroPadded[n * NFFT:n * NFFT + M] = UnfilteredProjections[n * M:(n + 1) * M]

        FilteredSignalSDZeroPadded = Allocation.zeros(N180 * NFFT)
        for n in range(N180):
            block = UnfilteredProjectionsZeroPadded[n * NFFT:(n + 1) * NFFT]
            # No normalization: the 1/Nt factor of the discrete transform cancels in the inverse.
            FilteredSignalSDZeroPadded[n * NFFT:(n + 1) * NFFT] = np.real(np.fft.ifft(np.fft.fft(block) * Kernel))

        FilteredSignalSD = Allocation.zeros(N180 * M)
        for n in range(N180):
            FilteredSignalSD[n * M:(n + 1) * M] = FilteredSignalSDZeroPadded[n * NFFT:n * NFFT + M]
        return FilteredSignalSD
