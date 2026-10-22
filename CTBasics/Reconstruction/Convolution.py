import numpy as np

from CTBasics.Descriptors.ScanTypes import ScanTypes
from CTBasics.Memory.Allocation import Allocation
from CTBasics.ReadAndWrite.IOHelperFunctions import IOHelperFunctions
from CTBasics.Reconstruction.Rebinning import _CachedFilename


def _ConvolutionMatrix(Kernel, M, start):
    """Toeplitz matrix T with (row @ T)[i] = full convolution of row and kernel at start + i."""
    k = start + np.arange(M)[None, :] - np.arange(M)[:, None]
    valid = (k >= 0) & (k < Kernel.size)
    return np.where(valid, Kernel[np.where(valid, k, 0)], 0.0)


class Convolution:
    """Filtering of rebinned sinograms with a reconstruction kernel."""

    # ==============================
    # Caching Mechanism
    # ==============================
    @staticmethod
    def Apply(Projections, Kernel, Descriptor, ParallelGeometry):
        ConvolutedFilename = Convolution.ConvolutedFilename(Descriptor)

        try:
            result = IOHelperFunctions.Load(ConvolutedFilename)
            print(f"Using cached convoluted data [{ConvolutedFilename}]")
            return result
        except OSError:
            pass

        M = ParallelGeometry.M
        Npar = ParallelGeometry.Npar
        dxi = ParallelGeometry.dxi

        Geometry = Descriptor.Geometry
        Parameter = Descriptor.Parameter
        Ns = Geometry.Ns

        if Parameter.ScanType == ScanTypes.Spiral:
            result = Convolution.Convovle3DSpiralSinogram(Projections, Kernel, M, Npar, dxi, Ns)
        elif Parameter.ScanType == ScanTypes.Sequence:
            result = Convolution.Convovle3DSequenceSinogram(
                Projections, Kernel, M, Npar, dxi, Ns, Parameter.NzScans
            )
        else:
            raise ValueError("Unknown ScanType")

        IOHelperFunctions.Save(result, ConvolutedFilename)
        return result

    @staticmethod
    def ConvolutedFilename(ScanDescriptor):
        return _CachedFilename(ScanDescriptor, "convoluted")

    @staticmethod
    def ConvolveSinogram(Projections2D, Kernel, M, N, dxi):
        Projections2D = np.asarray(Projections2D, dtype=float).ravel()
        Kernel = np.asarray(Kernel, dtype=float).ravel()

        # MATLAB conv(u, v, 'same') keeps the central part the size of u, while numpy's
        # 'same' keeps max(len(u), len(v)). Kernels here are longer than one projection,
        # so the central window has to be cut explicitly.
        start = (Kernel.size - 1) // 2

        # dxi turns the discrete sum into the integral over the detector coordinate.
        ProjectionsConvolved = Allocation.zeros(M * N)
        ProjectionsConvolved[:] = (
            (Projections2D[:M * N].reshape(N, M) @ _ConvolutionMatrix(Kernel, M, start)) * dxi
        ).ravel()
        return ProjectionsConvolved

    @staticmethod
    def _ConvolveRowWise(Projections, Kernel, M, Npar, dxi, Ns):
        """Filter each detector row independently and relinearize to (Npar, Ns, M) order."""
        Projections = np.asarray(Projections, dtype=float).ravel()

        RegroupedProjections = Projections.reshape(Ns * Npar, M)

        RowWiseFilteredProjections = Allocation.zeros(Ns, M * Npar)
        for ns in range(1, Ns + 1):
            subgroup = RegroupedProjections[ns - 1::Ns, :].ravel()
            RowWiseFilteredProjections[ns - 1, :] = Convolution.ConvolveSinogram(
                subgroup, Kernel, M, Npar, dxi
            )

        LinearizedProjections = Allocation.zeros(Npar, Ns * M)
        for n in range(1, Npar + 1):
            LinearizedProjections[n - 1, :] = RowWiseFilteredProjections[:, (n - 1) * M:n * M].ravel()

        return LinearizedProjections.ravel()

    @staticmethod
    def Convovle3DSpiralSinogram(Projections, Kernel, M, Npar, dxi, Ns):
        return Convolution._ConvolveRowWise(Projections, Kernel, M, Npar, dxi, Ns)

    @staticmethod
    def Convovle3DSequenceSinogram(ProjectionsSequence, Kernel, M, Npar, dxi, Ns, NzScans):
        ProjectionsSequence = np.asarray(ProjectionsSequence, dtype=float).ravel()

        Block = M * Ns * Npar
        ProjectionsConvolved = Allocation.zeros(Block * NzScans)
        for nz in range(1, NzScans + 1):
            ProjectionsConvolved[(nz - 1) * Block:nz * Block] = Convolution._ConvolveRowWise(
                ProjectionsSequence[(nz - 1) * Block:nz * Block], Kernel, M, Npar, dxi, Ns
            )
        return ProjectionsConvolved
