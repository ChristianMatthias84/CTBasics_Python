import numpy as np

from ..Memory.Allocation import Allocation
from ..ReadAndWrite.IOHelperFunctions import IOHelperFunctions
from ..Reconstruction.Backprojection import Backprojection
from ..Reconstruction.Constants import Constants
from ..Reconstruction.Convolution import Convolution
from ..Reconstruction.ForwardProjection import ForwardProjection
from ..Reconstruction.Kernels import Kernels


class MAR:
    """Normalized metal artifact reduction on a reconstructed CT slice."""

    def __init__(self, InputImage, x0, dx, Nx, y0, dy, Ny, theta0, dtheta, N180, xi0, dxi, M):
        self.x0 = x0
        self.dx = dx
        self.Nx = Nx
        self.y0 = y0
        self.dy = dy
        self.Ny = Ny

        # The algorithm works on HU shifted to a positive scale.
        self.Input = np.asarray(InputImage, dtype=float).ravel() + 1000

        self.theta0 = theta0
        self.dtheta = dtheta
        self.N180 = N180
        self.xi0 = xi0
        self.dxi = dxi
        self.M = M

        self.TM = 3000
        self.TB = 1400
        self.TA = 400

    def Apply(self):
        """Runs the nine NMAR steps and writes every intermediate result to the output folder."""
        Projections = ForwardProjection.RayDrivenForwProjParCartesianMultiple(
            self.x0, self.dx, self.Nx, self.y0, self.dy, self.Ny,
            self.Input * Constants.MuW / 1000,
            self.theta0, self.dtheta, self.N180, self.xi0, self.dxi, self.M
        )
        IOHelperFunctions.Save(Projections, "Projections")

        MetalImage = self.ThresholdMetalImage()
        IOHelperFunctions.Save(MetalImage, "MetalImage")

        MetalProjections = ForwardProjection.RayDrivenForwProjParCartesianMultiple(
            self.x0, self.dx, self.Nx, self.y0, self.dy, self.Ny,
            MetalImage * Constants.MuW / 1000,
            self.theta0, self.dtheta, self.N180, self.xi0, self.dxi, self.M
        )
        IOHelperFunctions.Save(MetalProjections, "MetalProjections")

        InterolatedProjections = self.MetalTraceInterpolation(Projections, MetalProjections)
        IOHelperFunctions.Save(InterolatedProjections, "InterolatedProjections")

        Kernel = Kernels.RamLakKernelSD(self.M, self.dxi)
        InterolatedProjections = Convolution.ConvolveSinogram(
            InterolatedProjections, Kernel, self.M, self.N180, self.dxi
        )
        MARResult = Backprojection.PixDrivenBackProjParCartesian(
            self.theta0, self.dtheta, self.N180, 2 * self.N180, self.xi0, self.dxi, self.M,
            InterolatedProjections, self.x0, self.dx, self.Nx, self.y0, self.dy, self.Ny
        )
        MARResult = MARResult / Constants.MuW * 1000
        IOHelperFunctions.Save(MARResult, "MARResult")

        PriorImage = self.CreatePriorImage(MARResult)
        IOHelperFunctions.Save(PriorImage, "PriorImage")

        PriorSinogram = ForwardProjection.RayDrivenForwProjParCartesianMultiple(
            self.x0, self.dx, self.Nx, self.y0, self.dy, self.Ny,
            PriorImage * Constants.MuW / 1000,
            self.theta0, self.dtheta, self.N180, self.xi0, self.dxi, self.M
        )
        IOHelperFunctions.Save(PriorSinogram, "PriorSinogram")

        InterolatedProjections = self.NoramlizedMetalTraceInterpolation(
            Projections, MetalProjections, PriorSinogram
        )
        IOHelperFunctions.Save(InterolatedProjections, "InterolatedProjections")

        InterolatedProjections = Convolution.ConvolveSinogram(
            InterolatedProjections, Kernel, self.M, self.N180, self.dxi
        )
        MARResult = Backprojection.PixDrivenBackProjParCartesian(
            self.theta0, self.dtheta, self.N180, 2 * self.N180, self.xi0, self.dxi, self.M,
            InterolatedProjections, self.x0, self.dx, self.Nx, self.y0, self.dy, self.Ny
        )
        MARResult = MARResult / Constants.MuW * 1000
        IOHelperFunctions.Save(MARResult, "iMARResult")

        return MARResult

    def ThresholdMetalImage(self):
        """Keeps only the voxels above the metal threshold, everything else becomes zero."""
        MetalImage = Allocation.zeros(self.Nx * self.Ny)
        metal = self.Input > self.TM
        MetalImage[metal] = self.Input[metal]
        return MetalImage

    def MetalTraceInterpolation(self, Projections, MetalProjections):
        """Replaces every metal-shadowed channel by a linear ramp between its trace borders."""
        Projections = np.asarray(Projections, dtype=float).ravel()
        MetalProjections = np.asarray(MetalProjections, dtype=float).ravel()
        InterpolatedProjections = Projections.copy()

        Metal = MetalProjections.reshape(self.N180, self.M) > 0
        Rows = Projections.reshape(self.N180, self.M)
        Output = InterpolatedProjections.reshape(self.N180, self.M)

        for n in range(self.N180):
            shadowed = np.flatnonzero(Metal[n])
            if shadowed.size == 0:
                continue

            # Split the shadowed channels into contiguous traces.
            breaks = np.flatnonzero(np.diff(shadowed) > 1)
            starts = np.concatenate(([shadowed[0]], shadowed[breaks + 1]))
            stops = np.concatenate((shadowed[breaks], [shadowed[-1]]))

            for start, stop in zip(starts, stops):
                leftIndex = start - 1
                rightIndex = stop + 1
                if leftIndex < 0 or rightIndex >= self.M:
                    # MATLAB walks past the array end here and aborts with an index error.
                    raise IndexError("Metal trace touches the detector border")

                span = rightIndex - leftIndex
                m = np.arange(start, stop + 1)
                w = (m - leftIndex) / span
                Output[n, start:stop + 1] = (1 - w) * Rows[n, leftIndex] + w * Rows[n, rightIndex]

        return InterpolatedProjections

    def NoramlizedMetalTraceInterpolation(self, Projections, MetalProjections, PriorSinogram):
        """Interpolates in the prior-normalized domain so edges survive the replacement."""
        Projections = np.asarray(Projections, dtype=float).ravel()
        PriorSinogram = np.asarray(PriorSinogram, dtype=float).ravel()

        NormalizedSinogram = (Projections + 1000) / (PriorSinogram + 1000)

        InterpolatedProjections = self.MetalTraceInterpolation(NormalizedSinogram, MetalProjections)

        return InterpolatedProjections * (PriorSinogram + 1000) - 1000

    def CreatePriorImage(self, Input):
        """Flattens soft tissue to water so the prior only keeps bone, metal and air."""
        Input = np.asarray(Input, dtype=float).ravel()
        PriorImage = Input.copy()
        PriorImage[(Input > self.TA) & (Input < self.TB)] = 1000
        return PriorImage
