import numpy as np

from CTBasics.Memory.Allocation import Allocation
from CTBasics.Memory.Precision import Precision
from CTBasics.MultiCore.CPU import CPU
from CTBasics.MultiCore.RAM import RAM

_BYTES = {"double": 8, "single": 4}


def _fix(value):
    return int(value)


class SpiralStacks:
    """Splits a rebinned spiral sinogram into z-stacks that can be backprojected in parallel."""

    def __init__(self, FanBeam_in, Rebinning_in, Container_in):
        self.Geometry = FanBeam_in.Geometry
        self.Parameter = FanBeam_in.Parameter

        self.Rebinning = Rebinning_in
        self.Container = Container_in

        self.StackSizes = None

        self.Init()

    def Apply(self, Sino):
        Sino = np.asarray(Sino, dtype=float).ravel()

        N = self.Rebinning.Npar
        M = self.Rebinning.M
        Ns = self.Geometry.Ns

        theta0 = self.Rebinning.theta0
        dtheta = self.Rebinning.dtheta

        NrzStacks = len(self.StackSizes)

        Stacks = [None] * NrzStacks
        NSino = Allocation.zeros(NrzStacks)
        theta = Allocation.zeros(NrzStacks)

        _, FirstReadingStack, FinalReadingStack = self.GetStacks()

        for i in range(1, NrzStacks + 1):
            readingstart = FirstReadingStack[i - 1]
            readingend = FinalReadingStack[i - 1]

            if readingstart < N and readingend > 1:
                readingstart = max(readingstart, 1)
                readingend = min(readingend, N)

                Stacks[i - 1] = Sino[(readingstart - 1) * M * Ns:readingend * M * Ns]

                NSino[i - 1] = readingend - readingstart + 1
                theta[i - 1] = theta0 + (readingstart - 1) * dtheta

        return Stacks, NSino, theta

    def GetStacks(self):
        RF = self.Geometry.RF
        ds = self.Geometry.ds
        Ns = self.Geometry.Ns

        z0focus = self.Parameter.z0focus
        pitch = self.Parameter.pitch
        alpha0 = self.Parameter.alpha0

        theta0 = self.Rebinning.theta0
        N2pi = self.Rebinning.N2pi

        dx = self.Container.dx
        Nx = self.Container.Nx
        dz = self.Container.dz
        z0 = self.Container.z0

        zRot = Ns * ds * pitch

        z0FocusParallelCenterRay = z0focus + (theta0 - alpha0) * zRot / (2 * np.pi)

        # Number of parallel readings that can illuminate one plane, widened by the FOV radius.
        Na = (RF + Nx * dx / 2) / RF * N2pi / pitch

        NrzStacks = len(self.StackSizes)

        z0stack = Allocation.zeros(NrzStacks)
        FirstReadingStack = np.zeros(NrzStacks, dtype=np.int64)
        FinalReadingStack = np.zeros(NrzStacks, dtype=np.int64)

        for i in range(1, NrzStacks + 1):
            z0stack[i - 1] = z0 + float(np.sum(self.StackSizes[:i - 1])) * dz

            zstartStack = z0stack[i - 1] + 1 * dz
            zendStack = z0stack[i - 1] + self.StackSizes[i - 1] * dz

            rc_startslice = (zstartStack - z0FocusParallelCenterRay) / zRot * N2pi
            rc_endslide = (zendStack - z0FocusParallelCenterRay) / zRot * N2pi

            FirstReadingStack[i - 1] = _fix(rc_startslice - Na / 2)
            FinalReadingStack[i - 1] = _fix(rc_endslide + Na / 2) + 1

        return z0stack, FirstReadingStack, FinalReadingStack

    def Init(self, workload=0.8):
        """Picks the largest stack size whose sinogram slices still fit into the free RAM share."""
        free = RAM.GetFree()

        BestSize = 1
        Hardware = CPU()

        MaxStackSize = _fix(self.Container.Nz / Hardware.PhysicalCores)

        for StackSize in range(1, MaxStackSize + 1):
            self.SetSize(StackSize)

            if self.GetRequiredBytes() < workload * free:
                BestSize = StackSize

        self.SetSize(BestSize)
        return self

    def SetSize(self, size):
        """Distributes the division remainder over the leading stacks to avoid a tail effect."""
        FullStacks = _fix(self.Container.Nz / size)
        LastStacks = self.Container.Nz - FullStacks * size

        self.StackSizes = np.full(FullStacks, size, dtype=np.int64)

        for i in range(1, LastStacks + 1):
            self.StackSizes[i - 1] += 1

        return self

    def GetRequiredBytes(self):
        bytes_per_value = _BYTES[Precision.Value()]

        ItemSize = self.Geometry.Ns * self.Rebinning.M

        _, FirstReadingStack, FinalReadingStack = self.GetStacks()

        TotalBytes = 0
        for i in range(1, len(self.StackSizes) + 1):
            TotalItems = FinalReadingStack[i - 1] - FirstReadingStack[i - 1] + 1
            TotalBytes += TotalItems * ItemSize * bytes_per_value

        return TotalBytes
