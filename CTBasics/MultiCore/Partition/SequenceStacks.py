import numpy as np

from ...Memory.Allocation import Allocation
from ...Memory.Precision import Precision
from ..CPU import CPU
from ..RAM import RAM

_BYTES = {"double": 8, "single": 4}


def _fix(value):
    return int(value)


class SequenceStacks:
    """Groups the scans of a sequence acquisition into z-stacks for parallel reconstruction."""

    def __init__(self, FanBeam_in, Rebinning_in, Container_in):
        self.Geometry = FanBeam_in.Geometry
        self.Parameter = FanBeam_in.Parameter

        self.Rebinning = Rebinning_in
        self.Container = Container_in

        self.StackSizes = None

        self.Init()

    def Apply(self, Sino):
        Sino = np.asarray(Sino, dtype=float).ravel()

        M = self.Rebinning.M
        N = self.Rebinning.Npar
        Ns = self.Geometry.Ns

        z0focus = self.Parameter.z0focus
        dzfocus = self.Parameter.dzfocus

        NrzStacks = len(self.StackSizes)

        SinoStacks = [None] * NrzStacks
        z0focousStacks = [None] * NrzStacks

        _, Stacks = self.GetStacks()

        for i in range(1, NrzStacks + 1):
            selectedScans = Stacks[i - 1]

            if len(selectedScans) > 0:
                SinoStacks[i - 1] = Sino[(selectedScans[0] - 1) * M * Ns * N:selectedScans[-1] * M * Ns * N]
                z0focousStacks[i - 1] = z0focus + (selectedScans[0] - 1) * dzfocus

        return SinoStacks, z0focousStacks

    def GetStacks(self):
        RF = self.Geometry.RF
        ds = self.Geometry.ds
        Ns = self.Geometry.Ns

        z0focus = self.Parameter.z0focus
        dzfocus = self.Parameter.dzfocus
        NzScans = self.Parameter.NzScans

        dx = self.Container.dx
        Nx = self.Container.Nx
        z0 = self.Container.z0
        dz = self.Container.dz

        NrzStacks = len(self.StackSizes)

        # Maximum focus-to-slice distance that still contributes: 1/2 * c * (RF + RFoV) / RF.
        delta_zfocusmax = 0.5 * Ns * ds * (RF + Nx * dx / 2) / RF

        z0stack = [None] * NrzStacks
        Scans = [None] * NrzStacks

        for i in range(1, NrzStacks + 1):
            z0stack[i - 1] = z0 + float(np.sum(self.StackSizes[:i - 1])) * dz

            zstartStack = z0stack[i - 1] + 1 * dz
            zendStack = z0stack[i - 1] + self.StackSizes[i - 1] * dz

            selectedScans = []
            for ScanIdx in range(1, NzScans + 1):
                znScan = z0focus + (ScanIdx - 1) * dzfocus

                if abs(znScan - zstartStack) < delta_zfocusmax or abs(znScan - zendStack) < delta_zfocusmax:
                    selectedScans.append(ScanIdx)

            Scans[i - 1] = selectedScans

        return z0stack, Scans

    def Init(self, workload=0.8):
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

        ItemSize = self.Rebinning.M * self.Geometry.Ns * self.Rebinning.Npar

        _, Stacks = self.GetStacks()

        TotalBytes = 0
        for i in range(1, len(self.StackSizes) + 1):
            selectedScans = Stacks[i - 1]

            if len(selectedScans) > 0:
                TotalItems = selectedScans[-1] - selectedScans[0] + 1
                TotalBytes += TotalItems * ItemSize * bytes_per_value

        return TotalBytes
