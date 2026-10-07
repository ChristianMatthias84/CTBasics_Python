import math

from ..ScannerGeometryTypes import ScannerGeometryTypes
from ..Sampling.AzimuthalSampling import AzimuthalSampling
from .BaseScannerGeometry import BaseScannerGeometry

_COPIED_PROPERTIES = (
    "RF", "RD", "RFD", "RC", "RM", "c", "Ns", "ds", "s0",
    "zCollimatorUpper", "zCollimatorLower", "thicknessInZLowerCollimator",
    "thicknessInZLightAbsorber", "additionalShiftDetector", "zdetectorLengthPhysical",
    "collimatorOpeningDisplacementDetector", "collimatorOpeningDisplacementSource",
    "soffset", "soffsetPhysical", "MDelta", "dDelta", "delta0", "MbetaDash",
    "dbetaDash", "betaDash0", "arbitraryLimitangle", "betaMax",
)


class StaticCTBeam(BaseScannerGeometry):
    def __init__(self) -> None:
        super().__init__(ScannerGeometryTypes.StaticCT, AzimuthalSampling(1470))

    def Init(self) -> None:
        self.RF = 1172.6 / 2
        self.RD = 850 / 2
        self.RFD = self.RF + self.RD
        self.RC = self.RD - 11.5
        self.MDelta = 2428
        self.dDelta = 2 * math.pi / self.MDelta
        self.delta0 = 0
        self.Ns = 16
        self.ds = 1.2858
        self.zdetectorLengthPhysical = 20.440
        self.c = self.Ns * self.ds
        self.thicknessInZLowerCollimator = 2.5
        self.thicknessInZLightAbsorber = 4.8
        self.additionalShiftDetector = 0.0
        self.zCollimatorLower = -7.18
        self.soffsetPhysical = -14.48
        self.zCollimatorUpper = -1.39
        self.soffset = self.soffsetPhysical - self.zdetectorLengthPhysical / 2 + self.Ns * self.ds / 2
        self.s0 = self.soffset - self.Ns * self.ds - self.ds / 2
        self.collimatorOpeningDisplacementDetector = 0.0
        self.collimatorOpeningDisplacementSource = 0.0
        self.MbetaDash = 2428
        self.dbetaDash = 2 * math.pi / self.MbetaDash
        self.betaDash0 = -math.pi
        self.arbitraryLimitangle = math.pi / 2
        # Largest fan angle still intersecting the measurement field for the chosen limit angle.
        self.betaMax = math.atan(
            math.sin(self.arbitraryLimitangle) * self.RD
            / (math.cos(self.arbitraryLimitangle) * self.RD + self.RF)
        )
        self.RM = math.sin(self.betaMax) * self.RF

    def Copy(self) -> "StaticCTBeam":
        instance = StaticCTBeam()
        instance.BeamType = self.BeamType
        instance.Sampling = self.Sampling.Copy()
        for name in _COPIED_PROPERTIES:
            setattr(instance, name, getattr(self, name))
        return instance
