import math

import numpy as np

from ..ScannerGeometryTypes import ScannerGeometryTypes
from .BaseScannerGeometry import BaseScannerGeometry

_COPIED_PROPERTIES = (
    "RM", "M", "dxi", "xi0", "N2pi", "thetaN_min", "dtheta", "thetaN", "Npar", "theta0",
)


class ParallelBeam(BaseScannerGeometry):
    """Virtual parallel-beam geometry, either stand-alone or rebinned from a scan descriptor."""

    def __init__(self, Descriptor_in=None) -> None:
        super().__init__(ScannerGeometryTypes.Parallel, None)

        if Descriptor_in is None:
            return

        Geometry = Descriptor_in.Geometry

        self.N2pi = Geometry.Sampling.N2pi
        self.dtheta = Geometry.Sampling.dalpha

        if Geometry.BeamType == ScannerGeometryTypes.Fan:
            self.RM = Geometry.RM
            self.M = Geometry.MFan
            self.dxi = 2 * self.RM / (self.M - 1)
            self.xi0 = -self.RM - self.dxi

            betaMax = max(Geometry.beta)

        elif Geometry.BeamType == ScannerGeometryTypes.StaticCT:
            self.RM = Geometry.RM
            self.dxi = Geometry.dDelta * Geometry.RF
            self.xi0 = -Geometry.RD - self.dxi
            self.M = int(2 * Geometry.RD / self.dxi + 1)

            betaMax = Geometry.betaMax

        else:
            raise ValueError("Invalid Beam Type")

        Parameter = Descriptor_in.Parameter

        # The parallel range is the fan range shrunk by the fan opening on both sides.
        self.thetaN = Parameter.alphaN - 2 * betaMax
        self.Npar = int(self.thetaN / self.dtheta)
        # alpha0 is a list for sequence scans, so theta0 stays one start angle per scan.
        self.theta0 = (np.asarray(Parameter.alpha0, dtype=float) + betaMax
                       if isinstance(Parameter.alpha0, (list, tuple, np.ndarray))
                       else Parameter.alpha0 + betaMax)

    def Init(self) -> None:
        self.RM = 250
        self.M = 500
        self.dxi = 2 * self.RM / (self.M - 1)
        self.xi0 = -self.RM - self.dxi

        self.N2pi = 1024
        self.thetaN_min = math.pi
        self.dtheta = 2 * math.pi / self.N2pi

        self.thetaN = 0.0
        self.Npar = 0
        self.theta0 = 0.0

    def Set(self, M_in) -> None:
        self.M = M_in
        self.dxi = 2 * self.RM / (self.M - 1)
        self.xi0 = -self.RM - self.dxi

    def Copy(self) -> "ParallelBeam":
        instance = ParallelBeam()
        instance.BeamType = self.BeamType
        instance.Sampling = self.Sampling.Copy() if self.Sampling is not None else None
        for name in _COPIED_PROPERTIES:
            setattr(instance, name, getattr(self, name))
        return instance
