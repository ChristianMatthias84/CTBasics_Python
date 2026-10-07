import math

from ..ScannerGeometryTypes import ScannerGeometryTypes
from ..ScanTypes import ScanTypes

TWO_PI = 2 * math.pi


class Simulation:
    """Simulation volume geometry.

    Either freely parametrized or derived from the CT scan geometry and the spiral /
    sequence path. The volume must cover the simulated collimation, otherwise rays run
    out of the volume abruptly and the line integrals are truncated.
    """

    def __init__(self, ScanDescriptor_in=None, Nxsim=None, Nysim=None, Nzsim=None,
                 dxsim=None, dysim=None, dzsim=None, xcsim=None, ycsim=None, zcsim=None,
                 x0sim=None, y0sim=None, z0sim=None, zStartsim=None, zStopsim=None,
                 ScanType=None) -> None:
        if ScanDescriptor_in is None and Nxsim is None:
            RM = 250
            self.Nxsim = 256
            self.Nysim = 256
            self.Nzsim = 40
            self.dxsim = 2 * RM / self.Nxsim
            self.dysim = 2 * RM / self.Nysim
            self.dzsim = 1.0
            self.xcsim = 0
            self.ycsim = 0
            self.zcsim = 0
            self.x0sim = -RM - self.dxsim / 2.0 + self.xcsim
            self.y0sim = -RM - self.dysim / 2.0 + self.ycsim
            self.z0sim = -self.Nzsim / 2 * self.dzsim - self.dzsim / 2.0 + self.zcsim
            self.zStartsim = 1
            self.zStopsim = self.Nzsim
            self.ScanType = None
            return

        if Nxsim is None:
            BeamType = ScanDescriptor_in.Geometry.BeamType

            if BeamType == ScannerGeometryTypes.StaticCT:
                self.InitiateSimulationFromStaticCTGeometry(ScanDescriptor_in)
            elif BeamType == ScannerGeometryTypes.Fan:
                self.InitiateSimulationFromFanBeamGeometry(ScanDescriptor_in)
            else:
                raise ValueError("Unknown GeometryType!")
            return

        self.Nxsim = Nxsim
        self.Nysim = Nysim
        self.Nzsim = Nzsim
        self.dxsim = dxsim
        self.dysim = dysim
        self.dzsim = dzsim
        self.xcsim = xcsim
        self.ycsim = ycsim
        self.zcsim = zcsim
        self.x0sim = x0sim
        self.y0sim = y0sim
        self.z0sim = z0sim
        self.zStartsim = zStartsim
        self.zStopsim = zStopsim
        self.ScanType = ScanType

    @staticmethod
    def _FirstZ(z0focus):
        return z0focus[0] if isinstance(z0focus, (list, tuple)) else z0focus

    def _ScanRange(self, Parameter):
        zScanRange = 0
        if self.ScanType == ScanTypes.Spiral:
            zScanRange += Parameter.alphaN / TWO_PI * Parameter.zrot
        elif self.ScanType == ScanTypes.Sequence:
            zScanRange += (Parameter.NzScans - 1) * Parameter.dzfocus
        return zScanRange

    def _Finalize(self, RM, zScanRange, zScanRange_Center):
        self.Nzsim = int(zScanRange / self.dzsim / 2) * 2
        self.xcsim = 0
        self.ycsim = 0
        self.zcsim = 0
        self.x0sim = -RM - self.dxsim / 2.0 + self.xcsim
        self.y0sim = -RM - self.dysim / 2.0 + self.ycsim
        self.z0sim = -self.Nzsim / 2 * self.dzsim - self.dzsim / 2.0 + zScanRange_Center + self.zcsim
        self.zStartsim = 1
        self.zStopsim = self.Nzsim

    def InitiateSimulationFromStaticCTGeometry(self, ScanDescriptor_in):
        self.Nxsim = 256
        self.Nysim = 256

        Geometry = ScanDescriptor_in.Geometry
        Parameter = ScanDescriptor_in.Parameter

        RF = Geometry.RF
        RFD = RF + Geometry.RD

        self.dxsim = 2 * Geometry.RM / self.Nxsim
        self.dysim = 2 * Geometry.RM / self.Nysim
        self.dzsim = 1.0
        self.ScanType = Parameter.ScanType

        zScanRange = self._ScanRange(Parameter)
        zScanRange_Center = (self._FirstZ(Parameter.z0focus) + zScanRange / 2
                             + (RF / RFD) * Geometry.soffset - (RF / RFD) * Geometry.c / 2)

        # Worst-case margin of 2*c; in practice less is needed because RM is much smaller.
        zScanRange = zScanRange + (RF / RFD) * 2 * Geometry.c

        self._Finalize(Geometry.RM, zScanRange, zScanRange_Center)
        return self

    def InitiateSimulationFromFanBeamGeometry(self, ScanDescriptor_in):
        self.Nxsim = 256
        self.Nysim = 256

        Geometry = ScanDescriptor_in.Geometry
        Parameter = ScanDescriptor_in.Parameter

        self.dxsim = 2 * Geometry.RM / self.Nxsim
        self.dysim = 2 * Geometry.RM / self.Nysim
        self.dzsim = 1.0
        self.ScanType = Parameter.ScanType

        zScanRange = self._ScanRange(Parameter)
        zScanRange_Center = self._FirstZ(Parameter.z0focus) + zScanRange / 2

        zScanRange = zScanRange + 2 * Geometry.c

        self._Finalize(Geometry.RM, zScanRange, zScanRange_Center)
        return self
