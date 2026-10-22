import math

from CTBasics.Descriptors.ScannerGeometryTypes import ScannerGeometryTypes
from CTBasics.Descriptors.ScanTypes import ScanTypes

TWO_PI = 2 * math.pi


class Reconstruction:
    """Reconstruction volume geometry.

    Usable stand-alone, derived from a scan descriptor, or additionally tied to the
    simulated volume so that reconstruction and simulation share the in-plane raster.
    """

    def __init__(self, ScanDescriptor_in=None, SimulatedGeometry_in=None) -> None:
        if ScanDescriptor_in is None:
            RM = 250
            self.Nx = 256
            self.Ny = 256
            self.Nz = 20
            self.dx = 2 * RM / self.Nx
            self.dy = 2 * RM / self.Ny
            self.dz = 2.0
            self.xc = 0
            self.yc = 0
            self.zc = 0
            self.x0 = -RM - self.dx / 2.0 + self.xc
            self.y0 = -RM - self.dy / 2.0 + self.yc
            self.z0 = -self.Nz / 2 * self.dz - self.dz / 2.0 + self.zc
            self.ScanType = None
            return

        BeamType = ScanDescriptor_in.Geometry.BeamType

        if BeamType == ScannerGeometryTypes.StaticCT:
            self.InitiateReconstructionFromStaticCTGeometry(ScanDescriptor_in, SimulatedGeometry_in)
        elif BeamType == ScannerGeometryTypes.Fan:
            self.InitiateReconstructionFromFanBeamGeometry(ScanDescriptor_in, SimulatedGeometry_in)
        else:
            raise ValueError("Unknown GeometryType!")

    def _SetInPlaneRaster(self, SimulatedGeometry_in):
        if SimulatedGeometry_in is None:
            self.Nx = 256
            self.Ny = 256
            self.dz = 2.0
        else:
            self.Nx = SimulatedGeometry_in.Nxsim
            self.Ny = SimulatedGeometry_in.Nysim
            self.dz = SimulatedGeometry_in.dzsim

    def _Finalize(self, RM, zScanRange, zScanRange_Center):
        # An even slice count keeps the volume symmetric around the scan center.
        self.Nz = int(zScanRange / self.dz / 2) * 2
        self.dx = 2 * RM / self.Nx
        self.dy = 2 * RM / self.Ny

        self.xc = 0
        self.yc = 0
        self.zc = 0
        self.x0 = -RM - self.dx / 2.0 + self.xc
        self.y0 = -RM - self.dy / 2.0 + self.yc
        self.z0 = -self.Nz / 2 * self.dz - self.dz / 2.0 + zScanRange_Center + self.zc

    @staticmethod
    def _FirstZ(z0focus):
        return z0focus[0] if isinstance(z0focus, (list, tuple)) else z0focus

    def InitiateReconstructionFromStaticCTGeometry(self, ScanDescriptor_in, SimulatedGeometry_in=None):
        self._SetInPlaneRaster(SimulatedGeometry_in)

        Geometry = ScanDescriptor_in.Geometry
        Parameter = ScanDescriptor_in.Parameter

        RF = Geometry.RF
        RFD = RF + Geometry.RD

        self.ScanType = Parameter.ScanType

        zScanRange = 0
        zScanRange_Center = 0

        if self.ScanType == ScanTypes.Spiral:
            zScanRange += Parameter.alphaN / TWO_PI * Parameter.zrot + (RF / RFD) * Geometry.c
            zScanRange_Center = (Parameter.z0focus
                                 + Parameter.alphaN / TWO_PI * Parameter.zrot / 2
                                 + (RF / RFD) * Geometry.soffset - (RF / RFD) * Geometry.c / 2)
        elif self.ScanType == ScanTypes.Sequence:
            zScanRange += (Parameter.NzScans - 1) * Parameter.dzfocus + (RF / RFD) * Geometry.c
            zScanRange_Center = (self._FirstZ(Parameter.z0focus)
                                 + (Parameter.NzScans - 1) * Parameter.dzfocus / 2
                                 + (RF / RFD) * Geometry.soffset - (RF / RFD) * Geometry.c / 2)

        self._Finalize(Geometry.RM, zScanRange, zScanRange_Center)
        return self

    def InitiateReconstructionFromFanBeamGeometry(self, ScanDescriptor_in, SimulatedGeometry_in=None):
        self._SetInPlaneRaster(SimulatedGeometry_in)

        Geometry = ScanDescriptor_in.Geometry
        Parameter = ScanDescriptor_in.Parameter

        self.ScanType = Parameter.ScanType

        zScanRange = 0
        zScanRange_Center = 0

        if self.ScanType == ScanTypes.Spiral:
            zScanRange += Parameter.alphaN / TWO_PI * Parameter.zrot + Geometry.c / 2
            zScanRange_Center = (Parameter.z0focus
                                 + Parameter.alphaN / TWO_PI * Parameter.zrot / 2)
        elif self.ScanType == ScanTypes.Sequence:
            zScanRange += (Parameter.NzScans - 1) * Parameter.dzfocus + Geometry.c
            zScanRange_Center = (self._FirstZ(Parameter.z0focus)
                                 + (Parameter.NzScans - 1) * Parameter.dzfocus / 2)

        self._Finalize(Geometry.RM, zScanRange, zScanRange_Center)
        return self
