import math

from CTBasics.Descriptors.ScannerGeometryTypes import ScannerGeometryTypes
from CTBasics.Descriptors.ScanTypes import ScanTypes
from CTBasics.Descriptors.ScanParameter.BaseScanParameter import BaseScanParameter


class Spiral(BaseScanParameter):
    """Container class for spiral scan parameters."""

    def __init__(self, Geometry_in) -> None:
        super().__init__(ScanTypes.Spiral, Geometry_in)

    def Copy(self) -> "Spiral":
        instance = Spiral(self.Geometry.Copy())

        for name in ("z0focus", "zEndfocus", "pitch", "zrot", "alpha0", "alphaN", "N"):
            setattr(instance, name, getattr(self, name))

        return instance

    def Init(self) -> "Spiral":
        self.z0focus = 0
        self.pitch = 0.5

        if self.Geometry.BeamType == ScannerGeometryTypes.Fan:
            betaMax = max(self.Geometry.beta)

            self.alpha0 = -betaMax
            self.alphaN = 2 * math.pi + 2 * betaMax
            self.N = int(self.alphaN / self.Geometry.Sampling.dalpha)

            self.zrot = self.pitch * self.Geometry.c
            self.zEndfocus = self.N / self.Geometry.Sampling.N2pi * self.zrot

        elif self.Geometry.BeamType == ScannerGeometryTypes.StaticCT:
            self.alpha0 = -self.Geometry.betaMax
            self.alphaN = 4 * math.pi + 2 * self.Geometry.betaMax
            self.N = int(self.alphaN / self.Geometry.Sampling.dalpha)

            # c is the real detector width here, not a collimation defined in the isocenter,
            # so it has to be projected back with RF/RFD.
            self.zrot = self.Geometry.RF / self.Geometry.RFD * self.pitch * self.Geometry.c
            self.zEndfocus = self.N / self.Geometry.Sampling.N2pi * self.zrot

        else:
            raise ValueError("Invalid Geometry!")

        return self

    def SetPitch(self, pitch_in) -> "Spiral":
        self.pitch = pitch_in
        self.zrot = self.pitch * self.Geometry.c

        self.SetScanRange(self.z0focus, self.zEndfocus)

        return self

    def SetScanRange(self, zstart_in, zend_in) -> "Spiral":
        self.z0focus = zstart_in
        nRotations = (zend_in - zstart_in) / self.zrot

        self.alphaN = nRotations * 2 * math.pi + 2 * max(self.Geometry.beta)
        self.N = int(self.alphaN / self.Geometry.Sampling.dalpha)

        self.zEndfocus = self.N / self.Geometry.Sampling.N2pi * self.zrot

        return self

    def Load(self, DataDescriptor_in):
        """Reads pitch, scan range and start angle of a measured spiral acquisition."""
        from CTBasics.RawData.RawDataScanDescriptor import RawDataScanDescriptor

        RotTime = DataDescriptor_in.ModeParXML.ModePar.RotTime

        TableDirectionEnum = RawDataScanDescriptor.Load(DataDescriptor_in, "TableDirection")
        TableDirection = +1.0 if "DirOut" in str(TableDirectionEnum) else -1.0

        TableSpeed = float(RawDataScanDescriptor.Load(DataDescriptor_in, "TableSpeed"))

        TableFeed = abs((TableDirection * TableSpeed * RotTime) / 1000000.0)

        self.pitch = TableFeed / self.Geometry.c
        self.zrot = TableFeed

        SpiralScan = DataDescriptor_in.ScanDescr.Det[0]

        self.z0focus = SpiralScan.FirstZPosition / 1000
        self.zEndfocus = SpiralScan.LastZPosition / 1000

        # -pi/2 because the raw-data coordinate system is rotated against the spec.
        self.alpha0 = SpiralScan.FirstTubeAngle / 1000 / 360 * 2 * math.pi - (math.pi / 2)
        self.alphaN = SpiralScan.NoOfReadings * self.Geometry.Sampling.dalpha

        self.N = SpiralScan.NoOfReadings

        return self
