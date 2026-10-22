import math

from CTBasics.Descriptors.ScannerGeometryTypes import ScannerGeometryTypes
from CTBasics.Descriptors.ScanTypes import ScanTypes
from CTBasics.Descriptors.ScanParameter.BaseScanParameter import BaseScanParameter


class Sequence(BaseScanParameter):
    """Container class for sequence scan parameters."""

    def __init__(self, Geometry_in) -> None:
        super().__init__(ScanTypes.Sequence, Geometry_in)

    def Copy(self) -> "Sequence":
        instance = Sequence(self.Geometry.Copy())

        instance.tableincrement = self.tableincrement
        instance.z0focus = self.z0focus
        instance.zEndfocus = self.zEndfocus
        instance.dzfocus = self.dzfocus
        instance.NzScans = self.NzScans

        instance.alpha0 = list(self.alpha0) if isinstance(self.alpha0, list) else self.alpha0
        instance.alphaN = self.alphaN
        instance.N = self.N

        return instance

    def ClearState(self) -> "Sequence":
        self.NzScans = 0

        self.tableincrement = None
        self.z0focus = None
        self.dzfocus = None
        self.zEndfocus = None
        self.alpha0 = []
        self.alphaN = None
        self.N = None

        return self

    def Init(self) -> "Sequence":
        self.ClearState()

        if self.Geometry.BeamType != ScannerGeometryTypes.Fan:
            raise ValueError("Invalid Geometry!")

        betaMax = max(self.Geometry.beta)

        self.NzScans = 2
        self.z0focus = 0

        self.SetTableIncrement(0.5)

        # One start angle per scanned z-position; the fan opening shifts the start by -betaMax.
        self.alpha0 = [-betaMax, -betaMax]
        self.alphaN = 2 * math.pi + 2 * betaMax

        self.N = int(self.alphaN / self.Geometry.Sampling.dalpha)

        return self

    def SetTableIncrement(self, tableincrement_in) -> "Sequence":
        self.tableincrement = tableincrement_in
        self.dzfocus = self.Geometry.c * self.tableincrement

        self.zEndfocus = self.z0focus + (self.NzScans - 1) * self.dzfocus

        return self

    def Load(self, DataDescriptor_in):
        """Reads the scan blocks of a measured sequence acquisition."""
        from CTBasics.RawData.RawDataScanDescriptor import RawDataScanDescriptor

        self.ClearState()

        if not isinstance(DataDescriptor_in, (list, tuple)):
            DataDescriptor_in = [DataDescriptor_in]

        self.NzScans = len(DataDescriptor_in)

        FeedScan = float(RawDataScanDescriptor.Load(DataDescriptor_in[0], "FeedScan")) / 1000

        self.tableincrement = FeedScan / self.Geometry.c
        self.dzfocus = FeedScan

        ScanBlock = DataDescriptor_in[0].ScanDescr.Det[0]

        self.z0focus = ScanBlock.FirstZPosition / 1000
        self.zEndfocus = self.z0focus + (self.NzScans - 1) * self.dzfocus

        self.N = ScanBlock.NoOfReadings
        self.alphaN = ScanBlock.NoOfReadings * self.Geometry.Sampling.dalpha

        self.alpha0 = []
        for ScanIdx in range(1, self.NzScans + 1):
            ScanBlock = DataDescriptor_in[ScanIdx - 1].ScanDescr.Det[0]
            # -pi/2 because the raw-data coordinate system is rotated against the spec.
            self.alpha0.append(ScanBlock.FirstTubeAngle / 1000 / 360 * 2 * math.pi - (math.pi / 2))

        return self
