import math

import numpy as np

from ..ScannerGeometryTypes import ScannerGeometryTypes
from ..Sampling.AzimuthalSampling import AzimuthalSampling
from ..Sampling.FlyingFocalSpot import FlyingFocalSpot
from .BaseScannerGeometry import BaseScannerGeometry


class FanBeam(BaseScannerGeometry):
    def __init__(self) -> None:
        super().__init__(ScannerGeometryTypes.Fan, AzimuthalSampling(512))

    def Init(self) -> None:
        self.RF = 595.0
        self.RD = 490.6
        self.ds = 0.6
        self.Ns = 32
        self.MFan = 500
        betaMax = 24.8 / 180.0 * math.pi
        self.gamma = 8
        self.FFS = FlyingFocalSpot("None")
        self.c = self.Ns * self.ds
        self.s0 = -self.Ns / 2.0 * self.ds - self.ds / 2.0
        self.RM = math.sin(betaMax) * self.RF
        dbeta = -2 * betaMax / (self.MFan - 1)
        # beta runs from +betaMax down to -betaMax, i.e. detector element 1 is at the top.
        self.beta = betaMax + dbeta * np.arange(self.MFan)

    def Copy(self) -> "FanBeam":
        instance = FanBeam()
        instance.BeamType = self.BeamType
        instance.Sampling = self.Sampling.Copy()
        instance.RF = self.RF
        instance.RD = self.RD
        instance.RM = self.RM
        instance.c = self.c
        instance.Ns = self.Ns
        instance.ds = self.ds
        instance.s0 = self.s0
        instance.MFan = self.MFan
        instance.beta = np.array(self.beta, copy=True)
        instance.gamma = self.gamma
        instance.FFS = self.FFS.Copy() if self.FFS is not None else None
        return instance

    def Load(self, DataDescriptor_in):
        """Fills the geometry from the descriptors of a measured raw-data file."""
        from ...RawData.RawDataScanDescriptor import RawDataScanDescriptor
        from ...RawData.ReadRawDataFunctions import ReadRawDataFunctions

        Scans = len(DataDescriptor_in) if isinstance(DataDescriptor_in, (list, tuple)) else 1
        DataDescriptor = DataDescriptor_in[0] if Scans > 1 else DataDescriptor_in

        SystemType = DataDescriptor.ModeParXML.Type.ModelType

        Geometry = ReadRawDataFunctions.LoadSystemGeometryXML(SystemType)
        MeasurementSystem = Geometry.ScannerGeometry.MeasurementSystem
        if isinstance(MeasurementSystem, list):
            MeasurementSystem = MeasurementSystem[0]

        self.FFS = FlyingFocalSpot(RawDataScanDescriptor.Load(DataDescriptor, "FlyingFocalSpot"))

        self.RF = float(MeasurementSystem.RadiusFocusPath.Text)
        self.RD = float(MeasurementSystem.RadiusFocusDetector.Text) - self.RF

        self.gamma = float(MeasurementSystem.AnodeAngle.Text)

        self.MFan = int(np.atleast_1d(DataDescriptor.ScanDescr.Type.ChnNum)[0])

        self.beta = ReadRawDataFunctions.LoadBetaTable(SystemType).data
        self.RM = math.sin(self.beta[-1]) * self.RF

        self.Ns = int(DataDescriptor.ScanDescr.Type.Slices) + 1
        self.ds = DataDescriptor.ModeParXML.ModePar.SliceWidthDMS / 1000

        self.c = self.Ns * self.ds

        # One rotation holds as many readings as integration periods fit into the rotation time.
        N2pi = round(DataDescriptor.ModeParXML.ModePar.RotTime
                     / (DataDescriptor.ModeParXML.ModePar.IntegrationTime / 1000 / 1000))

        self.Sampling = AzimuthalSampling(N2pi)

        return self
