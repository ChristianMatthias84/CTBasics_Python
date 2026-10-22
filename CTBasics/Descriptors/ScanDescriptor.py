from pathlib import Path

from CTBasics.Descriptors.Geometries.FanBeam import FanBeam
from CTBasics.Descriptors.Geometries.ParallelBeam import ParallelBeam
from CTBasics.Descriptors.Geometries.StaticCTBeam import StaticCTBeam
from CTBasics.Descriptors.Sampling.AzimuthalSampling import AzimuthalSampling
from CTBasics.Descriptors.Sampling.FlyingFocalSpot import FlyingFocalSpot
from CTBasics.Descriptors.ScannerGeometryTypes import ScannerGeometryTypes
from CTBasics.Descriptors.ScanParameter.Sequence import Sequence
from CTBasics.Descriptors.ScanParameter.Spiral import Spiral
from CTBasics.Descriptors.ScanTypes import ScanTypes


class ScanDescriptor:
    """Describes the scan geometry and the scan parameter."""

    def __init__(self, Geometry, Parameter, Filename: str = "[virtual]") -> None:
        self.Geometry = Geometry
        self.Parameter = Parameter
        self.Filename = Filename

    def Copy(self) -> "ScanDescriptor":
        GeometryCopy = self.Geometry.Copy()
        ParameterCopy = self.Parameter.Copy()
        # The parameter must reference the copied geometry, not a second independent copy.
        ParameterCopy.Geometry = GeometryCopy
        return ScanDescriptor(GeometryCopy, ParameterCopy, self.Filename)

    def Get(self, Readings_in, rho):
        """Restricts a reading set to one flying-focal-spot sub-sampling and adapts the geometry.

        Every rho-th reading belongs to focal spot `rho`, so the azimuthal sampling drops by
        the number of focal spots and the focus moves by the spot-specific radial/angular offset.
        """
        ScanDescriptorRho = self.Copy()

        TotalFocalSpots = FlyingFocalSpot.GetSize(self.Geometry.FFS)

        if TotalFocalSpots == 1:
            return ScanDescriptorRho, Readings_in

        file = Path(ScanDescriptorRho.Filename)
        ScanDescriptorRho.Filename = f"{file.stem}_rho_[{rho}]{file.suffix}"

        ReadingsRho = Readings_in[rho::TotalFocalSpots]

        ScalingFactor = 1 / TotalFocalSpots

        ScanDescriptorRho.Geometry.Sampling = AzimuthalSampling(ScalingFactor * self.Geometry.Sampling.N2pi)
        ScanDescriptorRho.Parameter.N = ScalingFactor * ScanDescriptorRho.Parameter.N

        Delta_R, Delta_Phi = ScanDescriptorRho.Geometry.FFS.GetDelta(ScanDescriptorRho.Geometry, rho)

        ScanDescriptorRho.Geometry.RF += Delta_R
        ScanDescriptorRho.Parameter.alpha0 += Delta_Phi

        return ScanDescriptorRho, ReadingsRho

    def Load(self, DataDescriptor_in):
        self.Geometry.Load(DataDescriptor_in)
        self.Parameter.Load(DataDescriptor_in)
        return self

    @staticmethod
    def Create(BeamType_in, ScanType_in) -> "ScanDescriptor":
        if BeamType_in == ScannerGeometryTypes.Fan:
            Geometry = FanBeam()
        elif BeamType_in == ScannerGeometryTypes.Parallel:
            Geometry = ParallelBeam()
        elif BeamType_in == ScannerGeometryTypes.StaticCT:
            Geometry = StaticCTBeam()
        else:
            raise ValueError("Invalid Beam-Type!")

        if ScanType_in == ScanTypes.Spiral:
            Parameter = Spiral(Geometry)
        elif ScanType_in == ScanTypes.Sequence:
            Parameter = Sequence(Geometry)
        else:
            raise ValueError("Invalid Scan-Type!")

        return ScanDescriptor(Geometry, Parameter)
