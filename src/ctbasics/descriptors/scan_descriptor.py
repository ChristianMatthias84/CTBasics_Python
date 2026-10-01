from __future__ import annotations

from dataclasses import dataclass, field
from copy import deepcopy

from .enums import ScanTypes, ScannerGeometryTypes


@dataclass
class BaseScannerGeometry:
    BeamType: ScannerGeometryTypes = ScannerGeometryTypes.Fan
    Sampling: object = None
    MDelta: int = 8
    Ns: int = 4
    dDelta: float = 0.7853981633974483
    ds: float = 1.0
    soffset: float = -2.5
    s0: float = -6.0


@dataclass
class FanBeam(BaseScannerGeometry):
    BeamType: ScannerGeometryTypes = ScannerGeometryTypes.Fan


@dataclass
class ParallelBeam(BaseScannerGeometry):
    BeamType: ScannerGeometryTypes = ScannerGeometryTypes.Parallel


@dataclass
class StaticCTBeam(BaseScannerGeometry):
    BeamType: ScannerGeometryTypes = ScannerGeometryTypes.StaticCT


@dataclass
class BaseScanParameter:
    ScanType: ScanTypes = ScanTypes.Spiral
    Geometry: object = None


@dataclass
class Spiral(BaseScanParameter):
    ScanType: ScanTypes = ScanTypes.Spiral


@dataclass
class Sequence(BaseScanParameter):
    ScanType: ScanTypes = ScanTypes.Sequence


@dataclass
class ScanDescriptor:
    Geometry: BaseScannerGeometry = field(default_factory=FanBeam)
    Parameter: BaseScanParameter = field(default_factory=Spiral)

    @classmethod
    def create(cls, beam_type: ScannerGeometryTypes, scan_type: ScanTypes):
        if beam_type == ScannerGeometryTypes.Fan:
            geometry = FanBeam()
            geometry.Sampling = type("Sampling", (), {"N2pi": 512})()
        elif beam_type == ScannerGeometryTypes.Parallel:
            geometry = ParallelBeam()
            geometry.Sampling = type("Sampling", (), {"N2pi": 512})()
        elif beam_type == ScannerGeometryTypes.StaticCT:
            geometry = StaticCTBeam()
            geometry.Sampling = type("Sampling", (), {"N2pi": 512})()
            geometry.MDelta = 8
            geometry.Ns = 4
            geometry.dDelta = 2.0 * 3.141592653589793 / geometry.MDelta
            geometry.ds = 1.0
            geometry.soffset = -2.5
            geometry.s0 = -2.5 - geometry.Ns * geometry.ds - geometry.ds / 2.0
        else:
            geometry = BaseScannerGeometry()
            geometry.Sampling = type("Sampling", (), {"N2pi": 512})()

        if scan_type == ScanTypes.Spiral:
            parameter = Spiral(Geometry=geometry)
        else:
            parameter = Sequence(Geometry=geometry)
        parameter.Geometry = geometry
        return cls(Geometry=geometry, Parameter=parameter)

    def copy(self):
        copied_geometry = deepcopy(self.Geometry)
        copied_parameter = deepcopy(self.Parameter)
        copied_parameter.Geometry = copied_geometry
        return ScanDescriptor(Geometry=copied_geometry, Parameter=copied_parameter)
