from enum import Enum


class ScannerGeometryTypes(Enum):
    Fan = "fan"
    Parallel = "parallel"
    StaticCT = "static_ct"


class ScanTypes(Enum):
    Spiral = "spiral"
    Sequence = "sequence"
