from enum import Enum


class ScannerGeometryTypes(str, Enum):
    """Mirror of the MATLAB constant-property class ScannerGeometryTypes."""

    Fan = "Fan"
    Parallel = "Parallel"
    StaticCT = "StaticCT"
