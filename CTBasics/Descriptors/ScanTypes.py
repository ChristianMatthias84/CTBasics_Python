from enum import Enum


class ScanTypes(str, Enum):
    """Mirror of the MATLAB constant-property class ScanTypes."""

    Spiral = "Spiral"
    Sequence = "Sequence"
