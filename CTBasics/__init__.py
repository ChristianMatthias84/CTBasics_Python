"""CTBasics - Python port of the MATLAB CTBasics CT physics framework.

The package layout mirrors the MATLAB source tree one-to-one. Importing this package
re-exports the public classes of every domain, which is the Python equivalent of the
`addpath(genpath(...))` performed by `init_CTBasics.m`.
"""

from .Apps import MAR
from .Descriptors import (
    AzimuthalSampling,
    BaseScanParameter,
    BaseScannerGeometry,
    FanBeam,
    FlyingFocalSpot,
    FocalSpotSimulationStaticCT,
    ParallelBeam,
    ScanDescriptor,
    ScannerGeometryTypes,
    ScanTypes,
    Sequence,
    Spiral,
    StaticCTBeam,
)
from .ImageManipulation import ImageManipulation
from .Memory import Allocation, Precision, TypeCast
from .MultiCore import CPU, Parallelization, RAM
from .Preprocessing import WBHC
from .RawData import RawDataHelperFunctions, RawDataScanDescriptor, ReadRawDataFunctions
from .ReadAndWrite import IOHelperFunctions
from .Reconstruction import (
    Backprojection,
    Constants,
    Convolution,
    ForwardProjection,
    Kernels,
    MoireFilter,
    Rebinning,
)
from .Scripts import ColliScript
from .Simulation import (
    CreatePhantom,
    MaterialProperties,
    Materials,
    Noise,
    Spectra,
    SpectralProperties,
)
from .Visualization import VisualizationTests

__all__ = [
    "Allocation",
    "AzimuthalSampling",
    "Backprojection",
    "BaseScanParameter",
    "BaseScannerGeometry",
    "CPU",
    "ColliScript",
    "Constants",
    "Convolution",
    "CreatePhantom",
    "FanBeam",
    "FlyingFocalSpot",
    "FocalSpotSimulationStaticCT",
    "ForwardProjection",
    "IOHelperFunctions",
    "ImageManipulation",
    "Kernels",
    "MAR",
    "MaterialProperties",
    "Materials",
    "MoireFilter",
    "Noise",
    "Parallelization",
    "ParallelBeam",
    "Precision",
    "RAM",
    "RawDataHelperFunctions",
    "RawDataScanDescriptor",
    "ReadRawDataFunctions",
    "Rebinning",
    "ScanDescriptor",
    "ScanTypes",
    "ScannerGeometryTypes",
    "Sequence",
    "Spectra",
    "SpectralProperties",
    "Spiral",
    "StaticCTBeam",
    "TypeCast",
    "VisualizationTests",
    "WBHC",
]
