"""Reconstruction of measured scanner raw data.

Port of LecturesAndTestScripts/TestScriptScannerData3D.m. MATLAB picks the `.raw` file
with `uigetfile`; here the path is passed in. Reading measured raw data requires the
external reader toolchain, so this script raises transitively when it is unavailable.
"""
import numpy as np

from ..Descriptors.Geometries.ParallelBeam import ParallelBeam
from ..Descriptors.Sampling.FlyingFocalSpot import FlyingFocalSpot
from ..Descriptors.ScanTypes import ScanTypes
from ..Descriptors.Volumes.Reconstruction import Reconstruction
from ..Memory.Allocation import Allocation
from ..Memory.Precision import Precision
from ..RawData.RawDataHelperFunctions import RawDataHelperFunctions
from ..RawData.RawDataScanDescriptor import RawDataScanDescriptor
from ..Reconstruction.Backprojection import Backprojection
from ..Reconstruction.Constants import Constants
from ..Reconstruction.Convolution import Convolution
from ..Reconstruction.Kernels import Kernels
from ..Reconstruction.Rebinning import Rebinning
from .CTRecon101 import _Save


def main(filename, Outputdirectory=None, ShowFigures=False, Reconkernel=Kernels.RamLak,
         ApplyMoireFilter=False, isSpectral=False, FFS=None, EffectiveMuWater=None):
    Precision.Value(32)

    if FFS is None:
        FFS = FlyingFocalSpot("None")

    # =========================================================
    # Rawdata reading and linearization
    # =========================================================

    aScanDescriptor, Readings = RawDataScanDescriptor.LoadFromFile(filename)
    Readings = aScanDescriptor.SetFFS(FFS, Readings)

    RawImage = np.asarray(Readings)[0, :, :].T

    ReadingsLinearized = RawDataHelperFunctions.LinearizeReadings(Readings)
    Allocation.clear("Readings")

    aReconVol = Reconstruction(aScanDescriptor)
    aParGeo = ParallelBeam(aScanDescriptor)

    Geometry = aScanDescriptor.Geometry
    Parameter = aScanDescriptor.Parameter
    aScanType = Parameter.ScanType

    # =========================================================
    # Rebinning
    # =========================================================

    Projections = Rebinning.Apply(ReadingsLinearized, aScanDescriptor, aParGeo)
    Allocation.clear("ReadingsLinearized")

    # =========================================================
    # Filtering
    # =========================================================

    if Reconkernel == Kernels.SheppLogan:
        Kernel = Kernels.SheppLoganKernelSD(aParGeo.M, aParGeo.dxi)
    elif Reconkernel == Kernels.RamLak:
        Kernel = Kernels.RamLakKernelSD(aParGeo.M, aParGeo.dxi)
    else:
        raise ValueError(f"Unknown reconstruction kernel: {Reconkernel}")

    Projections2DFiltered = Convolution.Apply(Projections, Kernel, aScanDescriptor, aParGeo)
    Allocation.clear("Projections")

    # =========================================================
    # Reconstruction
    # =========================================================

    if aScanType == ScanTypes.Sequence:
        ReconstructedImage, Hxyz = Backprojection.SequenceWFBP3DFullyPixelDrivenParallelizedStacked(
            Projections2DFiltered, aScanDescriptor, aParGeo, aReconVol
        )
    else:
        ReconstructedImage, Hxyz = Backprojection.SpiralWFBP3DFullyPixelDrivenParallelizedStacked(
            Projections2DFiltered, aScanDescriptor, aParGeo, aReconVol
        )

    if not isSpectral:
        ReconstructedImageHU = (ReconstructedImage - Constants.MuW) / Constants.MuW * 1000
    else:
        ReconstructedImageHU = (ReconstructedImage - EffectiveMuWater) / EffectiveMuWater * 1000

    if Outputdirectory is not None:
        _Save(Outputdirectory, "ReadingsLinearized", ReadingsLinearized)
        _Save(Outputdirectory, "ProjectiondataParallelBeamRebinned", Projections)
        _Save(Outputdirectory, "Projections2DFiltered", Projections2DFiltered)
        _Save(Outputdirectory, "Hxyz", Hxyz)
        _Save(Outputdirectory, "Reconstruction", ReconstructedImage)
        _Save(Outputdirectory, "ReconstructionHU", ReconstructedImageHU)

    if ShowFigures:
        _ShowFigures(RawImage, Projections, Projections2DFiltered, Kernel,
                     ReconstructedImageHU, Geometry, Parameter, aParGeo, aReconVol, aScanType)

    return {
        "ScanDescriptor": aScanDescriptor,
        "ReconstructionVolume": aReconVol,
        "ParallelGeometry": aParGeo,
        "RawImage": RawImage,
        "Kernel": Kernel,
        "FilteredProjections": Projections2DFiltered,
        "Hxyz": Hxyz,
        "ReconstructedImage": ReconstructedImage,
        "ReconstructedImageHU": ReconstructedImageHU,
    }


def _ShowFigures(RawImage, Projections, Projections2DFiltered, Kernel,
                 ReconstructedImageHU, Geometry, Parameter, aParGeo, aReconVol, aScanType):
    import matplotlib.pyplot as plt

    Views = aParGeo.Npar * Parameter.NzScans if aScanType == ScanTypes.Sequence else aParGeo.Npar
    Columns = aParGeo.M * Geometry.Ns

    def Show(image, title):
        plt.figure()
        plt.imshow(image, cmap="gray", aspect="auto")
        plt.title(title)

    Show(RawImage, "First reading")
    Show(np.asarray(Projections).reshape(Views, Columns), "Rebinned projections")

    plt.figure()
    plt.plot(Kernel)
    plt.title("Reconstruction kernel")

    Show(np.asarray(Projections2DFiltered).reshape(Views, Columns), "Filtered projections")
    Show(ReconstructedImageHU.reshape(aReconVol.Nz * aReconVol.Ny, aReconVol.Nx),
         "Reconstruction [HU]")
    plt.show()
