"""Reconstruct every flying focal spot of a measured scan separately and fuse the results.

Port of LecturesAndTestScripts/TestFuseFFS.m. MATLAB opens a file dialog; here the raw
data file is passed in. Reading measured raw data needs the external reader toolchain,
so `RawDataScanDescriptor.LoadFromFile` raises until that is available.
"""
import numpy as np

from CTBasics.Descriptors.Geometries.ParallelBeam import ParallelBeam
from CTBasics.Descriptors.Sampling.FlyingFocalSpot import FlyingFocalSpot
from CTBasics.Descriptors.ScanTypes import ScanTypes
from CTBasics.Descriptors.Volumes.Reconstruction import Reconstruction
from CTBasics.Memory.Precision import Precision
from CTBasics.RawData.RawDataHelperFunctions import RawDataHelperFunctions
from CTBasics.RawData.RawDataScanDescriptor import RawDataScanDescriptor
from CTBasics.ReadAndWrite.IOHelperFunctions import IOHelperFunctions
from CTBasics.Reconstruction.Backprojection import Backprojection
from CTBasics.Reconstruction.Constants import Constants
from CTBasics.Reconstruction.Convolution import Convolution
from CTBasics.Reconstruction.Kernels import Kernels
from CTBasics.Reconstruction.Rebinning import Rebinning


def main(filename, Reconkernel=Kernels.RamLak, ApplyBetaOffset=False, isSpectral=False,
         EffectiveMuWater=None):
    Precision.Value(32)

    aIOHelper = IOHelperFunctions()

    aScanDescriptor, TotalReadings = RawDataScanDescriptor.LoadFromFile(filename)

    from pathlib import Path

    stem = Path(aScanDescriptor.Filename).stem
    extension = Path(aScanDescriptor.Filename).suffix

    if ApplyBetaOffset:
        name = FlyingFocalSpot.GetName(aScanDescriptor.Geometry.FFS)
        if name in ("1x1", "1x2"):
            beta_offset = 2.250
        elif name == "1p1":
            beta_offset = 2.000
        elif name in ("2x1", "2x2"):
            beta_offset = 2.125
        else:
            beta_offset = 0.0

        delta_beta = np.mean(aScanDescriptor.Geometry.beta)
        aScanDescriptor.Geometry.beta = aScanDescriptor.Geometry.beta - beta_offset * delta_beta
        aScanDescriptor.Filename = f"{stem}_beta_shift_[{beta_offset}]{extension}"
        stem = Path(aScanDescriptor.Filename).stem

    TotalFocalSpots = FlyingFocalSpot.GetSize(aScanDescriptor.Geometry.FFS)

    for rho in range(1, TotalFocalSpots + 1):
        print(f"Processing: FFS ({rho}/{TotalFocalSpots})")

        aScanDescriptorRho, ReadingsRho = aScanDescriptor.Get(TotalReadings, rho)

        ReadingsLinearized = RawDataHelperFunctions.LinearizeReadings(ReadingsRho)
        aReconVol = Reconstruction(aScanDescriptorRho)
        aParGeo = ParallelBeam(aScanDescriptorRho)
        aScanType = aScanDescriptorRho.Parameter.ScanType

        Projections = Rebinning.Apply(ReadingsLinearized, aScanDescriptorRho, aParGeo)

        if Reconkernel == Kernels.SheppLogan:
            Kernel = Kernels.SheppLoganKernelSD(aParGeo.M, aParGeo.dxi)
        elif Reconkernel == Kernels.RamLak:
            Kernel = Kernels.RamLakKernelSD(aParGeo.M, aParGeo.dxi)
        else:
            raise ValueError(f"Unknown reconstruction kernel: {Reconkernel}")

        Projections2DFiltered = Convolution.Apply(
            Projections, Kernel, aScanDescriptorRho, aParGeo
        )

        if aScanType == ScanTypes.Sequence:
            ReconstructedImage, Hxyz = Backprojection.SequenceWFBP3DFullyPixelDriven(
                Projections2DFiltered, aScanDescriptorRho, aParGeo, aReconVol
            )
        else:
            ReconstructedImage, Hxyz = Backprojection.SpiralWFBP3DFullyPixelDriven(
                Projections2DFiltered, aScanDescriptorRho, aParGeo, aReconVol
            )

        if not isSpectral:
            ReconstructedImageHU = (ReconstructedImage - Constants.MuW) / Constants.MuW * 1000
        else:
            ReconstructedImageHU = (ReconstructedImage - EffectiveMuWater) / EffectiveMuWater * 1000

        aIOHelper.Save(ReconstructedImageHU, f"{stem}_rho_[{rho}]_HU.bin")

    # Fusing the focal spot sub-reconstructions is a plain average.
    ReconConfig = Reconstruction(aScanDescriptor)
    TotalResult = np.zeros((ReconConfig.Nx, ReconConfig.Ny, ReconConfig.Nz))

    for rho in range(1, TotalFocalSpots + 1):
        Deserialize = aIOHelper.Load(f"{stem}_rho_[{rho}]_HU.bin")
        Reco_Rho = np.asarray(Deserialize).reshape(
            ReconConfig.Nx, ReconConfig.Ny, ReconConfig.Nz
        )
        TotalResult = TotalResult + (1.0 / TotalFocalSpots) * Reco_Rho

    aIOHelper.Save(TotalResult, f"{stem}_FFS_fused_HU.bin")

    return TotalResult


if __name__ == "__main__":
    import sys

    main(sys.argv[1])
