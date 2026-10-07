"""3D static CT simulation and reconstruction.

Port of Projects/Static CT/StaticCT3D.m.
"""
import numpy as np

from ...Descriptors.Geometries.ParallelBeam import ParallelBeam
from ...Descriptors.ScanDescriptor import ScanDescriptor
from ...Descriptors.ScannerGeometryTypes import ScannerGeometryTypes
from ...Descriptors.ScanTypes import ScanTypes
from ...Descriptors.Volumes.Reconstruction import Reconstruction
from ...Descriptors.Volumes.Simulation import Simulation
from ...LecturesAndTestScripts.CTRecon101 import _Save
from ...Reconstruction.Backprojection import Backprojection
from ...Reconstruction.Constants import Constants
from ...Reconstruction.Convolution import Convolution
from ...Reconstruction.ForwardProjection import ForwardProjection
from ...Reconstruction.Kernels import Kernels
from ...Reconstruction.Rebinning import Rebinning
from ...Simulation.CreatePhantom import CreatePhantom


def main(aScanType=ScanTypes.Spiral, Outputdirectory=None, ShowFigures=False,
         Reconkernel=Kernels.RamLak, FFTConvolution=False,
         SelectedPhantom="SimpleEllipseWithHips3D", EffectiveMuWater=None,
         Descriptor=None, SimVol=None, ReconVol=None):
    if Descriptor is None:
        Descriptor = ScanDescriptor.Create(ScannerGeometryTypes.StaticCT, aScanType)

    aParGeo = ParallelBeam(Descriptor)
    Geometry = Descriptor.Geometry
    Parameter = Descriptor.Parameter

    aSimVol = Simulation(Descriptor) if SimVol is None else SimVol
    aReconVol = Reconstruction(Descriptor, aSimVol) if ReconVol is None else ReconVol

    # =========================================================
    # Phantom
    # =========================================================

    Voxelphantom, isSpectral = CreatePhantom.CreateVoxelPhantom3D(SelectedPhantom, aSimVol)

    # dFVec = [dzF, dRF, dxyF]
    CollimationOfOneProjection = ForwardProjection.StaticCTCollimationSingle(
        Descriptor, np.pi / 2, [np.tan(10 * np.pi / 180) * 2.5, 2.5, 0.5]
    )

    # =========================================================
    # Rawdata simulation
    # =========================================================

    if aScanType == ScanTypes.Sequence:
        Projections = ForwardProjection.RayDrivenForwardProj3DStaticCTSequence(
            Voxelphantom, Descriptor, aSimVol
        )
    else:
        Projections = ForwardProjection.RayDrivenForwardProj3DStaticCTSpiral(
            Voxelphantom, Descriptor, aSimVol
        )
    StaticCTProjections = Projections

    # =========================================================
    # Rebinning
    # =========================================================

    if aScanType == ScanTypes.Sequence:
        Projections = Rebinning.RebinningStaticCTToParallelSequence(
            Projections, Descriptor, aParGeo
        )
    else:
        Projections = Rebinning.RebinningStaticCTToParallelSpiral(
            Projections, Descriptor, aParGeo
        )
    RebinnedProjections = np.array(Projections, copy=True)

    # =========================================================
    # Filtering
    # =========================================================

    if Reconkernel == Kernels.SheppLogan:
        Kernel = Kernels.SheppLoganKernelSD(aParGeo.M, aParGeo.dxi)
    elif Reconkernel == Kernels.RamLak:
        Kernel = Kernels.RamLakKernelSD(aParGeo.M, aParGeo.dxi)
    else:
        raise ValueError(f"Unknown reconstruction kernel: {Reconkernel}")

    if aScanType == ScanTypes.Sequence:
        # MATLAB passes Npar_seq here, not Npar.
        Projections = Convolution.Convovle3DSequenceSinogram(
            Projections, Kernel, aParGeo.M, aParGeo.Npar_seq, aParGeo.dxi,
            Geometry.Ns, Parameter.NzScans
        )
    else:
        Projections = Convolution.Convovle3DSpiralSinogram(
            Projections, Kernel, aParGeo.M, aParGeo.Npar, aParGeo.dxi, Geometry.Ns
        )
    FilteredProjections = Projections

    # =========================================================
    # Reconstruction
    # =========================================================

    if aScanType == ScanTypes.Sequence:
        ReconstructedImage, Hxyz = Backprojection.SequenceWFBP3DFullyPixelDrivenParallelized_StaticCT(
            FilteredProjections, Descriptor, aParGeo, aReconVol
        )
    else:
        ReconstructedImage, Hxyz = Backprojection.SpiralWFBP3DFullyPixelDrivenParallelized_StaticCT(
            FilteredProjections, Descriptor, aParGeo, aReconVol
        )

    if not isSpectral:
        ReconstructedImageHU = (ReconstructedImage - Constants.MuW) / Constants.MuW * 1000
    else:
        ReconstructedImageHU = (ReconstructedImage - EffectiveMuWater) / EffectiveMuWater * 1000

    if Outputdirectory is not None:
        _Save(Outputdirectory, "Phantom", Voxelphantom)
        _Save(Outputdirectory, "CollimationOfOneProjection", CollimationOfOneProjection)
        _Save(Outputdirectory, "Projection2DStaticCT", StaticCTProjections)
        _Save(Outputdirectory, "ProjectiondataParallelBeamRebinned", RebinnedProjections)
        _Save(Outputdirectory, "Projections2DFiltered", FilteredProjections)
        _Save(Outputdirectory, "Hxyz", Hxyz)
        _Save(Outputdirectory, "Reconstruction", ReconstructedImage)
        _Save(Outputdirectory, "ReconstructionHU", ReconstructedImageHU)

    if ShowFigures:
        _ShowFigures(Voxelphantom, CollimationOfOneProjection, Kernel,
                     ReconstructedImageHU, Geometry, aSimVol, aReconVol)

    return {
        "Descriptor": Descriptor,
        "SimulationVolume": aSimVol,
        "ReconstructionVolume": aReconVol,
        "ParallelGeometry": aParGeo,
        "Voxelphantom": Voxelphantom,
        "isSpectral": isSpectral,
        "CollimationOfOneProjection": CollimationOfOneProjection,
        "StaticCTProjections": StaticCTProjections,
        "RebinnedProjections": RebinnedProjections,
        "Kernel": Kernel,
        "FilteredProjections": FilteredProjections,
        "Hxyz": Hxyz,
        "ReconstructedImage": ReconstructedImage,
        "ReconstructedImageHU": ReconstructedImageHU,
    }


def _ShowFigures(Voxelphantom, CollimationOfOneProjection, Kernel,
                 ReconstructedImageHU, Geometry, aSimVol, aReconVol):
    import matplotlib.pyplot as plt

    plt.figure()
    plt.imshow(Voxelphantom.reshape(aSimVol.Nysim * aSimVol.Nzsim, aSimVol.Nxsim),
               cmap="gray", aspect="auto")
    plt.title("Phantom")

    plt.figure()
    plt.imshow(np.asarray(CollimationOfOneProjection).reshape(Geometry.Ns, Geometry.MDelta),
               cmap="gray", aspect="auto")
    plt.title("Collimation of one projection")

    plt.figure()
    plt.plot(Kernel)
    plt.title("Reconstruction kernel")

    plt.figure()
    plt.imshow(ReconstructedImageHU.reshape(aReconVol.Nz * aReconVol.Ny, aReconVol.Nx),
               cmap="gray", vmin=-200, vmax=400, aspect="auto")
    plt.title("Reconstruction [HU]")
    plt.show()


if __name__ == "__main__":
    result = main()
    print(f"Reconstruction range {result['ReconstructedImageHU'].min():.0f} .. "
          f"{result['ReconstructedImageHU'].max():.0f} HU")
