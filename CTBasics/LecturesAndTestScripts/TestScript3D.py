"""Full 3D spiral and sequence simulation and reconstruction chain.

Port of LecturesAndTestScripts/TestScript3D.m. The MATLAB script loops over both scan
types; here `main` runs one scan type so it can be driven from tests, and `DemoAll`
reproduces the MATLAB loop.
"""
import numpy as np

from CTBasics.Descriptors.Geometries.ParallelBeam import ParallelBeam
from CTBasics.Descriptors.ScanDescriptor import ScanDescriptor
from CTBasics.Descriptors.ScannerGeometryTypes import ScannerGeometryTypes
from CTBasics.Descriptors.ScanTypes import ScanTypes
from CTBasics.Descriptors.Volumes.Reconstruction import Reconstruction
from CTBasics.Descriptors.Volumes.Simulation import Simulation
from CTBasics.ImageManipulation.ImageManipulation import ImageManipulation
from CTBasics.Preprocessing.WBHC import WBHC as WBHCorrection
from CTBasics.ReadAndWrite.IOHelperFunctions import IOHelperFunctions
from CTBasics.Reconstruction.Backprojection import Backprojection
from CTBasics.Reconstruction.Constants import Constants
from CTBasics.Reconstruction.Convolution import Convolution
from CTBasics.Reconstruction.ForwardProjection import ForwardProjection
from CTBasics.Reconstruction.Kernels import Kernels
from CTBasics.Reconstruction.MoireFilter import MoireFilter
from CTBasics.Reconstruction.Rebinning import Rebinning
from CTBasics.Simulation.CreatePhantom import CreatePhantom
from CTBasics.Simulation.Noise import Noise
from CTBasics.Simulation.Spectra import Spectra
from CTBasics.LecturesAndTestScripts.CTRecon101 import _Save


def main(aScanType=ScanTypes.Spiral, Outputdirectory=None, ShowFigures=False,
         Reconkernel=Kernels.RamLak, SimNoise=False, ApplyMoireFilter=False,
         PatientData=False, WBHC=False, N_I0=10000, DataDirectory=None,
         SelectedPhantom="SimpleEllipseWithHips3D",
         spectrum=Spectra.kV120, E0=20, deltaE=10, N_Ebins=12,
         ScanRange=(0, 50), Pitch=1.5, TableIncrement=0.5, Descriptor=None,
         SimVol=None, ReconVol=None):
    # =========================================================
    # Descriptor and derived geometry
    # =========================================================

    if Descriptor is None:
        Descriptor = ScanDescriptor.Create(ScannerGeometryTypes.Fan, aScanType)

        if aScanType == ScanTypes.Spiral:
            Descriptor.Parameter.SetScanRange(*ScanRange)
            Descriptor.Parameter.SetPitch(Pitch)
        elif aScanType == ScanTypes.Sequence:
            Descriptor.Parameter.SetTableIncrement(TableIncrement)

    aSimVol = Simulation(Descriptor) if SimVol is None else SimVol
    aReconVol = Reconstruction(Descriptor, aSimVol) if ReconVol is None else ReconVol
    aParGeo = ParallelBeam(Descriptor)

    Geometry = Descriptor.Geometry
    Parameter = Descriptor.Parameter

    # =========================================================
    # Phantom Definition & Phantom Creation
    # =========================================================

    if not PatientData:
        Voxelphantom, isSpectral = CreatePhantom.CreateVoxelPhantom3D(SelectedPhantom, aSimVol)
    else:
        from pathlib import Path

        if DataDirectory is None:
            DataDirectory = IOHelperFunctions().GlobalDataPath

        isSpectral = False
        V_HU, V_info, V_mapped = IOHelperFunctions.read_Volume_with_Dicom(
            str(Path(DataDirectory) / "AbdomenPelvis3D")
        )
        V_HU = ImageManipulation.Resample3D(
            np.asarray(V_HU, dtype=float), aSimVol.Nxsim, aSimVol.Nysim, aSimVol.Nzsim
        )
        # MATLAB `Slice = V(:,:,nz)'; Slice(:)'` is the row-major flattening of each slice.
        Voxelphantom = np.concatenate(
            [V_HU[:, :, nz].ravel() for nz in range(aSimVol.Nzsim)]
        )
        Voxelphantom = (Voxelphantom + 1000) * Constants.MuW / 1000

    # =========================================================
    # Rawdata Simulation
    # =========================================================

    if isSpectral:
        # MATLAB has no spectral 3D forward projector; the branch is empty there.
        raise NotImplementedError("Spectral 3D forward projection is not available in MATLAB either")

    if aScanType == ScanTypes.Sequence:
        Projections = ForwardProjection.RayDrivenForwardProj3DFanSequence(
            Voxelphantom, Descriptor, aSimVol
        )
    else:
        Projections = ForwardProjection.RayDrivenForwardProj3DFanSpiral(
            Voxelphantom, Descriptor, aSimVol
        )

    FanProjections = Projections

    # =========================================================
    # WBHC correction
    # =========================================================

    EffectiveMuWater = Constants.MuW
    if isSpectral and WBHC:
        Projections, EffectiveMuWater = WBHCorrection.LUTBasedWBHCCorrection(
            spectrum, E0, N_Ebins, deltaE, Projections
        )

    # =========================================================
    # Rebinning
    # =========================================================

    if aScanType == ScanTypes.Sequence:
        Projections = Rebinning.RebinningFanToParallelSequence(Projections, Descriptor, aParGeo)
    else:
        Projections = Rebinning.RebinningFanToParallelSpiral(Projections, Descriptor, aParGeo)

    RebinnedProjections = np.array(Projections, copy=True)

    # =========================================================
    # Noise Simulation
    # =========================================================

    if SimNoise:
        if aScanType == ScanTypes.Sequence:
            Nsamples = aParGeo.M * Geometry.Ns * aParGeo.Npar * Parameter.NzScans
        else:
            Nsamples = aParGeo.M * Geometry.Ns * aParGeo.Npar
        Projections = Noise.ApplyPoissonNoiseParallel(Projections, Nsamples, N_I0)

    if ApplyMoireFilter:
        Projections = MoireFilter.ApplyMoireFilterOnProjections3D(Projections, aParGeo, Descriptor)

    # =========================================================
    # Convolution Kernel definition & Projection convolution
    # =========================================================

    if Reconkernel == Kernels.SheppLogan:
        Kernel = Kernels.SheppLoganKernelSD(aParGeo.M, aParGeo.dxi)
    elif Reconkernel == Kernels.RamLak:
        Kernel = Kernels.RamLakKernelSD(aParGeo.M, aParGeo.dxi)
    else:
        raise ValueError(f"Unknown reconstruction kernel: {Reconkernel}")

    if aScanType == ScanTypes.Sequence:
        Projections = Convolution.Convovle3DSequenceSinogram(
            Projections, Kernel, aParGeo.M, aParGeo.Npar, aParGeo.dxi,
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
        ReconstructedImage, Hxyz = Backprojection.SequenceWFBP3DFullyPixelDrivenParallelized(
            FilteredProjections, Descriptor, aParGeo, aReconVol
        )
    else:
        ReconstructedImage, Hxyz = Backprojection.SpiralWFBP3DFullyPixelDrivenParallelized(
            FilteredProjections, Descriptor, aParGeo, aReconVol
        )

    if not isSpectral:
        ReconstructedImageHU = (ReconstructedImage - Constants.MuW) / Constants.MuW * 1000
    else:
        ReconstructedImageHU = (ReconstructedImage - EffectiveMuWater) / EffectiveMuWater * 1000

    if Outputdirectory is not None:
        _Save(Outputdirectory, "Patientdata" if PatientData else "Phantom", Voxelphantom)
        _Save(Outputdirectory, "Projection2DFanBeam", FanProjections)
        _Save(Outputdirectory, "ProjectiondataParallelBeamRebinned", RebinnedProjections)
        _Save(Outputdirectory, "Projections2DFiltered", FilteredProjections)
        _Save(Outputdirectory, "Hxyz", Hxyz)
        _Save(Outputdirectory, "Reconstruction", ReconstructedImage)
        _Save(Outputdirectory, "ReconstructionHU", ReconstructedImageHU)

    if ShowFigures:
        _ShowFigures(ReconstructedImageHU, aReconVol)

    return {
        "Descriptor": Descriptor,
        "SimulationVolume": aSimVol,
        "ReconstructionVolume": aReconVol,
        "ParallelGeometry": aParGeo,
        "Voxelphantom": Voxelphantom,
        "isSpectral": isSpectral,
        "FanProjections": FanProjections,
        "RebinnedProjections": RebinnedProjections,
        "Kernel": Kernel,
        "FilteredProjections": FilteredProjections,
        "Hxyz": Hxyz,
        "ReconstructedImage": ReconstructedImage,
        "ReconstructedImageHU": ReconstructedImageHU,
    }


def DemoAll(**kwargs):
    """Runs both scan types exactly like the MATLAB loop does."""
    results = {}
    for aScanType in (ScanTypes.Spiral, ScanTypes.Sequence):
        print(f"Testing ScanType: {aScanType}")
        results[aScanType] = main(aScanType=aScanType, **kwargs)
    return results


def _ShowFigures(ReconstructedImageHU, aReconVol):
    import matplotlib.pyplot as plt

    Volume = ReconstructedImageHU.reshape(aReconVol.Nz, aReconVol.Ny, aReconVol.Nx)
    for nz in range(aReconVol.Nz):
        plt.figure()
        plt.imshow(Volume[nz], cmap="gray", vmin=-200, vmax=400)
        plt.title(f"Reconstruction slice {nz + 1} [HU]")
        plt.colorbar()
    plt.show()


if __name__ == "__main__":
    for aScanType, result in DemoAll().items():
        recon = result["ReconstructedImageHU"]
        print(f"{aScanType}: min {recon.min():.1f} HU, max {recon.max():.1f} HU")
