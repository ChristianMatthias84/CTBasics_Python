"""Coarse end-to-end checks of the 3D and 4D lecture scripts."""
import numpy as np
import pytest

from CTBasics.Descriptors.Sampling.AzimuthalSampling import AzimuthalSampling
from CTBasics.Descriptors.ScanDescriptor import ScanDescriptor
from CTBasics.Descriptors.ScannerGeometryTypes import ScannerGeometryTypes
from CTBasics.Descriptors.ScanTypes import ScanTypes
from CTBasics.Descriptors.Volumes.Reconstruction import Reconstruction
from CTBasics.Descriptors.Volumes.Simulation import Simulation
from CTBasics.LecturesAndTestScripts import Test4DCT, TestScript3D, TestScriptScannerData3D

RM = 250.0


def _CoarseSetup(aScanType):
    """Descriptor and volumes coarse enough to keep the 3D chain below ten seconds.

    N2pi is lowered together with MFan, so the ratio Npar/Npi stays large enough for the
    half-turn data completeness check of the weighted filtered backprojection.
    """
    Descriptor = ScanDescriptor.Create(ScannerGeometryTypes.Fan, aScanType)

    if aScanType == ScanTypes.Spiral:
        Descriptor.Parameter.SetScanRange(0, 30)
        Descriptor.Parameter.SetPitch(1.5)
    else:
        Descriptor.Parameter.SetTableIncrement(0.5)

    Descriptor.Geometry.Sampling = AzimuthalSampling(64)
    betaMax = Descriptor.Geometry.beta[0]
    Descriptor.Geometry.MFan = 97
    Descriptor.Geometry.beta = betaMax + (-2 * betaMax / 96) * np.arange(97)

    SimVol = Simulation(Descriptor)
    SimVol.Nxsim = SimVol.Nysim = 48
    SimVol.dxsim = SimVol.dysim = 2 * RM / 48
    SimVol.x0sim = SimVol.y0sim = -RM - SimVol.dxsim / 2.0

    return Descriptor, SimVol, Reconstruction(Descriptor, SimVol)


def _SliceWaterMeans(result, radius=40.0):
    ReconVol = result["ReconstructionVolume"]
    Volume = result["ReconstructedImageHU"].reshape(ReconVol.Nz, ReconVol.Ny, ReconVol.Nx)

    x = ReconVol.x0 + np.arange(1, ReconVol.Nx + 1) * ReconVol.dx
    y = ReconVol.y0 + np.arange(1, ReconVol.Ny + 1) * ReconVol.dy
    X, Y = np.meshgrid(x, y)
    Mask = np.hypot(X, Y) < radius

    return np.array([Volume[nz][Mask].mean() for nz in range(ReconVol.Nz)])


@pytest.fixture(scope="module")
def spiral3d():
    Descriptor, SimVol, ReconVol = _CoarseSetup(ScanTypes.Spiral)
    return TestScript3D.main(ScanTypes.Spiral, Descriptor=Descriptor,
                             SimVol=SimVol, ReconVol=ReconVol)


@pytest.fixture(scope="module")
def sequence3d():
    Descriptor, SimVol, ReconVol = _CoarseSetup(ScanTypes.Sequence)
    return TestScript3D.main(ScanTypes.Sequence, Descriptor=Descriptor,
                             SimVol=SimVol, ReconVol=ReconVol)


@pytest.fixture(scope="module")
def fourdct():
    return Test4DCT.main(M=129, N2pi=256, Nx=96, Ny=96, Nxsim=96, Nysim=96)


def test_spiral_chain_produces_the_expected_data_sizes(spiral3d):
    Geometry = spiral3d["Descriptor"].Geometry
    Parameter = spiral3d["Descriptor"].Parameter
    ParGeo = spiral3d["ParallelGeometry"]

    assert spiral3d["FanProjections"].size == Geometry.MFan * Geometry.Ns * Parameter.N
    assert spiral3d["RebinnedProjections"].size == ParGeo.M * Geometry.Ns * ParGeo.Npar
    assert spiral3d["FilteredProjections"].size == spiral3d["RebinnedProjections"].size

    ReconVol = spiral3d["ReconstructionVolume"]
    assert spiral3d["ReconstructedImage"].size == ReconVol.Nx * ReconVol.Ny * ReconVol.Nz
    assert spiral3d["Hxyz"].size == spiral3d["ReconstructedImage"].size


def test_spiral_reconstruction_hits_water_in_the_fully_sampled_slices(spiral3d):
    Means = _SliceWaterMeans(spiral3d)

    # The outer slices lack a full half turn of data and are zeroed by the completeness
    # volume, which maps to -1000 HU. The interior must be water.
    assert np.count_nonzero(np.abs(Means) < 20.0) >= 15
    assert np.isclose(Means.min(), -1000.0, atol=1.0)


def test_spiral_reconstruction_resolves_the_bone_inserts(spiral3d):
    Volume = spiral3d["ReconstructedImageHU"]

    # The hips are modelled with 1.8 * MuW, i.e. roughly 800 HU.
    assert Volume.max() > 600.0


def test_sequence_chain_produces_the_expected_data_sizes(sequence3d):
    Geometry = sequence3d["Descriptor"].Geometry
    Parameter = sequence3d["Descriptor"].Parameter
    ParGeo = sequence3d["ParallelGeometry"]

    assert Parameter.NzScans >= 2
    assert (sequence3d["FanProjections"].size
            == Geometry.MFan * Geometry.Ns * Parameter.N * Parameter.NzScans)
    assert (sequence3d["RebinnedProjections"].size
            == ParGeo.M * Geometry.Ns * ParGeo.Npar * Parameter.NzScans)


def test_sequence_reconstruction_hits_water(sequence3d):
    Means = _SliceWaterMeans(sequence3d)

    # Every sequence slice is fully sampled, so none of them is blanked.
    assert np.all(Means > -100.0)
    assert np.count_nonzero(np.abs(Means) < 20.0) >= 15


def test_unknown_three_d_phantom_is_rejected():
    Descriptor, SimVol, ReconVol = _CoarseSetup(ScanTypes.Spiral)

    # SimpleEllipseWithHips3D is the only 3D phantom MATLAB offers.
    with pytest.raises(ValueError):
        TestScript3D.main(ScanTypes.Spiral, Descriptor=Descriptor, SimVol=SimVol,
                          ReconVol=ReconVol, SelectedPhantom="WaterPhantomSpectral")


def test_four_d_scan_covers_four_turns(fourdct):
    Geometry = fourdct["Geometry"]

    assert Geometry["thetaN"] == pytest.approx(8 * np.pi)
    assert Geometry["N"] == 1024
    assert Geometry["Npi"] == 128
    assert Geometry["Nphases"] == 8
    assert fourdct["MultiphaseRecon"].shape == (8, 96 * 96)


def test_four_d_phase_images_differ_from_each_other(fourdct):
    Phases = fourdct["MultiphaseReconHU"]

    # The lung phantom breathes, so consecutive half turns must not agree.
    assert np.abs(Phases[0] - Phases[2]).max() > 100.0


def test_four_d_full_scan_reconstruction_is_lung_tissue(fourdct):
    Geometry = fourdct["Geometry"]
    Image = fourdct["ReconstructedImageHU"].reshape(Geometry["Ny"], Geometry["Nx"])

    # Air outside the thorax.
    assert Image[0, 0] == pytest.approx(-1000.0, abs=60.0)
    assert Image.mean() < -500.0


def test_scanner_data_script_requires_a_raw_file(tmp_path):
    with pytest.raises(Exception):
        TestScriptScannerData3D.main(str(tmp_path / "missing.raw"))
