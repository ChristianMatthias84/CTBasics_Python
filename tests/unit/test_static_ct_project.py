"""Checks of the static CT project scripts and their sampling helpers."""
import math

import numpy as np
import pytest

from CTBasics.Descriptors.Geometries.ParallelBeam import ParallelBeam
from CTBasics.Descriptors.Geometries.StaticCTBeam import StaticCTBeam
from CTBasics.Descriptors.Sampling.AzimuthalSampling import AzimuthalSampling
from CTBasics.Descriptors.ScanDescriptor import ScanDescriptor
from CTBasics.Descriptors.ScanParameter.Spiral import Spiral
from CTBasics.Descriptors.Volumes.Reconstruction import Reconstruction
from CTBasics.Descriptors.Volumes.Simulation import Simulation
from CTBasics.Projects.StaticCT import StaticCT, StaticCT3D
from CTBasics.Projects.StaticCT.StaticCTHelperFunctions import StaticCTSystemHelperFunctions


@pytest.fixture(scope="module")
def staticct2d():
    return StaticCT.main(MDelta=256, MbetaDash=256, N2pi=128,
                         Nxsim=96, Nysim=96, Nx=96, Ny=96)


@pytest.fixture(scope="module")
def staticct3d():
    Geometry = StaticCTBeam()
    Geometry.Init()
    # Only the azimuthal sampling and the ring resolution are coarsened; Ns and soffset
    # define the detector row offset A of the backprojector and must stay consistent.
    Geometry.MDelta = 240
    Geometry.dDelta = 2 * math.pi / 240
    Geometry.MbetaDash = 240
    Geometry.dbetaDash = 2 * math.pi / 240
    Geometry.Sampling = AzimuthalSampling(120)

    Descriptor = ScanDescriptor(Geometry, Spiral(Geometry).Init())

    SimVol = Simulation(Descriptor)
    SimVol.Nxsim = SimVol.Nysim = 48
    SimVol.dxsim = SimVol.dysim = 2 * Geometry.RM / 48
    SimVol.x0sim = SimVol.y0sim = -Geometry.RM - SimVol.dxsim / 2

    ReconVol = Reconstruction(Descriptor, SimVol)

    return StaticCT3D.main(Descriptor=Descriptor, SimVol=SimVol, ReconVol=ReconVol)


def _WaterMask(x0, dx, Nx, y0, dy, Ny, radius=40.0):
    x = x0 + np.arange(1, Nx + 1) * dx
    y = y0 + np.arange(1, Ny + 1) * dy
    X, Y = np.meshgrid(x, y)
    return np.hypot(X, Y) < radius


def test_static_ct_geometry_follows_the_arbitrary_limit_angle(staticct2d):
    Geometry = staticct2d["Geometry"]

    RF, RD = Geometry["RF"], Geometry["RD"]
    assert RF == 1150 / 2
    assert RD == 850 / 2
    # The limit angle is pi/2, so the tangent reduces to RD / RF.
    assert Geometry["betaMax"] == pytest.approx(math.atan(RD / RF))
    assert Geometry["RM"] == pytest.approx(math.sin(Geometry["betaMax"]) * RF)
    assert Geometry["thetaN"] == pytest.approx(math.pi)


def test_static_ct_two_d_reconstructs_water(staticct2d):
    Geometry = staticct2d["Geometry"]
    Image = staticct2d["ReconstructedImageHU"].reshape(Geometry["Ny"], Geometry["Nx"])
    Mask = _WaterMask(Geometry["x0"], Geometry["dx"], Geometry["Nx"],
                      Geometry["y0"], Geometry["dy"], Geometry["Ny"], radius=60.0)

    assert Image[Mask].mean() == pytest.approx(0.0, abs=10.0)
    assert Image[Mask].std() < 20.0
    # The hip inserts use 1.8 * MuW.
    assert Image.max() > 600.0


def test_static_ct_two_d_data_sizes(staticct2d):
    Geometry = staticct2d["Geometry"]

    assert staticct2d["StaticCTProjections"].size == Geometry["MDelta"] * Geometry["N"]
    assert staticct2d["RebinnedProjections"].size == Geometry["MPar"] * Geometry["NPar"]
    assert staticct2d["FilteredProjections"].size == staticct2d["RebinnedProjections"].size
    assert staticct2d["ReconstructedImage"].size == Geometry["Nx"] * Geometry["Ny"]


def test_static_ct_two_d_fft_path_matches_the_spatial_convolution():
    Direct = StaticCT.main(MDelta=256, MbetaDash=256, N2pi=128,
                           Nxsim=64, Nysim=64, Nx=64, Ny=64)
    Fourier = StaticCT.main(MDelta=256, MbetaDash=256, N2pi=128,
                            Nxsim=64, Nysim=64, Nx=64, Ny=64, FFTConvolution=True)

    Mask = _WaterMask(Direct["Geometry"]["x0"], Direct["Geometry"]["dx"], 64,
                      Direct["Geometry"]["y0"], Direct["Geometry"]["dy"], 64, radius=60.0)
    A = Direct["ReconstructedImageHU"].reshape(64, 64)[Mask]
    B = Fourier["ReconstructedImageHU"].reshape(64, 64)[Mask]

    # Both paths implement the same ramp filter, only the discretization differs.
    assert np.abs(A - B).max() < 50.0


def test_static_ct_three_d_reconstructs_water_in_the_covered_slices(staticct3d):
    ReconVol = staticct3d["ReconstructionVolume"]
    Volume = staticct3d["ReconstructedImageHU"].reshape(ReconVol.Nz, ReconVol.Ny, ReconVol.Nx)
    Mask = _WaterMask(ReconVol.x0, ReconVol.dx, ReconVol.Nx,
                      ReconVol.y0, ReconVol.dy, ReconVol.Ny)

    Means = np.array([Volume[nz][Mask].mean() for nz in range(ReconVol.Nz)])

    assert np.count_nonzero(np.abs(Means) < 20.0) >= 10
    assert Volume.max() > 400.0


def test_static_ct_three_d_collimation_blocks_part_of_the_detector(staticct3d):
    Geometry = staticct3d["Descriptor"].Geometry
    Collimation = staticct3d["CollimationOfOneProjection"]

    assert Collimation.size == Geometry.MDelta * Geometry.Ns
    assert set(np.unique(Collimation)) <= {0.0, 1.0}
    assert 0 < np.count_nonzero(Collimation) < Collimation.size


def test_greedy_gap_sampling_halves_the_largest_gap():
    angles = StaticCTSystemHelperFunctions.GreedyGapSamplingSubset(4, 0, 90)

    assert angles == [45.0, 22.5, 67.5, 11.25]


def test_greedy_gap_sampling_wraps_into_zero_to_three_sixty():
    angles = StaticCTSystemHelperFunctions.GreedyGapSamplingSubset(8, 300, 120)

    assert all(0.0 <= a < 360.0 for a in angles)


def test_subset_interleaving_360_alternates_between_the_subsets():
    angles = np.rad2deg(StaticCTSystemHelperFunctions.SubsetInterleaving(8, 2, "360"))

    assert np.allclose(angles, [90, 270, 45, 225, 135, 315, 22.5, 202.5])


def test_subset_interleaving_180_mirrors_the_half_turn():
    angles = np.rad2deg(StaticCTSystemHelperFunctions.SubsetInterleaving(8, 2, "180"))
    half = angles[:4]

    assert np.allclose(angles[4:], (half + 180) % 360)


def test_subset_interleaving_rejects_odd_counts_for_180():
    with pytest.raises(ValueError):
        StaticCTSystemHelperFunctions.SubsetInterleaving(7, 2, "180")


def test_subset_interleaving_rejects_unknown_recon_types():
    with pytest.raises(ValueError):
        StaticCTSystemHelperFunctions.SubsetInterleaving(8, 2, "90")


def test_subset_interleaving_applies_the_global_offset():
    Base = StaticCTSystemHelperFunctions.SubsetInterleaving(8, 2, "360")
    Shifted = StaticCTSystemHelperFunctions.SubsetInterleaving(8, 2, "360", 30)

    assert np.allclose(np.rad2deg(Shifted), (np.rad2deg(Base) + 30) % 360)


def test_parallel_geometry_of_the_static_ct_descriptor_is_consistent(staticct3d):
    Descriptor = staticct3d["Descriptor"]
    ParGeo = ParallelBeam(Descriptor)

    assert ParGeo.M == staticct3d["ParallelGeometry"].M
    assert ParGeo.dxi == pytest.approx(Descriptor.Geometry.dDelta * Descriptor.Geometry.RF)
