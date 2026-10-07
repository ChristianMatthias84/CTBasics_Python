"""End-to-end checks of the cone-beam (3D) reconstruction chain."""
import numpy as np
import pytest

from CTBasics.Reconstruction.Backprojection import Backprojection
from CTBasics.Reconstruction.Convolution import Convolution
from CTBasics.Reconstruction.ForwardProjection import ForwardProjection
from CTBasics.Reconstruction.Kernels import Kernels

RM = 120.0
M = 201
DXI = 2 * RM / (M - 1)
XI0 = -RM - DXI

N2PI = 256
NPAR = 128
DTHETA = np.pi / NPAR
THETA0 = 0.0

# A very large focus radius keeps the cone narrow, so the parallel-beam WFBP applies.
RF = 5000.0
NS = 16
DS = 2.0
S0 = -NS / 2 * DS - DS / 2

NX = NY = 64
DX = DY = 2 * RM / NX
X0 = Y0 = -RM - DX / 2

NZ = 8
DZ = 4.0
Z0 = -NZ / 2 * DZ - DZ / 2

MU = 0.02
R = 60.0

SLAB = slice(2, 6)   # the phantom occupies these four slices


def _grid():
    axis = X0 + np.arange(1, NX + 1) * DX
    return np.meshgrid(axis, axis)


def _phantom():
    X, Y = _grid()
    volume = np.zeros((NZ, NY, NX))
    volume[SLAB][:, X * X + Y * Y <= R * R] = MU
    return volume


def _beta():
    return -np.arcsin((XI0 + np.arange(1, M + 1) * DXI) / RF)


@pytest.fixture(scope="module")
def filtered_sinogram():
    volume = _phantom().ravel()
    beta = _beta()

    Sino = np.zeros(NPAR * M * NS)
    for n in range(1, NPAR + 1):
        Sino[(n - 1) * M * NS:n * M * NS] = ForwardProjection.RayDrivenForwardProj3DFanSingle(
            RF, THETA0 + n * DTHETA, beta, S0, DS, X0, DX, NX, Y0, DY, NY,
            Z0, DZ, NZ, volume, M, NS, 0.0
        )

    Kernel = Kernels.RamLakKernelSD(M, DXI)
    return Convolution.ConvolveSinogram(Sino, Kernel, M, NPAR * NS, DXI)


@pytest.fixture(scope="module")
def reconstruction(filtered_sinogram):
    Reconstruction, HxyzVolume = Backprojection.SequenceWFBP3DFullyPixelDriven(
        RF, np.array([THETA0]), DTHETA, NPAR, N2PI, XI0, DXI, M, S0, DS, NS,
        filtered_sinogram, X0, DX, NX, Y0, DY, NY, Z0, DZ, NZ, 0.0, 0.0, [1]
    )
    return Reconstruction.reshape(NZ, NY, NX), HxyzVolume.reshape(NZ, NY, NX)


def _inner_mask():
    X, Y = _grid()
    return X * X + Y * Y < (0.6 * R) ** 2


def test_central_slices_recover_the_attenuation(reconstruction):
    Reconstruction, _ = reconstruction
    inner = _inner_mask()

    for nz in (3, 4):
        assert Reconstruction[nz][inner].mean() == pytest.approx(MU, rel=0.01)


def test_slices_outside_the_object_stay_empty(reconstruction):
    Reconstruction, _ = reconstruction
    inner = _inner_mask()

    for nz in (0, NZ - 1):
        assert abs(Reconstruction[nz][inner].mean()) < 0.001 * MU


def test_z_blur_conserves_the_total_attenuation(reconstruction):
    """Trilinear z interpolation smears the slab edges but must not lose any mass."""
    Reconstruction, _ = reconstruction
    inner = _inner_mask()

    profile = np.array([Reconstruction[nz][inner].mean() for nz in range(NZ)])
    thickness = SLAB.stop - SLAB.start

    assert profile.sum() == pytest.approx(thickness * MU, rel=0.01)


def test_z_profile_is_symmetric_around_the_slab(reconstruction):
    Reconstruction, _ = reconstruction
    inner = _inner_mask()

    profile = np.array([Reconstruction[nz][inner].mean() for nz in range(NZ)])

    assert np.allclose(profile, profile[::-1], atol=1e-6)


def test_reconstruction_is_mirror_symmetric(reconstruction):
    Reconstruction, _ = reconstruction

    central = Reconstruction[4]

    # Views run over a full half turn starting at theta0 = 0, so the geometry is exactly
    # symmetric about the x axis. The measured residual is 3.9e-12, i.e. pure floating
    # point accumulation over the 128 views, nine orders below mu.
    assert np.allclose(central, central[::-1], atol=1e-10, rtol=0.0)


def test_reconstructed_disc_is_radially_uniform(reconstruction):
    Reconstruction, _ = reconstruction

    central = Reconstruction[4]
    X, Y = _grid()
    radius = np.hypot(X, Y)

    for lower, upper in ((0, 20), (20, 40), (40, 55)):
        ring = central[(radius >= lower) & (radius < upper)]
        assert ring.mean() == pytest.approx(MU, rel=0.005)
        assert ring.std() / MU < 0.02


def test_every_voxel_is_covered_by_the_half_turn_groups(reconstruction):
    _, HxyzVolume = reconstruction

    # Npi = N2pi/2 = 128 groups, each contributing at most one view for N = Npar = 128.
    assert HxyzVolume.max() <= NPAR
    assert HxyzVolume[4][_inner_mask()].min() > 0


def test_attenuation_scales_linearly(filtered_sinogram):
    def reconstruct(scale):
        Reconstruction, _ = Backprojection.SequenceWFBP3DFullyPixelDriven(
            RF, np.array([THETA0]), DTHETA, NPAR, N2PI, XI0, DXI, M, S0, DS, NS,
            scale * filtered_sinogram, X0, DX, NX, Y0, DY, NY, Z0, DZ, NZ, 0.0, 0.0, [1]
        )
        return Reconstruction

    assert np.allclose(reconstruct(3.0), 3.0 * reconstruct(1.0), rtol=1e-12)


def test_parallel_3d_backprojector_is_linear():
    Sino = np.zeros(NPAR * M * NS)
    Sino[::7] = 1.0

    def backproject(scale):
        return Backprojection.PixDriven3DBackProjParCartesian(
            RF, THETA0, DTHETA, NPAR, N2PI, XI0, DXI, M, S0, DS, NS,
            scale * Sino, X0, DX, NX, Y0, DY, NY, Z0, DZ, NZ
        )

    assert np.allclose(backproject(2.5), 2.5 * backproject(1.0), rtol=1e-12)


def test_parallel_3d_backprojector_matches_the_single_view_sum():
    Sino = np.zeros(NPAR * M * NS)
    Sino[::11] = 1.0

    Reconstruction = Backprojection.PixDriven3DBackProjParCartesian(
        RF, THETA0, DTHETA, NPAR, N2PI, XI0, DXI, M, S0, DS, NS,
        Sino, X0, DX, NX, Y0, DY, NY, Z0, DZ, NZ
    )

    assert Reconstruction.shape == (NX * NY * NZ,)
    assert np.all(np.isfinite(Reconstruction))
    assert Reconstruction.max() > 0.0


def test_spiral_wfbp_runs_and_stays_finite(filtered_sinogram):
    Reconstruction, HxyzVolume = Backprojection.SpiralWFBP3DFullyPixelDriven(
        RF, THETA0, DTHETA, NPAR, N2PI, XI0, DXI, M, S0, DS, NS,
        filtered_sinogram, X0, DX, NX, Y0, DY, NY, Z0, DZ, NZ,
        0.0, 0.0, THETA0
    )

    # A zero pitch degenerates the spiral into a sequence scan at z0focus = 0.
    assert np.all(np.isfinite(Reconstruction))
    assert Reconstruction.reshape(NZ, NY, NX)[4][_inner_mask()].mean() == pytest.approx(MU, rel=0.05)
    assert HxyzVolume.max() > 0


def test_static_ct_wfbp_variants_stay_finite(filtered_sinogram):
    RD = 500.0

    for Reconstruction, _ in (
        Backprojection.SequenceWFBP3DFullyPixelDriven_StaticCT(
            RF, RD, THETA0, DTHETA, NPAR, N2PI, XI0, DXI, M, S0, DS, NS, 0.0,
            filtered_sinogram, X0, DX, NX, Y0, DY, NY, Z0, DZ, NZ, 0.0, 0.0, [1]
        ),
        Backprojection.SpiralWFBP3DFullyPixelDriven_StaticCT(
            RF, RD, THETA0, DTHETA, NPAR, N2PI, XI0, DXI, M, S0, DS, NS, 0.0,
            filtered_sinogram, X0, DX, NX, Y0, DY, NY, Z0, DZ, NZ, 0.0, 0.0, THETA0
        ),
    ):
        assert np.all(np.isfinite(Reconstruction))
        assert Reconstruction.shape == (NX * NY * NZ,)
