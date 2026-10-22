"""parfor ports: a worker pool must reproduce the serial results."""
import numba
import numpy as np
import pytest

from CTBasics.Memory.Precision import Precision
from CTBasics.MultiCore.Parallelization import Parallelization
from CTBasics.Reconstruction.Backprojection import Backprojection
from CTBasics.Reconstruction.ForwardProjection import ForwardProjection
from CTBasics.Simulation.Materials import Materials

RM = 120.0
NX = NY = 24
DX = DY = 2 * RM / NX
X0 = Y0 = -RM - DX / 2
M = 31
DXI = 2 * RM / (M - 1)
XI0 = -RM - DXI
N360 = 12
TROT = 0.5

RF = 595.0
NS = 4
DS = 2.0
S0 = -NS / 2 * DS - DS / 2
NZ = 4
DZ = 4.0
Z0 = -NZ / 2 * DZ - DZ / 2


def _order():
    return np.random.default_rng(3).permutation(N360) * 2 * np.pi / N360


def _volume():
    axis = X0 + np.arange(1, NX + 1) * DX
    X, Y = np.meshgrid(axis, axis)
    volume = np.zeros((NZ, NY, NX))
    volume[1:3][:, X * X + Y * Y <= 60.0 ** 2] = 0.02
    return volume.ravel()


def _beta():
    return -np.arcsin((XI0 + np.arange(1, M + 1) * DXI) / RF)


def _motion():
    Projections, Angles, points, _ = \
        ForwardProjection.RayDrivenForwProjParCartesianMultipleWithMotionArbitraryOrderParallelized(
            X0, DX, NX, Y0, DY, NY, 2 * N360, XI0, DXI, M, N360, TROT, _order()
        )
    return [Projections, Angles, points]


def _dual():
    Projections, Angles, points, _, _ = \
        ForwardProjection.RayDrivenForwProjParCartesianMultipleWithMotionDual_ParallelSimple(
            X0, DX, NX, Y0, DY, NY, N360, XI0, DXI, M, N360, TROT, _order(), Shift_GT=0.1
        )
    return [Projections, Angles, points]


def _spectral():
    Mat = np.full(NX * NY, int(Materials.Nyst_water))
    return [ForwardProjection.SpectralRayDrivenForwProjParCartesianMultiple(
        X0, DX, NX, Y0, DY, NY, Mat, 0.0, np.pi / 8, 8, XI0, DXI, M, "kV120", 40.0, 3, 20.0
    )]


def _fan3d():
    return [ForwardProjection.RayDrivenForwardProj3DFanMultipleSequence(
        RF, 0.0, 2 * np.pi / N360, _beta(), S0, DS, X0, DX, NX, Y0, DY, NY,
        Z0, DZ, NZ, _volume(), M, NS, N360, 0.0
    )]


def _pixdriven3d():
    Sino = np.zeros(N360 * M * NS)
    Sino[::5] = 1.0
    return [Backprojection.PixDriven3DBackProjParCartesian(
        RF, 0.0, np.pi / N360, N360, N360, XI0, DXI, M, S0, DS, NS,
        Sino, X0, DX, NX, Y0, DY, NY, Z0, DZ, NZ
    )]


def _projectors2d():
    """Threaded numba kernels; rays and image rows are independent of the thread count."""
    Image = _volume()[NX * NY:2 * NX * NY]
    Sino = ForwardProjection.RayDrivenForwProjParCartesianMultiple(
        X0, DX, NX, Y0, DY, NY, Image, 0.0, np.pi / N360, N360, XI0, DXI, M
    )
    return [
        Sino,
        ForwardProjection.RayDrivenForwardProj2DFanMultiple(
            RF, 0.0, 2 * np.pi / N360, _beta(), X0, DX, NX, Y0, DY, NY, Image, M, N360
        ),
        ForwardProjection.RayDrivenForwardProj2DStaticCTMultiple(
            RF, 400.0, 0.0, 2 * np.pi / N360, 0.0, 2 * np.pi / 64, X0, DX, NX, Y0, DY, NY,
            Image, 64, N360
        ),
        Backprojection.PixDrivenBackProjParCartesian(
            0.0, np.pi / N360, N360, 2 * N360, XI0, DXI, M, Sino, X0, DX, NX, Y0, DY, NY
        ),
        Backprojection.WFBP2DFullyPixelDriven(
            0.0, np.pi / N360, N360, 2 * N360, XI0, DXI, M, Sino, X0, DX, NX, Y0, DY, NY
        ),
    ]


def _projectors3d():
    """Threaded cone-beam kernels; rays and voxel rows are independent of the thread count."""
    dDelta = 2 * np.pi / 64
    Sino = np.random.default_rng(5).random(2 * N360 * NS * M)
    return [
        ForwardProjection.RayDrivenForwardProj3DStaticCTMultipleSequence(
            RF, 400.0, 0.0, 2 * np.pi / N360, -dDelta, dDelta, S0, DS, 0.5,
            X0, DX, NX, Y0, DY, NY, Z0, DZ, NZ, _volume(), 64, NS, N360, 1.0
        ),
        *Backprojection.SpiralWFBP3DFullyPixelDriven(
            RF, 0.0, np.pi / N360, 2 * N360, 2 * N360, XI0, DXI, M, S0, DS, NS,
            Sino, X0, DX, NX, Y0, DY, NY, Z0, DZ, NZ, -2.0, 1.0, 0.0
        ),
        *Backprojection.SequenceWFBP3DFullyPixelDriven_StaticCT(
            RF, 400.0, 0.0, np.pi / N360, N360, 2 * N360, XI0, DXI, M, S0, DS, NS, 4.0,
            Sino, X0, DX, NX, Y0, DY, NY, Z0, DZ, NZ, -2.0, 4.0, range(1, 3)
        ),
    ]


CASES = {"motion": _motion, "dual": _dual, "spectral": _spectral, "fan3d": _fan3d,
         "pixdriven3d": _pixdriven3d, "projectors2d": _projectors2d,
         "projectors3d": _projectors3d}


@pytest.fixture(scope="module")
def results():
    """Every case once serially (session default) and once on a two-worker pool."""
    serial = {name: case() for name, case in CASES.items()}

    Parallelization.DeletePool()
    Parallelization.Init(2)
    try:
        parallel = {name: case() for name, case in CASES.items()}
        parallel["threads"] = numba.get_num_threads()

        Precision.Value(32)
        single = _fan3d()[0]
    finally:
        Precision.Value(64)
        Parallelization.DeletePool()
        Parallelization.Init(1)

    return serial, parallel, single


@pytest.mark.parametrize("name", ["motion", "dual", "spectral", "fan3d", "projectors2d",
                                  "projectors3d"])
def test_pool_reproduces_the_serial_result_bitwise(results, name):
    serial, parallel, _ = results

    for expected, actual in zip(serial[name], parallel[name]):
        assert np.array_equal(actual, expected)


def test_init_sets_the_kernel_threads(results):
    _, parallel, _ = results

    assert parallel["threads"] == 2
    assert numba.get_num_threads() == 1


def test_pool_reduction_matches_the_serial_sum(results):
    """Workers sum their own half-turn groups, so only the summation order differs."""
    serial, parallel, _ = results

    assert np.allclose(parallel["pixdriven3d"][0], serial["pixdriven3d"][0], rtol=1e-12, atol=0.0)


def test_workers_inherit_the_working_precision(results):
    _, _, single = results

    assert single.dtype == np.float32


def test_init_with_one_worker_runs_without_pool():
    assert Parallelization.GetPool() is None
    assert Parallelization.GetNumWorkers() == 0
