"""Dynamic-phantom and polychromatic forward projection.

The dynamic projectors are checked against the static projector in the degenerate cases
(motion switched off, a single energy bin), where both must agree exactly.
"""
import numpy as np
import pytest

from CTBasics.Reconstruction.ForwardProjection import ForwardProjection
from CTBasics.Simulation.CreatePhantom import CreatePhantom
from CTBasics.Simulation.MaterialProperties import MaterialProperties
from CTBasics.Simulation.Materials import Materials

RM = 120.0
NX = NY = 64
DX = DY = 2 * RM / NX
X0 = Y0 = -RM - DX / 2
M = 101
DXI = 2 * RM / (M - 1)
XI0 = -RM - DXI
N360 = 64
TROT = 0.5


def _material_phantom():
    axis = X0 + np.arange(1, NX + 1) * DX
    X, Y = np.meshgrid(axis, axis)
    Mat = np.full((NY, NX), int(Materials.Air))
    Mat[X ** 2 + Y ** 2 <= 80.0 ** 2] = int(Materials.Nyst_water)
    Mat[(X - 30) ** 2 + Y ** 2 <= 25.0 ** 2] = int(Materials.Nyst_CorticalBone)
    return Mat.ravel()


# --------------------------------------------------------------- scan order

def test_rearrange_projections_sorts_every_full_rotation():
    Angles = np.array([2.0, 0.0, 3.0, 1.0])
    Projections = np.arange(8 * 3, dtype=float).reshape(8, 3)
    nTs = np.arange(1, 9, dtype=float)

    Sorted, Angles_new, nTs_new = ForwardProjection.RearrangeProjections(Angles, Projections, nTs)

    assert np.array_equal(Angles_new, [0.0, 1.0, 2.0, 3.0])
    # Permutation [1, 3, 0, 2] applied inside each of the two rotations.
    assert np.array_equal(nTs_new, [2, 4, 1, 3, 6, 8, 5, 7])
    assert np.array_equal(Sorted[0], Projections[1])
    assert np.array_equal(Sorted[4], Projections[5])


def test_rearrange_projections_handles_an_incomplete_rotation():
    Angles = np.array([2.0, 0.0, 3.0, 1.0])
    Projections = np.arange(6 * 2, dtype=float).reshape(6, 2)
    nTs = np.arange(1, 7, dtype=float)

    Sorted, _, nTs_new = ForwardProjection.RearrangeProjections(Angles, Projections, nTs)

    # The trailing two views are sorted by their own leading angles [2.0, 0.0].
    assert np.array_equal(nTs_new[4:], [6, 5])
    assert np.array_equal(Sorted[4], Projections[5])


# ------------------------------------------------------------ motion projectors

@pytest.fixture(scope="module")
def arbitrary_order_result():
    rng = np.random.default_rng(0)
    Order = rng.permutation(N360) * 2 * np.pi / N360
    N = 2 * N360

    result = ForwardProjection.RayDrivenForwProjParCartesianMultipleWithMotionArbitraryOrder(
        X0, DX, NX, Y0, DY, NY, N, XI0, DXI, M, N360, TROT, Order, Motion=True
    )
    return Order, N, result


def test_arbitrary_order_returns_ascending_angles(arbitrary_order_result):
    Order, N, (Projections, Angles, points, PhantomTruth, nT_Phases, nTs, RoI) = arbitrary_order_result

    assert Projections.shape == (N * M,)
    assert np.all(np.diff(Angles) >= 0)
    assert np.array_equal(np.sort(Order), Angles)
    assert points.shape == (2, N)
    assert len(RoI) == N


def test_arbitrary_order_stores_one_ground_truth_per_half_rotation(arbitrary_order_result):
    _, N, (_, _, _, PhantomTruth, nT_Phases, _, _) = arbitrary_order_result

    assert PhantomTruth.shape == (2 * N // N360, NX * NY)
    assert np.array_equal(nT_Phases, [N360 // 4 + k * N360 // 2 for k in range(len(nT_Phases))])
    assert np.all(PhantomTruth.max(axis=1) > 0)


def test_arbitrary_order_without_motion_equals_the_static_projector():
    Theta = np.arange(N360) * 2 * np.pi / N360

    Projections, _, _, _, _, _, _ = \
        ForwardProjection.RayDrivenForwProjParCartesianMultipleWithMotionArbitraryOrder(
            X0, DX, NX, Y0, DY, NY, N360, XI0, DXI, M, N360, TROT, Theta, Motion=False
        )

    Image, _, _, _, _ = CreatePhantom.CreateVoxelPhantom2DWithMotionUnified(
        NX, DX, X0, NY, DY, Y0, 0, TROT / N360, 1, "Cardiac", Motion=False
    )
    dtheta = 2 * np.pi / N360
    Reference = np.asarray(ForwardProjection.RayDrivenForwProjParCartesianMultiple(
        X0, DX, NX, Y0, DY, NY, Image, -dtheta, dtheta, N360, XI0, DXI, M
    ))

    assert np.allclose(Projections, Reference, atol=1e-12, rtol=0.0)


def test_parallelized_variant_matches_the_serial_one(arbitrary_order_result):
    Order, _, (Projections, Angles, points, _, _, _, _) = arbitrary_order_result

    Par, AnglesPar, pointsPar, _ = \
        ForwardProjection.RayDrivenForwProjParCartesianMultipleWithMotionArbitraryOrderParallelized(
            X0, DX, NX, Y0, DY, NY, 2 * N360, XI0, DXI, M, N360, TROT, Order, Motion=True
        )

    assert np.array_equal(Par, Projections)
    assert np.array_equal(AnglesPar, Angles)
    assert np.array_equal(pointsPar, points)


def test_dual_dispatches_to_the_parallel_simple_variant():
    rng = np.random.default_rng(1)
    Order = rng.permutation(N360) * 2 * np.pi / N360
    N = N360

    Dual = ForwardProjection.RayDrivenForwProjParCartesianMultipleWithMotionDual(
        X0, DX, NX, Y0, DY, NY, N, XI0, DXI, M, N360, TROT, Order
    )
    Simple = ForwardProjection.RayDrivenForwProjParCartesianMultipleWithMotionDual_ParallelSimple(
        X0, DX, NX, Y0, DY, NY, N, XI0, DXI, M, N360, TROT, Order
    )

    assert np.array_equal(Dual[0], Simple[0])
    assert Dual[3].shape == (N, M)


def test_dual_linear_matches_the_parallel_simple_projections():
    rng = np.random.default_rng(2)
    Order = rng.permutation(N360) * 2 * np.pi / N360

    Linear = ForwardProjection.RayDrivenForwProjParCartesianMultipleWithMotionDual_linear(
        X0, DX, NX, Y0, DY, NY, N360, XI0, DXI, M, N360, TROT, Order
    )
    Simple = ForwardProjection.RayDrivenForwProjParCartesianMultipleWithMotionDual_ParallelSimple(
        X0, DX, NX, Y0, DY, NY, N360, XI0, DXI, M, N360, TROT, Order
    )

    assert np.allclose(Linear[0], Simple[0])
    assert np.array_equal(Linear[1], Simple[1])


def test_respiratory_projector_produces_a_moving_sinogram():
    Projections = ForwardProjection.RayDrivenForwProjParCartesianMultipleWithMotion(
        X0, DX, NX, Y0, DY, NY, "CardiacPhantom", 0.0, np.pi / N360, N360, XI0, DXI, M, N360, TROT
    ).reshape(N360, M)

    assert np.all(np.isfinite(Projections))
    assert Projections.max() > 0
    # Motion means the opposing half-rotation views do not repeat exactly.
    assert not np.allclose(Projections[0], Projections[N360 // 2][::-1])


# ----------------------------------------------------------- spectral projectors

def test_single_energy_bin_equals_the_monochromatic_projection():
    Mat = _material_phantom()
    N = 64
    dtheta = np.pi / N

    Spectral = ForwardProjection.SpectralRayDrivenForwProjParCartesianMultiple(
        X0, DX, NX, Y0, DY, NY, Mat, 0.0, dtheta, N, XI0, DXI, M, "kV120", 0.0, 1, 70.0
    )

    Image = MaterialProperties().MaterialToSpectralAttenuationCoefficientConversion(Mat, NX, NY, 70)
    Reference = np.asarray(ForwardProjection.RayDrivenForwProjParCartesianMultiple(
        X0, DX, NX, Y0, DY, NY, Image, 0.0, dtheta, N, XI0, DXI, M
    ))

    assert np.allclose(Spectral, Reference, atol=1e-12, rtol=0.0)


def test_polychromatic_projection_stays_within_the_monochromatic_range():
    Mat = _material_phantom()
    N = 16
    dtheta = np.pi / N
    E0, N_Ebins, deltaE = 20.0, 8, 12.0

    Spectral = ForwardProjection.SpectralRayDrivenForwProjParCartesianMultiple(
        X0, DX, NX, Y0, DY, NY, Mat, 0.0, dtheta, N, XI0, DXI, M, "kV120", E0, N_Ebins, deltaE
    )

    Properties = MaterialProperties()
    Monochromatic = np.array([
        ForwardProjection.RayDrivenForwProjParCartesianMultiple(
            X0, DX, NX, Y0, DY, NY,
            Properties.MaterialToSpectralAttenuationCoefficientConversion(
                Mat, NX, NY, int(E0 + nbin * deltaE)),
            0.0, dtheta, N, XI0, DXI, M
        ) for nbin in range(1, N_Ebins + 1)
    ])

    assert np.all(Spectral >= Monochromatic.min(axis=0) - 1e-9)
    assert np.all(Spectral <= Monochromatic.max(axis=0) + 1e-9)


def test_spectral_fan_projection_matches_its_monochromatic_limit():
    Mat = _material_phantom()
    RF = 595.0
    MFan = 101
    betaMax = np.arcsin(RM / RF)
    beta = betaMax - 2 * betaMax / (MFan - 1) * np.arange(MFan)
    N = 32
    dalpha = 2 * np.pi / N

    Spectral = ForwardProjection.SpectralRayDrivenForwardProj2DFanMultiple(
        RF, 0.0, dalpha, beta, X0, DX, NX, Y0, DY, NY, Mat, MFan, N, "kV120", 0.0, 1, 70.0
    )

    Image = MaterialProperties().MaterialToSpectralAttenuationCoefficientConversion(Mat, NX, NY, 70)
    Reference = np.asarray(ForwardProjection.RayDrivenForwardProj2DFanMultiple(
        RF, 0.0, dalpha, beta, X0, DX, NX, Y0, DY, NY, Image, MFan, N
    ))

    assert np.allclose(Spectral, Reference, atol=1e-12, rtol=0.0)
