"""Analytic checks of the cone-beam (3D) forward projectors."""
import numpy as np
import pytest

from CTBasics.Reconstruction.ForwardProjection import ForwardProjection

NX = NY = 201
DX = DY = 1.0
X0 = Y0 = -NX / 2 * DX - DX / 2

NZ = 41
DZ = 2.0
Z0 = -NZ / 2 * DZ - DZ / 2

R = 60.0          # cylinder radius in mm
RF = 570.0        # focus to iso
RD = 500.0        # detector ring radius of the static CT geometry

M = 401           # fan channels
NS = 8            # detector rows
DS = 1.0
S0 = -NS / 2 * DS - DS / 2


def _disc():
    axis = X0 + np.arange(1, NX + 1) * DX
    X, Y = np.meshgrid(axis, axis)
    return X * X + Y * Y <= R * R


def _cylinder(value=1.0):
    volume = np.zeros((NZ, NY, NX))
    volume[:, _disc()] = value
    return volume.ravel()


def _z_ramp():
    """Every slice carries its own index as attenuation, so z sampling becomes visible."""
    volume = np.zeros((NZ, NY, NX))
    disc = _disc()
    for nz in range(NZ):
        volume[nz][disc] = float(nz)
    return volume.ravel()


def _beta():
    betaMax = np.arcsin(250.0 / RF)
    return betaMax - 2 * betaMax / (M - 1) * np.arange(M)


def _fan(Ima, alpha=0.0, zsource=0.0):
    return ForwardProjection.RayDrivenForwardProj3DFanSingle(
        RF, alpha, _beta(), S0, DS, X0, DX, NX, Y0, DY, NY,
        Z0, DZ, NZ, Ima, M, NS, zsource
    ).reshape(NS, M)


def test_fan_cone_beam_matches_the_analytic_chord_length():
    Rows = _fan(_cylinder())

    beta = _beta()
    xi = -RF * np.sin(beta)
    inside = np.abs(xi) < 0.9 * R
    chord = 2 * np.sqrt(np.clip(R * R - xi * xi, 0.0, None))

    for row in range(NS):
        deviation = np.abs(Rows[row][inside] - chord[inside]) / chord[inside]
        # 1 mm voxels on a 60 mm cylinder leave about one percent discretization error.
        assert deviation.max() < 0.02


def test_fan_cone_beam_is_invariant_along_z_for_a_cylinder():
    Rows = _fan(_cylinder())

    # The cylinder is z-invariant, so the tilted rows may only differ by the sub-voxel
    # z interpolation, which stays far below the 1 mm in-plane voxel size.
    assert np.abs(Rows - Rows[0]).max() < 0.01


def test_detector_rows_sample_increasing_z():
    Rows = _fan(_z_ramp())

    central = Rows[:, M // 2]

    # The cone opens in z, so each row integrates a higher slice index than the previous.
    assert np.all(np.diff(central) > 0)


def test_source_z_position_shifts_the_sampled_slice():
    central = [_fan(_z_ramp(), zsource=zs)[0, M // 2] / (2 * R) for zs in (0.0, 10.0, 20.0)]

    # A 10 mm focus shift must move the sampled slice index by exactly 10 mm / dz.
    assert central[1] - central[0] == pytest.approx(10.0 / DZ, rel=0.02)
    assert central[2] - central[1] == pytest.approx(10.0 / DZ, rel=0.02)


def test_empty_volume_projects_to_zero():
    Rows = _fan(np.zeros(NX * NY * NZ))

    assert np.count_nonzero(Rows) == 0


def test_attenuation_scales_linearly():
    single = _fan(_cylinder(1.0))
    double = _fan(_cylinder(2.0))

    assert np.allclose(double, 2 * single, rtol=1e-12)


def test_static_ct_cone_beam_matches_the_analytic_chord_length():
    MDelta = 721
    dDelta = 2 * np.pi / MDelta

    Projection = ForwardProjection.RayDrivenForwardProj3DStaticCTSingle(
        RF, RD, 0.0, -dDelta, dDelta, S0, DS, 0.0,
        X0, DX, NX, Y0, DY, NY, Z0, DZ, NZ, _cylinder(), MDelta, NS, 0.0
    ).reshape(NS, MDelta)

    assert Projection.max() == pytest.approx(2 * R, rel=0.02)

    # The arbitrary fan limitation blocks every element that would be hit from behind.
    active = np.count_nonzero(Projection[0] > 0)
    assert 0 < active < MDelta


def test_static_ct_rows_are_invariant_for_a_cylinder():
    MDelta = 361
    dDelta = 2 * np.pi / MDelta

    Projection = ForwardProjection.RayDrivenForwardProj3DStaticCTSingle(
        RF, RD, 0.0, -dDelta, dDelta, S0, DS, 0.0,
        X0, DX, NX, Y0, DY, NY, Z0, DZ, NZ, _cylinder(), MDelta, NS, 0.0
    ).reshape(NS, MDelta)

    assert np.abs(Projection - Projection[0]).max() < 0.01


def test_multiple_sequence_stacks_all_readings():
    N = 4
    dalpha = 2 * np.pi / N

    Projections = ForwardProjection.RayDrivenForwardProj3DFanMultipleSequence(
        RF, 0.0, dalpha, _beta(), S0, DS, X0, DX, NX, Y0, DY, NY,
        Z0, DZ, NZ, _cylinder(), M, NS, N, 0.0
    )

    assert Projections.shape == (N * M * NS,)

    block = M * NS
    for n in range(1, N + 1):
        reading = Projections[(n - 1) * block:n * block]
        expected = _fan(_cylinder(), alpha=n * dalpha).ravel()
        assert np.allclose(reading, expected, rtol=1e-12)


def test_rotating_a_cylinder_converges_to_a_rotation_invariant_projection():
    """A centred cylinder is rotationally symmetric, so views may only differ by the
    voxelisation of the circle. The residual must shrink with the voxel size."""
    beta = _beta()
    inside = np.abs(-RF * np.sin(beta)) < 0.9 * R

    nz, dz = 21, 2.0
    z0 = -nz / 2 * dz - dz / 2

    def worst_deviation(N):
        d = 160.0 / N
        origin = -N / 2 * d - d / 2
        axis = origin + np.arange(1, N + 1) * d
        X, Y = np.meshgrid(axis, axis)

        volume = np.zeros((nz, N, N))
        volume[:, X * X + Y * Y <= R * R] = 1.0
        volume = volume.ravel()

        def project(alpha):
            return ForwardProjection.RayDrivenForwardProj3DFanSingle(
                RF, alpha, beta, S0, DS, origin, d, N, origin, d, N,
                z0, dz, nz, volume, M, NS, 0.0
            ).reshape(NS, M)[:, inside]

        reference = project(0.0)
        return max(np.abs(project(alpha) - reference).max() / reference.min()
                   for alpha in (0.3, 1.1, 2.7))

    deviations = [worst_deviation(N) for N in (101, 201, 401)]

    assert all(a > b for a, b in zip(deviations, deviations[1:]))
    assert deviations[-1] < 0.01
