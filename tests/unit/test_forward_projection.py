"""Numerical validation of the ray-driven forward projectors against analytic solutions."""
import numpy as np
import pytest

from CTBasics.Reconstruction import ForwardProjection


def _disk(N, d, origin, radius, mu):
    grid = origin + np.arange(1, N + 1) * d
    X, Y = np.meshgrid(grid, grid)
    return np.where(X**2 + Y**2 <= radius**2, mu, 0.0).ravel()


@pytest.mark.parametrize("theta", [0.0, np.pi / 6.0, np.pi / 3.0, np.pi / 2.0, 2.0])
def test_parallel_projection_of_disk_matches_analytic_chord_length(theta):
    N, d, radius, mu = 161, 1.0, 40.0, 0.02
    origin = -0.5 * N * d
    image = _disk(N, d, origin, radius, mu)

    M, dxi = 141, 1.0
    xi0 = -0.5 * M * dxi

    projection = ForwardProjection.RayDrivenForwProjParCartesianSingle(
        origin, d, N, origin, d, N, image, theta, xi0, dxi, M
    )

    xi = xi0 + np.arange(1, M + 1) * dxi
    analytic = mu * 2.0 * np.sqrt(np.clip(radius**2 - xi**2, 0.0, None))
    interior = np.abs(xi) < radius - 3.0

    # 1.5% tolerance covers the partial-volume error at the voxelized disk boundary.
    assert np.abs(projection[interior] - analytic[interior]).max() < 0.015 * analytic[interior].max()


def test_parallel_projection_is_zero_outside_the_object():
    N, d, radius, mu = 81, 1.0, 20.0, 0.02
    origin = -0.5 * N * d
    image = _disk(N, d, origin, radius, mu)

    M, dxi = 81, 1.0
    xi0 = -0.5 * M * dxi
    projection = ForwardProjection.RayDrivenForwProjParCartesianSingle(
        origin, d, N, origin, d, N, image, 0.3, xi0, dxi, M
    )

    xi = xi0 + np.arange(1, M + 1) * dxi
    assert np.allclose(projection[np.abs(xi) > radius + 2.0], 0.0)


def test_parallel_projection_total_mass_is_angle_invariant():
    N, d, radius, mu = 101, 1.0, 25.0, 0.02
    origin = -0.5 * N * d
    image = _disk(N, d, origin, radius, mu)
    M, dxi = 101, 1.0
    xi0 = -0.5 * M * dxi

    masses = [
        ForwardProjection.RayDrivenForwProjParCartesianSingle(
            origin, d, N, origin, d, N, image, theta, xi0, dxi, M
        ).sum() * dxi
        for theta in (0.0, 0.4, 0.9, 1.3)
    ]
    expected = mu * np.pi * radius**2

    assert np.allclose(masses, expected, rtol=0.02)


def test_parallel_multiple_stacks_single_projections_in_order():
    N, d, radius, mu = 41, 1.0, 12.0, 0.02
    origin = -0.5 * N * d
    image = _disk(N, d, origin, radius, mu)
    M, dxi, Nproj = 41, 1.0, 4
    xi0 = -0.5 * M * dxi
    theta0, dtheta = 0.0, np.pi / Nproj

    sino = ForwardProjection.RayDrivenForwProjParCartesianMultiple(
        origin, d, N, origin, d, N, image, theta0, dtheta, Nproj, xi0, dxi, M
    )

    assert sino.shape == (Nproj * M,)
    for n in range(1, Nproj + 1):
        single = ForwardProjection.RayDrivenForwProjParCartesianSingle(
            origin, d, N, origin, d, N, image, theta0 + n * dtheta, xi0, dxi, M
        )
        assert np.allclose(sino[(n - 1) * M:n * M], single)


def test_fan_projection_of_centered_disk_matches_parallel_geometry():
    N, d, radius, mu = 121, 1.0, 30.0, 0.02
    origin = -0.5 * N * d
    image = _disk(N, d, origin, radius, mu)

    RF = 600.0
    M = 101
    beta = np.linspace(-0.06, 0.06, M)

    projection = ForwardProjection.RayDrivenForwardProj2DFanSingle(
        RF, 0.0, beta, origin, d, N, origin, d, N, image, M
    )

    # For a fan beam the ray at angle beta has distance xi = -RF*sin(beta) from the centre.
    xi = -RF * np.sin(beta)
    analytic = mu * 2.0 * np.sqrt(np.clip(radius**2 - xi**2, 0.0, None))
    interior = np.abs(xi) < radius - 3.0

    assert np.abs(projection[interior] - analytic[interior]).max() < 0.02 * analytic[interior].max()


def test_fan_projection_rotation_error_converges_with_grid_refinement():
    """A voxelized disk is only approximately rotation invariant.

    The pointwise deviation between two source angles must therefore shrink as the
    grid is refined; the integrated mass is already angle invariant on a coarse grid.
    """
    radius, mu = 25.0, 0.02
    RF, M = 600.0, 61
    beta = np.linspace(-0.04, 0.04, M)

    deviations = []
    for N, d in ((101, 1.0), (201, 0.5), (401, 0.25)):
        origin = -0.5 * N * d
        image = _disk(N, d, origin, radius, mu)

        reference = ForwardProjection.RayDrivenForwardProj2DFanSingle(
            RF, 0.0, beta, origin, d, N, origin, d, N, image, M
        )
        rotated = ForwardProjection.RayDrivenForwardProj2DFanSingle(
            RF, 1.0, beta, origin, d, N, origin, d, N, image, M
        )

        deviations.append(np.abs(reference - rotated).max() / reference.max())
        assert abs(reference.sum() - rotated.sum()) < 0.002 * reference.sum()

    assert deviations[0] > deviations[1] > deviations[2]
    assert deviations[-1] < 0.01


def test_static_ct_projection_sees_the_object_and_respects_the_fan_limit():
    N, d, radius, mu = 81, 1.0, 20.0, 0.02
    origin = -0.5 * N * d
    image = _disk(N, d, origin, radius, mu)

    RF, RD, MDelta = 586.3, 425.0, 64
    dDelta = 2.0 * np.pi / MDelta

    projection = ForwardProjection.RayDrivenForwardProj2DStaticCTSingle(
        RF, RD, np.pi / 2.0, 0.0, dDelta, origin, d, N, origin, d, N, image, MDelta
    )

    assert projection.shape == (MDelta,)
    assert np.isfinite(projection).all()
    # Rays outside the +/-pi/2 acceptance stay exactly zero, rays through the disk do not.
    assert np.count_nonzero(projection) > 0
    assert np.count_nonzero(projection == 0.0) > 0
    assert projection.max() <= mu * 2.0 * radius * 1.1
