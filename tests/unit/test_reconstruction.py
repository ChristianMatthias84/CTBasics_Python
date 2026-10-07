"""Numerical validation of the reconstruction kernels and backprojectors."""
import numpy as np
import pytest

from CTBasics.Reconstruction import Backprojection, ForwardProjection, Kernels


def test_ramlak_matches_the_closed_form_and_is_symmetric():
    dxi = 0.8
    kernel = Kernels.RamLakKernelSD(4, dxi)

    assert kernel.shape == (9,)
    assert np.allclose(kernel, kernel[::-1])
    assert np.isclose(kernel[4], 1.0 / (4.0 * dxi * dxi))
    # Even taps vanish, odd taps follow -1/(pi^2 dxi^2 m^2).
    assert np.isclose(kernel[4 + 2], 0.0)
    assert np.isclose(kernel[4 + 1], -1.0 / (np.pi**2 * dxi**2))
    assert np.isclose(kernel[4 + 3], -1.0 / (np.pi**2 * dxi**2 * 9.0))


def test_ramlak_kernel_sums_to_approximately_zero():
    kernel = Kernels.RamLakKernelSD(400, 1.0)
    # A ramp filter has no DC response, so the discrete taps must nearly cancel.
    assert abs(kernel.sum()) < 0.02


def test_shepp_logan_kernel_is_symmetric_and_negative_off_centre():
    kernel = Kernels.SheppLoganKernelSD(5, 1.0)

    assert kernel.shape == (11,)
    assert np.allclose(kernel, kernel[::-1])
    assert kernel[5] > 0.0
    assert (kernel[:5] < 0.0).all()


def test_hann_apodized_ramlak_tapers_to_zero_at_the_edges():
    M = 16
    plain = Kernels.RamLakKernelSD(M, 1.0)
    hann = Kernels.RamLakHannKernelSD(M, 1.0)

    assert hann.shape == plain.shape
    assert np.isclose(hann[0], 0.0)
    assert np.isclose(hann[-1], 0.0)
    assert np.isclose(hann[M], plain[M])
    assert np.abs(hann).max() <= np.abs(plain).max() + 1e-12


def test_moire_filter_is_a_lowpass_with_cosine_squared_roll_off():
    NFFT, ds = 64, 1.0
    moire = Kernels.GetMoireFilter(NFFT, ds, 0.1, 0.3)

    assert moire.shape == (NFFT,)
    assert np.isclose(moire[0], 1.0)
    assert (moire >= -1e-12).all()
    assert (moire <= 1.0 + 1e-12).all()


def test_backprojection_of_a_single_spike_draws_a_line():
    Nx = Ny = 41
    d = 1.0
    origin = -0.5 * Nx * d
    M, dxi = 41, 1.0
    xi0 = -0.5 * M * dxi

    projection = np.zeros(M)
    projection[M // 2] = 1.0

    image = Backprojection.PixDrivenBackProjParCartesianSingle(
        0.0, xi0, dxi, M, projection, origin, d, Nx, origin, d, Ny
    ).reshape(Ny, Nx)

    # theta = 0 smears along y, so exactly one column band carries the signal.
    column_mass = image.sum(axis=0)
    assert column_mass.max() > 0.0
    assert np.count_nonzero(column_mass > 0.1 * column_mass.max()) <= 3
    assert np.allclose(image.sum(axis=1), image[:, column_mass.argmax()].max(), atol=1e-9)


def test_fbp_chain_reconstructs_the_attenuation_of_a_disk():
    N, d, radius, mu = 129, 1.0, 40.0, 0.02
    origin = -0.5 * N * d
    grid = origin + np.arange(1, N + 1) * d
    X, Y = np.meshgrid(grid, grid)
    image = np.where(X**2 + Y**2 <= radius**2, mu, 0.0).ravel()

    M, dxi = 129, 1.0
    xi0 = -0.5 * M * dxi
    Nproj, N2pi = 180, 360
    theta0, dtheta = 0.0, np.pi / Nproj

    sino = ForwardProjection.RayDrivenForwProjParCartesianMultiple(
        origin, d, N, origin, d, N, image, theta0, dtheta, Nproj, xi0, dxi, M
    )

    kernel = Kernels.RamLakKernelSD(M // 2, dxi)
    filtered = np.zeros_like(sino)
    for n in range(Nproj):
        filtered[n * M:(n + 1) * M] = np.convolve(sino[n * M:(n + 1) * M], kernel, mode="same") * dxi

    recon = Backprojection.WFBP2DFullyPixelDriven(
        theta0, dtheta, Nproj, N2pi, xi0, dxi, M, filtered, origin, d, N, origin, d, N
    ).reshape(N, N)

    interior = X**2 + Y**2 <= (radius - 10.0) ** 2
    exterior = X**2 + Y**2 >= (radius + 10.0) ** 2

    assert np.isclose(recon[interior].mean(), mu, rtol=0.02)
    assert recon[interior].std() < 0.02 * mu
    assert abs(recon[exterior].mean()) < 0.1 * mu


def test_wfbp_and_pixel_driven_backprojection_agree_over_a_half_turn():
    N, d = 31, 1.0
    origin = -0.5 * N * d
    M, dxi = 31, 1.0
    xi0 = -0.5 * M * dxi
    Nproj, N2pi = 12, 24
    theta0, dtheta = 0.0, np.pi / Nproj

    rng = np.random.default_rng(0)
    sino = rng.random(Nproj * M)

    wfbp = Backprojection.WFBP2DFullyPixelDriven(
        theta0, dtheta, Nproj, N2pi, xi0, dxi, M, sino, origin, d, N, origin, d, N
    )
    pixel_driven = Backprojection.PixDrivenBackProjParCartesian(
        theta0, dtheta, Nproj, N2pi, xi0, dxi, M, sino, origin, d, N, origin, d, N
    )

    # With one projection per half-turn group both weightings reduce to the same sum.
    assert np.allclose(wfbp, pixel_driven)
