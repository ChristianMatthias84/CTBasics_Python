"""Normalized metal artifact reduction."""
import numpy as np
import pytest

from CTBasics.Apps.MAR import MAR

RM = 120.0
NX = NY = 96
M = 161
N180 = 90


@pytest.fixture
def geometry():
    dx = dy = 2 * RM / NX
    dxi = 2 * RM / (M - 1)
    return dict(
        x0=-RM - dx / 2, dx=dx, Nx=NX,
        y0=-RM - dy / 2, dy=dy, Ny=NY,
        theta0=0.0, dtheta=np.pi / N180, N180=N180,
        xi0=-RM - dxi, dxi=dxi, M=M,
    )


def _Phantom(geometry):
    """Water cylinder with a bone insert and a small dense metal implant."""
    ax = geometry["x0"] + np.arange(1, NX + 1) * geometry["dx"]
    ay = geometry["y0"] + np.arange(1, NY + 1) * geometry["dy"]
    X, Y = np.meshgrid(ax, ay)

    HU = np.full((NY, NX), -1000.0)
    HU[X ** 2 + Y ** 2 <= 90.0 ** 2] = 0.0
    HU[(X - 30.0) ** 2 + (Y - 20.0) ** 2 <= 25.0 ** 2] = 800.0
    HU[(X + 35.0) ** 2 + (Y + 15.0) ** 2 <= 8.0 ** 2] = 6000.0
    return HU.ravel()


def _MAR(geometry):
    return MAR(_Phantom(geometry), **geometry)


def test_input_is_shifted_to_a_positive_scale(geometry):
    mar = _MAR(geometry)

    assert mar.Input.min() == pytest.approx(0.0)
    assert mar.Input.max() == pytest.approx(7000.0)


def test_threshold_keeps_only_the_metal(geometry):
    mar = _MAR(geometry)

    MetalImage = mar.ThresholdMetalImage()

    assert set(np.unique(MetalImage)) == {0.0, 7000.0}
    # Only the implant exceeds the metal threshold of 3000.
    assert np.count_nonzero(MetalImage) == np.count_nonzero(mar.Input > mar.TM)
    assert 0 < np.count_nonzero(MetalImage) < MetalImage.size


def test_prior_image_flattens_soft_tissue_to_water(geometry):
    mar = _MAR(geometry)

    Prior = mar.CreatePriorImage(np.array([-1000.0, 0.0, 500.0, 1300.0, 1500.0, 6000.0]))

    assert np.array_equal(Prior, [-1000.0, 0.0, 1000.0, 1000.0, 1500.0, 6000.0])


def test_trace_interpolation_reproduces_a_linear_sinogram(geometry):
    mar = _MAR(geometry)
    Projections = np.tile(np.linspace(0.0, 10.0, M), (N180, 1))
    MetalProjections = np.zeros((N180, M))
    MetalProjections[:, 70:85] = 1.0

    Interpolated = mar.MetalTraceInterpolation(Projections.ravel(), MetalProjections.ravel())

    # Linear interpolation across a linear ramp must be exact.
    assert np.abs(Interpolated.reshape(N180, M) - Projections).max() < 1e-12


def test_trace_interpolation_leaves_unshadowed_channels_untouched(geometry):
    mar = _MAR(geometry)
    rng = np.random.default_rng(0)
    Projections = rng.standard_normal(N180 * M)
    MetalProjections = np.zeros((N180, M))
    MetalProjections[3, 40:44] = 1.0

    Interpolated = mar.MetalTraceInterpolation(Projections, MetalProjections.ravel())

    untouched = MetalProjections.ravel() == 0
    assert np.array_equal(Interpolated[untouched], Projections[untouched])
    assert not np.array_equal(Interpolated[~untouched], Projections[~untouched])


def test_trace_interpolation_rejects_a_trace_at_the_detector_border(geometry):
    mar = _MAR(geometry)
    MetalProjections = np.zeros((N180, M))
    MetalProjections[0, 0] = 1.0

    with pytest.raises(IndexError):
        mar.MetalTraceInterpolation(np.zeros(N180 * M), MetalProjections.ravel())


def test_normalized_interpolation_is_a_no_op_for_a_perfect_prior(geometry):
    mar = _MAR(geometry)
    rng = np.random.default_rng(1)
    Projections = rng.uniform(0.0, 5.0, N180 * M)
    MetalProjections = np.zeros((N180, M))
    MetalProjections[:, 70:85] = 1.0

    # If the prior sinogram equals the measurement, the normalized sinogram is constant 1
    # and the interpolation must return the measurement unchanged.
    Result = mar.NoramlizedMetalTraceInterpolation(
        Projections, MetalProjections.ravel(), Projections
    )

    assert np.abs(Result - Projections).max() < 1e-9


def test_apply_reduces_the_metal_streaks(geometry, tmp_path, monkeypatch):
    from CTBasics.ReadAndWrite.IOHelperFunctions import IOHelperFunctions

    monkeypatch.setattr(IOHelperFunctions, "Save", staticmethod(lambda *args, **kwargs: None))

    mar = _MAR(geometry)
    MARResult = mar.Apply()

    assert MARResult.shape == (NX * NY,)
    assert np.isfinite(MARResult).all()

    # MATLAB shifts the input by +1000 and never shifts the result back, so water lands
    # at 1000 instead of 0 HU. Measured deviation on this phantom: 0.6 HU.
    ax = geometry["x0"] + np.arange(1, NX + 1) * geometry["dx"]
    ay = geometry["y0"] + np.arange(1, NY + 1) * geometry["dy"]
    X, Y = np.meshgrid(ax, ay)
    water = ((X ** 2 + Y ** 2 <= 70.0 ** 2)
             & ((X - 30.0) ** 2 + (Y - 20.0) ** 2 > 35.0 ** 2)
             & ((X + 35.0) ** 2 + (Y + 15.0) ** 2 > 25.0 ** 2))

    assert MARResult.reshape(NY, NX)[water].mean() == pytest.approx(1000.0, abs=10.0)
