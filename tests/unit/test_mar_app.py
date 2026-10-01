import numpy as np

from ctbasics.apps import MAR


def test_mar_threshold_and_prior_helpers():
    mar = MAR(0.0, 1.0, 3, 0.0, 1.0, 3, np.array([0.0, 2500.0, 500.0, 3200.0, 0.0, 100.0, 0.0, 0.0, 0.0]), 0.0, 1.0, 1, 0.0, 1.0, 5)

    metal_image = mar.threshold_metal_image()
    prior_image = mar.create_prior_image(np.array([0.0, 500.0, 1000.0, 3200.0, 0.0, 100.0, 0.0, 0.0, 0.0]))

    assert metal_image.shape == (9,)
    assert metal_image[1] == 3500.0
    assert metal_image[3] == 4200.0
    assert prior_image[2] == 1000.0
    assert prior_image[3] == 3200.0


def test_mar_interpolation_helpers_fill_metal_traces():
    mar = MAR(0.0, 1.0, 2, 0.0, 1.0, 2, np.zeros(4), 0.0, 1.0, 1, 0.0, 1.0, 4)
    projections = np.array([0.0, 10.0, 20.0, 30.0])
    metal_projections = np.array([0.0, 1.0, 1.0, 0.0])
    prior_sinogram = np.zeros(4)

    interpolated = mar.metal_trace_interpolation(projections, metal_projections)
    normalized = mar.normalized_metal_trace_interpolation(projections, metal_projections, prior_sinogram)

    assert np.isclose(interpolated[1], 10.0)
    assert np.isclose(interpolated[2], 20.0)
    assert normalized.shape == projections.shape


def test_mar_apply_runs_on_tiny_geometry():
    mar = MAR(
        0.0,
        1.0,
        2,
        0.0,
        1.0,
        2,
        np.array([0.0, 2500.0, 0.0, 0.0]),
        0.0,
        np.pi / 2.0,
        2,
        -1.5,
        1.0,
        4,
    )

    result = mar.apply()

    assert result.shape == (4,)
    assert np.isfinite(result).all()