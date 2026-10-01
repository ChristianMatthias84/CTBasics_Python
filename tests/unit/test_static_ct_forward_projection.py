import numpy as np

from ctbasics.reconstruction import ForwardProjection


def test_static_ct_forward_projection_returns_finite_projection():
    image = np.zeros(16)
    image[5] = 1.0

    projection = ForwardProjection.ray_driven_forward_proj_2d_static_ct_single(
        580.0,
        425.0,
        np.pi / 2.0,
        0.0,
        2.0 * np.pi / 8.0,
        -1.0,
        1.0,
        4,
        -1.0,
        1.0,
        4,
        image,
        8,
    )

    assert projection.shape == (8,)
    assert np.isfinite(projection).all()