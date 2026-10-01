import numpy as np

from ctbasics.reconstruction import ForwardProjection


def test_parallel_motion_forward_projection_returns_expected_length():
    projections = ForwardProjection.ray_driven_forward_proj_par_cartesian_multiple_with_motion(
        -1.0,
        1.0,
        2,
        -1.0,
        1.0,
        2,
        "CardiacPhantom",
        0.0,
        np.pi / 2.0,
        2,
        -1.5,
        1.0,
        4,
        4,
        1.0,
        Motion=False,
    )

    assert projections.shape == (8,)
    assert np.isfinite(projections).all()