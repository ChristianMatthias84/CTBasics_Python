import numpy as np

from ctbasics.reconstruction import ForwardProjection


def test_motion_dual_forward_projection_wrapper_returns_expected_shapes():
    theta_array = np.array([0.0, np.pi / 2.0, np.pi, 3.0 * np.pi / 2.0])
    projections, angles_new, points, phantom_truth, n_t_phases = ForwardProjection.ray_driven_forward_proj_par_cartesian_multiple_with_motion_dual(
        -1.0,
        1.0,
        2,
        -1.0,
        1.0,
        2,
        4,
        -1.5,
        1.0,
        4,
        4,
        1.0,
        theta_array,
        SelectedPhantom="Cardiac",
        Motion=False,
    )

    assert projections.shape == (1, 16)
    assert angles_new.shape == (4,)
    assert points.shape == (2, 4)
    assert phantom_truth.shape[1] == 4
    assert n_t_phases.ndim == 1