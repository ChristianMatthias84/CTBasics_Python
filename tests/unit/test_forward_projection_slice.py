import numpy as np

from ctbasics.reconstruction import Backprojection, ForwardProjection, Kernels, Rebinning
from ctbasics.simulation import CreatePhantom


def test_fan_forward_projection_produces_finite_projections():
    phantom, _ = CreatePhantom.create_voxel_phantom_2d(31, 1.0, -16.0, 31, 1.0, -16.0, "WaterPhantom")
    beta = np.linspace(-0.2, 0.2, 9)
    projections = ForwardProjection.ray_driven_forward_proj_2d_fan_multiple(20.0, -0.1, 0.05, beta, -16.0, 1.0, 31, -16.0, 1.0, 31, phantom, 9, 4)

    assert projections.shape == (36,)
    assert np.isfinite(projections).all()


def test_simple_projection_rebin_and_backproject_chain_runs():
    sino = np.arange(12, dtype=float)
    rebinned = Rebinning.rebinning_fan_to_parallel(sino, 20.0, 3, 4, -0.5, 0.5, np.array([-0.1, 0.0, 0.1]), 3, 2, -0.5, 0.5, -1.0, 0.2)
    kernel = Kernels.ramlak_kernel_sd(1, 1.0)
    assert kernel.shape == (3,)
    recon = Backprojection.wfbp2d_fully_pixel_driven(-0.5, 0.5, 2, 4, -1.0, 0.2, 3, rebinned, -1.0, 1.0, 2, -1.0, 1.0, 2)

    assert recon.shape == (4,)
    assert np.isfinite(recon).all()