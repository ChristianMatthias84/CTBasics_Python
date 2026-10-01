import numpy as np

from ctbasics.reconstruction import Backprojection, Kernels, Rebinning


def test_ramlak_kernel_is_symmetric_and_centered():
    kernel = Kernels.ramlak_kernel_sd(3, 1.0)

    assert kernel.shape == (7,)
    assert np.isclose(kernel[3], 0.25)
    assert np.allclose(kernel, kernel[::-1])


def test_convolution_slice_rebins_and_reconstructs_shape():
    sino = np.linspace(0.0, 1.0, 12)
    kernel = Kernels.ramlak_kernel_sd(1, 1.0)
    filtered = Rebinning.rebinning_fan_to_parallel(sino, 1.0, 3, 4, 0.0, 1.0, np.array([-0.1, 0.0, 0.1]), 3, 2, 0.0, 1.0, -1.0, 1.0)

    assert filtered.shape == (6,)

    recon = Backprojection.wfbp2d_fully_pixel_driven(0.0, 1.0, 2, 2, -1.0, 1.0, 3, np.arange(6, dtype=float), -1.0, 1.0, 2, -1.0, 1.0, 1)
    assert recon.shape == (2,)