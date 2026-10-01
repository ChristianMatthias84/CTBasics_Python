import numpy as np

from ctbasics.image_manipulation import ImageManipulation


def test_resample_2d_preserves_gradient_shape():
    image = np.arange(9, dtype=float).reshape(3, 3)
    resampled = ImageManipulation.resample_2d(image, 5, 5)

    assert resampled.shape == (5, 5)
    assert np.isfinite(resampled).all()