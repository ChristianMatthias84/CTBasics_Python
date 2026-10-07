"""Resampling helpers.

MATLAB builds the query grid as `1:sz/N:sz`, so the sample count only equals N when
`sz/N` divides `sz-1` exactly. The tests pin that behaviour down.
"""
import numpy as np
import pytest

from CTBasics.ImageManipulation.ImageManipulation import ImageManipulation


def test_downsampling_by_an_integer_factor_picks_every_other_sample():
    image = np.arange(64, dtype=float).reshape(8, 8)

    resampled = ImageManipulation.Resample2D(image, 4, 4)

    assert resampled.shape == (4, 4)
    # Query grid 1, 3, 5, 7 lands exactly on the original samples.
    assert np.array_equal(resampled, image[::2, ::2])


def test_resampling_to_the_same_size_is_the_identity():
    rng = np.random.default_rng(0)
    image = rng.standard_normal((6, 5))

    resampled = ImageManipulation.Resample2D(image, 6, 5)

    assert resampled.shape == image.shape
    assert np.allclose(resampled, image)


def test_upsampling_stops_short_of_the_last_row():
    image = np.arange(9, dtype=float).reshape(3, 3)

    resampled = ImageManipulation.Resample2D(image, 5, 5)

    # MATLAB `1:0.6:3` yields 1, 1.6, 2.2, 2.8 -- four samples, not five.
    assert resampled.shape == (4, 4)
    assert np.isfinite(resampled).all()
    assert resampled[0, 0] == pytest.approx(image[0, 0])


def test_interpolation_is_linear_between_the_samples():
    image = np.array([[0.0, 0.0], [0.0, 0.0], [10.0, 10.0], [10.0, 10.0]])

    resampled = ImageManipulation.Resample2D(image, 8, 2)

    # Query step is 0.5, so the sample between rows 2 and 3 must be the midpoint.
    assert resampled[3, 0] == pytest.approx(5.0)


def test_resample_3d_applies_the_2d_resampling_per_slice():
    rng = np.random.default_rng(1)
    volume = rng.standard_normal((8, 8, 3))

    resampled = ImageManipulation.Resample3D(volume, 4, 4, 3)

    assert resampled.shape == (4, 4, 3)
    for nz in range(3):
        expected = ImageManipulation.Resample2D(volume[:, :, nz], 4, 4)
        assert np.array_equal(resampled[:, :, nz], expected)
