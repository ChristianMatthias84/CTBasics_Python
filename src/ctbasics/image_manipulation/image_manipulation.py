from __future__ import annotations

import numpy as np
from scipy.interpolate import RegularGridInterpolator


class ImageManipulation:
    @staticmethod
    def resample_2d(image_matrix_2d_in, nx_new: int, ny_new: int):
        image = np.asarray(image_matrix_2d_in, dtype=float)
        sx, sy = image.shape
        xg = np.arange(1, sx + 1, dtype=float)
        yg = np.arange(1, sy + 1, dtype=float)
        interpolator = RegularGridInterpolator((xg, yg), image, bounds_error=False, fill_value=None)
        xq = np.linspace(1, sx, nx_new)
        yq = np.linspace(1, sy, ny_new)
        xx, yy = np.meshgrid(xq, yq, indexing="ij")
        result = interpolator(np.column_stack([xx.ravel(order="C"), yy.ravel(order="C")]))
        return result.reshape(nx_new, ny_new)

    @staticmethod
    def resample_3d(image_matrix_3d_in, nx_new: int, ny_new: int, nz: int):
        image = np.asarray(image_matrix_3d_in)
        result = np.zeros((nx_new, ny_new, nz), dtype=float)
        for k in range(nz):
            result[:, :, k] = ImageManipulation.resample_2d(image[:, :, k], nx_new, ny_new)
        return result