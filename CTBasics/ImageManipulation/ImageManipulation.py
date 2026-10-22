import numpy as np
from scipy.interpolate import RegularGridInterpolator

from CTBasics.Memory.Allocation import Allocation


class ImageManipulation:
    @staticmethod
    def Resample2D(ImageMatrix2D_in, Nx_new, Ny_new):
        """Resamples a 2D image onto a new grid with linear interpolation.

        MATLAB builds the query grid as `1:sz/N:sz`, i.e. a fixed step starting at 1.
        The last sample therefore lands at `sz + 1 - sz/N`, not at `sz`.
        """
        ImageMatrix2D_in = np.asarray(ImageMatrix2D_in, dtype=float)
        sz = ImageMatrix2D_in.shape

        xg = np.arange(1, sz[0] + 1, dtype=float)
        yg = np.arange(1, sz[1] + 1, dtype=float)
        F = RegularGridInterpolator((xg, yg), ImageMatrix2D_in,
                                    bounds_error=False, fill_value=None)

        xq = np.arange(1.0, sz[0] + 1e-9, sz[0] / Nx_new)
        yq = np.arange(1.0, sz[1] + 1e-9, sz[1] / Ny_new)

        XQ, YQ = np.meshgrid(xq, yq, indexing="ij")
        vq = F(np.stack((XQ.ravel(), YQ.ravel()), axis=-1)).reshape(xq.size, yq.size)

        return vq

    @staticmethod
    def Resample3D(ImageMatrix3D_in, Nx_new, Ny_new, Nz):
        """Resamples every slice of a volume independently."""
        ImageMatrix3D_in = np.asarray(ImageMatrix3D_in, dtype=float)
        Result = Allocation.zeros(Nx_new, Ny_new, Nz)
        for nz in range(Nz):
            Result[:, :, nz] = ImageManipulation.Resample2D(
                ImageMatrix3D_in[:, :, nz], Nx_new, Ny_new
            )
        return Result
