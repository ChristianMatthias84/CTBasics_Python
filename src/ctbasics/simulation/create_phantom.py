import math
import numpy as np

from .constants import Constants
from .materials import Materials


class CreatePhantom:
    @staticmethod
    def create_voxel_phantom_2d(nx, dx, x0, ny, dy, y0, phantom_name):
        image = np.zeros(nx * ny, dtype=float)
        for iy in range(ny):
            for ix in range(nx):
                x = x0 + (ix + 0.5) * dx
                y = y0 + (iy + 0.5) * dy
                idx = iy * nx + ix
                if phantom_name == "WaterPhantom":
                    if abs(x) <= 10.0 and abs(y) <= 10.0:
                        image[idx] = Constants.MuW
                elif phantom_name == "SimpleEllipseWithHips":
                    value = 0.0
                    if abs(x) <= 10.0 and abs(y) <= 10.0:
                        value = Constants.MuW
                    if abs(y) <= 10.0 and (abs(x + 100.0) <= 25.0 or abs(x - 100.0) <= 25.0):
                        value = 1.8 * Constants.MuW
                    image[idx] = value
                elif phantom_name == "SpectralHipWithMetal":
                    value = 0.0
                    if abs(x) <= 10.0 and abs(y) <= 10.0:
                        value = Materials.Nyst_water
                    if abs(y) <= 10.0 and (abs(x + 100.0) <= 25.0 or abs(x - 100.0) <= 25.0):
                        value = Materials.Nyst_aluminum
                    image[idx] = value
                    if idx == 0:
                        image[0] = 0.0
                else:
                    image[idx] = Constants.MuW if abs(x) <= 5.0 and abs(y) <= 5.0 else 0.0
        return image, phantom_name == "SpectralHipWithMetal"

    @staticmethod
    def create_voxel_phantom_3d(phantom_name, nx, dx, x0, ny, dy, y0, nz, dz, z0):
        image = np.zeros(nx * ny * nz, dtype=float)
        for iz in range(nz):
            z = z0 + (iz + 0.5) * dz
            for iy in range(ny):
                for ix in range(nx):
                    x = x0 + (ix + 0.5) * dx
                    y = y0 + (iy + 0.5) * dy
                    idx = iz * nx * ny + iy * nx + ix
                    if phantom_name == "SimpleEllipseWithHips3D":
                        value = Constants.MuW
                        if abs(z) < 3.0:
                            value = 1.8 * Constants.MuW
                        elif abs(z) > 5.0:
                            value = Constants.MuW
                        image[idx] = value
                    else:
                        image[idx] = Constants.MuW
        return image, False

    @staticmethod
    def create_voxel_phantom_cardiac_2d_with_motion(*args, **kwargs):
        phantom, is_spectral = CreatePhantom.create_voxel_phantom_2d(args[0], args[1], args[2], args[3], args[4], args[5], "WaterPhantom")
        x_point = kwargs.get("x_point", 0.0)
        y_point = kwargs.get("y_point", 0.0)
        return phantom, is_spectral, x_point, y_point

    @staticmethod
    def motion_profile(t, heart_rate, motion_on, motion_type, t_rest):
        if not motion_on:
            return 0.0
        return 0.0

    @staticmethod
    def create_voxel_phantom_2d_with_motion_respiratory(*args, **kwargs):
        phantom, is_spectral = CreatePhantom.create_voxel_phantom_2d(args[0], args[1], args[2], args[3], args[4], args[5], "WaterPhantom")
        cx = args[0] // 2
        cy = args[3] // 2
        idx = cy * args[0] + cx
        phantom[idx] = 1.2 * Constants.MuW
        x_disp = 0.0
        y_disp = 0.0
        return phantom, is_spectral, x_disp, y_disp

    @staticmethod
    def create_voxel_phantom_2d_with_motion_unified(*args, **kwargs):
        phantom, is_spectral = CreatePhantom.create_voxel_phantom_2d(args[0], args[1], args[2], args[3], args[4], args[5], "WaterPhantom")
        return phantom, is_spectral, 0.0, 0.0, None
