import numpy as np

from ctbasics.reconstruction import ForwardProjection
from ctbasics.simulation import CreatePhantom


def main():
    phantom, _ = CreatePhantom.create_voxel_phantom_2d(31, 1.0, -16.0, 31, 1.0, -16.0, 'WaterPhantom')
    beta = np.linspace(-0.2, 0.2, 9)
    proj = ForwardProjection.ray_driven_forward_proj_2d_fan_multiple(
        20.0, -0.1, 0.05, beta, -16.0, 1.0, 31, -16.0, 1.0, 31, phantom, 9, 4
    )
    return {'shape': proj.shape, 'finite': bool(np.isfinite(proj).all()), 'mean': float(proj.mean())}


if __name__ == '__main__':
    print(main())