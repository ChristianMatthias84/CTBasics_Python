import numpy as np

from ctbasics.simulation import CreatePhantom


def main():
    phantom, _ = CreatePhantom.create_voxel_phantom_3d('SimpleEllipseWithHips3D', 31, 1.0, -16.0, 31, 1.0, -16.0, 8, 1.0, 0.0)
    volume = phantom.reshape(8, 31, 31)
    return {'shape': volume.shape, 'finite': bool(np.isfinite(volume).all())}


if __name__ == '__main__':
    print(main())