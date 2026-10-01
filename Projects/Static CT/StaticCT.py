import numpy as np

from ctbasics.simulation import CreatePhantom


def main():
    phantom, _ = CreatePhantom.create_voxel_phantom_2d(41, 1.0, -20.5, 41, 1.0, -20.5, 'WaterPhantom')
    image = phantom.reshape(41, 41)
    return {'shape': image.shape, 'center': float(image[20, 20]), 'finite': bool(np.isfinite(image).all())}


if __name__ == '__main__':
    print(main())