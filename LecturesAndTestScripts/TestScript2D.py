from ctbasics.simulation import CreatePhantom


def main():
    phantom, is_spectral = CreatePhantom.create_voxel_phantom_2d(301, 1.0, -151.0, 301, 1.0, -151.0, 'WaterPhantom')
    return {'shape': phantom.shape, 'is_spectral': bool(is_spectral), 'center': float(phantom.reshape(301, 301)[150, 150])}


if __name__ == '__main__':
    print(main())