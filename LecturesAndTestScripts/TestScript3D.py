from ctbasics.simulation import CreatePhantom


def main():
    phantom, is_spectral = CreatePhantom.create_voxel_phantom_3d('SimpleEllipseWithHips3D', 31, 1.0, -16.0, 31, 1.0, -16.0, 10, 1.0, 0.0)
    volume = phantom.reshape(10, 31, 31)
    return {'shape': volume.shape, 'is_spectral': bool(is_spectral)}


if __name__ == '__main__':
    print(main())