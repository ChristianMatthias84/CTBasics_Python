import numpy as np

from ctbasics.reconstruction import Backprojection, Rebinning


def main():
    sino = np.arange(12, dtype=float)
    rebinned = Rebinning.rebinning_fan_to_parallel(sino, 20.0, 3, 4, -0.5, 0.5, np.array([-0.1, 0.0, 0.1]), 3, 2, -0.5, 0.5, -1.0, 0.2)
    recon = Backprojection.wfbp2d_fully_pixel_driven(-0.5, 0.5, 2, 4, -1.0, 0.2, 3, rebinned, -1.0, 1.0, 2, -1.0, 1.0, 2)
    return {'shape': recon.shape, 'finite': bool(np.isfinite(recon).all())}


if __name__ == '__main__':
    print(main())