import numpy as np

from ctbasics.reconstruction import Kernels, Rebinning


def main():
    kernel = Kernels.ramlak_kernel_sd(1, 1.0)
    sino = np.linspace(0.0, 1.0, 12)
    filtered = Rebinning.rebinning_fan_to_parallel(sino, 1.0, 3, 4, 0.0, 1.0, np.array([-0.1, 0.0, 0.1]), 3, 2, 0.0, 1.0, -1.0, 1.0)
    return {'kernel_shape': kernel.shape, 'filtered_shape': filtered.shape, 'finite': bool(np.isfinite(filtered).all())}


if __name__ == '__main__':
    print(main())