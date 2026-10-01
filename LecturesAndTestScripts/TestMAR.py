import numpy as np

from ctbasics.apps import MAR


def main():
    mar = MAR(0.0, 1.0, 3, 0.0, 1.0, 3, np.array([0.0, 2500.0, 500.0, 3200.0, 0.0, 100.0, 0.0, 0.0, 0.0]), 0.0, 1.0, 1, 0.0, 1.0, 5)
    metal_image = mar.threshold_metal_image()
    return {'shape': metal_image.shape, 'values': metal_image.tolist()}


if __name__ == '__main__':
    print(main())