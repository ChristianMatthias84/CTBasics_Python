import numpy as np


class Kernels:
    @staticmethod
    def ramlak_kernel_sd(order, spacing):
        if order < 0:
            order = 0
        support = 2 * order + 1
        kernel = np.zeros(support, dtype=float)
        center = order
        for i in range(support):
            x = (i - center) * spacing
            if x == 0:
                kernel[i] = 0.25
            else:
                kernel[i] = np.sin(x) / x
        kernel[center] = 0.25
        return kernel
