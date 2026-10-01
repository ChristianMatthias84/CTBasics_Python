import numpy as np

from .precision import Precision


class Allocation:
    @staticmethod
    def zeros(*shape):
        dtype = np.float32 if Precision.value() == 32 else np.float64
        return np.zeros(shape, dtype=dtype)

    @staticmethod
    def ones(*shape):
        dtype = np.float32 if Precision.value() == 32 else np.float64
        return np.ones(shape, dtype=dtype)
