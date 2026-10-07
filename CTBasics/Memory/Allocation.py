import numpy as np

from .Precision import Precision

_DTYPES = {"double": np.float64, "single": np.float32}


class Allocation:
    @staticmethod
    def _dtype():
        return _DTYPES[Precision.Value()]

    @staticmethod
    def zeros(*shape):
        return np.zeros(shape, dtype=Allocation._dtype())

    @staticmethod
    def ones(*shape):
        return np.ones(shape, dtype=Allocation._dtype())

    @staticmethod
    def clear(*arrays):
        for array in arrays:
            array.fill(0)
