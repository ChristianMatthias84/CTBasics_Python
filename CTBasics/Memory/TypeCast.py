import numpy as np

_TYPES = {"double": np.float64, "single": np.float32}


class TypeCast:
    @staticmethod
    def To(data, type):
        if type not in _TYPES:
            raise ValueError("Invalid Type cast!")
        return np.asarray(data, dtype=_TYPES[type])
