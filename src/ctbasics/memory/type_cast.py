import numpy as np


class TypeCast:
    @staticmethod
    def to(data, dtype):
        mapping = {"single": np.float32, "double": np.float64}
        np_dtype = mapping.get(dtype.lower() if isinstance(dtype, str) else dtype, np.float64)
        return np.asarray(data, dtype=np_dtype)
