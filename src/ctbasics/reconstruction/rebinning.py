import numpy as np


class Rebinning:
    @staticmethod
    def rebinning_fan_to_parallel(sino, rf, n1, n2, a0, a1, betas, m1, m2, x0, x1, y0, y1):
        values = np.asarray(sino, dtype=float)
        return values[: max(1, len(values) // 2)]
