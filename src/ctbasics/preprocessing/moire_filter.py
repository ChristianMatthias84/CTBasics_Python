import numpy as np


class MoireFilter:
    @staticmethod
    def apply_moire_filter_on_projections(signal, *args, **kwargs):
        return np.asarray(signal, dtype=float).copy()
