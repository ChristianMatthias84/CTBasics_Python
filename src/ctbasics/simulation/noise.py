import numpy as np


class Noise:
    @staticmethod
    def poisson_noise(values):
        return np.random.poisson(np.clip(values, 0, None)).astype(float)
