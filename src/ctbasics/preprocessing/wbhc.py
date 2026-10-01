import numpy as np


class WBHC:
    @staticmethod
    def generate_lut_with_wedge_simulation(kv, nr_of_samples, width, height):
        lut = np.linspace(0.0, 1.0, nr_of_samples, dtype=float)
        lut = np.vstack([lut, np.linspace(0.0, 1.0, nr_of_samples, dtype=float)])
        effective_mu_water = 0.02
        return lut, effective_mu_water
