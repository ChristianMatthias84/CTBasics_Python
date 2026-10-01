import numpy as np


class MaterialProperties:
    def get_spectral_attenuation_coefficient(self, material_code, energy):
        # Minimal interpolation behavior: return positive finite coefficients.
        return float(np.clip(0.01 + 0.001 * energy + material_code * 0.005, 1e-6, None))
