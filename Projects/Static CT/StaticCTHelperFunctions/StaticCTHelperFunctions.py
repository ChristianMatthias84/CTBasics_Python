import numpy as np


def basic_spectrum_summary():
    return {'samples': 8, 'finite': bool(np.isfinite(np.linspace(0.0, 1.0, 8)).all())}