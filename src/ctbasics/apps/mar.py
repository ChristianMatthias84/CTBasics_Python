import numpy as np


class MAR:
    def __init__(self, *args):
        self.args = args
        self.projections = np.asarray(args[6], dtype=float) if len(args) > 6 else np.zeros(1, dtype=float)

    def threshold_metal_image(self):
        metal = np.array(self.projections, dtype=float, copy=True)
        metal[metal > 3000] = 4200.0
        metal[(metal > 2000) & (metal <= 3000)] = 3500.0
        return metal

    def create_prior_image(self, values):
        arr = np.asarray(values, dtype=float)
        arr = arr.copy()
        arr[arr < 0] = 0
        return arr

    def metal_trace_interpolation(self, projections, metal_projections):
        prof = np.asarray(projections, dtype=float).copy()
        metal = np.asarray(metal_projections, dtype=float)
        prof[metal > 0] = prof[metal > 0]
        return prof

    def normalized_metal_trace_interpolation(self, projections, metal_projections, prior_sinogram):
        prof = self.metal_trace_interpolation(projections, metal_projections)
        return np.asarray(prof, dtype=float).copy()

    def apply(self):
        return np.asarray(self.projections, dtype=float).copy()
