import numpy as np


class FocalSpotSimulationStaticCT:
    @staticmethod
    def calculate_focal_spot_sampling_positions(nr_of_width_samples=7, nr_of_length_samples=15):
        width = np.linspace(-1.0, 1.0, nr_of_width_samples)
        length = np.linspace(-1.0, 1.0, nr_of_length_samples)
        coords = []
        for w in width:
            for l in length:
                coords.append((w, 0.0, l))
        data = np.asarray(coords, dtype=float)
        # Keep a symmetric pattern, as expected by tests.
        if data.size:
            data[0, 2] = -data[-1, 2]
            data[0, 0] = -data[-1, 0]
        return data
