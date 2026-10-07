import numpy as np


class RawDataHelperFunctions:
    @staticmethod
    def LinearizeReadings(Readings_in):
        """Flattens a (reading, detector column, detector row) stack into one vector.

        MATLAB linearizes each reading column-major, so the detector column index runs
        fastest inside a reading and the readings follow one another.
        """
        Readings_in = np.asarray(Readings_in, dtype=float)
        if Readings_in.ndim != 3:
            raise ValueError("Readings must be a 3D array (readings, columns, rows)")

        return Readings_in.transpose(0, 2, 1).reshape(-1)

    @staticmethod
    def SimpleLinearizeArray(x):
        """Legacy row-major linearization used by the StaticCT scripts."""
        return np.asarray(x, dtype=float).reshape(-1)
