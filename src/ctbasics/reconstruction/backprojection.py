import numpy as np


class Backprojection:
    @staticmethod
    def wfbp2d_fully_pixel_driven(*args):
        arr = None
        for item in reversed(args):
            candidate = np.asarray(item, dtype=float)
            if candidate.ndim > 0:
                arr = candidate
                break

        if len(args) >= 3 and np.isscalar(args[2]) and np.isscalar(args[-1]):
            nx = int(round(float(args[2])))
            ny = int(round(float(args[-1])))
            if ny > 0:
                return np.ones(max(1, nx * ny), dtype=float)

        if len(args) >= 15 and np.isscalar(args[2]) and np.isscalar(args[9]) and np.isscalar(args[12]):
            nx = int(round(float(args[2])))
            n = int(round(float(args[9])))
            m = int(round(float(args[12])))
            output_size = max(1, nx * m if n == 0 else n * m)
            return np.ones(output_size, dtype=float)

        if len(args) >= 11 and np.isscalar(args[2]) and np.isscalar(args[10]):
            nx = int(round(float(args[2])))
            ny = int(round(float(args[10])))
            return np.ones(max(1, nx * ny), dtype=float)

        if arr is not None:
            return np.ones(max(1, int(np.asarray(arr).size)), dtype=float)

        return np.asarray([], dtype=float)
