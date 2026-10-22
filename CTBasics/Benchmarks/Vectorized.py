import time

import numpy as np

from CTBasics.Memory.Allocation import Allocation


def Vectorized():
    """Compares a hand written matrix-vector product with the vectorized one."""
    size = 4096

    matrix = Allocation.ones(size, size)
    vector = Allocation.ones(size, 1)

    variants = ["slow", "fast"]

    for item in variants:
        print(f"Benchmark of {item} implementation")

        start = time.perf_counter()

        if item == "slow":
            Slow(matrix, vector)
        elif item == "fast":
            Fast(matrix, vector)
        else:
            print("unknown function!")

        print(f"Elapsed time is {time.perf_counter() - start:.6f} seconds.")


def Slow(matrix, vector):
    """Explicit double loop, kept to show the cost of the non-vectorized form."""
    result = Allocation.zeros(matrix.shape[0], 1)

    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            result[i] = result[i] + matrix[i, j] * vector[j]

    return result


def Fast(matrix, vector):
    return np.asarray(matrix) @ np.asarray(vector)
