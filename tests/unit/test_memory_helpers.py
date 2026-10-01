import numpy as np

from ctbasics.memory import Allocation, Precision, TypeCast


def test_allocation_respects_precision_switch():
    Precision.value(32)
    array = Allocation.zeros(2, 3)

    assert array.shape == (2, 3)
    assert array.dtype == np.float32

    ones = Allocation.ones(2, 3, 4)
    assert ones.shape == (2, 3, 4)
    assert ones.dtype == np.float32

    Precision.value(64)


def test_typecast_converts_to_requested_dtype():
    data = [1, 2, 3]

    assert TypeCast.to(data, "single").dtype == np.float32
    assert TypeCast.to(data, "double").dtype == np.float64
