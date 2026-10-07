import numpy as np
import pytest

from CTBasics.Memory import Allocation, Precision, TypeCast


@pytest.fixture(autouse=True)
def restore_default_precision():
    # Precision is global state in MATLAB too, so other scripts may have switched it.
    Precision.Value(64)
    yield
    Precision.Value(64)


def test_precision_defaults_to_double_like_matlab():
    assert Precision.Value() == "double"
    assert Precision.Default == "double"
    assert Precision.IsDefault()
    assert Allocation.zeros(2).dtype == np.float64


def test_precision_switch_changes_allocation_dtype():
    assert Precision.Value(32) == "single"
    assert not Precision.IsDefault()

    array = Allocation.zeros(2, 3)
    assert array.shape == (2, 3)
    assert array.dtype == np.float32

    ones = Allocation.ones(2, 3, 4)
    assert ones.shape == (2, 3, 4)
    assert ones.dtype == np.float32
    assert (ones == 1.0).all()


def test_precision_rejects_unsupported_widths():
    with pytest.raises(ValueError):
        Precision.Value(16)


def test_typecast_converts_to_requested_dtype():
    data = [1, 2, 3]

    assert TypeCast.To(data, "single").dtype == np.float32
    assert TypeCast.To(data, "double").dtype == np.float64


def test_allocation_clear_zeroes_in_place():
    array = Allocation.ones(3)
    Allocation.clear(array)
    assert (array == 0.0).all()
