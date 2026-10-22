import pytest

from CTBasics.MultiCore.Parallelization import Parallelization


@pytest.fixture(scope="session", autouse=True)
def serial_parfor():
    """The tiny test cases run faster serially than with spawned workers.

    Tests that need the pool switch to it explicitly, see test_parallelization.py.
    """
    Parallelization.Init(1)
    yield
    Parallelization.DeletePool()
