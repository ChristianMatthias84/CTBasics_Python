class StressTest:
    """Allocation stress test.

    The MATLAB body is commented out apart from an empty `parfor` loop that only warms up
    the worker pool, so `Run` has no effect other than touching the parallel backend.
    """

    def __init__(self):
        pass

    def Run(self, size, loop=None):
        for _ in range(1, 5):
            pass

        return None
