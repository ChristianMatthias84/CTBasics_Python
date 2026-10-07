from concurrent.futures import ProcessPoolExecutor

from .CPU import CPU

_Pool = None
_NumWorkers = 0


class Parallelization:
    """Process pool management, replacing MATLAB's `parpool`/`gcp`."""

    @staticmethod
    def Init(nDesired=None):
        global _Pool, _NumWorkers

        if _Pool is None:
            _NumWorkers = int(nDesired) if nDesired is not None else CPU.GetThreads()
            _Pool = ProcessPoolExecutor(max_workers=_NumWorkers)

        return _NumWorkers

    @staticmethod
    def GetPool():
        return _Pool

    @staticmethod
    def GetNumWorkers() -> int:
        return _NumWorkers if _Pool is not None else 0

    @staticmethod
    def DeletePool():
        global _Pool, _NumWorkers

        if _Pool is not None:
            _Pool.shutdown(wait=True)
            _Pool = None
            _NumWorkers = 0

    @staticmethod
    def PrintInfo():
        if _Pool is None:
            print("No parallel pool active.")
        else:
            print(f"Parallel pool active with {_NumWorkers} workers.")
