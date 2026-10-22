import os
from concurrent.futures import ProcessPoolExecutor

import numba
import numpy as np

from CTBasics.Memory.Precision import Precision
from CTBasics.MultiCore.CPU import CPU

_Pool = None
_NumWorkers = 0
_Initialized = False
_InWorker = False

_BITS = {"double": 64, "single": 32}


def _WorkerInit():
    """Marks the process as a worker so that nested parfor loops run serially, as in MATLAB."""
    global _InWorker
    _InWorker = True
    numba.set_num_threads(1)


def _RunChunk(func, chunk, broadcast, precision):
    """Executes one contiguous chunk of parfor iterations inside a worker."""
    Precision.Value(precision)
    return [func(**sliced, **broadcast) for sliced in chunk]


class Parallelization:
    """Process pool management, replacing MATLAB's `parpool`/`gcp`/`parfor`."""

    @staticmethod
    def Init(nDesired=None):
        """Creates the worker pool; `nDesired=1` selects serial execution without a pool.

        The threaded numba kernels of the projectors use the same number of threads.
        """
        global _Pool, _NumWorkers, _Initialized

        if not _Initialized:
            _NumWorkers = int(nDesired) if nDesired is not None else CPU.GetThreads()
            numba.set_num_threads(min(max(_NumWorkers, 1), numba.config.NUMBA_NUM_THREADS))
            if _NumWorkers > 1:
                # Each worker is one parfor lane; nested BLAS threads would oversubscribe the
                # cores. numpy is already loaded here, so only the spawned workers see this.
                for variable in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
                    os.environ.setdefault(variable, "1")
                _Pool = ProcessPoolExecutor(max_workers=_NumWorkers, initializer=_WorkerInit)
            _Initialized = True

        return _NumWorkers

    @staticmethod
    def GetPool():
        return _Pool

    @staticmethod
    def GetNumWorkers() -> int:
        return _NumWorkers if _Pool is not None else 0

    @staticmethod
    def DeletePool():
        global _Pool, _NumWorkers, _Initialized

        if _Pool is not None:
            _Pool.shutdown(wait=True)
        _Pool = None
        _NumWorkers = 0
        _Initialized = False

    @staticmethod
    def ParFor(func, iterations, **broadcast):
        """MATLAB parfor: returns `[func(**sliced, **broadcast) for sliced in iterations]`.

        `iterations` holds one dict of sliced keyword arguments per loop pass; `broadcast`
        holds the arguments shared by all passes. Like MATLAB, the first call creates the
        pool, and loops inside a worker run serially. Every worker receives one contiguous
        chunk, so broadcast arrays are transferred once per worker, and the results come
        back in iteration order, which keeps them identical to a serial run.
        `func` must be picklable, i.e. a module-level function or a class staticmethod.
        """
        chunks = Parallelization.Chunks(iterations)

        if len(chunks) == 1:
            return [func(**sliced, **broadcast) for sliced in chunks[0]]

        precision = _BITS[Precision.Value()]
        futures = [_Pool.submit(_RunChunk, func, chunk, broadcast, precision) for chunk in chunks]

        return [result for future in futures for result in future.result()]

    @staticmethod
    def Chunks(items):
        """Splits `items` into one contiguous chunk per worker, e.g. for parfor reductions.

        Without a pool, inside a worker or for a single item there is exactly one chunk,
        so the serial order is kept.
        """
        items = list(items)

        if not _InWorker and len(items) > 1:
            Parallelization.Init()

        count = 1 if _InWorker or _Pool is None else min(_NumWorkers, len(items))
        indices = np.array_split(np.arange(len(items)), max(count, 1))

        return [[items[i] for i in chunk] for chunk in indices]

    @staticmethod
    def PrintInfo():
        if _Pool is None:
            print("No parallel pool active.")
        else:
            print(f"Parallel pool active with {_NumWorkers} workers.")
