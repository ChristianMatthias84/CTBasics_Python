import os

import psutil

from CTBasics.MultiCore.RAM import RAM


class CPU:
    """Physical/logical core layout, mirroring MATLAB's `feature('numcores')` report."""

    def __init__(self):
        details = CPU.GetDetails().splitlines()

        self.PhysicalCores = None
        self.VirtualyCores = None
        self.HyperThreaded = None

        for line in details:
            if "MATLAB detected:" in line:
                cores = int("".join(c for c in line if c.isdigit()))
                if "physical cores" in line:
                    self.PhysicalCores = cores
                if "logical cores" in line:
                    self.VirtualyCores = cores
            if "hyper-threading" in line:
                self.HyperThreaded = "enabled" in line

    @staticmethod
    def GetDetails():
        physical = psutil.cpu_count(logical=False) or os.cpu_count() or 1
        logical = psutil.cpu_count(logical=True) or os.cpu_count() or 1
        state = "enabled" if logical > physical else "disabled"
        return (
            f"MATLAB detected: {physical} physical cores.\n"
            f"MATLAB detected: {logical} logical cores.\n"
            f"MATLAB was assigned: {logical} logical cores by the OS.\n"
            f"MATLAB detected hyper-threading {state}.\n"
        )

    @staticmethod
    def GetOptions():
        """Worker count for the process pool; replaces MATLAB's `parforOptions`."""
        return {"NumWorkers": CPU.GetThreads(), "RangePartitionMethod": "fixed"}

    @staticmethod
    def GetThreads(requests=None, workload=0.8):
        """Largest worker count whose memory request still fits into the free RAM share.

        `requests[i]` is the memory needed when running with i+1 threads.
        """
        if requests is None:
            return psutil.cpu_count(logical=True) or os.cpu_count() or 1

        free = RAM.GetFree()

        threads = 1
        for i, request in enumerate(requests, start=1):
            if request < workload * free:
                threads = i
        return threads
