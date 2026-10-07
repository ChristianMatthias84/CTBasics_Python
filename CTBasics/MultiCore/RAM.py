import psutil


class RAM:
    @staticmethod
    def GetFree():
        """Available physical memory in bytes (MATLAB: systemview.PhysicalMemory.Available)."""
        return psutil.virtual_memory().available

    @staticmethod
    def GetUsed():
        return psutil.virtual_memory().total - RAM.GetFree()
