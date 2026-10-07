class BaseScanParameter:
    """Base class for all scan-parameter types."""

    def __init__(self, ScanType_in, Geometry_in) -> None:
        self.ScanType = ScanType_in
        self.Geometry = Geometry_in
        self.Init()

    def Init(self) -> None:
        raise NotImplementedError

    def Copy(self):
        raise NotImplementedError
