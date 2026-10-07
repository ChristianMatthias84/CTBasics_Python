class BaseScannerGeometry:
    """Base class for all geometry types."""

    def __init__(self, BeamType_in, Sampling_in) -> None:
        self.BeamType = BeamType_in
        self.Sampling = Sampling_in
        self.Init()

    def Init(self) -> None:
        raise NotImplementedError

    def Copy(self):
        raise NotImplementedError
