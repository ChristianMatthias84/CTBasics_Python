import math


class AzimuthalSampling:
    """Container class for azimuthal sampling parameter."""

    def __init__(self, N2pi_in: int) -> None:
        self.N2pi = N2pi_in
        self.dalpha = 2 * math.pi / self.N2pi

    def Copy(self) -> "AzimuthalSampling":
        return AzimuthalSampling(self.N2pi)
