import math


class FlyingFocalSpot:
    """Container class for flying focal spot."""

    def __init__(self, xml: str = "") -> None:
        self.phi = "Phi" in xml
        self.z = "Z" in xml
        self.diag = "Diag" in xml

    def Copy(self) -> "FlyingFocalSpot":
        instance = FlyingFocalSpot()
        instance.phi = self.phi
        instance.z = self.z
        instance.diag = self.diag
        return instance

    def ToString(self) -> str:
        return f"FFS_[{FlyingFocalSpot.GetName(self)}]"

    def GetDelta(self, Geometry, rho):
        Delta_R = 1
        Delta_Phi = 1

        name = FlyingFocalSpot.GetName(self)
        if name == "1x1":
            if rho == 0:
                Delta_R, Delta_Phi = 0, 0
        elif name == "1x2":
            Delta_R, Delta_Phi = (+1, 0) if rho == 0 else (-1, 0)
        elif name == "1p1":
            Delta_R, Delta_Phi = (+1, +1) if rho == 0 else (-1, -1)
        elif name == "2x1":
            Delta_R, Delta_Phi = (0, +1) if rho == 0 else (0, -1)
        elif name == "2x2":
            Delta_R, Delta_Phi = {0: (+1, +1), 1: (+1, -1), 2: (-1, +1), 3: (-1, -1)}[rho]

        RFD = Geometry.RF + Geometry.RD
        s = Geometry.ds
        Delta_beta = sum(Geometry.beta) / len(Geometry.beta)
        gamma = (Geometry.gamma / 360) * (2 * math.pi)

        Delta_R = Delta_R * (RFD / Geometry.RD) * (s / 2) * (1 / math.tan(gamma))
        Delta_Phi = Delta_Phi * (RFD / Geometry.RD) * (Delta_beta / 2) * Geometry.RF
        return Delta_R, Delta_Phi

    @staticmethod
    def GetMask(FFS):
        """Returns the (2x2) flying-focal-spot flag mask, MATLAB index order (row, column)."""
        Mask = [[0, 0], [0, 0]]
        Mask[0][0] = 1
        if FFS.phi:
            Mask[1][0] = 1
        if FFS.z:
            Mask[0][1] = 1
        if FFS.diag:
            Mask[1][1] = 1
        if FFS.phi and FFS.z:
            Mask[1][1] = 1
        return Mask

    @staticmethod
    def GetName(FFS) -> str:
        RPP = FlyingFocalSpot.GetSize(FFS)
        if RPP == 1:
            return "1x1"
        if RPP == 2:
            if FFS.z:
                return "1x2"
            if FFS.phi:
                return "2x1"
            if FFS.diag:
                return "1p1"
        if RPP == 4:
            return "2x2"
        raise ValueError(f"Invalid flying focal spot size: {RPP}")

    @staticmethod
    def GetSize(FFS) -> int:
        return sum(sum(row) for row in FlyingFocalSpot.GetMask(FFS))

    @staticmethod
    def AreEqual(lhs, rhs) -> bool:
        return FlyingFocalSpot.GetMask(lhs) == FlyingFocalSpot.GetMask(rhs)
