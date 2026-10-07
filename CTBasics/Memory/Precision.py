class Precision:
    """Global switch between 32-bit and 64-bit working precision.

    `Value()` returns the MATLAB type name, not the bit count, because `Allocation`
    and `TypeCast` pass it straight through as a dtype selector.
    """

    Default = "double"

    _value = None

    @staticmethod
    def Value(bytes=None):
        if bytes is not None:
            Precision._value = int(bytes)
        if Precision._value is None:
            Precision._value = 64

        if Precision._value == 64:
            return "double"
        if Precision._value == 32:
            return "single"
        raise ValueError("Invalid size of bytes!")

    @staticmethod
    def IsDefault() -> bool:
        return Precision.Value() == Precision.Default
