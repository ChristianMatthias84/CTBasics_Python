import numpy as np


class Precision:
    _value = 32

    @staticmethod
    def value(bits=None):
        if bits is not None:
            Precision._value = int(bits)
        return Precision._value
