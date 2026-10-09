"""Two concrete known digital memories; neither computes the task rule."""

from .task import state_bits


def _set_request(register, value):
    if type(register) is not int or not 0 <= register < 4:
        raise ValueError("Register must be an integer in [0, 3]")
    if type(value) is not int or value not in (0, 1):
        raise ValueError("SET value must be an integer bit")


class BinaryMemory:
    """Four directly stored digital bits, with exact deterministic reset."""

    def __init__(self):
        self._bits = [0, 0, 0, 0]

    def reset(self):
        self._bits[:] = (0, 0, 0, 0)

    def read(self):
        return tuple(self._bits)

    def set_bit(self, register, value):
        _set_request(register, value)
        self._bits[register] = value


class EncodedMemory:
    """A known channel permutation/inversion; only encoded bits persist.

    This is an explicit digital codec, not discovery of an unknown substrate.
    """

    _channels = (2, 0, 3, 1)
    _inversions = (1, 0, 1, 0)

    def __init__(self):
        self._encoded = [0, 0, 0, 0]
        self.reset()

    def reset(self):
        for register in range(4):
            self._encoded[self._channels[register]] = self._inversions[register]

    def read(self):
        return state_bits(tuple(self._encoded[channel] ^ invert
                                for channel, invert in
                                zip(self._channels, self._inversions)))

    def set_bit(self, register, value):
        _set_request(register, value)
        self._encoded[self._channels[register]] = value ^ self._inversions[register]
