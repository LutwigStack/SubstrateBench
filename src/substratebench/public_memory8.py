"""Scalar public reset/step/observe memory; no task rule or host bit cache."""

import hashlib
import json
import math
from contextlib import contextmanager
from pathlib import Path

from .carry8 import _set_request

MAX_ARTIFACT_BYTES = 64 * 1024
ALGORITHM = "affine-public-reader-scalar-bootstrap-v1"
_ZERO = ((0.0,) * 16,)


def _number(value, limit):
    if type(value) not in (int, float):
        raise ValueError("Expected a finite numeric coefficient")
    try:
        value = float(value)
    except (OverflowError, ValueError):
        raise ValueError("Expected a finite numeric coefficient") from None
    if not math.isfinite(value) or abs(value) > limit:
        raise ValueError("Numeric value exceeds its finite bound")
    return value


def _array(value, dimensions, limit):
    if not isinstance(value, (tuple, list)) or len(value) != dimensions[0]:
        raise ValueError("Invalid public array shape")
    if len(dimensions) == 1:
        return tuple(_number(item, limit) for item in value)
    return tuple(_array(item, dimensions[1:], limit) for item in value)


def _unique_pairs(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("Duplicate JSON key")
        result[key] = value
    return result


def _reject_constant(_):
    raise ValueError("Nonfinite JSON number")


def load_coefficients(path, sha256):
    """Read a bounded, checksum-verified public-only coefficient artifact."""
    if (not isinstance(sha256, str) or len(sha256) != 64
            or any(char not in "0123456789abcdef" for char in sha256)):
        raise ValueError("Expected a lowercase SHA256 digest")
    with Path(path).open("rb") as stream:
        encoded = stream.read(MAX_ARTIFACT_BYTES + 1)
    if len(encoded) > MAX_ARTIFACT_BYTES:
        raise ValueError("Coefficient artifact exceeds 64 KiB")
    if hashlib.sha256(encoded).hexdigest() != sha256:
        raise ValueError("Coefficient artifact checksum differs")
    try:
        value = json.loads(encoded, object_pairs_hook=_unique_pairs,
                           parse_constant=_reject_constant)
    except (UnicodeError, json.JSONDecodeError, RecursionError):
        raise ValueError("Invalid coefficient JSON") from None
    if (not isinstance(value, dict)
            or set(value) != {"schema_version", "family", "algorithm", "reader", "pulses"}
            or type(value["schema_version"]) is not int or value["schema_version"] != 1
            or value["family"] != "discrete_tanh" or value["algorithm"] != ALGORITHM):
        raise ValueError("Unsupported public coefficient schema")
    return (_array(value["reader"], (2, 33, 4), 1e6),
            _array(value["pulses"], (2, 8, 16), 1.0))


class PublicMemory8:
    """Concrete scalar adapter; port operations accept/return Python rows.

    The port owns transport validation, authentic IDs, deadlines and cleanup.
    observe(lane) returns a list/tuple shaped [1,32]; step takes [1,16].
    reset(lane,1) performs the known eight-tick cold reset. No constructor I/O.
    """

    __slots__ = ("_port", "_reader", "_pulses", "_ready", "_broken", "_busy",
                 "_attempts", "_completed", "_ticks", "_attempted_ticks",
                 "_operation_id", "_last_operation", "coefficient_sha256")

    def __init__(self, port, reader, pulses):
        self._reader = _array(reader, (2, 33, 4), 1e6)
        self._pulses = _array(pulses, (2, 8, 16), 1.0)
        if any(not callable(getattr(port, name, None))
               for name in ("reset", "step", "observe")):
            raise ValueError("Expected a public reset/step/observe port")
        self._port = port
        encoded = json.dumps([self._reader, self._pulses], allow_nan=False,
                             separators=(",", ":")).encode()
        self.coefficient_sha256 = hashlib.sha256(encoded).hexdigest()
        self._ready = self._broken = self._busy = False
        self._attempts = self._completed = self._ticks = self._attempted_ticks = 0
        self._operation_id = 0
        self._last_operation = None

    @classmethod
    def from_json(cls, port, path, *, sha256):
        return cls(port, *load_coefficients(path, sha256))

    def counts(self):
        return {"attempted_primitive_calls": self._attempts,
                "completed_primitive_calls": self._completed,
                "attempted_common_ticks": self._attempted_ticks,
                "completed_common_ticks": self._ticks,
                "map_ticks": None if self._broken else self._ticks,
                "broken": self._broken, "operations": self._operation_id}

    @property
    def last_operation(self):
        return None if self._last_operation is None else dict(self._last_operation)

    @contextmanager
    def _operation(self, operation, register=None, value=None):
        if self._broken or self._busy:
            raise RuntimeError("Public memory is broken or already in use")
        if operation != "reset" and not self._ready:
            raise RuntimeError("Public memory requires reset")
        self._busy = True
        self._operation_id += 1
        first, completed, ticks = self._attempts + 1, self._completed, self._ticks
        outcome = "interrupted"
        try:
            yield
            outcome = "completed"
        except BaseException:
            self._broken = True
            raise
        finally:
            self._last_operation = {
                "macro_id": self._operation_id, "operation": operation,
                "register": register, "value": value,
                "first_primitive_id": first, "last_primitive_id": self._attempts,
                "attempted_primitive_calls": self._attempts - first + 1,
                "completed_primitive_calls": self._completed - completed,
                "common_ticks_before": ticks,
                "common_ticks_after": None if self._broken else self._ticks,
                "outcome": outcome}
            self._busy = False

    def _call(self, operation, lane, argument=None):
        self._attempts += 1
        method = getattr(self._port, operation)
        result = method(lane) if argument is None else method(lane, argument)
        if operation == "observe":
            result = _array(result, (1, 32), 1e6)[0]
        elif result is not None:
            raise ValueError("Unexpected response from public mutation")
        self._completed += 1
        return result

    def _read(self):
        observations = [self._call("observe", lane) for lane in (0, 1)]
        logits = tuple(sum(y * self._reader[lane][index][bit]
                           for index, y in enumerate(observations[lane]))
                       + self._reader[lane][32][bit]
                       for lane in (0, 1) for bit in range(4))
        if any(not math.isfinite(value) for value in logits):
            raise ValueError("Nonfinite public reader result")
        return tuple(int(value >= 0) for value in logits)

    def _tick(self, low, high):
        self._attempted_ticks += 1
        self._call("step", 0, low)
        self._call("step", 1, high)
        self._ticks += 1

    def _slot(self, low, high):
        self._tick(low, high)
        for lane in (0, 1):
            self._call("observe", lane)
        for _ in range(8):
            self._tick(_ZERO, _ZERO)

    def reset(self):
        with self._operation("reset"):
            self._attempted_ticks += 8
            for lane in (0, 1):
                self._call("reset", lane, 1)
            self._ticks += 8
            for bit in range(4):
                self._slot((self._pulses[0][2 * bit],),
                           (self._pulses[1][2 * bit],))
            self._read()
            self._ready = True

    def read(self):
        with self._operation("read"):
            return self._read()

    def set_bit(self, register, value):
        _set_request(register, value)
        with self._operation("set_bit", register, value):
            pulse = (self._pulses[register // 4][2 * (register % 4) + value],)
            self._slot(pulse if register < 4 else _ZERO,
                       _ZERO if register < 4 else pulse)
            self._read()
