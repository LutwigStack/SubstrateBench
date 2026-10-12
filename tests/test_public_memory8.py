import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from substratebench.public_memory8 import ALGORITHM, PublicMemory8, load_coefficients


def coefficients():
    reader = [[[float(row == bit) for bit in range(4)] for row in range(33)]
              for _ in range(2)]
    pulses = [[[float((operation // 2 == channel) * (2 * (operation % 2) - 1))
                for channel in range(16)] for operation in range(8)] for _ in range(2)]
    return reader, pulses


class Port:
    """Instrumented software contract fixture, not physical qualification."""
    def __init__(self):
        self.calls, self.bits, self.fail_at = [], [[0] * 4 for _ in range(2)], None

    def _call(self, op, lane, value=None):
        self.calls.append((op, lane, value))
        if len(self.calls) == self.fail_at:
            raise OSError("interrupted public port")

    def reset(self, lane, batch):
        self._call("reset", lane, batch)
        self.bits[lane][:] = [0] * 4

    def step(self, lane, value):
        self._call("step", lane, value)
        for bit, pulse in enumerate(value[0][:4]):
            if pulse:
                self.bits[lane][bit] = int(pulse > 0)

    def observe(self, lane):
        self._call("observe", lane)
        return [[2.0 * bit - 1 for bit in self.bits[lane]] + [0.0] * 28]


class PublicMemoryTests(unittest.TestCase):
    def setUp(self):
        self.port = Port()
        self.reader, self.pulses = coefficients()
        self.memory = PublicMemory8(self.port, self.reader, self.pulses)

    def test_constructor_no_io_and_requires_reset(self):
        self.assertEqual(self.port.calls, [])
        with self.assertRaises(RuntimeError):
            self.memory.read()
        self.assertEqual(self.port.calls, [])

    def test_reset_exact_bootstrap_and_evaluator_initialization(self):
        self.memory.reset()
        self.assertEqual(len(self.port.calls), 84)
        self.assertEqual(self.memory.counts()["completed_common_ticks"], 44)
        self.assertEqual(self.port.calls[:2], [("reset", 0, 1), ("reset", 1, 1)])
        for register in range(8):
            self.memory.set_bit(register, 0)
        self.assertEqual(len(self.port.calls), 260)
        self.assertEqual(self.memory.counts()["completed_common_ticks"], 116)

    def test_literal_selected_pulse_other_zero_and_settling(self):
        self.memory.reset()
        for register in range(8):
            first = len(self.port.calls)
            self.memory.set_bit(register, 1)
            calls = self.port.calls[first:]
            self.assertEqual(len(calls), 22)
            self.assertEqual([row[:2] for row in calls[:4]],
                             [("step", 0), ("step", 1), ("observe", 0), ("observe", 1)])
            for lane in (0, 1):
                expected = tuple(self.pulses[lane][2 * (register % 4) + 1]) if lane == register // 4 else (0.0,) * 16
                self.assertEqual(calls[lane][2], (expected,))
            self.assertEqual(calls[4:20], [("step", lane, ((0.0,) * 16,))
                                           for _ in range(8) for lane in (0, 1)])
            self.assertEqual([row[:2] for row in calls[20:]], [("observe", 0), ("observe", 1)])
        self.assertEqual(self.memory.read(), (1,) * 8)

    def test_fresh_read_and_defensive_freeze_without_host_q(self):
        self.memory.reset()
        self.memory.set_bit(0, 1)
        self.reader[0][0][0] = -99
        self.pulses[0][1][0] = -99
        self.port.bits[0][0] = 0
        first = len(self.port.calls)
        self.assertEqual(self.memory.read(), (0,) * 8)
        self.assertEqual(self.port.calls[first:], [("observe", 0, None), ("observe", 1, None)])
        self.assertFalse(hasattr(self.memory, "__dict__"))
        self.memory.set_bit(0, 1)
        self.assertEqual(self.memory.read()[0], 1)

    def test_invalid_set_before_mutation(self):
        self.memory.reset()
        before = len(self.port.calls)
        for args in ((True, 0), (0, False), (-1, 0), (8, 0), (1.0, 0), (0, 2)):
            with self.assertRaises(ValueError):
                self.memory.set_bit(*args)
        self.assertEqual(len(self.port.calls), before)
        self.assertFalse(self.memory.counts()["broken"])

    def test_partial_tick_breaks_permanently_and_counts_prefix(self):
        self.memory.reset()
        self.port.fail_at = len(self.port.calls) + 2
        with self.assertRaises(OSError):
            self.memory.set_bit(0, 1)
        self.assertEqual(self.memory.last_operation["attempted_primitive_calls"], 2)
        self.assertEqual(self.memory.last_operation["completed_primitive_calls"], 1)
        self.assertIsNone(self.memory.last_operation["common_ticks_after"])
        self.assertEqual(self.memory.counts()["completed_common_ticks"], 44)
        before = len(self.port.calls)
        for method in (self.memory.reset, self.memory.read, lambda: self.memory.set_bit(0, 0)):
            with self.assertRaises(RuntimeError):
                method()
        self.assertEqual(len(self.port.calls), before)

    def test_malformed_observation_breaks_before_further_pulse(self):
        self.memory.reset()
        for bad in ([], [[0] * 31], [[False] * 32], [[float("nan")] * 32], [[1e7] * 32]):
            port = Port()
            memory = PublicMemory8(port, *coefficients())
            memory.reset()
            port.observe = lambda lane: bad
            with self.assertRaises(ValueError):
                memory.set_bit(0, 1)
            self.assertEqual(memory.last_operation["attempted_primitive_calls"], 3)
            self.assertTrue(memory.counts()["broken"])

    def test_unexpected_mutation_response_breaks_without_retry(self):
        self.port.reset = lambda lane, batch: "unexpected"
        with self.assertRaises(ValueError):
            self.memory.reset()
        self.assertEqual(self.memory.last_operation["attempted_primitive_calls"], 1)
        self.assertEqual(self.memory.last_operation["completed_primitive_calls"], 0)
        self.assertTrue(self.memory.counts()["broken"])

    def test_invalid_coefficient_shapes_types_bounds(self):
        for field, value in (("reader", True), ("reader", float("inf")),
                             ("reader", 1e7), ("pulses", 1.1)):
            reader, pulses = coefficients()
            (reader if field == "reader" else pulses)[0][0][0] = value
            with self.assertRaises(ValueError):
                PublicMemory8(Port(), reader, pulses)
        with self.assertRaises(ValueError):
            PublicMemory8(Port(), self.reader[:1], self.pulses)

    def test_loader_checksum_schema_duplicates_and_byte_bound(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "coefficients.json"
            valid = {"schema_version": 1, "family": "discrete_tanh", "algorithm": ALGORITHM,
                     "reader": self.reader, "pulses": self.pulses}
            for value in (valid, dict(valid, extra=0), dict(valid, schema_version=True)):
                encoded = json.dumps(value).encode()
                path.write_bytes(encoded)
                digest = hashlib.sha256(encoded).hexdigest()
                if value is valid:
                    loaded = load_coefficients(path, digest)
                    self.assertEqual(loaded[0][0][0][0], 1.0)
                    with self.assertRaises(ValueError):
                        load_coefficients(path, "0" * 64)
                else:
                    with self.assertRaises(ValueError):
                        load_coefficients(path, digest)
            for encoded in (b'{"x":1,"x":2}', b'{"x":NaN}', b" " * (65536 + 1)):
                path.write_bytes(encoded)
                with self.assertRaises(ValueError):
                    load_coefficients(path, hashlib.sha256(encoded).hexdigest())


if __name__ == "__main__":
    unittest.main()
