import hashlib
import json
import tempfile
import unittest
from contextlib import redirect_stdout, redirect_stderr
from io import StringIO
from pathlib import Path
from unittest.mock import patch

from substratebench import Command as FourBitCommand, sequence_id as four_bit_id, write_result
from substratebench import carry8
from substratebench.cli import main


class CarryLawTests(unittest.TestCase):
    def test_all_8192_transitions_against_bitwise_carry_and_borrow(self):
        for state in carry8.all_states():
            for cmd in carry8.all_commands():
                expected = list(state)
                carry = cmd.gate
                for bit in range(cmd.register, 8):
                    old = expected[bit]
                    expected[bit] ^= carry
                    carry &= old if cmd.value else 1 - old
                self.assertEqual(carry8.reference_step(state, cmd), tuple(expected))
        self.assertEqual(carry8.reference_step((1, 1, 1, 1, 0, 0, 0, 0), carry8.Command(0, 1, 1)),
                         (0, 0, 0, 0, 1, 0, 0, 0))
        self.assertEqual(carry8.reference_step((1,) * 8, carry8.Command(0, 1, 1)), (0,) * 8)
        self.assertEqual(carry8.reference_step((0,) * 8, carry8.Command(0, 1, 0)), (1,) * 8)

    def test_separate_eight_bit_trust_boundaries(self):
        for args in ((8, 1, 1), (True, 1, 0), (7, False, 1), (0, 1, 0.0)):
            with self.assertRaises(ValueError):
                carry8.Command(*args)
        for state in ((0,) * 4, (0,) * 7, (0,) * 7 + (False,), "00000000"):
            with self.assertRaises(ValueError):
                carry8.state_bits(state)
        with self.assertRaises(ValueError):
            carry8.reference_step((0,) * 8, FourBitCommand(0, 1, 1))
        for memory in (carry8.BinaryMemory(), carry8.EncodedMemory()):
            for register, value in ((8, 1), (True, 1), (0, False)):
                with self.assertRaises(ValueError):
                    memory.set_bit(register, value)

    def test_normative_identity_and_unchanged_legacy_id(self):
        commands = (carry8.Command(7, 1, 0), carry8.Command(0, 0, 1))
        prefix = b"substratebench-carry8-sequence-v1\x00\x08onehot8-gate-value\x00"
        encoded = bytes((0, 0, 0, 0, 0, 0, 0, 1, 1, 0,
                         1, 0, 0, 0, 0, 0, 0, 0, 0, 1))
        self.assertEqual(carry8.sequence_id(commands),
                         hashlib.sha256(prefix + b"\x00\x00\x00\x02" + encoded).hexdigest())
        old = (FourBitCommand(2, 1, 0), FourBitCommand(0, 1, 1))
        self.assertEqual(four_bit_id(old), hashlib.sha256(bytes((0, 0, 1, 0, 1, 0,
                                                              1, 0, 0, 0, 1, 1))).hexdigest())
        for invalid in ((), (FourBitCommand(0, 0, 0),), (commands[0],) * 129):
            with self.assertRaises(ValueError):
                carry8.sequence_id(invalid)


class CarrySuiteTests(unittest.TestCase):
    def test_reproducibility_both_route_exclusion_and_upper_active(self):
        first, count = carry8.make_suite(42, 4)
        self.assertEqual(count, 0)
        self.assertEqual((first, count), carry8.make_suite(42, 4))
        identities = []
        for role in first:
            for case in role["cases"]:
                for stream in (case["commands"], tuple(carry8.permute_command(cmd)
                                                       for cmd in case["commands"])):
                    self.assertTrue(any(cmd.gate and cmd.register >= 4 for cmd in stream))
                    identities.append(carry8.sequence_id(stream))
        self.assertEqual(len(identities), len(set(identities)))
        replacement, count = carry8.make_suite(42, 4, identities)
        self.assertEqual(count, len(identities))
        for role in replacement:
            for case in role["cases"]:
                self.assertNotIn(case["id"], identities)
                self.assertNotIn(carry8.sequence_id(tuple(carry8.permute_command(cmd)
                                                        for cmd in case["commands"])), identities)

    def test_invalid_sampling_and_bounded_exclusion(self):
        for kwargs in ({"seed": True}, {"seed": -1}, {"episodes": 129}, {"episodes": 3},
                       {"exclude_ids": ("A" * 64,)}, {"exclude_ids": ("0" * 63,)}):
            with self.assertRaises(ValueError):
                carry8.make_suite(**kwargs)
        with patch.object(carry8, "MAX_EXCLUDED_IDS", 2):
            with self.assertRaises(ValueError):
                carry8.make_suite(exclude_ids=("0" * 64,) * 3)
        # Duplicate entries still consume the resource bound; counts report unique IDs.
        _, count = carry8.make_suite(episodes=2, exclude_ids=("0" * 64,) * 3)
        self.assertEqual(count, 1)


class CarryBenchmarkTests(unittest.TestCase):
    def test_reference_run_complete_interface_and_real_memory_controls(self):
        result = carry8.evaluate(carry8.reference_step, carry8.EncodedMemory(), episodes=4,
                                 controller_label="reference_oracle")
        self.assertEqual(result["verdict"], "pass")
        self.assertEqual(result["status"], "completed")
        self.assertEqual(result["interface_qualification"]["reader_correct"], 256)
        self.assertEqual(result["interface_qualification"]["SET_correct"], 4096)
        self.assertEqual(result["source_law_qualification"]["correct"], 8192)
        self.assertIsNone(result["source_law_qualification"]["signed_margin"])
        qualification = result["cost"]["phases"]["interface_qualification"]
        self.assertEqual(qualification["memory_SET_calls"], 38912)
        self.assertEqual(qualification["memory_reads"], 4352)
        self.assertEqual(qualification["memory_resets"], 4352)
        self.assertEqual(qualification["controller_calls"], 0)
        for name in ("random-L24", "random-L128"):
            for route in ("normal", "permutation"):
                self.assertEqual(result["routes"][route][name]["metrics"]["whole_sequence_accuracy"], 1)
            for route in ("zero_write", "reset"):
                self.assertLessEqual(result["routes"][route][name]["metrics"]["whole_sequence_accuracy"], .7)
        self.assertEqual(result["cost"]["phases"]["zero_write"]["memory_SET_calls"], 8 * 8)
        self.assertEqual(result["cost"]["phases"]["normal"]["controller_calls"], 4 * (24 + 128))

    def test_zero_source_failure_censors_long_routes_and_preserves_nulls(self):
        result = carry8.evaluate(carry8.zero_step, carry8.BinaryMemory(), episodes=2)
        self.assertEqual(result["status"], "source_not_qualified")
        self.assertEqual(result["verdict"], "fail")
        self.assertEqual(result["route_status"], "not_attempted")
        self.assertIsNone(result["criteria"]["per_length"])
        for route in carry8.ROUTES:
            self.assertTrue(all(value is None for value in result["routes"][route].values()))
            self.assertTrue(all(value == 0 for value in result["cost"]["phases"][route].values()))
        self.assertEqual(result["cost"]["totals"]["controller_calls"], 8192)

    def test_collateral_SET_failure_prevents_source_calls(self):
        class Broken(carry8.BinaryMemory):
            def set_bit(self, register, value):
                self._bits[:] = (0,) * 8
                super().set_bit(register, value)

        def forbidden(*args):
            self.fail("Source must not run when memory fails admission")

        result = carry8.evaluate(forbidden, Broken(), episodes=2)
        self.assertEqual(result["status"], "interface_not_qualified")
        self.assertIsNone(result["source_law_qualification"]["qualified"])
        self.assertEqual(result["cost"]["totals"]["controller_calls"], 0)
        self.assertTrue(all(value is None for value in result["routes"]["normal"].values()))

    def test_callback_gets_own_state_only_and_malformed_output_is_error(self):
        calls = 0

        def source(state, command):
            nonlocal calls
            calls += 1
            self.assertIs(type(command), carry8.Command)
            self.assertEqual(len(state), 8)
            return carry8.reference_step(state, command)

        result = carry8.evaluate(source, carry8.BinaryMemory(), episodes=2)
        self.assertEqual(calls, result["cost"]["totals"]["controller_calls"])
        for invalid in ((0,) * 4, (False,) + (0,) * 7):
            with self.assertRaises(ValueError):
                carry8.evaluate(lambda state, cmd: invalid, carry8.BinaryMemory(), episodes=2)

    def test_invalid_inputs_fail_before_external_memory_access(self):
        class Forbidden:
            def reset(self):
                self.fail("Memory must not run on invalid input")

        for kwargs in ({"controller_label": ""}, {"episodes": 1}, {"exclude_ids": ("bad",)}):
            with self.assertRaises(ValueError):
                carry8.evaluate(carry8.reference_step, Forbidden(), **kwargs)

    def test_encoded_memory_has_no_logical_sidecar(self):
        memory = carry8.EncodedMemory()
        memory.set_bit(4, 1)
        self.assertEqual(set(memory.__dict__), {"_encoded"})
        self.assertEqual(len(memory._encoded), 8)
        self.assertEqual(memory.read(), (0, 0, 0, 0, 1, 0, 0, 0))


class CarryCLITests(unittest.TestCase):
    def test_cli_pass_failure_invalid_and_no_overwrite(self):
        with tempfile.TemporaryDirectory() as directory, redirect_stdout(StringIO()), redirect_stderr(StringIO()):
            path = Path(directory) / "reference.json"
            args = ["--task", "carry_counter_8", "--episodes", "2", "--output", str(path)]
            self.assertEqual(main(args), 0)
            initial = path.read_bytes()
            self.assertEqual(json.loads(initial)["profile"], carry8.PROFILE)
            self.assertEqual(main(args), 2)
            self.assertEqual(path.read_bytes(), initial)
            zero = Path(directory) / "zero.json"
            self.assertEqual(main(args[:-1] + [str(zero), "--controller", "zero"]), 1)
            invalid = Path(directory) / "invalid.json"
            self.assertEqual(main(args[:-1] + [str(invalid), "--episodes", "1"]), 2)
            self.assertFalse(invalid.exists())
            malformed = Path(directory) / "malformed.json"
            with patch.object(carry8, "reference_step", lambda state, cmd: (0,) * 4):
                self.assertEqual(main(args[:-1] + [str(malformed)]), 2)
            self.assertFalse(malformed.exists())

    def test_dangling_symlink_and_oversized_report_never_overwritten(self):
        with tempfile.TemporaryDirectory() as directory, redirect_stdout(StringIO()), redirect_stderr(StringIO()):
            path = Path(directory) / "dangling.json"
            path.symlink_to(Path(directory) / "absent")
            self.assertEqual(main(["--task", "carry_counter_8", "--output", str(path)]), 2)
            self.assertTrue(path.is_symlink())
            self.assertFalse(path.exists())
            large = Path(directory) / "oversized.json"
            with self.assertRaises(ValueError):
                write_result({"value": "x" * (16 * 1024 * 1024)}, large)
            self.assertFalse(large.exists())
            self.assertEqual(list(Path(directory).iterdir()), [path])


if __name__ == "__main__":
    unittest.main()
