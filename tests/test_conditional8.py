import hashlib
import importlib.util
import json
import tempfile
import unittest
from contextlib import redirect_stdout, redirect_stderr
from io import StringIO
from pathlib import Path
from unittest.mock import patch

from substratebench import Command as FourBitCommand, carry8, conditional8, write_result
from substratebench.cli import main


class ConditionalLawTests(unittest.TestCase):
    def test_all_8192_transitions_against_independent_integer_masks(self):
        checked = 0
        for number in range(256):
            state = tuple((number >> bit) & 1 for bit in range(8))
            for register in range(8):
                neighbor, target = (register + 1) % 8, (register + 2) % 8
                for gate in (0, 1):
                    for value in (0, 1):
                        expected = number
                        if gate and value:
                            expected ^= (1 << register) | (((number >> neighbor) & 1) << target)
                        elif gate and ((number >> register) ^ (number >> neighbor)) & 1:
                            expected ^= (1 << register) | (1 << neighbor)
                        actual = conditional8.reference_step(state, carry8.Command(register, gate, value))
                        self.assertEqual(sum(bit << index for index, bit in enumerate(actual)), expected)
                        checked += 1
        self.assertEqual(checked, 8192)

    def test_cross_bank_wrap_old_neighbor_and_malformed_inputs(self):
        self.assertEqual(conditional8.reference_step((0, 0, 0, 1, 0, 0, 0, 0), carry8.Command(3, 1, 0)),
                         (0, 0, 0, 0, 1, 0, 0, 0))
        self.assertEqual(conditional8.reference_step((0, 0, 0, 0, 1, 0, 0, 0), carry8.Command(3, 1, 1)),
                         (0, 0, 0, 1, 1, 1, 0, 0))
        self.assertEqual(conditional8.reference_step((1, 0, 0, 0, 0, 0, 0, 0), carry8.Command(7, 1, 1)),
                         (1, 1, 0, 0, 0, 0, 0, 1))
        self.assertEqual(conditional8.reference_step((1, 0, 0, 0, 0, 0, 0, 0), carry8.Command(7, 1, 0)),
                         (0, 0, 0, 0, 0, 0, 0, 1))
        for state in ((0,) * 4, (False,) + (0,) * 7, "00000000"):
            with self.assertRaises(ValueError):
                conditional8.reference_step(state, carry8.Command(0, 1, 1))
        with self.assertRaises(ValueError):
            conditional8.reference_step((0,) * 8, FourBitCommand(0, 1, 1))

    def test_same_content_identity_and_exclusions_across_laws(self):
        self.assertIs(conditional8.Command, carry8.Command)
        commands = (carry8.Command(7, 1, 0), carry8.Command(0, 0, 1))
        prefix = b"substratebench-carry8-sequence-v1\x00\x08onehot8-gate-value\x00"
        encoded = bytes((0, 0, 0, 0, 0, 0, 0, 1, 1, 0,
                         1, 0, 0, 0, 0, 0, 0, 0, 0, 1))
        self.assertEqual(conditional8.sequence_id(commands),
                         hashlib.sha256(prefix + b"\x00\x00\x00\x02" + encoded).hexdigest())
        suite, count = conditional8.make_suite(42, 2)
        self.assertEqual((suite, count), carry8.make_suite(42, 2))
        excluded = [carry8.sequence_id(stream) for role in suite for case in role["cases"]
                    for stream in (case["commands"], tuple(carry8.permute_command(cmd)
                                                           for cmd in case["commands"]))]
        fresh, count = conditional8.make_suite(42, 2, excluded)
        self.assertEqual(count, 8)
        for role in fresh:
            for case in role["cases"]:
                for stream in (case["commands"], tuple(carry8.permute_command(cmd)
                                                       for cmd in case["commands"])):
                    self.assertNotIn(carry8.sequence_id(stream), excluded)


class ConditionalBenchmarkTests(unittest.TestCase):
    def test_conditional_owns_labels_and_complete_qualification_and_controls(self):
        with patch.object(carry8, "evaluate", side_effect=AssertionError("wrong law owner")), \
                patch.object(carry8, "_run_routes", side_effect=AssertionError("wrong Gold owner")):
            result = conditional8.evaluate(conditional8.reference_step, carry8.EncodedMemory(), episodes=4)
        self.assertEqual(result["task_id"], "conditional_logic_8")
        self.assertEqual(result["benchmark_version"], "0.4.0")
        self.assertEqual(result["profile"], conditional8.PROFILE)
        self.assertEqual(result["verdict"], "pass")
        self.assertEqual(result["interface_qualification"]["reader_correct"], 256)
        self.assertEqual(result["interface_qualification"]["SET_correct"], 4096)
        self.assertEqual(result["source_law_qualification"]["correct"], 8192)
        self.assertIsNone(result["source_law_qualification"]["signed_margin"])
        self.assertEqual(result["source"]["kind"], "reference_oracle")
        self.assertIsNone(result["source"]["creation_wall_seconds"])
        self.assertEqual(result["source"]["creation_cost_status"], "UNKNOWN")
        qualification = result["cost"]["phases"]["interface_qualification"]
        self.assertEqual(qualification["memory_reads"], 4352)
        self.assertEqual(qualification["memory_SET_calls"], 38912)
        for name in ("random-L24", "random-L128"):
            for route in ("normal", "permutation"):
                self.assertEqual(result["routes"][route][name]["metrics"]["whole_sequence_accuracy"], 1)
            for route in ("zero_write", "reset"):
                self.assertLessEqual(result["routes"][route][name]["metrics"]["whole_sequence_accuracy"], .7)
        self.assertEqual(result["cost"]["phases"]["zero_write"]["memory_SET_calls"], 8 * 8)
        self.assertEqual(result["cost"]["phases"]["normal"]["controller_calls"], 4 * (24 + 128))

    def test_routes_use_own_read_state_after_source_drift_without_gold_repair(self):
        memory = carry8.BinaryMemory()
        calls, inputs = 0, []

        def drifting_source(state, command):
            nonlocal calls
            calls += 1
            proposed = conditional8.reference_step(state, command)
            if calls > 8192:
                self.assertEqual(state, memory.read())
                inputs.append(state)
                proposed = (1 - proposed[0],) + proposed[1:]
            return proposed

        result = conditional8.evaluate(drifting_source, memory, episodes=2,
                                       controller_label="drifting-caller", source_creation_wall_seconds=12.5)
        self.assertTrue(result["source_law_qualification"]["qualified"])
        self.assertEqual(result["route_status"], "completed")
        self.assertEqual(result["verdict"], "fail")
        expected_inputs = []
        for route in carry8.ROUTES:
            for role in result["suite"]:
                for outputs in result["routes"][route][role["role"]]["outputs"]:
                    expected_inputs.extend([(0,) * 8] * len(outputs) if route in ("zero_write", "reset")
                                           else [(0,) * 8] + [tuple(state) for state in outputs[:-1]])
        self.assertEqual(inputs, expected_inputs)
        self.assertEqual(result["source"]["kind"], "caller_supplied")
        self.assertEqual(result["source"]["identity"], "drifting-caller")
        self.assertEqual(result["source"]["creation_wall_seconds"], 12.5)
        self.assertEqual(result["source"]["creation_cost_status"], "caller_reported_not_audited")
        self.assertIsNone(result["source"]["energy"])
        self.assertEqual(calls, result["cost"]["totals"]["controller_calls"])

    def test_zero_source_saved_fail_null_routes_and_zero_route_calls(self):
        result = conditional8.evaluate(conditional8.zero_step, carry8.BinaryMemory(), episodes=2)
        self.assertEqual(result["source"]["kind"], "zero_source_control")
        self.assertEqual(result["status"], "source_not_qualified")
        self.assertEqual(result["route_status"], "not_attempted")
        self.assertIsNone(result["criteria"]["per_length"])
        for route in carry8.ROUTES:
            self.assertTrue(all(value is None for value in result["routes"][route].values()))
            self.assertTrue(all(value == 0 for value in result["cost"]["phases"][route].values()))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "fail.json"
            write_result(result, path)
            self.assertEqual(json.loads(path.read_text())["verdict"], "fail")

    def test_corrupt_set_skips_source_and_saves_interface_failure(self):
        class Broken(carry8.BinaryMemory):
            def set_bit(self, register, value):
                self._bits[:] = (0,) * 8
                super().set_bit(register, value)

        def forbidden(*args):
            self.fail("Source must not run when memory fails admission")

        result = conditional8.evaluate(forbidden, Broken(), episodes=2)
        self.assertEqual(result["status"], "interface_not_qualified")
        self.assertIsNone(result["source_law_qualification"]["qualified"])
        self.assertEqual(result["cost"]["totals"]["controller_calls"], 0)
        self.assertTrue(all(value is None for value in result["routes"]["normal"].values()))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "corrupt.json"
            write_result(result, path)
            self.assertEqual(json.loads(path.read_text())["status"], "interface_not_qualified")

    def test_invalid_inputs_before_memory_and_malformed_outputs_propagate(self):
        class Forbidden:
            def reset(self):
                raise AssertionError("Memory must not run on invalid input")

        invalid_costs = (-1, float("nan"), float("inf"), True, "1", 10**500)
        invalid = [{"source_creation_wall_seconds": cost} for cost in invalid_costs]
        invalid += [{"controller_label": ""}, {"episodes": 1}, {"seed": True},
                    {"exclude_ids": ("bad",)}]
        for kwargs in invalid:
            with self.assertRaises(ValueError):
                conditional8.evaluate(conditional8.reference_step, Forbidden(), **kwargs)
        with self.assertRaises(ValueError):
            conditional8.evaluate(None, Forbidden())
        for output in ((0,) * 4, (False,) + (0,) * 7):
            with self.assertRaises(ValueError):
                conditional8.evaluate(lambda state, cmd: output, carry8.BinaryMemory(), episodes=2)
        class Unreadable(carry8.BinaryMemory):
            def read(self):
                return (0,) * 7
        with self.assertRaises(ValueError):
            conditional8.evaluate(conditional8.reference_step, Unreadable(), episodes=2)


class ConditionalCLITests(unittest.TestCase):
    def test_cli_pass_fail_error_and_exclusive_output(self):
        with tempfile.TemporaryDirectory() as directory, redirect_stdout(StringIO()), redirect_stderr(StringIO()):
            path = Path(directory) / "reference.json"
            args = ["--task", "conditional_logic_8", "--episodes", "2", "--output", str(path)]
            self.assertEqual(main(args), 0)
            initial = path.read_bytes()
            self.assertEqual(json.loads(initial)["profile"], conditional8.PROFILE)
            self.assertEqual(main(args), 2)
            self.assertEqual(path.read_bytes(), initial)
            zero = Path(directory) / "zero.json"
            self.assertEqual(main(args[:-1] + [str(zero), "--controller", "zero"]), 1)
            invalid = Path(directory) / "invalid.json"
            self.assertEqual(main(args[:-1] + [str(invalid), "--episodes", "1"]), 2)
            self.assertFalse(invalid.exists())
            malformed = Path(directory) / "malformed.json"
            with patch.object(conditional8, "reference_step", lambda state, cmd: (0,) * 4):
                self.assertEqual(main(args[:-1] + [str(malformed)]), 2)
            self.assertFalse(malformed.exists())

    def test_example_reuses_same_memory_object_and_fixed_codec_between_tasks(self):
        path = Path(__file__).resolve().parents[1] / "examples" / "reuse_memory_two_tasks.py"
        spec = importlib.util.spec_from_file_location("reuse_memory_two_tasks", path)
        example = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(example)
        memories, codecs, reports = [], [], []
        carry_evaluate, conditional_evaluate = carry8.evaluate, conditional8.evaluate

        def observe(evaluator, next_state, memory, **kwargs):
            memories.append(memory)
            codecs.append((memory._channels, memory._inversions))
            self.assertEqual(memory.read(), (0,) * 8)
            result = evaluator(next_state, memory, **kwargs)
            reports.append(result)
            return result

        with tempfile.TemporaryDirectory() as directory, redirect_stdout(StringIO()), \
                patch.object(carry8, "evaluate", side_effect=lambda *args, **kwargs:
                             observe(carry_evaluate, *args, **kwargs)), \
                patch.object(conditional8, "evaluate", side_effect=lambda *args, **kwargs:
                             observe(conditional_evaluate, *args, **kwargs)):
            args = ["--output-dir", directory, "--episodes", "2"]
            self.assertEqual(example.main(args), 0)
            self.assertIs(memories[0], memories[1])
            self.assertEqual(codecs[0], codecs[1])
            self.assertEqual(reports[0]["suite"], reports[1]["suite"])
            self.assertEqual({path.name for path in Path(directory).iterdir()},
                             {"carry_counter_8.json", "conditional_logic_8.json"})
            with self.assertRaises(FileExistsError):
                example.main(args)
            self.assertEqual(len(memories), 2)


if __name__ == "__main__":
    unittest.main()
