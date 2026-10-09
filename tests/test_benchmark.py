import json
from pathlib import Path
import tempfile
import unittest
from concurrent.futures import ThreadPoolExecutor

from substratebench import (BinaryMemory, Command, EncodedMemory, evaluate,
                            make_suite, reference_step, sequence_id, write_result,
                            zero_step)
from substratebench.task import all_commands, all_states, permute_command, state_bits


class TaskTests(unittest.TestCase):
    def test_entire_law_preserves_other_registers_and_obeys_gate(self):
        for state in all_states():
            for command in all_commands():
                actual = reference_step(state, command)
                expected = list(state)
                if command.gate and command.value:
                    expected[command.register] = 1 - expected[command.register]
                self.assertEqual(actual, tuple(expected))

    def test_command_and_state_trust_boundaries(self):
        for args in ((True, 1, 1), (4, 1, 1), (0, .5, 1), (0, 1, False)):
            with self.assertRaises(ValueError):
                Command(*args)
        for state in ((1, 2, 0, 0), (True, 0, 0, 0), (0, 0, 0), "0000"):
            with self.assertRaises(ValueError):
                state_bits(state)

    def test_reproducible_cases_exclusion_and_counterfactuals(self):
        first = make_suite(42, 4)
        self.assertEqual(first, make_suite(42, 4))
        ids = [case["id"] for role in first for case in role["cases"]]
        self.assertEqual(len(ids), len(set(ids)))
        second = make_suite(42, 4, ids)
        self.assertFalse(set(ids).intersection(case["id"] for role in second
                                              for case in role["cases"]))
        for role in first:
            if role["kind"] not in ("history", "gate"):
                continue
            for left, right in zip(role["cases"][::2], role["cases"][1::2]):
                states = []
                for case in (left, right):
                    state = (0, 0, 0, 0)
                    for command in case["commands"]:
                        state = reference_step(state, command)
                    states.append(state)
                self.assertEqual(sum(a != b for a, b in zip(*states)), 1)
                if role["kind"] == "history":
                    self.assertEqual(left["commands"][-1], right["commands"][-1])

    def test_historical_identity_codec(self):
        import hashlib
        commands = (Command(2, 1, 0), Command(0, 1, 1))
        encoded = bytes((0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 1, 1))
        self.assertEqual(sequence_id(commands), hashlib.sha256(encoded).hexdigest())

    def test_exclusion_covers_permuted_queries_and_rejects_pairs_atomically(self):
        original = make_suite(42, 4)
        transformed_ids = [sequence_id(tuple(permute_command(cmd) for cmd in case["commands"]))
                           for role in original for case in role["cases"]]
        replacement = make_suite(42, 4, transformed_ids)
        all_ids = []
        for role in replacement:
            for case in role["cases"]:
                all_ids.append(case["id"])
                all_ids.append(sequence_id(tuple(permute_command(cmd)
                                                 for cmd in case["commands"])))
        self.assertEqual(len(all_ids), len(set(all_ids)))
        self.assertFalse(set(transformed_ids).intersection(all_ids))
        for role in replacement:
            if role["kind"] in ("history", "gate"):
                for left, right in zip(role["cases"][::2], role["cases"][1::2]):
                    differences = sum(a != b for a, b in
                                      zip(left["commands"], right["commands"]))
                    self.assertEqual(differences, 1)


class MemoryTests(unittest.TestCase):
    def test_all_generic_SET_operations_preserve_other_bits(self):
        for memory_type in (BinaryMemory, EncodedMemory):
            memory = memory_type()
            for state in all_states():
                for register in range(4):
                    for value in (0, 1):
                        memory.reset()
                        for bit, initial in enumerate(state):
                            memory.set_bit(bit, initial)
                        memory.set_bit(register, value)
                        expected = list(state)
                        expected[register] = value
                        self.assertEqual(memory.read(), tuple(expected))

    def test_encoded_carrier_has_only_encoded_persistent_storage(self):
        memory = EncodedMemory()
        memory.set_bit(0, 1)
        self.assertEqual(memory.__dict__, {"_encoded": [0, 0, 0, 1]})
        self.assertEqual(memory.read(), (1, 0, 0, 0))


class BenchmarkTests(unittest.TestCase):
    def test_real_reference_run_and_memory_controls(self):
        result = evaluate(reference_step, EncodedMemory(), seed=42, episodes=4,
                          controller_label="reference_oracle")
        self.assertEqual(result["verdict"], "pass")
        self.assertEqual(result["interface_qualification"]["reader_correct"], 16)
        self.assertEqual(result["interface_qualification"]["SET_correct"], 128)
        self.assertEqual(result["source_law_diagnostic"]["correct"], 256)
        for role, data in result["routes"]["normal"].items():
            self.assertEqual(data["metrics"]["whole_sequence_accuracy"], 1)
        for phase in ("zero", "reset"):
            self.assertLess(result["routes"][phase]["random-L128"]["metrics"]
                            ["whole_sequence_accuracy"], .7)
        qualification_cost = result["cost"]["phases"]["interface_qualification"]
        self.assertEqual(qualification_cost["memory_SET_calls"], 704)
        self.assertEqual(qualification_cost["memory_reads"], 144)
        self.assertEqual(qualification_cost["memory_resets"], 144)
        self.assertEqual(qualification_cost["controller_calls"], 0)

    def test_zero_source_is_a_scientific_failure(self):
        result = evaluate(zero_step, BinaryMemory(), episodes=2)
        self.assertEqual(result["status"], "completed")
        self.assertEqual(result["verdict"], "fail")
        self.assertLess(result["source_law_diagnostic"]["correct"], 256)

    def test_collateral_writer_failure_censors_task_and_calls_no_source(self):
        class BrokenMemory(BinaryMemory):
            def set_bit(self, register, value):
                self._bits[:] = (0, 0, 0, 0)
                super().set_bit(register, value)

        def forbidden(state, command):
            self.fail("Source must not run after interface qualification fails")

        result = evaluate(forbidden, BrokenMemory(), episodes=2)
        self.assertEqual(result["status"], "interface_not_qualified")
        self.assertFalse(result["interface_qualification"]["qualified"])
        self.assertEqual(result["routes"], {})
        self.assertEqual(result["source_law_diagnostic"]["status"], "not_attempted")
        self.assertEqual(result["cost"]["totals"]["controller_calls"], 0)

    def test_callback_only_gets_current_state_and_command_and_invalid_output_fails(self):
        observed = []

        def source(state, command):
            observed.append((state, command))
            self.assertIsInstance(state, tuple)
            self.assertIsInstance(command, Command)
            return reference_step(state, command)

        result = evaluate(source, BinaryMemory(), episodes=2)
        self.assertEqual(len(observed), result["cost"]["totals"]["controller_calls"])
        with self.assertRaises(ValueError):
            evaluate(lambda state, command: (False, 0, 0, 0), BinaryMemory(), episodes=2)


class ArtifactTests(unittest.TestCase):
    def test_exclusive_atomic_publication_with_competing_writers(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "result.json"

            def publish(value):
                try:
                    write_result({"value": value}, path)
                    return "ok"
                except FileExistsError:
                    return "exists"

            with ThreadPoolExecutor(max_workers=2) as pool:
                outcomes = list(pool.map(publish, (1, 2)))
            self.assertEqual(sorted(outcomes), ["exists", "ok"])
            original = path.read_bytes()
            self.assertIn(json.loads(original)["value"], (1, 2))
            with self.assertRaises(FileExistsError):
                write_result({"value": 3}, path)
            self.assertEqual(path.read_bytes(), original)
            self.assertEqual(list(Path(directory).iterdir()), [path])

    def test_nonfinite_result_never_creates_artifact(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "result.json"
            with self.assertRaises(ValueError):
                write_result({"value": float("nan")}, path)
            self.assertFalse(path.exists())


if __name__ == "__main__":
    unittest.main()
