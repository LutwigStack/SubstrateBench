import hashlib
import unittest
from functools import partial

from substratebench import BinaryMemory, Command, evaluate, reference_step, sequence_id
from substratebench.task import all_commands, all_states


class AdditionalTaskTests(unittest.TestCase):
    def test_complete_carry_and_conditional_laws(self):
        for state in all_states():
            number = sum(bit << index for index, bit in enumerate(state))
            for command in all_commands():
                j = command.register
                expected_carry = number
                expected_conditional = number
                if command.gate:
                    expected_carry = (number + (1 if command.value else -1) * (1 << j)) % 16
                    if command.value:
                        expected_conditional ^= (1 << j) | (((number >> ((j + 1) % 4)) & 1) << ((j + 2) % 4))
                    else:
                        difference = ((number >> j) ^ (number >> ((j + 1) % 4))) & 1
                        expected_conditional ^= (difference << j) | (difference << ((j + 1) % 4))
                for task, expected in (("carry_counter", expected_carry),
                                       ("conditional_logic", expected_conditional)):
                    actual = reference_step(state, command, task=task)
                    self.assertEqual(sum(bit << index for index, bit in enumerate(actual)), expected)

    def test_wraparound_and_simultaneous_conditional_update(self):
        self.assertEqual(reference_step((1, 1, 1, 1), Command(0, 1, 1), task="carry_counter"), (0, 0, 0, 0))
        self.assertEqual(reference_step((0, 0, 0, 0), Command(0, 1, 0), task="carry_counter"), (1, 1, 1, 1))
        self.assertEqual(reference_step((0, 1, 0, 0), Command(0, 1, 0), task="conditional_logic"), (1, 0, 0, 0))
        self.assertEqual(reference_step((0, 1, 0, 0), Command(0, 1, 1), task="conditional_logic"), (1, 1, 1, 0))

    def test_new_task_reports_and_transformed_gold_use_selected_law(self):
        for task in ("carry_counter", "conditional_logic"):
            result = evaluate(partial(reference_step, task=task), BinaryMemory(), task=task, episodes=4)
            self.assertEqual(result["verdict"], "pass")
            self.assertEqual(result["task_id"], task)
            self.assertEqual(result["benchmark_version"], "0.2.0")
            self.assertEqual(result["source_law_diagnostic"]["correct"], 256)
            for route in ("normal", "permutation"):
                self.assertTrue(all(row["metrics"]["whole_sequence_accuracy"] == 1
                                    for row in result["routes"][route].values()))

    def test_invalid_task_rejected_before_controller_or_memory_access(self):
        def forbidden(*args):
            self.fail("Invalid task must fail before invoking caller code")

        class ForbiddenMemory:
            reset = read = set_bit = forbidden

        with self.assertRaises(ValueError):
            evaluate(forbidden, ForbiddenMemory(), task="unknown", episodes=2)
        with self.assertRaises(ValueError):
            reference_step((0, 0, 0, 0), Command(0, 0, 0), task="unknown")

    def test_lab_raw_commands_regenerate_public_ids_without_changing_public_codec(self):
        rows = [(0, 0, 1, 0, 1, 0), (1, 0, 0, 0, 0, 1)]
        commands = [Command(list(row[:4]).index(1), gate=row[5], value=row[4]) for row in rows]
        lab_digest = hashlib.sha256(bytes(value for row in rows for value in row)).hexdigest()
        public_digest = sequence_id(commands)
        self.assertNotEqual(public_digest, lab_digest)
        expected_public = bytes((0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 1, 0))
        self.assertEqual(public_digest, hashlib.sha256(expected_public).hexdigest())


if __name__ == "__main__":
    unittest.main()
