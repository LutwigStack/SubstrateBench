"""Bounded eight-bit carry reference profile; no trained source is bundled."""

import hashlib
import random
import time
from dataclasses import dataclass
from itertools import islice
from pathlib import Path

from .benchmark import _ledger, _metrics
from .task import _bounded

PROFILE = "carry-counter-eight-bit-memory-reference-0.3"
LENGTHS = (24, 128)
ROUTES = ("normal", "permutation", "zero_write", "reset")
MAX_EXCLUDED_IDS = 2_000_000
SAMPLER = "python-mt19937-getrandbits-rejection-upper-active-v1"
IDENTITY = "sha256-carry8-width-layout-length-onehot8-gate-value-v1"
_IDENTITY_PREFIX = b"substratebench-carry8-sequence-v1\x00\x08onehot8-gate-value\x00"
State = tuple[int, int, int, int, int, int, int, int]


def state_bits(state) -> State:
    if not isinstance(state, (tuple, list)) or len(state) != 8:
        raise ValueError("State must contain exactly eight integer bits")
    if any(type(bit) is not int or bit not in (0, 1) for bit in state):
        raise ValueError("State must contain exactly eight integer bits")
    return tuple(state)


def _set_request(register, value):
    if type(register) is not int or not 0 <= register < 8:
        raise ValueError("Register must be an integer in [0, 7]")
    if type(value) is not int or value not in (0, 1):
        raise ValueError("SET value must be an integer bit")


@dataclass(frozen=True)
class Command:
    register: int
    gate: int
    value: int

    def __post_init__(self):
        _set_request(self.register, self.value)
        if type(self.gate) is not int or self.gate not in (0, 1):
            raise ValueError("Gate must be an integer bit")


def reference_step(state: State, command: Command) -> State:
    """Exact modulo-256 arithmetic oracle, not learned-transfer evidence."""
    old = state_bits(state)
    if not isinstance(command, Command):
        raise ValueError("Expected carry8.Command")
    if not command.gate:
        return old
    number = sum(bit << index for index, bit in enumerate(old))
    number = (number + (2 * command.value - 1) * (1 << command.register)) % 256
    return tuple((number >> index) & 1 for index in range(8))


def zero_step(state: State, command: Command) -> State:
    return (0,) * 8


def all_states():
    return tuple(tuple((number >> bit) & 1 for bit in range(8))
                 for number in range(256))


def all_commands():
    return tuple(Command(register, gate, value) for register in range(8)
                 for gate in (0, 1) for value in (0, 1))


def permute_command(command):
    return Command((command.register + 1) % 8, command.gate, command.value)


def sequence_id(commands):
    """Hash prefix, four-byte big-endian length, then onehot8/gate/value bytes."""
    ordered = tuple(islice(commands, 129))
    if not 1 <= len(ordered) <= 128 or any(not isinstance(cmd, Command) for cmd in ordered):
        raise ValueError("A sequence must contain 1 to 128 carry8.Command values")
    encoded = bytes(value for command in ordered
                    for value in (*[int(bit == command.register) for bit in range(8)],
                                  command.gate, command.value))
    return hashlib.sha256(_IDENTITY_PREFIX + len(ordered).to_bytes(4, "big") + encoded).hexdigest()


def _excluded_ids(exclude_ids):
    excluded = set()
    for count, identity in enumerate(exclude_ids, 1):
        if count > MAX_EXCLUDED_IDS:
            raise ValueError(f"Excluded IDs exceed {MAX_EXCLUDED_IDS} entries")
        if (not isinstance(identity, str) or len(identity) != 64
                or any(char not in "0123456789abcdef" for char in identity)):
            raise ValueError("Excluded IDs must be lowercase SHA256 strings")
        excluded.add(identity)
    return excluded


def make_suite(seed=42, episodes=64, exclude_ids=()):
    """Reserve both base and transformed IDs, without inspecting source outputs."""
    if type(seed) is not int or not 0 <= seed < 2**32:
        raise ValueError("Seed must be an integer in [0, 2**32)")
    if type(episodes) is not int or not 2 <= episodes <= 128 or episodes % 2:
        raise ValueError("Episodes per role must be an even integer in [2, 128]")
    seen = _excluded_ids(exclude_ids)
    excluded_count = len(seen)
    suite = []
    for role_index, length in enumerate(LENGTHS):
        rng = random.Random(seed + role_index * 100_003)
        cases = []
        for _ in range(episodes * 100):
            if len(cases) == episodes:
                break
            commands = tuple(Command(_bounded(rng, 8), _bounded(rng, 2),
                                     _bounded(rng, 2)) for _ in range(length))
            permuted = tuple(permute_command(cmd) for cmd in commands)
            if not all(any(cmd.gate and cmd.register >= 4 for cmd in stream)
                       for stream in (commands, permuted)):
                continue
            identity = sequence_id(commands)
            transformed = sequence_id(permuted)
            if identity == transformed or seen.intersection((identity, transformed)):
                continue
            cases.append({"id": identity, "commands": commands})
            seen.update((identity, transformed))
        if len(cases) != episodes:
            raise ValueError("Cannot construct a distinct evaluation suite")
        suite.append({"role": f"random-L{length}", "kind": "random",
                      "length": length, "cases": cases})
    return suite, excluded_count


class BinaryMemory:
    """Eight directly stored bits; the memory computes no task arithmetic."""

    def __init__(self):
        self._bits = [0] * 8

    def reset(self):
        self._bits[:] = (0,) * 8

    def read(self):
        return tuple(self._bits)

    def set_bit(self, register, value):
        _set_request(register, value)
        self._bits[register] = value


class EncodedMemory:
    """Known digital permutation/inversion; only eight encoded bits persist."""

    _channels = (2, 0, 3, 1, 6, 4, 7, 5)
    _inversions = (1, 0, 1, 0, 1, 0, 1, 0)

    def __init__(self):
        self._encoded = [0] * 8
        self.reset()

    def reset(self):
        for register in range(8):
            self._encoded[self._channels[register]] = self._inversions[register]

    def read(self):
        return tuple(self._encoded[channel] ^ invert for channel, invert
                     in zip(self._channels, self._inversions))

    def set_bit(self, register, value):
        _set_request(register, value)
        self._encoded[self._channels[register]] = value ^ self._inversions[register]


def _read(memory, cost):
    cost["memory_reads"] += 1
    return state_bits(memory.read())


def _reset(memory, cost):
    cost["memory_resets"] += 1
    memory.reset()


def _set(memory, register, value, cost):
    cost["memory_SET_calls"] += 1
    memory.set_bit(register, value)


def _write(memory, state, cost):
    for register, value in enumerate(state):
        _set(memory, register, value, cost)


def _qualification(memory, cost):
    reads, writes = [], []
    for state in all_states():
        _reset(memory, cost)
        _write(memory, state, cost)
        observed = _read(memory, cost)
        reads.append({"requested": state, "observed": observed, "correct": observed == state})
        for register in range(8):
            for value in (0, 1):
                _reset(memory, cost)
                _write(memory, state, cost)
                _set(memory, register, value, cost)
                observed = _read(memory, cost)
                expected = list(state)
                expected[register] = value
                writes.append({"initial": state, "register": register, "value": value,
                               "observed": observed, "correct": observed == tuple(expected)})
    reader_correct = sum(row["correct"] for row in reads)
    SET_correct = sum(row["correct"] for row in writes)
    return {"reader_correct": reader_correct, "reader_total": 256,
            "SET_correct": SET_correct, "SET_total": 4096,
            "qualified": reader_correct == 256 and SET_correct == 4096,
            "domain": "exhaustive known finite digital-memory domain",
            "reader_rows": reads, "SET_rows": writes}


def evaluate(next_state, memory, *, seed=42, episodes=64, exclude_ids=(),
             controller_label="user-supplied"):
    """Admit the finite law, then roll out only the supplied memory's own state.

    next_state(bits8, Command) must return eight integer bits and retain no task
    history; fixed weights and callback purity are explicit caller obligations.
    Technical exceptions propagate. Confidence and Torch batch contexts are not
    measured by this scalar bits-only profile.
    """
    started = time.perf_counter()
    if not callable(next_state):
        raise ValueError("next_state must be callable")
    if not isinstance(controller_label, str) or not 1 <= len(controller_label) <= 160:
        raise ValueError("Controller label must contain 1 to 160 characters")
    suite, excluded_count = make_suite(seed, episodes, exclude_ids)
    costs = {"interface_qualification": _ledger(), "source_law_qualification": _ledger(),
             **{route: _ledger() for route in ROUTES}}
    qualification = _qualification(memory, costs["interface_qualification"])
    result = {
        "schema_version": 1, "benchmark_version": "0.3.0", "profile": PROFILE,
        "task_id": "carry_counter_8", "status": "interface_not_qualified", "verdict": "fail",
        "controller": controller_label, "substrate": type(memory).__name__,
        "scope": "known digital eight-bit memory replacement; reference harness",
        "replacement_semantics": "cold: deterministic reset between episodes",
        "calibration": {"regime": "none; supplied controller and memory are fixed",
                        "optimizer_updates": 0, "probe_cost": "interface_qualification ledger",
                        "external_training_cost": None,
                        "external_training_cost_status": "not_measured_by_runner"},
        "memory_contract": {"persistent_task_bits": 8, "write_SET_slots": 8,
                            "callback_statelessness": "caller contract; not verified",
                            "model_freezing": "caller contract; not verified"},
        "evaluation_split": {"sampler": SAMPLER, "seed": seed,
                             "sequence_identity": IDENTITY, "excluded_ID_count": excluded_count,
                             "exclusion_entry_limit": MAX_EXCLUDED_IDS,
                             "training_overlap": "base and permuted supplied IDs excluded; completeness not verified",
                             "permutation_queries": "derived paired commands, not independent holdouts"},
        "suite": [{"role": role["role"], "length": role["length"],
                   "cases": [{"id": case["id"], "commands": [[cmd.register, cmd.gate, cmd.value]
                                                            for cmd in case["commands"]]}
                             for case in role["cases"]]} for role in suite],
        "interface_qualification": qualification,
        "source_law_qualification": {"status": "not_attempted", "qualified": None,
                                     "correct": None, "total": 8192,
                                     "signed_margin": None, "margin_status": "not_measured"},
        "route_status": "not_attempted",
        "routes": {route: {role["role"]: None for role in suite} for route in ROUTES},
        "criteria": {"thresholds": {"prefix_state_accuracy": .99,
                                    "whole_sequence_accuracy": .95, "control_whole_drop": .30},
                     "per_length": None},
        "cost": {"phases": costs},
        "implementation_sha256": {
            name: hashlib.sha256((Path(__file__).parent / name).read_bytes()).hexdigest()
            for name in ("carry8.py", "task.py", "benchmark.py")},
    }
    if qualification["qualified"]:
        law_cost = costs["source_law_qualification"]
        rows = []
        for state in all_states():
            for command in all_commands():
                law_cost["controller_calls"] += 1
                predicted = state_bits(next_state(state, command))
                law_cost["reference_label_steps"] += 1
                expected = reference_step(state, command)
                rows.append({"state": state, "command": [command.register, command.gate, command.value],
                             "predicted": predicted, "correct": predicted == expected})
        correct = sum(row["correct"] for row in rows)
        result["source_law_qualification"] = {
            "status": "completed", "qualified": correct == 8192, "correct": correct, "total": 8192,
            "signed_margin": None, "margin_status": "not_measured",
            "interpretation": "finite supported-law admission; not unseen-state generalization",
            "rows": rows}
        result["status"] = "completed" if correct == 8192 else "source_not_qualified"
        if correct == 8192:
            _run_routes(result, suite, next_state, memory, costs)
    result["cost"]["totals"] = {key: sum(phase[key] for phase in costs.values()) for key in _ledger()}
    result["cost"]["units"] = "visible Python calls; each whole-state write is eight SET calls"
    result["cost"]["runner_wall_seconds"] = time.perf_counter() - started
    result["cost"]["wall_scope"] = "evaluate() including suite generation; excludes import/install/report I/O"
    return result


def _run_routes(result, suite, next_state, memory, costs):
    for route in ROUTES:
        cost = costs[route]
        for role in suite:
            outputs, targets, identities = [], [], []
            for case in role["cases"]:
                _reset(memory, cost)
                _write(memory, (0,) * 8, cost)
                predicted, gold_states = [], []
                gold = (0,) * 8
                commands = tuple(permute_command(cmd) if route == "permutation" else cmd
                                 for cmd in case["commands"])
                identities.append(sequence_id(commands))
                for command in commands:
                    if route == "reset":
                        _reset(memory, cost)
                        _write(memory, (0,) * 8, cost)
                    current = _read(memory, cost)
                    cost["controller_calls"] += 1
                    proposed = state_bits(next_state(current, command))
                    if route != "zero_write":
                        _write(memory, proposed, cost)
                    predicted.append(_read(memory, cost))
                    cost["reference_label_steps"] += 1
                    gold = reference_step(gold, command)
                    gold_states.append(gold)
                outputs.append(predicted)
                targets.append(gold_states)
            result["routes"][route][role["role"]] = {
                "metrics": _metrics(outputs, targets), "sequence_ids": identities,
                "outputs": outputs, "role": "primary"}
    per_length = {}
    for length in LENGTHS:
        name = f"random-L{length}"
        normal = result["routes"]["normal"][name]["metrics"]
        permuted = result["routes"]["permutation"][name]["metrics"]
        per_length[name] = {
            "normal_quality": normal["prefix_state_accuracy"] >= .99 and normal["whole_sequence_accuracy"] >= .95,
            "permuted_task_quality": permuted["prefix_state_accuracy"] >= .99 and permuted["whole_sequence_accuracy"] >= .95,
            **{f"{control}_whole_drop": normal["whole_sequence_accuracy"] -
               result["routes"][control][name]["metrics"]["whole_sequence_accuracy"] >= .30
               for control in ("zero_write", "reset")}}
    result["route_status"] = "completed"
    result["criteria"]["per_length"] = per_length
    if all(all(values.values()) for values in per_length.values()):
        result["verdict"] = "pass"
