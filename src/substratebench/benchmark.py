"""Memory-ownership evaluation and atomic, exclusive result persistence."""

import hashlib
import json
import os
import tempfile
import time
from itertools import islice
from pathlib import Path

from .task import (
    LENGTHS,
    SAMPLER,
    TASKS,
    all_commands,
    all_states,
    make_suite,
    permute_command,
    reference_step,
    sequence_id,
    state_bits,
)

PROFILE = "gated-xor-memory-reference-0.1"
PROFILES = {"gated_xor": PROFILE,
            "carry_counter": "carry-counter-memory-reference-0.2",
            "conditional_logic": "conditional-logic-memory-reference-0.2"}
ROUTES = ("normal", "permutation", "zero", "reset")


def _ledger():
    return {"memory_reads": 0, "memory_resets": 0, "memory_SET_calls": 0,
            "controller_calls": 0, "reference_label_steps": 0}


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


def _metrics(outputs, targets):
    correct = [[actual == target for actual, target in zip(episode, gold)]
               for episode, gold in zip(outputs, targets)]
    episodes = len(correct)
    prefixes = sum(len(item) for item in correct)
    return {"episodes": episodes, "prefixes": prefixes,
            "prefix_state_accuracy": sum(sum(item) for item in correct) / prefixes,
            "final_state_accuracy": sum(item[-1] for item in correct) / episodes,
            "whole_sequence_accuracy": sum(all(item) for item in correct) / episodes}


def _qualification(memory, cost):
    reads, writes = [], []
    for state in all_states():
        _reset(memory, cost)
        _write(memory, state, cost)
        observed = _read(memory, cost)
        reads.append({"requested": state, "observed": observed,
                      "correct": observed == state})
    for state in all_states():
        for register in range(4):
            for value in (0, 1):
                _reset(memory, cost)
                _write(memory, state, cost)
                _set(memory, register, value, cost)
                observed = _read(memory, cost)
                target = list(state)
                target[register] = value
                writes.append({"initial": state, "register": register,
                               "value": value, "observed": observed,
                               "correct": observed == tuple(target)})
    read_correct = sum(item["correct"] for item in reads)
    write_correct = sum(item["correct"] for item in writes)
    return {"reader_correct": read_correct, "reader_total": 16,
            "SET_correct": write_correct, "SET_total": 128,
            "qualified": read_correct == 16 and write_correct == 128,
            "domain": "exhaustive known finite digital-memory domain",
            "reader_rows": reads, "SET_rows": writes}


def evaluate(next_state, memory, *, seed=42, episodes=64, exclude_ids=(),
             controller_label="user-supplied", task="gated_xor"):
    """Run a stateless controller against memory, without gold or history inputs.

    next_state(current_bits, Command) must return four Python integer bits.
    memory owns persistent history and exposes read(), set_bit(j, bit), reset().
    These are caller contracts: Python cannot prove that a callback has no cache.
    Exceptions propagate; no successful artifact is produced for a technical error.
    """
    started = time.perf_counter()
    if not callable(next_state):
        raise ValueError("next_state must be callable")
    if task not in TASKS:
        raise ValueError("Unknown task law")
    if not isinstance(controller_label, str) or not 1 <= len(controller_label) <= 160:
        raise ValueError("Controller label must contain 1 to 160 characters")
    excluded = tuple(islice(exclude_ids, 100_001))
    suite = make_suite(seed, episodes, excluded)
    costs = {"interface_qualification": _ledger(), "source_law_diagnostic": _ledger(),
             **{route: _ledger() for route in ROUTES}}
    qualification = _qualification(memory, costs["interface_qualification"])
    result = {
        "schema_version": 1, "benchmark_version": "0.2.0", "profile": PROFILES[task], "task_id": task,
        "status": "completed" if qualification["qualified"] else "interface_not_qualified",
        "verdict": "fail", "controller": controller_label,
        "substrate": type(memory).__name__, "scope": "known digital memory replacement",
        "replacement_semantics": "cold: deterministic reset between episodes",
        "calibration": {"regime": "none; supplied controller and memory interface are fixed",
                        "optimizer_updates": 0, "probe_cost": "interface_qualification ledger",
                        "external_training_cost": None,
                        "external_training_cost_status": "not_measured_by_runner"},
        "memory_contract": {"persistent_task_bits": 4, "write_SET_slots": 4,
                            "callback_statelessness": "caller contract; not verified"},
        "evaluation_split": {"sampler": SAMPLER, "seed": seed,
                             "sequence_identity": "sha256-uint8-onehot4-gate-value-v1",
                             "excluded_ID_count": len(set(excluded)),
                             "training_overlap": "base and permuted supplied IDs excluded; completeness not verified",
                             "permutation_queries": "derived paired commands, not independent holdouts"},
        "suite": [{"role": role["role"], "kind": role["kind"], "length": role["length"],
                   "cases": [{"id": case["id"],
                              "commands": [[cmd.register, cmd.gate, cmd.value]
                                           for cmd in case["commands"]]}
                             for case in role["cases"]]} for role in suite],
        "interface_qualification": qualification,
        "source_law_diagnostic": {"status": "not_attempted"},
        "routes": {}, "criteria": {}, "cost": {"phases": costs},
        "implementation_sha256": {
            name: hashlib.sha256((Path(__file__).parent / name).read_bytes()).hexdigest()
            for name in ("task.py", "memory.py", "benchmark.py")},
    }
    if qualification["qualified"]:
        diagnostic_cost = costs["source_law_diagnostic"]
        rows = []
        for state in all_states():
            for command in all_commands():
                diagnostic_cost["controller_calls"] += 1
                predicted = state_bits(next_state(state, command))
                diagnostic_cost["reference_label_steps"] += 1
                expected = reference_step(state, command, task=task)
                rows.append({"state": state,
                             "command": [command.register, command.gate, command.value],
                             "predicted": predicted, "correct": predicted == expected})
        result["source_law_diagnostic"] = {
            "status": "completed", "correct": sum(row["correct"] for row in rows),
            "total": 256, "interpretation": "finite-domain diagnostic, not generalization",
            "rows": rows}
        for route in ROUTES:
            route_cost = costs[route]
            route_result = {}
            for role in suite:
                outputs, targets, actual_ids = [], [], []
                for case in role["cases"]:
                    _reset(memory, route_cost)
                    _write(memory, (0, 0, 0, 0), route_cost)
                    predicted, gold_states = [], []
                    gold = (0, 0, 0, 0)
                    commands = tuple(permute_command(cmd) if route == "permutation" else cmd
                                     for cmd in case["commands"])
                    actual_ids.append(sequence_id(commands))
                    for command in commands:
                        if route == "reset":
                            _reset(memory, route_cost)
                            _write(memory, (0, 0, 0, 0), route_cost)
                        current = _read(memory, route_cost)
                        route_cost["controller_calls"] += 1
                        proposed = state_bits(next_state(current, command))
                        if route != "zero":
                            _write(memory, proposed, route_cost)
                        observed = _read(memory, route_cost)
                        predicted.append(observed)
                        route_cost["reference_label_steps"] += 1
                        gold = reference_step(gold, command, task=task)
                        gold_states.append(gold)
                    outputs.append(predicted)
                    targets.append(gold_states)
                route_result[role["role"]] = {
                    "metrics": _metrics(outputs, targets), "sequence_ids": actual_ids,
                    "outputs": outputs,
                    "role": "primary" if role["kind"] == "random" else "descriptive"}
            result["routes"][route] = route_result
        criteria = {}
        for length in LENGTHS:
            name = f"random-L{length}"
            normal = result["routes"]["normal"][name]
            permuted = result["routes"]["permutation"][name]
            criteria[name] = {
                "normal_quality": normal["metrics"]["prefix_state_accuracy"] >= .99
                                  and normal["metrics"]["whole_sequence_accuracy"] >= .95,
                "permuted_task_quality": permuted["metrics"]["prefix_state_accuracy"] >= .99
                                        and permuted["metrics"]["whole_sequence_accuracy"] >= .95,
                **{f"{control}_whole_drop":
                   normal["metrics"]["whole_sequence_accuracy"] -
                   result["routes"][control][name]["metrics"]["whole_sequence_accuracy"] >= .30
                   for control in ("zero", "reset")},
            }
        result["criteria"] = {"thresholds": {"prefix_state_accuracy": .99,
                                               "whole_sequence_accuracy": .95,
                                               "control_whole_drop": .30},
                              "per_length": criteria}
        if all(all(values.values()) for values in criteria.values()):
            result["verdict"] = "pass"
    result["cost"]["totals"] = {key: sum(phase[key] for phase in costs.values())
                                      for key in _ledger()}
    result["cost"]["units"] = (
        "visible Python interface calls; each write is four SET calls; no physical energy units")
    result["cost"]["runner_wall_seconds"] = time.perf_counter() - started
    result["cost"]["wall_scope"] = "evaluate() including suite generation; excludes import/install/report I/O"
    return result


def write_result(result, destination):
    """Publish complete JSON atomically, failing if the destination already exists."""
    destination = Path(destination)
    encoded = (json.dumps(result, allow_nan=False, separators=(",", ":")) + "\n").encode()
    if len(encoded) > 16 * 1024 * 1024:
        raise ValueError("Result exceeds the 16 MiB artifact limit")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=destination.parent, prefix=".substratebench-",
                                         delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
        # Hard linking makes publication exclusive even when another writer races.
        os.link(temporary, destination)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
