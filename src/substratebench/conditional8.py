"""Bounded eight-bit OLD-state conditional law; no trained source is bundled."""

import hashlib
import math
import time
from pathlib import Path

from . import carry8
from .benchmark import _ledger, _metrics
from .carry8 import (
    BinaryMemory, Command, EncodedMemory, State, all_commands, all_states,
    make_suite, permute_command, sequence_id, state_bits,
)

PROFILE = "conditional-logic-eight-bit-memory-reference-0.4"


def reference_step(state: State, command: Command) -> State:
    """Apply simultaneous updates from OLD bits; this is a reference oracle."""
    old = state_bits(state)
    if not isinstance(command, Command):
        raise ValueError("Expected carry8.Command")
    result = list(old)
    if command.gate:
        register = command.register
        neighbor = (register + 1) % 8
        if command.value == 0:
            result[register], result[neighbor] = old[neighbor], old[register]
        else:
            result[register] ^= 1
            result[(register + 2) % 8] ^= old[neighbor]
    return tuple(result)


def zero_step(state: State, command: Command) -> State:
    """A deliberately incapable source control."""
    return (0,) * 8


def evaluate(next_state, memory, *, seed=42, episodes=64, exclude_ids=(),
             controller_label="user-supplied", source_creation_wall_seconds=None):
    """Qualify the conditional law, then execute only memory's actual read state.

    Weights, freezing and absence of hidden callback history are caller contracts.
    Optional source creation seconds are caller-reported, never runner-measured.
    Technical exceptions propagate; scientific admission failures return reports.
    """
    started = time.perf_counter()
    if not callable(next_state):
        raise ValueError("next_state must be callable")
    if not isinstance(controller_label, str) or not 1 <= len(controller_label) <= 160:
        raise ValueError("Controller label must contain 1 to 160 characters")
    if source_creation_wall_seconds is not None:
        if type(source_creation_wall_seconds) not in (int, float):
            raise ValueError("Source creation wall seconds must be finite and nonnegative")
        try:
            valid_cost = math.isfinite(source_creation_wall_seconds) and source_creation_wall_seconds >= 0
        except OverflowError:
            valid_cost = False
        if not valid_cost:
            raise ValueError("Source creation wall seconds must be finite and nonnegative")
    suite, excluded_count = make_suite(seed, episodes, exclude_ids)
    costs = {"interface_qualification": _ledger(), "source_law_qualification": _ledger(),
             **{route: _ledger() for route in carry8.ROUTES}}
    qualification = carry8._qualification(memory, costs["interface_qualification"])
    kind = ("reference_oracle" if next_state is reference_step else
            "zero_source_control" if next_state is zero_step else "caller_supplied")
    result = {
        "schema_version": 1, "benchmark_version": "0.4.0", "profile": PROFILE,
        "task_id": "conditional_logic_8", "status": "interface_not_qualified", "verdict": "fail",
        "controller": controller_label, "substrate": type(memory).__name__,
        "scope": "known digital eight-bit memory replacement; reference harness",
        "replacement_semantics": "cold: deterministic reset between episodes",
        "source": {"kind": kind, "identity": controller_label,
                   "identity_status": "caller-declared; provenance not verified",
                   "creation_wall_seconds": source_creation_wall_seconds,
                   "creation_cost_status": "UNKNOWN" if source_creation_wall_seconds is None else
                                           "caller_reported_not_audited",
                   "creation_cost_scope": "external source creation; excluded from runner totals",
                   "energy": None, "energy_status": "UNKNOWN"},
        "calibration": {"regime": "none; supplied controller and memory are fixed",
                        "optimizer_updates": 0, "probe_cost": "interface_qualification ledger",
                        "external_training_cost": None, "external_training_cost_status": "UNKNOWN"},
        "memory_contract": {"persistent_task_bits": 8, "write_SET_slots": 8,
                            "callback_statelessness": "caller contract; not verified",
                            "model_freezing": "caller contract; not verified"},
        "evaluation_split": {"sampler": carry8.SAMPLER, "seed": seed,
                             "sequence_identity": carry8.IDENTITY,
                             "identity_scope": "command content shared with carry8; task_id identifies law",
                             "excluded_ID_count": excluded_count,
                             "exclusion_entry_limit": carry8.MAX_EXCLUDED_IDS,
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
        "routes": {route: {role["role"]: None for role in suite} for route in carry8.ROUTES},
        "criteria": {"thresholds": {"prefix_state_accuracy": .99,
                                    "whole_sequence_accuracy": .95, "control_whole_drop": .30},
                     "per_length": None},
        "cost": {"phases": costs},
        "implementation_sha256": {
            name: hashlib.sha256((Path(__file__).parent / name).read_bytes()).hexdigest()
            for name in ("conditional8.py", "carry8.py", "task.py", "benchmark.py")},
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
    result["cost"]["physical_time_status"] = "not measured; memory ABI has no wait or clock"
    return result


def _run_routes(result, suite, next_state, memory, costs):
    for route in carry8.ROUTES:
        cost = costs[route]
        for role in suite:
            outputs, targets, identities = [], [], []
            for case in role["cases"]:
                carry8._reset(memory, cost)
                carry8._write(memory, (0,) * 8, cost)
                predicted, gold_states = [], []
                gold = (0,) * 8
                commands = tuple(permute_command(cmd) if route == "permutation" else cmd
                                 for cmd in case["commands"])
                identities.append(sequence_id(commands))
                for command in commands:
                    if route == "reset":
                        carry8._reset(memory, cost)
                        carry8._write(memory, (0,) * 8, cost)
                    current = carry8._read(memory, cost)
                    cost["controller_calls"] += 1
                    proposed = state_bits(next_state(current, command))
                    if route != "zero_write":
                        carry8._write(memory, proposed, cost)
                    predicted.append(carry8._read(memory, cost))
                    cost["reference_label_steps"] += 1
                    gold = reference_step(gold, command)
                    gold_states.append(gold)
                outputs.append(predicted)
                targets.append(gold_states)
            result["routes"][route][role["role"]] = {
                "metrics": _metrics(outputs, targets), "sequence_ids": identities,
                "outputs": outputs, "role": "primary"}
    per_length = {}
    for length in carry8.LENGTHS:
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
