"""Three four-bit finite task laws and reproducible public evaluation cases."""

import hashlib
import random
from dataclasses import dataclass
from itertools import islice

REGISTERS = 4
LENGTHS = (12, 24, 64, 128)
SAMPLER = "python-mt19937-getrandbits-rejection-v1"
TASKS = ("gated_xor", "carry_counter", "conditional_logic")
State = tuple[int, int, int, int]


def state_bits(state) -> State:
    """Reject malformed outputs at the user-controller boundary."""
    if not isinstance(state, (tuple, list)) or len(state) != REGISTERS:
        raise ValueError("State must contain exactly four integer bits")
    if any(type(bit) is not int or bit not in (0, 1) for bit in state):
        raise ValueError("State must contain exactly four integer bits")
    return tuple(state)


@dataclass(frozen=True)
class Command:
    register: int
    gate: int
    value: int

    def __post_init__(self):
        if type(self.register) is not int or not 0 <= self.register < REGISTERS:
            raise ValueError("Register must be an integer in [0, 3]")
        if any(type(value) is not int or value not in (0, 1)
               for value in (self.gate, self.value)):
            raise ValueError("Gate and value must be integer bits")


def reference_step(state: State, command: Command, *, task="gated_xor") -> State:
    """An exact-rule oracle for testing the harness, not a trained model."""
    old = state_bits(state)
    if task not in TASKS:
        raise ValueError("Unknown task law")
    result = list(old)
    register = command.register
    if task == "gated_xor":
        result[register] ^= command.gate & command.value
    elif command.gate:
        if task == "carry_counter":
            number = sum(bit << index for index, bit in enumerate(old))
            number = (number + (2 * command.value - 1) * (1 << register)) % 16
            result = [(number >> index) & 1 for index in range(4)]
        elif command.value == 0:
            neighbor = (register + 1) % 4
            result[register], result[neighbor] = old[neighbor], old[register]
        else:
            result[register] ^= 1
            result[(register + 2) % 4] ^= old[(register + 1) % 4]
    return tuple(result)


def zero_step(state: State, command: Command) -> State:
    """A deliberately incapable source control."""
    return (0, 0, 0, 0)


def all_states():
    return tuple(tuple((number >> bit) & 1 for bit in range(REGISTERS))
                 for number in range(16))


def all_commands():
    return tuple(Command(register, gate, value)
                 for register in range(REGISTERS)
                 for gate in (0, 1) for value in (0, 1))


def sequence_id(commands):
    """SHA256 of public six-byte one-hot/gate/value commands; Lab swaps the last two."""
    return hashlib.sha256(bytes(value for command in commands
                                for value in (*[int(bit == command.register)
                                                for bit in range(4)],
                                              command.gate, command.value))).hexdigest()


def permute_command(command):
    return Command((command.register + 1) % REGISTERS, command.gate, command.value)


def _bounded(rng, bound):
    value = rng.getrandbits(bound.bit_length())
    while value >= bound:
        value = rng.getrandbits(bound.bit_length())
    return value


def make_suite(seed=42, episodes=64, exclude_ids=()):
    """Generate ordered sequences, excluding only the IDs supplied by the caller.

    The benchmark cannot verify that the supplied training-ID list is complete.
    Diagnostic pairs differ at one earlier or gated command; they are descriptive.
    """
    if type(seed) is not int or not 0 <= seed < 2**32:
        raise ValueError("Seed must be an integer in [0, 2**32)")
    if type(episodes) is not int or not 2 <= episodes <= 128 or episodes % 2:
        raise ValueError("Episodes per role must be an even integer in [2, 128]")
    excluded = tuple(islice(exclude_ids, 100_001))
    if len(excluded) > 100_000 or any(
        not isinstance(item, str) or len(item) != 64
        or any(char not in "0123456789abcdef" for char in item)
        for item in excluded
    ):
        raise ValueError("Excluded IDs must be at most 100000 lowercase SHA256 strings")
    seen = set(excluded)
    suite = []
    roles = [(f"random-L{length}", length, "random") for length in LENGTHS]
    roles += [(f"diagnostic-{kind}", 24, kind)
              for kind in ("history", "gate", "repeat")]
    for role_index, (name, length, kind) in enumerate(roles):
        rng = random.Random(seed + role_index * 100_003)
        cases = []
        attempts = 0
        while len(cases) < episodes:
            attempts += 1
            if attempts > episodes * 100:
                raise ValueError("Cannot construct a distinct evaluation suite")
            commands = tuple(Command(_bounded(rng, 4), _bounded(rng, 2),
                                     _bounded(rng, 2)) for _ in range(length))
            pair = [commands]
            if kind in ("history", "gate"):
                tick = 0 if kind == "history" else length - 1
                selected = commands[tick].register
                left, right = list(commands), list(commands)
                left[tick] = Command(selected, 1, 0)
                right[tick] = Command(selected, 1, 1)
                pair = [tuple(left), tuple(right)]
            elif kind == "repeat":
                values = list(commands)
                selected = _bounded(rng, 4)
                for tick in (0, length // 2, length - 1):
                    values[tick] = Command(selected, 1, 1)
                pair = [tuple(values)]
            identities = [sequence_id(item) for item in pair]
            transformed = [sequence_id(tuple(permute_command(cmd) for cmd in item))
                           for item in pair]
            reserved = identities + transformed
            if len(set(reserved)) != len(reserved) or seen.intersection(reserved):
                continue
            for item, identity in zip(pair, identities):
                cases.append({"id": identity, "commands": item})
            seen.update(reserved)
        suite.append({"role": name, "kind": kind, "length": length,
                      "cases": cases})
    return suite
