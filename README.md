# SubstrateBench

**STATUS: implemented — public reference benchmark, version 0.2.0.**

SubstrateBench measures whether a fixed task controller can keep using a four-bit
memory after replacing its storage implementation. The package is a small,
reproducible digital benchmark that other people can install and extend. It has
no runtime dependencies.

The initial harness is scoped in [SB-001](docs/promotion/SB-001.md) and the two
additional task laws in [SB-002](docs/promotion/SB-002.md);
repository boundaries are described in [AGENTS.md](AGENTS.md).

The bundled controller is an exact-rule **reference oracle**, not a trained model.
Its successful run validates the benchmark harness. Historical learned-source
experiments, failures, reviews, and cost comparisons belong in
[substratebench-lab](https://github.com/LutwigStack/substratebench-lab).
The lab results are separate evidence; this package does not reproduce them.

## Run it

Requires Python 3.10 or newer. From a checkout:

```sh
python -m pip install .
substratebench --output results/reference.json
python -m substratebench --memory binary --output results/binary.json
python -m substratebench --controller zero --output results/negative.json
substratebench --task carry_counter --output results/carry.json
substratebench --task conditional_logic --output results/conditional.json
```

The oracle commands run the reference controller. The zero control intentionally
fails the quality criteria and returns exit code 1. Exit code 0 means the declared
reference-profile criteria passed. Exit code 2 means invalid arguments, malformed
controller output, or an I/O error. Output paths must be new: existing artifacts
are never overwritten. Results are compact JSON with raw commands and predictions.

The default run evaluates 448 sequences: 64 per role, four random lengths
(12, 24, 64, 128), and three descriptive roles (history, gate, repeat).
`--seed 42` and `--episodes 64` are explicit reproducibility settings. Episode
count must be even and between 2 and 128.

## Use your own controller

```python
from substratebench import EncodedMemory, evaluate, write_result

def my_source_step(current_bits, command):
    # A deliberately incapable custom source, so this example is runnable.
    return (0, 0, 0, 0)

result = evaluate(
    my_source_step,
    EncodedMemory(),
    seed=71,
    controller_label="custom-zero-source",
)
write_result(result, "results/custom-source.json")
```

The complete runnable example is [examples/run_custom.py](examples/run_custom.py).
It reports a failed quality verdict. Replace the callback with your own model
wrapper and pass `exclude_ids=training_sequence_ids` when applicable. Model and
training identities remain owned by the caller. The callback must return four
integer bits, keep its weights
frozen, and retain no task history outside the supplied memory. It receives only
the currently read bits and `Command(register, gate, value)`. Python does not
enforce callback purity or inspect model parameters. A passing result is therefore
a benchmark result under that caller contract, not an independent audit of model
freezing or absence of a hidden sidecar.

The memory integration boundary is equally small: `reset()`, `read()` returning
four logical bits, and `set_bit(register, value)`. `BinaryMemory` stores bits
directly. `EncodedMemory` stores a channel permutation and inversion, with an
explicit known codec and no second copy of logical state. Both support exact
digital reset; this release makes no claim about physical or unknown substrates.

## What is measured

Before task execution, the runner checks all 16 readable states and all 128
generic SET operations (16 initial states × four registers × two values).
A SET must preserve the other three bits. Failure is reported as
`interface_not_qualified`; task routes remain unattempted, with their costs zero.
This is qualification in a known finite domain, not generalization.

The default gated-XOR task changes the selected register only when both command bits are 1.
A descriptive source-law check evaluates all 16 states × 16 commands. The runner
then executes each sequence freely, reading from memory on every step and writing
the proposed state through four fixed SET calls. Gold labels stay in the evaluator
and never enter the controller callback.

Four paired routes measure normal operation, register-permuted controller inputs
against the corresponding transformed task, disabled writing, and a reset before
every command. Primary random-role thresholds are 99% prefix-state accuracy,
95% whole-sequence accuracy for normal and permuted tasks, and a drop of at least
30 percentage points in whole-sequence accuracy for each memory ablation, at
every random length. History/gate/repeat roles are descriptive. These controls
test dependence on visible memory operations; they cannot prove callback purity
or the causal contribution of trained weights.

Select the task with `--task` or `evaluate(..., task="carry_counter")`. The
default remains `gated_xor`. All laws use `Command(register, gate, value)`;
`gate=0` preserves the complete state. Register0 is the least significant bit.

| Task | Active command (`gate=1`) |
| --- | --- |
| `gated_xor` | Flip the selected bit when `value=1`. |
| `carry_counter` | Add `2**register` when `value=1`, subtract it when `value=0`, modulo 16; carries and borrows may change other bits. |
| `conditional_logic` | With `value=0`, swap the selected bit and its next cyclic neighbor. With `value=1`, flip the selected bit and also its opposite bit if the old next-neighbor bit is 1. Updates are simultaneous. |

For a reference callback in the API, bind the same task explicitly:

```python
from functools import partial
from substratebench import BinaryMemory, evaluate, reference_step

result = evaluate(partial(reference_step, task="carry_counter"),
                  BinaryMemory(), task="carry_counter", episodes=4)
```

Custom callbacks receive the same two arguments for every task; the evaluator
does not tell the callback the Gold state. `diagnostic-history` pairs change the
first command's value. The legacy `diagnostic-gate` pairs change the last
command's value while keeping gate=1 in both members; this is a payload contrast
(subtract/add for the counter), not a gate-on/off causal test. Role names and
command construction remain unchanged. Descriptive pairs do not guarantee
different final states for every law.

Reports separate completion status from the pass/fail verdict and contain role
metrics, raw outputs, sequence identities, qualification rows, implementation
hashes, and costs per phase. One task-memory write is four SET calls. Visible
reads, resets, SET calls, controller calls, and reference-label steps are counted;
wall time covers `evaluate()` including case generation. Import, installation,
report writing, external training, energy, and hardware interaction costs are not
included. External training cost is explicitly unknown. Do not compare this
runner's wall time with a model's full training/deployment cost.
Custom memory implementations may do additional internal integration or wait
steps; this runner counts the visible interface only. Full physical/training cost
requires separate instrumentation supplied by the integrator.

## Reproducibility and scope

The default profile remains `gated-xor-memory-reference-0.1`; new task profiles
are `carry-counter-memory-reference-0.2` and
`conditional-logic-memory-reference-0.2`. Reports include `task_id` as an additive
schema-version-1 field and package version 0.2.0. The sampler uses Python's seeded
MT19937 `getrandbits` with explicit rejection sampling. Sequence IDs are SHA256
of ordered uint8 six-byte commands (four one-hot register bits, gate, value),
unchanged from public version 0.1.0. Roles and task names do not affect IDs.
The suite rejects duplicates and any supplied `exclude_ids` across both original
and register-permuted commands; it cannot verify that
the supplied training-ID list is complete. New random seeds do not establish
independent holdout substrate families. Permuted queries are derived paired cases.

Historical Lab command arrays order the last two features as **value, gate**;
the public codec orders them as **gate, value**. The former claim of direct
historical compatibility was incorrect for asymmetric commands. To exclude Lab
training episodes, regenerate public IDs from their original ordered commands:

```python
from substratebench import Command, sequence_id

lab_episode_rows = [(0, 0, 1, 0, 1, 0), (1, 0, 0, 0, 0, 1)]
commands = [Command(list(row[:4]).index(1), gate=int(row[5]), value=int(row[4]))
            for row in lab_episode_rows]
public_id = sequence_id(commands)
```

Pass those regenerated IDs through `exclude_ids`. A digest-only Lab ID list
cannot be converted without the original command arrays.

This release replaces **memory storage**, while the controller's computation stays
in Python. It does not implement the broader proposed black-box dynamical-slot
benchmark, learned adapters, held-out substrate families, a cost advantage from
pretraining, or a leaderboard. The research vision is recorded in the lab without
presenting planned features as implemented.

Licensing terms have not yet been selected. No redistribution license is granted
by this initial repository import.

## Development

```sh
python -m pip install -e .
python -m unittest discover -s tests -v
```

The tests exercise the complete finite task law, generic memory preservation,
paired sequence construction, actual free rollouts and ablations, rejection of
malformed controller output, censored interface failure, and competing atomic
artifact writers. CI installs the package before running these checks.
