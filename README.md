# SubstrateBench

**STATUS: implemented — public reference benchmark, version 0.1.0.**

SubstrateBench measures whether a fixed task controller can keep using a four-bit
memory after replacing its storage implementation. This first release is a small,
reproducible digital benchmark that other people can install and extend. It has
no runtime dependencies.

This initial promotion is scoped in [SB-001](docs/promotion/SB-001.md);
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
```

The first two commands run the reference controller; the third intentionally
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

The task changes the selected register only when both command bits are 1.
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

The profile is `gated-xor-memory-reference-0.1`; the sampler uses Python's seeded
MT19937 `getrandbits` with explicit rejection sampling. Sequence IDs are SHA256
of ordered uint8 six-byte commands (four one-hot register bits, gate, value),
compatible with the historical lab task identity codec. Roles do not affect IDs.
The suite rejects duplicates and any supplied `exclude_ids` across both original
and register-permuted commands; it cannot verify that
the supplied training-ID list is complete. New random seeds do not establish
independent holdout substrate families. Permuted queries are derived paired cases.

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
