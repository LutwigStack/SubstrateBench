"""CLI for the public reference harness, not the historical learned-source run."""

import argparse
import os
import sys
from functools import partial

from .benchmark import evaluate, write_result
from .memory import BinaryMemory, EncodedMemory
from .task import TASKS, reference_step, zero_step


def main(argv=None):
    parser = argparse.ArgumentParser(description="Known digital memory reference benchmark")
    parser.add_argument("--output", required=True, help="New JSON artifact; existing files are never replaced")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--episodes", type=int, default=64, help="Even episode count per role, 2 to 128")
    parser.add_argument("--memory", choices=("binary", "encoded"), default="encoded")
    parser.add_argument("--controller", choices=("oracle", "zero"), default="oracle")
    parser.add_argument("--task", choices=(*TASKS, "carry_counter_8"), default="gated_xor")
    args = parser.parse_args(argv)
    if args.task == "carry_counter_8":
        from . import carry8
        memory = carry8.BinaryMemory() if args.memory == "binary" else carry8.EncodedMemory()
        controller = carry8.reference_step if args.controller == "oracle" else carry8.zero_step
    else:
        memory = BinaryMemory() if args.memory == "binary" else EncodedMemory()
        controller = partial(reference_step, task=args.task) if args.controller == "oracle" else zero_step
    label = "reference_oracle" if args.controller == "oracle" else "zero_source_control"
    try:
        if os.path.lexists(args.output):
            raise FileExistsError(f"Artifact already exists: {args.output}")
        if args.task == "carry_counter_8":
            result = carry8.evaluate(controller, memory, seed=args.seed, episodes=args.episodes,
                                     controller_label=label)
        else:
            result = evaluate(controller, memory, seed=args.seed, episodes=args.episodes,
                              controller_label=label, task=args.task)
        write_result(result, args.output)
    except (ValueError, OSError) as error:
        print(f"substratebench: {error}", file=sys.stderr)
        return 2
    print(f"{result['verdict'].upper()} — {args.task}, {label}, {result['substrate']}; artifact: {args.output}")
    print("Reference harness: this run does not demonstrate learned-model transfer.")
    return 0 if result["verdict"] == "pass" else 1
