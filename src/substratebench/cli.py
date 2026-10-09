"""CLI for the public reference harness, not the historical learned-source run."""

import argparse
import os
import sys

from .benchmark import evaluate, write_result
from .memory import BinaryMemory, EncodedMemory
from .task import reference_step, zero_step


def main(argv=None):
    parser = argparse.ArgumentParser(description="Known digital four-bit memory reference benchmark")
    parser.add_argument("--output", required=True, help="New JSON artifact; existing files are never replaced")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--episodes", type=int, default=64, help="Even episode count per role, 2 to 128")
    parser.add_argument("--memory", choices=("binary", "encoded"), default="encoded")
    parser.add_argument("--controller", choices=("oracle", "zero"), default="oracle")
    args = parser.parse_args(argv)
    memory = BinaryMemory() if args.memory == "binary" else EncodedMemory()
    controller = reference_step if args.controller == "oracle" else zero_step
    label = "reference_oracle" if args.controller == "oracle" else "zero_source_control"
    try:
        if os.path.lexists(args.output):
            raise FileExistsError(f"Artifact already exists: {args.output}")
        result = evaluate(controller, memory, seed=args.seed, episodes=args.episodes,
                          controller_label=label)
        write_result(result, args.output)
    except (ValueError, OSError) as error:
        print(f"substratebench: {error}", file=sys.stderr)
        return 2
    print(f"{result['verdict'].upper()} — {label}, {result['substrate']}; artifact: {args.output}")
    print("Reference harness: this run does not demonstrate learned-model transfer.")
    return 0 if result["verdict"] == "pass" else 1
