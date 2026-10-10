"""Reuse one fixed digital memory interface across two reference task laws."""

import argparse
import os
from pathlib import Path

from substratebench import carry8, conditional8, write_result


def main(argv=None):
    parser = argparse.ArgumentParser(description="Same-memory reference interface reuse")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--episodes", type=int, default=64)
    args = parser.parse_args(argv)
    paths = [Path(args.output_dir) / f"{task}.json"
             for task in ("carry_counter_8", "conditional_logic_8")]
    for path in paths:
        if os.path.lexists(path):
            raise FileExistsError(f"Artifact already exists: {path}")
    memory = carry8.EncodedMemory()
    carry = carry8.evaluate(carry8.reference_step, memory, seed=args.seed,
                            episodes=args.episodes, controller_label="reference_oracle")
    # Lifecycle reset is outside both evaluation ledgers; the codec stays fixed.
    memory.reset()
    conditional = conditional8.evaluate(conditional8.reference_step, memory, seed=args.seed,
                                        episodes=args.episodes, controller_label="reference_oracle")
    for result, path in zip((carry, conditional), paths):
        write_result(result, path)
        print(f"{result['task_id']}: {result['verdict']}; artifact: {path}")
    print("One fixed memory object reused; reference interface checks, no learned-transfer evidence.")
    return 0 if carry["verdict"] == conditional["verdict"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
