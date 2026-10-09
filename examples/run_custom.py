"""Run after installing the package, from any working directory."""

import argparse

from substratebench import EncodedMemory, evaluate, write_result


def custom_source(current_bits, command):
    return (0, 0, 0, 0)


def main():
    parser = argparse.ArgumentParser(description="A deliberately incapable user callback")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    result = evaluate(custom_source, EncodedMemory(), seed=71,
                      controller_label="custom-zero-source")
    write_result(result, args.output)
    print(f"Custom source verdict: {result['verdict']} (expected: fail)")


if __name__ == "__main__":
    main()
