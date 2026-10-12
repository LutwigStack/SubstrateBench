"""Optional NumPy recipe: newly generated, fully journaled scalar reference run.

Run with ``python -m substratebench.public_memory8_demo --output NEW_DIRECTORY``.
This module is a concrete known-family example, not a learned model or transport.
"""

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import struct
import sys
import time

from . import carry8, conditional8
from .benchmark import write_result
from .public_memory8 import ALGORITHM, PublicMemory8

SEEDS = (8205005, 8205006)
TRACE_MAGIC = b"SB005P1\n"
MACRO_MAGIC = b"SB005M1\n"
HEADER = struct.Struct("<BQBBH")  # kind, request ID, operation, lane, batch
MACRO = struct.Struct("<QBBBQQQQQQB")
OPS = {"reset": 0, "step": 1, "observe": 2}
PRIMITIVE_CAP = 1536 * 1024 * 1024
MACRO_CAP = 128 * 1024 * 1024
MAX_FRAME = 16 * 1024


class FitNonattainment(ValueError):
    """Fixed public observations completed but identification gates did not pass."""

    def __init__(self, message, lane, gates):
        super().__init__(message)
        self.diagnostic = {"lane": lane, "gates": [bool(value) for value in gates]}


def _digest(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return {"path": Path(path).name, "bytes": Path(path).stat().st_size,
            "sha256": digest.hexdigest()}


class Journal:
    def __init__(self, path, magic, limit):
        self.stream = Path(path).open("xb")
        self.limit, self.bytes, self.records = limit, 0, 0
        self._write(magic)

    def _write(self, value):
        if self.bytes + len(value) > self.limit:
            raise RuntimeError("Journal byte quota exhausted")
        written = self.stream.write(value)
        self.bytes += written
        if written != len(value):
            raise OSError("Partial journal append")

    def append(self, value):
        if not 0 < len(value) <= MAX_FRAME:
            raise ValueError("Journal frame exceeds its fixed bound")
        self._write(struct.pack("<I", len(value)) + value)
        self.records += 1

    def close(self):
        if not self.stream.closed:
            try:
                self.stream.flush()
                os.fsync(self.stream.fileno())
            finally:
                self.stream.close()


class _Bank:
    """Private dynamics owner; only Port exposes public observations."""

    def __init__(self, np, seed):
        rng = np.random.Generator(np.random.PCG64(seed))
        self.alpha = rng.uniform(1.8, 2.2, 4)
        self.drive = 3.5 * np.linalg.qr(rng.normal(size=(16, 4)))[0].T
        self.observation = 1.5 * np.linalg.qr(rng.normal(size=(32, 4)))[0].T
        for value in (self.alpha, self.drive, self.observation):
            value.flags.writeable = False
        self.position, self.np = None, np

    def tick(self, stimulus):
        forcing = self.np.clip(stimulus @ self.drive.T, -4.0, 4.0)
        self.position = self.np.tanh(self.alpha * self.position + forcing)


class Port:
    """Bounded concrete in-process public port and authentic primitive journal."""

    def __init__(self, np, banks, journal, deadline):
        self.np, self._banks, self.journal, self.deadline = np, banks, journal, deadline
        self.request_id = self.completed = self.microsteps = 0
        self.broken = False

    def _call(self, operation, lane, argument=None):
        if self.broken or time.monotonic() >= self.deadline:
            raise TimeoutError("Public port is broken or expired")
        if type(lane) is not int or lane not in (0, 1):
            raise ValueError("Expected public lane 0 or 1")
        bank, np = self._banks[lane], self.np
        if operation == "reset":
            if type(argument) is not int or not 1 <= argument <= 32:
                raise ValueError("Reset batch must be an integer in [1,32]")
            batch, stimulus = argument, None
        else:
            if bank.position is None:
                raise RuntimeError("Public lane requires reset")
            batch, stimulus = len(bank.position), None
            if operation == "step":
                stimulus = np.asarray(argument, dtype=np.float64)
                if (stimulus.shape != (batch, 16) or not np.isfinite(stimulus).all()
                        or np.max(abs(stimulus)) > 1):
                    raise ValueError("Invalid public stimulus")
        self.request_id += 1
        head = (self.request_id, OPS[operation], lane, batch)
        try:
            request = HEADER.pack(0, *head)
            if stimulus is not None:
                request += stimulus.astype("<f8", copy=False).tobytes()
            self.journal.append(request)
            if operation == "reset":
                bank.position = np.full((batch, 4), -1.0, np.float64)
                for _ in range(8):
                    bank.tick(np.zeros((batch, 16), np.float64))
                self.microsteps += batch * 8
            elif operation == "step":
                bank.tick(stimulus)
                self.microsteps += batch
            y = bank.position @ bank.observation if operation == "observe" else None
            if (not np.isfinite(bank.position).all()
                    or y is not None and not np.isfinite(y).all()):
                raise FloatingPointError("Nonfinite simulated dynamics")
            response = HEADER.pack(1, *head)
            if y is not None:
                response += y.astype("<f8", copy=False).tobytes()
            response += bank.position.astype("<f8", copy=False).tobytes()
            self.journal.append(response)
            self.completed += 1
            return None if y is None else y.tolist()
        except BaseException:
            self.broken = True
            raise

    def reset(self, lane, batch):
        return self._call("reset", lane, batch)

    def step(self, lane, stimulus):
        return self._call("step", lane, stimulus)

    def observe(self, lane):
        return self._call("observe", lane)


def fit_public(port, np):
    """One fixed public-only probe per bank; no labels, geometry or fallback."""
    readers, pulses, diagnostics = [], [], []
    epsilon = 1e-4
    stimulus = epsilon * np.concatenate((np.eye(16), -np.eye(16)))
    for lane in (0, 1):
        port.reset(lane, 32)
        y0 = np.asarray(port.observe(lane), dtype=np.float64)
        port.step(lane, stimulus)
        y1 = np.asarray(port.observe(lane), dtype=np.float64)
        port.step(lane, np.zeros((32, 16)))
        y2 = np.asarray(port.observe(lane), dtype=np.float64)
        reset = y0.mean(axis=0)
        g0 = (y1[:16] - y1[16:]) / (2 * epsilon)
        g1 = (y2[:16] - y2[16:]) / (2 * epsilon)
        _, singular, vt = np.linalg.svd(g0, full_matrices=False)
        v = vt[:4].T
        r0, r1 = g0 @ v, g1 @ v
        m, _, rank, _ = np.linalg.lstsq(r0, r1, rcond=None)
        eigen, e = np.linalg.eig(m)
        order = np.argsort(eigen.real)
        eigen, e = eigen[order], e[:, order]
        imaginary = max(float(np.max(abs(eigen.imag))), float(np.max(abs(e.imag))))
        eigen, e = eigen.real, e.real
        p = v @ e
        projection = reset @ p
        response_residual = np.linalg.norm(r1 - r0 @ m) / max(np.linalg.norm(r1), 1e-12)
        gates = [np.max(abs(y0 - reset)) <= 1e-9,
                 singular[3] >= 1e-5, singular[4] <= 1e-8,
                 singular[4] / singular[3] <= 1e-6, rank == 4,
                 np.linalg.cond(r0) <= 1e5, response_residual <= 1e-5,
                 imaginary <= 1e-9, np.all((eigen > 0) & (eigen < 1)),
                 np.min(np.diff(eigen)) >= 1e-5, np.linalg.cond(e) <= 1e5,
                 np.linalg.norm(m @ e - e * eigen) / max(np.linalg.norm(m) * np.linalg.norm(e), 1e-12) <= 1e-9,
                 np.all(abs(projection) >= 1e-5)]
        if not all(gates):
            raise FitNonattainment("Fixed public identification gates failed", lane, gates)
        p *= (-np.sign(projection) / abs(projection))[None]
        h = g0 @ p
        inverse = np.linalg.pinv(h)
        h_singular = np.linalg.svd(h, compute_uv=False)
        drive_gates = [h_singular[3] >= 1e-5, np.linalg.cond(h) <= 1e5,
                       np.linalg.norm(inverse @ h - np.eye(4)) <= 1e-8]
        if not all(drive_gates):
            raise FitNonattainment("Fixed public drive gates failed", lane, drive_gates)
        axis = inverse / np.max(abs(inverse), axis=1)[:, None]
        readers.append(np.vstack((p, np.zeros((1, 4)))).tolist())
        pulses.append(np.stack((-axis, axis), axis=1).reshape(8, 16).tolist())
        diagnostics.append({"lane": lane, "eigenvalues": eigen.tolist(),
                            "response_residual": float(response_residual),
                            "public_calls": 6, "numerical_solves": 8})
    return {"schema_version": 1, "family": "discrete_tanh", "algorithm": ALGORITHM,
            "reader": readers, "pulses": pulses}, diagnostics


class _TracedMemory:
    """Example-only audit consumer; the supplied adapter owns all operations."""

    def __init__(self, memory, port, journal):
        self.memory, self.port, self.journal = memory, port, journal

    def _call(self, operation, *args):
        first = self.port.request_id + 1
        previous = self.memory.counts()["operations"]
        try:
            return getattr(self.memory, operation)(*args)
        finally:
            row = self.memory.last_operation
            if row is not None and row["macro_id"] > previous:
                self.journal.append(MACRO.pack(
                    row["macro_id"], {"reset": 0, "read": 1, "set_bit": 2}[operation],
                    255 if row["register"] is None else row["register"],
                    255 if row["value"] is None else row["value"],
                    first, self.port.request_id, row["attempted_primitive_calls"],
                    row["completed_primitive_calls"], row["common_ticks_before"],
                    2**64 - 1 if row["common_ticks_after"] is None else row["common_ticks_after"],
                    int(row["outcome"] == "completed")))

    def reset(self):
        return self._call("reset")

    def read(self):
        return self._call("read")

    def set_bit(self, register, value):
        return self._call("set_bit", register, value)


def _retention(memory, port):
    memory.reset()
    expected, rows = [0] * 8, []
    for tick in range(32):
        register, value = tick * 3 % 8, tick // 8 % 2
        memory.set_bit(register, value)
        expected[register] = value
        observed = memory.read()
        rows.append({"stage": "SET", "tick": tick, "requested": list(expected),
                     "observed": observed, "correct": observed == tuple(expected)})
    idle_ranges = []
    for tick in range(64):
        first = port.request_id + 1
        port.step(0, ((0.0,) * 16,))
        port.step(1, ((0.0,) * 16,))
        idle_ranges.append([first, port.request_id])
        observed = memory.read()
        rows.append({"stage": "idle", "tick": tick, "requested": list(expected),
                     "observed": observed, "correct": observed == tuple(expected)})
    return {"scope": "separate 32 SET / 64 idle diagnostic; not a stock timed control",
            "correct": sum(row["correct"] for row in rows), "total": 96,
            "qualified": all(row["correct"] for row in rows),
            "idle_request_ranges": idle_ranges, "rows": rows}


def run(output, *, episodes=2, seconds=1800):
    """Generate all assets locally and run the unchanged two reference profiles."""
    enclosing_started, enclosing_cpu = time.monotonic(), time.process_time()
    if type(episodes) is not int or not 2 <= episodes <= 128 or episodes % 2:
        raise ValueError("Episodes must be an even integer in [2,128]")
    if type(seconds) not in (int, float) or not 1 <= seconds <= 7200 or not math.isfinite(seconds):
        raise ValueError("Seconds must be finite in [1,7200]")
    # ponytail: fixed journals admit E2/E4; larger profiles need a new quota review.
    primitive_bound = (2 * 1_230_336 + 248_576 * episodes + 1108) * 450 + 120_000
    if primitive_bound > PRIMITIVE_CAP:
        raise ValueError("This episode count exceeds the fixed complete-trace quota")
    try:
        import numpy as np
    except ImportError:
        raise ValueError("The optional demo requires NumPy; install substratebench[demo]") from None
    if np.__version__ != "2.2.6":
        raise ValueError("The reproducible demo requires the pinned NumPy 2.2.6 extra")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    started, cpu_started = time.monotonic(), time.process_time()
    primitive = macro = port = None
    receipt = {"schema_version": 1, "status": "interrupted", "seed": 42,
               "episodes": episodes, "lengths": [24, 128], "body_seeds": list(SEEDS),
               "numpy_version": np.__version__, "python_version": sys.version.split()[0],
               "deadline_seconds": seconds, "energy": "UNKNOWN_NOT_METERED",
               "primitive_byte_upper_bound": primitive_bound,
               "scope": "new noiseless discrete_tanh scalar simulated-memory / CPU reference oracle",
               "implementation": {name: _digest(Path(__file__).parent / name)
                                  for name in ("public_memory8.py", "public_memory8_demo.py",
                                               "carry8.py", "conditional8.py", "benchmark.py", "task.py")}}
    try:
        primitive = Journal(output / "primitives.bin", TRACE_MAGIC, PRIMITIVE_CAP)
        macro = Journal(output / "macros.bin", MACRO_MAGIC, MACRO_CAP)
        banks = [_Bank(np, seed) for seed in SEEDS]
        write_result({"schema_version": 1, "family": "discrete_tanh",
                      "reset_ticks": 8, "seeds": list(SEEDS),
                      "banks": [{name: getattr(bank, name).tolist()
                                 for name in ("alpha", "drive", "observation")} for bank in banks]},
                     output / "geometry-private.json")
        port = Port(np, banks, primitive, started + seconds)
        coefficients, fit = fit_public(port, np)
        write_result(coefficients, output / "coefficients.json")
        coefficient_file = _digest(output / "coefficients.json")
        candidate = PublicMemory8.from_json(port, output / "coefficients.json",
                                          sha256=coefficient_file["sha256"])
        memory = _TracedMemory(candidate, port, macro)
        receipt.update(fit=fit, fit_last_request_id=port.request_id,
                       coefficients=coefficient_file,
                       coefficient_sha256=candidate.coefficient_sha256,
                       frozen_before_qualification=True)
        for name, profile in (("carry", carry8), ("conditional", conditional8)):
            first = port.request_id + 1
            result = profile.evaluate(profile.reference_step, memory, seed=42,
                                      episodes=episodes, controller_label="reference_oracle")
            result["scope"] = receipt["scope"]
            result["substrate"] = "PublicMemory8"
            result["public_memory8"] = {
                "first_request_id": first, "last_request_id": port.request_id,
                "coefficient_artifact_sha256": coefficient_file["sha256"],
                "adapter_sha256": receipt["implementation"]["public_memory8.py"]["sha256"],
                "physical_qualification": "pending independent complete trace reconstruction",
                "source_kind": "reference_oracle", "learned_transfer": False}
            write_result(result, output / (name + ".json"))
            receipt[name] = {"verdict": result["verdict"], "first_request_id": first,
                             "last_request_id": port.request_id}
        first = port.request_id + 1
        retention = _retention(memory, port)
        write_result(retention, output / "retention.json")
        receipt.update(retention={"qualified": retention["qualified"],
                                  "first_request_id": first, "last_request_id": port.request_id},
                       memory_counts=candidate.counts(), status="completed")
    except BaseException as error:
        receipt.update(error_type=type(error).__name__, error=str(error))
        if isinstance(error, FitNonattainment):
            receipt.update(status="scientific_nonattainment", failed_stage="public_identification",
                           fit_failure=error.diagnostic, conformance_status="not_attempted")
        else:
            receipt["unfinished_cost_tail"] = "UNKNOWN"
        raise
    finally:
        cleanup_error = None
        for journal in (primitive, macro):
            if journal is not None:
                try:
                    journal.close()
                except BaseException as error:
                    cleanup_error = error
                    receipt.setdefault("cleanup_errors", []).append(type(error).__name__)
                    receipt.update(status="interrupted", unfinished_cost_tail="UNKNOWN")
        receipt.update(request_attempts=None if port is None else port.request_id,
                       request_completions=None if port is None else port.completed,
                       lane_row_microsteps=None if port is None else port.microsteps,
                       runner_wall_seconds=time.monotonic() - started,
                       runner_cpu_seconds=time.process_time() - cpu_started,
                       timing_scope="run after NumPy import through journal flush; excludes final receipt/hash I/O",
                       trace_bytes=None if primitive is None else primitive.bytes,
                       macro_bytes=None if macro is None else macro.bytes)
        write_result(receipt, output / "receipt.json")
        write_result({"schema_version": 1,
                      "enclosing_wall_seconds": time.monotonic() - enclosing_started,
                      "enclosing_cpu_seconds": time.process_time() - enclosing_cpu,
                      "scope": "run including NumPy import and receipt I/O; excludes interpreter/module imports, this meter and manifest final writes",
                      "energy": "UNKNOWN_NOT_METERED"}, output / "outer-meter.json")
        files = [_digest(path) for path in sorted(output.iterdir()) if path.is_file()]
        write_result({"schema_version": 1, "availability": "locally generated; no historical assets required",
                      "files": files}, output / "manifest.json")
        if cleanup_error is not None:
            raise cleanup_error
    return receipt


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path, help="New external output directory")
    parser.add_argument("--episodes", type=int, default=2)
    parser.add_argument("--seconds", type=float, default=1800)
    args = parser.parse_args(argv)
    try:
        result = run(args.output, episodes=args.episodes, seconds=args.seconds)
    except FitNonattainment as error:
        print(f"public_memory8_demo: scientific nonattainment: {error}", file=sys.stderr)
        return 1
    except (ValueError, OSError, RuntimeError, TimeoutError) as error:
        print(f"public_memory8_demo: {error}", file=sys.stderr)
        return 2
    print(f"Generated {result['request_completions']} public primitive calls at {args.output}")
    print("CPU reference oracles; physical claims require an independent full trace audit.")
    return 0 if all(result[name]["verdict"] == "pass" for name in ("carry", "conditional")) and result["retention"]["qualified"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
