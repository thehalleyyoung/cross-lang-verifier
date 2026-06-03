#!/usr/bin/env python3
"""Cold-start CI gate for 100_STEPS Step 150.

The gate is intentionally two-phase:

1. run the full pytest suite with external compiler discovery masked, so only
   toolchain-independent tests execute regardless of what happens to be on PATH;
2. unmask toolchains and confirm one representative C->Rust signed-overflow
   witness against real clang/UBSan and rustc.

The total wall clock is budgeted (default: 300 seconds) and each phase reports a
machine-readable timing summary for CI logs.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import List, Mapping, Optional


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ub_oracle.reexec import NO_TOOLCHAIN_ENV  # noqa: E402

DEFAULT_BUDGET_SECONDS = 300.0
DEFAULT_SAMPLE_RESERVE_SECONDS = 20.0
DEFAULT_SAMPLE_TIMEOUT_SECONDS = 20


@dataclass
class PhaseResult:
    name: str
    elapsed_seconds: float
    returncode: int
    detail: str


def _elapsed(start: float) -> float:
    return round(time.monotonic() - start, 3)


def _tail(text: str, *, limit: int = 4000) -> str:
    return text if len(text) <= limit else text[-limit:]


def _xdist_available() -> bool:
    return importlib.util.find_spec("xdist") is not None


def pytest_command(*, parallel: bool = True, extra_args: Optional[List[str]] = None) -> List[str]:
    cmd = [
        sys.executable,
        "-m",
        "pytest",
        str(ROOT / "tests"),
        "-q",
        "-p",
        "no:cacheprovider",
    ]
    if parallel and _xdist_available():
        cmd.extend(["-n", "auto"])
    if extra_args:
        cmd.extend(extra_args)
    return cmd


def noncompiler_env(base: Optional[Mapping[str, str]] = None) -> dict:
    env = dict(os.environ if base is None else base)
    env[NO_TOOLCHAIN_ENV] = "1"
    return env


def run_noncompiler_pytest(
    *,
    timeout_seconds: float,
    parallel: bool = True,
    extra_args: Optional[List[str]] = None,
    runner=subprocess.run,
) -> PhaseResult:
    cmd = pytest_command(parallel=parallel, extra_args=extra_args)
    started = time.monotonic()
    try:
        proc = runner(
            cmd,
            cwd=str(ROOT),
            env=noncompiler_env(),
            capture_output=True,
            text=True,
            timeout=max(1, int(timeout_seconds)),
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        stdout = exc.stdout or ""
        stderr = exc.stderr or ""
        return PhaseResult(
            "noncompiler_pytest",
            _elapsed(started),
            124,
            "pytest timed out\n" + _tail(str(stdout) + "\n" + str(stderr)),
        )

    detail = _tail((proc.stdout or "") + ("\n" + proc.stderr if proc.stderr else ""))
    return PhaseResult("noncompiler_pytest", _elapsed(started), int(proc.returncode), detail)


def run_compiled_sample(*, timeout_seconds: int = DEFAULT_SAMPLE_TIMEOUT_SECONDS) -> PhaseResult:
    started = time.monotonic()

    from ub_oracle import ReexecHarness, get_oracle  # noqa: WPS433
    from ub_oracle import oracles as _oracles  # noqa: F401,WPS433
    from ub_oracle.reexec import toolchain_available  # noqa: WPS433

    previous_mask = os.environ.pop(NO_TOOLCHAIN_ENV, None)
    try:
        status = toolchain_available()
    finally:
        if previous_mask is not None:
            os.environ[NO_TOOLCHAIN_ENV] = previous_mask

    if not status.full_for("rust"):
        return PhaseResult(
            "compiled_sample",
            _elapsed(started),
            2,
            f"needs C+UBSan+rustc toolchain; discovered {status}",
        )

    oracle = get_oracle("signed_overflow")
    unit = {
        "name": "cold-start-signed-overflow",
        "kind": "binop_const",
        "op": "add",
        "const": 1,
        "width": 32,
        "var": "x",
        "signed": True,
        "probe": "signed_overflow",
        "source_lang": "c",
        "target_lang": "rust",
    }
    found = oracle.find_divergence(unit)
    confirmed = oracle.confirm(found, ReexecHarness(status, timeout=timeout_seconds))
    rr = confirmed.reexec
    if rr is None or not rr.available or not rr.confirmed:
        return PhaseResult(
            "compiled_sample",
            _elapsed(started),
            1,
            rr.summary() if rr is not None else "oracle did not produce re-exec evidence",
        )

    ce = confirmed.counterexample
    detail = json.dumps(
        {
            "divergence_class": confirmed.divergence_class,
            "witness": ce.inputs if ce is not None else {},
            "ub_reachable": rr.ub_reachable,
            "ub_consequential": rr.ub_consequential,
            "rust_defined": rr.rust_defined,
        },
        sort_keys=True,
    )
    return PhaseResult("compiled_sample", _elapsed(started), 0, detail)


def run_gate(args: argparse.Namespace) -> int:
    budget = float(args.budget_seconds)
    started = time.monotonic()
    phases: List[PhaseResult] = []

    pytest_timeout = max(1.0, budget - float(args.sample_reserve_seconds))
    if not args.skip_pytest:
        phase = run_noncompiler_pytest(
            timeout_seconds=pytest_timeout,
            parallel=not args.no_parallel,
            extra_args=args.pytest_arg,
        )
        phases.append(phase)
        if phase.returncode != 0:
            print(json.dumps({"ok": False, "phases": [asdict(p) for p in phases]}, indent=2))
            return phase.returncode

    elapsed = time.monotonic() - started
    remaining = budget - elapsed
    if remaining <= 0:
        phases.append(PhaseResult("budget", round(elapsed, 3), 124, "budget exhausted before compiled sample"))
        print(json.dumps({"ok": False, "budget_seconds": budget, "phases": [asdict(p) for p in phases]}, indent=2))
        return 124

    if not args.skip_sample:
        phase = run_compiled_sample(timeout_seconds=min(args.sample_timeout_seconds, max(1, int(remaining))))
        phases.append(phase)
        if phase.returncode != 0:
            print(json.dumps({"ok": False, "phases": [asdict(p) for p in phases]}, indent=2))
            return phase.returncode

    total = round(time.monotonic() - started, 3)
    ok = total <= budget
    summary = {
        "ok": ok,
        "budget_seconds": budget,
        "total_seconds": total,
        "phases": [asdict(p) for p in phases],
    }
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if ok else 124


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--budget-seconds",
        type=float,
        default=float(os.environ.get("COLD_START_BUDGET_S", DEFAULT_BUDGET_SECONDS)),
        help="total wall-clock budget for pytest + representative compiled sample",
    )
    parser.add_argument(
        "--sample-reserve-seconds",
        type=float,
        default=DEFAULT_SAMPLE_RESERVE_SECONDS,
        help="minimum time reserved for the compiled sample before starting pytest",
    )
    parser.add_argument(
        "--sample-timeout-seconds",
        type=int,
        default=DEFAULT_SAMPLE_TIMEOUT_SECONDS,
        help="per-subprocess timeout used by the representative compiled sample",
    )
    parser.add_argument("--no-parallel", action="store_true", help="disable pytest-xdist even when installed")
    parser.add_argument("--skip-pytest", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--skip-sample", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument(
        "--pytest-arg",
        action="append",
        default=[],
        help="extra argument to append to the pytest command (repeatable)",
    )
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    return run_gate(parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
