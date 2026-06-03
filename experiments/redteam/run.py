#!/usr/bin/env python3
"""
Internal red-team driver (100_STEPS step 84).

``python -m experiments.redteam.run`` regenerates the *deterministic* adversarial
grid ``experiments/redteam/cases.json`` — every semantics-preserving divergent
mutation, across every oracle and language pair, with the **symbolic** verdict
each receives when no toolchain is present (always ``candidate`` — a sound,
no-equivalence-claim verdict).  Because this path never shells out to a compiler,
the grid is byte-reproducible and ``make redteam-check`` asserts it.

With ``--attack`` it additionally runs the *real* adversary: it confirms every
case against the actual ``clang``+UBSan + target compilers and asserts that **no**
genuinely-divergent case is ever reported ``NO_DIVERGENCE_FOUND`` (a soundness
breach).  The environment-dependent verdicts are written to ``attack.json``
(gitignored).  A non-empty breach set exits non-zero.

With ``--table`` it prints the human-readable adversarial report.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(_HERE))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from src.ub_oracle import oracles  # noqa: F401  (registers all pairs)
from src.ub_oracle import redteam as RT
from src.ub_oracle.reexec import ReexecHarness, ToolchainStatus, toolchain_available
from src.ub_oracle.cache import ToolchainMismatch, validate_toolchain_file

CASES_PATH = os.path.join(_HERE, "cases.json")
ATTACK_PATH = os.path.join(_HERE, "attack.json")

#: a deliberately empty toolchain so the deterministic grid never compiles.
_NO_TOOLCHAIN = ToolchainStatus(cc=None, ubsan=False, targets=())


def _write_json(path: str, obj) -> None:
    with open(path, "w") as f:
        f.write(json.dumps(obj, indent=2, sort_keys=True))
        f.write("\n")


def _deterministic_grid() -> dict:
    # symbolic-only run (no toolchain): every divergent case is a sound CANDIDATE.
    report = RT.run_redteam(status=_NO_TOOLCHAIN, confirm=False)
    return report.to_dict()


def _render(report: RT.RedTeamReport) -> str:
    lines = [f"red-team: {report.n_cases} adversarial divergent cases, "
             f"toolchain_full={report.toolchain_full}"]
    lines.append(f"  confirmed DIVERGENT : {report.n_confirmed_divergent}")
    lines.append(f"  soundness breaches  : {len(report.breaches)}")
    lines.append(f"  SOUND               : {report.sound}")
    if report.breaches:
        lines.append("  --- BREACHES ---")
        for c in report.breaches:
            lines.append(f"    {c.label}: {c.verdict} — {c.detail}")
    return "\n".join(lines)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--attack", action="store_true",
                    help="run the real adversary against actual compilers")
    ap.add_argument("--check", action="store_true",
                    help="regenerate the grid and assert it matches cases.json")
    ap.add_argument("--table", action="store_true",
                    help="print the human-readable adversarial report")
    args = ap.parse_args(argv)

    if args.check and args.attack:
        if not os.path.exists(ATTACK_PATH):
            print("attack.json missing; run --attack first", file=sys.stderr)
            return 1
        try:
            validation = validate_toolchain_file(ATTACK_PATH)
        except ToolchainMismatch as exc:
            print(f"TOOLCHAIN MISMATCH: {exc}", file=sys.stderr)
            return 1
        print(f"OK: red-team attack replays under pinned toolchain ({validation.detail})")
        return 0

    grid = _deterministic_grid()

    if args.check:
        if not os.path.exists(CASES_PATH):
            print("cases.json missing; run without --check first", file=sys.stderr)
            return 1
        with open(CASES_PATH) as f:
            on_disk = f.read()
        regenerated = json.dumps(grid, indent=2, sort_keys=True) + "\n"
        if on_disk != regenerated:
            print("MISMATCH: red-team cases.json is not byte-identical to regeneration",
                  file=sys.stderr)
            return 1
        print("OK: red-team adversarial grid reproduces byte-identically")
        return 0

    _write_json(CASES_PATH, grid)
    print(f"wrote {CASES_PATH}: {grid['n_cases']} adversarial cases "
          f"(symbolic verdict, no toolchain) — {grid['n_breaches']} breaches")

    if args.attack:
        status = toolchain_available()
        report = RT.run_redteam(harness=ReexecHarness(status), status=status)
        _write_json(ATTACK_PATH, report.to_dict())
        print(f"wrote {ATTACK_PATH} (environment-dependent, gitignored)")
        if args.table:
            print()
            print(_render(report))
        if report.breaches:
            print(f"SOUNDNESS BREACH: {len(report.breaches)} divergent cases "
                  f"reported as NO_DIVERGENCE_FOUND", file=sys.stderr)
            return 1
        print(f"red-team PASSED: {report.n_confirmed_divergent}/{report.n_cases} "
              f"cases confirmed DIVERGENT, 0 soundness breaches")
    elif args.table:
        print()
        print(f"deterministic grid: {grid['n_cases']} cases, "
              f"{grid['n_breaches']} breaches (symbolic)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
