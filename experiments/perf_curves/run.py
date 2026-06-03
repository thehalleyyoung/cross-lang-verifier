#!/usr/bin/env python3
"""
Performance / scalability study driver (100_STEPS step 50).

``python -m experiments.perf_curves.run`` regenerates the *deterministic* grid
``experiments/perf_curves/grid.json`` (classes x pairs x widths x SMT sizes, with
verdicts and the SMT result at each width) — **no timings**, so ``make perf-check``
can assert byte-identical regeneration.  The grid is the version-controlled
evidence that the scalability study covers the whole matrix and that every search
stays satisfiable as the bit width grows to 512.

With ``--measure`` it additionally times the *real* Z3 searches and the SMT
scaling curve and writes the environment-dependent ``timings.json`` (gitignored).
``--table`` prints the human-readable curves. ``--budget-check`` times the real
CI-sized searches and fails if any row exceeds its p50/p95 latency budget.
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
from src.ub_oracle import perf
from src.ub_oracle.cache import (
    ToolchainMismatch,
    toolchain_provenance,
    validate_toolchain_file,
)

GRID_PATH = os.path.join(_HERE, "grid.json")
TIMINGS_PATH = os.path.join(_HERE, "timings.json")
BUDGET_PATH = os.path.join(_HERE, "budget.json")


def _write_json(path: str, obj) -> None:
    with open(path, "w") as f:
        f.write(json.dumps(obj, indent=2, sort_keys=True))
        f.write("\n")


def _render_table(timings: dict) -> str:
    lines = []
    lines.append("== class x pair search time (median seconds) ==")
    for row in timings["class_pair_profile"]:
        lines.append(f"  {row['label']:<34} {row['seconds']:.5f}  {row['verdict']}")
    lines.append("")
    lines.append("== integer search vs bit width ==")
    for row in timings["width_scaling"]:
        lines.append(f"  {row['label']:<20} {row['seconds']:.5f}  {row['verdict']}")
    lines.append("")
    c = timings["smt_scaling"]
    lines.append("== SMT overflow-search vs bit width ==")
    for w, s, v in zip(c["sizes"], c["seconds"], c["verdicts"]):
        lines.append(f"  width {w:>4}  {s:.5f}s  {v}")
    lines.append(f"  growth_ratio/doubling = {c['growth_ratio']:.3f}  "
                 f"(pathological={c['pathological']}, threshold={c['threshold']})")
    return "\n".join(lines)


def _render_budget_summary(report: dict) -> str:
    sections = (
        ("class/pair", report["class_pair_profile"]),
        ("width", report["width_scaling"]),
        ("smt", report["smt_scaling"]),
    )
    lines = ["== perf latency budget (p50/p95 seconds) =="]
    for name, rows in sections:
        bad = [r for r in rows if not r["ok"]]
        p95s = [r["p95_seconds"] for r in rows]
        max_p95 = max(p95s) if p95s else 0.0
        lines.append(f"  {name:<10} rows={len(rows):>3} "
                     f"max_p95={max_p95:.6f} failures={len(bad)}")
    if report["failures"]:
        lines.append("  failures:")
        for failure in report["failures"][:20]:
            lines.append(f"    - {failure}")
    return "\n".join(lines)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--measure", action="store_true",
                    help="time the real searches and write timings.json")
    ap.add_argument("--check", action="store_true",
                    help="regenerate the grid in-memory and assert it matches grid.json")
    ap.add_argument("--table", action="store_true",
                    help="print the human-readable curves (implies --measure)")
    ap.add_argument("--repeats", type=int, default=5,
                    help="timing repeats (median); default 5")
    ap.add_argument("--budget-check", action="store_true",
                    help="time real searches and enforce p50/p95 latency budgets")
    ap.add_argument("--budget-repeats", type=int, default=3,
                    help="budget-check timing repeats; default 3")
    ap.add_argument("--write-budget-report", action="store_true",
                    help="write environment-dependent budget.json")
    args = ap.parse_args(argv)

    if args.check and (args.measure or args.table):
        if not os.path.exists(TIMINGS_PATH):
            print("timings.json missing; run --measure first", file=sys.stderr)
            return 1
        try:
            validation = validate_toolchain_file(TIMINGS_PATH)
        except ToolchainMismatch as exc:
            print(f"TOOLCHAIN MISMATCH: {exc}", file=sys.stderr)
            return 1
        print(f"OK: perf timings replay under pinned toolchain ({validation.detail})")
        return 0

    grid = perf.deterministic_grid()

    if args.check:
        if not os.path.exists(GRID_PATH):
            print("grid.json missing; run without --check first", file=sys.stderr)
            return 1
        with open(GRID_PATH) as f:
            on_disk = f.read()
        regenerated = json.dumps(grid, indent=2, sort_keys=True) + "\n"
        if on_disk != regenerated:
            print("MISMATCH: perf grid.json is not byte-identical to regeneration",
                  file=sys.stderr)
            return 1
        print("OK: perf scalability grid reproduces byte-identically")
        if args.budget_check:
            report = perf.latency_budget_report(repeats=args.budget_repeats)
            print(_render_budget_summary(report))
            if args.write_budget_report:
                _write_json(BUDGET_PATH, report)
                print(f"wrote {BUDGET_PATH} (environment-dependent, gitignored)")
            if not report["ok"]:
                return 1
        return 0

    _write_json(GRID_PATH, grid)
    print(f"wrote {GRID_PATH}: {len(grid['class_pair_profile'])} class/pair cells, "
          f"{len(grid['width_scaling'])} width cells, "
          f"{len(grid['smt_scaling'])} SMT sizes")

    if args.measure or args.table:
        provenance = toolchain_provenance()
        timings = {
            "schema": "perf-timings/v2",
            "toolchain_fingerprint": provenance["fingerprint"],
            "toolchain_provenance": provenance,
            "z3_version": perf.z3_version(),
            "class_pair_profile": [r.to_dict() for r in
                                   perf.class_pair_profile(repeats=args.repeats)],
            "width_scaling": [r.to_dict() for r in
                              perf.width_scaling_curve(repeats=args.repeats)],
            "smt_scaling": perf.smt_scaling_curve(repeats=args.repeats).to_dict(),
        }
        _write_json(TIMINGS_PATH, timings)
        print(f"wrote {TIMINGS_PATH} (environment-dependent, gitignored)")
        if args.table:
            print()
            print(_render_table(timings))
    if args.budget_check:
        report = perf.latency_budget_report(repeats=args.budget_repeats)
        if args.write_budget_report:
            _write_json(BUDGET_PATH, report)
            print(f"wrote {BUDGET_PATH} (environment-dependent, gitignored)")
        print()
        print(_render_budget_summary(report))
        if not report["ok"]:
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
