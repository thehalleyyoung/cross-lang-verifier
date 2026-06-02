#!/usr/bin/env python3
"""
Coverage baseline + ratchet gate (100_STEPS step 8).

The core analysis modules of the verifier — the catalogue/IR, the oracle SPI, the
product-program/decision layer (``verify``), the symbolic searches, the soundness
red-team, and the reporting/triage/cache machinery — must stay well-tested. This
gate measures *branch* coverage of a curated list of those modules over the fast,
toolchain-independent test subset and enforces a **ratchet**: coverage may never
drop below the committed floor in ``coverage_floor.json``, and when it rises you
re-baseline with ``--update`` so the floor only ever moves up.

It is deliberately toolchain-free (it deselects the real-compiler tests) so it is
fast and deterministic and runs anywhere — the re-execution paths are covered
separately by the ``@_requires_toolchain`` suite.

Usage::

    python scripts/coverage_gate.py            # measure + enforce the floor
    python scripts/coverage_gate.py --update   # ratchet the floor up to current
    python scripts/coverage_gate.py --report   # print the per-module table
"""

from __future__ import annotations

import argparse
import json
import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FLOOR_PATH = os.path.join(_ROOT, "coverage_floor.json")

#: The curated set of core analysis modules whose coverage we ratchet. These are
#: the toolchain-independent brains of the tool — the IR/taxonomy, the oracle SPI,
#: the product-program/decision layer, the symbolic searches, and the finding
#: pipeline. The re-execution harness (reexec.py), CLI glue, and the heavyweight
#: study drivers (perf/redteam — each guarded by their own byte-reproducible
#: ``*-check`` gates) are intentionally excluded so this gate stays fast.
CORE_MODULES = [
    "src/ub_oracle/catalogue.py",
    "src/ub_oracle/plugin.py",
    "src/ub_oracle/verify.py",
    "src/ub_oracle/replay.py",
    "src/ub_oracle/report.py",
    "src/ub_oracle/triage.py",
    "src/ub_oracle/suppress.py",
    "src/ub_oracle/cache.py",
    "src/ub_oracle/metrics.py",
    "src/ub_oracle/minimizer.py",
    "src/ub_oracle/regression_matrix.py",
    "src/ub_oracle/dashboard.py",
    "src/ub_oracle/ablation.py",
    "src/ub_oracle/target_semantics.py",
    "src/ub_oracle/oracles/signed_overflow.py",
    "src/ub_oracle/oracles/integer_ub.py",
    "src/ub_oracle/oracles/memory_shape.py",
    "src/ub_oracle/oracles/floating_point.py",
    "src/ub_oracle/oracles/target_pairs.py",
    "src/ub_oracle/oracles/c_to_go.py",
]

#: tests that shell out to real compilers OR drive heavyweight symbolic studies
#: (perf at 512-bit widths, the red-team's all-oracle sweeps) — deselected so the
#: gate is fast and deterministic. Those paths are covered by the toolchain suite
#: and by the ``perf-check`` / ``redteam-check`` reproducibility gates.
_DESELECT_K = (
    "not toolchain and not swift and not go and not real_compiler "
    "and not confirms and not external_plugin and not reexec and not roundtrip "
    "and not head_to_head and not differential_fuzz "
    "and not perf and not redteam"
)

#: how far above the floor we must climb before --update will move it (avoids
#: churn from sub-percent jitter while still ratcheting on real gains).
_RATCHET_MARGIN = 1.0


def _measure() -> dict:
    """Run the fast suite under coverage and return a module->percent map."""
    import tempfile

    import coverage

    cov = coverage.Coverage(branch=True, source=["src/ub_oracle"])
    cov.start()

    import pytest  # noqa: WPS433

    # Run in-process so coverage sees everything; quiet, no cacheprovider.
    code = pytest.main([
        os.path.join(_ROOT, "tests", "test_ub_oracle.py"),
        "-q", "-p", "no:cacheprovider", "-k", _DESELECT_K,
    ])
    cov.stop()
    if code not in (0,):
        print(f"FAIL: the fast test subset did not pass (pytest rc={code})",
              file=sys.stderr)
        raise SystemExit(2)

    # Use the stable JSON-report API for branch-aware per-file percentages.
    with tempfile.NamedTemporaryFile("r", suffix=".json", delete=False) as tf:
        json_path = tf.name
    try:
        cov.json_report(outfile=json_path)
        with open(json_path) as f:
            report = json.load(f)
    finally:
        os.unlink(json_path)

    files = report.get("files", {})
    # normalise keys to repo-relative paths.
    by_rel = {}
    for path, info in files.items():
        rel = os.path.relpath(os.path.join(_ROOT, path), _ROOT) \
            if not os.path.isabs(path) else os.path.relpath(path, _ROOT)
        by_rel[rel] = info

    result = {}
    for rel in CORE_MODULES:
        info = by_rel.get(rel)
        if info is None:
            print(f"WARNING: {rel} was never imported by the fast suite",
                  file=sys.stderr)
            result[rel] = 0.0
            continue
        result[rel] = round(info["summary"]["percent_covered"], 2)
    return result


def _total(pcts: dict) -> float:
    if not pcts:
        return 0.0
    return round(sum(pcts.values()) / len(pcts), 2)


def evaluate(current: dict, floor: dict, *, tol: float = 1e-9):
    """Pure ratchet check: compare measured ``current`` against ``floor``.

    Returns ``(ok, failures, headroom)`` where ``failures`` is a list of human
    strings describing any module/mean that dropped below its committed floor and
    ``headroom`` is how far the mean sits above the mean floor. Extracted as a
    pure function so the ratchet policy is unit-testable without re-running the
    whole suite under coverage.
    """
    failures = []
    for rel, floor_pct in floor.get("modules", {}).items():
        cur = current.get(rel, 0.0)
        if cur + tol < floor_pct:
            failures.append(f"  {rel}: {cur:.2f}% < floor {floor_pct:.2f}%")
    cur_total = _total(current)
    mean_floor = floor.get("mean_floor", 0.0)
    if cur_total + tol < mean_floor:
        failures.append(f"  MEAN: {cur_total:.2f}% < floor {mean_floor:.2f}%")
    return (not failures, failures, round(cur_total - mean_floor, 2))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--update", action="store_true",
                    help="ratchet the committed floor up to the current coverage")
    ap.add_argument("--report", action="store_true",
                    help="print the per-module coverage table")
    args = ap.parse_args(argv)

    os.chdir(_ROOT)
    current = _measure()
    cur_total = _total(current)

    if args.report:
        print(f"{'module':<46} {'cover':>7}")
        print("-" * 55)
        for rel in CORE_MODULES:
            print(f"{rel:<46} {current[rel]:>6.2f}%")
        print("-" * 55)
        print(f"{'MEAN (core modules)':<46} {cur_total:>6.2f}%")

    if args.update:
        floor = {"mean_floor": cur_total, "modules": current}
        with open(FLOOR_PATH, "w") as f:
            f.write(json.dumps(floor, indent=2, sort_keys=True))
            f.write("\n")
        print(f"ratcheted floor -> mean {cur_total:.2f}% across "
              f"{len(current)} core modules ({FLOOR_PATH})")
        return 0

    if not os.path.exists(FLOOR_PATH):
        print("coverage_floor.json missing; run with --update to baseline first",
              file=sys.stderr)
        return 1
    with open(FLOOR_PATH) as f:
        floor = json.load(f)

    ok, failures, head = evaluate(current, floor)
    if not ok:
        print("COVERAGE REGRESSION below the committed ratchet floor:",
              file=sys.stderr)
        for line in failures:
            print(line, file=sys.stderr)
        return 1

    print(f"OK: core coverage {cur_total:.2f}% >= floor "
          f"{floor.get('mean_floor', 0.0):.2f}% (+{head:.2f}%)")
    if head >= _RATCHET_MARGIN:
        print(f"   coverage rose by >= {_RATCHET_MARGIN:.0f}%; run "
              f"`python scripts/coverage_gate.py --update` to ratchet the floor.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
