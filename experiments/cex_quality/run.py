#!/usr/bin/env python3
"""
Counterexample-quality study driver (100_STEPS step 52).

A confirmed divergence is only as useful as the witness that demonstrates it.
This study takes every anchor (C->Rust) oracle's *raw* symbolic witness and
shrinks it to a locally-minimal canonical form, then proves — by really
compiling and running the C (under UBSan) and the Rust target — that the
minimized witness still triggers the SAME undefined behavior.

Two layers, mirroring the cross-pair matrix:

* The deterministic layer (no toolchain) records each oracle's raw witness and
  its symbolic simplicity cost in ``baseline.json``, which is version-controlled
  and regenerates byte-identically (``--check``).
* The empirical layer (``--minimize``, needs clang+UBSan+rustc) actually runs the
  minimizer against real compilers and writes the environment-dependent
  ``minimized.json`` (gitignored), printing a quality table.

The faithfulness guarantee under test: magnitude minimization must never drift a
witness into a *different* UB (INT_MIN/-1 must not collapse to division-by-zero;
a too-large shift must not become a negative shift).
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
from src.ub_oracle.plugin import ALL_ORACLES
from src.ub_oracle.minimizer import minimize_counterexample, simplicity_cost
from src.ub_oracle.regression_matrix import canonical_unit_for
from src.ub_oracle.reexec import ReexecHarness, toolchain_available
from src.ub_oracle.cache import (
    ToolchainMismatch,
    toolchain_provenance,
    validate_toolchain_file,
)

BASELINE_PATH = os.path.join(_HERE, "baseline.json")
MINIMIZED_PATH = os.path.join(_HERE, "minimized.json")


def _anchor_oracles():
    """The C->Rust anchor oracles, sorted by class for determinism."""
    anchors = [o for o in ALL_ORACLES
               if o.source_lang == "c" and o.target_lang == "rust"]
    return sorted(anchors, key=lambda o: o.divergence_class)


def _symbolic_cost(inputs) -> int:
    return sum(simplicity_cost(v)[0] for v in inputs.values()
               if isinstance(v, int))


def build_baseline() -> dict:
    """The deterministic, toolchain-free record of each raw witness."""
    rows = []
    for orc in _anchor_oracles():
        res = orc.find_divergence(canonical_unit_for(orc))
        if res.counterexample is None:
            continue
        inputs = dict(res.counterexample.inputs)
        rows.append({
            "divergence_class": orc.divergence_class,
            "source_lang": orc.source_lang,
            "target_lang": orc.target_lang,
            "raw_inputs": inputs,
            "raw_cost": _symbolic_cost(inputs),
        })
    return {"n_oracles": len(rows), "rows": rows}


def _write_json(path: str, obj) -> None:
    with open(path, "w") as f:
        f.write(json.dumps(obj, indent=2, sort_keys=True))
        f.write("\n")


def _render_table(results) -> str:
    head = (f"{'class':18s} {'pair':10s} {'raw':>22s} {'minimized':>22s} "
            f"{'red?':4s} {'locmin?':7s} {'probes':>6s}")
    lines = [head, "-" * len(head)]
    for r in results:
        pair = f"{r['source_lang']}->{r['target_lang']}"
        lines.append(
            f"{r['divergence_class']:18s} {pair:10s} "
            f"{str(r['original_inputs']):>22s} {str(r['minimized_inputs']):>22s} "
            f"{('yes' if r['reduced'] else 'no'):4s} "
            f"{('yes' if r['certified_locally_minimal'] else 'no'):7s} "
            f"{r['probes']:>6d}")
    return "\n".join(lines)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--minimize", action="store_true",
                    help="minimize every witness against real compilers")
    ap.add_argument("--check", action="store_true",
                    help="regenerate baseline in-memory and assert byte match")
    ap.add_argument("--table", action="store_true",
                    help="print the human-readable quality table")
    ap.add_argument("--max-probes", type=int, default=400,
                    help="per-witness real-compiler probe budget")
    args = ap.parse_args(argv)

    if args.check and args.minimize:
        if not os.path.exists(MINIMIZED_PATH):
            print("minimized.json missing; run --minimize first", file=sys.stderr)
            return 1
        try:
            validation = validate_toolchain_file(MINIMIZED_PATH)
        except ToolchainMismatch as exc:
            print(f"TOOLCHAIN MISMATCH: {exc}", file=sys.stderr)
            return 1
        print(f"OK: minimized witnesses replay under pinned toolchain ({validation.detail})")
        return 0

    baseline = build_baseline()

    if args.check:
        if not os.path.exists(BASELINE_PATH):
            print("baseline.json missing; run without --check first",
                  file=sys.stderr)
            return 1
        with open(BASELINE_PATH) as f:
            on_disk = f.read()
        regenerated = json.dumps(baseline, indent=2, sort_keys=True) + "\n"
        if on_disk != regenerated:
            print("MISMATCH: cex-quality baseline.json is not byte-identical",
                  file=sys.stderr)
            return 1
        print("OK: cex-quality baseline reproduces byte-identically")
        return 0

    _write_json(BASELINE_PATH, baseline)
    print(f"wrote {BASELINE_PATH}: {baseline['n_oracles']} anchor witnesses")

    if args.minimize:
        status = toolchain_available()
        harness = ReexecHarness(status)
        provenance = toolchain_provenance(status)
        results = []
        for orc in _anchor_oracles():
            res = orc.find_divergence(canonical_unit_for(orc))
            if res.counterexample is None:
                continue
            m = minimize_counterexample(orc, res, harness,
                                        max_probes=args.max_probes)
            results.append(m.to_dict())
        n_confirmed = sum(1 for r in results if r["confirmed"])
        n_reduced = sum(1 for r in results if r["reduced"])
        n_locmin = sum(1 for r in results if r["certified_locally_minimal"])
        payload = {
            "schema": "cex-quality-minimized/v2",
            "toolchain_fingerprint": provenance["fingerprint"],
            "toolchain_provenance": provenance,
            "n_oracles": len(results),
            "n_confirmed": n_confirmed,
            "n_reduced": n_reduced,
            "n_locally_minimal": n_locmin,
            "results": results,
        }
        _write_json(MINIMIZED_PATH, payload)
        print(f"wrote {MINIMIZED_PATH}: {n_confirmed}/{len(results)} confirmed, "
              f"{n_reduced} reduced, {n_locmin} certified locally minimal")
        if args.table:
            print()
            print(_render_table(results))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
