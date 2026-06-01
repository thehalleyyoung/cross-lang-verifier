#!/usr/bin/env python3
"""
Deterministic reproduction driver for the C->Rust UB-divergence anchor
(100_STEPS steps 2, 14, 24).

Running ``python -m experiments.ub_divergence.run`` regenerates
``experiments/ub_divergence/results.json`` *deterministically* (no toolchain,
no unseeded randomness), so ``make reproduce-check`` can assert byte-identical
regeneration.

With ``--confirm`` it additionally compiles & runs the real C (under UBSan) and
real Rust for each witness via the ground-truth harness and writes
``confirmations.json``.  That artifact is environment-dependent (it needs a C and
Rust toolchain) and is therefore NOT part of the byte-identical reproduce check.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from typing import Dict, List

# Allow running both as a module and as a script.
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(_HERE))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from src.ub_oracle import oracles  # noqa: F401  (registers plugins)
from src.ub_oracle.plugin import get_oracle
from src.ub_oracle.diff_testing import measure_fuzzing_gap
from src.ub_oracle.reexec import ReexecHarness, toolchain_available

# A fixed, ordered set of anchor units.  Deterministic by construction.
UNITS: List[Dict] = [
    {"kind": "binop_const", "op": "add", "const": 1, "width": 32, "var": "x", "signed": True},
    {"kind": "binop_const", "op": "add", "const": 7, "width": 32, "var": "x", "signed": True},
    {"kind": "binop_const", "op": "sub", "const": 1, "width": 32, "var": "x", "signed": True},
    {"kind": "binop_const", "op": "add", "const": 1, "width": 64, "var": "x", "signed": True},
]

FUZZ_TRIALS = 200_000
FUZZ_SEED = 0
RESULTS_PATH = os.path.join(_HERE, "results.json")
CONFIRMATIONS_PATH = os.path.join(_HERE, "confirmations.json")


def _sha(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()[:16]


def build_results() -> Dict:
    orc = get_oracle("signed_overflow")
    cases = []
    for unit in UNITS:
        res = orc.find_divergence(unit)
        ce = res.counterexample
        gap = measure_fuzzing_gap(unit["op"], unit["const"], unit["width"],
                                  trials=FUZZ_TRIALS, seed=FUZZ_SEED)
        cases.append({
            "divergence_class": res.divergence_class,
            "op": unit["op"],
            "const": unit["const"],
            "width": unit["width"],
            "verdict": str(res.verdict),
            "witness": ce.inputs,
            "source_definedness": ce.source_definedness,
            "c_src_sha16": _sha(ce.source_snippet),
            "rust_src_sha16": _sha(ce.target_snippet),
            "divergence_witness": ce.divergence_witness,
            "fuzzing_gap": {
                "total_inputs": gap.total_inputs,
                "overflow_inputs": gap.overflow_inputs,
                "hit_probability": gap.hit_probability,
                "trials": gap.trials,
                "empirical_hits": gap.empirical_hits,
                "seed": gap.seed,
                "oracle_hit_probability": gap.oracle_hit_probability,
            },
        })
    return {
        "artifact": "ub_divergence_anchor",
        "language_pair": "c->rust",
        "fuzz_trials": FUZZ_TRIALS,
        "fuzz_seed": FUZZ_SEED,
        "cases": cases,
    }


def build_confirmations() -> Dict:
    orc = get_oracle("signed_overflow")
    status = toolchain_available()
    harness = ReexecHarness(status)
    out = {"toolchain": {"cc": status.cc, "rustc": status.rustc, "ubsan": status.ubsan},
           "cases": []}
    for unit in UNITS:
        res = orc.confirm(orc.find_divergence(unit), harness)
        rr = res.reexec
        out["cases"].append({
            "op": unit["op"], "const": unit["const"], "width": unit["width"],
            "witness": res.counterexample.inputs,
            "available": rr.available,
            "ub_reachable": rr.ub_reachable,
            "ub_consequential": rr.ub_consequential,
            "rust_defined": rr.rust_defined,
            "confirmed": rr.confirmed,
            "reason": rr.reason,
        })
    return out


def _write_json(path: str, obj: Dict) -> None:
    with open(path, "w") as f:
        f.write(json.dumps(obj, indent=2, sort_keys=True))
        f.write("\n")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--confirm", action="store_true",
                    help="also run the real-compiler ground-truth confirmation")
    ap.add_argument("--check", action="store_true",
                    help="regenerate in-memory and assert it matches results.json")
    args = ap.parse_args(argv)

    results = build_results()
    if args.check:
        if not os.path.exists(RESULTS_PATH):
            print("results.json missing; run without --check first", file=sys.stderr)
            return 1
        with open(RESULTS_PATH) as f:
            on_disk = f.read()
        regenerated = json.dumps(results, indent=2, sort_keys=True) + "\n"
        if on_disk != regenerated:
            print("MISMATCH: results.json is not byte-identical to regeneration",
                  file=sys.stderr)
            return 1
        print("OK: results.json reproduces byte-identically")
        return 0

    _write_json(RESULTS_PATH, results)
    print(f"wrote {RESULTS_PATH} ({len(results['cases'])} cases)")
    for c in results["cases"]:
        g = c["fuzzing_gap"]
        print(f"  {c['op']} c={c['const']} w={c['width']}: witness={c['witness']} "
              f"UB inputs={g['overflow_inputs']}/{g['total_inputs']} "
              f"fuzzer={g['empirical_hits']}/{g['trials']} oracle_p={g['oracle_hit_probability']:.0f}")

    if args.confirm:
        conf = build_confirmations()
        _write_json(CONFIRMATIONS_PATH, conf)
        n_ok = sum(1 for c in conf["cases"] if c["confirmed"])
        print(f"wrote {CONFIRMATIONS_PATH}: {n_ok}/{len(conf['cases'])} confirmed "
              f"against real compilers")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
