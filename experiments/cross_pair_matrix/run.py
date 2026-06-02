#!/usr/bin/env python3
"""
Cross-pair regression-matrix driver (100_STEPS step 40).

``python -m experiments.cross_pair_matrix.run`` regenerates
``experiments/cross_pair_matrix/results.json`` *deterministically* (no toolchain),
so ``make matrix-check`` can assert byte-identical regeneration — the matrix is
the living, version-controlled evidence that the oracle suite generalises across
every supported ``(source, target)`` language pair.

With ``--confirm`` it additionally compiles & runs every cell's witness against
the real C (under UBSan) and the real target compiler for whatever pairs the host
toolchain supports, writing the environment-dependent ``confirmations.json``.

With ``--table`` it prints the human-readable grid.
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
from src.ub_oracle import regression_matrix as M
from src.ub_oracle.reexec import ReexecHarness, toolchain_available

RESULTS_PATH = os.path.join(_HERE, "results.json")
CONFIRMATIONS_PATH = os.path.join(_HERE, "confirmations.json")


def _write_json(path: str, obj) -> None:
    with open(path, "w") as f:
        f.write(json.dumps(obj, indent=2, sort_keys=True))
        f.write("\n")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--confirm", action="store_true",
                    help="also confirm every cell against real compilers")
    ap.add_argument("--check", action="store_true",
                    help="regenerate in-memory and assert it matches results.json")
    ap.add_argument("--table", action="store_true",
                    help="print the human-readable pair x class grid")
    args = ap.parse_args(argv)

    matrix = M.build_matrix()

    if args.check:
        if not os.path.exists(RESULTS_PATH):
            print("results.json missing; run without --check first", file=sys.stderr)
            return 1
        with open(RESULTS_PATH) as f:
            on_disk = f.read()
        regenerated = json.dumps(matrix, indent=2, sort_keys=True) + "\n"
        if on_disk != regenerated:
            print("MISMATCH: matrix results.json is not byte-identical to regeneration",
                  file=sys.stderr)
            return 1
        print("OK: cross-pair matrix reproduces byte-identically")
        return 0

    _write_json(RESULTS_PATH, matrix)
    print(f"wrote {RESULTS_PATH}: {matrix['n_cells']} cells across "
          f"{len(matrix['language_pairs'])} pairs "
          f"({', '.join(matrix['language_pairs'])})")
    if args.table:
        print()
        print(M.render_table(matrix))

    if args.confirm:
        harness = ReexecHarness(toolchain_available())
        conf = M.confirm_matrix(harness)
        _write_json(CONFIRMATIONS_PATH, conf)
        print(f"wrote {CONFIRMATIONS_PATH}: {conf['n_confirmed']}/{conf['n_attempted']} "
              f"cells confirmed against real compilers "
              f"({conf['n_cells'] - conf['n_attempted']} skipped: toolchain absent)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
