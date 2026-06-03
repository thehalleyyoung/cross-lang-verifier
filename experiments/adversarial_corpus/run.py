#!/usr/bin/env python3
"""Regenerate or check the Step-165 adversarial near-miss corpus."""

from __future__ import annotations

import argparse
import json
import sys

from src.ub_oracle import adversarial_corpus as AC


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true",
                        help="assert the committed manifest is byte-fresh")
    parser.add_argument("--table", action="store_true",
                        help="print a compact static-verdict summary")
    args = parser.parse_args(argv)

    if args.check:
        ok, detail = AC.check_results()
        if not ok:
            print(f"MISMATCH: {detail}", file=sys.stderr)
            return 1
        print("OK: adversarial corpus reproduces byte-identically")
        return 0

    AC.write_results()
    doc = AC.results_document()
    print(
        f"wrote {AC.RESULTS_PATH}: {doc['census']['n_cases']} cases, "
        f"{doc['static_verdicts']['n_breaches']} soundness breaches"
    )
    if args.table:
        print(json.dumps(doc["static_verdicts"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
