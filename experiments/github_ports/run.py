#!/usr/bin/env python3
"""Reproduce the GitHub-mined C->Rust port sample artifacts."""

from __future__ import annotations

import argparse
import json
import sys

from src.ub_oracle import github_port_miner as miner


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true",
                        help="assert results.json matches the checked-in corpus")
    parser.add_argument("--json", action="store_true",
                        help="print the results document to stdout")
    args = parser.parse_args(argv)

    if args.check:
        ok, detail = miner.check_results()
        if not ok:
            print(detail, file=sys.stderr)
            return 1
        print("OK: GitHub-port mining results reproduce byte-identically")
        return 0

    doc = miner.results_document()
    if args.json:
        print(json.dumps(doc, indent=2, sort_keys=True))
    else:
        miner.write_results()
        print(f"wrote {miner.RESULTS_PATH} ({doc['n_verified_samples']} samples, "
              f"{doc['n_candidates']} candidates, hash={doc['content_hash'][:16]})")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
