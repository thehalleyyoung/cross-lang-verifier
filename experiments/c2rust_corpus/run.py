#!/usr/bin/env python3
"""Reproduce the Tier-1 c2rust-output corpus artifacts."""

from __future__ import annotations

import argparse
import json
import sys

from src.ub_oracle import c2rust_corpus as corpus


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--regenerate", action="store_true",
                        help="run c2rust and overwrite generated/*.rs")
    parser.add_argument("--check", action="store_true",
                        help="assert results.json matches the checked-in corpus")
    parser.add_argument("--check-generated", action="store_true",
                        help="also regenerate with c2rust and compare generated/*.rs")
    parser.add_argument("--json", action="store_true",
                        help="print the results document to stdout")
    args = parser.parse_args(argv)

    if args.regenerate:
        hashes = corpus.regenerate_generated()
        print(f"regenerated {len(hashes)} c2rust Rust artifact(s)")

    if args.check_generated:
        if corpus.c2rust_path() is None:
            print("c2rust is not on PATH", file=sys.stderr)
            return 1
        cmp = corpus.compare_generated_to_c2rust()
        if not cmp["ok"]:
            print("generated Rust differs from c2rust output: "
                  + ", ".join(cmp["mismatches"]), file=sys.stderr)
            return 1
        print(f"OK: generated Rust matches {cmp['c2rust_version']}")

    if args.check:
        ok, detail = corpus.check_results()
        if not ok:
            print(detail, file=sys.stderr)
            return 1
        print("OK: c2rust corpus results reproduce byte-identically")
        return 0

    doc = corpus.results_document()
    if args.json:
        print(json.dumps(doc, indent=2, sort_keys=True))
    else:
        corpus.write_results()
        print(f"wrote {corpus.RESULTS_PATH} ({doc['n_items']} items, "
              f"{doc['n_source_libraries']} source libraries, "
              f"hash={doc['content_hash'][:16]})")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
