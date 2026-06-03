#!/usr/bin/env python3
"""Regenerate or check the Step-166 corpus datasheet."""

from __future__ import annotations

import argparse
import json
import sys

from src.ub_oracle import corpus_datasheet as datasheet


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write", action="store_true",
                        help="regenerate the JSON and Markdown datasheets")
    parser.add_argument("--check", action="store_true",
                        help="assert the committed datasheets are byte-fresh")
    parser.add_argument("--json", action="store_true",
                        help="print the generated JSON document")
    args = parser.parse_args(argv)

    if args.check:
        ok, detail = datasheet.check_all()
        if not ok:
            print(detail, file=sys.stderr)
            return 1
        print("OK: corpus datasheet reproduces byte-identically")
        return 0

    doc = datasheet.results_document()
    if args.json:
        print(json.dumps(doc, indent=2, sort_keys=True))
        return 0

    datasheet.write_all()
    print(
        f"wrote {datasheet.RESULTS_PATH} and {datasheet.DOC_PATH} "
        f"({doc['n_records']} records, hash={doc['content_hash'][:16]})"
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
