#!/usr/bin/env python3
"""Regenerate, check, or run the Step-167 continuous corpus CI plan."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from src.ub_oracle import continuous_corpus_ci as cci


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write", action="store_true", help="regenerate the deterministic plan artifacts")
    parser.add_argument("--check", action="store_true", help="assert the deterministic plan artifacts are fresh")
    parser.add_argument("--json", action="store_true", help="print the deterministic plan JSON")
    parser.add_argument("--run", action="store_true", help="execute the corpus checks and emit a runtime report")
    parser.add_argument(
        "--command",
        action="append",
        default=None,
        help="exact validation command to run; repeat to select a subset",
    )
    parser.add_argument(
        "--command-timeout-seconds",
        type=int,
        default=cci.DEFAULT_COMMAND_TIMEOUT_SECONDS,
        help="per-validation-command timeout for --run",
    )
    parser.add_argument(
        "--report-path",
        type=Path,
        default=cci.DEFAULT_REPORT_PATH,
        help="ephemeral runtime report path used by --run",
    )
    args = parser.parse_args(argv)

    if args.check:
        ok, detail = cci.check_all()
        if not ok:
            print(detail, file=sys.stderr)
            return 1
        print("OK: continuous corpus CI plan reproduces byte-identically")
        return 0

    if args.json:
        print(json.dumps(cci.plan_document(), indent=2, sort_keys=True))
        return 0

    if args.run:
        report = cci.run_nightly(
            commands=args.command,
            command_timeout=args.command_timeout_seconds,
            report_path=args.report_path,
        )
        print(json.dumps(report, indent=2, sort_keys=True))
        return 0 if report["ok"] else 1

    cci.write_all()
    doc = cci.plan_document()
    print(
        f"wrote {cci.RESULTS_PATH} and {cci.DOC_PATH} "
        f"({doc['n_corpora']} corpora, {doc['n_commands']} commands, "
        f"hash={doc['verdict_layer_hash'][:16]})"
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
