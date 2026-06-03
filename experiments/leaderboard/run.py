#!/usr/bin/env python3
"""Regenerate, check, or score the Step-168 public leaderboard."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from src.ub_oracle import leaderboard


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write", action="store_true",
                        help="regenerate the public cases, answer key, sample submission, and docs")
    parser.add_argument("--check", action="store_true",
                        help="assert committed leaderboard artifacts are byte-fresh")
    parser.add_argument("--score", metavar="SUBMISSION.json",
                        help="score a submission against the answer key")
    parser.add_argument("--answer-key", type=Path, default=leaderboard.ANSWER_KEY_PATH,
                        help="answer key used by --score (default: committed reference key)")
    parser.add_argument("--json", action="store_true",
                        help="print machine-readable JSON for --score or generated summary")
    args = parser.parse_args(argv)

    if args.check:
        ok, detail = leaderboard.check_all()
        if not ok:
            print(detail, file=sys.stderr)
            return 1
        print("OK: leaderboard artifacts reproduce byte-identically")
        return 0

    if args.score:
        submission_path = Path(args.score)
        submission = json.loads(submission_path.read_text(encoding="utf-8"))
        answer_key = json.loads(args.answer_key.read_text(encoding="utf-8"))
        score = leaderboard.score_submission(submission, answer_key)
        payload = score.to_dict()
        if args.json:
            print(json.dumps(payload, indent=2, sort_keys=True))
        else:
            print(
                f"{payload['submission_id']}: primary_score={payload['primary_score']:.2f} "
                f"coverage={payload['coverage']:.3f} macro_f1={payload['macro_f1']:.3f} "
                f"valid={payload['valid']}"
            )
            if payload["errors"]:
                print("errors:", "; ".join(payload["errors"]), file=sys.stderr)
        return 0 if score.valid else 2

    if args.json:
        print(json.dumps(leaderboard.results_document(), indent=2, sort_keys=True))
        return 0

    leaderboard.write_all()
    doc = leaderboard.results_document()
    print(
        f"wrote leaderboard artifacts under {leaderboard.EXPERIMENT_DIR} "
        f"({doc['n_cases']} cases, hash={doc['content_hash'][:16]})"
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
