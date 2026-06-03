from __future__ import annotations

import argparse
import json
import sys

from src.ub_oracle.existing_tools_study import (
    DEFAULT_SEED,
    DEFAULT_TRIALS,
    confirm_step162_existing_tools,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run the Step 162 same-corpus existing-tools comparison.")
    parser.add_argument("--trials", type=int, default=DEFAULT_TRIALS)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--skip-miri", action="store_true",
                        help="Record Miri as not-run instead of probing cargo miri.")
    parser.add_argument("--check", action="store_true",
                        help="Exit non-zero if the Step 162 acceptance check fails.")
    args = parser.parse_args(argv)

    conf = confirm_step162_existing_tools(
        trials=args.trials,
        seed=args.seed,
        run_miri=not args.skip_miri,
    )
    print(json.dumps(conf.report.summary(), indent=2, sort_keys=True))
    if args.check and not conf.ok:
        print(conf.detail, file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
