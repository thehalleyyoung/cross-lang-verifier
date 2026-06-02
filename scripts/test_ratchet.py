#!/usr/bin/env python3
"""
True-green test ratchet (100_STEPS step 3).

The whole point of a *divergence oracle* is trustworthiness, so the test suite
itself must be honest: it must be **green** (zero failures, zero errors, zero
surprise ``xpass``\\es) and it must **never silently regress** — neither by
losing passing tests nor by quietly skipping more of them over time.

This gate measures the suite's real outcome counts and enforces a ratchet
against a committed baseline in ``tests/green_baseline.json``:

* **failures / errors must be zero** — always.
* **xpassed must be zero** — an ``xfail`` test that unexpectedly passes is a
  lie in the suite (the bug it documents is gone, or the marker is wrong); fix
  the test or drop the marker, don't let it rot.
* **passed may never drop** below the recorded floor — no quietly deleting or
  ``@skip``-ing a test to make red go away.
* **skipped may never rise** above the recorded ceiling — "un-skip dead tests
  and forbid new skips": a newly-skipped test is invisible rot.

Two profiles share one baseline file:

* ``--fast`` (default): the **toolchain-independent** suite — every test file
  except the heavyweight real-compiler ``test_ub_oracle.py`` driver. It is
  deterministic and runs anywhere in ~10s, so it is the gate developers and the
  pre-commit path run. (The fast unit portion of ``ub_oracle`` is separately
  ratcheted for coverage by ``scripts/coverage_gate.py``.)
* ``--full``: the *entire* suite including the real ``clang``/``rustc``/``go``
  confirmation tests. This is the CI green gate; it needs the full toolchain
  and takes many minutes.

Usage::

    python scripts/test_ratchet.py            # enforce the fast floor
    python scripts/test_ratchet.py --full     # enforce the full floor (CI)
    python scripts/test_ratchet.py --report    # print current vs. baseline
    python scripts/test_ratchet.py --update    # re-baseline the *current* profile
"""

from __future__ import annotations

import argparse
import json
import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_SRC = os.path.join(_ROOT, "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)
BASELINE_PATH = os.path.join(_ROOT, "tests", "green_baseline.json")

#: The slow real-compiler driver. The fast profile runs everything *but* this.
_TOOLCHAIN_DRIVER = os.path.join("tests", "test_ub_oracle.py")


class _StatsPlugin:
    """A tiny pytest plugin that records terminal outcome counts."""

    def __init__(self) -> None:
        self.counts = {
            "passed": 0,
            "failed": 0,
            "error": 0,
            "skipped": 0,
            "xfailed": 0,
            "xpassed": 0,
        }

    def pytest_terminal_summary(self, terminalreporter) -> None:  # noqa: D401
        stats = terminalreporter.stats
        self.counts["passed"] = len(stats.get("passed", []))
        self.counts["failed"] = len(stats.get("failed", []))
        self.counts["error"] = len(stats.get("error", []))
        self.counts["skipped"] = len(stats.get("skipped", []))
        self.counts["xfailed"] = len(stats.get("xfailed", []))
        self.counts["xpassed"] = len(stats.get("xpassed", []))


def _measure(profile: str) -> dict:
    """Run the suite for ``profile`` in-process and return outcome counts."""
    import pytest  # noqa: WPS433

    args = ["-p", "no:cacheprovider", "-q", os.path.join(_ROOT, "tests")]
    if profile == "fast":
        # Deselect the heavyweight real-compiler driver entirely so the gate is
        # deterministic and fast on any machine.
        args += ["--ignore", os.path.join(_ROOT, _TOOLCHAIN_DRIVER)]

    plugin = _StatsPlugin()
    code = pytest.main(args, plugins=[plugin])
    counts = plugin.counts
    counts["_rc"] = int(code)
    return counts


def _load_baseline() -> dict:
    if not os.path.exists(BASELINE_PATH):
        return {}
    with open(BASELINE_PATH) as f:
        return json.load(f)


def _save_baseline(data: dict) -> None:
    with open(BASELINE_PATH, "w") as f:
        json.dump(data, f, indent=2, sort_keys=True)
        f.write("\n")


def _enforce(profile: str, current: dict, floor: dict) -> int:
    """Return process exit code; print violations to stderr."""
    from ub_oracle import test_ratchet_core as _core

    problems = _core.violations(profile, current, floor)
    if problems:
        print(f"test-ratchet [{profile}]: FAILED", file=sys.stderr)
        for p in problems:
            print(f"  - {p}", file=sys.stderr)
        if floor:
            print(
                f"  (floor: passed>={floor['passed']}, skipped<={floor['skipped']}; "
                f"got passed={current['passed']}, skipped={current['skipped']}, "
                f"failed={current['failed']}, xpassed={current['xpassed']})",
                file=sys.stderr,
            )
        return 1

    print(
        f"test-ratchet [{profile}]: PASSED "
        f"(passed={current['passed']}, skipped={current['skipped']}, "
        f"xfailed={current['xfailed']}, failed=0, error=0, xpassed=0)"
    )
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--fast", action="store_true",
                   help="toolchain-independent profile (default)")
    g.add_argument("--full", action="store_true",
                   help="entire suite incl. real-compiler tests (CI)")
    ap.add_argument("--update", action="store_true",
                    help="re-baseline the chosen profile to the current counts")
    ap.add_argument("--report", action="store_true",
                    help="print current vs. baseline and exit 0")
    args = ap.parse_args(argv)

    profile = "full" if args.full else "fast"
    current = _measure(profile)

    baseline = _load_baseline()
    floor = baseline.get(profile, {})

    if args.report:
        print(f"profile: {profile}")
        print(f"current : {json.dumps({k: v for k, v in current.items() if k != '_rc'})}")
        print(f"baseline: {json.dumps(floor) if floor else '(none)'}")
        return 0

    if args.update:
        from ub_oracle import test_ratchet_core as _core

        ok, why = _core.is_baselineable(current)
        if not ok:
            print(why, file=sys.stderr)
            return 2
        baseline[profile] = _core.floor_from(current)
        _save_baseline(baseline)
        print(
            f"test-ratchet [{profile}]: baseline updated -> "
            f"passed={current['passed']}, skipped={current['skipped']}, "
            f"xfailed={current['xfailed']}"
        )
        return 0

    return _enforce(profile, current, floor)


if __name__ == "__main__":
    raise SystemExit(main())
