"""
Pure decision core for the true-green test ratchet (100_STEPS step 3).

This module holds the *toolchain-free* part of ``scripts/test_ratchet.py``: the
function that, given a profile's measured outcome counts and a committed floor,
decides whether the suite is honest-green and non-regressed. It is factored out
of the script so the policy can be unit-tested and machine-verified
(``traceability._thm_true_green_ratchet``) without actually running pytest.

A run is rejected iff any of these hold:

* ``failed`` or ``error`` is non-zero (the suite is red),
* ``xpassed`` is non-zero (an ``xfail`` unexpectedly passed -- a stale lie),
* ``passed`` dropped below the floor (a passing test was lost), or
* ``skipped`` rose above the floor (a test was silently skipped).
"""

from __future__ import annotations

from typing import Dict, List, Mapping, Tuple

Counts = Mapping[str, int]


def violations(profile: str, current: Counts, floor: Counts) -> List[str]:
    """Return a list of human-readable ratchet violations (empty == green)."""
    problems: List[str] = []

    if int(current.get("failed", 0)) != 0:
        problems.append(f"{current.get('failed')} test(s) FAILED (must be 0)")
    if int(current.get("error", 0)) != 0:
        problems.append(f"{current.get('error')} test(s) ERRORED (must be 0)")
    if int(current.get("xpassed", 0)) != 0:
        problems.append(
            f"{current.get('xpassed')} test(s) XPASSED -- an xfail unexpectedly "
            "passed; fix the test or drop the marker (must be 0)"
        )

    if not floor:
        problems.append(
            f"no baseline for profile '{profile}'. Run with --update to record "
            "the current green state."
        )
    else:
        if int(current.get("passed", 0)) < int(floor.get("passed", 0)):
            problems.append(
                f"passed regressed: {current.get('passed')} < floor "
                f"{floor.get('passed')} (a passing test was lost)"
            )
        if int(current.get("skipped", 0)) > int(floor.get("skipped", 0)):
            problems.append(
                f"skipped grew: {current.get('skipped')} > ceiling "
                f"{floor.get('skipped')} (a new test was silently skipped)"
            )

    return problems


def enforce_counts(profile: str, current: Counts, floor: Counts) -> int:
    """Return a process exit code: 0 if green and non-regressed, else 1."""
    return 0 if not violations(profile, current, floor) else 1


def is_baselineable(current: Counts) -> Tuple[bool, str]:
    """Whether ``current`` is green enough to record as a new floor."""
    f, e, x = (int(current.get(k, 0)) for k in ("failed", "error", "xpassed"))
    if f or e or x:
        return False, (
            f"refusing to baseline a non-green run "
            f"(failed={f}, error={e}, xpassed={x})"
        )
    return True, ""


def floor_from(current: Counts) -> Dict[str, int]:
    """Project a measurement onto the persisted floor shape."""
    return {
        "passed": int(current.get("passed", 0)),
        "skipped": int(current.get("skipped", 0)),
        "xfailed": int(current.get("xfailed", 0)),
    }
