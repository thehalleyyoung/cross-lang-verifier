"""
N-language consistency oracle (100_STEPS step 125).

Given *one* C source translated to **three or more** target languages, run the
*same* witnessing input through every translation and flag any target that
disagrees with the majority.  This is the multi-target generalisation of the
pairwise divergence oracle: instead of asking "does C diverge from target T?", it
asks "do the N translations of this C unit *agree with each other*?", and pins the
blame on the minority.

Why this catches real bugs the pairwise view misses: a UB-rooted C unit is
undefined, so each *safe* target is free to resolve it to its own defined value —
and they need not agree.  For example, the out-of-range shift probe ``1 << 32``
*masks* the shift amount in Rust (``wrapping_shl`` → ``1 << (32 mod 32) == 1``) but
yields ``0`` in Go and Swift.  Two targets agree, one dissents: Rust is flagged as
the outlier, with a live, reproducible witness.  Conversely a divide-by-zero probe
makes *every* safe target abort deterministically — they all canonicalise to
``ABORT`` and the unit is reported **consistent**.

The oracle is built entirely on the existing pluggable machinery:

  * the participating targets for a class are exactly the registered oracles
    ``oracles_for("c", T, class)`` — no new per-target code;
  * each oracle already emits its target program from the *shared* anchor witness
    search, so all N translations are driven by one identical C source + input;
  * the live comparison uses the re-execution harness's single-shot build+run
    helpers and each target pack's *defined-outcome* data to canonicalise results.

:func:`plan_consistency` is the pure-data plan (which targets participate, no
toolchain); :func:`check_consistency` is the live, compiler-backed comparison; and
:func:`consistency_units` is the curated sample the test-suite witnesses.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from .plugin import oracles_for
from .reexec import ReexecHarness, RunOutcome, toolchain_available
from .target_semantics import get_pack


# ── canonicalising an observable outcome ─────────────────────────────────────

def canonical_outcome(outcome: RunOutcome, target_lang: str) -> str:
    """Reduce a target run to a comparable *observable* token.

    Two translations are "the same" iff they produce the same token:

      * ``VALUE:<stdout>``   — a normal exit (return code 0) printing a value;
      * ``ABORT``            — a *defined*, deterministic abort (a panic / trap /
                               uncaught exception, per the target's pack);
      * ``TIMEOUT``          — the run did not terminate in budget;
      * ``UNDEFINED:rc=<n>`` — a non-defined termination (should never happen for
                               a safe target; surfaced rather than hidden).
    """
    if outcome.timed_out:
        return "TIMEOUT"
    if outcome.returncode == 0:
        return f"VALUE:{outcome.stdout}"
    if outcome.returncode in get_pack(target_lang).defined_returncodes:
        return "ABORT"
    return f"UNDEFINED:rc={outcome.returncode}"


# ── the plan (pure data, no toolchain) ───────────────────────────────────────

@dataclass(frozen=True)
class ConsistencyPlan:
    divergence_class: str
    targets: Tuple[str, ...]

    @property
    def is_n_language(self) -> bool:
        """At least three translations to compare (the step's ``>= 3`` bar)."""
        return len(self.targets) >= 3


def plan_consistency(divergence_class: str) -> ConsistencyPlan:
    """Which target languages have a translation for this class (sorted)."""
    targets = sorted({o.target_lang for o in oracles_for(
        source_lang="c", divergence_class=divergence_class)})
    return ConsistencyPlan(divergence_class, tuple(targets))


# ── the live comparison ──────────────────────────────────────────────────────

@dataclass
class TargetObservation:
    target_lang: str
    token: str
    returncode: int
    stdout: str


@dataclass
class ConsistencyReport:
    divergence_class: str
    available: bool
    inputs: Dict[str, object] = field(default_factory=dict)
    observations: List[TargetObservation] = field(default_factory=list)
    majority_token: Optional[str] = None
    flagged: Tuple[str, ...] = ()
    c_ub_reachable: bool = False
    reason: str = ""

    @property
    def n_targets(self) -> int:
        return len(self.observations)

    @property
    def consistent(self) -> bool:
        """All compared translations produced the same observable token."""
        return (self.available and self.n_targets >= 3
                and len(self.flagged) == 0)

    @property
    def has_outlier(self) -> bool:
        return self.available and len(self.flagged) > 0


def _participating_oracles(divergence_class: str, harness: ReexecHarness):
    """Registered (target -> oracle) for the class whose compiler is present."""
    out = []
    for o in oracles_for(source_lang="c", divergence_class=divergence_class):
        if harness.status.target_available(o.target_lang):
            out.append(o)
    # deterministic order, one oracle per target.
    by_target: Dict[str, object] = {}
    for o in out:
        by_target.setdefault(o.target_lang, o)
    return [by_target[t] for t in sorted(by_target)]


def check_consistency(divergence_class: str, unit: Dict,
                      harness: Optional[ReexecHarness] = None) -> ConsistencyReport:
    """Build the one C source and its N translations, run the shared witness
    through each, and flag every target that disagrees with the majority.

    The unit is the same shape the pairwise oracles consume; ``unit`` drives the
    (shared) anchor witness search, so every target receives a byte-identical C
    source and identical input."""
    harness = harness or ReexecHarness()
    report = ConsistencyReport(divergence_class=divergence_class, available=False)

    oracles = _participating_oracles(divergence_class, harness)
    if not harness.status.c_available or len(oracles) < 3:
        report.reason = (
            f"need a C compiler and >=3 available targets; have C="
            f"{harness.status.c_available}, targets="
            f"{[o.target_lang for o in oracles]}")
        return report

    # Drive the shared witness search once via the anchor (or first) oracle to
    # obtain the canonical C source + input; every target oracle reproduces the
    # identical C source from the same unit.
    seed = oracles[0].find_divergence(dict(unit))
    if seed.counterexample is None:
        report.reason = "no witness found for the unit"
        return report
    c_src = seed.counterexample.source_snippet
    argv = [str(v) for v in seed.counterexample.inputs.values()]
    report.inputs = dict(seed.counterexample.inputs)
    report.available = True

    # context: is the C source actually UB on this input? (UBSan trap)
    if harness.status.ubsan:
        san = harness.build_and_run_c(
            c_src, ["-O1", "-fsanitize=undefined", "-fno-sanitize-recover=all"], argv)
        report.c_ub_reachable = bool(san and san.ub_trapped)

    for o in oracles:
        ce = o.find_divergence(dict(unit)).counterexample
        # all oracles for a class share the C source; guard the invariant.
        if ce.source_snippet != c_src:
            report.reason = (f"C source mismatch across targets at "
                             f"{o.target_lang}; aborting comparison")
            report.available = False
            return report
        run = harness.build_and_run_target(ce.target_snippet, o.target_lang, argv)
        if run is None:
            report.reason = f"compile/run failed for {o.target_lang}"
            report.available = False
            return report
        report.observations.append(TargetObservation(
            o.target_lang, canonical_outcome(run, o.target_lang),
            run.returncode, run.stdout))

    tokens = [obs.token for obs in report.observations]
    counts = Counter(tokens)
    # majority = the most common token (ties broken by token order for determinism)
    top = max(sorted(counts), key=lambda t: counts[t])
    report.majority_token = top
    report.flagged = tuple(sorted(obs.target_lang for obs in report.observations
                                  if obs.token != top))
    if report.flagged:
        outliers = {obs.target_lang: obs.token for obs in report.observations
                    if obs.target_lang in report.flagged}
        report.reason = (f"{len(report.observations)} translations compared; "
                         f"majority={top!r}; outlier(s) {outliers}")
    else:
        report.reason = (f"all {len(report.observations)} translations agree "
                         f"({top!r}) — consistent")
    return report


# ── the curated sample the suite witnesses ───────────────────────────────────

def consistency_units() -> List[Tuple[str, Dict]]:
    """(class, unit) cases with >= 3 translations, spanning both outcomes:

      * ``shift_oob``   — on an out-of-range shift, Rust's ``wrapping_shl`` *masks*
        the shift amount (``1 << (32 mod 32) == 1``) while Go and Swift yield
        ``0``; three real compilers, one shared C source, and Rust is the lone
        flagged *outlier*;
      * ``div_by_zero`` — every safe target aborts deterministically, so the
        translations are *consistent* (all canonicalise to ``ABORT``).
    """
    return [
        ("shift_oob",
         {"kind": "shift", "width": 32, "value": 1,
          "source_lang": "c", "target_lang": "rust"}),
        ("div_by_zero",
         {"kind": "div", "width": 32, "a": "a", "b": "b",
          "source_lang": "c", "target_lang": "rust"}),
    ]
