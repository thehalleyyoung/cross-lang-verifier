"""
Head-to-head: this oracle vs. equal-budget differential testing (step 48).

The central empirical claim of the project is that **random/differential testing
with sanitizers misses the divergences this oracle finds**, because the inputs
that trigger them are a vanishing fraction of the input space. This module makes
that a measured, reproducible head-to-head rather than a slogan.

For a translation unit, the oracle produces a real `(C, Rust)` program pair (the
*same* pair a differential tester would exercise) that reads its inputs from
``argv``. We then:

* **oracle side** — find the divergence symbolically (Z3, deterministic) and
  confirm it by re-executing the real binaries on the witness; and
* **fuzzer side** — give an *equal-budget* differential tester the identical
  binaries: it draws ``trials`` uniform-random inputs from the natural input
  domain, runs the C build under UndefinedBehaviorSanitizer and the Rust build,
  and flags a divergence when the C run **traps on UB** or the two observable
  outputs **differ**.

The reported gap — units the oracle confirms but the equal-budget fuzzer never
hits — is the oracle's quantified advantage. The harness is deliberately *not*
rigged against fuzzing: for divergence classes whose triggering inputs are dense
(e.g. out-of-range shift amounts) the fuzzer finds them immediately, and the
report shows that too.

This harness needs a real toolchain, so it is exercised by the (skippable) test
suite rather than wired into the toolchain-free reproducible artifact.
"""

from __future__ import annotations

import os
import random
import tempfile
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from .plugin import REGISTRY, OracleVerdict
from .reexec import ReexecHarness, RunOutcome

# A per-variable sampling domain: ("int", lo, hi) or ("float", lo, hi).
Domain = Tuple[str, float, float]

_C_UBSAN_FLAGS = ["-O0", "-fsanitize=undefined", "-fno-sanitize-recover=all"]


@dataclass
class FuzzUnit:
    """A labelled unit + the natural input domain a differential tester samples."""
    name: str
    unit: Dict
    #: divergence-class key whose oracle owns this unit.
    divergence_class: str
    #: domain per input variable, keyed by the same names the oracle emits.
    domains: Dict[str, Domain]


@dataclass
class HeadToHeadRow:
    name: str
    divergence_class: str
    oracle_confirmed: bool
    fuzz_trials: int
    fuzz_hits: int
    fuzz_first_hit_trial: Optional[int]
    seed: int
    detail: str = ""

    @property
    def fuzz_found(self) -> bool:
        return self.fuzz_hits > 0

    @property
    def is_false_negative_gap(self) -> bool:
        """The oracle confirmed a divergence the equal-budget fuzzer never hit."""
        return self.oracle_confirmed and not self.fuzz_found

    def as_dict(self) -> Dict:
        return {
            "name": self.name,
            "divergence_class": self.divergence_class,
            "oracle_confirmed": self.oracle_confirmed,
            "fuzz_trials": self.fuzz_trials,
            "fuzz_hits": self.fuzz_hits,
            "fuzz_first_hit_trial": self.fuzz_first_hit_trial,
            "fuzz_found": self.fuzz_found,
            "false_negative_gap": self.is_false_negative_gap,
            "seed": self.seed,
            "detail": self.detail,
        }


def _sample(rng: random.Random, dom: Domain) -> str:
    kind, lo, hi = dom
    if kind == "int":
        return str(rng.randint(int(lo), int(hi)))
    if kind == "float":
        return repr(rng.uniform(lo, hi))
    raise ValueError(f"unknown domain kind {kind!r}")


def _trial_diverges(harness: ReexecHarness, c_bin: str, rust_bin: str,
                    argv: List[str]) -> bool:
    """One differential-test trial: does the sanitized C vs Rust pair diverge?"""
    c: RunOutcome = harness._run([c_bin, *argv])
    if c.ub_trapped:
        return True  # sanitizer caught real UB — a divergence from defined Rust
    r: RunOutcome = harness._run([rust_bin, *argv])
    if c.returncode != 0 or not r.rust_outcome_defined:
        return False  # cannot compare cleanly on this input
    return c.stdout != r.stdout


def differential_fuzz(harness: ReexecHarness, fu: FuzzUnit, *,
                      trials: int, seed: int) -> Tuple[int, Optional[int]]:
    """Compile the unit's real pair once and fuzz it for ``trials`` inputs.

    Returns ``(hits, first_hit_trial)``. Stops sampling at the first hit (a
    differential tester only needs one counterexample), so dense-UB classes
    finish immediately while sparse ones exhaust the budget.
    """
    oracle = REGISTRY[fu.divergence_class]
    res = oracle.find_divergence(fu.unit)
    if res.verdict is not OracleVerdict.DIVERGENT or res.counterexample is None:
        return (0, None)
    ce = res.counterexample
    order = list(ce.inputs.keys())  # argv order matches inputs.values() order

    rng = random.Random(seed)
    with tempfile.TemporaryDirectory() as d:
        c_bin = harness._compile_c(ce.source_snippet, _C_UBSAN_FLAGS, d, "c_fuzz")
        rust_bin = harness._compile_target(ce.target_snippet, ce.target_lang,
                                           d, "rs_fuzz")
        if not (c_bin and rust_bin):
            return (0, None)
        hits = 0
        first = None
        for t in range(1, trials + 1):
            argv = [_sample(rng, fu.domains[name]) for name in order]
            if _trial_diverges(harness, c_bin, rust_bin, argv):
                hits += 1
                if first is None:
                    first = t
                break
    return (hits, first)


def head_to_head(units: List[FuzzUnit], *, trials: int = 2000, seed: int = 0,
                 harness: Optional[ReexecHarness] = None) -> Dict:
    """Run the oracle and an equal-budget differential tester on each unit."""
    harness = harness or ReexecHarness()
    rows: List[HeadToHeadRow] = []
    for fu in units:
        oracle = REGISTRY[fu.divergence_class]
        res = oracle.find_divergence(fu.unit)
        confirmed = False
        detail = ""
        if res.verdict is OracleVerdict.DIVERGENT:
            res = oracle.confirm(res, harness)
            confirmed = bool(res.reexec and res.reexec.available
                             and res.reexec.confirmed)
            detail = res.reexec.reason if res.reexec else ""
        hits, first = differential_fuzz(harness, fu, trials=trials, seed=seed)
        rows.append(HeadToHeadRow(
            name=fu.name, divergence_class=fu.divergence_class,
            oracle_confirmed=confirmed, fuzz_trials=trials, fuzz_hits=hits,
            fuzz_first_hit_trial=first, seed=seed, detail=detail))

    gap = [r.name for r in rows if r.is_false_negative_gap]
    return {
        "trials_per_unit": trials,
        "seed": seed,
        "num_units": len(rows),
        "oracle_confirmed": sum(1 for r in rows if r.oracle_confirmed),
        "fuzzer_found": sum(1 for r in rows if r.fuzz_found),
        "false_negative_gap_units": sorted(gap),
        "false_negative_gap": len(gap),
        "rows": [r.as_dict() for r in rows],
    }


def default_units() -> List[FuzzUnit]:
    """A curated two-sided demonstration set (sparse-UB gap + dense-UB parity).

    * signed overflow (w32, w64): the UB inputs are ~1 / 2^width of the domain,
      so an equal-budget uniform fuzzer essentially never hits them — the gap.
    * out-of-range shift: roughly all shift amounts in the natural domain are
      out of range, so the fuzzer finds it on the first trial — parity, proving
      the harness is not rigged against differential testing.
    """
    i32 = ("int", -(2 ** 31), 2 ** 31 - 1)
    i64 = ("int", -(2 ** 63), 2 ** 63 - 1)
    shift_amt = ("int", 0, 2 ** 16 - 1)  # non-negative so Rust's u32 parse is valid
    return [
        FuzzUnit("ovf_add1_w32",
                 {"kind": "binop_const", "op": "add", "const": 1, "width": 32,
                  "var": "x", "signed": True, "probe": "signed_overflow"},
                 "signed_overflow", {"x": i32}),
        FuzzUnit("ovf_add1_w64",
                 {"kind": "binop_const", "op": "add", "const": 1, "width": 64,
                  "var": "x", "signed": True, "probe": "signed_overflow"},
                 "signed_overflow", {"x": i64}),
        FuzzUnit("shift_w32",
                 {"kind": "shift", "width": 32, "probe": "shift_oob"},
                 "shift_oob", {"x": i32, "s": shift_amt}),
    ]
