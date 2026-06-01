"""
The divergence-oracle plugin interface + registry (100_STEPS step 11).

This is the *generality backbone*: every divergence class (signed overflow,
shift-out-of-range, OOB, ...) is implemented as an independent ``DivergenceOracle``
plugin behind one uniform contract, so new classes — and eventually new language
pairs — are additive rather than forks of the engine.

A plugin's job is to: decide whether it *applies* to a translation unit, search
(symbolically) for a divergence witness, package it as a re-executable
:class:`~src.ub_oracle.replay.Counterexample`, and optionally confirm it via the
ground-truth re-execution harness.
"""

from __future__ import annotations

import abc
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional

from .replay import Counterexample
from .reexec import ReexecHarness, ReexecResult


class OracleVerdict(Enum):
    DIVERGENT = "divergent"            # a confirmed (or solver-found) divergence witness exists
    NO_DIVERGENCE_FOUND = "no_divergence_found"  # none found within the search bound
    NOT_APPLICABLE = "not_applicable"  # this oracle does not apply to the unit
    UNKNOWN = "unknown"                # solver returned unknown / timed out

    def __str__(self) -> str:  # pragma: no cover - trivial
        return self.value


@dataclass
class OracleResult:
    verdict: OracleVerdict
    divergence_class: str
    counterexample: Optional[Counterexample] = None
    reexec: Optional[ReexecResult] = None
    detail: str = ""

    @property
    def is_divergent(self) -> bool:
        return self.verdict is OracleVerdict.DIVERGENT


class DivergenceOracle(abc.ABC):
    """Uniform contract for a divergence-class plugin."""

    #: stable catalogue key, e.g. "signed_overflow"
    divergence_class: str = ""
    #: anchor language pair this plugin is validated on
    source_lang: str = "c"
    target_lang: str = "rust"
    #: how the ground-truth harness should confirm this class.
    #: "exploited"           — UB flips the observable value across opt levels.
    #: "trap_vs_defined"     — C is UB on a defined input while Rust is defined.
    #: "optimizer_exploited" — same C source yields different output under two
    #:                         conforming compilations (no sanitizer can trap it)
    #:                         while Rust is defined & deterministic.
    confirmation_mode: str = "exploited"
    #: for "optimizer_exploited" oracles, the pair of C flag-sets whose
    #: disagreement evidences the divergence (e.g. FP contraction off vs fast).
    #: None falls back to the harness default (-O0 vs -O2 -fstrict-aliasing).
    optimizer_flag_variants: Optional[tuple] = None

    @abc.abstractmethod
    def applies_to(self, unit: Dict) -> bool:
        """Whether this oracle is relevant to the given translation unit."""

    @abc.abstractmethod
    def find_divergence(self, unit: Dict) -> OracleResult:
        """Search for a divergence witness and package it as a counterexample."""

    def confirm(self, result: OracleResult,
                harness: Optional[ReexecHarness] = None) -> OracleResult:
        """Confirm a found counterexample via the ground-truth harness, if possible."""
        if result.counterexample is None or not result.is_divergent:
            return result
        ce = result.counterexample
        if not (ce.source_snippet and ce.target_snippet):
            return result
        harness = harness or ReexecHarness()
        argv = [str(v) for v in ce.inputs.values()]
        if self.confirmation_mode == "trap_vs_defined":
            rr = harness.confirm_trap_vs_defined(
                ce.source_snippet, ce.target_snippet, argv, ce.divergence_class,
                target_lang=self.target_lang)
        elif self.confirmation_mode == "optimizer_exploited":
            flags = self.optimizer_flag_variants
            if flags is not None:
                rr = harness.confirm_optimizer_exploited(
                    ce.source_snippet, ce.target_snippet, argv, ce.divergence_class,
                    c_flags_a=list(flags[0]), c_flags_b=list(flags[1]),
                    target_lang=self.target_lang)
            else:
                rr = harness.confirm_optimizer_exploited(
                    ce.source_snippet, ce.target_snippet, argv, ce.divergence_class,
                    target_lang=self.target_lang)
        else:
            rr = harness.confirm_ub_divergence(
                ce.source_snippet, ce.target_snippet, argv, ce.divergence_class,
                target_lang=self.target_lang)
        result.reexec = rr
        if rr.available:
            ce.confirmed = rr.confirmed
            ce.source_observed = {k: v.stdout for k, v in rr.c_runs.items()}
            ce.target_observed = rr.rust_run.stdout if rr.rust_run else None
        return result


# ── registry ─────────────────────────────────────────────────────────────────

#: Anchor (C->Rust) index, keyed by divergence_class. Preserved verbatim so the
#: precision/recall, ablation and head-to-head harnesses keep their exact
#: C->Rust semantics. Only the anchor pair lands here.
REGISTRY: Dict[str, DivergenceOracle] = {}

#: Every registered oracle across *all* language pairs. This is the
#: pair-agnostic backbone the verifier consults so that adding a second target
#: language (C->Go, ...) is additive rather than a fork of the engine.
ALL_ORACLES: List[DivergenceOracle] = []

#: anchor language pair the legacy REGISTRY indexes
ANCHOR_PAIR = ("c", "rust")


def register(oracle: DivergenceOracle) -> DivergenceOracle:
    if not oracle.divergence_class:
        raise ValueError("oracle must declare a divergence_class")
    ALL_ORACLES.append(oracle)
    if (oracle.source_lang, oracle.target_lang) == ANCHOR_PAIR:
        REGISTRY[oracle.divergence_class] = oracle
    return oracle


def get_oracle(divergence_class: str) -> DivergenceOracle:
    return REGISTRY[divergence_class]


def oracles_for(source_lang: Optional[str] = None,
                target_lang: Optional[str] = None,
                divergence_class: Optional[str] = None) -> List[DivergenceOracle]:
    """All registered oracles matching the given (optional) filters."""
    out = []
    for o in ALL_ORACLES:
        if source_lang is not None and o.source_lang != source_lang:
            continue
        if target_lang is not None and o.target_lang != target_lang:
            continue
        if divergence_class is not None and o.divergence_class != divergence_class:
            continue
        out.append(o)
    return out


def get_oracle_for(divergence_class: str, source_lang: str = "c",
                   target_lang: str = "rust") -> DivergenceOracle:
    matches = oracles_for(source_lang, target_lang, divergence_class)
    if not matches:
        raise KeyError(
            f"no oracle for {source_lang}->{target_lang}:{divergence_class}")
    return matches[0]


def language_pairs() -> List[tuple]:
    """The distinct (source_lang, target_lang) pairs that have oracles."""
    seen = []
    for o in ALL_ORACLES:
        pair = (o.source_lang, o.target_lang)
        if pair not in seen:
            seen.append(pair)
    return seen


def list_oracles() -> List[str]:
    return sorted(REGISTRY.keys())
