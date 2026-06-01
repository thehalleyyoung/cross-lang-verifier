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
    #: "exploited"        — UB flips the observable value across opt levels.
    #: "trap_vs_defined"  — C is UB on a defined input while Rust is defined.
    confirmation_mode: str = "exploited"

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
                ce.source_snippet, ce.target_snippet, argv, ce.divergence_class)
        else:
            rr = harness.confirm_ub_divergence(
                ce.source_snippet, ce.target_snippet, argv, ce.divergence_class)
        result.reexec = rr
        if rr.available:
            ce.confirmed = rr.confirmed
            ce.source_observed = {k: v.stdout for k, v in rr.c_runs.items()}
            ce.target_observed = rr.rust_run.stdout if rr.rust_run else None
        return result


# ── registry ─────────────────────────────────────────────────────────────────

REGISTRY: Dict[str, DivergenceOracle] = {}


def register(oracle: DivergenceOracle) -> DivergenceOracle:
    if not oracle.divergence_class:
        raise ValueError("oracle must declare a divergence_class")
    REGISTRY[oracle.divergence_class] = oracle
    return oracle


def get_oracle(divergence_class: str) -> DivergenceOracle:
    return REGISTRY[divergence_class]


def list_oracles() -> List[str]:
    return sorted(REGISTRY.keys())
