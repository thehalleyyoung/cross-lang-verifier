"""
Sound-for-divergence multi-oracle entry point (100_STEPS steps 5 & 33).

``verify_unit`` runs *every applicable* divergence oracle against a translation
unit and combines their results under a deliberately conservative soundness
policy:

* This tool is an **oracle for divergence**, not a proof of equivalence. It is
  *sound for divergence* (Step 5): it only ever returns
  :data:`VerifyVerdict.DIVERGENT` when an actual divergence has been **confirmed
  by ground-truth re-execution** of real compiled C and Rust. A merely symbolic
  witness that cannot be re-executed (no toolchain) is downgraded to
  ``CANDIDATE`` — never asserted as a divergence.

* It is **loud about the unknown** (Step 33): the absence of a found divergence
  is reported as :data:`VerifyVerdict.NO_DIVERGENCE_FOUND` and **never** as
  "equivalent". A solver timeout becomes :data:`VerifyVerdict.UNKNOWN`, and a
  unit no oracle understands becomes :data:`VerifyVerdict.NOT_COVERED`. Only
  ``DIVERGENT`` is a positive, ground-truth-backed claim; everything else
  explicitly disclaims any equivalence guarantee.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional

from .plugin import (
    REGISTRY,
    ALL_ORACLES,
    DivergenceOracle,
    OracleResult,
    OracleVerdict,
)
from .reexec import ReexecHarness, ToolchainStatus, toolchain_available
from . import abstract_interp as _ai

# import the oracles package so the registry is populated on use.
from . import oracles as _oracles  # noqa: F401


class VerifyVerdict(Enum):
    #: a divergence was found AND confirmed by ground-truth re-execution.
    DIVERGENT = "divergent"
    #: a symbolic divergence witness exists but could not be re-executed
    #: (e.g. toolchain unavailable); NOT asserted as a divergence.
    CANDIDATE = "candidate"
    #: oracles applied but found no divergence within bounds.
    #: This is NOT a proof of equivalence.
    NO_DIVERGENCE_FOUND = "no_divergence_found"
    #: at least one applicable oracle returned unknown (e.g. solver timeout).
    UNKNOWN = "unknown"
    #: no registered oracle applies to this unit; nothing can be said.
    NOT_COVERED = "not_covered"

    def __str__(self) -> str:  # pragma: no cover - trivial
        return self.value

    @property
    def claims_divergence(self) -> bool:
        """Only DIVERGENT is a positive, ground-truth-backed claim."""
        return self is VerifyVerdict.DIVERGENT

    @property
    def claims_equivalence(self) -> bool:
        """This tool never claims equivalence."""
        return False


@dataclass
class VerifyReport:
    verdict: VerifyVerdict
    unit: Dict
    #: per-oracle results that were produced (only applicable oracles).
    oracle_results: List[OracleResult] = field(default_factory=list)
    #: the confirmed divergence, if any.
    divergence: Optional[OracleResult] = None
    toolchain_available: bool = False
    detail: str = ""
    #: classes the abstract-interpretation pre-pass soundly discharged (proved
    #: their UB unreachable under the unit's declared range) before any SMT call.
    prepass_pruned: List[str] = field(default_factory=list)

    @property
    def is_sound_claim(self) -> bool:
        return self.verdict is VerifyVerdict.DIVERGENT

    def banner(self) -> str:
        """A human-facing, *loud* one-line summary."""
        if self.verdict is VerifyVerdict.DIVERGENT:
            cls = self.divergence.divergence_class if self.divergence else "?"
            return f"DIVERGENT [{cls}] — confirmed by re-execution. {self.detail}"
        if self.verdict is VerifyVerdict.CANDIDATE:
            return ("CANDIDATE — symbolic witness found but NOT re-executed; "
                    "this is not a confirmed divergence. " + self.detail)
        if self.verdict is VerifyVerdict.NO_DIVERGENCE_FOUND:
            return ("NO DIVERGENCE FOUND — this is NOT a proof of equivalence; "
                    "only the covered classes were checked. " + self.detail)
        if self.verdict is VerifyVerdict.UNKNOWN:
            return ("UNKNOWN — a solver abstained; no claim is made. " + self.detail)
        return ("NOT COVERED — no oracle understands this unit; no claim is made. "
                + self.detail)


def applicable_oracles(unit: Dict) -> List[DivergenceOracle]:
    """Oracles that apply to ``unit``, respecting any declared language pair.

    A unit may declare ``source_lang``/``target_lang``; oracles validated on a
    *different* pair are never eligible, so a unit declaring an unsupported pair
    (e.g. ``go``->``rust``) honestly matches *no* oracle and is reported as
    ``NOT_COVERED`` rather than silently treated as the anchor pair. Units that
    omit the languages default to the C->Rust anchor pair, so the legacy
    benchmark/verify behaviour is preserved exactly while a unit that *opts in*
    to a second target (``target_lang: "go"``) is routed to that pair's oracles.
    """
    src = unit.get("source_lang") or "c"
    tgt = unit.get("target_lang") or "rust"

    return [o for o in ALL_ORACLES
            if o.source_lang == src and o.target_lang == tgt
            and o.applies_to(unit)]


def verify_unit(
    unit: Dict,
    harness: Optional[ReexecHarness] = None,
    *,
    confirm: bool = True,
    status: Optional[ToolchainStatus] = None,
    prepass: bool = True,
) -> VerifyReport:
    """
    Run all applicable oracles and combine under the sound-for-divergence policy.

    When ``confirm`` is true and the toolchain is available, every symbolic
    divergence witness is re-executed; only confirmed witnesses yield a
    ``DIVERGENT`` verdict. Otherwise such a witness is reported as ``CANDIDATE``.

    When ``prepass`` is true (default) an interval-domain abstract-interpretation
    pre-pass runs first: for any class it can *prove* has no reachable UB under
    the unit's declared operating range, the corresponding oracle's SMT search is
    skipped entirely (the class is recorded in ``prepass_pruned``). This is a
    sound accelerator — it only ever discharges a class as no-divergence, never
    asserts one — so the verdict is identical to ``prepass=False`` while avoiding
    solver calls on obviously-equivalent fragments.
    """
    status = status or toolchain_available()
    tgt_lang = unit.get("target_lang") or "rust"
    tool_ok = status.full_for(tgt_lang)
    if confirm and tool_ok and harness is None:
        harness = ReexecHarness(status)

    applicable = applicable_oracles(unit)
    if not applicable:
        return VerifyReport(VerifyVerdict.NOT_COVERED, unit,
                            toolchain_available=tool_ok,
                            detail="no registered oracle applies")

    pruned: Dict[str, str] = {}
    if prepass:
        try:
            for cls, res in _ai.prunable_classes(unit).items():
                pruned[cls] = res.reason
        except (ValueError, KeyError):
            pruned = {}  # malformed range etc.: fall back to the full search.

    results: List[OracleResult] = []
    candidates: List[OracleResult] = []
    saw_unknown = False

    for oracle in applicable:
        if oracle.divergence_class in pruned:
            results.append(OracleResult(
                OracleVerdict.NO_DIVERGENCE_FOUND, oracle.divergence_class,
                detail=f"pruned by abstract-interpretation pre-pass: "
                       f"{pruned[oracle.divergence_class]}"))
            continue
        res = oracle.find_divergence(unit)
        if res.verdict is OracleVerdict.DIVERGENT:
            if confirm and tool_ok:
                res = oracle.confirm(res, harness)
                if res.reexec is not None and res.reexec.available and res.reexec.confirmed:
                    results.append(res)
                    return VerifyReport(
                        VerifyVerdict.DIVERGENT, unit,
                        oracle_results=results + candidates,
                        divergence=res,
                        toolchain_available=tool_ok,
                        detail=res.reexec.reason,
                        prepass_pruned=list(pruned),
                    )
                # symbolic witness that did not confirm: keep as candidate, but
                # do NOT assert divergence (soundness).
                candidates.append(res)
            else:
                candidates.append(res)
        elif res.verdict is OracleVerdict.UNKNOWN:
            saw_unknown = True
            results.append(res)
        else:
            results.append(res)

    if candidates:
        cls = ", ".join(sorted({c.divergence_class for c in candidates}))
        return VerifyReport(
            VerifyVerdict.CANDIDATE, unit,
            oracle_results=results + candidates,
            toolchain_available=tool_ok,
            detail=(f"symbolic witness(es) for [{cls}] not re-executed "
                    f"(toolchain_available={tool_ok}, confirm={confirm})"),
            prepass_pruned=list(pruned),
        )

    if saw_unknown:
        return VerifyReport(
            VerifyVerdict.UNKNOWN, unit, oracle_results=results,
            toolchain_available=tool_ok,
            detail="at least one oracle returned UNKNOWN",
            prepass_pruned=list(pruned),
        )

    covered = ", ".join(sorted({o.divergence_class for o in applicable}))
    detail = f"checked classes: [{covered}]"
    if pruned:
        detail += (f"; {len(pruned)} class(es) discharged by abstract-"
                   f"interpretation pre-pass without SMT")
    return VerifyReport(
        VerifyVerdict.NO_DIVERGENCE_FOUND, unit, oracle_results=results,
        toolchain_available=tool_ok,
        detail=detail,
        prepass_pruned=list(pruned),
    )
