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
    DivergenceOracle,
    OracleResult,
    OracleVerdict,
)
from .reexec import ReexecHarness, ToolchainStatus, toolchain_available

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

    If the unit declares ``source_lang``/``target_lang``, only oracles validated
    on exactly that pair are eligible — a unit declaring an unsupported pair
    (e.g. ``go``->``rust``) therefore matches *no* oracle and is honestly
    reported as ``NOT_COVERED`` rather than silently treated as the anchor pair.
    Units that omit the languages default to the anchor pair each oracle exposes.
    """
    src = unit.get("source_lang")
    tgt = unit.get("target_lang")

    def pair_ok(o: DivergenceOracle) -> bool:
        if src is not None and src != o.source_lang:
            return False
        if tgt is not None and tgt != o.target_lang:
            return False
        return True

    return [o for o in REGISTRY.values() if pair_ok(o) and o.applies_to(unit)]


def verify_unit(
    unit: Dict,
    harness: Optional[ReexecHarness] = None,
    *,
    confirm: bool = True,
    status: Optional[ToolchainStatus] = None,
) -> VerifyReport:
    """
    Run all applicable oracles and combine under the sound-for-divergence policy.

    When ``confirm`` is true and the toolchain is available, every symbolic
    divergence witness is re-executed; only confirmed witnesses yield a
    ``DIVERGENT`` verdict. Otherwise such a witness is reported as ``CANDIDATE``.
    """
    status = status or toolchain_available()
    tool_ok = status.full
    if confirm and tool_ok and harness is None:
        harness = ReexecHarness(status)

    applicable = applicable_oracles(unit)
    if not applicable:
        return VerifyReport(VerifyVerdict.NOT_COVERED, unit,
                            toolchain_available=tool_ok,
                            detail="no registered oracle applies")

    results: List[OracleResult] = []
    candidates: List[OracleResult] = []
    saw_unknown = False

    for oracle in applicable:
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
        )

    if saw_unknown:
        return VerifyReport(
            VerifyVerdict.UNKNOWN, unit, oracle_results=results,
            toolchain_available=tool_ok,
            detail="at least one oracle returned UNKNOWN",
        )

    covered = ", ".join(sorted({o.divergence_class for o in applicable}))
    return VerifyReport(
        VerifyVerdict.NO_DIVERGENCE_FOUND, unit, oracle_results=results,
        toolchain_available=tool_ok,
        detail=f"checked classes: [{covered}]",
    )
