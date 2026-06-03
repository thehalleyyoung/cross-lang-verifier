"""
Internal red-team: try to make the oracle falsely look "equivalent" (step 84).

The flagship soundness claim of this project is *sound-for-divergence*: the tool
must **never** silently bless a genuinely divergent translation as
equivalence-looking.  The verifier has no ``EQUIVALENT`` verdict, so the single
dangerous failure mode is returning :data:`VerifyVerdict.NO_DIVERGENCE_FOUND` for
a unit that is *actually* divergent — that is the closest thing to a false "looks
equivalent", and it is exactly what a soundness hole would produce (an applicable
oracle that fails to find the witness it should, or no oracle applying where one
should).

This module is an *adversary*.  For **every** registered oracle on **every**
supported language pair it synthesises a battery of genuinely-divergent units —
the canonical witness plus semantics-preserving *adversarial mutations* (renamed
variables, an explicit conflicting-but-valid ``probe``, alternate bit widths,
alternate overflowing constants) — and asserts the verifier's verdict stays in
the **sound** set:

* with a real toolchain, every case must be ``DIVERGENT`` (confirmed by
  re-execution), and
* in *no* configuration may a genuinely-divergent case be reported
  ``NO_DIVERGENCE_FOUND``.

A case that lands on ``NO_DIVERGENCE_FOUND`` is a **soundness breach** and is
reported as such.  ``NOT_COVERED`` and ``CANDIDATE`` are *not* breaches: they make
no equivalence claim (``CANDIDATE`` happens only when the toolchain is absent).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

from . import oracles as _oracles  # noqa: F401  (registers all pairs)
from .plugin import ALL_ORACLES, DivergenceOracle
from .reexec import ReexecHarness, ToolchainStatus, toolchain_available
from .regression_matrix import CANONICAL_UNITS, canonical_unit_for
from .verify import verify_unit, VerifyVerdict
from .cache import toolchain_provenance


# Verdicts that make NO equivalence claim and are therefore sound for a
# genuinely-divergent input. NO_DIVERGENCE_FOUND is the one forbidden verdict.
_SOUND_FOR_DIVERGENT = frozenset({
    VerifyVerdict.DIVERGENT,
    VerifyVerdict.CANDIDATE,
    VerifyVerdict.UNKNOWN,
    VerifyVerdict.NOT_COVERED,
})


def _mutations_for(oracle: DivergenceOracle) -> List[Dict]:
    """Semantics-preserving adversarial variants of ``oracle``'s known witness.

    Each returned unit is *still genuinely divergent* — the mutation only renames
    things, pins the matching probe, or moves to another valid width/constant — so
    the oracle has no excuse to miss it. The mutations probe the seams an attacker
    would poke at to slip a divergence past the matcher.
    """
    cls = oracle.divergence_class
    base = canonical_unit_for(oracle)
    out: List[Dict] = [dict(base)]

    # (1) explicit, *matching* probe — must not change the verdict.
    out.append(dict(base, probe=cls))

    # (2) renamed program variables — pure alpha-renaming, still divergent.
    renamed = dict(base)
    if "var" in renamed:
        renamed["var"] = "zz"
    if base.get("kind") == "shift":
        renamed["shift_var"] = "amt"
    if base.get("kind") in ("div", "rem"):
        renamed["a"], renamed["b"] = "p", "q"
    if renamed != base:
        out.append(renamed)

    # (3) alternate bit width for the width-parametric integer classes.
    if cls in ("signed_overflow", "shift_oob", "div_by_zero", "intmin_div_neg1"):
        other = 64 if base.get("width", 32) == 32 else 32
        out.append(dict(base, width=other))

    # (4) a *different* overflowing constant for signed overflow — the witness
    #     search must generalise beyond INT_MAX, not pattern-match one literal.
    if cls == "signed_overflow":
        out.append(dict(base, const=1234567))
        out.append(dict(base, op="sub", const=2147483647))

    # de-duplicate while preserving order.
    seen = set()
    uniq = []
    for u in out:
        key = tuple(sorted((k, repr(v)) for k, v in u.items()))
        if key not in seen:
            seen.add(key)
            uniq.append(u)
    return uniq


@dataclass
class RedTeamCase:
    label: str
    divergence_class: str
    source_lang: str
    target_lang: str
    unit: Dict
    verdict: str = ""
    breach: bool = False
    detail: str = ""

    def to_dict(self) -> Dict:
        return {
            "label": self.label,
            "divergence_class": self.divergence_class,
            "source_lang": self.source_lang,
            "target_lang": self.target_lang,
            "verdict": self.verdict,
            "breach": self.breach,
        }


@dataclass
class RedTeamReport:
    cases: List[RedTeamCase] = field(default_factory=list)
    toolchain_full: bool = False
    toolchain_provenance: Dict[str, object] = field(default_factory=dict)

    @property
    def breaches(self) -> List[RedTeamCase]:
        return [c for c in self.cases if c.breach]

    @property
    def n_cases(self) -> int:
        return len(self.cases)

    @property
    def n_confirmed_divergent(self) -> int:
        return sum(1 for c in self.cases if c.verdict == VerifyVerdict.DIVERGENT.value)

    @property
    def sound(self) -> bool:
        return not self.breaches

    def to_dict(self) -> Dict:
        provenance = dict(self.toolchain_provenance)
        out = {
            "n_cases": self.n_cases,
            "n_confirmed_divergent": self.n_confirmed_divergent,
            "n_breaches": len(self.breaches),
            "sound": self.sound,
            "toolchain_full": self.toolchain_full,
            "cases": [c.to_dict() for c in
                      sorted(self.cases, key=lambda c: c.label)],
        }
        if provenance:
            out.update({
                "schema": "redteam-attack/v2",
                "toolchain_fingerprint": provenance.get("fingerprint", {}),
                "toolchain_provenance": provenance,
            })
        return out


def build_cases() -> List[RedTeamCase]:
    """Every adversarial divergent case across every oracle/pair (deterministic)."""
    cases: List[RedTeamCase] = []
    for oracle in sorted(ALL_ORACLES,
                         key=lambda o: (o.source_lang, o.target_lang,
                                        o.divergence_class)):
        if oracle.divergence_class not in CANONICAL_UNITS:
            continue
        for i, unit in enumerate(_mutations_for(oracle)):
            cases.append(RedTeamCase(
                label=f"{oracle.source_lang}->{oracle.target_lang}:"
                      f"{oracle.divergence_class}#{i}",
                divergence_class=oracle.divergence_class,
                source_lang=oracle.source_lang,
                target_lang=oracle.target_lang,
                unit=unit,
            ))
    return cases


def run_redteam(harness: Optional[ReexecHarness] = None,
                status: Optional[ToolchainStatus] = None,
                confirm: bool = True) -> RedTeamReport:
    """Run every adversarial case and flag any soundness breach."""
    status = status or toolchain_available()
    provenance = toolchain_provenance(status) if confirm else {}
    report = RedTeamReport(toolchain_full=status.full,
                           toolchain_provenance=provenance)
    for case in build_cases():
        rep = verify_unit(case.unit, harness=harness, confirm=confirm,
                          status=status)
        case.verdict = rep.verdict.value
        # A genuinely-divergent unit reported NO_DIVERGENCE_FOUND is the breach.
        case.breach = rep.verdict not in _SOUND_FOR_DIVERGENT
        case.detail = rep.detail
        report.cases.append(case)
    return report
