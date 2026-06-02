"""
Divergence triage UX (100_STEPS step 65).

On a big migration the raw per-unit verdict list is too flat to act on: a reviewer
needs to know *what to look at first*.  This module ranks a list of
:class:`~ub_oracle.verify.VerifyReport` into a small number of **priority tiers**
so the highest-consequence, most-certain findings float to the top:

    Tier 0  CONFIRMED_CRITICAL   ground-truth-confirmed divergence, catalogue
                                 severity CRITICAL  (real UB / definite behavioral
                                 difference is reachable) — fix first.
    Tier 1  CONFIRMED_MODERATE   confirmed divergence, severity MODERATE.
    Tier 2  CONFIRMED_MINOR      confirmed divergence, severity MINOR.
    Tier 3  CANDIDATE            a *symbolic* witness that was not re-executed
                                 (a spec-gap to investigate, not yet a bug),
                                 sub-ordered by the underlying class severity.
    Tier 4  UNKNOWN              a solver abstained / timed out — needs attention
                                 but makes no claim.
    Tier 5  NOT_COVERED          no oracle understands the unit — a coverage gap.
    Tier 6  NO_DIVERGENCE        nothing fired in the covered classes
                                 (informational; NOT a proof of equivalence).

The ordering encodes the project's honesty contract: a *confirmed* divergence
always outranks a *symbolic candidate*, which always outranks an *abstention*.
Within a tier, items are ordered by catalogue severity then class then unit name
so the output is deterministic and screenshot-stable.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum
from typing import Dict, Iterable, List, Optional

from .catalogue import CATALOGUE, Severity
from .report import _class_of, pair_of
from .verify import VerifyReport, VerifyVerdict

#: numeric rank of each catalogue severity (smaller == more urgent).
_SEVERITY_RANK: Dict[Severity, int] = {
    Severity.CRITICAL: 0,
    Severity.MODERATE: 1,
    Severity.MINOR: 2,
}


class Tier(IntEnum):
    """Triage priority tiers, lowest value == most urgent."""

    CONFIRMED_CRITICAL = 0
    CONFIRMED_MODERATE = 1
    CONFIRMED_MINOR = 2
    CANDIDATE = 3
    UNKNOWN = 4
    NOT_COVERED = 5
    NO_DIVERGENCE = 6

    @property
    def label(self) -> str:
        return {
            Tier.CONFIRMED_CRITICAL: "confirmed-divergence (critical)",
            Tier.CONFIRMED_MODERATE: "confirmed-divergence (moderate)",
            Tier.CONFIRMED_MINOR: "confirmed-divergence (minor)",
            Tier.CANDIDATE: "candidate (symbolic, unconfirmed)",
            Tier.UNKNOWN: "unknown (solver abstained)",
            Tier.NOT_COVERED: "not-covered (no oracle applies)",
            Tier.NO_DIVERGENCE: "no-divergence-found (informational)",
        }[self]

    @property
    def actionable(self) -> bool:
        """Tiers a reviewer must actually look at (everything but clean/no-div)."""
        return self <= Tier.NOT_COVERED and self is not Tier.NO_DIVERGENCE


def severity_of_class(divergence_class: str) -> Optional[Severity]:
    """The catalogue severity of a divergence class, or ``None`` if unlisted."""
    entry = CATALOGUE.get(divergence_class)
    return entry.severity if entry is not None else None


def _severity_rank(divergence_class: str) -> int:
    sev = severity_of_class(divergence_class)
    # unknown classes sort *after* the three known severities but before nothing.
    return _SEVERITY_RANK.get(sev, len(_SEVERITY_RANK)) if sev else len(_SEVERITY_RANK)


@dataclass(frozen=True)
class TriageItem:
    """One ranked finding."""

    tier: Tier
    verdict: VerifyVerdict
    divergence_class: str
    severity: Optional[str]
    pair: str
    unit_label: str
    detail: str = ""

    @property
    def sort_key(self):
        # tier first, then severity within the tier, then class, then unit name —
        # fully deterministic.
        return (int(self.tier), _severity_rank(self.divergence_class),
                self.divergence_class, self.unit_label)

    def to_dict(self) -> Dict:
        return {
            "tier": int(self.tier),
            "tier_label": self.tier.label,
            "verdict": self.verdict.value,
            "divergence_class": self.divergence_class,
            "severity": self.severity,
            "pair": self.pair,
            "unit": self.unit_label,
            "detail": self.detail,
        }


def _tier_for(report: VerifyReport, cls: str) -> Tier:
    v = report.verdict
    if v is VerifyVerdict.DIVERGENT:
        sev = severity_of_class(cls)
        if sev is Severity.CRITICAL:
            return Tier.CONFIRMED_CRITICAL
        if sev is Severity.MODERATE:
            return Tier.CONFIRMED_MODERATE
        if sev is Severity.MINOR:
            return Tier.CONFIRMED_MINOR
        # confirmed but the class carries no catalogue severity: treat as critical
        # (a confirmed divergence is never less than the lowest confirmed tier).
        return Tier.CONFIRMED_CRITICAL
    if v is VerifyVerdict.CANDIDATE:
        return Tier.CANDIDATE
    if v is VerifyVerdict.UNKNOWN:
        return Tier.UNKNOWN
    if v is VerifyVerdict.NOT_COVERED:
        return Tier.NOT_COVERED
    return Tier.NO_DIVERGENCE


def _unit_label(report: VerifyReport, i: int) -> str:
    u = report.unit
    return str(u.get("name") or u.get("id") or u.get("kind") or f"unit[{i}]")


def triage_items(reports: Iterable[VerifyReport]) -> List[TriageItem]:
    """Rank reports into a deterministic, severity-ordered list of items."""
    items: List[TriageItem] = []
    for i, r in enumerate(reports):
        cls = (r.divergence.divergence_class
               if r.divergence is not None else _class_of(r))
        sev = severity_of_class(cls)
        items.append(TriageItem(
            tier=_tier_for(r, cls),
            verdict=r.verdict,
            divergence_class=cls,
            severity=sev.value if sev else None,
            pair=pair_of(r),
            unit_label=_unit_label(r, i),
            detail=r.detail,
        ))
    items.sort(key=lambda it: it.sort_key)
    return items


@dataclass
class TriageSummary:
    items: List[TriageItem] = field(default_factory=list)
    by_tier: Dict[int, int] = field(default_factory=dict)

    @property
    def total(self) -> int:
        return len(self.items)

    @property
    def actionable(self) -> int:
        return sum(1 for it in self.items if it.tier.actionable)

    @property
    def top_tier(self) -> Optional[Tier]:
        return self.items[0].tier if self.items else None

    def items_in(self, tier: Tier) -> List[TriageItem]:
        return [it for it in self.items if it.tier is tier]

    def to_dict(self) -> Dict:
        return {
            "total": self.total,
            "actionable": self.actionable,
            "top_tier": int(self.top_tier) if self.top_tier is not None else None,
            "by_tier": {int(k): v for k, v in sorted(self.by_tier.items())},
            "items": [it.to_dict() for it in self.items],
        }


def triage_reports(reports: Iterable[VerifyReport]) -> TriageSummary:
    """Build a :class:`TriageSummary` from a list of verify reports."""
    items = triage_items(reports)
    by_tier: Dict[int, int] = {}
    for it in items:
        by_tier[int(it.tier)] = by_tier.get(int(it.tier), 0) + 1
    return TriageSummary(items=items, by_tier=by_tier)


def render_triage(summary: TriageSummary, *, color=None, max_per_tier: int = 0) -> str:
    """A digestible, grouped triage view.

    ``max_per_tier`` (0 == unlimited) caps how many items are listed per tier so
    huge projects stay screenshot-friendly; the per-tier count is always shown in
    full.  ``color`` is an optional ``callable(code, text) -> str`` (the CLI's
    ``_Color``); when ``None`` the output is plain.
    """
    def paint(code: str, text: str) -> str:
        return color(code, text) if color is not None else text

    lines: List[str] = []
    lines.append(paint("1", "Triage") +
                 f" — {summary.total} unit(s); "
                 f"{summary.actionable} need attention")
    if summary.total == 0:
        return "\n".join(lines)

    tier_color = {
        Tier.CONFIRMED_CRITICAL: "31;1",
        Tier.CONFIRMED_MODERATE: "31",
        Tier.CONFIRMED_MINOR: "33",
        Tier.CANDIDATE: "33;1",
        Tier.UNKNOWN: "35",
        Tier.NOT_COVERED: "90",
        Tier.NO_DIVERGENCE: "32",
    }

    for tier in Tier:
        group = summary.items_in(tier)
        if not group:
            continue
        marker = "‼" if tier is Tier.CONFIRMED_CRITICAL else " "
        lines.append("")
        lines.append(f"{marker} " + paint(tier_color[tier], f"[{tier.label}]") +
                     f"  ({len(group)})")
        shown = group if max_per_tier <= 0 else group[:max_per_tier]
        for it in shown:
            sev = f" {it.severity}" if it.severity else ""
            lines.append(f"    {it.unit_label:<24} "
                         f"{paint('2', it.divergence_class)}{sev} "
                         f"{paint('2', '[' + it.pair + ']')}")
        if max_per_tier > 0 and len(group) > max_per_tier:
            lines.append(f"    … {len(group) - max_per_tier} more")
    return "\n".join(lines)
