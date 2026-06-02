"""
Config + suppression files (100_STEPS step 67).

To adopt a divergence checker on a *large existing* migration, a team needs to
**baseline** the divergences they already know about — exactly like a linter's
suppression / baseline file — so CI fails only on *new* divergences while the
known-accepted ones stay visible but non-blocking.

A suppression file is JSON::

    {
      "version": 1,
      "suppressions": [
        {"divergence_class": "signed_overflow", "pair": "c->rust",
         "unit": "checksum_*", "reason": "release wrap is intended here"},
        {"fingerprint": "9f3a...", "reason": "triaged 2026-05, accepted"}
      ]
    }

A rule matches a report iff **every field it specifies** matches (an empty rule
matches every finding, like a bare ``# noqa`` — allowed but loud).  Fields:

* ``divergence_class`` — exact class key (e.g. ``signed_overflow``).
* ``pair``             — exact language pair (e.g. ``c->rust``).
* ``unit``             — ``fnmatch`` glob over the unit's name/id/kind.
* ``fingerprint``      — the SARIF ``partialFingerprint`` (stable per unit+class).
* ``reason``           — required free text (a baseline without a reason is a
                         silent suppression; we *require* it).
* ``expires``          — optional ISO ``YYYY-MM-DD``; an expired rule never
                         matches and is surfaced as a warning.

Only **findings** (``DIVERGENT`` / ``CANDIDATE``) are suppressible; abstentions
and clean results are never silently hidden.  A suppressed ``DIVERGENT`` is kept
in the output and counted in its own ``suppressed`` bucket, but does **not** trip
the CI fail gate — that is the whole point of a baseline.
"""

from __future__ import annotations

import datetime as _dt
import fnmatch
import json
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional

from .report import _class_of, _fingerprint, pair_of
from .verify import VerifyReport, VerifyVerdict

SUPPRESSION_VERSION = 1

#: verdicts that represent an actual *finding* (and so can be baselined).
_SUPPRESSIBLE = {VerifyVerdict.DIVERGENT, VerifyVerdict.CANDIDATE}


def _unit_label(report: VerifyReport) -> str:
    u = report.unit
    return str(u.get("name") or u.get("id") or u.get("kind") or "unit")


def finding_class(report: VerifyReport) -> str:
    return (report.divergence.divergence_class
            if report.divergence is not None else _class_of(report))


def fingerprint_of(report: VerifyReport) -> str:
    """The stable SARIF fingerprint of a report's finding (unit+class hash)."""
    return _fingerprint(report, finding_class(report))


@dataclass(frozen=True)
class Suppression:
    """One baseline / suppression rule."""

    reason: str
    divergence_class: Optional[str] = None
    pair: Optional[str] = None
    unit: Optional[str] = None
    fingerprint: Optional[str] = None
    expires: Optional[str] = None

    def is_empty_match(self) -> bool:
        """True iff the rule constrains nothing (matches every finding)."""
        return not any((self.divergence_class, self.pair, self.unit,
                        self.fingerprint))

    def expired(self, today: Optional[_dt.date] = None) -> bool:
        if not self.expires:
            return False
        today = today or _dt.date.today()
        try:
            exp = _dt.date.fromisoformat(self.expires)
        except ValueError:
            # a malformed date is treated as already-expired (fail safe: do not
            # silently suppress on a typo).
            return True
        return today > exp

    def matches(self, report: VerifyReport,
                today: Optional[_dt.date] = None) -> bool:
        if report.verdict not in _SUPPRESSIBLE:
            return False
        if self.expired(today):
            return False
        if self.divergence_class is not None and \
                self.divergence_class != finding_class(report):
            return False
        if self.pair is not None and self.pair != pair_of(report):
            return False
        if self.unit is not None and \
                not fnmatch.fnmatchcase(_unit_label(report), self.unit):
            return False
        if self.fingerprint is not None and \
                self.fingerprint != fingerprint_of(report):
            return False
        return True

    def to_dict(self) -> Dict:
        d: Dict[str, str] = {"reason": self.reason}
        for k in ("divergence_class", "pair", "unit", "fingerprint", "expires"):
            v = getattr(self, k)
            if v is not None:
                d[k] = v
        return d

    @staticmethod
    def from_dict(d: Dict) -> "Suppression":
        if not isinstance(d, dict):
            raise ValueError("each suppression must be a JSON object")
        reason = d.get("reason")
        if not reason or not str(reason).strip():
            raise ValueError("each suppression must carry a non-empty 'reason'")
        return Suppression(
            reason=str(reason),
            divergence_class=d.get("divergence_class"),
            pair=d.get("pair"),
            unit=d.get("unit"),
            fingerprint=d.get("fingerprint"),
            expires=d.get("expires"),
        )


def load_suppressions(path: str) -> List[Suppression]:
    """Parse a suppression file; raises ``ValueError`` on a malformed file."""
    with open(path, "r", encoding="utf-8") as fh:
        data = json.load(fh)
    if isinstance(data, list):
        rules = data
    elif isinstance(data, dict):
        rules = data.get("suppressions")
        if rules is None:
            raise ValueError("suppression file object needs a 'suppressions' array")
    else:
        raise ValueError("suppression file must be a JSON object or array")
    if not isinstance(rules, list):
        raise ValueError("'suppressions' must be an array")
    return [Suppression.from_dict(r) for r in rules]


@dataclass
class SuppressionOutcome:
    """A report paired with whether (and why) it was suppressed."""

    report: VerifyReport
    suppressed: bool = False
    rule: Optional[Suppression] = None

    @property
    def effective_finding(self) -> bool:
        """A *blocking* finding: a non-suppressed DIVERGENT/CANDIDATE."""
        return (self.report.verdict in _SUPPRESSIBLE) and not self.suppressed


@dataclass
class SuppressionResult:
    outcomes: List[SuppressionOutcome] = field(default_factory=list)
    expired_rules: List[Suppression] = field(default_factory=list)
    unused_rules: List[Suppression] = field(default_factory=list)
    empty_rules: List[Suppression] = field(default_factory=list)

    @property
    def suppressed_count(self) -> int:
        return sum(1 for o in self.outcomes if o.suppressed)

    def to_dict(self) -> Dict:
        return {
            "suppressed": self.suppressed_count,
            "expired_rules": [r.to_dict() for r in self.expired_rules],
            "unused_rules": [r.to_dict() for r in self.unused_rules],
            "empty_rules": [r.to_dict() for r in self.empty_rules],
        }


def apply_suppressions(reports: Iterable[VerifyReport],
                       suppressions: List[Suppression],
                       today: Optional[_dt.date] = None) -> SuppressionResult:
    """Annotate each report with whether a suppression rule baselines it.

    The first matching, non-expired rule wins.  Diagnostics about expired,
    unused and overly-broad (empty) rules are collected so the baseline file
    itself stays honest and maintainable.
    """
    reports = list(reports)
    used: set = set()
    outcomes: List[SuppressionOutcome] = []
    for r in reports:
        matched: Optional[Suppression] = None
        for idx, rule in enumerate(suppressions):
            if rule.matches(r, today):
                matched = rule
                used.add(idx)
                break
        outcomes.append(SuppressionOutcome(report=r, suppressed=matched is not None,
                                           rule=matched))

    expired = [s for s in suppressions if s.expired(today)]
    empty = [s for s in suppressions if s.is_empty_match()]
    unused = [s for i, s in enumerate(suppressions)
              if i not in used and not s.expired(today)]
    return SuppressionResult(outcomes=outcomes, expired_rules=expired,
                             unused_rules=unused, empty_rules=empty)


def build_baseline(reports: Iterable[VerifyReport], *,
                   reason: str = "baselined: pre-existing divergence",
                   by_fingerprint: bool = True) -> Dict:
    """Emit a suppression file capturing every *current* finding.

    With ``by_fingerprint`` (default) each finding is pinned by its stable
    fingerprint so the baseline matches exactly that unit+class and a genuinely
    *new* divergence is never accidentally suppressed.
    """
    rules: List[Dict] = []
    for r in reports:
        if r.verdict not in _SUPPRESSIBLE:
            continue
        cls = finding_class(r)
        rule: Dict[str, str] = {
            "reason": reason,
            "divergence_class": cls,
            "pair": pair_of(r),
            "unit": _unit_label(r),
        }
        if by_fingerprint:
            rule["fingerprint"] = fingerprint_of(r)
        rules.append(rule)
    # deterministic ordering for a stable, diff-friendly baseline file.
    rules.sort(key=lambda d: (d.get("divergence_class", ""), d.get("unit", ""),
                              d.get("fingerprint", "")))
    return {"version": SUPPRESSION_VERSION, "suppressions": rules}
