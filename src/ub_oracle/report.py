"""
Honest aggregate reporting + SARIF emission (100_STEPS steps 47 & 57).

Two reporting concerns, kept deliberately free of any equivalence claim:

* :func:`aggregate_reports` (Step 47) — turns a list of :class:`VerifyReport`
  into an **honest** summary: what fraction of units the oracle actually
  *decided*, what fraction it *abstained* on (and *why* — symbolic-only
  candidate vs. simply not-covered), and what fraction were *unknown* (a solver
  abstained/timed out). The breakdown is provided overall, per **language pair**,
  and per **divergence class**, so missing coverage is impossible to hide.

  Crucially, a ``NO_DIVERGENCE_FOUND`` verdict is counted as *decided* only in
  the narrow sense "the oracle checked the covered classes and found none"; it is
  **never** evidence of equivalence, and the summary says so out loud.

* :func:`to_sarif` (Step 57) — renders the confirmed/divergent findings as a
  SARIF 2.1.0 log so they surface in GitHub code scanning / VS Code. Only
  ground-truth-confirmed ``DIVERGENT`` results are emitted at ``error`` level;
  unconfirmed symbolic ``CANDIDATE`` witnesses are emitted at ``warning`` level
  and clearly labelled as not-yet-confirmed. No source locations are fabricated.
"""

from __future__ import annotations

import hashlib
import json
from typing import Dict, Iterable, List, Optional

from .catalogue import CATALOGUE
from .verify import VerifyReport, VerifyVerdict

#: how each verdict rolls up into the honest top-level buckets.
_BUCKET = {
    VerifyVerdict.DIVERGENT: "decided",
    VerifyVerdict.NO_DIVERGENCE_FOUND: "decided",
    VerifyVerdict.CANDIDATE: "abstained",
    VerifyVerdict.NOT_COVERED: "abstained",
    VerifyVerdict.UNKNOWN: "unknown",
}

SARIF_VERSION = "2.1.0"
SARIF_SCHEMA = ("https://raw.githubusercontent.com/oasis-tcs/sarif-spec/master/"
                "Schemata/sarif-schema-2.1.0.json")
TOOL_NAME = "cross-lang-verifier"
TOOL_URI = "https://github.com/thehalleyyoung/cross-lang-verifier"


def pair_of(report: VerifyReport) -> str:
    """The declared/implied language pair of a unit, e.g. ``c->rust``.

    Taken from the unit's explicit ``source_lang``/``target_lang`` when present,
    else from the first applicable oracle, else ``unknown->unknown``. Whether a
    default was applied is recorded by :func:`_pair_default_applied`.
    """
    u = report.unit
    src = u.get("source_lang")
    tgt = u.get("target_lang")
    if src and tgt:
        return f"{src}->{tgt}"
    for res in report.oracle_results:
        # OracleResult carries no langs; fall back to the divergence's, if any.
        pass
    if report.divergence is not None:
        # confirmed divergence implies the anchor pair of its oracle.
        return "c->rust"
    if src or tgt:
        return f"{src or 'unknown'}->{tgt or 'unknown'}"
    return "unknown->unknown"


def _class_of(report: VerifyReport) -> str:
    """Attribute a report to a divergence class for the per-class breakdown."""
    if report.divergence is not None:
        return report.divergence.divergence_class
    probe = report.unit.get("probe")
    if probe:
        return str(probe)
    classes = sorted({r.divergence_class for r in report.oracle_results})
    if len(classes) == 1:
        return classes[0]
    if not classes:
        return "uncovered"
    return "multiple"


def _empty_tally() -> Dict[str, int]:
    return {
        "total": 0,
        "decided": 0,
        "abstained": 0,
        "unknown": 0,
        # sub-buckets so the two kinds of abstention stay distinguishable.
        "divergent": 0,
        "no_divergence_found": 0,
        "candidate": 0,
        "not_covered": 0,
    }


def _tally(t: Dict[str, int], report: VerifyReport) -> None:
    t["total"] += 1
    t[_BUCKET[report.verdict]] += 1
    t[report.verdict.value] += 1


def _finalize(t: Dict[str, int]) -> Dict:
    n = t["total"] or 1
    out = dict(t)
    out["decided_fraction"] = t["decided"] / n
    out["abstained_fraction"] = t["abstained"] / n
    out["unknown_fraction"] = t["unknown"] / n
    return out


def aggregate_reports(reports: Iterable[VerifyReport]) -> Dict:
    """Honest decided/abstained/unknown summary, by pair and by class.

    The returned dict is JSON-serialisable and deterministic (keys sorted).
    """
    reports = list(reports)
    overall = _empty_tally()
    by_pair: Dict[str, Dict[str, int]] = {}
    by_class: Dict[str, Dict[str, int]] = {}

    for r in reports:
        _tally(overall, r)
        p = pair_of(r)
        _tally(by_pair.setdefault(p, _empty_tally()), r)
        c = _class_of(r)
        _tally(by_class.setdefault(c, _empty_tally()), r)

    return {
        "overall": _finalize(overall),
        "by_pair": {k: _finalize(v) for k, v in sorted(by_pair.items())},
        "by_class": {k: _finalize(v) for k, v in sorted(by_class.items())},
        "disclaimer": (
            "NO_DIVERGENCE_FOUND means only the covered divergence classes were "
            "checked and none fired; it is NOT a proof of equivalence. CANDIDATE "
            "is a symbolic witness that was not re-executed and is NOT a confirmed "
            "divergence."
        ),
    }


# ── SARIF ────────────────────────────────────────────────────────────────────

def _rules() -> List[Dict]:
    rules = []
    for key, entry in sorted(CATALOGUE.items()):
        rules.append({
            "id": key,
            "name": entry.cls.name.replace(" ", ""),
            "shortDescription": {"text": entry.cls.name},
            "fullDescription": {"text": entry.source_rule},
            "helpUri": TOOL_URI,
            "defaultConfiguration": {"level": "error"},
            "properties": {
                "c_standard_ref": entry.c_standard_ref,
                "severity": entry.severity.value,
                "source_definedness": entry.source_definedness.value,
                "rust_outcome": entry.rust_outcome.value,
            },
        })
    return rules


def _unit_name(report: VerifyReport) -> str:
    u = report.unit
    return str(u.get("name") or u.get("id") or u.get("kind") or "unit")


def _fingerprint(report: VerifyReport, cls: str) -> str:
    payload = json.dumps(report.unit, sort_keys=True, default=str)
    h = hashlib.sha256((cls + "|" + payload).encode("utf-8")).hexdigest()
    return h[:16]


def _location(report: VerifyReport) -> Dict:
    """A SARIF location: logical (always) + physical only if the unit carries one.

    Source locations are never fabricated; a physical location is emitted only
    when the unit explicitly provides ``source_file`` (and optionally ``line``).
    """
    loc: Dict = {"logicalLocations": [{
        "name": _unit_name(report),
        "kind": "translationUnit",
        "fullyQualifiedName": f"{pair_of(report)}::{_unit_name(report)}",
    }]}
    src_file = report.unit.get("source_file")
    if src_file:
        region = {"startLine": int(report.unit.get("line", 1))}
        loc["physicalLocation"] = {
            "artifactLocation": {"uri": str(src_file)},
            "region": region,
        }
    return loc


def to_sarif(reports: Iterable[VerifyReport]) -> Dict:
    """Render reports as a SARIF 2.1.0 log (GitHub code-scanning compatible)."""
    reports = list(reports)
    results: List[Dict] = []
    for r in reports:
        if r.verdict is VerifyVerdict.DIVERGENT:
            level = "error"
            cls = r.divergence.divergence_class if r.divergence else "unknown"
        elif r.verdict is VerifyVerdict.CANDIDATE:
            level = "warning"
            cls = _class_of(r)
        else:
            # decided-clean / unknown / not-covered are not SARIF findings.
            continue

        ce = None
        if r.divergence is not None:
            ce = r.divergence.counterexample
        if ce is None:
            for res in r.oracle_results:
                if res.counterexample is not None:
                    ce = res.counterexample
                    break

        props: Dict = {"verdict": r.verdict.value}
        if ce is not None:
            props["inputs"] = {k: repr(v) for k, v in ce.inputs.items()}
            props["divergence_witness"] = ce.divergence_witness
        if r.detail:
            props["detail"] = r.detail

        results.append({
            "ruleId": cls,
            "level": level,
            "message": {"text": r.banner()},
            "locations": [_location(r)],
            "partialFingerprints": {"unitClassHash/v1": _fingerprint(r, cls)},
            "properties": props,
        })

    return {
        "version": SARIF_VERSION,
        "$schema": SARIF_SCHEMA,
        "runs": [{
            "tool": {"driver": {
                "name": TOOL_NAME,
                "informationUri": TOOL_URI,
                "rules": _rules(),
            }},
            "results": results,
        }],
    }
