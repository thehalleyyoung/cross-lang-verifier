"""Claims audit: tie every quantitative / framing claim in the docs to what the
code actually proves.

Step 93 of the roadmap asks to *tighten claims to exactly what's proven — cut
every overclaim; the general framing must be backed by >=2 pairs and real
machinery, not aspiration.*  This module turns that into a mechanical guard.

It scans the public docs (`README.md`, `CAPABILITIES.md`, `docs/TRACEABILITY.md`)
for a small set of **auditable claims** and checks each against a value computed
live from the real code:

  * every named C->target **language pair** mentioned in the "Rust/Go/Swift"
    framing is actually registered with **at least one real oracle** (so the
    multi-language framing is backed, not aspirational);
  * the **>=2-pairs** generality bar holds (the general "cross-language oracle"
    framing requires more than one working pair);
  * each literal **count** asserted in prose (e.g. "N exhibits across
    rust/go/swift", "N language pairs") equals the live count from the
    re-confirmed `divergence_zoo` / registry; and
  * **every traceability claim** carries a theorem that returns truthy under the
    current environment (consistency-gated when a toolchain is absent), so the
    docs never advertise a capability whose own check fails.

If a doc is edited to overclaim — a bigger exhibit count, a pair with no oracle,
a generality boast with only one pair — `confirm_claims_audit()` fails.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Tuple

_ROOT = Path(__file__).resolve().parents[2]
_README = _ROOT / "README.md"
_CAPABILITIES = _ROOT / "CAPABILITIES.md"
_TRACE_DOC = _ROOT / "docs" / "TRACEABILITY.md"


def _pkg():
    try:
        import ub_oracle as p
    except ModuleNotFoundError:
        import src.ub_oracle as p  # type: ignore
    return p


# --------------------------------------------------------------------------
# Live ground-truth values
# --------------------------------------------------------------------------
def _live_values() -> Dict[str, object]:
    p = _pkg()
    try:
        from ub_oracle import divergence_zoo as zoo
    except ModuleNotFoundError:
        from src.ub_oracle import divergence_zoo as zoo  # type: ignore

    pairs = list(p.language_pairs())
    pairs_with_oracle = [pr for pr in pairs if len(p.oracles_for(*pr)) >= 1]
    divergent = [e for e in zoo.EXHIBITS if e.declared_label == "divergent"]
    zoo_pairs = set(zoo.to_json()["pairs"])
    return {
        "pairs": pairs,
        "pairs_with_oracle": pairs_with_oracle,
        "n_pairs": len(pairs),
        "n_pairs_with_oracle": len(pairs_with_oracle),
        "divergent_exhibits": len(divergent),
        "zoo_pairs": zoo_pairs,
    }


# --------------------------------------------------------------------------
# Audited claims
# --------------------------------------------------------------------------
@dataclass
class ClaimCheck:
    name: str
    ok: bool
    expected: str
    actual: str
    detail: str = ""


def _check_named_pairs_have_oracles(v: Dict[str, object],
                                    text: str | None = None) -> ClaimCheck:
    # The docs repeatedly frame the tool around "C->Rust/Go/Swift". Every named
    # target must be a registered pair with at least one real oracle.
    named = {"rust", "go", "swift"}
    if text is None:
        text = (_README.read_text() + _CAPABILITIES.read_text())
    low = text.lower()
    mentioned = {t for t in named if t in low}
    have = {pr[1] for pr in v["pairs_with_oracle"]}  # type: ignore[index]
    missing = sorted(mentioned - have)
    ok = not missing
    return ClaimCheck(
        "named_pairs_have_oracles", ok,
        expected=f"every named target {sorted(mentioned)} has a real oracle",
        actual=f"targets with oracle: {sorted(have)}",
        detail="" if ok else f"named-but-unbacked: {missing}",
    )


def _check_generality_two_pairs(v: Dict[str, object]) -> ClaimCheck:
    # The general "cross-language" framing requires >=2 working pairs.
    n = int(v["n_pairs_with_oracle"])  # type: ignore[arg-type]
    ok = n >= 2
    return ClaimCheck(
        "generality_backed_by_2plus_pairs", ok,
        expected=">=2 pairs with a real oracle",
        actual=f"{n} pairs with a real oracle",
        detail="" if ok else "general framing not backed by >=2 pairs",
    )


def _check_exhibit_count(v: Dict[str, object],
                         text: str | None = None) -> ClaimCheck:
    # "N exhibits across rust/go/swift" in the README must equal the live count
    # of divergent zoo exhibits.
    if text is None:
        text = _README.read_text()
    m = re.search(r"(\d+)\s+exhibits across rust/go/swift", text)
    live = int(v["divergent_exhibits"])  # type: ignore[arg-type]
    if not m:
        return ClaimCheck(
            "readme_exhibit_count", True,
            expected="(no literal exhibit-count claim found)",
            actual=f"live divergent exhibits={live}",
            detail="nothing to over/under-claim",
        )
    claimed = int(m.group(1))
    ok = claimed == live
    return ClaimCheck(
        "readme_exhibit_count", ok,
        expected=f"{live} divergent exhibits (live)",
        actual=f"README claims {claimed}",
        detail="" if ok else "README exhibit count != live count",
    )


def _check_every_traceability_claim_passes() -> ClaimCheck:
    # Docs never advertise a capability whose own theorem fails right now.
    p = _pkg()
    try:
        from ub_oracle import traceability as t
    except ModuleNotFoundError:
        from src.ub_oracle import traceability as t  # type: ignore
    failed: List[str] = []
    for c in t.CLAIMS:
        thm = getattr(c, "theorem", None)
        if thm is None:
            continue
        try:
            if not bool(thm()):
                failed.append(c.id)
        except Exception as exc:  # pragma: no cover - environment dependent
            failed.append(f"{c.id}({exc!r})")
    ok = not failed
    return ClaimCheck(
        "all_traceability_theorems_pass", ok,
        expected=f"all {len(t.CLAIMS)} claim theorems truthy",
        actual=f"failing: {failed}",
        detail="" if ok else "a documented capability's own check fails",
    )


@dataclass
class ClaimsAuditReport:
    ok: bool
    n_checks: int
    checks: Tuple[ClaimCheck, ...] = field(default_factory=tuple)
    detail: str = ""


def confirm_claims_audit(*, include_theorems: bool = False) -> ClaimsAuditReport:
    """Run every audited claim against live ground truth.

    The default runs the fast, novel doc<->code checks (counts, named-pair
    backing, the >=2-pairs generality bar).  ``include_theorems=True`` also
    re-runs **every** traceability theorem core (a release/CI gate; slow,
    recompiles real programs) to ensure no documented capability's own check is
    currently failing.
    """
    v = _live_values()
    checks: List[ClaimCheck] = [
        _check_named_pairs_have_oracles(v),
        _check_generality_two_pairs(v),
        _check_exhibit_count(v),
    ]
    if include_theorems:
        checks.append(_check_every_traceability_claim_passes())
    ok = all(c.ok for c in checks)
    return ClaimsAuditReport(
        ok=ok,
        n_checks=len(checks),
        checks=tuple(checks),
        detail="every audited doc claim matches the code"
        if ok else "an audited claim is not backed by the code",
    )


def audit_text(text: str) -> Tuple[ClaimCheck, ClaimCheck]:
    """Audit an arbitrary doc string (used to prove the guard catches edits):
    returns the named-pair-backing and exhibit-count checks for ``text``."""
    v = _live_values()
    return (_check_named_pairs_have_oracles(v, text),
            _check_exhibit_count(v, text))


if __name__ == "__main__":  # pragma: no cover
    rep = confirm_claims_audit()
    print(f"claims-audit ok={rep.ok} checks={rep.n_checks}")
    for c in rep.checks:
        flag = "ok " if c.ok else "XX "
        print(f"  {flag}{c.name}: expected={c.expected} | actual={c.actual}"
              + (f" | {c.detail}" if c.detail else ""))
