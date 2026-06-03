"""Generated soundness compendium (100_STEPS step 140).

The soundness gate enforces one statement per registered oracle instance; the
traceability table maps public claims to modules and executable checks; the Lean
driver names the mechanized theorem surface.  This module joins those three
sources into one deterministic oracle -> theorem -> witness document so the paper
and artifact review have a single auditable map.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

from . import mechanized_soundness as ms
from . import soundness_gate as sg
from . import traceability


ROOT = Path(__file__).resolve().parents[2]
SOUNDNESS_COMPENDIUM_DOC = ROOT / "docs" / "SOUNDNESS_COMPENDIUM.md"


@dataclass(frozen=True)
class CompendiumRow:
    statement_id: str
    source_lang: str
    target_lang: str
    divergence_class: str
    confirmation_mode: str
    source_definedness: str
    theorem_refs: Tuple[str, ...]
    witness_unit_json: str

    @property
    def oracle(self) -> str:
        return f"{self.source_lang}->{self.target_lang}:{self.divergence_class}"


@dataclass(frozen=True)
class CompendiumCheck:
    ok: bool
    doc_fresh: bool
    audit_ok: bool
    missing_claim_refs: Tuple[str, ...]
    missing_lean_refs: Tuple[str, ...]
    row_count: int

    def detail(self) -> str:
        if self.ok:
            return f"{self.row_count} oracle rows; compendium fresh"
        parts: List[str] = []
        if not self.doc_fresh:
            parts.append("docs/SOUNDNESS_COMPENDIUM.md is stale")
        if not self.audit_ok:
            parts.append("soundness registry audit failed")
        if self.missing_claim_refs:
            parts.append(f"missing claim refs: {list(self.missing_claim_refs)}")
        if self.missing_lean_refs:
            parts.append(f"missing Lean refs: {list(self.missing_lean_refs)}")
        return "; ".join(parts)


def _statement_rows() -> Tuple[CompendiumRow, ...]:
    rows: List[CompendiumRow] = []
    for st in sorted(sg.SOUNDNESS_STATEMENTS, key=lambda s: s.key):
        rows.append(CompendiumRow(
            statement_id=st.id,
            source_lang=st.source_lang,
            target_lang=st.target_lang,
            divergence_class=st.divergence_class,
            confirmation_mode=st.expected_confirmation_mode,
            source_definedness=st.source_definedness,
            theorem_refs=tuple(st.theorem_refs),
            witness_unit_json=json.dumps(
                st.witness_unit,
                sort_keys=True,
                separators=(",", ":"),
            ),
        ))
    return tuple(rows)


def _claim_by_id() -> Dict[str, traceability.Claim]:
    return {claim.id: claim for claim in traceability.CLAIMS}


def _missing_refs(rows: Sequence[CompendiumRow]) -> Tuple[Tuple[str, ...], Tuple[str, ...]]:
    claims = set(traceability.claim_ids())
    lean = set(ms.REQUIRED_THEOREMS)
    missing_claims: List[str] = []
    missing_lean: List[str] = []
    for row in rows:
        for ref in row.theorem_refs:
            if ref.startswith("claim:") and ref.split(":", 1)[1] not in claims:
                missing_claims.append(ref)
            if ref.startswith("lean:") and ref.split(":", 1)[1] not in lean:
                missing_lean.append(ref)
    return tuple(sorted(set(missing_claims))), tuple(sorted(set(missing_lean)))


def compendium_rows() -> Tuple[CompendiumRow, ...]:
    """Return the stable oracle -> theorem -> witness rows."""

    return _statement_rows()


def render_compendium() -> str:
    rows = compendium_rows()
    claims = _claim_by_id()
    lines: List[str] = [
        "# Soundness compendium",
        "",
        "This file is generated from `src/ub_oracle/soundness_compendium.py`, "
        "which joins `soundness_gate.SOUNDNESS_STATEMENTS`, "
        "`traceability.CLAIMS`, and the Lean theorem contract in "
        "`mechanized_soundness.REQUIRED_THEOREMS`. It is intentionally stable: "
        "it records declared theorem/claim references and concrete witness units, "
        "not host-dependent proof-assistant output.",
        "",
        f"- Registered oracle statements: **{len(rows)}**",
        f"- Traceability claims: **{len(claims)}**",
        f"- Required ProductSoundness Lean theorems: **{len(ms.REQUIRED_THEOREMS)}**",
        "",
        "## Mechanized theorem surface",
        "",
    ]
    lines.extend(f"- `{name}`" for name in ms.REQUIRED_THEOREMS)
    lines.extend([
        "",
        "## Oracle-to-evidence matrix",
        "",
        "| Statement | Oracle | Mode | Source premise | Evidence refs | Witness unit |",
        "| --- | --- | --- | --- | --- | --- |",
    ])
    for row in rows:
        refs = ", ".join(f"`{ref}`" for ref in row.theorem_refs)
        lines.append(
            f"| `{row.statement_id}` | `{row.oracle}` | `{row.confirmation_mode}` | "
            f"`{row.source_definedness}` | {refs} | "
            f"`{row.witness_unit_json}` |"
        )
    lines.extend([
        "",
        "## Traceability claim index",
        "",
        "| Claim | Module | Symbols |",
        "| --- | --- | --- |",
    ])
    for claim in traceability.CLAIMS:
        symbols = ", ".join(f"`{sym}`" for sym in claim.symbols) or "—"
        lines.append(f"| `{claim.id}` | `{claim.module}` | {symbols} |")
    lines.append("")
    return "\n".join(lines)


def confirm_compendium(
    *,
    doc_path: Path = SOUNDNESS_COMPENDIUM_DOC,
    probe_witnesses: bool = False,
) -> CompendiumCheck:
    """Check registry coverage, reference validity, and generated-doc freshness."""

    rows = compendium_rows()
    missing_claims, missing_lean = _missing_refs(rows)
    audit = sg.confirm_soundness_registry(probe_witnesses=probe_witnesses)
    expected = render_compendium()
    try:
        actual = doc_path.read_text(encoding="utf-8")
    except OSError:
        actual = ""
    doc_fresh = actual == expected
    ok = bool(doc_fresh and audit.ok and not missing_claims and not missing_lean)
    return CompendiumCheck(
        ok=ok,
        doc_fresh=doc_fresh,
        audit_ok=audit.ok,
        missing_claim_refs=missing_claims,
        missing_lean_refs=missing_lean,
        row_count=len(rows),
    )


__all__ = [
    "CompendiumRow",
    "CompendiumCheck",
    "SOUNDNESS_COMPENDIUM_DOC",
    "compendium_rows",
    "render_compendium",
    "confirm_compendium",
]
