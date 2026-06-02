"""Step 99 — a structured, re-confirmable "divergence zoo".

A gallery is nice for humans; a *zoo* is a **machine-readable, indexed, and
continuously-verifiable** catalogue of the cross-language divergence patterns the
tool catches.  This module aggregates every catalogued pattern from the live
corpora (`idiomatic_corpus`, `multipair_corpus`) into:

  * a structured **index** keyed by `(divergence_class, language_pair)`, where
    each entry carries its provenance, its C source, the divergent translation,
    and — crucially — a concrete **witnessing input** that triggers the
    divergence;
  * a deterministic **JSON** export (`zoo.json`) and a human **markdown** page
    (`docs/zoo.md`), both generated, never hand-maintained, so the zoo cannot
    drift from the real catalogue.

What makes it a *zoo* rather than a static list is that every exhibit is
**re-confirmable**: `confirm_zoo()` takes each divergent entry's witness and
**re-runs the real oracle** on it, requiring the divergence to still be flagged
on the witnessing input and to stay silent on the safe input.  An exhibit that
cannot be reproduced live is rejected.  Consistency-only (``available=False``)
when no target toolchain is present — never fabricated.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from . import idiomatic_corpus as _idio
from . import multipair_corpus as _multi
from .reexec import ReexecHarness, toolchain_available

_ROOT = Path(__file__).resolve().parents[2]
_DOCS = _ROOT / "docs"
_ZOO_MD = _DOCS / "zoo.md"


@dataclass(frozen=True)
class ZooExhibit:
    exhibit_id: str
    divergence_class: str
    source_lang: str
    target_lang: str
    provenance: str
    declared_label: str           # "divergent" | "equivalent"
    c_src: str
    target_src: str
    witness: Tuple[str, ...]      # input that triggers the divergence
    safe: Tuple[str, ...]         # input that is well-defined on both sides

    @property
    def pair(self) -> str:
        return f"{self.source_lang}->{self.target_lang}"


def _exhibits() -> List[ZooExhibit]:
    out: List[ZooExhibit] = []
    for it in _idio.CORPUS:
        for lang, tgt in sorted(it.targets.items()):
            out.append(ZooExhibit(
                exhibit_id=f"idio:{it.item_id}:{lang}",
                divergence_class=it.klass, source_lang="c", target_lang=lang,
                provenance=it.provenance, declared_label=it.declared_label,
                c_src=it.c_src, target_src=tgt,
                witness=tuple(it.ub_inputs), safe=tuple(it.safe_inputs)))
    for fn in _multi.CORPUS:
        for lang, tgt in sorted(fn.targets.items()):
            out.append(ZooExhibit(
                exhibit_id=f"multi:{fn.func_id}:{lang}",
                divergence_class=fn.klass, source_lang="c", target_lang=lang,
                provenance=fn.provenance, declared_label=fn.declared_label,
                c_src=fn.c_src, target_src=tgt,
                witness=tuple(fn.ub_inputs), safe=tuple(fn.safe_inputs)))
    out.sort(key=lambda e: e.exhibit_id)
    return out


EXHIBITS: Tuple[ZooExhibit, ...] = tuple(_exhibits())


def index_by_class_and_pair() -> Dict[str, Dict[str, List[str]]]:
    """`{divergence_class: {pair: [exhibit_id, ...]}}` over divergent exhibits."""
    idx: Dict[str, Dict[str, List[str]]] = {}
    for e in EXHIBITS:
        if e.declared_label != "divergent":
            continue
        idx.setdefault(e.divergence_class, {}).setdefault(e.pair, []).append(
            e.exhibit_id)
    return idx


# ── exports ──────────────────────────────────────────────────────────────────


def to_json() -> dict:
    payload = {
        "schema": "divergence-zoo/1",
        "classes": sorted({e.divergence_class for e in EXHIBITS}),
        "pairs": sorted({e.pair for e in EXHIBITS}),
        "index": index_by_class_and_pair(),
        "exhibits": [
            {
                "id": e.exhibit_id,
                "class": e.divergence_class,
                "pair": e.pair,
                "provenance": e.provenance,
                "label": e.declared_label,
                "witness": list(e.witness),
                "safe": list(e.safe),
            }
            for e in EXHIBITS
        ],
    }
    return payload


def content_hash() -> str:
    blob = json.dumps(to_json(), sort_keys=True).encode()
    return hashlib.sha256(blob).hexdigest()[:16]


def _zoo_markdown() -> str:
    idx = index_by_class_and_pair()
    lines: List[str] = [
        "# The divergence zoo",
        "",
        "*Auto-generated from the live corpora — do not edit by hand; run "
        "`python -m ub_oracle.divergence_zoo`.*",
        "",
        "A machine-readable, indexed catalogue of the cross-language divergence "
        "patterns this tool catches, **indexed by class and language pair**. "
        "Every divergent exhibit carries a concrete **witnessing input** and is "
        "**re-confirmed live** by `confirm_zoo()` (the oracle must still flag the "
        "divergence on the witness and stay silent on the safe input).",
        "",
        f"*content hash: `{content_hash()}` — "
        f"{sum(len(p) for c in idx.values() for p in c.values())} divergent "
        f"exhibits across {len(idx)} classes.*",
        "",
        "## Index — class × pair",
        "",
        "| divergence class | language pair | exhibits |",
        "|------------------|---------------|----------|",
    ]
    for klass in sorted(idx):
        for pair in sorted(idx[klass]):
            ids = ", ".join(f"`{i}`" for i in sorted(idx[klass][pair]))
            lines.append(f"| `{klass}` | `{pair}` | {ids} |")
    lines += ["", "## Exhibits", ""]
    for e in EXHIBITS:
        if e.declared_label != "divergent":
            continue
        lines += [
            f"### `{e.exhibit_id}` — {e.divergence_class} ({e.pair})",
            "",
            f"*Mirrors:* {e.provenance}. *Witness:* `{list(e.witness)}` "
            f"(safe: `{list(e.safe)}`).",
            "",
            "```c",
            e.c_src.strip(),
            "```",
            "",
            f"```{ 'rust' if e.target_lang=='rust' else e.target_lang }",
            e.target_src.strip(),
            "```",
            "",
        ]
    return "\n".join(lines)


def generate_zoo() -> Path:
    _ZOO_MD.write_text(_zoo_markdown(), encoding="utf-8")
    return _ZOO_MD


# ── live re-confirmation ─────────────────────────────────────────────────────


@dataclass
class ExhibitCheck:
    exhibit_id: str
    confirmed: bool
    detail: str


@dataclass
class ZooReport:
    available: bool
    ok: bool
    content_hash: str
    n_divergent: int
    n_confirmed: int
    checks: Tuple[ExhibitCheck, ...] = field(default_factory=tuple)
    detail: str = ""


def confirm_zoo() -> ZooReport:
    """Re-confirm every divergent exhibit live: the oracle must flag the
    divergence on the witnessing input and stay silent on the safe input.
    Consistency-only when no target toolchain is present."""
    generate_zoo()
    status = toolchain_available()
    h = ReexecHarness(status)
    divergent = [e for e in EXHIBITS if e.declared_label == "divergent"]

    def _available(e: ZooExhibit) -> bool:
        if e.divergence_class == "memcpy_overlap":
            return status.full_libc_contract_for(e.target_lang)
        return status.full_for(e.target_lang)

    if not any(_available(e) for e in divergent):
        return ZooReport(
            available=False, ok=True, content_hash=content_hash(),
            n_divergent=len(divergent), n_confirmed=0,
            detail="no target toolchain; zoo generated, witnesses not re-run")

    checks: List[ExhibitCheck] = []
    confirmed = 0
    for e in divergent:
        if not _available(e):
            continue
        w = [a for a in e.witness if a != ""]
        s = [a for a in e.safe if a != ""]
        if e.divergence_class == "memcpy_overlap":
            ub = h.confirm_libc_contract_trap_vs_defined(
                e.c_src, e.target_src, w, e.divergence_class, e.target_lang)
            safe = h.confirm_libc_contract_trap_vs_defined(
                e.c_src, e.target_src, s, e.divergence_class, e.target_lang)
        else:
            ub = h.confirm_trap_vs_defined(e.c_src, e.target_src, w,
                                           e.divergence_class, e.target_lang)
            safe = h.confirm_trap_vs_defined(e.c_src, e.target_src, s,
                                             e.divergence_class, e.target_lang)
        good = bool(ub.confirmed) and not bool(safe.confirmed)
        confirmed += int(good)
        checks.append(ExhibitCheck(
            e.exhibit_id, good,
            f"witness_flagged={ub.confirmed} safe_silent={not safe.confirmed}"))

    ran = [c for c in checks]
    ok = len(ran) > 0 and all(c.confirmed for c in ran)
    return ZooReport(
        available=True, ok=bool(ok), content_hash=content_hash(),
        n_divergent=len(divergent), n_confirmed=confirmed,
        checks=tuple(checks),
        detail=f"{confirmed}/{len(ran)} exhibits re-confirmed live "
               f"({len(divergent)} catalogued)")


if __name__ == "__main__":  # pragma: no cover
    rep = confirm_zoo()
    print("divergence-zoo:", rep.detail, "hash", rep.content_hash)
    for c in rep.checks:
        print(f"  {c.exhibit_id}: {c.detail} ok={c.confirmed}")
    print("=> ok" if rep.ok else "=> FAILED")
    raise SystemExit(0 if rep.ok else 1)
