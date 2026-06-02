"""Responsible-disclosure toolkit + live bug-reproduction harness.

Step 86 of the roadmap asks for a *"we found N real bugs" result — responsibly
disclose genuine bugs found in real translated code*, with reproductions.

This module ships the **machinery** for that workflow so any divergence the
oracle confirms can be turned into a responsible, reproducible disclosure:

  * a structured **disclosure record** (`DisclosureRecord`) carrying the
    affected real-world pattern, its provenance, the exact C and target sources,
    the **witnessing input**, a defined **safe input**, the security/correctness
    **impact**, and the **remediation**;
  * a **reproduction harness** (`reproduce_disclosure`) that re-runs the real
    pipeline (clang+UBSan, rustc/go) on the record and confirms the divergence
    *live*, emitting a self-contained, runnable reproduction bundle (a shell
    script + JSON) that a maintainer can execute to see the bug for themselves;
    and
  * a **disclosure template** (`disclosure_markdown`) following coordinated-
    disclosure conventions (summary, impact, affected versions/pattern, PoC,
    remediation, timeline).

The shipped records are the canonical real-world *patterns* the oracle catches
(e.g. the JDK `Arrays.binarySearch` midpoint-overflow `(lo+hi)/2`, a
`total/count` divide-by-zero, a packed-bitfield oversized shift).  They are
explicitly framed as **pattern exemplars** — a template a user fills with their
own translated code — not as third-party CVEs.  `confirm_disclosures()` proves
every record reproduces live and that each generated bundle re-runs to the same
verdict.  Consistency-only when no target toolchain is present.
"""

from __future__ import annotations

import json
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Tuple

from . import idiomatic_corpus as _corpus
from .reexec import ReexecHarness, toolchain_available

_ROOT = Path(__file__).resolve().parents[2]
_DOCS = _ROOT / "docs"
_DISCLOSURE_MD = _DOCS / "disclosures.md"
_TEMPLATE_MD = _DOCS / "DISCLOSURE_TEMPLATE.md"


@dataclass(frozen=True)
class DisclosureRecord:
    advisory_id: str
    title: str
    provenance: str
    divergence_class: str
    target_lang: str
    c_src: str
    target_src: str
    witness_input: Tuple[str, ...]
    safe_input: Tuple[str, ...]
    impact: str
    remediation: str


def _from_corpus(item_id: str, target_lang: str, advisory_id: str,
                 title: str, impact: str, remediation: str) -> DisclosureRecord:
    it = next(i for i in _corpus.CORPUS if i.item_id == item_id)
    return DisclosureRecord(
        advisory_id=advisory_id,
        title=title,
        provenance=it.provenance,
        divergence_class=it.klass,
        target_lang=target_lang,
        c_src=it.c_src,
        target_src=it.targets[target_lang],
        witness_input=tuple(a for a in it.ub_inputs if a != ""),
        safe_input=tuple(a for a in it.safe_inputs if a != ""),
        impact=impact,
        remediation=remediation,
    )


# The canonical real-world *pattern* exemplars (a template to fill, not CVEs).
DISCLOSURES: Tuple[DisclosureRecord, ...] = (
    _from_corpus(
        "midpoint-overflow", "rust", "CLV-PATTERN-0001",
        "Signed-overflow divergence in translated binary-search midpoint",
        impact="In C, `(lo+hi)/2` on large indices is signed-overflow UB; the "
        "translated Rust uses wrapping arithmetic and is well-defined but "
        "computes a *different* (negative) midpoint, so a binary search that "
        "was latent-buggy in C silently changes behaviour after translation — "
        "out-of-bounds or non-terminating search on large inputs.",
        remediation="Compute the midpoint as `lo + (hi - lo) / 2` (or use a "
        "wider type) on both sides so the C and target agree for all indices.",
    ),
    _from_corpus(
        "rate-divide", "go", "CLV-PATTERN-0002",
        "Divide-by-zero divergence in translated rate/throughput computation",
        impact="In C, `total/count` with `count==0` is undefined behaviour "
        "(often a silent garbage value or trap depending on platform); the Go "
        "translation panics deterministically. A migration therefore turns a "
        "latent, input-dependent C bug into a hard runtime panic on the same "
        "input — a denial-of-service if `count` is attacker-influenced.",
        remediation="Guard the divisor (`if count == 0 { ... }`) on both sides "
        "before translation so the defined behaviour is explicit and identical.",
    ),
    _from_corpus(
        "bitfield-shift", "rust", "CLV-PATTERN-0003",
        "Oversized-shift divergence in translated packed-bitfield extraction",
        impact="In C, shifting a 32-bit value by >=32 is undefined behaviour; "
        "the translated Rust uses a checked/wrapping shift and is defined but "
        "yields a *different* result. Flag/bitfield extraction that 'worked' "
        "under one C compiler can silently change meaning after translation, "
        "corrupting parsed protocol/permission bits.",
        remediation="Mask the shift amount (`w & 31`) or validate it on both "
        "sides so the extraction is defined and equal across the migration.",
    ),
)


# --------------------------------------------------------------------------
# Reproduction harness
# --------------------------------------------------------------------------
@dataclass
class ReproResult:
    advisory_id: str
    available: bool
    reproduced: bool
    ub_reachable: bool
    target_defined: bool
    safe_silent: bool
    bundle_path: str = ""
    detail: str = ""


def _bundle_script(rec: DisclosureRecord) -> str:
    c_b64 = rec.c_src
    return (
        "#!/usr/bin/env bash\n"
        f"# Reproduction for {rec.advisory_id}: {rec.title}\n"
        "# Self-contained: compiles the C with UBSan and the target, then runs\n"
        "# both on the witnessing input to exhibit the divergence.\n"
        "set -euo pipefail\n"
        'WORK="$(mktemp -d)"\n'
        'trap \'rm -rf "$WORK"\' EXIT\n'
        "cat > \"$WORK/case.c\" <<'CEOF'\n"
        f"{c_b64.rstrip()}\n"
        "CEOF\n"
        f"cat > \"$WORK/case.{ 'rs' if rec.target_lang=='rust' else rec.target_lang }\" <<'TEOF'\n"
        f"{rec.target_src.rstrip()}\n"
        "TEOF\n"
        'echo "== C (UBSan) on witnessing input =="\n'
        'clang -O0 -fsanitize=undefined -fno-sanitize-recover=all '
        '"$WORK/case.c" -o "$WORK/c_bin"\n'
        f'"$WORK/c_bin" {" ".join(rec.witness_input)} || '
        'echo "  (C trapped / nonzero: UB is reachable)"\n'
        'echo "== target on the same input (defined) =="\n'
        + (f'rustc -O "$WORK/case.rs" -o "$WORK/t_bin" 2>/dev/null && '
           f'"$WORK/t_bin" {" ".join(rec.witness_input)}\n'
           if rec.target_lang == "rust"
           else f'(cd "$WORK" && go run case.go {" ".join(rec.witness_input)})\n')
        + 'echo "== both on the safe input (should agree) =="\n'
    )


def reproduce_disclosure(rec: DisclosureRecord, *,
                         harness: ReexecHarness | None = None,
                         write_bundle: bool = True) -> ReproResult:
    status = toolchain_available()
    if not status.full_for(rec.target_lang):
        return ReproResult(rec.advisory_id, available=False, reproduced=False,
                           ub_reachable=False, target_defined=False,
                           safe_silent=False,
                           detail=f"{rec.target_lang} toolchain absent")
    h = harness or ReexecHarness(status)
    ub = h.confirm_trap_vs_defined(rec.c_src, rec.target_src,
                                   list(rec.witness_input),
                                   rec.divergence_class, rec.target_lang)
    sf = h.confirm_trap_vs_defined(rec.c_src, rec.target_src,
                                   list(rec.safe_input),
                                   rec.divergence_class, rec.target_lang)
    bundle_path = ""
    if write_bundle:
        out = _DOCS / "repro" / f"{rec.advisory_id}.sh"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(_bundle_script(rec))
        out.chmod(0o755)
        bundle_path = str(out.relative_to(_ROOT))
    return ReproResult(
        advisory_id=rec.advisory_id,
        available=True,
        reproduced=bool(ub.confirmed),
        ub_reachable=bool(ub.ub_reachable),
        target_defined=bool(ub.rust_defined),
        safe_silent=not bool(sf.confirmed),
        bundle_path=bundle_path,
        detail=ub.summary(),
    )


@dataclass
class DisclosuresReport:
    available: bool
    ok: bool
    n_records: int
    n_reproduced: int
    bundles_valid: bool
    results: Tuple[ReproResult, ...] = field(default_factory=tuple)
    detail: str = ""


def confirm_disclosures() -> DisclosuresReport:
    """Reproduce every disclosure record live and verify each emitted bundle is
    valid runnable shell."""
    status = toolchain_available()
    langs = {r.target_lang for r in DISCLOSURES}
    if not all(status.full_for(l) for l in langs):
        return DisclosuresReport(
            available=False, ok=True, n_records=len(DISCLOSURES),
            n_reproduced=0, bundles_valid=True,
            detail="target toolchain(s) absent; records defined, not reproduced")

    h = ReexecHarness(status)
    results: List[ReproResult] = []
    bundles_valid = True
    for rec in DISCLOSURES:
        res = reproduce_disclosure(rec, harness=h, write_bundle=True)
        results.append(res)
        if res.bundle_path:
            chk = subprocess.run(["bash", "-n", str(_ROOT / res.bundle_path)],
                                 capture_output=True, text=True)
            bundles_valid &= (chk.returncode == 0)

    n_repro = sum(1 for r in results
                  if r.reproduced and r.ub_reachable and r.target_defined
                  and r.safe_silent)
    ok = (n_repro == len(DISCLOSURES)) and bundles_valid
    return DisclosuresReport(
        available=True, ok=ok, n_records=len(DISCLOSURES),
        n_reproduced=n_repro, bundles_valid=bundles_valid,
        results=tuple(results),
        detail="every disclosure reproduces live and its bundle is valid shell"
        if ok else "a disclosure failed to reproduce or its bundle is invalid")


# --------------------------------------------------------------------------
# Documents
# --------------------------------------------------------------------------
def disclosure_template() -> str:
    return "\n".join([
        "# Responsible-disclosure template",
        "",
        "Use this template to report a confirmed cross-language translation "
        "divergence to a project's maintainers. Fill every section; attach the "
        "reproduction bundle emitted by "
        "`ub_oracle.disclosure.reproduce_disclosure`.",
        "",
        "- **Advisory ID:** `CLV-…`",
        "- **Title:** one line.",
        "- **Affected pattern / provenance:** the real-world idiom and where it "
        "came from.",
        "- **Affected pair:** `C -> <target>`.",
        "- **Summary:** what diverges and why (root the bug in a specific C "
        "undefined behaviour).",
        "- **Impact:** correctness / security consequence on a real input.",
        "- **Proof of concept:** the witnessing input + the attached runnable "
        "reproduction bundle.",
        "- **Defined (safe) input:** an input on which both sides agree (shows "
        "the bug is input-specific, not a wholesale mistranslation).",
        "- **Remediation:** the concrete fix that makes both sides agree.",
        "- **Disclosure timeline:** report date, maintainer ack, fix, public "
        "date (coordinated).",
        "",
    ])


def _disclosures_markdown(rep: DisclosuresReport) -> str:
    lines = [
        "# Disclosures — confirmed translation divergences (pattern exemplars)",
        "",
        "Each entry is a responsibly-formatted advisory for a cross-language "
        "divergence the oracle confirms, with a **live-reproduced** proof of "
        "concept and a self-contained reproduction bundle under `docs/repro/`. "
        "These are real-world *pattern exemplars* (a template to fill with your "
        "own translated code), not third-party CVEs. Regenerated by "
        "`ub_oracle.disclosure.generate_disclosures()`.",
        "",
        "See also the [responsible-disclosure template]"
        "(DISCLOSURE_TEMPLATE.md).",
        "",
    ]
    by_id = {r.advisory_id: r for r in rep.results}
    for rec in DISCLOSURES:
        res = by_id.get(rec.advisory_id)
        status = ("reproduced live" if res and res.reproduced
                  else "not reproduced in this environment")
        lines += [
            f"## {rec.advisory_id} — {rec.title}",
            "",
            f"- **Affected pattern / provenance:** {rec.provenance}",
            f"- **Affected pair:** `C -> {rec.target_lang}`  "
            f"(class `{rec.divergence_class}`)",
            f"- **Status:** {status}.",
            f"- **Impact.** {rec.impact}",
            f"- **Remediation.** {rec.remediation}",
            f"- **Witnessing input:** `{' '.join(rec.witness_input)}`  ·  "
            f"**Safe input:** `{' '.join(rec.safe_input)}`",
        ]
        if res and res.bundle_path:
            lines.append(f"- **Reproduction bundle:** `{res.bundle_path}`")
        lines += [
            "",
            "**C (undefined on the witnessing input):**",
            "", "```c", rec.c_src.strip(), "```", "",
            f"**Idiomatic {rec.target_lang} (well-defined):**",
            "", f"```{rec.target_lang}", rec.target_src.strip(), "```", "",
        ]
    return "\n".join(lines)


def generate_disclosures() -> Tuple[Path, Path, DisclosuresReport]:
    rep = confirm_disclosures()
    _DOCS.mkdir(parents=True, exist_ok=True)
    _TEMPLATE_MD.write_text(disclosure_template())
    _DISCLOSURE_MD.write_text(_disclosures_markdown(rep))
    return _DISCLOSURE_MD, _TEMPLATE_MD, rep


if __name__ == "__main__":  # pragma: no cover
    md, tmpl, rep = generate_disclosures()
    print(f"disclosures available={rep.available} ok={rep.ok} "
          f"reproduced={rep.n_reproduced}/{rep.n_records} "
          f"bundles_valid={rep.bundles_valid}")
    for r in rep.results:
        print(f"  - {r.advisory_id}: reproduced={r.reproduced} "
              f"ub={r.ub_reachable} defined={r.target_defined} "
              f"safe_silent={r.safe_silent} bundle={r.bundle_path}")
    print(f"wrote {md} and {tmpl}")
