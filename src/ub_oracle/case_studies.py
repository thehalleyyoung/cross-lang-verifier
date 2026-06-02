"""End-to-end case studies with cost/benefit.

Step 87 of the roadmap: *walk several real migrations end to end; show
cost/benefit.*  This module turns three real-world-derived C->target migrations
into fully **executed** case studies and quantifies the cost/benefit of the
oracle against an equal-budget differential fuzzer running on the **same real
binaries**.

Each case study is sourced from `idiomatic_corpus` (real provenance — e.g. the
classic JDK/`Arrays.binarySearch` midpoint-overflow bug `(lo+hi)/2`) and is
walked through the full pipeline against real compilers:

  1. **UB is reachable in C** — clang+UBSan on the witnessing input traps
     (the source execution is genuinely undefined).
  2. **The target is well-defined** — the idiomatic Rust/Go translation runs
     deterministically on the same input (a *defined* outcome).
  3. **The oracle confirms the divergence** on the witness and stays **silent**
     on the safe input (no false alarm), and we time how long that costs.
  4. **Cost/benefit vs fuzzing** — an equal-budget random differential tester is
     run on the *same compiled C and target binaries*; for sparse-UB classes it
     burns its whole budget without ever hitting the bug, while the oracle finds
     it deterministically.  The headline benefit is the **false-negative gap**:
     bugs the oracle confirms that equal-budget fuzzing misses.

`confirm_case_studies()` executes every case live and asserts the walk holds and
at least one case exhibits the fuzzing gap.  `generate_case_studies()` writes a
reproducible `docs/case_studies.md`.  Consistency-only when no toolchain present.
"""

from __future__ import annotations

import random
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from . import idiomatic_corpus as _corpus
from .reexec import ReexecHarness, toolchain_available

_ROOT = Path(__file__).resolve().parents[2]
_DOCS = _ROOT / "docs"
_CASE_MD = _DOCS / "case_studies.md"

_C_UBSAN_FLAGS = ["-O0", "-fsanitize=undefined", "-fno-sanitize-recover=all"]

# The three real-world-derived migrations we walk, with the target language each
# is most idiomatically translated into in the corpus.
_CASE_PLAN: Tuple[Tuple[str, str], ...] = (
    ("midpoint-overflow", "rust"),
    ("rate-divide", "go"),
    ("bitfield-shift", "rust"),
)


def _item(item_id: str) -> _corpus.IdiomaticItem:
    for it in _corpus.CORPUS:
        if it.item_id == item_id:
            return it
    raise KeyError(item_id)


# --------------------------------------------------------------------------
# Equal-budget differential fuzz over the *real* compiled binaries.
# --------------------------------------------------------------------------
def _int32() -> Tuple[int, int]:
    return (-(2 ** 31), 2 ** 31 - 1)


def _differential_fuzz_binaries(
    h: ReexecHarness, c_bin: str, tgt_bin: str, arity: int,
    target_lang: str, *, trials: int, seed: int,
) -> Tuple[int, Optional[int]]:
    """Sample ``trials`` random integer inputs and look for a divergence on the
    already-compiled C (UBSan) and target binaries.  Returns
    ``(hits, first_hit_trial)``; stops at the first hit (a differential tester
    only needs one counterexample)."""
    lo, hi = _int32()
    rng = random.Random(seed)
    hits = 0
    first: Optional[int] = None
    for t in range(1, trials + 1):
        argv = [str(rng.randint(lo, hi)) for _ in range(arity)]
        c = h._run([c_bin, *argv])
        if c.ub_trapped:
            hits += 1
            first = first if first is not None else t
            break
        if c.returncode != 0:
            continue
        r = h._run([tgt_bin, *argv])
        if not r.rust_outcome_defined:
            continue
        if c.stdout != r.stdout:
            hits += 1
            first = first if first is not None else t
            break
    return hits, first


# --------------------------------------------------------------------------
# Case study data model
# --------------------------------------------------------------------------
@dataclass
class CaseResult:
    item_id: str
    provenance: str
    divergence_class: str
    target_lang: str
    ub_reachable: bool
    target_defined: bool
    oracle_confirms_witness: bool
    oracle_silent_on_safe: bool
    oracle_confirm_ms: float
    fuzz_trials: int
    fuzz_hits: int
    fuzz_first_hit_trial: Optional[int]
    witness: Tuple[str, ...]
    safe: Tuple[str, ...]

    @property
    def walk_ok(self) -> bool:
        return (self.ub_reachable and self.target_defined
                and self.oracle_confirms_witness
                and self.oracle_silent_on_safe)

    @property
    def fuzzing_gap(self) -> bool:
        """The oracle confirmed a bug the equal-budget fuzzer never hit."""
        return self.oracle_confirms_witness and self.fuzz_hits == 0


@dataclass
class CaseStudiesReport:
    available: bool
    ok: bool
    n_cases: int
    n_walked: int
    n_gap: int
    cases: Tuple[CaseResult, ...] = field(default_factory=tuple)
    detail: str = ""


def _run_one(h: ReexecHarness, item_id: str, target_lang: str, *,
             trials: int, seed: int) -> CaseResult:
    it = _item(item_id)
    tgt = it.targets[target_lang]
    witness = tuple(a for a in it.ub_inputs if a != "")
    safe = tuple(a for a in it.safe_inputs if a != "")

    t0 = time.perf_counter()
    ub = h.confirm_trap_vs_defined(it.c_src, tgt, list(witness),
                                   it.klass, target_lang)
    sf = h.confirm_trap_vs_defined(it.c_src, tgt, list(safe),
                                   it.klass, target_lang)
    confirm_ms = (time.perf_counter() - t0) * 1000.0

    hits, first = 0, None
    if ub.available:
        with tempfile.TemporaryDirectory() as d:
            c_bin = h._compile_c(it.c_src, _C_UBSAN_FLAGS, d, "c_case")
            tgt_bin = h._compile_target(tgt, target_lang, d, "t_case")
            if c_bin and tgt_bin:
                hits, first = _differential_fuzz_binaries(
                    h, c_bin, tgt_bin, len(witness), target_lang,
                    trials=trials, seed=seed)

    return CaseResult(
        item_id=item_id,
        provenance=it.provenance,
        divergence_class=it.klass,
        target_lang=target_lang,
        ub_reachable=bool(ub.ub_reachable),
        target_defined=bool(ub.rust_defined),
        oracle_confirms_witness=bool(ub.confirmed),
        oracle_silent_on_safe=not bool(sf.confirmed),
        oracle_confirm_ms=round(confirm_ms, 1),
        fuzz_trials=trials,
        fuzz_hits=hits,
        fuzz_first_hit_trial=first,
        witness=witness,
        safe=safe,
    )


def confirm_case_studies(*, trials: int = 4000,
                         seed: int = 0) -> CaseStudiesReport:
    """Walk every case study end to end against real compilers and measure the
    cost/benefit vs an equal-budget differential fuzzer on the same binaries."""
    status = toolchain_available()
    needed = {lang for _, lang in _CASE_PLAN}
    if not all(status.full_for(lang) for lang in needed):
        return CaseStudiesReport(
            available=False, ok=True, n_cases=len(_CASE_PLAN),
            n_walked=0, n_gap=0,
            detail="target toolchain(s) absent; case studies not executed")

    h = ReexecHarness(status)
    cases: List[CaseResult] = []
    for item_id, lang in _CASE_PLAN:
        cases.append(_run_one(h, item_id, lang, trials=trials, seed=seed))

    n_walked = sum(1 for c in cases if c.walk_ok)
    n_gap = sum(1 for c in cases if c.fuzzing_gap)
    ok = (n_walked == len(cases)) and (n_gap >= 1)
    return CaseStudiesReport(
        available=True, ok=ok, n_cases=len(cases),
        n_walked=n_walked, n_gap=n_gap, cases=tuple(cases),
        detail="every case walked end-to-end; at least one fuzzing gap shown"
        if ok else "a case failed to walk or no fuzzing gap was demonstrated")


# --------------------------------------------------------------------------
# Reproducible document
# --------------------------------------------------------------------------
def _case_markdown(rep: CaseStudiesReport) -> str:
    lines = [
        "# Case studies — real migrations, walked end to end",
        "",
        "Three real-world-derived `C->target` migrations executed against real "
        "compilers (clang+UBSan, rustc, go), with the cost/benefit of the "
        "oracle measured against an **equal-budget differential fuzzer running "
        "on the same binaries**. Regenerated by "
        "`ub_oracle.case_studies.generate_case_studies()`.",
        "",
    ]
    if not rep.available:
        lines += ["> Target toolchain unavailable in this environment; the "
                  "studies are defined but were not executed here.", ""]
        return "\n".join(lines)

    lines += [
        "## Cost / benefit summary",
        "",
        "| case | provenance | class | pair | oracle | fuzzer "
        f"(equal budget) | benefit |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for c in rep.cases:
        fuzz = (f"hit at trial {c.fuzz_first_hit_trial}/{c.fuzz_trials}"
                if c.fuzz_hits else f"0/{c.fuzz_trials} (never hit)")
        oracle = (f"confirmed in {c.oracle_confirm_ms:.0f} ms"
                  if c.oracle_confirms_witness else "—")
        benefit = ("**found a bug fuzzing missed**" if c.fuzzing_gap
                   else "parity (dense UB)")
        lines.append(
            f"| `{c.item_id}` | {c.provenance.split('(')[0].strip()} | "
            f"`{c.divergence_class}` | c->{c.target_lang} | {oracle} | "
            f"{fuzz} | {benefit} |")
    lines.append("")
    lines.append(f"Walked end-to-end: **{rep.n_walked}/{rep.n_cases}**; "
                 f"false-negative gaps (oracle confirms, equal-budget fuzz "
                 f"misses): **{rep.n_gap}**.")
    lines.append("")

    for c in rep.cases:
        it = _item(c.item_id)
        lines += [
            f"## `{c.item_id}` — {c.divergence_class} (c->{c.target_lang})",
            "",
            f"**Provenance.** {c.provenance}",
            "",
            "**The C source (undefined on the witness input):**",
            "",
            "```c",
            it.c_src.strip(),
            "```",
            "",
            f"**The idiomatic {c.target_lang} translation (well-defined):**",
            "",
            f"```{c.target_lang}",
            it.targets[c.target_lang].strip(),
            "```",
            "",
            "**Walk.**",
            "",
            f"- Witnessing input `{' '.join(c.witness)}`: C UB reachable "
            f"(UBSan trapped) = **{c.ub_reachable}**; {c.target_lang} defined "
            f"= **{c.target_defined}**; oracle confirms divergence = "
            f"**{c.oracle_confirms_witness}** "
            f"(in {c.oracle_confirm_ms:.0f} ms).",
            f"- Safe input `{' '.join(c.safe)}`: oracle stays silent = "
            f"**{c.oracle_silent_on_safe}** (no false alarm).",
            f"- Equal-budget differential fuzz ({c.fuzz_trials} random int32 "
            f"trials on the same binaries): "
            + (f"hit at trial {c.fuzz_first_hit_trial}."
               if c.fuzz_hits else "**0 hits — never found the bug.**"),
            "",
        ]
    return "\n".join(lines)


def generate_case_studies(*, trials: int = 4000,
                          seed: int = 0) -> Tuple[Path, CaseStudiesReport]:
    rep = confirm_case_studies(trials=trials, seed=seed)
    _DOCS.mkdir(parents=True, exist_ok=True)
    _CASE_MD.write_text(_case_markdown(rep))
    return _CASE_MD, rep


if __name__ == "__main__":  # pragma: no cover
    path, rep = generate_case_studies()
    print(f"case-studies available={rep.available} ok={rep.ok} "
          f"walked={rep.n_walked}/{rep.n_cases} gaps={rep.n_gap}")
    for c in rep.cases:
        print(f"  - {c.item_id} (c->{c.target_lang}): walk_ok={c.walk_ok} "
              f"oracle={c.oracle_confirm_ms:.0f}ms "
              f"fuzz_hits={c.fuzz_hits}/{c.fuzz_trials} gap={c.fuzzing_gap}")
    print(f"wrote {path}")
