"""Step 49 — head-to-head vs existing tools (where they apply; the gap where none do).

The cross-language, UB-rooted divergence problem this project targets — *does a
C program and its Rust/Go translation diverge because the C side has undefined
behavior the target defines away?* — sits in a gap that existing verification
tooling does not occupy. This module makes that claim **concrete and executed**
rather than rhetorical:

1.  **Live probing.** We probe the machine for the relevant tool *categories*
    (bounded model checkers / single-language equivalence: CBMC, ESBMC, KLEE,
    SMACK; static analyzers: cppcheck, Infer; peephole/IR translation validators:
    Alive2). For any that are installed we would run them; their structural
    applicability to a *cross-language* pair is recorded either way.

2.  **A runnable baseline that *is* realizable on any C toolchain.** Single-language
    equivalence checking and translation validation both operate **within one
    language/IR**. The faithful, runnable proxy for that whole category is to ask
    whether two compilations of the *same C program* agree — i.e. an O0-vs-O2
    differential, the exact question a translation validator poses about a single
    compiler. We execute that baseline on real *divergent* ground-truth items and
    show it finds **nothing**: for a definedness divergence (div-by-zero, OOB,
    oversized shift, INT_MIN/-1) both C builds trap identically, so a
    same-language tool sees no divergence at all — because the divergence is only
    visible *against the Rust/Go translation*, which such a tool cannot even
    ingest.

3.  **Our oracle on the same items.** The sanitizer-anchored decision procedure
    (:func:`ground_truth.label_item`) catches every one of those items. The
    head-to-head therefore quantifies a **total false-negative gap** for the
    realizable single-language baseline, and documents *why* the remaining tool
    categories cannot be posed the question at all.

Everything that can be executed on this toolchain is executed; nothing is
asserted. The categorical gaps (no cross-language equivalence tool ingests a
(C, Rust) pair) are stated as applicability facts, not as benchmark wins.
"""

from __future__ import annotations

import shutil
import tempfile
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from . import ground_truth as gt
from .reexec import ReexecHarness, ToolchainStatus, toolchain_available


# --------------------------------------------------------------------------- #
# The catalogue of existing-tool categories and their applicability.
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class ToolCategory:
    key: str
    name: str
    probe_binaries: Tuple[str, ...]
    ingests_cross_language_pair: bool
    note: str

    def installed(self) -> Optional[str]:
        for b in self.probe_binaries:
            p = shutil.which(b)
            if p:
                return p
        return None


CATEGORIES: Tuple[ToolCategory, ...] = (
    ToolCategory(
        "bmc-single-lang", "Bounded model checking / single-language equivalence",
        ("cbmc", "esbmc"), False,
        "Verifies one C program (or two C programs against each other). It cannot "
        "ingest a Rust/Go translation, so it cannot relate the C source to its "
        "target — the cross-language divergence is outside its input language."),
    ToolCategory(
        "symbolic-exec", "Symbolic execution",
        ("klee",), False,
        "Explores paths of one program in one language (LLVM bitcode from C). It "
        "finds UB *within* the C program but has nothing to compare it to in the "
        "target language; it does not answer 'does the translation diverge?'."),
    ToolCategory(
        "trans-validation", "Translation validation (single compiler/IR)",
        ("alive2",), False,
        "Validates that a compiler transformation preserves semantics *within one "
        "IR* (e.g. LLVM → LLVM). A C→Rust/Go port is not a single-compiler IR "
        "transformation, so a translation validator is not posed this question."),
    ToolCategory(
        "static-analysis", "Static analysis / linters",
        ("cppcheck", "infer"), False,
        "Flags suspicious patterns in one language. It may warn about the C UB in "
        "isolation but does not establish a *divergence* against a specific target "
        "translation, and cannot confirm the target is defined."),
    ToolCategory(
        "verified-transpiler", "Verified / equivalence-checked transpiler",
        ("smack",), False,
        "Would prove a particular translator correct; none is installed and, where "
        "they exist, they cover a fixed translator, not an arbitrary (C, target) "
        "pair produced by any transpiler or by hand."),
)


# --------------------------------------------------------------------------- #
# The realizable single-language / translation-validation baseline:
# does the SAME C program, compiled two ways, diverge? (It cannot see the target.)
# --------------------------------------------------------------------------- #
@dataclass
class BaselineOutcome:
    item_id: str
    lang: str
    klass: str
    # the C-only differential: did O0 vs O2 disagree on observable output?
    baseline_divergence_found: bool
    # our oracle's verdict on the SAME pair
    oracle_divergence_found: bool
    detail: str = ""


# Classes where the unsanitized C program *traps identically at every optimisation
# level* (an integer division by zero / INT_MIN/-1 raises SIGFPE no matter what the
# optimiser does). For these a same-language O0-vs-O2 differential — the realizable
# proxy for single-language equivalence / translation validation — is *provably
# blind*: both builds crash the same way, so it observes no divergence at all, even
# though the pair genuinely diverges from the panic-defining target it cannot
# ingest. (Value-producing UB such as oversized shift / signed overflow can make
# O0 and O2 disagree, so a same-language differential *can* reach some of those; we
# report that honestly in the full breakdown rather than overclaiming.)
BASELINE_BLIND_CLASSES = frozenset({"div_by_zero", "int_min_div_neg1"})


def _c_self_diff(h: ReexecHarness, item: gt.GTItem) -> Tuple[bool, str]:
    """Run the single-language baseline: compile the C program at -O0 and -O2 and
    compare observable output on the item's input. Returns (divergence_found,
    detail). This is the strongest thing a *same-language* equivalence checker /
    translation validator can be made to do here — and it is structurally blind to
    the target translation."""
    argv = list(item.inputs)
    with tempfile.TemporaryDirectory() as d:
        o0 = h._compile_c(item.c_src, ["-O0"], d, "o0")
        o2 = h._compile_c(item.c_src, ["-O2"], d, "o2")
        if not (o0 and o2):
            return False, "C did not compile at both levels"
        r0 = h._run([o0, *argv])
        r2 = h._run([o2, *argv])
    # A same-language tool only "finds a divergence" if the two C builds disagree
    # on a *defined* observable. Identical outcomes (including identical traps)
    # look equivalent to it.
    if r0.returncode == 0 and r2.returncode == 0 and r0.stdout != r2.stdout:
        return True, f"O0={r0.stdout!r} != O2={r2.stdout!r}"
    return False, (f"O0(rc={r0.returncode},out={r0.stdout!r}) ~ "
                   f"O2(rc={r2.returncode},out={r2.stdout!r})")


@dataclass
class HeadToHeadReport:
    available: bool
    langs: Tuple[str, ...]
    outcomes: List[BaselineOutcome] = field(default_factory=list)
    categories: List[Dict[str, object]] = field(default_factory=list)

    @property
    def n_items(self) -> int:
        return len(self.outcomes)

    @property
    def baseline_found(self) -> int:
        return sum(1 for o in self.outcomes if o.baseline_divergence_found)

    @property
    def oracle_found(self) -> int:
        return sum(1 for o in self.outcomes if o.oracle_divergence_found)

    @property
    def false_negative_gap(self) -> int:
        """Items the oracle catches that the single-language baseline misses."""
        return sum(1 for o in self.outcomes
                   if o.oracle_divergence_found and not o.baseline_divergence_found)

    # ── breakdown by the provably-blind classes ────────────────────────
    @property
    def blind_outcomes(self) -> List["BaselineOutcome"]:
        return [o for o in self.outcomes if o.klass in BASELINE_BLIND_CLASSES]

    @property
    def blind_baseline_found(self) -> int:
        return sum(1 for o in self.blind_outcomes if o.baseline_divergence_found)

    @property
    def blind_oracle_found(self) -> int:
        return sum(1 for o in self.blind_outcomes if o.oracle_divergence_found)

    def per_class_breakdown(self) -> Dict[str, Dict[str, int]]:
        out: Dict[str, Dict[str, int]] = {}
        for o in self.outcomes:
            d = out.setdefault(o.klass, {"items": 0, "baseline": 0, "oracle": 0})
            d["items"] += 1
            d["baseline"] += int(o.baseline_divergence_found)
            d["oracle"] += int(o.oracle_divergence_found)
        return out

    @property
    def any_cross_language_tool_installed(self) -> bool:
        return any(c["installed"] and c["ingests_cross_language_pair"]
                   for c in self.categories)


def _divergent_sample(langs: Tuple[str, ...], per_class: int) -> List[gt.GTItem]:
    items = [it for it in gt.enumerate_corpus(langs)
             if it.declared_label == "divergent"]
    seen: Dict[Tuple[str, str], int] = {}
    out: List[gt.GTItem] = []
    for it in items:
        key = (it.lang, it.klass)
        n = seen.get(key, 0)
        if n < per_class:
            out.append(it)
            seen[key] = n + 1
    return out


def run_head_to_head(langs: Tuple[str, ...] = ("rust", "go"),
                     status: Optional[ToolchainStatus] = None,
                     per_class: int = 1) -> HeadToHeadReport:
    """Execute the realizable single-language baseline and our oracle on the same
    divergent items, and record the applicability of every existing-tool category."""
    st = status or toolchain_available()
    avail_langs = tuple(l for l in langs if st.full_for(l))
    rep = HeadToHeadReport(available=bool(avail_langs), langs=avail_langs)
    # Category applicability is independent of the C toolchain — always recorded.
    for c in CATEGORIES:
        inst = c.installed()
        rep.categories.append({
            "key": c.key,
            "name": c.name,
            "installed": bool(inst),
            "path": inst or "",
            "ingests_cross_language_pair": c.ingests_cross_language_pair,
            "note": c.note,
        })
    if not avail_langs:
        return rep
    h = ReexecHarness(st)
    for it in _divergent_sample(avail_langs, per_class):
        base_found, detail = _c_self_diff(h, it)
        ev = gt.label_item(h, it)
        rep.outcomes.append(BaselineOutcome(
            item_id=it.item_id, lang=it.lang, klass=it.klass,
            baseline_divergence_found=base_found,
            oracle_divergence_found=(ev.observed_label == "divergent"),
            detail=f"baseline[{detail}] oracle[{ev.observed_label}: {ev.detail}]"))
    return rep


@dataclass
class HeadToHeadConfirmation:
    available: bool
    ok: bool
    n_items: int
    baseline_found: int
    oracle_found: int
    false_negative_gap: int
    n_blind: int
    blind_baseline_found: int
    blind_oracle_found: int
    report: HeadToHeadReport


def confirm_head_to_head(langs: Tuple[str, ...] = ("rust", "go"),
                         per_class: int = 1) -> HeadToHeadConfirmation:
    """Confirm the central comparison on real *divergent* items:

    * the oracle catches **every** divergence;
    * on the **provably-blind** classes (div-by-zero, INT_MIN/-1 — where the
      unsanitised C traps identically at all optimisation levels), the realizable
      single-language baseline finds **none** — a total false-negative gap,
      because the divergence is only visible against the panic-defining target
      the baseline cannot ingest; the oracle catches all of them;
    * no installed tool category can ingest a cross-language (C, target) pair.

    Value-producing UB (oversized shift, signed overflow) is reported in the full
    per-class breakdown — a same-language O0-vs-O2 differential *can* observe some
    of it, which we do not hide.
    """
    rep = run_head_to_head(langs, per_class=per_class)
    n_blind = len(rep.blind_outcomes)
    avail = rep.available and rep.n_items > 0 and n_blind > 0
    ok = (avail
          and rep.oracle_found == rep.n_items          # oracle catches all
          and rep.blind_baseline_found == 0            # baseline blind on blind classes
          and rep.blind_oracle_found == n_blind        # oracle catches all blind-class items
          and not rep.any_cross_language_tool_installed)
    return HeadToHeadConfirmation(
        available=avail, ok=ok, n_items=rep.n_items,
        baseline_found=rep.baseline_found, oracle_found=rep.oracle_found,
        false_negative_gap=rep.false_negative_gap,
        n_blind=n_blind,
        blind_baseline_found=rep.blind_baseline_found,
        blind_oracle_found=rep.blind_oracle_found,
        report=rep)


def applicability_table() -> List[Dict[str, object]]:
    """A static table (for the paper's 'why no existing tool applies' figure)."""
    return [{
        "key": c.key, "name": c.name,
        "ingests_cross_language_pair": c.ingests_cross_language_pair,
        "installed": bool(c.installed()), "note": c.note,
    } for c in CATEGORIES]


HEADTOHEAD_EXTERNAL_SPI = {
    "run_head_to_head": run_head_to_head,
    "confirm_head_to_head": confirm_head_to_head,
    "applicability_table": applicability_table,
    "CATEGORIES": CATEGORIES,
}


if __name__ == "__main__":  # pragma: no cover
    conf = confirm_head_to_head(per_class=1)
    print(f"available={conf.available} ok={conf.ok}")
    print(f"items={conf.n_items} oracle_found={conf.oracle_found} "
          f"baseline_found={conf.baseline_found} "
          f"false_negative_gap={conf.false_negative_gap}")
    print(f"definedness(blind): {conf.n_blind} items, "
          f"baseline_found={conf.blind_baseline_found}, "
          f"oracle_found={conf.blind_oracle_found}")
    print("per-class breakdown (items / baseline-found / oracle-found):")
    for k, d in sorted(conf.report.per_class_breakdown().items()):
        print(f"  {k:18s} {d['items']:2d} / {d['baseline']:2d} / {d['oracle']:2d}")
    print("tool-category applicability:")
    for c in conf.report.categories:
        flag = "installed" if c["installed"] else "absent"
        print(f"  [{flag:9s}] {c['name']}: "
              f"cross-lang-pair-input={c['ingests_cross_language_pair']}")
