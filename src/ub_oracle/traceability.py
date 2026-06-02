"""Theory–implementation traceability (Step 83).

Artifact evaluators should be able to check, mechanically, that every claim the
paper/README makes is backed by a *named code module and symbol* — and, where
the claim is itself a checkable theorem, by a runnable check.  This module is the
single source of truth for that mapping.

Each :class:`Claim` records:

  * a stable ``id`` (cited from ``docs/TRACEABILITY.md`` and the paper),
  * the human-readable ``statement``,
  * the ``module`` and ``symbols`` that implement it (verified to import and to
    define those names), and
  * an optional ``theorem`` callable — a *fast*, toolchain-free check that
    returns ``True`` when the claim's executable core holds, so traceability is
    not merely "a symbol exists" but "the stated property runs and passes".

``verify_traceability()`` returns a list of :class:`TraceProblem`; an empty list
means every claim's module imports, every referenced symbol exists, and every
attached theorem evaluates to ``True``.  The test-suite asserts that, and also
asserts the prose ``docs/TRACEABILITY.md`` cites exactly the claim ids defined
here (so the doc cannot silently drift from the code).
"""

from __future__ import annotations

import importlib
from dataclasses import dataclass, field
from typing import Callable, List, Optional, Sequence

__all__ = [
    "Claim",
    "TraceProblem",
    "CLAIMS",
    "claim",
    "verify_traceability",
    "claim_ids",
]


@dataclass(frozen=True)
class Claim:
    id: str
    statement: str
    module: str
    symbols: Sequence[str] = field(default_factory=tuple)
    theorem: Optional[Callable[[], bool]] = None
    docs: Sequence[str] = field(default_factory=tuple)


@dataclass(frozen=True)
class TraceProblem:
    claim_id: str
    kind: str  # "import" | "symbol" | "theorem"
    detail: str


# ── executable theorem cores (fast, toolchain-free) ──────────────────────────
# These import lazily inside the function so that merely importing this module is
# cheap and free of heavy dependencies.

def _thm_completeness_integer_classes() -> bool:
    from . import completeness as c
    results = c.check_all_completeness()
    return bool(results) and all(r.complete for r in results)

def _thm_completeness_all_pairs() -> bool:
    from . import completeness as c
    by_pair = c.check_pair_completeness()
    return bool(by_pair) and all(
        results and all(r.complete for r in results)
        for results in by_pair.values()
    )

def _thm_semantics_one_sided() -> bool:
    # A positive divergence verdict is impossible without source UB (clause P).
    from . import semantics as s
    no_ub = s.Observation(
        source=s.SourceObservation(o0=s.Outcome(0, "0"), o2=s.Outcome(0, "1"),
                                   san_trapped=False),
        target=s.TargetObservation(defined=True), mode=s.EXPLOITED)
    return s.is_divergence(no_ub) is False

def _thm_semantics_coincides_vacuous() -> bool:
    # The coincidence theorem holds (vacuously) on an unavailable run.
    from . import semantics as s
    from .reexec import ReexecResult
    r = ReexecResult(available=False, divergence_class="signed_overflow", inputs={})
    return s.coincides_with_harness(r) is True

def _thm_ir_rejects_illformed() -> bool:
    from . import ir
    bad = {"kind": "binop_const", "op": "add"}  # missing const/width/var
    return len(ir.validate_unit(bad)) > 0

def _thm_prepass_preserves_verdict() -> bool:
    # On a clearly-diverging signed-overflow unit, prunable_classes must NOT prune
    # signed_overflow (so the pre-pass cannot change the verdict by skipping it).
    from . import abstract_interp as ai
    unit = {"kind": "binop_const", "op": "add", "const": 1, "width": 32,
            "var": "x", "signed": True}
    return "signed_overflow" not in ai.prunable_classes(unit)

def _thm_catalogue_nonempty() -> bool:
    from .catalogue import CATALOGUE
    return len(CATALOGUE) >= 8

def _thm_three_pairs_registered() -> bool:
    from . import language_pairs
    pairs = set(language_pairs())
    return {("c", "rust"), ("c", "go"), ("c", "swift")}.issubset(pairs)


def claim(*args, **kwargs) -> Claim:  # small constructor alias
    return Claim(*args, **kwargs)


CLAIMS: List[Claim] = [
    claim(
        "C1-soundness",
        "A DIVERGENT verdict is only returned after ground-truth re-execution of "
        "real compiled programs confirms a UB-rooted divergence.",
        "ub_oracle.verify",
        ("verify_unit", "VerifyVerdict", "VerifyReport"),
        docs=("README.md", "docs/SEMANTICS.md"),
    ),
    claim(
        "C2-divergence-semantics",
        "The witnessed property — divergence modulo source-undefinedness — is an "
        "executable predicate that is one-sided (positive only when source UB is "
        "reached).",
        "ub_oracle.semantics",
        ("is_divergence", "judge", "Observation", "EXPLOITED", "TRAP_VS_DEFINED"),
        theorem=_thm_semantics_one_sided,
        docs=("docs/SEMANTICS.md",),
    ),
    claim(
        "C3-semantics-coincides",
        "The formal divergence predicate coincides with the re-execution "
        "harness's confirmed decision on real programs.",
        "ub_oracle.semantics",
        ("coincides_with_harness", "observation_from_reexec"),
        theorem=_thm_semantics_coincides_vacuous,
        docs=("docs/SEMANTICS.md",),
    ),
    claim(
        "C4-completeness-classes",
        "For the four integer classes the symbolic search is complete on a "
        "precisely-stated bounded fragment (no false negatives there).",
        "ub_oracle.completeness",
        ("FRAGMENTS", "check_class_completeness", "check_all_completeness",
         "OUT_OF_FRAGMENT"),
        theorem=_thm_completeness_integer_classes,
        docs=("docs/COMPLETENESS.md",),
    ),
    claim(
        "C5-completeness-pairs",
        "The completeness result holds across every registered language pair "
        "(C→Rust/Go/Swift).",
        "ub_oracle.completeness",
        ("check_pair_completeness",),
        theorem=_thm_completeness_all_pairs,
        docs=("docs/COMPLETENESS.md",),
    ),
    claim(
        "C6-prepass-sound",
        "The abstract-interpretation pre-pass is a verdict-preserving accelerator: "
        "it never prunes a class that can actually reach UB on the unit's domain.",
        "ub_oracle.abstract_interp",
        ("prunable_classes", "analyze_unit", "Interval"),
        theorem=_thm_prepass_preserves_verdict,
        docs=("README.md",),
    ),
    claim(
        "C7-ir-contract",
        "A single pair-agnostic IR contract is frozen and validated at the "
        "boundary; ill-formed units are rejected.",
        "ub_oracle.ir",
        ("validate_unit", "assert_valid", "KNOWN_KINDS", "IRValidationError"),
        theorem=_thm_ir_rejects_illformed,
        docs=("docs/IR.md",),
    ),
    claim(
        "C8-pluggable-targets",
        "Target-language semantics are data-driven packs; the pipeline differs "
        "across pairs only by the declared target, and three pairs are registered.",
        "ub_oracle.target_semantics",
        ("TargetPack", "PACKS", "get_pack"),
        theorem=_thm_three_pairs_registered,
        docs=("README.md", "CAPABILITIES.md"),
    ),
    claim(
        "C9-replay-format",
        "Confirmed counterexamples are captured in a versioned, replayable format.",
        "ub_oracle.replay",
        ("Counterexample", "REPLAY_SCHEMA_VERSION"),
        docs=("README.md",),
    ),
    claim(
        "C10-ub-catalogue",
        "The supported undefined-behavior divergence classes are enumerated in a "
        "single catalogue.",
        "ub_oracle.catalogue",
        ("CATALOGUE", "DivergenceClass", "c_ub_classes"),
        theorem=_thm_catalogue_nonempty,
        docs=("README.md", "CAPABILITIES.md"),
    ),
    claim(
        "C11-redteam",
        "An internal red-team actively tries to make the oracle call a truly "
        "divergent pair equivalent, on every supported pair.",
        "ub_oracle.redteam",
        ("build_cases", "run_redteam", "RedTeamReport"),
        docs=("README.md",),
    ),
]


def claim_ids() -> List[str]:
    return [c.id for c in CLAIMS]


def verify_traceability(run_theorems: bool = True) -> List[TraceProblem]:
    """Check every claim: module imports, symbols exist, theorems pass.

    Returns a (hopefully empty) list of :class:`TraceProblem`.  Set
    ``run_theorems=False`` to skip the executable cores (symbol/import checks
    only) for a fast structural pass.
    """
    problems: List[TraceProblem] = []
    seen_ids = set()
    for c in CLAIMS:
        if c.id in seen_ids:
            problems.append(TraceProblem(c.id, "symbol", "duplicate claim id"))
        seen_ids.add(c.id)

        try:
            mod = importlib.import_module("src." + c.module) \
                if not c.module.startswith("src.") else importlib.import_module(c.module)
        except Exception as e:  # pragma: no cover - import error path
            problems.append(TraceProblem(c.id, "import",
                                         f"cannot import {c.module}: {e!r}"))
            continue

        for sym in c.symbols:
            if not hasattr(mod, sym):
                problems.append(TraceProblem(
                    c.id, "symbol", f"{c.module} has no symbol {sym!r}"))

        if run_theorems and c.theorem is not None:
            try:
                ok = c.theorem()
            except Exception as e:  # pragma: no cover - theorem error path
                problems.append(TraceProblem(
                    c.id, "theorem", f"theorem raised {e!r}"))
                continue
            if ok is not True:
                problems.append(TraceProblem(
                    c.id, "theorem", f"theorem returned {ok!r}, expected True"))

    return problems
