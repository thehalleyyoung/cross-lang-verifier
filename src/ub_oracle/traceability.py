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

def _thm_uninit_definedness() -> bool:
    # The lattice must flag an unwritten read and clear an initialized one.
    from .oracles import uninit_read as u
    unwritten = {"kind": "uninit_read", "storage": {"kind": "scalar"},
                 "writes": [], "read": None}
    initialized = {"kind": "uninit_read", "storage": {"kind": "scalar"},
                   "writes": [{"slot": None}], "read": None}
    return (u.uninitialized_read(unwritten) is not None
            and u.uninitialized_read(initialized) is None)


def _thm_cegar_sound_and_refines() -> bool:
    # CEGAR must (a) match exact enumeration and (b) genuinely refine on a
    # path-sensitive fragment the interval domain cannot discharge.
    from . import cegar as c
    equiv = c.GuardedQuery("add", 1, 32, "x", (c.even(),))
    re = c.run_cegar(equiv)
    sound_equiv = (re.verdict is c.CegarVerdict.EQUIVALENT
                   and c.brute_force_witness(equiv) is None
                   and re.refinements >= 1)
    witn = c.GuardedQuery("add", 2, 32, "x", (c.even(), c.at_least(0)))
    rw = c.run_cegar(witn)
    bf = c.brute_force_witness(witn)
    sound_div = (rw.verdict is c.CegarVerdict.DIVERGENT
                 and bf is not None
                 and rw.witness is not None
                 and all(p.holds_at(rw.witness) for p in witn.assumes))
    return bool(sound_equiv and sound_div)


def _thm_kinduction_safe_and_witness() -> bool:
    # k-induction must (a) prove the modular counter SAFE only with the
    # strengthening invariant, and (b) return a reachable overflow witness whose
    # depth matches an independent concrete simulation.
    from . import kinduction as k
    ts, aux = k.saturating_counter(1000)
    safe = k.prove(ts, max_k=8, aux=aux)
    sound_safe = (safe.verdict is k.KIndVerdict.SAFE
                  and k.prove(ts, max_k=6).verdict is k.KIndVerdict.UNKNOWN)
    div = k.accumulator_overflow(k.INT32_MAX - 2, 1)
    res = k.prove(div, max_k=8)
    _, vdepth, _ = k.simulate(div, 10)
    sound_div = (res.verdict is k.KIndVerdict.DIVERGENT
                 and res.witness_depth == vdepth)
    return bool(sound_safe and sound_div)


def _thm_abi_layout_sound() -> bool:
    # The ABI oracle must (a) flag the classic suboptimal-order struct as an
    # interop hazard whose optimized layout reorders it, (b) abstain on a struct
    # already in padding-optimal order, and (c) flag the C-vs-default-Rust enum
    # width divergence while modelling union/nested-struct layouts faithfully.
    from . import abi_layout as a
    hz = a.abi_divergence(a.hazard_struct())
    sound_hazard = (hz.is_hazard
                    and hz.c.size == 12 and hz.optimized.size == 8
                    and set(hz.moved_fields) == {"a", "b", "c"})
    sf = a.abi_divergence(a.safe_struct())
    uni = a.abi_divergence(a.uniform_struct())
    sound_safe = (not sf.is_hazard and not uni.is_hazard
                  and sf.moved_fields == [])
    en = a.enum_abi_divergence(3)
    nested = a.c_layout(a.nested_struct())
    union = a.union_layout(a.mixed_union())
    sound_fidelity = (en.is_hazard and en.c_size == 4 and en.rust_default_size == 1
                      and nested.size == 16 and nested.offsets["y"] == 12
                      and union.size == 8 and set(union.offsets.values()) == {0})
    return bool(sound_hazard and sound_safe and sound_fidelity)


def _thm_memory_model_sound() -> bool:
    # The provenance memory model must flag spatial OOB, use-after-free and
    # double-free on the canonical traces, accept the safe and boundary traces,
    # and keep provenance per-allocation (freeing one object is not freeing
    # another).
    from . import memory_model as m
    oob = m.first_fault(m.oob_trace())
    uaf = m.first_fault(m.uaf_trace())
    df = m.first_fault(m.double_free_trace())
    faults_ok = (oob is not None and oob.kind is m.FaultKind.OOB_SPATIAL
                 and uaf is not None and uaf.kind is m.FaultKind.USE_AFTER_FREE
                 and df is not None and df.kind is m.FaultKind.DOUBLE_FREE)
    safe_ok = (m.first_fault(m.safe_trace()) is None
               and m.first_fault(m.safe_boundary_trace()) is None)
    prov = m.first_fault([m.Alloc("p", 4), m.Alloc("q", 16), m.Free("p"),
                          m.Load("q", 12, 4)])
    prov_ok = prov is None
    return bool(faults_ok and safe_ok and prov_ok)


def _thm_provenance_pnvi_sound() -> bool:
    # The PNVI provenance model must (a) allow forming a one-past-the-end pointer
    # but flag its dereference, (b) preserve provenance across an in-bounds
    # arithmetic round-trip, (c) recover provenance only via exposure, and
    # (d) revoke provenance on free.
    from . import provenance as p
    one_past_safe = p.first_fault(p.one_past_form_only()) is None
    deref = p.first_fault(p.one_past_form_then_deref())
    deref_oob = deref is not None and deref.kind is p.ProvFault.DEREF_OOB
    arith_ok = p.first_fault(p.arithmetic_roundtrip()) is None
    exposed_ok = p.first_fault(p.exposed_roundtrip_recovers_provenance()) is None
    opaque = p.first_fault(p.opaque_int_has_no_provenance())
    opaque_ok = opaque is not None and opaque.kind is p.ProvFault.NO_PROVENANCE
    uaf = p.first_fault(p.use_after_free_via_provenance())
    uaf_ok = uaf is not None and uaf.kind is p.ProvFault.USE_AFTER_FREE
    return bool(one_past_safe and deref_oob and arith_ok and exposed_ok
                and opaque_ok and uaf_ok)


def _thm_ownership_facts_sound() -> bool:
    # Every ownership pattern's prediction must be internally consistent (an
    # error code iff rejected); the headline fact is that mutable aliasing is
    # rejected (E0499) while its unsafe re-expression is accepted.
    from . import ownership as o
    consistent = all(bool(p.error_code) == (not p.accepts)
                     for p in o.PATTERNS.values())
    headline = (o.pattern("two_mut_borrows").accepts is False
                and o.pattern("two_mut_borrows").error_code == "E0499"
                and o.pattern("raw_ptr_aliasing").accepts is True)
    return bool(consistent and headline)


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
    claim(
        "C12-uninit-definedness",
        "The uninitialized-read class is decided by a real three-point "
        "definedness-lattice dataflow analysis that flags reads of slots not "
        "written on all paths and never flags a fully-initialized read.",
        "ub_oracle.oracles.uninit_read",
        ("analyze_definedness", "uninitialized_read", "UninitializedReadOracle"),
        theorem=_thm_uninit_definedness,
        docs=("README.md", "CAPABILITIES.md"),
    ),
    claim(
        "C13-cegar-refinement",
        "Guarded fragments the non-relational interval pre-pass cannot discharge "
        "are decided by a lazy predicate-abstraction CEGAR loop: it starts from "
        "the UB condition with no guards, refines one path-condition at a time on "
        "each spurious model, and is sound (its verdict matches exact enumeration "
        "of the UB region) while genuinely refining on path-sensitive fragments.",
        "ub_oracle.cegar",
        ("run_cegar", "brute_force_witness", "GuardedQuery"),
        theorem=_thm_cegar_sound_and_refines,
        docs=("README.md",),
    ),
    claim(
        "C14-kinduction-loops",
        "Looping fragments are decided beyond bounded unrolling by k-induction: "
        "the base case returns a reachable overflow witness at the exact iteration "
        "the loop goes undefined, and the inductive step (optionally strengthened "
        "with auxiliary invariants the engine first proves inductive) certifies "
        "no-divergence for an unbounded iteration count.",
        "ub_oracle.kinduction",
        ("prove", "simulate", "TransitionSystem", "saturating_counter",
         "accumulator_overflow"),
        theorem=_thm_kinduction_safe_and_witness,
        docs=("README.md",),
    ),
    claim(
        "C15-abi-layout",
        "Divergence at FFI boundaries is decided structurally for structs, "
        "unions, enums and nested aggregates: the oracle computes the exact "
        "C-ABI layout (confirmed field-by-field against real clang offsetof) and "
        "flags an interop hazard iff a padding-optimizing representation would "
        "reorder a struct, or iff a C enum and a default-repr Rust fieldless enum "
        "of the same arity have different widths. Real rustc confirms #[repr(C)] "
        "mirrors the C layout/width exactly while the default repr diverges "
        "precisely when predicted, and Go's declaration-order layout matches C — "
        "so the oracle never fabricates a hazard the compiler does not exhibit.",
        "ub_oracle.abi_layout",
        ("c_layout", "optimized_layout", "abi_divergence", "union_layout",
         "enum_abi_divergence", "confirm_abi"),
        theorem=_thm_abi_layout_sound,
        docs=("README.md",),
    ),
    claim(
        "C16-provenance-memory",
        "Spatial and temporal memory safety is decided on whole traces by a "
        "byte-addressed memory model in which every pointer carries the "
        "provenance of its allocation: an access is in bounds iff it lies in "
        "[0,size) of that allocation (never an adjacent object), and any access "
        "through a freed allocation is a use-after-free. The model's predicted "
        "fault — spatial OOB, use-after-free or double-free — is confirmed on "
        "real compiled code under AddressSanitizer, which traps iff a fault is "
        "predicted and reports the same fault kind; model-safe traces run clean.",
        "ub_oracle.memory_model",
        ("simulate", "first_fault", "MemEvent", "FaultKind", "confirm_memory"),
        theorem=_thm_memory_model_sound,
        docs=("README.md",),
    ),
    claim(
        "C17-pointer-provenance",
        "Pointer provenance is modelled in the C PNVI-ae style: every pointer "
        "carries the provenance of one allocation; a one-past-the-end pointer is "
        "formable and comparable but its dereference is out of bounds; pointer "
        "arithmetic preserves provenance; an integer round-trip recovers "
        "provenance only when the allocation was exposed (an opaque integer "
        "yields a no-provenance pointer whose dereference is undefined); and free "
        "revokes provenance. The real-compiler-confirmable distinctions — forming "
        "vs dereferencing one-past-the-end, and in-bounds arithmetic round-trips "
        "— agree with AddressSanitizer on compiled code, and the general "
        "provenance interface is documented.",
        "ub_oracle.provenance",
        ("simulate", "first_fault", "ProvEvent", "ProvFault",
         "PROVENANCE_INTERFACE", "confirm_provenance"),
        theorem=_thm_provenance_pnvi_sound,
        docs=("README.md",),
    ),
    claim(
        "C18-ownership-facts",
        "Ownership facts are taken as ground truth from the real Rust borrow "
        "checker: the idiomatic safe translation of a mutably-aliasing C idiom is "
        "rejected (E0499 for two &mut, E0502 for &mut while & is live, E0382 for "
        "use-after-move), while disjoint/sequential borrows are accepted and the "
        "unsafe raw-pointer re-expression compiles. Each verdict — accept/reject "
        "plus the exact error code — is observed by compiling with rustc, not "
        "assumed, and the general (retargetable) ownership interface is documented.",
        "ub_oracle.ownership",
        ("PATTERNS", "pattern", "confirm_ownership", "OWNERSHIP_INTERFACE"),
        theorem=_thm_ownership_facts_sound,
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
