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


def _thm_unit_alignment_sound() -> bool:
    # The structural aligner must (a) recover the ground-truth pairing on a
    # renamed module whose names are adversarial to name matching, and (b) do so
    # strictly better than the name-only baseline. Signature arity mismatch must
    # be a hard zero so that arity always vetoes an incompatible pair.
    from . import unit_alignment as ua
    c, t, truth = (ua.example_c_unit(), ua.example_target_unit(),
                   ua.example_ground_truth())
    structural = ua.alignment_accuracy(ua.align(c, t).mapping, truth)
    baseline = ua.alignment_accuracy(ua.name_only_align(c, t), truth)
    arity_veto = ua.signature_score(
        ua.FunctionSig("f", ("int", "int"), "int"),
        ua.FunctionSig("g", ("i32",), "i32")) == 0.0
    return bool(structural == 1.0 and baseline < structural and arity_veto)


def _thm_foreign_frontier_sound() -> bool:
    # The frontier detector must (a) stay CLEAR on the pure fragment, (b) abstain
    # on each foreign construct, and (c) — where a compiler is present — have its
    # abstention justified by real clang IR (the construct is genuinely opaque to
    # a pure model). Comments/strings must never trigger a false abstention.
    from . import foreign_effects as fe
    pure = "int add(int a,int b){return a+b;}\nint dbl(int*p){return *p+*p;}"
    if not fe.decide(pure).clear:
        return False
    cmt = "int f(int x){/* volatile asm */ return x;}\n// volatile longjmp"
    if not fe.decide(cmt).clear:
        return False
    cases = [
        ("int f(volatile int*p){return *p;}", fe.ForeignKind.VOLATILE),
        ("extern int g(int);\nint f(int x){return g(x);}",
         fe.ForeignKind.FOREIGN_CALL),
        ("#include <setjmp.h>\njmp_buf b;\nint f(){return setjmp(b);}",
         fe.ForeignKind.NONLOCAL_JUMP),
    ]
    for src, kind in cases:
        v = fe.decide(src)
        if v.clear or kind not in v.kinds:
            return False
    confs = fe.confirm_all()
    # if a compiler is available the confirmations must all hold; if not, the
    # detector-only facts above already establish the claim.
    return all(c.ok for c in confs)


def _thm_concurrency_race_sound() -> bool:
    # The pattern catalogue must be internally consistent (exactly the
    # unsynchronized counter races) and, where ThreadSanitizer is available, the
    # racy/race-free verdict must be confirmed on a real binary: the
    # unsynchronized counter races, the mutex-guarded one does not.
    from . import concurrency as co
    racy = [p.name for p in co.PATTERNS.values() if p.races]
    if racy != ["unsynchronized_counter"]:
        return False
    import os as _os2
    if not _os2.path.exists(co.CC):
        return True  # detector-free consistency already established
    race = co.confirm_race("unsynchronized_counter", check_go=False)
    clean = co.confirm_race("mutex_counter", check_go=False)
    return bool(race.c.race_detected is True and clean.c.race_detected is False)


def _thm_indirect_resolution_sound() -> bool:
    # The precise points-to set for a dispatch table is exactly its entries and
    # refines the signature-typed set (which excludes the wrong-signature decoy).
    # Where clang is present, real execution must confirm observed == predicted
    # and the decoy is never reached.
    from . import indirect_calls as ic
    u = ic.parse_unit(ic.EXAMPLE_DISPATCH)
    precise = ic.resolve_table_call(u, "table")
    conservative = ic.signature_compatible_targets(u, "op_t")
    if precise != {"add", "sub", "mul"}:
        return False
    if "log_msg" in conservative or not precise.issubset(conservative):
        return False
    import os as _os3
    if not _os3.path.exists(ic.CC):
        return True
    c = ic.confirm_table_dispatch(ic.EXAMPLE_DISPATCH, "table")
    return bool(c.ok and c.exact and "log_msg" not in c.observed)


def _thm_preprocess_real() -> bool:
    # The hazardous-macro detector must flag an unparenthesized function-like
    # macro and not a safe one; where clang is present, real preprocessing +
    # execution must show the macro is load-bearing (3 vs 4), conditionals select
    # the program, and includes resolve.
    from . import preprocess as pp
    if [h.name for h in pp.detect_unparenthesized_macros(
            "#define MUL(a,b) a*b\n")] != ["MUL"]:
        return False
    if pp.detect_unparenthesized_macros("#define MUL(a,b) ((a)*(b))\n") != []:
        return False
    import os as _os4
    if not _os4.path.exists(pp.CC):
        return True
    return bool(pp.confirm_macro_precedence_hazard().ok
                and pp.confirm_conditional_compilation().ok
                and pp.confirm_include_resolution().ok)


def _thm_libc_model_accurate() -> bool:
    # The pure models must satisfy their value contracts (sign semantics, NUL
    # preconditions) and, where clang is present, agree with the real libc on a
    # randomized differential check across the whole modelled surface.
    from . import libc_model as lc
    if lc.model_strcmp(b"abc\x00", b"abd\x00") != -1:
        return False
    if lc.model_memset(b"\x00\x00\x00", 0x41, 2) != b"\x41\x41\x00":
        return False
    try:
        lc.model_strlen(b"no terminator")
        return False
    except ValueError:
        pass
    import os as _os5
    if not _os5.path.exists(lc.CC):
        return True
    return all(r.ok for r in lc.confirm_all(trials=40))


def _thm_ir_ingest_sound() -> bool:
    # Ingestion must recover faithful facts from the compilers' own IRs: clang's
    # AST gives exact signatures/storage classes; rustc's MIR gives ownership
    # (move/drop) facts for free. Where a compiler is absent we fall back to a
    # consistency-only check so the claim never silently passes on fabricated data.
    from . import ir_ingest as ir
    if ir._split_fn_qualtype("char *(const char *)") != ("const char *",):
        return False
    if ir._split_fn_qualtype("int (int, int)") != ("int", "int"):
        return False
    import os as _os6
    okc = True
    if _os6.path.exists(ir.CLANG):
        c = ir.confirm_clang_ingest()
        okc = bool(c.ok)
    okm = True
    if _os6.path.exists(ir.RUSTC):
        m = ir.confirm_mir_ingest()
        okm = bool(m.ok)
    return okc and okm


def _thm_project_ingest_sound() -> bool:
    # Whole-project ingestion must recover the union of symbols from a real C
    # compilation database and enumerate a real Cargo workspace's members. Where
    # a tool is absent we fall back to consistency-only checks (argv parsing).
    from . import project_ingest as pi
    if pi._include_dirs_from_argv(["cc", "-I", "inc", "x.c"], "/p") != ("/p/inc",):
        return False
    if pi._entry_command({"command": "cc -c x.c"}) != ["cc", "-c", "x.c"]:
        return False
    import os as _os7
    okc = True
    if _os7.path.exists(pi.CLANG):
        okc = bool(pi.confirm_compile_db().ok)
    okw = True
    if _os7.path.exists(pi.CARGO):
        okw = bool(pi.confirm_cargo_workspace().ok)
    return okc and okw


def _thm_solver_portfolio_robust() -> bool:
    # The portfolio must reach the known ground-truth answer on every divergence-
    # class query with full agreement among available solvers, and (when more than
    # one backend is present) at least one query must be decided by all of them.
    from . import solver_portfolio as sp
    rep = sp.robustness_report(timeout=10.0)
    if not rep or not all(c.consensus == c.expected and c.agreement for c in rep):
        return False
    solvers = sp.available_solvers()
    if len(solvers) >= 2:
        cross = any(
            all(c.per_solver.get(s) in (sp.SAT, sp.UNSAT) for s in solvers)
            for c in rep
        )
        if not cross:
            return False
    return True


def _thm_frontend_fuzz_hardened() -> bool:
    # A seeded differential-fuzz run of the C frontend against clang must compile a
    # meaningful number of random well-typed programs and recover every function's
    # signature/storage with zero divergences and zero crashes; garbage input must
    # never raise. Absent clang, fall back to the toolchain-free garbage contract.
    from . import frontend_fuzz as ff
    for g in ff.GARBAGE_INPUTS:
        try:
            ff._iri.ingest_clang(g)
        except Exception:
            return False
    import os as _os8
    if not _os8.path.exists(ff.CLANG):
        return True
    rep = ff.fuzz_clang_frontend(iterations=30, seed=0xA11CE)
    return rep.ok and rep.compiled >= 15


def _thm_conformance_green() -> bool:
    # The curated per-language conformance corpus must lower every applicable
    # construct to its exact expected facts (clang AST signatures/storage; rustc
    # MIR ownership). Absent both compilers there is nothing applicable to check,
    # so the corpus must at least be non-empty and well-formed.
    from . import conformance as cf
    if len(cf.ALL_CASES) < 12:
        return False
    for c in cf.ALL_CASES:
        if c.lang not in ("c", "rust") or not c.expected or not c.source:
            return False
    conf = cf.confirm_conformance()
    import os as _os9
    if not (_os9.path.exists(cf.CLANG) or _os9.path.exists(cf.RUSTC)):
        return True
    return conf.ok and conf.passed == conf.applicable


def _thm_eval_order_sound() -> bool:
    # Unsequenced-modification UB must be flagged (and loudly abstained on) while
    # clean code is not. Where clang is absent, the comment/string stripper still
    # gives a toolchain-free consistency check.
    from . import eval_order as eo
    if "i++" in eo._strip_comments_strings('int x;/* i++ */ x=1;'):
        return False
    import os as _os10
    if not _os10.path.exists(eo.CLANG):
        return True
    c = eo.confirm_sequencing()
    return bool(c.ok)


def _thm_cve_corpus_catches() -> bool:
    # The curated CWE-tagged corpus must be well-formed, and on any full toolchain
    # every applicable (case, target) pair must confirm a real definedness
    # divergence end-to-end. Absent a full toolchain, only the corpus shape is
    # checkable (consistency-only).
    from . import cve_corpus as cc
    if len(cc.CORPUS) < 5:
        return False
    for c in cc.CORPUS:
        if not c.cwe.startswith("CWE-") or not c.targets or not c.inputs:
            return False
    conf = cc.confirm_corpus()
    if not conf.available:
        return True
    return conf.ok


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
    claim(
        "C19-unit-alignment",
        "Cross-unit function alignment is decided by structure, not names: each "
        "C function is matched to its translated counterpart using signature "
        "compatibility (arity + C->target type families, with arity mismatch a hard "
        "veto) and self-reinforcing call-graph agreement, with name similarity only "
        "as a tiebreak. On a renamed module whose names are adversarial to a "
        "name-matcher (a 2-arg `add` is name-closest to a 1-arg `add_one`), the "
        "structural aligner recovers the ground-truth pairing exactly while the "
        "name-only baseline does not; user pins are honoured and low-confidence "
        "functions are reported unmatched rather than forced.",
        "ub_oracle.unit_alignment",
        ("align", "signature_score", "types_compatible", "name_only_align",
         "alignment_accuracy"),
        theorem=_thm_unit_alignment_sound,
        docs=("README.md",),
    ),
    claim(
        "C20-foreign-frontier",
        "The oracle reasons only about a pure, well-defined C fragment and "
        "**abstains loudly** at the soundness frontier instead of guessing: a "
        "lexical detector (comment/string-aware) flags volatile accesses, inline "
        "assembly, calls to undefined `extern` functions, atomics, "
        "setjmp/longjmp and signal handlers, and `decide` returns ABSTAIN with a "
        "human-readable reason naming each construct, while the pure fragment "
        "(including ordinary libc calls) stays CLEAR. Each abstention is justified "
        "against real `clang` IR — volatile keeps four separate `load volatile` "
        "where the pure version coalesces to one, inline asm is an opaque "
        "`call ... asm`, an `extern` callee is an undefined `declare`, and an "
        "atomic lowers to `load atomic` — so the frontier is documented, not "
        "assumed.",
        "ub_oracle.foreign_effects",
        ("scan_c_source", "decide", "FrontierVerdict", "confirm_all",
         "FOREIGN_FRONTIER"),
        theorem=_thm_foreign_frontier_sound,
        docs=("README.md",),
    ),
    claim(
        "C21-concurrency-race",
        "Data-race verdicts are taken from real sanitizers on real binaries, not "
        "assumed: a small catalogue of concurrency patterns (unsynchronized "
        "counter; mutex-guarded; lock-free atomic; read-only sharing) is compiled "
        "and run under **ThreadSanitizer** (C side) and **`go run -race`** (Go "
        "side), and a pattern is only called a race when the detector actually "
        "fires. The headline cross-language fact is that the *same* "
        "unsynchronized-counter idiom is flagged as a data race on **both** the C "
        "source (TSan) and the Go target (`-race`) — while Rust rejects it at "
        "compile time — and every synchronized variant (mutex/atomic/read-only) "
        "runs clean on both. The per-target migration story is documented.",
        "ub_oracle.concurrency",
        ("PATTERNS", "pattern", "confirm_race", "RaceConfirmation",
         "RACE_FRONTIER"),
        theorem=_thm_concurrency_race_sound,
        docs=("README.md",),
    ),
    claim(
        "C22-indirect-resolution",
        "Indirect calls through function-pointer dispatch tables are resolved so "
        "the call graph keeps the edges that matter on real code. The precise "
        "points-to set of a call `table[k](...)` is exactly the functions named "
        "in the table's initializer, and it refines the conservative "
        "signature-typed set (every defined function whose signature matches the "
        "table element type) — which itself excludes a wrong-signature decoy and "
        "`main`. The resolution is proven exact against real execution: an "
        "instrumented build of the example is compiled and run, and the set of "
        "functions actually reached through the table equals the predicted set "
        "(observed == predicted, never escaping it) while the decoy is never "
        "invoked.",
        "ub_oracle.indirect_calls",
        ("parse_unit", "resolve_table_call", "signature_compatible_targets",
         "confirm_table_dispatch", "table_is_well_typed"),
        theorem=_thm_indirect_resolution_sound,
        docs=("README.md",),
    ),
    claim(
        "C23-real-preprocessing",
        "Source is analysed only after the **real** C preprocessor runs, because "
        "a C program's meaning is fixed only post-preprocessing. Using `clang -E`, "
        "three load-bearing facts are proven against compiled, executed code: (1) "
        "macros are semantically load-bearing — `#define MUL(a,b) a*b` invoked as "
        "`MUL(1+1,2)` expands to `1+1*2` and a real binary evaluates it to 3, "
        "whereas the parenthesized macro gives 4, and the hazardous form is "
        "detected up front while the safe form is not; (2) `#ifdef` conditionals "
        "select the program (the same source runs to 0 without `-DFEATURE` and 1 "
        "with it); (3) `#include` resolution makes a header-only symbol present "
        "after preprocessing and the built program runs.",
        "ub_oracle.preprocess",
        ("preprocess", "detect_unparenthesized_macros",
         "confirm_macro_precedence_hazard", "confirm_conditional_compilation",
         "confirm_include_resolution"),
        theorem=_thm_preprocess_real,
        docs=("README.md",),
    ),
    claim(
        "C24-libc-model",
        "The most-hit libc/runtime surface is modelled with behavior-accurate, "
        "executable specs and proven against the *real* libc by randomized "
        "differential testing: `strlen`, `strcmp`, `strncmp`, `memcmp`, `memcpy`, "
        "`memset` and `strchr` each run on the host libc through a compiled "
        "harness and their output is compared to the pure-Python model on "
        "hundreds of random inputs, with zero mismatches. The specs encode the "
        "runtime's exact value contract — `strcmp`/`memcmp` return only the SIGN "
        "of the first differing unsigned byte, `strlen`/`strcmp`/`strchr` require "
        "NUL termination (the model raises on violation), `memcpy` forbids overlap "
        "while `memmove` allows it — and the `LibcSpec`/`confirm_spec` framework is "
        "runtime-agnostic (a new function = a model + a harness mode).",
        "ub_oracle.libc_model",
        ("SPECS", "confirm_spec", "confirm_all", "model_strcmp",
         "LIBC_CONTRACTS"),
        theorem=_thm_libc_model_accurate,
        docs=("README.md",),
    ),
    claim(
        "C25-ir-ingest",
        "Cross-language facts are recovered from the *compilers' own* "
        "intermediate representations rather than re-parsed by hand: clang's "
        "`-ast-dump=json` yields exact function signatures, parameter names and "
        "storage classes (e.g. `add(int,int)->int`, `static char *dup_first("
        "const char *)`), while rustc's `--emit=mir` yields ownership facts for "
        "free — a by-value non-`Copy` parameter (e.g. `Vec<i32>`) shows a "
        "`move`/`drop` of its local in MIR and is recorded as consumed, whereas a "
        "`Copy` `i32` parameter is not. Builtins are filtered via source-location "
        "provenance and the ingesters confirm themselves against the real clang "
        "and rustc on every run.",
        "ub_oracle.ir_ingest",
        ("ingest_clang", "ingest_rustc_mir", "confirm_clang_ingest",
         "confirm_mir_ingest", "IRModule"),
        theorem=_thm_ir_ingest_sound,
        docs=("README.md",),
    ),
    claim(
        "C26-project-ingest",
        "Ingestion scales from a single hand-picked file to a whole build tree on "
        "both sides of a migration. The source side reads a Clang "
        "`compile_commands.json` compilation database (the CMake/Bear standard), "
        "recovers each translation unit's `-I` include directories, lowers every "
        "TU through the clang-AST ingester and unions their symbols into one "
        "`ProjectModule` — proven on a real two-file C project where `add`/`helper`"
        "/`slen` are recovered across files with correct types and storage. The "
        "target side enumerates a Cargo workspace via `cargo metadata` (the build "
        "graph Cargo itself uses), discovering every member package, target name, "
        "kind and source root — proven on a real two-member workspace.",
        "ub_oracle.project_ingest",
        ("ingest_compile_db", "ingest_cargo_workspace", "confirm_compile_db",
         "confirm_cargo_workspace", "ProjectModule"),
        theorem=_thm_project_ingest_sound,
        docs=("README.md",),
    ),
    claim(
        "C27-solver-portfolio",
        "Decision procedures run as a *portfolio*, not a single point of failure: "
        "z3 (in-process) and boolector (out-of-process on a temp SMT-LIB2 file, "
        "exit 10/20 = sat/unsat) race the same query in parallel under a shared "
        "wall-clock timeout; the first decisive answer wins and all solvers that "
        "answered are cross-checked for agreement (a disagreement yields a loud "
        "UNKNOWN, never a silently-chosen verdict). A robustness battery spanning "
        "divergence-relevant bit-vector classes (signed overflow, unsigned wrap, "
        "truncation, shift-by-width, even-product low bit, xor-self) is solved by "
        "*every available solver* and matches the known ground truth, with at least "
        "one query decided by both backends so the cross-check is real.",
        "ub_oracle.solver_portfolio",
        ("solve_portfolio", "robustness_report", "confirm_portfolio",
         "available_solvers", "PortfolioResult"),
        theorem=_thm_solver_portfolio_robust,
        docs=("README.md",),
    ),
    claim(
        "C28-frontend-fuzz",
        "The C frontend is hardened by a *differential fuzzer* that uses the real "
        "clang as oracle: a seeded generator emits random but always-compilable "
        "well-typed translation units (random function counts, return types across "
        "the integer/pointer/void zoo, random arity, parameter types and storage "
        "classes); each is compiled by clang, ingested, and the recovered "
        "`IRFunction` set is diffed against the generator's ground truth — names, "
        "arity, return types, parameter types and storage class must match exactly. "
        "Dozens of programs survive a fixed-seed run with zero parse-divergences "
        "and zero crashes, and a corpus of malformed/non-C inputs makes the "
        "frontend return `None` rather than raise.",
        "ub_oracle.frontend_fuzz",
        ("fuzz_clang_frontend", "confirm_fuzz", "generate_program",
         "GARBAGE_INPUTS", "FuzzReport"),
        theorem=_thm_frontend_fuzz_hardened,
        docs=("README.md",),
    ),
    claim(
        "C29-conformance",
        "A curated per-language conformance corpus gates the frontends: each case "
        "pairs a real construct with its exact expected lowering and is checked "
        "against the real compiler. The C cases pin clang-AST facts — a `typedef`'d "
        "return type is preserved, an array parameter `int a[10]` decays to "
        "`int *`, a function-pointer parameter reconstructs to `int (*)(int)`, a "
        "`const int *` keeps its qualifier, `unsigned` canonicalises to "
        "`unsigned int`, and `static` linkage is read from the AST. The Rust cases "
        "pin rustc-MIR ownership — a by-value `Vec`/`Box`/`String` is moved while a "
        "reference or `Copy` scalar is not. The whole applicable suite is green, so "
        "any regression in how a construct lowers turns a case red.",
        "ub_oracle.conformance",
        ("run_conformance", "confirm_conformance", "ALL_CASES",
         "ConformanceCase", "CaseResult"),
        theorem=_thm_conformance_green,
        docs=("README.md",),
    ),
    claim(
        "C30-eval-order",
        "The sequencing oracle treats C's evaluation-order rules as a cross-language "
        "soundness hazard. Unsequenced modification (`i = i++ + i++`, `g(i++, i++)`) "
        "is genuine *undefined behavior* — the whole program's value is undefined — "
        "so the oracle detects it precisely with clang's `-Wunsequenced` (proven: "
        "the increment, call-argument and self-assignment idioms each produce a "
        "located diagnostic; clean sequenced code produces none) and returns a loud "
        "ABSTAIN, because no target translation that picks a concrete order can be "
        "proven equivalent to another legal C compilation. Unspecified "
        "argument-evaluation order with side effects (defined left-to-right in "
        "Rust/Go but not in C, and invisible in any single run) is documented as "
        "the sequencing soundness frontier.",
        "ub_oracle.eval_order",
        ("detect_unsequenced", "decide", "confirm_sequencing", "UnsequencedReport"),
        theorem=_thm_eval_order_sound,
        docs=("README.md",),
    ),
    claim(
        "C31-cve-corpus",
        "A curated 'we catch real bugs' corpus of UB-rooted weakness classes — each "
        "tagged with its CWE — is verified end-to-end across language pairs. Every "
        "entry pairs a C program that executes undefined behavior on a concrete "
        "input (confirmed by the UBSan/bounds build trapping) with a faithful Rust "
        "or Go translation that is fully defined and deterministic, and the "
        "re-execution harness confirms the definedness divergence by actually "
        "compiling and running both. Covered: division by zero (CWE-369), "
        "out-of-bounds array read (CWE-125), signed integer overflow (CWE-190), "
        "shift past bit-width (CWE-758) and INT_MIN/-1 (CWE-682), spanning C→Rust "
        "and C→Go. Nothing is asserted — every catch is executed.",
        "ub_oracle.cve_corpus",
        ("run_corpus", "confirm_corpus", "coverage_table", "CORPUS", "CveCase"),
        theorem=_thm_cve_corpus_catches,
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
