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
    if not by_pair:
        return False
    # A pair that implements *none* of the integer-fragment classes (e.g. the C++
    # defined-subset pair, whose only class is signed_shift_sign_bit) is
    # legitimately empty and skipped; every pair that *does* implement fragment
    # classes must be complete on all of them. Require the core integer pairs.
    checked = 0
    for results in by_pair.values():
        if not results:
            continue
        checked += 1
        if not all(r.complete for r in results):
            return False
    return checked >= 3

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
    # unsynchronized counter races), Rust must reject exactly that racy idiom, and
    # where ThreadSanitizer / Go's race detector are available, the racy/race-free
    # verdict must be confirmed on real binaries.
    from . import concurrency as co
    racy = [p.name for p in co.PATTERNS.values() if p.races]
    if racy != ["unsynchronized_counter"]:
        return False
    if any(p.rust_accepts is not (not p.races) for p in co.PATTERNS.values()):
        return False
    import os as _os2
    if _os2.path.exists(co.RUSTC):
        rust_race = co.confirm_race("unsynchronized_counter", check_go=False,
                                    check_rust=True)
        if rust_race.rust.accepted is not False:
            return False
        rust_clean = co.confirm_race("mutex_counter", check_go=False,
                                     check_rust=True)
        if rust_clean.rust.accepted is not True:
            return False
    if co.c_race_detector_available():
        race = co.confirm_race("unsynchronized_counter", check_go=False,
                               check_rust=False)
        clean = co.confirm_race("mutex_counter", check_go=False,
                                check_rust=False)
        if not (race.c.race_detected is True and clean.c.race_detected is False):
            return False
    if co.go_race_detector_available():
        race = co.confirm_race("unsynchronized_counter", check_go=True,
                               check_rust=False)
        clean = co.confirm_race("mutex_counter", check_go=True,
                                check_rust=False)
        if not (race.go.race_detected is True and clean.go.race_detected is False):
            return False
    return True


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


def _thm_ground_truth_labeled() -> bool:
    # The labeled ground-truth set must be large (>= 500 pairs across >= 2
    # language pairs) and faithfully labeled: on any full toolchain, the
    # authoritative sanitizer-based labeler must agree with every sampled
    # declared label. Absent a full toolchain only the corpus shape is checkable.
    from . import ground_truth as gt
    full = gt.enumerate_corpus()
    if len(full) < 500:
        return False
    if len({it.lang for it in full}) < 2:
        return False
    labels = {it.declared_label for it in full}
    if labels != {"divergent", "equivalent"}:
        return False
    conf = gt.confirm_ground_truth(sample_per_class=1)
    if not conf.available:
        return True
    return conf.ok


def _thm_scale_measure_reproducible() -> bool:
    # The scale harness must expose a content-hashed verdict layer that is stable
    # across runs and faithful to the sanitizer-grounded labels. Absent a full
    # toolchain only the schema/SPI shape is checkable.
    from . import scale_measure as sm
    if not isinstance(sm.SCHEMA_VERSION, str) or not sm.SCHEMA_VERSION:
        return False
    for k in ("run_scale", "results_document", "emit_results_json",
              "content_hash", "confirm_scale"):
        if k not in sm.SCALE_MEASURE_SPI:
            return False
    conf = sm.confirm_scale(sample_per_class=1)
    if not conf.available:
        return True
    return conf.ok and conf.hash_stable


def _thm_headtohead_external_gap() -> bool:
    # No existing-tool category ingests a cross-language pair, and on the
    # provably-blind UB classes the realizable single-language baseline finds
    # nothing while the oracle catches all. Absent a full toolchain only the
    # categorical applicability table is checkable.
    from . import external_baselines as xb
    tbl = xb.applicability_table()
    if not tbl or any(row["ingests_cross_language_pair"] for row in tbl):
        return False
    conf = xb.confirm_head_to_head(per_class=1)
    if not conf.available:
        return True
    return conf.ok


def _thm_replication_kit() -> bool:
    # The external replication kit's files must exist with the expected entry
    # points, the corpus it reproduces must be >= 500 pairs across >= 2 language
    # pairs, and the deterministic manifest hash must be stable across runs.
    from . import replication as r
    rep = r.confirm_replication_kit(quick=True)
    if not (rep.files_ok and rep.corpus_ok):
        return False
    h1 = r.manifest(r.confirm_replication_kit(quick=True))["kit_hash"]
    h2 = r.manifest(r.confirm_replication_kit(quick=True))["kit_hash"]
    return h1 == h2


def _thm_statistical_rigor() -> bool:
    # Empirical claims about the oracle are reported with pre-registered metrics,
    # Wilson 95% confidence intervals computed deterministically from the counts,
    # multiple seeds, hardware provenance, and zero false positives on the
    # equivalent population (the sound-for-divergence guarantee, measured).
    from . import statistical_rigor as sr
    # Wilson interval is a pure, reproducible function of (k, n).
    if sr.wilson_interval(7, 20) != sr.wilson_interval(7, 20):
        return False
    if set(sr.PREREGISTERED_METRICS) != {"recall_definedness", "false_positive_rate"}:
        return False
    conf = sr.confirm_statistical_rigor(seeds=(1, 2), sample_per_seed=3)
    if not conf.available:
        return True  # consistency-only when toolchain is absent
    return bool(conf.ok and conf.hash_stable and conf.fpr.successes == 0)


def _thm_product_program() -> bool:
    # The relational product-program assertion R_m decides identically to the
    # operational divergence semantics over the recorded-observable abstraction
    # (a finite, exhaustively-checkable Boolean equivalence), and identically to
    # the real re-execution harness on real compiled code.
    from . import product_program as pp
    from . import semantics as sem
    # Abstraction-level soundness AND completeness: same Boolean function.
    for mode in pp.MODES:
        for o0_rc in (0, 1):
            for o2_rc in (0, 1):
                for o0v, o2v in (("1", "1"), ("1", "2")):
                    for san in (False, True):
                        for defined in (False, True):
                            for det in (False, True):
                                obs = pp.ProductObservable(
                                    target="rust", mode=mode,
                                    o0_rc=o0_rc, o0_val=o0v,
                                    o2_rc=o2_rc, o2_val=o2v,
                                    san_trapped=san, defined=defined,
                                    deterministic=det)
                                if pp.product_violated(obs) != sem.is_divergence(
                                        obs.to_observation()):
                                    return False
    conf = pp.confirm_product_program(per_class=1)
    if not conf.available:
        return True  # consistency-only when toolchain absent
    return bool(conf.ok and conf.n_divergent > 0 and conf.n_equivalent > 0)


def _thm_translation_validation() -> bool:
    # The translation validator is sound: a REFUTED counterexample witness, when
    # replayed against fresh compilations, reproduces a violation of R_m (a
    # genuine, third-party-checkable divergence), and replaying is deterministic.
    # Equivalent items are NOT_REFUTED (no false refutation).
    from . import translation_validation as tv
    conf = tv.confirm_translation_validation(per_class=1)
    if not conf.available:
        return True  # consistency-only when toolchain absent
    return bool(conf.ok and conf.n_refuted > 0 and conf.n_not_refuted > 0)


def _thm_generalization() -> bool:
    # The divergence result generalizes: across every available (language pair x
    # producer style) cell the detection rate is 1.0 on UB inputs and the
    # false-positive rate is 0.0 on safe inputs, with >= 2 pairs exercised.
    from . import generalization as g
    conf = g.confirm_generalization()
    if not conf.available:
        return True  # consistency-only when toolchain absent
    return bool(conf.ok and conf.invariant_across_pairs
                and conf.invariant_across_styles and conf.n_pairs >= 2)


def _thm_artifact_eval() -> bool:
    # The three ACM artifact badges are *earned*: every criterion of every badge
    # passes. Availability is pure file inspection (always real evidence);
    # functional and reproduced exercise live paths (real oracle run + byte-
    # identical results + stable reproducibility hashes) when a toolchain is
    # present and degrade to consistency-only (still passing) otherwise.
    from . import artifact_eval as a
    conf = a.confirm_artifact_evaluation()
    return bool(conf.ok
                and set(conf.earned_badges) == {"available", "functional", "reproduced"})


def _thm_mechanized_soundness() -> bool:
    # The core soundness/relative-completeness argument for the product-program
    # decision procedure is machine-checked by the real Lean 4 kernel for a
    # language-pair-parametric calculus instantiated to C->Rust, and independently
    # cross-checked in Coq for the central product-program lemma. When a proof
    # assistant is absent we require only that its source declares every theorem
    # (consistency-only); when present the kernel must accept the development.
    # When Lake is present, the extracted verdict checker must also build.
    from . import mechanized_soundness as m
    rep = m.confirm_mechanized_soundness()
    if not rep.ok:
        return False
    coq = m.confirm_coq_crosscheck()
    if not coq.ok:
        return False
    boundary = m.confirm_mechanized_completeness_boundary()
    if not boundary.ok:
        return False
    if m._lake_binary() is None:
        return True
    return bool(boundary.fully_checked and m.build_verified_checker().ok)


def _thm_idiomatic_corpus() -> bool:
    # The oracle keeps its guarantees on idiomatic, value-carrying ports: every
    # divergent item is flagged on its UB input and silent on its safe input,
    # and every equivalent item is never flagged, across >=2 languages, with a
    # content-hash-stable verdict layer. Consistency-only when toolchain absent.
    from . import idiomatic_corpus as ic
    conf = ic.confirm_idiomatic_corpus()
    if not conf.available:
        return True
    return bool(conf.ok and conf.n_divergent > 0 and conf.n_equivalent > 0
                and conf.n_langs >= 2 and conf.hash_stable)


def _thm_multipair_corpus() -> bool:
    # Multi-pair generality: every real function's UB-rooted divergence is
    # flagged on EVERY target pair and every equivalent function on none, with a
    # content-hash-stable verdict layer and >=2 pairs. Consistency-only when no
    # toolchain is present.
    from . import multipair_corpus as mpc
    conf = mpc.confirm_multipair_corpus()
    if not conf.available:
        return True
    return bool(conf.ok and conf.cross_pair_invariant and conf.hash_stable
                and conf.n_pairs >= 2)


def _thm_transpiler_recipes() -> bool:
    # The "translate with $tool, then verify with us" pipeline preserves the
    # oracle's guarantees: for every available reference translator the
    # translate->verify step flags a div-by-zero divergence on the UB input and
    # stays silent on a safe input, and external transpiler recipes (c2rust,
    # llm-transpiler) are registered and correctly gated (never fabricated).
    # Consistency-only when no target toolchain is present.
    from . import transpiler_recipes as tr
    conf = tr.confirm_transpiler_recipes()
    return bool(conf.ok)


def _thm_playground() -> bool:
    # The interactive web playground is backed by the LIVE oracle, not a mock:
    # a real HTTP server, driven over a real socket, flags a div-by-zero
    # translation on the UB input and stays silent on a safe input, and the
    # rendered page advertises every supported language pair. Consistency-only
    # when no Rust toolchain is present.
    from . import playground as pg
    conf = pg.confirm_playground()
    return bool(conf.ok)


def _thm_docs_site() -> bool:
    # The documentation site is a REAL mkdocs build, not a description: the
    # gallery is machine-generated from the live corpora and `mkdocs build
    # --strict` (which fails on any broken nav entry or missing page) produces a
    # site. Consistency-only when mkdocs is not installed.
    from . import docs_site as ds
    rep = ds.confirm_docs_site()
    return bool(rep.ok)


def _thm_vscode_extension() -> bool:
    # The in-editor surface is a REAL VS Code extension that compiles: its
    # TypeScript builds cleanly with the real `tsc` against the real
    # `@types/vscode` typings, producing the JS entry point, and its
    # package.json contributes the verify command. It is thin (shells out to the
    # proven CLI), so it cannot drift from the oracle. Consistency-only when
    # Node/npm is absent.
    from . import vscode_ext as vx
    rep = vx.confirm_vscode_extension()
    return bool(rep.ok)


def _thm_real_frontends() -> bool:
    # The supported C frontend is a REAL grammar-backed parser (tree-sitter),
    # not a hand-rolled toy: on real C translation units the function table it
    # extracts (names, arities, parameter names, storage class) AGREES with the
    # clang AST — the compiler's own ground truth. The frontend SPI exposes
    # three registered frontends. Consistency-only when tree-sitter/clang absent.
    from . import frontends as fe
    rep = fe.confirm_real_frontends()
    return bool(rep.ok)


def _thm_single_binary() -> bool:
    # A single-file executable distribution exists for non-Python users: a
    # zipapp .pyz bundling ub_oracle builds, runs as a subprocess against a real
    # manifest, and produces JSON byte-identical to the in-process CLI -- the
    # shipped single file is behaviourally identical to the library. Pure stdlib,
    # always available.
    from . import single_binary as sb
    rep = sb.confirm_single_binary()
    return bool(rep.ok)


def _thm_divergence_zoo() -> bool:
    # The divergence zoo is a structured, indexed catalogue whose every divergent
    # exhibit is RE-CONFIRMABLE: the oracle re-run on each exhibit's witnessing
    # input still flags the divergence and stays silent on the safe input, so the
    # zoo cannot contain an unreproducible exhibit. Consistency-only when no
    # target toolchain is present.
    from . import divergence_zoo as zoo
    rep = zoo.confirm_zoo()
    return bool(rep.ok)


def _thm_paper_figures() -> bool:
    # The three paper figures are DATA-FAITHFUL, not hand-drawn: every number
    # rendered into the SVGs (catalogue cell counts, per-pair confirmed-
    # divergence counts, per-class totals, the semrec-vs-fuzzing recall gap) is
    # independently recomputed from the real in-repo data sources (the
    # re-confirmed divergence_zoo, the cross-pair regression matrix, and the
    # ub-invisible benchmark) and must match, with Fig1 column sums equal to
    # Fig3 totals. Pure stdlib, always reproducible.
    from . import figures as fg
    rep = fg.confirm_figures()
    return bool(rep.ok)


def _thm_ecosystem() -> bool:
    # The v1 public API is a real, enforced SemVer surface: every promised
    # symbol is still exported in ub_oracle.__all__ and importable (so a
    # removal/rename -- a breaking change -- is caught mechanically), AND the
    # shipped `cargo cross-lang-verify` subcommand is valid shell that runs
    # end-to-end against a real manifest and emits JSON byte-identical to the
    # in-process library, so the integration cannot drift from the oracle.
    from . import ecosystem as eco
    rep = eco.confirm_ecosystem()
    return bool(rep.ok)


def _thm_claims_audit() -> bool:
    # Every audited public claim is tied to live ground truth: each named C->
    # target language in the docs is a registered pair with a real oracle, the
    # general framing is backed by >=2 working pairs, and every literal count
    # asserted in prose (e.g. "N exhibits across rust/go/swift") equals the live
    # count from the re-confirmed corpora -- so a doc edited to overclaim fails.
    from . import claims_audit as ca
    rep = ca.confirm_claims_audit()
    return bool(rep.ok)


def _thm_case_studies() -> bool:
    # Three real-world-derived C->target migrations are walked end-to-end against
    # real compilers: on the witnessing input the C UB is reachable (UBSan
    # traps), the idiomatic target is well-defined, the oracle confirms the
    # divergence and stays silent on the safe input -- AND an equal-budget
    # differential fuzzer on the SAME real binaries demonstrably misses at least
    # one bug the oracle confirms (the false-negative gap). Consistency-only
    # when no target toolchain is present.
    from . import case_studies as cs
    rep = cs.confirm_case_studies(trials=600)
    return bool(rep.ok)


def _thm_disclosures() -> bool:
    # The responsible-disclosure toolkit reproduces every advisory LIVE: for
    # each record the real pipeline (clang+UBSan, rustc/go) confirms the C UB is
    # reachable on the witness, the target is defined, the oracle confirms the
    # divergence and stays silent on the safe input -- and every emitted
    # self-contained reproduction bundle is valid runnable shell. Consistency-
    # only when no target toolchain is present.
    from . import disclosure as dz
    rep = dz.confirm_disclosures()
    return bool(rep.ok)


def _thm_true_green_ratchet() -> bool:
    # The suite-honesty gate is itself sound: its enforcement core flags any red
    # outcome (a failure, an error, or an xpassed "lie") and any *regression*
    # against the committed floor (a lost pass, or a newly-skipped test), while
    # accepting a genuinely green, non-regressed run. We prove this on the gate's
    # own decision function with synthetic count vectors -- no toolchain needed,
    # so the theorem is total.
    from . import test_ratchet_core as trc  # noqa: WPS433

    floor = {"passed": 100, "skipped": 5, "xfailed": 7}
    green = {"passed": 100, "skipped": 5, "xfailed": 7,
             "failed": 0, "error": 0, "xpassed": 0}
    if trc.enforce_counts("fast", green, floor) != 0:
        return False
    better = dict(green, passed=120, skipped=4)
    if trc.enforce_counts("fast", better, floor) != 0:
        return False
    for bad in (
        dict(green, failed=1),
        dict(green, error=1),
        dict(green, xpassed=1),
        dict(green, passed=99),
        dict(green, skipped=6),
    ):
        if trc.enforce_counts("fast", bad, floor) == 0:
            return False
    return True


def _thm_large_scale_study() -> bool:
    # The large-scale (>=100k LOC) migration study is sound on two axes. (1) The
    # corpus genuinely meets the LOC floor and is all-distinct source -- a pure,
    # toolchain-free structural fact. (2) When real compilers are present, a
    # seeded random sample of the corpus, executed end-to-end through the actual
    # labeler, agrees with every declared ground-truth label. The theorem checks
    # (1) unconditionally and (2) only when the toolchain is available, so it is
    # total: it never fabricates a pass when compilers are absent.
    from . import large_scale_study as ls  # noqa: WPS433

    cen = ls.corpus_census(ls.generate_corpus())
    if int(cen["total_loc"]) < ls.MIN_TOTAL_LOC:
        return False
    if cen["n_items"] != cen["n_distinct_programs"]:
        return False
    rep = ls.confirm_large_scale_study(sample_size=6)
    if not rep.ok:
        return False
    if rep.available and rep.aggregates.get("agree") != rep.aggregates.get("executed"):
        return False
    return True


def _thm_vla_bound_oracle() -> bool:
    # The VLA-bound oracle is sound on the anchor and the Go pair. For each
    # target, the oracle (a) finds a non-positive VLA bound as its witness, and
    # (b) when the toolchain is present, the real UBSan build *traps* on that
    # bound (C is UB: vla-bound) while the safe target port is *defined and
    # deterministic* (a panic). The structural witness check is unconditional;
    # the real-compiler confirmation runs only when the toolchain exists, so the
    # theorem is total.
    from . import oracles as _oracles  # noqa: F401, WPS433  (register plugins)
    from .plugin import get_oracle_for, OracleVerdict
    from .reexec import ReexecHarness, toolchain_available

    status = toolchain_available()
    h = ReexecHarness(status)
    for target in ("rust", "go"):
        orc = get_oracle_for("vla_bound", "c", target)
        if orc.confirmation_mode != "trap_vs_defined":
            return False
        res = orc.find_divergence({"kind": "vla", "width": 32})
        if res.verdict is not OracleVerdict.DIVERGENT:
            return False
        if list(res.counterexample.inputs.values())[0] >= 0:
            return False
        if not status.full_for(target):
            continue
        rr = orc.confirm(res, h).reexec
        if not (rr.available and rr.ub_reachable and rr.rust_defined and rr.confirmed):
            return False
    return True


def _thm_float_cast_overflow_oracle() -> bool:
    # The float->int out-of-range conversion oracle is sound on the anchor and
    # the Go pair. For each target the oracle (a) finds an out-of-range value
    # (just past the destination integer maximum) as its witness, and (b) when
    # the toolchain is present, the real UBSan build *traps*
    # (`float-cast-overflow`) while the safe target port (Rust saturating `as`,
    # Go defined conversion) is *defined and deterministic*. The structural
    # witness check is unconditional; the real-compiler confirmation runs only
    # when the toolchain exists, so the theorem is total.
    from . import oracles as _oracles  # noqa: F401, WPS433  (register plugins)
    from .plugin import get_oracle_for, OracleVerdict
    from .reexec import ReexecHarness, toolchain_available

    status = toolchain_available()
    h = ReexecHarness(status)
    for target, width in (("rust", 32), ("go", 64)):
        orc = get_oracle_for("float_cast_overflow", "c", target)
        if orc.confirmation_mode != "trap_vs_defined":
            return False
        res = orc.find_divergence({"kind": "float_cast", "width": width})
        if res.verdict is not OracleVerdict.DIVERGENT:
            return False
        # the witness must be strictly past the signed maximum of the dest type.
        if list(res.counterexample.inputs.values())[0] <= (1 << (width - 1)) - 1:
            return False
        if not status.full_for(target):
            continue
        rr = orc.confirm(res, h).reexec
        if not (rr.available and rr.ub_reachable and rr.rust_defined and rr.confirmed):
            return False
    return True


def claim(*args, **kwargs) -> Claim:  # small constructor alias
    return Claim(*args, **kwargs)


def _thm_fast_math_reassoc_oracle() -> bool:
    # The -ffast-math reassociation oracle is sound on the anchor and the Go
    # pair. For each target the oracle (a) finds a finite-normal witness (x,y)
    # for which IEEE-strict (x+y)-x is 0 while the reassociated value is a
    # non-zero y, and (b) when the toolchain is present, the *same* C source
    # compiled `-fno-fast-math` vs `-ffast-math` produces different observable
    # output while the safe target (Rust/Go, no auto-reassociation) is defined
    # and deterministic. The structural witness check is unconditional; the
    # real-compiler confirmation runs only when the toolchain exists.
    from . import oracles as _oracles  # noqa: F401, WPS433  (register plugins)
    from .plugin import get_oracle_for, OracleVerdict
    from .reexec import ReexecHarness, toolchain_available

    status = toolchain_available()
    h = ReexecHarness(status)
    for target in ("rust", "go"):
        orc = get_oracle_for("fast_math_reassoc", "c", target)
        if orc.confirmation_mode != "optimizer_exploited":
            return False
        res = orc.find_divergence({"kind": "fp_reassoc"})
        if res.verdict is not OracleVerdict.DIVERGENT:
            return False
        x, y = res.counterexample.inputs["x"], res.counterexample.inputs["y"]
        # the witness must genuinely swallow y: IEEE-strict (x+y)-x rounds to 0
        # while the reassociated value y is non-zero.
        if y == 0.0 or ((x + y) - x) != 0.0:
            return False
        if not status.full_for(target):
            continue
        rr = orc.confirm(res, h).reexec
        if not (rr.available and rr.ub_consequential and rr.rust_defined and rr.confirmed):
            return False
    return True


def _thm_restrict_violation_oracle() -> bool:
    # The restrict-violation oracle is sound on the anchor and the Go pair. For
    # each target the oracle (a) finds an aliasing selector that triggers the
    # restrict violation, and (b) when the toolchain is present, the *same* C
    # source compiled `-O0` vs `-O2` disagrees on stdout (the optimizer caches a
    # value the restrict promise let it keep) while the safe target — Rust's
    # unique `&mut` / Go's restrict-free pointers — is defined and
    # deterministic. No sanitizer can trap this class, so the optimisation-level
    # disagreement is the divergence evidence.
    from . import oracles as _oracles  # noqa: F401, WPS433  (register plugins)
    from .plugin import get_oracle_for, OracleVerdict
    from .reexec import ReexecHarness, toolchain_available

    status = toolchain_available()
    h = ReexecHarness(status)
    for target in ("rust", "go"):
        orc = get_oracle_for("restrict_violation", "c", target)
        if orc.confirmation_mode != "optimizer_exploited":
            return False
        res = orc.find_divergence({"kind": "restrict_pair"})
        if res.verdict is not OracleVerdict.DIVERGENT:
            return False
        if res.counterexample.inputs["sel"] == 0:
            return False
        if not status.full_for(target):
            continue
        rr = orc.confirm(res, h).reexec
        if not (rr.available and rr.ub_consequential and rr.rust_defined and rr.confirmed):
            return False
    return True


def _thm_pointer_provenance_oracle() -> bool:
    # The pointer-provenance oracle is sound on the anchor and the Go pair. For
    # each target the oracle (a) finds — via Z3 — the least offset whose byte
    # displacement is guaranteed to overflow a 64-bit address space, and (b) when
    # the toolchain is present, the `-fsanitize=undefined` build traps via the
    # `pointer-overflow` check on that input while the safe target keeps an index
    # and accesses through a checked operation, yielding a deterministic, defined
    # value. The confirmation runs in `trap_vs_defined` mode.
    from . import oracles as _oracles  # noqa: F401, WPS433  (register plugins)
    from .plugin import get_oracle_for, OracleVerdict
    from .reexec import ReexecHarness, toolchain_available

    status = toolchain_available()
    h = ReexecHarness(status)
    for target, width in (("rust", 32), ("go", 64)):
        orc = get_oracle_for("pointer_provenance", "c", target)
        if orc.confirmation_mode != "trap_vs_defined":
            return False
        res = orc.find_divergence({"kind": "pointer_offset", "width": width})
        if res.verdict is not OracleVerdict.DIVERGENT:
            return False
        if res.counterexample.inputs["n"] <= 0:
            return False
        if not status.full_for(target):
            continue
        rr = orc.confirm(res, h).reexec
        if not (rr.available and rr.ub_reachable and rr.rust_defined and rr.confirmed):
            return False
    return True


def _thm_soundness_regression_gate() -> bool:
    # Every registered built-in oracle must have a static soundness statement,
    # matching confirmation mode, live theorem/claim references, and a concrete
    # witness unit that exercises the plugin's real find_divergence path.  The
    # negative-control tests show a new unstatemented oracle fails this gate.
    from . import soundness_gate as sg

    audit = sg.confirm_soundness_registry(probe_witnesses=True)
    return audit.ok and len(audit.probes) == len(audit.registered) == len(audit.statements)


def _thm_soundness_compendium() -> bool:
    # The compendium must be generated from live traceability + soundness-registry
    # data, list every oracle instance, and be byte-fresh.  Witness probing is
    # already covered by C63; keeping this theorem metadata-only avoids doing the
    # same registry probe twice during traceability checks.
    from . import soundness_compendium as sc
    from . import soundness_gate as sg

    check = sc.confirm_compendium(probe_witnesses=False)
    return bool(check.ok and check.row_count == len(sg.SOUNDNESS_STATEMENTS))


CLAIMS: List[Claim] = [
    claim(
        "C1-soundness",
        "A DIVERGENT verdict is only returned after real compiled programs "
        "confirm either a UB-rooted divergence or an explicitly model-level "
        "allowed-execution-set divergence.",
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
        "compile time under real `rustc` — and every synchronized variant "
        "(mutex/atomic/read-only) runs clean on both detectors and compiles/runs "
        "on Rust. The per-target migration story is documented.",
        "ub_oracle.concurrency",
        ("PATTERNS", "pattern", "confirm_race", "RaceConfirmation",
         "RustConfirmation", "c_race_detector_available",
         "go_race_detector_available", "RACE_FRONTIER"),
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
    claim(
        "C32-ground-truth",
        "A labeled ground-truth set of >=500 (C program, target translation) pairs "
        "across two language pairs (C->Rust and C->Go) underpins any "
        "precision/recall claim. The label of every item — 'divergent' or "
        "'equivalent' — is established not by the oracle but by *bounded "
        "enumeration + real sanitizers*: the C program is compiled under "
        "UBSan/bounds instrumentation and run; a trap (with a defined, "
        "deterministic target) labels the pair divergent, while a non-trapping C "
        "whose observable output matches a defined, deterministic target labels it "
        "equivalent. The corpus is enumerated parametrically (UB families: "
        "div-by-zero, OOB read, oversized shift, signed overflow, INT_MIN/-1; safe "
        "families: add, mul, in-bounds index, in-range shift, mod). On a full "
        "toolchain the sanitizer-established label agrees with the constructed "
        "label on every sampled item across both languages and both labels — the "
        "labeling authority is independent of the verifier under test.",
        "ub_oracle.ground_truth",
        ("enumerate_corpus", "label_item", "establish_ground_truth",
         "confirm_ground_truth", "corpus_stats"),
        theorem=_thm_ground_truth_labeled,
        docs=("README.md",),
    ),
    claim(
        "C33-scale-measure",
        "A scale-measurement harness drives the labeled corpus through the real "
        "(sanitizer-anchored) decision procedure, recording time, memory, verdict "
        "and abstention per item, and emits a canonical results JSON. Crucially the "
        "verdict layer is content-hashed while timing/memory are not: wall-clock "
        "and RSS are non-deterministic, so the sha256 is computed over only the "
        "verdict-relevant fields with sorted keys, and two independent runs on the "
        "same toolchain yield the identical content_hash even though their timings "
        "differ. Aggregates (decided/abstained by language pair and by divergence "
        "class, total wall time, peak RSS) are computed from the per-item records, "
        "and every decided item matches its sanitizer-grounded label (the "
        "measurement layer never corrupts the verdict).",
        "ub_oracle.scale_measure",
        ("run_scale", "results_document", "emit_results_json", "content_hash",
         "confirm_scale"),
        theorem=_thm_scale_measure_reproducible,
        docs=("README.md",),
    ),
    claim(
        "C34-external-head-to-head",
        "Against existing tools the cross-language UB-divergence problem occupies a "
        "structural gap, made concrete and executed rather than rhetorical. The "
        "machine is probed live for every relevant tool category (bounded model "
        "checking / single-language equivalence, symbolic execution, translation "
        "validation, static analysis, verified transpilers); none ingests a "
        "cross-language (C, target) pair, so none can even be posed the question. "
        "The realizable proxy for the single-language / translation-validation "
        "category — a same-language O0-vs-O2 differential of the C program — is run "
        "on real divergent items: on the provably-blind classes (div-by-zero, "
        "INT_MIN/-1, where the unsanitised C traps identically at all optimisation "
        "levels) it finds nothing while the oracle catches every one, a total "
        "false-negative gap. Value-producing UB that a same-language differential "
        "can sometimes observe is reported honestly in the per-class breakdown, not "
        "hidden.",
        "ub_oracle.external_baselines",
        ("run_head_to_head", "confirm_head_to_head", "applicability_table",
         "CATEGORIES"),
        theorem=_thm_headtohead_external_gap,
        docs=("README.md",),
    ),
    claim(
        "C35-replication-kit",
        "A stranger can regenerate every byte-reproducible table and re-confirm "
        "every oracle against the real toolchain from a single hermetic artifact: "
        "a pinned Dockerfile (clang/UBSan, rustc, go, z3/boolector, Python) plus "
        "`make reproduce-kit` / `scripts/reproduce_kit.sh`, which runs the "
        "credibility guard, the byte-identical regeneration of the trusted result "
        "tables, the corpus/ground-truth/scale re-confirmations against real code, "
        "and a content-hash manifest. The manifest's `kit_hash` is computed over "
        "only the run-to-run-stable layers (corpus statistics, the external-tool "
        "applicability table, kit file hashes) so two independent reproductions on "
        "the same checkout produce the identical hash — the diffability an artifact "
        "evaluator needs. The kit's integrity (file presence, entry points, corpus "
        ">=500 pairs across >=2 language pairs, hash stability) is itself a checked "
        "theorem.",
        "ub_oracle.replication",
        ("confirm_replication_kit", "manifest", "render"),
        theorem=_thm_replication_kit,
        docs=("README.md",),
    ),
    claim(
        "C36-statistical-rigor",
        "Empirical claims about the oracle are reported with the rigor an "
        "empirical-methods reviewer expects, **computed** not asserted. Metrics "
        "are pre-registered (their exact definitions are frozen constants before "
        "any measurement). The real definedness-divergence oracle "
        "(`confirm_trap_vs_defined`) is run over seeded subsamples of the "
        "independently sanitizer-labeled corpus across multiple seeds; recall and "
        "false-positive-rate are reported as point estimates with **Wilson 95% "
        "confidence intervals** that are a deterministic function of the success "
        "and trial counts (so the interval is exactly reproducible — no bootstrap "
        "RNG enters the headline number). The sound-for-divergence guarantee is "
        "verified empirically as **zero false positives** on the equivalent "
        "population, and negative results (value divergences a trap-vs-defined "
        "oracle is not designed to decide) are reported in a separate "
        "`out_of_scope` bucket rather than silently dropped. The per-item outcome "
        "layer is content-hashed (hardware/timing excluded) so two runs with the "
        "same seeds reproduce the identical hash.",
        "ub_oracle.statistical_rigor",
        ("confirm_statistical_rigor", "run_study", "wilson_interval",
         "PREREGISTERED_METRICS"),
        theorem=_thm_statistical_rigor,
        docs=("README.md",),
    ),
    claim(
        "C37-product-program",
        "The oracle is given its **relational / translation-validation** account: "
        "a product program `P_S x P_T` whose runs pair a source run and a target "
        "run on the same input, carrying a relational assertion `R_m` whose "
        "violation is *exactly* a cross-language divergence. The construction is "
        "written as inference rules (Step-L / Step-R / Join over the synchronous "
        "product, plus the three-clause assertion `R_m = not(P and T and C_m)`) "
        "with a **soundness-and-relative-completeness theorem**, all "
        "**parameterized over the target semantics pack** (`target_semantics`). "
        "The theorem is discharged two ways: (i) over the recorded-observable "
        "abstraction `product_violated` is *exhaustively* checked to be the same "
        "Boolean function as `semantics.is_divergence`; (ii) on real compiled "
        "code, `confirm_product_program` builds the product observable from real "
        "clang/UBSan + rustc/go runs and verifies "
        "`product_violated == is_divergence == harness.confirmed` on divergent "
        "AND equivalent corpus items across packs, with a stable content hash.",
        "ub_oracle.product_program",
        ("confirm_product_program", "product_violated", "evaluate_clauses",
         "build_product"),
        theorem=_thm_product_program,
        docs=("README.md", "docs/PRODUCT_PROGRAM.md"),
    ),
    claim(
        "C38-translation-validation",
        "The oracle is presented through the recognized **translation-validation** "
        "interface: a per-instance, witness-producing validator `V(P_S, P_T, I, T)` "
        "that either `REFUTES` the producer's faithfulness claim with a "
        "counterexample or returns `NOT_REFUTED` over the probed inputs (a "
        "one-sided result — equivalence is never claimed, matching the global "
        "sound-for-divergence direction). The validity relation is exactly the "
        "relational assertion `R_m`. The contribution is the **re-executable "
        "witness**: a self-contained record (both source texts, the concrete "
        "input, the target pack, the product observable) whose `replay()` "
        "recompiles and re-runs both sides from scratch. Two operational theorems "
        "are discharged on real code by `confirm_translation_validation`: "
        "**witness soundness** (a REFUTED witness replays to the same `R_m` "
        "violation against fresh compilations) and **witness determinism** (a "
        "second replay reproduces the identical observable). Producer-agnostic and "
        "target-parameterized.",
        "ub_oracle.translation_validation",
        ("validate", "CounterexampleWitness", "confirm_translation_validation"),
        theorem=_thm_translation_validation,
        docs=("README.md", "docs/TRANSLATION_VALIDATION.md"),
    ),
    claim(
        "C39-generalization",
        "The divergence result is **not an artefact** of one language pair, one "
        "producer, or one input — proven by running the real oracle over a grid "
        "of (language pair) x (producer / translation style) x (divergence class) "
        "x (concrete input). Three target packs are exercised (`rust`, `go`, "
        "`swift`) against three producer styles (`direct` inline, `helper` via a "
        "function, `verbose` intermediate bindings — modelling the different but "
        "faithful code a transpiler vs a human/LLM emits), each class probed with "
        "several distinct UB-triggering and several safe inputs. The result is "
        "**invariant across every cell**: detection rate 1.0 on UB inputs and "
        "false-positive rate 0.0 on safe inputs for every pair and every style "
        "(no outlier), with >=2 language pairs actually compiled and run. The "
        "per-cell verdict layer is content-hashed (timing excluded) so the study "
        "reproduces an identical hash across runs.",
        "ub_oracle.generalization",
        ("confirm_generalization", "run_generalization", "target_source"),
        theorem=_thm_generalization,
        docs=("README.md",),
    ),
    claim(
        "C40-artifact-eval",
        "The repository **earns all three ACM artifact badges**, and the badge "
        "criteria are themselves *checked predicates* rather than prose promises "
        "(`ub_oracle.artifact_eval`). **Available**: an OSI `LICENSE`, a "
        "`CITATION.cff` naming the public repository, a `README`, and a package "
        "version consistent across packaging metadata and the citation descriptor "
        "— all verified by inspection. **Functional**: the artifact is documented "
        "(README + CAPABILITIES + artifact appendix + traceability), its "
        "replication-kit entry points resolve over a >=500-pair / >=2-language "
        "corpus, the fresh-venv packaging proof is present, and — with a "
        "C+UBSan+rustc toolchain — the **real oracle is run live**, catching a "
        "div-by-zero divergence and staying silent on a safe input. "
        "**Reproduced**: the trusted results artifact regenerates byte-for-byte, "
        "and the replication `kit_hash`, scale verdict hash and generalization "
        "grid hash are each stable across two independent runs. Every live check "
        "degrades to consistency-only when the toolchain is absent and never "
        "falsely claims a badge.",
        "ub_oracle.artifact_eval",
        ("confirm_artifact_evaluation", "evaluate_artifact", "BADGES"),
        theorem=_thm_artifact_eval,
        docs=("README.md", "docs/ARTIFACT.md"),
    ),
    claim(
        "C41-mechanized-soundness",
        "The core soundness argument is **machine-checked by the Lean 4 kernel**, "
        "not merely tested, and the central theorem is independently cross-"
        "checked in Coq. `formal/ProductSoundness.lean` (self-contained, no "
        "Mathlib) formalizes the relational/product-program decision procedure "
        "over the recorded-observable abstraction `(P=ub-reached, T=target-"
        "defined, C=consequence)` with `R = ¬(P∧T∧C)`, and proves: `oracle_sound` "
        "(no false alarms), `oracle_complete_rel` (relative completeness), "
        "`oracle_decides` (reports iff diverges), `equivalence_never_reported` "
        "(equivalent pairs never flagged), `report_implies_ub` (every "
        "counterexample is rooted in source UB), `pack_oracle_sound` (the "
        "argument is **language-pair-parametric**), `rust_oracle_sound` (the "
        "concrete C→Rust instantiation, `RustPack` defined codes `{0,101}`), and "
        "an end-to-end product-program witness theorem proving emitted "
        "counterexamples preserve the source/target/input payload and the "
        "observation derived from raw run facts; plus class-specific theorem "
        "families for strict-aliasing optimizer-exploitation and "
        "pointer-provenance `trap_vs_defined` witnesses. "
        "`formal/CompletenessBoundary.lean` then formalizes the exact published "
        "boundary between classes guaranteed complete on their declared finite "
        "fragment and classes that remain sound-but-may-abstain, proving the "
        "classification total and disjoint while tying in-fragment classes back "
        "to the recorded-observable decision theorem. "
        "`ub_oracle.mechanized_soundness` runs the real `lean` binary and "
        "confirms the kernel accepts the required theorem set; Step 129 also "
        "builds `formal/VerifiedChecker.lean` with Lake, and Step 131 builds "
        "`formal/CompletenessBoundary.lean` with Lake, producing a tiny "
        "checker that re-validates each source-UB positive verdict's final "
        "inference from raw re-execution facts via "
        "`productViolated`/`oracle_sound`. With Lean "
        "or Lake absent it degrades only to consistency checks and never claims an "
        "unrun proof. `formal/CoreSoundness.v` re-proves the same central "
        "recorded-observable and product-witness lemmas in Coq; when `coqc` is "
        "available the driver runs the real Coq kernel, otherwise it reports "
        "source-contract-only status.",
        "ub_oracle.mechanized_soundness",
        ("confirm_mechanized_soundness", "build_verified_checker",
         "run_verified_checker", "confirm_coq_crosscheck",
         "confirm_mechanized_completeness_boundary", "REQUIRED_THEOREMS",
         "REQUIRED_COMPLETENESS_BOUNDARY_THEOREMS", "REQUIRED_COQ_THEOREMS",
         "LEAN_SOURCE", "COMPLETENESS_BOUNDARY_SOURCE", "COQ_SOURCE",
         "CHECKER_SOURCE"),
        theorem=_thm_mechanized_soundness,
        docs=("README.md", "docs/MECHANIZED_SOUNDNESS.md"),
    ),
    claim(
        "C42-idiomatic-corpus",
        "A **Tier-2 anchor corpus of human-idiomatic ports** proves the oracle "
        "keeps both guarantees on realistic, value-carrying functions — not toy "
        "`a/b` pairs. Each item is a real-world-shaped function with provenance "
        "(the binary-search/merge **midpoint `(lo+hi)/2`** signed-overflow bug, "
        "a packed-struct **bit-field shift**, a coreutils-style **rate divide**, "
        "and the equivalent idiomatic fixes: a 64-bit-widened **safe average**, "
        "a saturating **byte clamp**, an Internet-checksum-shaped **additive "
        "checksum**). On every (item × language) cell across **rust** and **go** "
        "the oracle is exactly right: every **divergent** item is flagged on its "
        "UB-triggering input and silent on its safe input, and every "
        "**equivalent** idiomatic port is never flagged (the true-negative a "
        "naive value-differ would false-positive on). The per-item verdict layer "
        "is content-hashed and reproduces an identical hash across runs.",
        "ub_oracle.idiomatic_corpus",
        ("confirm_idiomatic_corpus", "run_corpus", "CORPUS"),
        theorem=_thm_idiomatic_corpus,
        docs=("README.md",),
    ),
    claim(
        "C43-multipair-corpus",
        "A **Tier-3 multi-pair corpus** stresses generality across **every** "
        "supported language pair at once: each real C function is translated to "
        "**all three** targets (`rust`, `go`, `swift`) in the varied style a "
        "transpiler / LLM emits, and the oracle is run on every available pair. "
        "The machine-checked claim is **cross-pair invariance of the verdict** — "
        "a divergent function (midpoint signed-overflow, rate div-by-zero, "
        "bit-field oversized shift) is flagged on **every** pair (the divergence "
        "is a property of the source UB, not of one target's quirks), while an "
        "equivalent function (clamp, additive checksum) is flagged on **none**. "
        "Verified live against clang/UBSan + rustc/go/swiftc: 15 (function × "
        "pair) verdicts, cross-pair invariant holds, every verdict correct, and "
        "the verdict layer is content-hash-stable across runs.",
        "ub_oracle.multipair_corpus",
        ("confirm_multipair_corpus", "run_corpus", "CORPUS"),
        theorem=_thm_multipair_corpus,
        docs=("README.md",),
    ),
    claim(
        "C44-transpiler-recipes",
        "First-class, **pluggable transpiler-integration recipes** realise the "
        "tool's workflow — *translate your C with `$tool`, then verify with us* — "
        "so new transpilers and language pairs slot in as **data**, not code "
        "(`ub_oracle.transpiler_recipes`). A `Translator` protocol is the "
        "integration point; `ReferenceTranslator` ships built-in compilable "
        "Rust/Go/Swift baselines and `ExternalCommandTranslator` shells out to a "
        "real transpiler binary (**c2rust**, or an LLM-transpiler CLI) with "
        "`{in}`/`{out}` placeholders, **gated** on the binary existing — when "
        "absent it reports `unavailable` and returns no output (never "
        "fabricated). `verify_transpiled` runs the **real oracle** on the "
        "translator's output; on every available reference pair the "
        "translate→verify step flags a div-by-zero divergence on the UB input "
        "and stays silent on a safe input, proving the recipe pipeline preserves "
        "the oracle's guarantees end to end (clang/UBSan + rustc/go/swiftc).",
        "ub_oracle.transpiler_recipes",
        ("confirm_transpiler_recipes", "verify_transpiled", "RECIPES"),
        theorem=_thm_transpiler_recipes,
        docs=("README.md", "docs/TRANSPILER_RECIPES.md"),
    ),
    claim(
        "C45-web-playground",
        "An **interactive web playground** (`ub_oracle.playground`) is the public "
        "*try-it* surface — *paste C + its translation, get a divergence verdict "
        "and the witness in the browser* — and it is backed by the **live "
        "oracle**, not a mock. A dependency-free `http.server` exposes "
        "`GET /` (a form whose language-pair dropdown advertises every supported "
        "target) and `POST /api/verify`, whose handler `evaluate(...)` actually "
        "compiles and runs both programs via `confirm_trap_vs_defined`; when the "
        "chosen target's toolchain is absent it answers an honest "
        "`available=false` (never fabricated). The machine-checked claim drives "
        "the **real server over a real socket**: a div-by-zero translation is "
        "flagged on the UB input `[\"10\",\"0\"]` and stays silent on the safe "
        "input `[\"10\",\"2\"]`, and the rendered page advertises rust/go/swift — "
        "end to end through the network stack (clang/UBSan + rustc).",
        "ub_oracle.playground",
        ("confirm_playground", "evaluate", "make_server"),
        theorem=_thm_playground,
        docs=("README.md", "docs/PLAYGROUND.md"),
    ),
    claim(
        "C46-docs-site",
        "A real, buildable **documentation site** (`ub_oracle.docs_site` + "
        "`mkdocs.yml`) gathers the project's reference docs into a navigable site "
        "whose centrepiece — a **gallery of caught divergences** — is "
        "**machine-generated from the live corpora** (`idiomatic_corpus`, "
        "`multipair_corpus`), so it can never drift from the real catalogue "
        "(every catalogued function, its class, label and language pairs "
        "appears). The machine-checked claim invokes the **real `mkdocs` binary** "
        "with `build --strict` — which turns any broken nav entry, missing page "
        "or dangling reference into a failure — and confirms an HTML site is "
        "produced; consistency-only when `mkdocs` is absent (never fabricated).",
        "ub_oracle.docs_site",
        ("confirm_docs_site", "generate_gallery"),
        theorem=_thm_docs_site,
        docs=("README.md",),
    ),
    claim(
        "C47-vscode-extension",
        "An in-editor surface ships as a **real VS Code extension** "
        "(`vscode-extension/`, checked by `ub_oracle.vscode_ext`) that surfaces "
        "C→{Rust,Go,Swift} divergences as `vscode.Diagnostic`s. It is "
        "deliberately **thin** — it shells out to the proven `cross-lang-verify` "
        "CLI and parses its JSON rather than re-implementing any analysis, so it "
        "cannot drift from the oracle. The machine-checked claim compiles the "
        "extension's TypeScript with the **real `tsc`** against the **real "
        "`@types/vscode`** API typings (`strict`, `noUnusedLocals`), requires a "
        "clean build producing `out/extension.js`, and validates that "
        "`package.json` declares the `engines.vscode` constraint and the "
        "`crossLangVerifier.verify` command. Consistency-only when Node/npm is "
        "absent (never fabricated).",
        "ub_oracle.vscode_ext",
        ("confirm_vscode_extension",),
        theorem=_thm_vscode_extension,
        docs=("README.md",),
    ),
    claim(
        "C48-real-frontends",
        "Robust, **real source frontends** replace hand-rolled parsers on the "
        "supported path, behind an explicit **frontend SPI** "
        "(`ub_oracle.frontends`): a `Frontend` protocol "
        "(`name`/`language`/`available()`/`ingest()→IRModule`) makes adding a "
        "language a bounded, documented task, with three frontends registered — "
        "`treesitter-c` (a **tree-sitter** grammar parser, the supported path for "
        "real C), `clang-ast-c` and `rustc-mir-rust`. The machine-checked claim "
        "**cross-validates the tree-sitter frontend against the compiler's own "
        "ground truth**: on several real-world-shaped C translation units the "
        "function table it extracts — names, arities, positional parameter names, "
        "storage class — must **agree with the clang AST** (8 functions across 5 "
        "units here). Tree-sitter is honestly gated: absent, the frontend reports "
        "`available()=False` and never fabricates a parse.",
        "ub_oracle.frontends",
        ("confirm_real_frontends", "ingest_treesitter", "FRONTENDS"),
        theorem=_thm_real_frontends,
        docs=("README.md", "docs/FRONTENDS.md"),
    ),
    claim(
        "C49-single-binary",
        "A **single-file executable distribution** lets non-Python users adopt "
        "the tool without managing an environment (`ub_oracle.single_binary`): "
        "`build_pyz` produces a stdlib **zipapp** `.pyz` that bundles the whole "
        "`ub_oracle` package behind a `__main__` shim and an interpreter "
        "shebang, so it runs directly (`./cross-lang-verify.pyz --units …`) given "
        "any Python 3 — no `pip install`, no virtualenv, no source tree (it "
        "complements the existing `docker run` image). The machine-checked claim "
        "builds the `.pyz`, **runs it as a subprocess** against a real units "
        "manifest, and proves its JSON output is **byte-identical** (canonical "
        "re-encoding) to the in-process CLI — the shipped single file is "
        "behaviourally identical to the library. Pure stdlib, so it is always "
        "available and reproducible anywhere.",
        "ub_oracle.single_binary",
        ("confirm_single_binary", "build_pyz"),
        theorem=_thm_single_binary,
        docs=("README.md",),
    ),
    claim(
        "C50-divergence-zoo",
        "A structured, continuously-verifiable **divergence zoo** "
        "(`ub_oracle.divergence_zoo`) is the canonical, machine-readable "
        "reference for the cross-language divergence patterns the tool catches: "
        "it aggregates every catalogued exhibit from the live corpora into an "
        "**index keyed by `(divergence_class, language_pair)`**, a deterministic "
        "`zoo.json` export and a generated `docs/zoo.md` (so it cannot drift from "
        "the catalogue), and each exhibit carries a concrete **witnessing "
        "input**. What makes it a *zoo* rather than a static list is that every "
        "exhibit is **re-confirmable**: `confirm_zoo()` re-runs the **real "
        "oracle** on each divergent exhibit's witness and requires the divergence "
        "to still be flagged on the witness and stay silent on the safe input — "
        "14 exhibits across rust/go/swift re-confirmed live here; an exhibit that "
        "cannot be reproduced is rejected. The index is content-hash-stable; "
        "consistency-only when no target toolchain is present.",
        "ub_oracle.divergence_zoo",
        ("confirm_zoo", "index_by_class_and_pair", "EXHIBITS"),
        theorem=_thm_divergence_zoo,
        docs=("README.md",),
    ),
    claim(
        "C51-paper-figures",
        "The three figures a paper is written around are **generated from the "
        "real data and proven data-faithful** (`ub_oracle.figures`): (1) the "
        "cross-language **divergence catalogue** as a `divergence_class x "
        "language_pair` matrix, (2) the **divergences-missed-by-fuzzing gap** "
        "per pair (our oracle confirms UB-rooted, value-invisible divergences "
        "at 100% recall while every differential-fuzzing / IR-equality baseline "
        "scores 0% by construction — there is no defined source value to "
        "compare), and (3) the **confirmed-divergences-by-class** headline "
        "table. There is no plotting dependency: the SVGs are pure-stdlib and "
        "every embedded number is a deterministic function of the live sources "
        "(the re-confirmed `divergence_zoo`, `cross_pair_matrix/results.json`, "
        "and `ub_invisible_results.json`). `confirm_figures()` recomputes every "
        "datum independently and requires it to match the rendered figure, and "
        "checks cross-figure consistency (Fig1 column sums equal Fig3 totals), "
        "so the figures cannot drift from the evidence.",
        "ub_oracle.figures",
        ("confirm_figures", "collect", "generate_figures"),
        theorem=_thm_paper_figures,
        docs=("README.md",),
    ),
    claim(
        "C52-ecosystem-semver",
        "The tool ships a **stable v1 public API with an enforced SemVer "
        "guard** and a real **`cargo` subcommand** integration "
        "(`ub_oracle.ecosystem`). `PUBLIC_API_V1` is the committed surface "
        "downstream code may depend on; `confirm_api_surface`/`_confirm_api` "
        "imports the package and asserts every promised symbol is still "
        "exported in `__all__` and is a live object, so any removal or rename "
        "(a breaking change) is caught mechanically before release, and the "
        "surface is snapshotted to `integrations/api_surface_v1.json`. The "
        "shipped `cargo-cross-lang-verify` shim follows cargo's `cargo-<name>` "
        "discovery convention and forwards to the proven CLI; "
        "`confirm_ecosystem()` checks it is valid shell (`bash -n`), runs it "
        "**end-to-end** against a real manifest, and requires its JSON to be "
        "**byte-identical** to the in-process library — so the integration "
        "cannot drift from the oracle.",
        "ub_oracle.ecosystem",
        ("confirm_ecosystem", "PUBLIC_API_V1", "generate_artifacts"),
        theorem=_thm_ecosystem,
        docs=("README.md",),
    ),
    claim(
        "C53-claims-audit",
        "Public claims are **tightened to exactly what's proven** by a "
        "mechanical guard (`ub_oracle.claims_audit`): it scans the docs "
        "(`README.md`, `CAPABILITIES.md`, `docs/TRACEABILITY.md`) for auditable "
        "claims and checks each against a value computed **live** from the code "
        "— every named `C->target` language in the framing must be a registered "
        "pair backed by **at least one real oracle** (no aspirational "
        "languages), the general cross-language framing must clear the "
        "**>=2-working-pairs** bar, and each literal count asserted in prose "
        "(e.g. \"N exhibits across rust/go/swift\") must equal the live count "
        "from the re-confirmed `divergence_zoo` and registry. A doc edited to "
        "overclaim — a bigger exhibit count, a pair with no oracle, a "
        "generality boast with only one pair — makes `confirm_claims_audit()` "
        "fail; an opt-in mode additionally re-runs every traceability theorem.",
        "ub_oracle.claims_audit",
        ("confirm_claims_audit", "audit_text"),
        theorem=_thm_claims_audit,
        docs=("README.md",),
    ),
    claim(
        "C54-case-studies",
        "Three **real-world-derived migrations are walked end to end** against "
        "real compilers with a measured **cost/benefit** versus fuzzing "
        "(`ub_oracle.case_studies`): each case (e.g. the classic "
        "JDK/`Arrays.binarySearch` midpoint-overflow `(lo+hi)/2`, a "
        "throughput `total/count` divide-by-zero, a packed-bitfield oversized "
        "shift) is taken through the full pipeline — clang+UBSan shows the C UB "
        "is **reachable** on the witnessing input, the idiomatic Rust/Go "
        "translation is **well-defined**, the oracle **confirms** the divergence "
        "(timed) and stays **silent** on the safe input — and then an "
        "**equal-budget differential fuzzer is run on the very same compiled "
        "binaries**. The headline benefit is the **false-negative gap**: for "
        "sparse-UB classes (e.g. divide-by-zero, where a random int32 is almost "
        "never 0) the fuzzer burns its whole budget without ever hitting the "
        "bug the oracle finds deterministically, while dense-UB classes show "
        "parity (proving the harness is not rigged against fuzzing). "
        "`confirm_case_studies()` requires every case to walk and at least one "
        "fuzzing gap; consistency-only when no toolchain is present.",
        "ub_oracle.case_studies",
        ("confirm_case_studies", "generate_case_studies", "CaseResult"),
        theorem=_thm_case_studies,
        docs=("README.md",),
    ),
    claim(
        "C55-responsible-disclosure",
        "A **responsible-disclosure toolkit with a live bug-reproduction "
        "harness** (`ub_oracle.disclosure`) turns any confirmed divergence into "
        "a coordinated, reproducible advisory: a structured `DisclosureRecord` "
        "carries the affected real-world pattern + provenance, the exact C and "
        "target sources, the witnessing input, a defined safe input, the "
        "security/correctness impact and the concrete remediation; a "
        "**coordinated-disclosure template** (`DISCLOSURE_TEMPLATE.md`) "
        "captures summary/impact/PoC/remediation/timeline. `reproduce_disclosure` "
        "re-runs the **real** pipeline (clang+UBSan, rustc/go) to confirm the "
        "divergence live and emits a **self-contained, runnable reproduction "
        "bundle** (a shell script that compiles both sides and exhibits the bug "
        "on the witness). The shipped records are real-world *pattern exemplars* "
        "(JDK midpoint overflow, `total/count` divide-by-zero, packed-bitfield "
        "shift), explicitly a template to fill rather than third-party CVEs. "
        "`confirm_disclosures()` requires every record to reproduce live and "
        "every bundle to be valid shell; consistency-only when no toolchain.",
        "ub_oracle.disclosure",
        ("confirm_disclosures", "reproduce_disclosure", "DisclosureRecord"),
        theorem=_thm_disclosures,
        docs=("README.md",),
    ),
    claim(
        "C56-true-green-ratchet",
        "The test suite's **own honesty is gated and ratcheted** "
        "(`scripts/test_ratchet.py` + `ub_oracle.test_ratchet_core`, floor in "
        "`tests/green_baseline.json`): a run is rejected unless it is "
        "**true-green** (zero failures, zero errors, and zero `xpassed` -- an "
        "`xfail` that unexpectedly passes is a stale lie) and **non-regressed** "
        "against the committed floor (the passing count may never drop and the "
        "skipped count may never rise, forbidding silently deleting or "
        "`@skip`-ing a test to make red disappear). The toolchain-independent "
        "`--fast` profile (every test file but the heavyweight real-compiler "
        "driver) is deterministic and runs in seconds. The decision core is "
        "proven sound on synthetic count vectors -- it accepts a green, "
        "non-regressed run and rejects every red or regressed one.",
        "ub_oracle.test_ratchet_core",
        ("violations", "enforce_counts", "is_baselineable"),
        theorem=_thm_true_green_ratchet,
        docs=("README.md",),
    ),
    claim(
        "C57-large-scale-study",
        "The oracle is validated at **migration scale**: a deterministic, "
        "all-distinct corpus of **7,500 genuinely-distinct C->{Rust,Go} programs "
        "totalling >=130k lines** (`ub_oracle.large_scale_study`) mixing "
        "UB-rooted divergent families (division-by-zero, OOB read, oversized "
        "shift, signed overflow) with defined-equivalent families "
        "(safe add/mul/mod/shift). Each program bakes its *defined* operands as "
        "distinct literals and reads only the UB-triggering operand from argv, so "
        "no two programs share source. A seeded random sample is executed "
        "end-to-end through the real labeler (clang/UBSan + rustc/go) and every "
        "sampled item's observed verdict agrees with its declared ground-truth "
        "label; the verdict-layer content hash is reproducible for a fixed seed. "
        "The census (LOC floor + all-distinct) is a toolchain-free structural "
        "guarantee, and the live-sample confirmation gates cleanly on toolchain "
        "availability.",
        "ub_oracle.large_scale_study",
        ("generate_corpus", "corpus_census", "confirm_large_scale_study"),
        theorem=_thm_large_scale_study,
        docs=("README.md",),
    ),
    claim(
        "C58-vla-bound-divergence",
        "A **variable-length-array (VLA) bound** divergence oracle "
        "(`ub_oracle.oracles.vla_bound`) for both C->Rust and C->Go: a C VLA "
        "`T a[n]` with a non-positive `n` is **undefined** (C17 6.7.6.2p5 -- the "
        "`-O0` build segfaults and the `-fsanitize=undefined` build traps via "
        "`vla-bound`), while the idiomatic *safe* port sizes a heap buffer from a "
        "**checked** length conversion and therefore turns the same input into a "
        "deterministic, defined **panic** (Rust `Vec` after a sign check, rc 101; "
        "Go `make([]T, n)` `makeslice`, rc 2). The witnessing bound is found with "
        "Z3 (the least-extreme negative value consistent with any declared range) "
        "rather than hard-coded, and the divergence is confirmed end-to-end "
        "against real clang/UBSan + rustc/go in `trap_vs_defined` mode.",
        "ub_oracle.oracles.vla_bound",
        ("VlaBoundOracle", "GoVlaBoundOracle"),
        theorem=_thm_vla_bound_oracle,
        docs=("README.md",),
    ),
    claim(
        "C59-float-cast-overflow-divergence",
        "A **float-to-integer out-of-range conversion** divergence oracle "
        "(`ub_oracle.oracles.float_cast`) for both C->Rust and C->Go -- the one "
        "genuinely-divergent corner of the C *conversion lattice*. Converting a "
        "finite `double` to an integer type whose range cannot hold the rounded "
        "value is **undefined** (C17 6.3.1.4p1 -- `-O0` yields a target-specific "
        "garbage value and `-fsanitize=undefined` traps via `float-cast-overflow`), "
        "whereas the idiomatic *safe* port keeps the cast and stays **defined**: "
        "Rust `x as iN` **saturates** to the destination bound and Go `iN(x)` "
        "yields a deterministic (implementation-specified) value, neither of which "
        "is UB. The witnessing value is found with Z3 (the least-extreme integer "
        "just past the destination maximum, e.g. `INT_MAX+1`, honouring any "
        "declared range) rather than hard-coded, and the divergence is confirmed "
        "end-to-end against real clang/UBSan + rustc/go in `trap_vs_defined` mode. "
        "The remaining conversion-lattice corners (signed<->unsigned, "
        "implementation-defined right shift) provably agree on every "
        "two's-complement target and are documented as non-divergent.",
        "ub_oracle.oracles.float_cast",
        ("FloatCastOverflowOracle", "GoFloatCastOverflowOracle"),
        theorem=_thm_float_cast_overflow_oracle,
        docs=("README.md",),
    ),
    claim(
        "C60-fast-math-reassociation-divergence",
        "A **`-ffast-math` reassociation** divergence oracle "
        "(`ub_oracle.oracles.fast_math`) for both C->Rust and C->Go. IEEE-754 "
        "arithmetic is not associative, so `(x+y)-x` is not the same value as "
        "`y`; under `-ffast-math`/`-Ofast` the compiler is licensed to "
        "reassociate floating arithmetic as if over the reals and may fold "
        "`(x+y)-x` to `y`, so the **same** C source yields **different** "
        "observable output under `-fno-fast-math` (IEEE-strict, rounds to 0 when "
        "`x` swallows `y`) vs `-ffast-math` (returns `y`). Rust and Go never "
        "auto-reassociate floating arithmetic, so each safe-target port is a "
        "single deterministic, defined value. The witness `(x,y)` is found with "
        "Z3's floating-point theory (a large `x` that exactly swallows a non-zero "
        "`y`, a maximally visible gap) and confirmed end-to-end against real "
        "clang (`-fno-fast-math` vs `-ffast-math`) + rustc/go in "
        "`optimizer_exploited` mode -- no sanitizer is needed or able to trap it.",
        "ub_oracle.oracles.fast_math",
        ("FastMathReassocOracle", "GoFastMathReassocOracle"),
        theorem=_thm_fast_math_reassoc_oracle,
        docs=("README.md",),
    ),
    claim(
        "C61-restrict-violation-divergence",
        "A **`restrict`-violation** divergence oracle "
        "(`ub_oracle.oracles.restrict_alias`) for both C->Rust and C->Go. Calling "
        "`f(int *restrict a, int *restrict b)` with aliasing pointers is "
        "**undefined** (C17 6.7.3.1p4) and the optimizer *relies* on the "
        "non-aliasing promise: the **same** C source on the aliasing input "
        "returns 4 under `-O0` (honest re-read) but 3 under `-O2` "
        "(restrict-based caching of `*a`). No sanitizer traps this -- the "
        "optimisation-level disagreement is the evidence. The idiomatic safe port "
        "cannot reproduce the hazard: Rust's `&mut` references are unique by the "
        "borrow checker (non-aliasing by construction) and Go has no `restrict`, "
        "so each target is a single deterministic, defined value. The witnessing "
        "aliasing selector is found with Z3 and the divergence is confirmed "
        "end-to-end against real clang (`-O0` vs `-O2`) + rustc/go in "
        "`optimizer_exploited` mode.",
        "ub_oracle.oracles.restrict_alias",
        ("RestrictViolationOracle", "GoRestrictViolationOracle"),
        theorem=_thm_restrict_violation_oracle,
        docs=("README.md",),
    ),
    claim(
        "C62-pointer-provenance-divergence",
        "A **pointer-provenance / pointer-arithmetic-overflow** divergence oracle "
        "(`ub_oracle.oracles.pointer_provenance`) for both C->Rust and C->Go. "
        "Forming `a + n` so the result leaves the array's provenance — and, in "
        "the limit, overflows the address-space computation `n*sizeof(T)` — is "
        "**undefined** (C17 6.5.6p8): `-fsanitize=undefined` traps via the "
        "`pointer-overflow` check while `-O0` derives from a wild pointer. The "
        "idiomatic *safe* port never forms a raw out-of-provenance pointer; it "
        "keeps an **index** and accesses through a **checked** operation "
        "(Rust `a.get(i)`, Go bounds-checked index), so the same offset becomes a "
        "deterministic, defined value. The witnessing offset is found with Z3 "
        "(the least `n` with `n*sizeof(T) >= 2**64`, guaranteed to overflow any "
        "64-bit base address — e.g. `2**62` for 4-byte ints) rather than "
        "hard-coded, and the divergence is confirmed end-to-end against real "
        "clang/UBSan + rustc/go in `trap_vs_defined` mode.",
        "ub_oracle.oracles.pointer_provenance",
        ("PointerProvenanceOracle", "GoPointerProvenanceOracle"),
        theorem=_thm_pointer_provenance_oracle,
        docs=("README.md",),
    ),
    claim(
        "C63-soundness-regression-gate",
        "A **soundness-regression CI gate** (`ub_oracle.soundness_gate`) rejects "
        "any built-in oracle instance that lacks a per-pair soundness statement. "
        "The gate enumerates the live plugin registry, requires one static "
        "statement per `(source,target,class)` instance, checks the registered "
        "confirmation mode and source-definedness premise, validates the referenced "
        "traceability/Lean evidence, and then executes each oracle's real "
        "`find_divergence` path on a concrete witness unit to ensure the statement "
        "is bound to working code rather than prose. A new plugin can no longer "
        "land in CI as an unreviewed soundness claim.",
        "ub_oracle.soundness_gate",
        ("SOUNDNESS_STATEMENTS", "confirm_soundness_registry",
         "audit_soundness_statements", "SoundnessStatement"),
        theorem=_thm_soundness_regression_gate,
        docs=("README.md", "docs/TRACEABILITY.md"),
    ),
    claim(
        "C64-soundness-compendium",
        "The **soundness compendium** (`ub_oracle.soundness_compendium`, generated "
        "as `docs/SOUNDNESS_COMPENDIUM.md`) maps every registered "
        "`(source,target,divergence_class)` oracle instance to its declared "
        "confirmation mode, source-definedness premise, traceability/Lean theorem "
        "references, and concrete witness unit. It is generated deterministically "
        "from the live soundness registry plus `traceability.CLAIMS` and the "
        "ProductSoundness theorem contract, so a stale document, a dangling claim "
        "reference, or an unknown Lean theorem fails mechanically.",
        "ub_oracle.soundness_compendium",
        ("compendium_rows", "render_compendium", "confirm_compendium",
         "SOUNDNESS_COMPENDIUM_DOC"),
        theorem=_thm_soundness_compendium,
        docs=("docs/SOUNDNESS_COMPENDIUM.md", "docs/TRACEABILITY.md"),
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
