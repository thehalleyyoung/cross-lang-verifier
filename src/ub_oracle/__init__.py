"""
ub_oracle — the cross-language semantic-divergence oracle framework.

This package implements the generality backbone of the project (see
``100_STEPS.md`` steps 9-14, 22-24): a *pluggable* divergence-oracle framework
whose flagship/anchor instance is C->Rust translation where the divergence is
rooted in C undefined behavior (UB).

The pieces:

* ``catalogue``  — a typed cross-language divergence catalogue, with the C-UB ->
  Rust-defined mapping as its richest sub-table.
* ``plugin``     — the uniform ``DivergenceOracle`` interface + a registry that
  makes new divergence classes (and language pairs) additive rather than forks.
* ``replay``     — a language-agnostic, re-executable counterexample JSON format.
* ``reexec``     — an independent ground-truth harness that *actually* compiles
  and runs the real source and target on a counterexample to confirm a verdict.
* ``oracles``    — concrete divergence-class plugins (signed-overflow first).
* ``diff_testing`` — a differential-testing baseline used to demonstrate the
  divergences that fuzzing misses but the oracle catches.

Everything here is designed to be *proven against real compiled code*: the
re-execution harness shells out to a real C compiler (with UndefinedBehavior-
Sanitizer) and a real Rust compiler.
"""

from .catalogue import (
    DivergenceClass,
    Severity,
    Definedness,
    RustOutcomeKind,
    DivergenceEntry,
    CATALOGUE,
    c_ub_classes,
    entry_for,
)
from .plugin import (
    DivergenceOracle,
    OracleVerdict,
    OracleResult,
    REGISTRY,
    register,
    get_oracle,
    get_oracle_for,
    oracles_for,
    language_pairs,
    list_oracles,
)
from .replay import (
    Counterexample,
    ProofCertificate,
    REPLAY_SCHEMA_VERSION,
    PROOF_CERTIFICATE_SCHEMA_VERSION,
    verify_certificate,
)
from .reexec import (
    ReexecResult,
    ReexecHarness,
    toolchain_available,
    ToolchainStatus,
)
from .verify import (
    verify_unit,
    VerifyVerdict,
    VerifyReport,
    applicable_oracles,
)
from .report import aggregate_reports, to_sarif, pair_of
from .ablation import ablate_each_class
from .headtohead import head_to_head, differential_fuzz, FuzzUnit
from .abstract_interp import (
    Interval,
    PrePassVerdict,
    PrePassResult,
    analyze_unit,
    prunable_classes,
)
from .ir import (
    KNOWN_KINDS,
    IRError,
    IRValidationError,
    validate_unit,
    is_valid,
    assert_valid,
)
from .completeness import (
    FRAGMENTS,
    Fragment,
    ClassCompleteness,
    check_class_completeness,
    check_all_completeness,
    check_pair_completeness,
)
from .semantics import (
    EXPLOITED,
    TRAP_VS_DEFINED,
    LIBC_CONTRACT_TRAP_VS_DEFINED,
    MODES,
    Outcome,
    SourceObservation,
    TargetObservation,
    Observation,
    DivergenceJudgment,
    is_divergence,
    judge,
    observation_from_reexec,
    coincides_with_harness,
)
from .traceability import (
    Claim,
    TraceProblem,
    CLAIMS,
    verify_traceability,
    claim_ids,
)
from .cache import (
    CacheEquivalenceProof,
    prove_cache_equivalence,
    report_signature_hash,
    unit_content_hash,
)
from .parallel_harness import (
    ParallelDeterminismProof,
    ParallelRunReport,
    run_parallel,
    confirm_parallel_determinism,
)

__all__ = [
    "DivergenceClass",
    "Severity",
    "Definedness",
    "RustOutcomeKind",
    "DivergenceEntry",
    "CATALOGUE",
    "c_ub_classes",
    "entry_for",
    "DivergenceOracle",
    "OracleVerdict",
    "OracleResult",
    "REGISTRY",
    "register",
    "get_oracle",
    "get_oracle_for",
    "oracles_for",
    "language_pairs",
    "list_oracles",
    "Counterexample",
    "ProofCertificate",
    "REPLAY_SCHEMA_VERSION",
    "PROOF_CERTIFICATE_SCHEMA_VERSION",
    "verify_certificate",
    "ReexecResult",
    "ReexecHarness",
    "toolchain_available",
    "ToolchainStatus",
    "verify_unit",
    "VerifyVerdict",
    "VerifyReport",
    "applicable_oracles",
    "aggregate_reports",
    "to_sarif",
    "pair_of",
    "ablate_each_class",
    "head_to_head",
    "differential_fuzz",
    "FuzzUnit",
    "Interval",
    "PrePassVerdict",
    "PrePassResult",
    "analyze_unit",
    "prunable_classes",
    "KNOWN_KINDS",
    "IRError",
    "IRValidationError",
    "validate_unit",
    "is_valid",
    "assert_valid",
    "FRAGMENTS",
    "Fragment",
    "ClassCompleteness",
    "check_class_completeness",
    "check_all_completeness",
    "check_pair_completeness",
    "EXPLOITED",
    "TRAP_VS_DEFINED",
    "LIBC_CONTRACT_TRAP_VS_DEFINED",
    "MODES",
    "Outcome",
    "SourceObservation",
    "TargetObservation",
    "Observation",
    "DivergenceJudgment",
    "is_divergence",
    "judge",
    "observation_from_reexec",
    "coincides_with_harness",
    "Claim",
    "TraceProblem",
    "CLAIMS",
    "verify_traceability",
    "claim_ids",
    "CacheEquivalenceProof",
    "prove_cache_equivalence",
    "report_signature_hash",
    "unit_content_hash",
    "ParallelDeterminismProof",
    "ParallelRunReport",
    "run_parallel",
    "confirm_parallel_determinism",
]
