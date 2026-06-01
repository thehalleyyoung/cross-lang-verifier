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
    list_oracles,
)
from .replay import Counterexample, REPLAY_SCHEMA_VERSION
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
    "list_oracles",
    "Counterexample",
    "REPLAY_SCHEMA_VERSION",
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
]
