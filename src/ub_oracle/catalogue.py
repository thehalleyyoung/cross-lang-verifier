"""
Cross-language semantic-divergence catalogue (100_STEPS steps 12-13).

This is a typed taxonomy of *cross-language* divergence classes.  Each entry
records, for a divergence class:

* the source-language rule (here: C), including whether the source leaves the
  behavior *undefined* (UB), *implementation-defined*, or *unspecified*;
* the target-language outcome (here: Rust), e.g. wrap / panic / saturate /
  bounds-check / ownership-rejection / defined-value;
* a severity, a C-standard citation, and a short witness recipe.

The C-UB -> Rust-defined mapping is the richest sub-table and is the beating
heart of the flagship paper claim.  The structure is intentionally
language-pair-agnostic: ``source_rule``/``target_rule`` could later describe
C->Go, Python->Rust, etc., but C->Rust is the anchor we prove everything on.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple


class Severity(Enum):
    """How consequential a divergence is."""

    CRITICAL = "critical"   # definite observable behavioral difference is reachable
    MODERATE = "moderate"   # difference reachable only on some inputs / configs
    MINOR = "minor"         # difference is observable but rarely consequential

    def __str__(self) -> str:  # pragma: no cover - trivial
        return self.value


class Definedness(Enum):
    """The status of a behavior in the *source* language."""

    UNDEFINED = "undefined"               # C UB: anything may happen, optimizer may assume it cannot occur
    IMPLEMENTATION_DEFINED = "impl_defined"
    UNSPECIFIED = "unspecified"           # e.g. evaluation order
    DEFINED = "defined"

    def is_ub(self) -> bool:
        return self is Definedness.UNDEFINED


class RustOutcomeKind(Enum):
    """What the Rust translation does where C is undefined/looser."""

    WRAP = "wrap"                     # two's-complement wraparound (wrapping_* / release +)
    PANIC = "panic"                   # debug overflow panic / bounds-check panic
    SATURATE = "saturate"            # saturating_*
    BOUNDS_CHECKED = "bounds_checked"  # slice indexing -> panic on OOB
    OWNERSHIP_REJECTED = "ownership_rejected"  # borrow checker forbids at compile time
    DEFINED_VALUE = "defined_value"  # fully defined, deterministic value
    OPTION_RESULT = "option_result"  # checked_* -> Option/Result

    def __str__(self) -> str:  # pragma: no cover - trivial
        return self.value


@dataclass(frozen=True)
class DivergenceClass:
    """A stable identifier + human name for a divergence class."""

    key: str
    name: str

    def __str__(self) -> str:  # pragma: no cover - trivial
        return self.key


# ── The divergence classes (anchor: C->Rust) ─────────────────────────────────

SIGNED_OVERFLOW = DivergenceClass("signed_overflow", "Signed integer overflow")
SHIFT_OUT_OF_RANGE = DivergenceClass("shift_oob", "Shift amount >= bit width")
DIVISION_BY_ZERO = DivergenceClass("div_by_zero", "Integer division/remainder by zero")
INT_MIN_DIV_NEG1 = DivergenceClass("intmin_div_neg1", "INT_MIN / -1 overflow")
NULL_DEREF = DivergenceClass("null_deref", "Null pointer dereference")
ARRAY_OOB = DivergenceClass("array_oob", "Out-of-bounds array/pointer access")
UNINIT_READ = DivergenceClass("uninit_read", "Read of uninitialized storage")
STRICT_ALIASING = DivergenceClass("strict_aliasing", "Type-based (strict) aliasing violation")
USE_AFTER_FREE = DivergenceClass("use_after_free", "Use of freed / dangling storage")
EVAL_ORDER = DivergenceClass("eval_order", "Unspecified evaluation order / sequencing")
INT_CONVERSION = DivergenceClass("int_conversion", "Out-of-range integer conversion")
FP_CONTRACTION = DivergenceClass("fp_contraction", "Floating-point contraction (FMA fusion)")
VLA_BOUND = DivergenceClass("vla_bound", "Variable-length-array bound is non-positive")


@dataclass(frozen=True)
class DivergenceEntry:
    """One row of the catalogue."""

    cls: DivergenceClass
    # source-language (C) side
    source_definedness: Definedness
    source_rule: str
    c_standard_ref: str
    # target-language (Rust) side
    rust_outcome: RustOutcomeKind
    target_rule: str
    # meta
    severity: Severity
    witness_recipe: str
    # which integer width(s) the entry is exercised at, if applicable
    int_widths: Tuple[int, ...] = field(default_factory=tuple)

    def is_ub_rooted(self) -> bool:
        """True iff the divergence is rooted in *source* undefined behavior."""
        return self.source_definedness.is_ub()


# ── The catalogue itself ─────────────────────────────────────────────────────

_ENTRIES: List[DivergenceEntry] = [
    DivergenceEntry(
        cls=SIGNED_OVERFLOW,
        source_definedness=Definedness.UNDEFINED,
        source_rule="Signed arithmetic that overflows the result type is undefined; "
                    "optimizers may assume it never happens (e.g. `x+1 > x` -> `true`).",
        c_standard_ref="C17 6.5p5",
        rust_outcome=RustOutcomeKind.WRAP,
        target_rule="`wrapping_add`/release `+` wrap two's-complement; debug `+` panics. "
                    "Either way the result is *defined*.",
        severity=Severity.CRITICAL,
        witness_recipe="Pick operands whose exact-integer result leaves [INT_MIN, INT_MAX].",
        int_widths=(8, 16, 32, 64),
    ),
    DivergenceEntry(
        cls=SHIFT_OUT_OF_RANGE,
        source_definedness=Definedness.UNDEFINED,
        source_rule="Shifting by >= width (or negative) is undefined.",
        c_standard_ref="C17 6.5.7p3",
        rust_outcome=RustOutcomeKind.PANIC,
        target_rule="`<<`/`>>` panic in debug on overshift; `wrapping_shl` masks the amount.",
        severity=Severity.CRITICAL,
        witness_recipe="Pick shift amount s with s >= bit width or s < 0.",
        int_widths=(8, 16, 32, 64),
    ),
    DivergenceEntry(
        cls=DIVISION_BY_ZERO,
        source_definedness=Definedness.UNDEFINED,
        source_rule="Integer division or remainder by zero is undefined.",
        c_standard_ref="C17 6.5.5p5",
        rust_outcome=RustOutcomeKind.PANIC,
        target_rule="`/` and `%` panic on a zero divisor (defined, deterministic).",
        severity=Severity.CRITICAL,
        witness_recipe="Pick divisor == 0.",
        int_widths=(8, 16, 32, 64),
    ),
    DivergenceEntry(
        cls=INT_MIN_DIV_NEG1,
        source_definedness=Definedness.UNDEFINED,
        source_rule="INT_MIN / -1 (and INT_MIN % -1) overflow the result type; undefined.",
        c_standard_ref="C17 6.5.5p6",
        rust_outcome=RustOutcomeKind.PANIC,
        target_rule="`/` panics on overflow in debug; `wrapping_div` returns INT_MIN.",
        severity=Severity.CRITICAL,
        witness_recipe="numerator == INT_MIN and divisor == -1.",
        int_widths=(8, 16, 32, 64),
    ),
    DivergenceEntry(
        cls=ARRAY_OOB,
        source_definedness=Definedness.UNDEFINED,
        source_rule="Accessing an array/pointer outside its bounds is undefined.",
        c_standard_ref="C17 6.5.6p8",
        rust_outcome=RustOutcomeKind.BOUNDS_CHECKED,
        target_rule="Slice/array indexing is bounds-checked and panics on OOB.",
        severity=Severity.CRITICAL,
        witness_recipe="Pick an index i with i < 0 or i >= len.",
    ),
    DivergenceEntry(
        cls=NULL_DEREF,
        source_definedness=Definedness.UNDEFINED,
        source_rule="Dereferencing a null pointer is undefined.",
        c_standard_ref="C17 6.5.3.2",
        rust_outcome=RustOutcomeKind.OWNERSHIP_REJECTED,
        target_rule="`&T`/`Option<&T>` cannot be null; null only via `unsafe` raw ptr.",
        severity=Severity.CRITICAL,
        witness_recipe="Pass/produce a null pointer where a value is read.",
    ),
    DivergenceEntry(
        cls=UNINIT_READ,
        source_definedness=Definedness.UNDEFINED,
        source_rule="Reading an object with indeterminate value is undefined.",
        c_standard_ref="C17 6.3.2.1p2",
        rust_outcome=RustOutcomeKind.OWNERSHIP_REJECTED,
        target_rule="The type system forbids reading uninitialized bindings.",
        severity=Severity.MODERATE,
        witness_recipe="Read a local before any store on some path.",
    ),
    DivergenceEntry(
        cls=STRICT_ALIASING,
        source_definedness=Definedness.UNDEFINED,
        source_rule="Accessing an object through an incompatible lvalue type is undefined.",
        c_standard_ref="C17 6.5p7",
        rust_outcome=RustOutcomeKind.DEFINED_VALUE,
        target_rule="Reinterpretation requires explicit `transmute`/unions with defined rules.",
        severity=Severity.MODERATE,
        witness_recipe="Write as type T, read as incompatible type U.",
    ),
    DivergenceEntry(
        cls=USE_AFTER_FREE,
        source_definedness=Definedness.UNDEFINED,
        source_rule="Using storage after its lifetime ends is undefined.",
        c_standard_ref="C17 6.2.4",
        rust_outcome=RustOutcomeKind.OWNERSHIP_REJECTED,
        target_rule="The borrow checker rejects dangling uses at compile time.",
        severity=Severity.CRITICAL,
        witness_recipe="Free/drop, then read.",
    ),
    DivergenceEntry(
        cls=EVAL_ORDER,
        source_definedness=Definedness.UNSPECIFIED,
        source_rule="Order of evaluation of subexpressions is unspecified; multiple "
                    "unsequenced side effects on one object are undefined.",
        c_standard_ref="C17 6.5p2-3",
        rust_outcome=RustOutcomeKind.DEFINED_VALUE,
        target_rule="Rust fixes left-to-right evaluation order.",
        severity=Severity.MODERATE,
        witness_recipe="Use an expression whose result depends on evaluation order.",
    ),
    DivergenceEntry(
        cls=INT_CONVERSION,
        source_definedness=Definedness.IMPLEMENTATION_DEFINED,
        source_rule="Converting an out-of-range value to a signed type is implementation-defined.",
        c_standard_ref="C17 6.3.1.3p3",
        rust_outcome=RustOutcomeKind.WRAP,
        target_rule="`as` casts are defined to truncate/wrap (modular) deterministically.",
        severity=Severity.MODERATE,
        witness_recipe="Convert a value outside the destination signed range.",
        int_widths=(8, 16, 32, 64),
    ),
    DivergenceEntry(
        cls=FP_CONTRACTION,
        source_definedness=Definedness.UNSPECIFIED,
        source_rule="An implementation may contract `a*b + c` into a single "
                    "fused multiply-add (one rounding) at its discretion; whether "
                    "it does is unspecified, so the result is not uniquely fixed.",
        c_standard_ref="C17 6.5p8 / FP_CONTRACT pragma 7.12.2",
        rust_outcome=RustOutcomeKind.DEFINED_VALUE,
        target_rule="Rust never auto-contracts: `a*b + c` always rounds twice "
                    "(fusion only via explicit `f64::mul_add`), so it is a single "
                    "deterministic, defined value.",
        severity=Severity.MODERATE,
        witness_recipe="Pick a,b,c with heavy cancellation so the rounding of "
                       "`a*b` is observable: fma(a,b,c) != round(round(a*b)+c).",
    ),
    DivergenceEntry(
        cls=VLA_BOUND,
        source_definedness=Definedness.UNDEFINED,
        source_rule="If the size expression of a variable-length array is not a "
                    "positive value at evaluation, the behavior is undefined; the "
                    "implementation may allocate garbage, smash the stack, or "
                    "(under UBSan) trap.",
        c_standard_ref="C17 6.7.6.2p5",
        rust_outcome=RustOutcomeKind.PANIC,
        target_rule="The idiomatic safe translation sizes a heap vector from a "
                    "checked length conversion (`Vec`/`make`): a negative bound "
                    "is rejected with a deterministic, defined panic instead of "
                    "undefined behavior.",
        severity=Severity.CRITICAL,
        witness_recipe="Read the VLA bound n from input and pick any n < 0; the "
                       "UBSan build traps (`vla-bound`) while the target panics "
                       "deterministically on the checked length.",
    ),
]

CATALOGUE: Dict[str, DivergenceEntry] = {e.cls.key: e for e in _ENTRIES}


def entry_for(key: str) -> DivergenceEntry:
    """Look up a catalogue entry by its class key (raises KeyError if absent)."""
    return CATALOGUE[key]


def c_ub_classes() -> List[DivergenceEntry]:
    """All catalogue entries rooted in C undefined behavior (the anchor sub-table)."""
    return [e for e in _ENTRIES if e.is_ub_rooted()]


def all_entries() -> List[DivergenceEntry]:
    return list(_ENTRIES)
