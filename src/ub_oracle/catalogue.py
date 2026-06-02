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
FLOAT_CAST_OVERFLOW = DivergenceClass("float_cast_overflow", "Float-to-integer conversion out of range")
FAST_MATH_REASSOC = DivergenceClass("fast_math_reassoc", "Floating-point reassociation under -ffast-math")
RESTRICT_VIOLATION = DivergenceClass("restrict_violation", "Aliasing through restrict-qualified pointers")
POINTER_PROVENANCE = DivergenceClass("pointer_provenance", "Pointer arithmetic out of object provenance / address overflow")
SIGNED_SHIFT_SIGN_BIT = DivergenceClass("signed_shift_sign_bit", "Left-shift of a 1 into the sign bit (UB in C, defined in C++20)")
BITFIELD_LAYOUT = DivergenceClass("bitfield_layout", "Implementation-defined bit-field layout / packing")
ENUM_OUT_OF_RANGE = DivergenceClass("enum_out_of_range", "Out-of-range value stored in / read from an enum")
MEMCPY_OVERLAP = DivergenceClass("memcpy_overlap", "Overlapping memcpy ranges")


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
    DivergenceEntry(
        cls=FLOAT_CAST_OVERFLOW,
        source_definedness=Definedness.UNDEFINED,
        source_rule="When a finite floating value is converted to an integer type "
                    "and the rounded result cannot be represented, the behavior "
                    "is undefined; under UBSan the `float-cast-overflow` check "
                    "traps, while `-O0` yields a target-specific garbage value.",
        c_standard_ref="C17 6.3.1.4p1",
        rust_outcome=RustOutcomeKind.DEFINED_VALUE,
        target_rule="The idiomatic safe translation keeps the `as`/conversion "
                    "cast, which is defined: Rust `x as iN` saturates to the "
                    "destination bound and Go `iN(x)` yields a deterministic "
                    "(implementation-specified) value — never undefined behavior.",
        severity=Severity.CRITICAL,
        witness_recipe="Read a double from input and pick the least-extreme value "
                       "just past the destination integer range (e.g. INT_MAX+1); "
                       "the UBSan build traps while the target is defined.",
        int_widths=(32, 64),
    ),
    DivergenceEntry(
        cls=FAST_MATH_REASSOC,
        source_definedness=Definedness.UNSPECIFIED,
        source_rule="Under `-ffast-math`/`-Ofast` the implementation is licensed "
                    "to reassociate floating-point arithmetic as if it were over "
                    "the reals (dropping the IEEE no-reassociation guarantee), so "
                    "the value of an expression like `(x+y)-x` is not uniquely "
                    "fixed: `-fno-fast-math` rounds at each step (IEEE) while "
                    "`-ffast-math` may fold `(x+y)-x` to `y`.",
        c_standard_ref="C17 Annex F (IEC 60559) / -ffast-math non-conformance",
        rust_outcome=RustOutcomeKind.DEFINED_VALUE,
        target_rule="Rust and Go never auto-reassociate floating arithmetic: "
                    "`(x+y)-x` always evaluates IEEE-strict (round at each step), "
                    "so the target is a single deterministic, defined value.",
        severity=Severity.MODERATE,
        witness_recipe="Pick x,y with x+y rounding back to x (y swallowed) but "
                       "y != 0; `-ffast-math` then yields y while IEEE-strict "
                       "yields 0 — a visible, reproducible reassociation gap.",
    ),
    DivergenceEntry(
        cls=RESTRICT_VIOLATION,
        source_definedness=Definedness.UNDEFINED,
        source_rule="If two `restrict`-qualified pointers in scope are used to "
                    "access the same object and at least one access is a store, "
                    "the behavior is undefined; the optimizer is entitled to "
                    "assume they never alias, so `-O2` may keep a stale cached "
                    "value where `-O0` re-reads memory.",
        c_standard_ref="C17 6.7.3.1p4",
        rust_outcome=RustOutcomeKind.OWNERSHIP_REJECTED,
        target_rule="Rust's `&mut` references are guaranteed unique (the borrow "
                    "checker forbids two live `&mut` to one object at compile "
                    "time), so the idiomatic safe port cannot alias and computes "
                    "a single deterministic, defined value; Go has no `restrict` "
                    "and never performs the aliasing-based rewrite.",
        severity=Severity.CRITICAL,
        witness_recipe="Feed the function two pointers to the *same* object; the "
                       "`-O0` build re-reads (aliased) memory while `-O2` assumes "
                       "no alias and returns the stale value — the same C source "
                       "diverges across optimisation levels, with no sanitizer "
                       "able to trap it.",
    ),
    DivergenceEntry(
        cls=POINTER_PROVENANCE,
        source_definedness=Definedness.UNDEFINED,
        source_rule="A pointer may only be computed within the elements of the "
                    "array object it points into, plus the one-past-the-end "
                    "position; forming a pointer further out (in the limit, one "
                    "whose address-space computation overflows) is undefined. The "
                    "pointer's *provenance* is the original object, so the "
                    "optimiser may assume the offset stays in range.",
        c_standard_ref="C17 6.5.6p8",
        rust_outcome=RustOutcomeKind.BOUNDS_CHECKED,
        target_rule="The idiomatic safe translation keeps an *index* and accesses "
                    "through a checked operation (`a.get(i)` / a bounds-checked "
                    "slice index), so a far-out offset becomes a deterministic, "
                    "defined value — a raw out-of-provenance pointer is never "
                    "formed.",
        severity=Severity.CRITICAL,
        witness_recipe="Read an offset n and pick the least n whose byte "
                       "displacement n*sizeof(T) overflows a 64-bit address space "
                       "(n*sizeof(T) >= 2**64); the UBSan `pointer-overflow` check "
                       "traps while the target's checked index is defined.",
        int_widths=(32, 64),
    ),
    DivergenceEntry(
        cls=SIGNED_SHIFT_SIGN_BIT,
        source_definedness=Definedness.UNDEFINED,
        source_rule="In C, `1 << 31` (more generally, a left shift whose "
                    "mathematical result E1*2**E2 is not representable in the "
                    "signed result type) is undefined; the optimizer may assume "
                    "it never happens.",
        c_standard_ref="C17 6.5.7p4",
        rust_outcome=RustOutcomeKind.DEFINED_VALUE,
        target_rule="C++20 mandates two's-complement and *defines* the same "
                    "`1 << 31` to yield INT_MIN by modular wraparound "
                    "(C++20 [expr.shift]/2), so the byte-identical source token "
                    "is a single deterministic, defined value when compiled as "
                    "C++ — the divergence is across the C/C++ language boundary.",
        severity=Severity.CRITICAL,
        witness_recipe="Pick the least shift amount n<width whose `1<<n` sets the "
                       "sign bit (n = width-1); the C UBSan build traps while the "
                       "C++20 build returns INT_MIN deterministically.",
        int_widths=(32, 64),
    ),
    DivergenceEntry(
        cls=BITFIELD_LAYOUT,
        source_definedness=Definedness.IMPLEMENTATION_DEFINED,
        source_rule="The allocation order, alignment and storage unit of a C "
                    "bit-field are implementation-defined: `unsigned a:3; b:5; "
                    "c:8;` is packed into a single addressable storage unit "
                    "(here a 4-byte `unsigned`), so the struct's size and in-"
                    "memory byte image are fixed by the ABI, not the source.",
        c_standard_ref="C17 6.7.2.1p11",
        rust_outcome=RustOutcomeKind.DEFINED_VALUE,
        target_rule="A faithful field-by-field translation gives each bit-field "
                    "its own integer field (`#[repr(C)]` `u8` / Go `uint8`), so "
                    "the target struct is *unpacked*: a different size and a "
                    "different, deterministic byte image. Serialising the struct "
                    "to a wire/ABI boundary therefore diverges.",
        severity=Severity.MODERATE,
        witness_recipe="Pick nonzero field values in range; the C packed image "
                       "(one storage unit) and the unpacked target image (one "
                       "byte per field) differ in both size and bytes — a "
                       "defined ABI/serialisation divergence on real layout.",
    ),
    DivergenceEntry(
        cls=ENUM_OUT_OF_RANGE,
        source_definedness=Definedness.IMPLEMENTATION_DEFINED,
        source_rule="A C enumeration has an implementation-defined integer type "
                    "able to hold every enumerator; storing a value outside the "
                    "enumerator set (but within that type) is permitted and "
                    "reads back the raw value — `(enum E)n` is just `n`.",
        c_standard_ref="C17 6.7.2.2p4 / 6.2.6.1",
        rust_outcome=RustOutcomeKind.DEFINED_VALUE,
        target_rule="A safe target enum has no representation for an out-of-range "
                    "discriminant: the idiomatic port matches on the integer and "
                    "collapses any unknown value to a default variant (Rust "
                    "`match { _ => Default }` / Go const + switch), so the same "
                    "input yields a different, deterministic value.",
        severity=Severity.MODERATE,
        witness_recipe="Feed the least value just past the largest enumerator; C "
                       "retains it verbatim while the safe target collapses it to "
                       "its default variant — a defined-but-different value.",
    ),
    DivergenceEntry(
        cls=MEMCPY_OVERLAP,
        source_definedness=Definedness.UNDEFINED,
        source_rule="The C library `memcpy` contract requires the source and "
                    "destination ranges not to overlap; copying between "
                    "overlapping objects has undefined behavior.",
        c_standard_ref="C17 7.24.2.1p2",
        rust_outcome=RustOutcomeKind.DEFINED_VALUE,
        target_rule="Idiomatic safe translations use `memmove`-equivalent slice "
                    "operations (`slice::copy_within` in Rust, `copy` on Go "
                    "slices), which define overlapping copies by first preserving "
                    "the source bytes.",
        severity=Severity.CRITICAL,
        witness_recipe="Use a single buffer and copy n runtime bytes from offset "
                       "0 to offset 1 with n >= 4; the checked-libc contract "
                       "build reports `memcpy-param-overlap`, while the target "
                       "slice copy deterministically produces the memmove result.",
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
