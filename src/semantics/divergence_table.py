"""
Complete C↔Rust semantic divergence table.

Enumerates every category of semantic difference between C and Rust,
with per-category C semantics, Rust semantics, divergence severity,
SMT encoding helpers, and fuzzing seed strategies.  A DivergenceAnalyzer
can take two IR operations and identify all applicable divergences.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import (
    Any,
    Callable,
    Dict,
    FrozenSet,
    List,
    Optional,
    Sequence,
    Set,
    Tuple,
)

from ..ir.types import (
    IRType,
    IntType,
    FloatType,
    PointerType,
    ArrayType,
    StructType,
    UnionType,
    FunctionType,
    VoidType,
    Signedness,
    FloatKind,
    Language,
    OverflowBehavior,
)
from ..ir.instructions import (
    Instruction,
    BinaryOp,
    UnaryOp,
    CompareOp,
    CastInst,
    LoadInst,
    StoreInst,
    AllocaInst,
    GetElementPtrInst,
    CallInst,
    ReturnInst,
    BranchInst,
    SelectInst,
    PhiInst,
    BinOpKind,
    UnaryOpKind,
    CastKind,
    CmpPredicate,
    Value,
    Constant,
)


# ── Divergence classification enums ──────────────────────────────────────

class DivergenceClass(Enum):
    """Each category of semantic divergence between C and Rust."""
    SignedOverflow = auto()
    UnsignedWrap = auto()
    IntPromotion = auto()
    NegativeShift = auto()
    DivisionByZero = auto()
    FloatToIntOOB = auto()
    NullDeref = auto()
    ArrayOOB = auto()
    PointerArith = auto()
    FloatPrecision = auto()
    ErrorHandling = auto()
    EnumRepr = auto()
    BitfieldLayout = auto()
    AlignmentReqs = auto()
    VolatileSemantics = auto()


class DivergenceType(Enum):
    """Severity of a divergence."""
    Critical = auto()   # Certain behavioural difference
    Moderate = auto()   # Potential difference depending on inputs
    None_ = auto()      # No divergence for this category

    def __str__(self) -> str:
        if self is DivergenceType.None_:
            return "None"
        return self.name


# ── Per-language semantic descriptions ───────────────────────────────────

@dataclass(frozen=True)
class CSemantics:
    """Description of C behaviour for a divergence class."""
    summary: str
    is_ub: bool = False
    is_impl_defined: bool = False
    standard_ref: str = ""
    detail: str = ""

    def __str__(self) -> str:
        tag = ""
        if self.is_ub:
            tag = " [UB]"
        elif self.is_impl_defined:
            tag = " [impl-defined]"
        return f"C: {self.summary}{tag}"


@dataclass(frozen=True)
class RustSemantics:
    """Description of Rust behaviour for a divergence class."""
    summary: str
    panics_in_debug: bool = False
    wraps_in_release: bool = False
    guaranteed: bool = True
    detail: str = ""

    def __str__(self) -> str:
        tags: list[str] = []
        if self.panics_in_debug:
            tags.append("panic-debug")
        if self.wraps_in_release:
            tags.append("wrap-release")
        tag_str = f" [{', '.join(tags)}]" if tags else ""
        return f"Rust: {self.summary}{tag_str}"


# ── Fuzzing seed strategy ────────────────────────────────────────────────

@dataclass(frozen=True)
class FuzzingSeedStrategy:
    """Describes how to generate targeted fuzzing inputs for a divergence."""
    description: str
    boundary_values: Tuple[Any, ...] = ()
    generator_hint: str = ""
    priority: int = 1  # higher = more important

    def generate_seeds(self, bit_width: int = 32) -> list[int]:
        """Generate concrete seed values for integer fuzzing."""
        seeds: list[int] = []
        if self.boundary_values:
            for v in self.boundary_values:
                if isinstance(v, int):
                    seeds.append(v)
                elif callable(v):
                    seeds.append(v(bit_width))
        else:
            seeds = self._default_seeds(bit_width)
        return seeds

    @staticmethod
    def _default_seeds(bit_width: int) -> list[int]:
        max_signed = (1 << (bit_width - 1)) - 1
        min_signed = -(1 << (bit_width - 1))
        max_unsigned = (1 << bit_width) - 1
        return [
            0, 1, -1,
            max_signed, max_signed - 1,
            min_signed, min_signed + 1,
            max_unsigned, max_unsigned - 1,
            42, -42,
        ]


# ── SMT encoding helpers ────────────────────────────────────────────────

class SMTEncoding:
    """Helpers to produce SMT-LIB-like constraint strings for divergence
    detection.  These are *templates*; a real solver integration would use
    a proper API (z3-py, etc.)."""

    @staticmethod
    def signed_overflow_add(a: str, b: str, width: int) -> str:
        max_val = (1 << (width - 1)) - 1
        min_val = -(1 << (width - 1))
        return (
            f"(or (> (+ {a} {b}) {max_val}) (< (+ {a} {b}) {min_val}))"
        )

    @staticmethod
    def signed_overflow_sub(a: str, b: str, width: int) -> str:
        max_val = (1 << (width - 1)) - 1
        min_val = -(1 << (width - 1))
        return (
            f"(or (> (- {a} {b}) {max_val}) (< (- {a} {b}) {min_val}))"
        )

    @staticmethod
    def signed_overflow_mul(a: str, b: str, width: int) -> str:
        max_val = (1 << (width - 1)) - 1
        min_val = -(1 << (width - 1))
        return (
            f"(let ((prod (* {a} {b}))) "
            f"(or (> prod {max_val}) (< prod {min_val})))"
        )

    @staticmethod
    def unsigned_wrap_add(a: str, b: str, width: int) -> str:
        max_val = (1 << width) - 1
        return f"(> (+ {a} {b}) {max_val})"

    @staticmethod
    def unsigned_wrap_sub(a: str, b: str, _width: int) -> str:
        return f"(< (- {a} {b}) 0)"

    @staticmethod
    def division_by_zero(divisor: str) -> str:
        return f"(= {divisor} 0)"

    @staticmethod
    def signed_division_overflow(a: str, b: str, width: int) -> str:
        min_val = -(1 << (width - 1))
        return f"(and (= {a} {min_val}) (= {b} (- 1)))"

    @staticmethod
    def negative_shift(shift_amount: str) -> str:
        return f"(< {shift_amount} 0)"

    @staticmethod
    def oversize_shift(shift_amount: str, width: int) -> str:
        return f"(>= {shift_amount} {width})"

    @staticmethod
    def float_to_int_oob(fval: str, int_width: int, signed: bool) -> str:
        if signed:
            max_v = (1 << (int_width - 1)) - 1
            min_v = -(1 << (int_width - 1))
        else:
            max_v = (1 << int_width) - 1
            min_v = 0
        return (
            f"(or (> {fval} {max_v}.0) (< {fval} {min_v}.0) (is_nan {fval}))"
        )

    @staticmethod
    def null_pointer(ptr: str) -> str:
        return f"(= {ptr} 0)"

    @staticmethod
    def array_oob(index: str, length: str) -> str:
        return f"(or (< {index} 0) (>= {index} {length}))"

    @staticmethod
    def alignment_check(addr: str, alignment: int) -> str:
        return f"(not (= (mod {addr} {alignment}) 0))"


# ── Divergence entry (one per DivergenceClass) ───────────────────────────

@dataclass
class DivergenceEntry:
    """A complete description of one semantic divergence category."""
    cls: DivergenceClass
    c_semantics: CSemantics
    rust_semantics: RustSemantics
    divergence_type: DivergenceType
    smt_encoder: Callable[..., str] | None = None
    fuzzing_strategy: FuzzingSeedStrategy | None = None
    applicable_opcodes: FrozenSet[str] = field(default_factory=frozenset)
    description: str = ""
    mitigation: str = ""
    test_priority: int = 1

    def matches_instruction(self, inst: Instruction) -> bool:
        """Return True if this divergence could apply to *inst*."""
        opcode = inst.opcode_name()
        if self.applicable_opcodes and opcode not in self.applicable_opcodes:
            return False
        return True

    def generate_smt_constraint(self, *args: str, width: int = 32) -> str | None:
        if self.smt_encoder is None:
            return None
        return self.smt_encoder(*args, width=width)


# ── The master divergence table ──────────────────────────────────────────

class DivergenceTable:
    """Complete table of C↔Rust semantic divergences.

    Initialised with all 15 divergence categories pre-populated.
    """

    def __init__(self) -> None:
        self._entries: Dict[DivergenceClass, DivergenceEntry] = {}
        self._build_table()

    # ── public API ───────────────────────────────────────────────────────

    def __getitem__(self, cls: DivergenceClass) -> DivergenceEntry:
        return self._entries[cls]

    def __iter__(self):
        return iter(self._entries.values())

    def __len__(self) -> int:
        return len(self._entries)

    def get(self, cls: DivergenceClass) -> DivergenceEntry | None:
        return self._entries.get(cls)

    def critical_entries(self) -> list[DivergenceEntry]:
        return [e for e in self._entries.values()
                if e.divergence_type is DivergenceType.Critical]

    def moderate_entries(self) -> list[DivergenceEntry]:
        return [e for e in self._entries.values()
                if e.divergence_type is DivergenceType.Moderate]

    def entries_for_instruction(self, inst: Instruction) -> list[DivergenceEntry]:
        return [e for e in self._entries.values() if e.matches_instruction(inst)]

    def summary(self) -> str:
        lines = ["Divergence Table Summary", "=" * 50]
        for e in self._entries.values():
            lines.append(
                f"  {e.cls.name:20s} | {str(e.divergence_type):8s} | "
                f"priority={e.test_priority}"
            )
        return "\n".join(lines)

    # ── table construction ───────────────────────────────────────────────

    def _build_table(self) -> None:
        self._add_signed_overflow()
        self._add_unsigned_wrap()
        self._add_int_promotion()
        self._add_negative_shift()
        self._add_division_by_zero()
        self._add_float_to_int_oob()
        self._add_null_deref()
        self._add_array_oob()
        self._add_pointer_arith()
        self._add_float_precision()
        self._add_error_handling()
        self._add_enum_repr()
        self._add_bitfield_layout()
        self._add_alignment_reqs()
        self._add_volatile_semantics()

    def _reg(self, entry: DivergenceEntry) -> None:
        self._entries[entry.cls] = entry

    # ── 1. SignedOverflow ────────────────────────────────────────────────

    def _add_signed_overflow(self) -> None:
        def _smt_add(a: str, b: str, width: int = 32) -> str:
            return SMTEncoding.signed_overflow_add(a, b, width)

        def _smt_sub(a: str, b: str, width: int = 32) -> str:
            return SMTEncoding.signed_overflow_sub(a, b, width)

        def _smt_mul(a: str, b: str, width: int = 32) -> str:
            return SMTEncoding.signed_overflow_mul(a, b, width)

        self._reg(DivergenceEntry(
            cls=DivergenceClass.SignedOverflow,
            c_semantics=CSemantics(
                summary="Signed integer overflow is undefined behavior",
                is_ub=True,
                standard_ref="C11 §6.5/5",
                detail=(
                    "If an exceptional condition occurs during the evaluation "
                    "of an expression (that is, if the result is not "
                    "mathematically defined or not in the range of "
                    "representable values for its type), the behavior is "
                    "undefined."
                ),
            ),
            rust_semantics=RustSemantics(
                summary="Signed overflow panics in debug, wraps in release",
                panics_in_debug=True,
                wraps_in_release=True,
                detail=(
                    "In debug mode, signed arithmetic operations check for "
                    "overflow and panic.  In release mode, two's complement "
                    "wrapping is performed."
                ),
            ),
            divergence_type=DivergenceType.Critical,
            smt_encoder=_smt_add,
            fuzzing_strategy=FuzzingSeedStrategy(
                description="Target signed boundary values",
                boundary_values=(
                    lambda w: (1 << (w - 1)) - 1,   # INT_MAX
                    lambda w: -(1 << (w - 1)),       # INT_MIN
                    lambda w: (1 << (w - 1)) - 2,
                    1, -1, 0,
                ),
                generator_hint="signed_boundary",
                priority=10,
            ),
            applicable_opcodes=frozenset({"add", "sub", "mul", "neg"}),
            description=(
                "Signed integer overflow is UB in C but has defined "
                "(wrapping) behaviour in Rust release builds."
            ),
            mitigation="Use wrapping_add/sub/mul or checked variants in Rust.",
            test_priority=10,
        ))

    # ── 2. UnsignedWrap ──────────────────────────────────────────────────

    def _add_unsigned_wrap(self) -> None:
        self._reg(DivergenceEntry(
            cls=DivergenceClass.UnsignedWrap,
            c_semantics=CSemantics(
                summary="Unsigned arithmetic wraps modulo 2^N (well-defined)",
                is_ub=False,
                standard_ref="C11 §6.2.5/9",
                detail=(
                    "A computation involving unsigned operands can never "
                    "overflow, because a result that cannot be represented "
                    "by the resulting unsigned integer type is reduced "
                    "modulo the number that is one greater than the largest "
                    "value that can be represented by the resulting type."
                ),
            ),
            rust_semantics=RustSemantics(
                summary="Unsigned overflow panics in debug, wraps in release",
                panics_in_debug=True,
                wraps_in_release=True,
                detail=(
                    "Even unsigned operations are checked in debug mode "
                    "and will panic on overflow."
                ),
            ),
            divergence_type=DivergenceType.Moderate,
            smt_encoder=lambda a, b, width=32: SMTEncoding.unsigned_wrap_add(a, b, width),
            fuzzing_strategy=FuzzingSeedStrategy(
                description="Target unsigned boundary values",
                boundary_values=(
                    lambda w: (1 << w) - 1,  # UINT_MAX
                    lambda w: (1 << w) - 2,
                    0, 1,
                ),
                generator_hint="unsigned_boundary",
                priority=7,
            ),
            applicable_opcodes=frozenset({"add", "sub", "mul"}),
            description=(
                "C unsigned wrapping is always well-defined. Rust panics "
                "in debug mode on unsigned overflow."
            ),
            mitigation="Use wrapping_* methods for intentional wrapping.",
            test_priority=7,
        ))

    # ── 3. IntPromotion ──────────────────────────────────────────────────

    def _add_int_promotion(self) -> None:
        self._reg(DivergenceEntry(
            cls=DivergenceClass.IntPromotion,
            c_semantics=CSemantics(
                summary="C performs implicit integer promotions",
                is_ub=False,
                is_impl_defined=True,
                standard_ref="C11 §6.3.1.1",
                detail=(
                    "Integer types smaller than int are promoted to int "
                    "(or unsigned int) before arithmetic operations.  The "
                    "width of int is implementation-defined."
                ),
            ),
            rust_semantics=RustSemantics(
                summary="Rust requires explicit casts for all integer conversions",
                guaranteed=True,
                detail="No implicit integer promotion occurs in Rust.",
            ),
            divergence_type=DivergenceType.Critical,
            fuzzing_strategy=FuzzingSeedStrategy(
                description="Small-type boundary values that trigger promotion",
                boundary_values=(127, 128, 255, 256, -128, -129, 0, 1),
                generator_hint="promotion_boundary",
                priority=8,
            ),
            applicable_opcodes=frozenset({
                "add", "sub", "mul", "sdiv", "udiv",
                "srem", "urem", "shl", "lshr", "ashr",
                "and", "or", "xor",
            }),
            description=(
                "C silently widens narrow integers before arithmetic; "
                "Rust never does implicit promotion."
            ),
            mitigation="Insert explicit casts to match C promotion semantics.",
            test_priority=8,
        ))

    # ── 4. NegativeShift ─────────────────────────────────────────────────

    def _add_negative_shift(self) -> None:
        self._reg(DivergenceEntry(
            cls=DivergenceClass.NegativeShift,
            c_semantics=CSemantics(
                summary="Shifting by a negative or >= width amount is UB",
                is_ub=True,
                standard_ref="C11 §6.5.7/3",
                detail=(
                    "The behavior is undefined if the right operand is "
                    "negative, or greater than or equal to the width of "
                    "the promoted left operand."
                ),
            ),
            rust_semantics=RustSemantics(
                summary="Shift overflow panics in debug, masks in release",
                panics_in_debug=True,
                wraps_in_release=True,
                detail=(
                    "In release mode Rust masks the shift amount to "
                    "bit_width - 1, so e.g. (x << 33) on u32 becomes "
                    "(x << 1)."
                ),
            ),
            divergence_type=DivergenceType.Critical,
            smt_encoder=lambda _a, shift, width=32: (
                f"(or {SMTEncoding.negative_shift(shift)} "
                f"{SMTEncoding.oversize_shift(shift, width)})"
            ),
            fuzzing_strategy=FuzzingSeedStrategy(
                description="Negative and oversized shift amounts",
                boundary_values=(-1, -32, 0, 31, 32, 33, 63, 64, 127, 128),
                generator_hint="shift_boundary",
                priority=9,
            ),
            applicable_opcodes=frozenset({"shl", "lshr", "ashr"}),
            description="C UB on bad shift amounts vs Rust panic/mask.",
            mitigation="Validate shift amounts before shifting.",
            test_priority=9,
        ))

    # ── 5. DivisionByZero ────────────────────────────────────────────────

    def _add_division_by_zero(self) -> None:
        self._reg(DivergenceEntry(
            cls=DivergenceClass.DivisionByZero,
            c_semantics=CSemantics(
                summary="Division by zero is undefined behavior",
                is_ub=True,
                standard_ref="C11 §6.5.5/5",
                detail="The result of the / operator is UB if the divisor is 0.",
            ),
            rust_semantics=RustSemantics(
                summary="Division by zero always panics",
                panics_in_debug=True,
                guaranteed=True,
                detail="Rust panics on division by zero in both debug and release.",
            ),
            divergence_type=DivergenceType.Critical,
            smt_encoder=lambda _a, b, width=32: SMTEncoding.division_by_zero(b),
            fuzzing_strategy=FuzzingSeedStrategy(
                description="Zero divisors and INT_MIN/-1",
                boundary_values=(0, 1, -1, lambda w: -(1 << (w - 1))),
                generator_hint="division_boundary",
                priority=10,
            ),
            applicable_opcodes=frozenset({"sdiv", "udiv", "srem", "urem"}),
            description="C UB vs Rust panic on division by zero.",
            mitigation="Guard division with a zero check.",
            test_priority=10,
        ))

    # ── 6. FloatToIntOOB ─────────────────────────────────────────────────

    def _add_float_to_int_oob(self) -> None:
        self._reg(DivergenceEntry(
            cls=DivergenceClass.FloatToIntOOB,
            c_semantics=CSemantics(
                summary="Float-to-int cast out of range is UB",
                is_ub=True,
                standard_ref="C11 §6.3.1.4/1",
                detail=(
                    "When a finite value of real floating type is converted "
                    "to an integer type other than _Bool, the fractional "
                    "part is discarded.  If the value cannot be represented, "
                    "the behavior is undefined."
                ),
            ),
            rust_semantics=RustSemantics(
                summary="Float-to-int saturates (since Rust 1.45)",
                guaranteed=True,
                detail=(
                    "Since Rust 1.45, `as` casts from float to int "
                    "saturate to the min/max of the integer type, and "
                    "NaN maps to 0."
                ),
            ),
            divergence_type=DivergenceType.Critical,
            smt_encoder=lambda fval, _b="", width=32: (
                SMTEncoding.float_to_int_oob(fval, width, signed=True)
            ),
            fuzzing_strategy=FuzzingSeedStrategy(
                description="Float values outside integer range, NaN, Inf",
                boundary_values=(),
                generator_hint="float_oob",
                priority=8,
            ),
            applicable_opcodes=frozenset({"fptosi", "fptoui"}),
            description="C UB vs Rust saturation for out-of-range float-to-int.",
            mitigation="Clamp float value before casting.",
            test_priority=8,
        ))

    # ── 7. NullDeref ─────────────────────────────────────────────────────

    def _add_null_deref(self) -> None:
        self._reg(DivergenceEntry(
            cls=DivergenceClass.NullDeref,
            c_semantics=CSemantics(
                summary="Dereferencing a null pointer is UB",
                is_ub=True,
                standard_ref="C11 §6.5.3.2/4",
                detail="Dereferencing a null pointer is undefined behaviour in C.",
            ),
            rust_semantics=RustSemantics(
                summary="Safe Rust prevents null pointers; unsafe deref is UB",
                guaranteed=True,
                detail=(
                    "References in safe Rust are always non-null.  Unsafe "
                    "code dereferencing a null raw pointer is UB."
                ),
            ),
            divergence_type=DivergenceType.Critical,
            smt_encoder=lambda ptr, _b="", width=64: SMTEncoding.null_pointer(ptr),
            fuzzing_strategy=FuzzingSeedStrategy(
                description="Null and near-null pointer values",
                boundary_values=(0, 1, -1),
                generator_hint="null_pointer",
                priority=10,
            ),
            applicable_opcodes=frozenset({"load", "store"}),
            description="Both C and Rust UB, but Rust safe code prevents it.",
            mitigation="Use Option<&T> instead of raw pointers.",
            test_priority=10,
        ))

    # ── 8. ArrayOOB ──────────────────────────────────────────────────────

    def _add_array_oob(self) -> None:
        self._reg(DivergenceEntry(
            cls=DivergenceClass.ArrayOOB,
            c_semantics=CSemantics(
                summary="Array out-of-bounds access is UB",
                is_ub=True,
                standard_ref="C11 §6.5.6/8",
                detail="Accessing past the end of an array is UB in C.",
            ),
            rust_semantics=RustSemantics(
                summary="Array indexing panics on out-of-bounds",
                panics_in_debug=True,
                guaranteed=True,
                detail="Rust slice/array indexing always performs bounds checks.",
            ),
            divergence_type=DivergenceType.Critical,
            smt_encoder=lambda idx, length, width=32: SMTEncoding.array_oob(idx, length),
            fuzzing_strategy=FuzzingSeedStrategy(
                description="Indices at and past array boundaries",
                boundary_values=(-1, 0, 1),
                generator_hint="array_boundary",
                priority=9,
            ),
            applicable_opcodes=frozenset({"gep", "load", "store"}),
            description="C UB vs Rust panic on array out-of-bounds.",
            mitigation="Use get() or get_unchecked() with manual bounds check.",
            test_priority=9,
        ))

    # ── 9. PointerArith ──────────────────────────────────────────────────

    def _add_pointer_arith(self) -> None:
        self._reg(DivergenceEntry(
            cls=DivergenceClass.PointerArith,
            c_semantics=CSemantics(
                summary="Pointer arithmetic outside object is UB",
                is_ub=True,
                standard_ref="C11 §6.5.6/8",
                detail=(
                    "Computing a pointer outside the bounds of an object "
                    "(more than one past the end) is UB."
                ),
            ),
            rust_semantics=RustSemantics(
                summary="Safe Rust disallows raw pointer arithmetic; unsafe follows C rules",
                guaranteed=True,
                detail=(
                    "In safe Rust, pointer arithmetic is not directly available.  "
                    "Unsafe pointer::offset has similar UB to C."
                ),
            ),
            divergence_type=DivergenceType.Moderate,
            fuzzing_strategy=FuzzingSeedStrategy(
                description="Large positive and negative offsets",
                boundary_values=(0, 1, -1, lambda w: (1 << (w - 1)) - 1),
                generator_hint="pointer_offset",
                priority=6,
            ),
            applicable_opcodes=frozenset({"gep"}),
            description="Pointer arithmetic UB in both languages for unsafe code.",
            mitigation="Use slice indexing instead of pointer arithmetic.",
            test_priority=6,
        ))

    # ── 10. FloatPrecision ───────────────────────────────────────────────

    def _add_float_precision(self) -> None:
        self._reg(DivergenceEntry(
            cls=DivergenceClass.FloatPrecision,
            c_semantics=CSemantics(
                summary="C allows excess precision in intermediate results",
                is_impl_defined=True,
                standard_ref="C11 §5.2.4.2.2",
                detail=(
                    "FLT_EVAL_METHOD controls whether intermediates are "
                    "evaluated at higher precision.  This can cause "
                    "different results depending on platform."
                ),
            ),
            rust_semantics=RustSemantics(
                summary="Rust does not guarantee evaluation precision",
                guaranteed=False,
                detail=(
                    "Rust float semantics mostly follow IEEE 754 but "
                    "do not specify intermediate precision handling."
                ),
            ),
            divergence_type=DivergenceType.Moderate,
            fuzzing_strategy=FuzzingSeedStrategy(
                description="Values near float precision boundaries",
                boundary_values=(),
                generator_hint="float_precision",
                priority=4,
            ),
            applicable_opcodes=frozenset({
                "fadd", "fsub", "fmul", "fdiv", "frem",
            }),
            description="Different intermediate precision may cause different results.",
            mitigation="Use volatile stores to force intermediate precision.",
            test_priority=4,
        ))

    # ── 11. ErrorHandling ────────────────────────────────────────────────

    def _add_error_handling(self) -> None:
        self._reg(DivergenceEntry(
            cls=DivergenceClass.ErrorHandling,
            c_semantics=CSemantics(
                summary="C uses errno / return codes for error handling",
                is_ub=False,
                detail=(
                    "C standard library functions typically signal errors "
                    "via errno or return values.  There is no unwinding."
                ),
            ),
            rust_semantics=RustSemantics(
                summary="Rust uses Result<T,E> and panic! for error handling",
                guaranteed=True,
                detail=(
                    "Rust errors are returned via Result or Option.  "
                    "Unrecoverable errors use panic! which unwinds or aborts."
                ),
            ),
            divergence_type=DivergenceType.Moderate,
            fuzzing_strategy=FuzzingSeedStrategy(
                description="Error-triggering inputs for standard library calls",
                boundary_values=(),
                generator_hint="error_trigger",
                priority=5,
            ),
            applicable_opcodes=frozenset({"call"}),
            description="Different error-handling paradigms.",
            mitigation="Map C errno patterns to Rust Result types.",
            test_priority=5,
        ))

    # ── 12. EnumRepr ─────────────────────────────────────────────────────

    def _add_enum_repr(self) -> None:
        self._reg(DivergenceEntry(
            cls=DivergenceClass.EnumRepr,
            c_semantics=CSemantics(
                summary="C enum underlying type is implementation-defined",
                is_impl_defined=True,
                standard_ref="C11 §6.7.2.2",
                detail=(
                    "The C standard allows the compiler to choose any "
                    "integer type that can represent all enumerator values."
                ),
            ),
            rust_semantics=RustSemantics(
                summary="Rust enums have isize repr by default, or explicit #[repr]",
                guaranteed=True,
                detail=(
                    "Without #[repr], Rust chooses the smallest type.  "
                    "With #[repr(C)], it matches C layout."
                ),
            ),
            divergence_type=DivergenceType.Moderate,
            fuzzing_strategy=FuzzingSeedStrategy(
                description="Enum values near type boundaries",
                boundary_values=(0, 127, 128, 255, 256, -128, -129),
                generator_hint="enum_boundary",
                priority=5,
            ),
            applicable_opcodes=frozenset(),
            description="Different enum representation between C and Rust.",
            mitigation="Use #[repr(C)] on Rust enums to match C layout.",
            test_priority=5,
        ))

    # ── 13. BitfieldLayout ───────────────────────────────────────────────

    def _add_bitfield_layout(self) -> None:
        self._reg(DivergenceEntry(
            cls=DivergenceClass.BitfieldLayout,
            c_semantics=CSemantics(
                summary="Bit-field layout is implementation-defined",
                is_impl_defined=True,
                standard_ref="C11 §6.7.2.1",
                detail=(
                    "The order of allocation of bit-fields within a unit, "
                    "alignment, and padding are all implementation-defined."
                ),
            ),
            rust_semantics=RustSemantics(
                summary="Rust has no native bit-fields; use bitflags or manual bit ops",
                guaranteed=True,
                detail=(
                    "Rust does not support bit-fields directly.  Libraries "
                    "like bitflags or manual shift/mask are used instead."
                ),
            ),
            divergence_type=DivergenceType.Moderate,
            fuzzing_strategy=FuzzingSeedStrategy(
                description="Values that differ across bit-field layouts",
                boundary_values=(),
                generator_hint="bitfield_layout",
                priority=3,
            ),
            applicable_opcodes=frozenset(),
            description="No direct Rust equivalent for C bit-fields.",
            mitigation="Use explicit shift/mask operations for bit-field access.",
            test_priority=3,
        ))

    # ── 14. AlignmentReqs ────────────────────────────────────────────────

    def _add_alignment_reqs(self) -> None:
        self._reg(DivergenceEntry(
            cls=DivergenceClass.AlignmentReqs,
            c_semantics=CSemantics(
                summary="Misaligned access may be UB or silently work",
                is_ub=True,
                standard_ref="C11 §6.3.2.3/7",
                detail=(
                    "Converting a pointer to a type with stricter alignment "
                    "and then dereferencing is UB."
                ),
            ),
            rust_semantics=RustSemantics(
                summary="Misaligned reference creation is instant UB",
                guaranteed=True,
                detail=(
                    "Creating a reference to a misaligned value is UB in "
                    "Rust, even without dereferencing."
                ),
            ),
            divergence_type=DivergenceType.Moderate,
            smt_encoder=lambda addr, _b="", width=64: (
                SMTEncoding.alignment_check(addr, 8)
            ),
            fuzzing_strategy=FuzzingSeedStrategy(
                description="Misaligned addresses",
                boundary_values=(1, 3, 5, 7),
                generator_hint="alignment",
                priority=6,
            ),
            applicable_opcodes=frozenset({"load", "store", "bitcast"}),
            description="Different alignment strictness between C and Rust.",
            mitigation="Use read_unaligned/write_unaligned for potentially misaligned access.",
            test_priority=6,
        ))

    # ── 15. VolatileSemantics ────────────────────────────────────────────

    def _add_volatile_semantics(self) -> None:
        self._reg(DivergenceEntry(
            cls=DivergenceClass.VolatileSemantics,
            c_semantics=CSemantics(
                summary="volatile prevents certain optimizations",
                is_impl_defined=True,
                standard_ref="C11 §6.7.3/7",
                detail=(
                    "Accesses to volatile objects are 'observable behavior' "
                    "and must not be optimized away.  The semantics of "
                    "volatile are otherwise implementation-defined."
                ),
            ),
            rust_semantics=RustSemantics(
                summary="Rust uses read_volatile/write_volatile intrinsics",
                guaranteed=True,
                detail=(
                    "Rust provides read_volatile and write_volatile as "
                    "explicit intrinsics rather than a type qualifier."
                ),
            ),
            divergence_type=DivergenceType.Moderate,
            fuzzing_strategy=FuzzingSeedStrategy(
                description="Volatile read/write sequences",
                boundary_values=(),
                generator_hint="volatile",
                priority=3,
            ),
            applicable_opcodes=frozenset({"load", "store"}),
            description="Different volatile semantics and syntax.",
            mitigation="Map C volatile accesses to read_volatile/write_volatile.",
            test_priority=3,
        ))


# ── Divergence Analyzer ──────────────────────────────────────────────────

@dataclass
class DivergenceReport:
    """Report produced for a single instruction or instruction pair."""
    instruction: Instruction | None
    c_instruction: Instruction | None = None
    rust_instruction: Instruction | None = None
    applicable: list[DivergenceEntry] = field(default_factory=list)
    critical: list[DivergenceEntry] = field(default_factory=list)
    moderate: list[DivergenceEntry] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    @property
    def has_divergence(self) -> bool:
        return len(self.critical) > 0 or len(self.moderate) > 0

    @property
    def max_severity(self) -> DivergenceType:
        if self.critical:
            return DivergenceType.Critical
        if self.moderate:
            return DivergenceType.Moderate
        return DivergenceType.None_

    def summary(self) -> str:
        lines: list[str] = []
        inst_name = ""
        if self.instruction:
            inst_name = self.instruction.opcode_name()
        lines.append(f"Divergence report for '{inst_name}':")
        lines.append(f"  Critical: {len(self.critical)}")
        lines.append(f"  Moderate: {len(self.moderate)}")
        for e in self.critical:
            lines.append(f"    [CRITICAL] {e.cls.name}: {e.description}")
        for e in self.moderate:
            lines.append(f"    [MODERATE] {e.cls.name}: {e.description}")
        for n in self.notes:
            lines.append(f"    NOTE: {n}")
        return "\n".join(lines)


class DivergenceAnalyzer:
    """Analyzes IR operations to identify applicable C↔Rust divergences.

    Can analyze:
    - A single instruction (reports all potentially applicable divergences)
    - A pair of (C-origin, Rust-origin) instructions (compares specifics)
    - An entire function
    """

    def __init__(self, table: DivergenceTable | None = None) -> None:
        self.table = table or DivergenceTable()

    def analyze_instruction(self, inst: Instruction) -> DivergenceReport:
        """Analyze a single instruction for all potentially applicable divergences."""
        report = DivergenceReport(instruction=inst)

        for entry in self.table:
            if not entry.matches_instruction(inst):
                continue

            # Additional type-aware filtering
            if not self._type_filter(inst, entry):
                continue

            report.applicable.append(entry)
            if entry.divergence_type is DivergenceType.Critical:
                report.critical.append(entry)
            elif entry.divergence_type is DivergenceType.Moderate:
                report.moderate.append(entry)

        # Add context-specific notes
        self._add_notes(inst, report)
        return report

    def analyze_pair(
        self,
        c_inst: Instruction,
        rust_inst: Instruction,
    ) -> DivergenceReport:
        """Compare a C-origin instruction with its Rust counterpart."""
        report = DivergenceReport(
            instruction=c_inst,
            c_instruction=c_inst,
            rust_instruction=rust_inst,
        )

        c_report = self.analyze_instruction(c_inst)
        for entry in c_report.applicable:
            report.applicable.append(entry)
            if entry.divergence_type is DivergenceType.Critical:
                report.critical.append(entry)
            elif entry.divergence_type is DivergenceType.Moderate:
                report.moderate.append(entry)

        # Check for type mismatches between the pair
        if c_inst.type != rust_inst.type:
            report.notes.append(
                f"Result type mismatch: C={c_inst.type} vs Rust={rust_inst.type}"
            )

        # Check for opcode differences
        if c_inst.opcode_name() != rust_inst.opcode_name():
            report.notes.append(
                f"Opcode mismatch: C={c_inst.opcode_name()} vs "
                f"Rust={rust_inst.opcode_name()}"
            )

        # Check operand type differences
        c_ops = c_inst.operands
        r_ops = rust_inst.operands
        if len(c_ops) == len(r_ops):
            for i, (co, ro) in enumerate(zip(c_ops, r_ops)):
                if co.type != ro.type:
                    report.notes.append(
                        f"Operand {i} type mismatch: C={co.type} vs Rust={ro.type}"
                    )

        return report

    def analyze_function(
        self,
        func: "Function",
    ) -> list[DivergenceReport]:
        """Analyze all instructions in a function."""
        from ..ir.function import Function
        reports: list[DivergenceReport] = []
        for inst in func.iter_instructions():
            report = self.analyze_instruction(inst)
            if report.has_divergence:
                reports.append(report)
        return reports

    def generate_fuzzing_seeds(
        self,
        inst: Instruction,
        bit_width: int = 32,
    ) -> dict[DivergenceClass, list[int]]:
        """Generate fuzzing seeds for all applicable divergences of *inst*."""
        seeds: dict[DivergenceClass, list[int]] = {}
        report = self.analyze_instruction(inst)
        for entry in report.applicable:
            if entry.fuzzing_strategy:
                entry_seeds = entry.fuzzing_strategy.generate_seeds(bit_width)
                if entry_seeds:
                    seeds[entry.cls] = entry_seeds
        return seeds

    def generate_smt_constraints(
        self,
        inst: Instruction,
        operand_names: list[str] | None = None,
        width: int = 32,
    ) -> dict[DivergenceClass, str]:
        """Generate SMT constraints for divergence detection."""
        constraints: dict[DivergenceClass, str] = {}
        if operand_names is None:
            operand_names = [f"op{i}" for i in range(len(inst.operands))]

        report = self.analyze_instruction(inst)
        for entry in report.applicable:
            if entry.smt_encoder is not None:
                try:
                    args = operand_names[:2]
                    while len(args) < 2:
                        args.append("_")
                    constraint = entry.smt_encoder(*args, width=width)
                    constraints[entry.cls] = constraint
                except Exception:
                    pass
        return constraints

    # ── Private helpers ──────────────────────────────────────────────────

    def _type_filter(self, inst: Instruction, entry: DivergenceEntry) -> bool:
        """Additional type-based filtering beyond opcode matching."""
        cls = entry.cls

        if cls is DivergenceClass.SignedOverflow:
            if isinstance(inst, BinaryOp):
                if isinstance(inst.lhs.type, IntType):
                    return inst.lhs.type.is_signed
            if isinstance(inst, UnaryOp) and inst.op is UnaryOpKind.NEG:
                if isinstance(inst.operands[0].type, IntType):
                    return inst.operands[0].type.is_signed
            return False

        if cls is DivergenceClass.UnsignedWrap:
            if isinstance(inst, BinaryOp):
                if isinstance(inst.lhs.type, IntType):
                    return inst.lhs.type.is_unsigned
            return False

        if cls is DivergenceClass.IntPromotion:
            if isinstance(inst, BinaryOp):
                if isinstance(inst.lhs.type, IntType):
                    return inst.lhs.type.width < 32
            return False

        if cls is DivergenceClass.FloatToIntOOB:
            if isinstance(inst, CastInst):
                return inst.cast_kind in (CastKind.FPTOSI, CastKind.FPTOUI)
            return False

        if cls is DivergenceClass.FloatPrecision:
            if isinstance(inst, BinaryOp):
                return inst.is_floating()
            return False

        return True

    def _add_notes(self, inst: Instruction, report: DivergenceReport) -> None:
        """Add context-specific notes to the report."""
        if isinstance(inst, BinaryOp):
            if inst.op in (BinOpKind.SDIV, BinOpKind.SREM):
                if isinstance(inst.lhs.type, IntType):
                    min_val = inst.lhs.type.min_value
                    report.notes.append(
                        f"INT_MIN / -1 also causes signed division overflow "
                        f"(INT_MIN = {min_val})"
                    )
            if inst.is_shift():
                if isinstance(inst.lhs.type, IntType):
                    report.notes.append(
                        f"Shift amount must be in [0, {inst.lhs.type.width})"
                    )

        if isinstance(inst, CastInst):
            if inst.cast_kind is CastKind.TRUNC:
                if isinstance(inst.src_type, IntType) and isinstance(inst.dest_type, IntType):
                    report.notes.append(
                        f"Truncation from {inst.src_type.width}-bit to "
                        f"{inst.dest_type.width}-bit may lose information"
                    )

        if isinstance(inst, (LoadInst, StoreInst)):
            report.notes.append(
                "Check pointer validity and alignment before memory access"
            )


# ── Convenience ──────────────────────────────────────────────────────────

def get_default_table() -> DivergenceTable:
    """Return a freshly-constructed default divergence table."""
    return DivergenceTable()


def quick_analyze(inst: Instruction) -> DivergenceReport:
    """One-shot analysis of a single instruction."""
    return DivergenceAnalyzer().analyze_instruction(inst)
