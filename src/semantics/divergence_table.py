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
    ExtractValueInst,
    InsertValueInst,
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
    # Original 15 classes
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
    # New classes: pointer semantics and memory model
    PointerCast = auto()
    PointerProvenance = auto()
    StructLayout = auto()
    UnionReinterpret = auto()
    EnumDiscriminant = auto()
    StringEncoding = auto()
    MallocFree = auto()
    FunctionPointer = auto()
    StackAlloc = auto()
    LifetimeDangle = auto()
    SliceVsRawPtr = auto()


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

    @staticmethod
    def pointer_cast_invalid(ptr: str, src_size: int, dst_size: int) -> str:
        return (
            f"(and (not (= {ptr} 0)) "
            f"(not (= (mod {ptr} {dst_size}) 0)))"
        )

    @staticmethod
    def provenance_violation(ptr: str, alloc_base: str, alloc_size: str) -> str:
        return (
            f"(or (< {ptr} {alloc_base}) "
            f"(>= {ptr} (+ {alloc_base} {alloc_size})))"
        )

    @staticmethod
    def struct_padding_mismatch(c_size: str, rust_size: str) -> str:
        return f"(not (= {c_size} {rust_size}))"

    @staticmethod
    def union_reinterpret(val: str, src_width: int, dst_width: int) -> str:
        if dst_width > src_width:
            return f"(> (bv2nat {val}) {(1 << src_width) - 1})"
        return f"(not (= ((_ extract {dst_width-1} 0) {val}) {val}))"

    @staticmethod
    def enum_discriminant_mismatch(c_disc: str, rust_disc: str) -> str:
        return f"(not (= {c_disc} {rust_disc}))"

    @staticmethod
    def string_encoding_mismatch(c_len: str, rust_len: str) -> str:
        return f"(not (= {c_len} {rust_len}))"

    @staticmethod
    def double_free(ptr: str, freed_set: str) -> str:
        return f"(select {freed_set} {ptr})"

    @staticmethod
    def dangling_pointer(ptr: str, live_set: str) -> str:
        return f"(not (select {live_set} {ptr}))"

    @staticmethod
    def function_pointer_null(fptr: str) -> str:
        return f"(= {fptr} 0)"

    @staticmethod
    def stack_lifetime_escape(ptr: str, frame_base: str, frame_size: str) -> str:
        return (
            f"(and (>= {ptr} {frame_base}) "
            f"(< {ptr} (+ {frame_base} {frame_size})))"
        )

    @staticmethod
    def slice_bounds_check(ptr: str, len_var: str, idx: str) -> str:
        return f"(or (< {idx} 0) (>= {idx} {len_var}))"

    @staticmethod
    def lifetime_violation(ptr: str, scope_start: str, scope_end: str) -> str:
        """Detect use-after-free: pointer used outside its valid scope."""
        return (
            f"(or (< {ptr} {scope_start}) (>= {ptr} {scope_end}))"
        )

    @staticmethod
    def borrow_aliasing(ptr1: str, ptr2: str) -> str:
        """Detect aliasing rule violations: two mutable borrows overlap."""
        return f"(= {ptr1} {ptr2})"

    @staticmethod
    def repr_c_layout(fields: str, alignments: str) -> str:
        """Check struct layout consistency under #[repr(C)]."""
        return (
            f"(not (= (c_layout {fields} {alignments}) "
            f"(rust_layout {fields} {alignments})))"
        )

    @staticmethod
    def transmute_safety(src_size: str, dst_size: str) -> str:
        """Check that transmute source and destination have equal sizes."""
        return f"(not (= {src_size} {dst_size}))"

    @staticmethod
    def integer_promotion_c(val: str, from_width: int, to_width: int) -> str:
        """Model C integer promotion from narrow to wider type."""
        max_from = (1 << (from_width - 1)) - 1
        min_from = -(1 << (from_width - 1))
        return (
            f"(and (>= {val} {min_from}) (<= {val} {max_from}) "
            f"(not (= (sext {val} {from_width} {to_width}) "
            f"(zext {val} {from_width} {to_width}))))"
        )


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

    def get_divergences_for_pair(
        self,
        c_inst: Instruction,
        rust_inst: Instruction,
    ) -> list[DivergenceEntry]:
        """Return all applicable divergences considering both instructions' types.

        Unions the divergence entries matched by either instruction, so that
        cross-language patterns (e.g. C cast paired with Rust bounds check)
        are captured.
        """
        seen: Set[DivergenceClass] = set()
        result: list[DivergenceEntry] = []
        for inst in (c_inst, rust_inst):
            for entry in self._entries.values():
                if entry.cls in seen:
                    continue
                if entry.matches_instruction(inst):
                    seen.add(entry.cls)
                    result.append(entry)
        return result

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
        # New pointer semantics and memory model classes
        self._add_pointer_cast()
        self._add_pointer_provenance()
        self._add_struct_layout()
        self._add_union_reinterpret()
        self._add_enum_discriminant()
        self._add_string_encoding()
        self._add_malloc_free()
        self._add_function_pointer()
        self._add_stack_alloc()
        self._add_lifetime_dangle()
        self._add_slice_vs_raw_ptr()

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
            applicable_opcodes=frozenset({
                "gep", "getelementptr", "getelementptr.inbounds",
                "load", "store", "extractvalue",
            }),
            description="C UB vs Rust panic on array out-of-bounds.",
            mitigation=(
                "Use get() or get_unchecked() with manual bounds check. "
                "For slices, prefer iterators over raw indexing."
            ),
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
            applicable_opcodes=frozenset({
                "gep", "getelementptr", "getelementptr.inbounds",
            }),
            description="Pointer arithmetic UB in both languages for unsafe code.",
            mitigation=(
                "Use slice indexing instead of pointer arithmetic. "
                "For unsafe code, use wrapping_offset or pointer::add with bounds checks."
            ),
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

    # ── 16. PointerCast ──────────────────────────────────────────────────

    def _add_pointer_cast(self) -> None:
        self._reg(DivergenceEntry(
            cls=DivergenceClass.PointerCast,
            c_semantics=CSemantics(
                summary="Pointer casts are implicit and unchecked",
                is_ub=False,
                is_impl_defined=True,
                standard_ref="C11 §6.3.2.3",
                detail=(
                    "C permits casting between pointer types freely. "
                    "Casts between incompatible types that are then "
                    "dereferenced violate strict aliasing (UB)."
                ),
            ),
            rust_semantics=RustSemantics(
                summary="Pointer casts require explicit 'as' or transmute in unsafe",
                guaranteed=True,
                detail=(
                    "Rust requires explicit casts between raw pointer types "
                    "and forbids safe reference-to-different-type casts. "
                    "Transmute is available but unsafe."
                ),
            ),
            divergence_type=DivergenceType.Critical,
            smt_encoder=lambda ptr, _b="", width=64: (
                SMTEncoding.pointer_cast_invalid(ptr, 8, 4)
            ),
            fuzzing_strategy=FuzzingSeedStrategy(
                description="Pointers at alignment boundaries",
                boundary_values=(0, 1, 2, 3, 4, 7, 8, 15, 16),
                generator_hint="pointer_cast",
                priority=8,
            ),
            applicable_opcodes=frozenset({"bitcast", "inttoptr", "ptrtoint"}),
            description=(
                "C allows implicit pointer casts; Rust requires explicit "
                "unsafe casts between pointer types."
            ),
            mitigation="Use safe Rust references or explicit as casts with safety comments.",
            test_priority=8,
        ))

    # ── 17. PointerProvenance ────────────────────────────────────────────

    def _add_pointer_provenance(self) -> None:
        self._reg(DivergenceEntry(
            cls=DivergenceClass.PointerProvenance,
            c_semantics=CSemantics(
                summary="C pointer provenance is weakly specified (PNVI-ae-udi model)",
                is_ub=False,
                is_impl_defined=True,
                standard_ref="C11 §6.5.6 + defect reports",
                detail=(
                    "C has no formal provenance model in the standard. "
                    "Compilers apply provenance-based alias analysis, "
                    "making integer-to-pointer roundtrips unreliable."
                ),
            ),
            rust_semantics=RustSemantics(
                summary="Rust strictly tracks pointer provenance via Stacked Borrows",
                guaranteed=True,
                detail=(
                    "Rust's Stacked Borrows / Tree Borrows model strictly "
                    "tracks pointer provenance. Integer-to-pointer casts "
                    "via strict_provenance API (ptr::with_addr)."
                ),
            ),
            divergence_type=DivergenceType.Critical,
            smt_encoder=lambda ptr, alloc_base, width=64: (
                SMTEncoding.provenance_violation(ptr, alloc_base, str(width))
            ),
            fuzzing_strategy=FuzzingSeedStrategy(
                description="Pointer roundtrip through integer and back",
                boundary_values=(0, 1),
                generator_hint="provenance_roundtrip",
                priority=7,
            ),
            applicable_opcodes=frozenset({
                "inttoptr", "ptrtoint",
                "gep", "getelementptr", "getelementptr.inbounds",
            }),
            description=(
                "C pointer provenance is undefined; Rust uses strict "
                "Stacked/Tree Borrows provenance model."
            ),
            mitigation="Use ptr::with_addr for integer-to-pointer roundtrips.",
            test_priority=7,
        ))

    # ── 18. StructLayout ─────────────────────────────────────────────────

    def _add_struct_layout(self) -> None:
        self._reg(DivergenceEntry(
            cls=DivergenceClass.StructLayout,
            c_semantics=CSemantics(
                summary="Struct layout follows platform ABI with padding",
                is_impl_defined=True,
                standard_ref="C11 §6.7.2.1",
                detail=(
                    "C struct layout is defined by the platform ABI. "
                    "Fields are laid out in declaration order with "
                    "implementation-defined padding for alignment."
                ),
            ),
            rust_semantics=RustSemantics(
                summary="Default Rust struct layout is unspecified; #[repr(C)] matches C",
                guaranteed=False,
                detail=(
                    "Without #[repr(C)], Rust may reorder fields and "
                    "use different padding. #[repr(C)] matches C layout."
                ),
            ),
            divergence_type=DivergenceType.Critical,
            smt_encoder=lambda c_size, rust_size, width=32: (
                SMTEncoding.struct_padding_mismatch(c_size, rust_size)
            ),
            fuzzing_strategy=FuzzingSeedStrategy(
                description="Structs with varying field sizes to trigger padding",
                boundary_values=(),
                generator_hint="struct_padding",
                priority=8,
            ),
            applicable_opcodes=frozenset({
                "gep", "getelementptr", "getelementptr.inbounds",
                "extractvalue", "insertvalue",
            }),
            description=(
                "C struct layout is ABI-defined; Rust default layout "
                "is unspecified and may differ."
            ),
            mitigation="Use #[repr(C)] on all FFI-visible structs.",
            test_priority=8,
        ))

    # ── 19. UnionReinterpret ─────────────────────────────────────────────

    def _add_union_reinterpret(self) -> None:
        self._reg(DivergenceEntry(
            cls=DivergenceClass.UnionReinterpret,
            c_semantics=CSemantics(
                summary="Union type-punning is implementation-defined (GCC: defined)",
                is_impl_defined=True,
                standard_ref="C11 §6.5.2.3, Annex J",
                detail=(
                    "Reading a union member other than the last one written "
                    "is implementation-defined in C11. GCC and Clang define "
                    "it as type-punning (reinterpret bytes)."
                ),
            ),
            rust_semantics=RustSemantics(
                summary="Rust unions require unsafe access; reading is always raw bytes",
                guaranteed=True,
                detail=(
                    "Accessing any field of a Rust union is unsafe. "
                    "The semantics are type-punning of the raw bytes. "
                    "No implicit conversion occurs."
                ),
            ),
            divergence_type=DivergenceType.Moderate,
            smt_encoder=lambda val, _b="", width=32: (
                SMTEncoding.union_reinterpret(val, width, width)
            ),
            fuzzing_strategy=FuzzingSeedStrategy(
                description="Values that differ under type-punning",
                boundary_values=(0, 1, -1, 0x7F800000, 0xFF800000),
                generator_hint="union_pun",
                priority=6,
            ),
            applicable_opcodes=frozenset({"bitcast", "load", "store"}),
            description="Union type-punning semantics differ subtly.",
            mitigation="Use transmute or explicit byte-level access.",
            test_priority=6,
        ))

    # ── 20. EnumDiscriminant ─────────────────────────────────────────────

    def _add_enum_discriminant(self) -> None:
        self._reg(DivergenceEntry(
            cls=DivergenceClass.EnumDiscriminant,
            c_semantics=CSemantics(
                summary="C enum values are plain integers with no variant data",
                is_impl_defined=True,
                standard_ref="C11 §6.7.2.2",
                detail=(
                    "C enums are integer constants. There is no tagged union "
                    "concept; values outside declared enumerators are valid."
                ),
            ),
            rust_semantics=RustSemantics(
                summary="Rust enums are tagged unions with discriminant + payload",
                guaranteed=True,
                detail=(
                    "Rust enums can carry data per variant. The discriminant "
                    "layout depends on #[repr]. Creating an enum value with "
                    "an invalid discriminant is UB."
                ),
            ),
            divergence_type=DivergenceType.Critical,
            smt_encoder=lambda c_disc, rust_disc, width=32: (
                SMTEncoding.enum_discriminant_mismatch(c_disc, rust_disc)
            ),
            fuzzing_strategy=FuzzingSeedStrategy(
                description="Discriminant values at and beyond valid range",
                boundary_values=(0, 1, 255, 256, -1, -128),
                generator_hint="enum_discriminant",
                priority=7,
            ),
            applicable_opcodes=frozenset({"switch", "extractvalue"}),
            description=(
                "C enums are plain integers; Rust enums are tagged unions "
                "where invalid discriminants are UB."
            ),
            mitigation="Use #[repr(C)] and validate discriminant values at FFI boundary.",
            test_priority=7,
        ))

    # ── 21. StringEncoding ───────────────────────────────────────────────

    def _add_string_encoding(self) -> None:
        self._reg(DivergenceEntry(
            cls=DivergenceClass.StringEncoding,
            c_semantics=CSemantics(
                summary="C strings are null-terminated byte arrays (no encoding guarantee)",
                is_ub=False,
                standard_ref="C11 §7.1.1",
                detail=(
                    "C strings are char arrays terminated by '\\0'. "
                    "The encoding is locale-dependent and not guaranteed "
                    "to be UTF-8."
                ),
            ),
            rust_semantics=RustSemantics(
                summary="Rust &str is guaranteed UTF-8; CStr for C interop",
                guaranteed=True,
                detail=(
                    "Rust &str and String are always valid UTF-8. "
                    "For C interop, CStr/CString handle null-terminated "
                    "byte strings without encoding guarantees."
                ),
            ),
            divergence_type=DivergenceType.Moderate,
            smt_encoder=lambda c_len, rust_len, width=32: (
                SMTEncoding.string_encoding_mismatch(c_len, rust_len)
            ),
            fuzzing_strategy=FuzzingSeedStrategy(
                description="Strings with non-ASCII bytes and embedded nulls",
                boundary_values=(0, 0x80, 0xFF),
                generator_hint="string_encoding",
                priority=5,
            ),
            applicable_opcodes=frozenset({
                "call", "gep", "getelementptr", "getelementptr.inbounds",
                "load", "store",
            }),
            description="C strings have no encoding; Rust strings must be UTF-8.",
            mitigation="Use CStr/CString for C string interop.",
            test_priority=5,
        ))

    # ── 22. MallocFree ───────────────────────────────────────────────────

    def _add_malloc_free(self) -> None:
        self._reg(DivergenceEntry(
            cls=DivergenceClass.MallocFree,
            c_semantics=CSemantics(
                summary="Manual malloc/free with no double-free or leak protection",
                is_ub=True,
                standard_ref="C11 §7.22.3",
                detail=(
                    "C provides malloc/calloc/realloc/free. Double-free "
                    "and use-after-free are UB. Memory leaks are allowed "
                    "but undesirable."
                ),
            ),
            rust_semantics=RustSemantics(
                summary="Ownership system prevents double-free and use-after-free",
                guaranteed=True,
                detail=(
                    "Rust's ownership and Drop trait ensure single-owner "
                    "deallocation. Box, Vec, etc. deallocate automatically. "
                    "Double-free is prevented at compile time."
                ),
            ),
            divergence_type=DivergenceType.Critical,
            smt_encoder=lambda ptr, freed_set, width=64: (
                SMTEncoding.double_free(ptr, freed_set)
            ),
            fuzzing_strategy=FuzzingSeedStrategy(
                description="Allocation/deallocation sequences",
                boundary_values=(0,),
                generator_hint="malloc_free",
                priority=9,
            ),
            applicable_opcodes=frozenset({"call"}),
            description=(
                "C manual memory management vs Rust ownership-based "
                "automatic deallocation."
            ),
            mitigation="Map malloc/free to Box::new/drop or Vec allocation.",
            test_priority=9,
        ))

    # ── 23. FunctionPointer ──────────────────────────────────────────────

    def _add_function_pointer(self) -> None:
        self._reg(DivergenceEntry(
            cls=DivergenceClass.FunctionPointer,
            c_semantics=CSemantics(
                summary="Function pointers are untyped at runtime; wrong type is UB",
                is_ub=True,
                standard_ref="C11 §6.5.2.2/6",
                detail=(
                    "Calling a function through a pointer of incompatible "
                    "type is UB. No runtime check is performed."
                ),
            ),
            rust_semantics=RustSemantics(
                summary="Fn traits provide type-safe function references; fn ptrs are unsafe",
                guaranteed=True,
                detail=(
                    "Rust closures implement Fn/FnMut/FnOnce traits with "
                    "compile-time type safety. Raw fn pointers exist but "
                    "calling them with wrong signature is UB."
                ),
            ),
            divergence_type=DivergenceType.Critical,
            smt_encoder=lambda fptr, _b="", width=64: (
                SMTEncoding.function_pointer_null(fptr)
            ),
            fuzzing_strategy=FuzzingSeedStrategy(
                description="Null and invalid function pointer values",
                boundary_values=(0, 1),
                generator_hint="function_pointer",
                priority=7,
            ),
            applicable_opcodes=frozenset({"call"}),
            description="C function pointers are untyped; Rust provides type-safe closures.",
            mitigation="Use extern \"C\" fn types and validate pointers at FFI boundary.",
            test_priority=7,
        ))

    # ── 24. StackAlloc ───────────────────────────────────────────────────

    def _add_stack_alloc(self) -> None:
        self._reg(DivergenceEntry(
            cls=DivergenceClass.StackAlloc,
            c_semantics=CSemantics(
                summary="Returning pointer to stack-local is UB (dangling pointer)",
                is_ub=True,
                standard_ref="C11 §6.2.4/2",
                detail=(
                    "The lifetime of automatic storage duration objects "
                    "ends when the block exits. Returning their address "
                    "creates a dangling pointer (UB to dereference)."
                ),
            ),
            rust_semantics=RustSemantics(
                summary="Borrow checker prevents returning references to locals",
                guaranteed=True,
                detail=(
                    "Rust's lifetime system prevents returning references "
                    "to stack-local variables at compile time."
                ),
            ),
            divergence_type=DivergenceType.Critical,
            smt_encoder=lambda ptr, frame_base, width=64: (
                SMTEncoding.stack_lifetime_escape(ptr, frame_base, str(width))
            ),
            fuzzing_strategy=FuzzingSeedStrategy(
                description="Addresses in stack frame range",
                boundary_values=(),
                generator_hint="stack_escape",
                priority=8,
            ),
            applicable_opcodes=frozenset({"alloca", "ret"}),
            description="C allows dangling stack pointers; Rust prevents at compile time.",
            mitigation="Allocate on heap (Box) or use output parameters.",
            test_priority=8,
        ))

    # ── 25. LifetimeDangle ───────────────────────────────────────────────

    def _add_lifetime_dangle(self) -> None:
        self._reg(DivergenceEntry(
            cls=DivergenceClass.LifetimeDangle,
            c_semantics=CSemantics(
                summary="Use-after-free is UB; no compiler enforcement",
                is_ub=True,
                standard_ref="C11 §6.2.4",
                detail=(
                    "Accessing memory after it has been freed is UB. "
                    "C compilers do not detect use-after-free statically "
                    "(requires dynamic tools like ASan)."
                ),
            ),
            rust_semantics=RustSemantics(
                summary="Lifetime annotations prevent use-after-free at compile time",
                guaranteed=True,
                detail=(
                    "Rust's borrow checker ensures references cannot outlive "
                    "their referent. Use-after-free is impossible in safe Rust."
                ),
            ),
            divergence_type=DivergenceType.Critical,
            smt_encoder=lambda ptr, live_set, width=64: (
                SMTEncoding.dangling_pointer(ptr, live_set)
            ),
            fuzzing_strategy=FuzzingSeedStrategy(
                description="Access patterns after deallocation",
                boundary_values=(),
                generator_hint="use_after_free",
                priority=9,
            ),
            applicable_opcodes=frozenset({"load", "store", "call"}),
            description="C allows use-after-free (UB); Rust prevents statically.",
            mitigation="Ensure all pointers are valid before use; use RAII patterns.",
            test_priority=9,
        ))

    # ── 26. SliceVsRawPtr ────────────────────────────────────────────────

    def _add_slice_vs_raw_ptr(self) -> None:
        self._reg(DivergenceEntry(
            cls=DivergenceClass.SliceVsRawPtr,
            c_semantics=CSemantics(
                summary="C uses pointer+length convention (unchecked)",
                is_ub=True,
                standard_ref="C11 §6.5.6",
                detail=(
                    "C represents arrays as raw pointers with a separate "
                    "length parameter (or sentinel). No bounds checking "
                    "is performed."
                ),
            ),
            rust_semantics=RustSemantics(
                summary="Rust slices carry length and bounds-check on access",
                guaranteed=True,
                detail=(
                    "Rust slices (&[T]) are fat pointers carrying both "
                    "pointer and length. Indexing performs bounds checks "
                    "(panics on OOB)."
                ),
            ),
            divergence_type=DivergenceType.Critical,
            smt_encoder=lambda ptr, len_var, width=64: (
                SMTEncoding.slice_bounds_check(ptr, len_var, "idx")
            ),
            fuzzing_strategy=FuzzingSeedStrategy(
                description="Buffer lengths at boundaries",
                boundary_values=(0, 1, -1),
                generator_hint="slice_bounds",
                priority=8,
            ),
            applicable_opcodes=frozenset({
                "gep", "getelementptr", "getelementptr.inbounds",
                "load", "store",
            }),
            description=(
                "C pointer+length is unchecked; Rust slices carry "
                "length and perform bounds checks."
            ),
            mitigation="Convert C pointer+length pairs to Rust slices at FFI boundary.",
            test_priority=8,
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

        if cls is DivergenceClass.PointerCast:
            if isinstance(inst, CastInst):
                return isinstance(inst.src_type, PointerType) or isinstance(inst.dest_type, PointerType)
            return False

        if cls is DivergenceClass.PointerProvenance:
            if isinstance(inst, CastInst):
                return (isinstance(inst.src_type, PointerType) and isinstance(inst.dest_type, IntType)) or \
                       (isinstance(inst.src_type, IntType) and isinstance(inst.dest_type, PointerType))
            if isinstance(inst, GetElementPtrInst):
                return True
            return False

        if cls is DivergenceClass.StructLayout:
            if isinstance(inst, GetElementPtrInst):
                return isinstance(inst.base_type, StructType)
            if isinstance(inst, (ExtractValueInst, InsertValueInst)):
                return True
            return False

        if cls is DivergenceClass.MallocFree:
            if isinstance(inst, CallInst):
                callee = getattr(inst, 'callee_name', '')
                return callee in ('malloc', 'calloc', 'realloc', 'free',
                                  'aligned_alloc', 'posix_memalign')
            return False

        if cls is DivergenceClass.FunctionPointer:
            if isinstance(inst, CallInst):
                # Indirect call: callee_name is empty or callee is a pointer
                callee = getattr(inst, 'callee_name', '')
                if not callee:
                    return True
                callee_val = inst.callee
                if isinstance(callee_val.type, PointerType):
                    pointee = getattr(callee_val.type, 'pointee', None)
                    if isinstance(pointee, FunctionType):
                        return True
            return False

        if cls is DivergenceClass.StringEncoding:
            if isinstance(inst, CallInst):
                callee = getattr(inst, 'callee_name', '')
                return callee in (
                    'strlen', 'strcmp', 'strncmp', 'strcpy', 'strncpy',
                    'strcat', 'strncat', 'memcpy', 'memmove', 'memset',
                    'memcmp', 'strdup', 'strndup',
                )
            return True  # Allow opcode-only matching for gep/load/store

        if cls is DivergenceClass.EnumDiscriminant:
            if isinstance(inst, ExtractValueInst):
                agg_type = inst.aggregate.type
                if isinstance(agg_type, StructType):
                    return True
            if isinstance(inst, (BranchInst, SelectInst)):
                return True
            return True  # Allow opcode-only matching for switch

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
            if isinstance(inst.src_type, PointerType) or isinstance(inst.dest_type, PointerType):
                report.notes.append(
                    "Pointer cast detected; check alignment and aliasing rules"
                )

        if isinstance(inst, (LoadInst, StoreInst)):
            report.notes.append(
                "Check pointer validity and alignment before memory access"
            )

        if isinstance(inst, GetElementPtrInst):
            if isinstance(inst.source_element_type, StructType):
                report.notes.append(
                    "Struct field access via GEP; verify #[repr(C)] layout compatibility"
                )

        if isinstance(inst, CallInst):
            callee = getattr(inst, 'callee_name', '')
            if callee in ('malloc', 'calloc', 'realloc', 'free',
                          'aligned_alloc', 'posix_memalign'):
                report.notes.append(
                    f"Memory allocation call '{callee}'; "
                    f"map to Rust ownership patterns (Box, Vec, etc.)"
                )
            if callee in ('strlen', 'strcmp', 'strncmp', 'strcpy', 'strncpy',
                          'strcat', 'strncat', 'memcpy', 'memmove', 'memset',
                          'memcmp', 'strdup', 'strndup'):
                report.notes.append(
                    f"String/memory operation '{callee}'; "
                    f"Rust &str requires valid UTF-8 unlike C char*"
                )
            if not callee:
                report.notes.append(
                    "Indirect function call; verify function pointer type safety"
                )


# ── Convenience ──────────────────────────────────────────────────────────

def get_default_table() -> DivergenceTable:
    """Return a freshly-constructed default divergence table."""
    return DivergenceTable()


def quick_analyze(inst: Instruction) -> DivergenceReport:
    """One-shot analysis of a single instruction."""
    return DivergenceAnalyzer().analyze_instruction(inst)
