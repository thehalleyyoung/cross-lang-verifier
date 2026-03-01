"""
Type coercion generation for the Cross-Language Equivalence Verifier.

Given a source type and target type, generates the sequence of IR cast
instructions needed to convert between them.  Handles integer widening /
narrowing, sign changes, float-int conversions, and pointer casts.
Tracks semantic information loss at each step.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, List, Optional, Sequence, Tuple

from ..ir.types import (
    IRType,
    IntType,
    FloatType,
    PointerType,
    ArrayType,
    StructType,
    VoidType,
    Signedness,
    FloatKind,
)
from ..ir.instructions import CastKind


# ── Information loss tracking ────────────────────────────────────────────

class InformationLoss(Enum):
    """Kind of information potentially lost during a coercion."""
    NONE = auto()           # Perfectly lossless
    PRECISION = auto()      # Float precision loss
    MAGNITUDE = auto()      # Value may be truncated / clamped
    SIGN = auto()           # Negative values lost (signed → unsigned)
    BITS = auto()           # High bits discarded (truncation)
    NAN_INF = auto()        # NaN / Inf special handling
    PROVENANCE = auto()     # Pointer provenance lost (ptrtoint)
    REPRESENTATION = auto() # Complete reinterpretation (bitcast)

    def __str__(self) -> str:
        return self.name

    @property
    def is_safe(self) -> bool:
        return self is InformationLoss.NONE


# ── Coercion step ────────────────────────────────────────────────────────

@dataclass(frozen=True)
class CoercionStep:
    """A single IR cast instruction in a coercion chain."""
    cast_kind: CastKind
    src_type: IRType
    dst_type: IRType
    information_loss: InformationLoss = InformationLoss.NONE
    description: str = ""
    is_implicit_in_c: bool = False
    is_implicit_in_rust: bool = False

    @property
    def is_lossless(self) -> bool:
        return self.information_loss is InformationLoss.NONE

    @property
    def is_widening(self) -> bool:
        if isinstance(self.src_type, IntType) and isinstance(self.dst_type, IntType):
            return self.dst_type.width > self.src_type.width
        if isinstance(self.src_type, FloatType) and isinstance(self.dst_type, FloatType):
            return self.dst_type.width > self.src_type.width
        return False

    @property
    def is_narrowing(self) -> bool:
        if isinstance(self.src_type, IntType) and isinstance(self.dst_type, IntType):
            return self.dst_type.width < self.src_type.width
        if isinstance(self.src_type, FloatType) and isinstance(self.dst_type, FloatType):
            return self.dst_type.width < self.src_type.width
        return False

    def __str__(self) -> str:
        loss_str = f" [{self.information_loss}]" if not self.is_lossless else ""
        return f"{self.cast_kind.value}: {self.src_type} → {self.dst_type}{loss_str}"


# ── Coercion chain ──────────────────────────────────────────────────────

@dataclass
class CoercionChain:
    """A sequence of coercion steps from source type to target type."""
    src_type: IRType
    dst_type: IRType
    steps: list[CoercionStep] = field(default_factory=list)
    is_valid: bool = True
    error_message: str = ""

    @property
    def is_empty(self) -> bool:
        return len(self.steps) == 0

    @property
    def is_identity(self) -> bool:
        return self.src_type == self.dst_type

    @property
    def is_lossless(self) -> bool:
        return all(s.is_lossless for s in self.steps)

    @property
    def num_steps(self) -> int:
        return len(self.steps)

    @property
    def total_information_loss(self) -> set[InformationLoss]:
        return {s.information_loss for s in self.steps if not s.is_lossless}

    @property
    def has_implicit_in_c(self) -> bool:
        return any(s.is_implicit_in_c for s in self.steps)

    @property
    def has_implicit_in_rust(self) -> bool:
        return any(s.is_implicit_in_rust for s in self.steps)

    @property
    def cast_kinds(self) -> list[CastKind]:
        return [s.cast_kind for s in self.steps]

    def append(self, step: CoercionStep) -> None:
        self.steps.append(step)

    def summary(self) -> str:
        if self.is_identity:
            return f"{self.src_type} (identity, no coercion needed)"
        if not self.is_valid:
            return f"INVALID: {self.error_message}"
        lines = [f"Coercion: {self.src_type} → {self.dst_type} ({self.num_steps} steps):"]
        for i, step in enumerate(self.steps):
            lines.append(f"  {i + 1}. {step}")
        if not self.is_lossless:
            losses = ", ".join(str(l) for l in self.total_information_loss)
            lines.append(f"  ⚠ Information loss: {losses}")
        return "\n".join(lines)

    def __str__(self) -> str:
        if self.is_identity:
            return "(identity)"
        return " → ".join(
            [str(self.src_type)]
            + [f"[{s.cast_kind.value}] {s.dst_type}" for s in self.steps]
        )


# ── Coercion generator ──────────────────────────────────────────────────

class CoercionGenerator:
    """Generates IR cast instruction sequences for type conversions.

    Given a source type and target type, determines the minimal sequence
    of CastKind operations to transform between them.
    """

    def __init__(self, pointer_size: int = 64) -> None:
        self.pointer_size = pointer_size

    def generate(self, src: IRType, dst: IRType) -> CoercionChain:
        """Generate the coercion chain from src to dst."""
        chain = CoercionChain(src_type=src, dst_type=dst)

        if src == dst:
            return chain  # Identity

        # Dispatch based on type combination
        if isinstance(src, IntType) and isinstance(dst, IntType):
            self._int_to_int(src, dst, chain)
        elif isinstance(src, FloatType) and isinstance(dst, FloatType):
            self._float_to_float(src, dst, chain)
        elif isinstance(src, IntType) and isinstance(dst, FloatType):
            self._int_to_float(src, dst, chain)
        elif isinstance(src, FloatType) and isinstance(dst, IntType):
            self._float_to_int(src, dst, chain)
        elif isinstance(src, PointerType) and isinstance(dst, IntType):
            self._ptr_to_int(src, dst, chain)
        elif isinstance(src, IntType) and isinstance(dst, PointerType):
            self._int_to_ptr(src, dst, chain)
        elif isinstance(src, PointerType) and isinstance(dst, PointerType):
            self._ptr_to_ptr(src, dst, chain)
        elif isinstance(src, IntType) and isinstance(dst, PointerType):
            self._int_to_ptr(src, dst, chain)
        elif isinstance(src, PointerType) and isinstance(dst, FloatType):
            # ptr → int → float (two-step)
            int_type = IntType(self.pointer_size, Signedness.UNSIGNED)
            self._ptr_to_int(src, int_type, chain)
            self._int_to_float(int_type, dst, chain)
        elif isinstance(src, FloatType) and isinstance(dst, PointerType):
            # float → int → ptr (two-step)
            int_type = IntType(self.pointer_size, Signedness.UNSIGNED)
            self._float_to_int(src, int_type, chain)
            self._int_to_ptr(int_type, dst, chain)
        else:
            # Unsupported conversion
            chain.is_valid = False
            chain.error_message = f"No known coercion from {src} to {dst}"

        return chain

    def can_coerce(self, src: IRType, dst: IRType) -> bool:
        """Return True if a coercion chain exists from src to dst."""
        chain = self.generate(src, dst)
        return chain.is_valid

    def coercion_cost(self, src: IRType, dst: IRType) -> int:
        """Return a cost metric for coercing src to dst.
        
        Lower = cheaper/safer. Used for overload resolution.
        0 = identity, 1 = widening, 2 = narrowing, 3+ = complex.
        """
        if src == dst:
            return 0
        chain = self.generate(src, dst)
        if not chain.is_valid:
            return 999

        cost = 0
        for step in chain.steps:
            if step.is_lossless:
                cost += 1
            elif step.information_loss is InformationLoss.PRECISION:
                cost += 2
            elif step.information_loss in (InformationLoss.MAGNITUDE,
                                           InformationLoss.SIGN,
                                           InformationLoss.BITS):
                cost += 3
            elif step.information_loss is InformationLoss.PROVENANCE:
                cost += 5
            else:
                cost += 4
        return cost

    def find_common_type(self, types: Sequence[IRType]) -> IRType | None:
        """Find the cheapest common type that all given types can coerce to."""
        if not types:
            return None
        if len(types) == 1:
            return types[0]

        # For integers: pick the widest, preserving signedness if possible
        if all(isinstance(t, IntType) for t in types):
            int_types = [t for t in types if isinstance(t, IntType)]
            max_width = max(t.width for t in int_types)
            has_signed = any(t.is_signed for t in int_types)
            has_unsigned = any(t.is_unsigned for t in int_types)
            if has_signed and has_unsigned:
                # If widths differ, use the wider signed; else unsigned
                if max_width > min(t.width for t in int_types):
                    return IntType(max_width, Signedness.SIGNED)
                return IntType(max_width, Signedness.UNSIGNED)
            sign = Signedness.SIGNED if has_signed else Signedness.UNSIGNED
            return IntType(max_width, sign)

        # For floats
        if all(isinstance(t, FloatType) for t in types):
            float_types = [t for t in types if isinstance(t, FloatType)]
            max_kind = max(float_types, key=lambda f: f.width)
            return max_kind

        # Mixed int/float: promote to float
        if all(isinstance(t, (IntType, FloatType)) for t in types):
            has_f64 = any(isinstance(t, FloatType) and t.kind is FloatKind.F64 for t in types)
            has_wide_int = any(isinstance(t, IntType) and t.width > 24 for t in types)
            if has_f64 or has_wide_int:
                return FloatType(FloatKind.F64)
            return FloatType(FloatKind.F32)

        return None

    # ── Integer ↔ Integer ────────────────────────────────────────────────

    def _int_to_int(
        self, src: IntType, dst: IntType, chain: CoercionChain,
    ) -> None:
        if src.width == dst.width and src.signedness != dst.signedness:
            # Same width, different sign: bitcast (reinterpret)
            chain.append(CoercionStep(
                cast_kind=CastKind.BITCAST,
                src_type=src,
                dst_type=dst,
                information_loss=(
                    InformationLoss.SIGN
                    if src.is_signed and dst.is_unsigned
                    else InformationLoss.NONE
                ),
                description="Sign change at same width",
                is_implicit_in_c=True,
                is_implicit_in_rust=False,
            ))
        elif src.width < dst.width:
            # Widening
            if src.is_signed:
                cast_kind = CastKind.SEXT
            else:
                cast_kind = CastKind.ZEXT

            # If also changing sign, may need two steps
            if src.signedness != dst.signedness and src.is_unsigned and dst.is_signed:
                # u8 → i32: zext then reinterpret (but zext to signed is fine)
                chain.append(CoercionStep(
                    cast_kind=CastKind.ZEXT,
                    src_type=src,
                    dst_type=dst,
                    information_loss=InformationLoss.NONE,
                    description="Unsigned widening to signed (all values fit)",
                    is_implicit_in_c=True,
                    is_implicit_in_rust=False,
                ))
            elif src.signedness != dst.signedness and src.is_signed and dst.is_unsigned:
                # i8 → u32: sext to same width as unsigned, then bitcast
                intermediate = IntType(dst.width, Signedness.SIGNED)
                chain.append(CoercionStep(
                    cast_kind=CastKind.SEXT,
                    src_type=src,
                    dst_type=intermediate,
                    information_loss=InformationLoss.NONE,
                    description="Sign-extend before unsigned conversion",
                    is_implicit_in_c=True,
                ))
                if intermediate != dst:
                    chain.append(CoercionStep(
                        cast_kind=CastKind.BITCAST,
                        src_type=intermediate,
                        dst_type=dst,
                        information_loss=InformationLoss.SIGN,
                        description="Reinterpret as unsigned",
                        is_implicit_in_c=True,
                    ))
            else:
                chain.append(CoercionStep(
                    cast_kind=cast_kind,
                    src_type=src,
                    dst_type=dst,
                    information_loss=InformationLoss.NONE,
                    description="Integer widening",
                    is_implicit_in_c=True,
                    is_implicit_in_rust=False,
                ))
        else:
            # Narrowing
            loss = InformationLoss.BITS
            if src.is_signed and dst.is_unsigned:
                loss = InformationLoss.SIGN  # Also sign loss
            chain.append(CoercionStep(
                cast_kind=CastKind.TRUNC,
                src_type=src,
                dst_type=dst,
                information_loss=loss,
                description="Integer narrowing (truncation)",
                is_implicit_in_c=True,
                is_implicit_in_rust=False,
            ))

    # ── Float ↔ Float ────────────────────────────────────────────────────

    def _float_to_float(
        self, src: FloatType, dst: FloatType, chain: CoercionChain,
    ) -> None:
        if src.width < dst.width:
            chain.append(CoercionStep(
                cast_kind=CastKind.FPEXT,
                src_type=src,
                dst_type=dst,
                information_loss=InformationLoss.NONE,
                description="Float widening (f32 → f64)",
                is_implicit_in_c=True,
                is_implicit_in_rust=False,
            ))
        else:
            chain.append(CoercionStep(
                cast_kind=CastKind.FPTRUNC,
                src_type=src,
                dst_type=dst,
                information_loss=InformationLoss.PRECISION,
                description="Float narrowing (f64 → f32)",
                is_implicit_in_c=True,
                is_implicit_in_rust=False,
            ))

    # ── Integer → Float ──────────────────────────────────────────────────

    def _int_to_float(
        self, src: IntType, dst: FloatType, chain: CoercionChain,
    ) -> None:
        cast_kind = CastKind.SITOFP if src.is_signed else CastKind.UITOFP

        # Determine if precision might be lost
        if dst.kind is FloatKind.F32:
            mantissa_bits = 24
        else:
            mantissa_bits = 53

        loss = InformationLoss.NONE
        if src.width > mantissa_bits:
            loss = InformationLoss.PRECISION

        chain.append(CoercionStep(
            cast_kind=cast_kind,
            src_type=src,
            dst_type=dst,
            information_loss=loss,
            description="Integer to float conversion",
            is_implicit_in_c=True,
            is_implicit_in_rust=False,
        ))

    # ── Float → Integer ──────────────────────────────────────────────────

    def _float_to_int(
        self, src: FloatType, dst: IntType, chain: CoercionChain,
    ) -> None:
        cast_kind = CastKind.FPTOSI if dst.is_signed else CastKind.FPTOUI

        chain.append(CoercionStep(
            cast_kind=cast_kind,
            src_type=src,
            dst_type=dst,
            information_loss=InformationLoss.MAGNITUDE,
            description="Float to integer conversion (truncates toward zero)",
            is_implicit_in_c=True,
            is_implicit_in_rust=False,
        ))

    # ── Pointer ↔ Integer ────────────────────────────────────────────────

    def _ptr_to_int(
        self, src: PointerType, dst: IntType, chain: CoercionChain,
    ) -> None:
        # First: ptrtoint to pointer-sized integer
        ptr_int_type = IntType(self.pointer_size, Signedness.UNSIGNED)
        chain.append(CoercionStep(
            cast_kind=CastKind.PTRTOINT,
            src_type=src,
            dst_type=ptr_int_type,
            information_loss=InformationLoss.PROVENANCE,
            description="Pointer to integer (loses provenance)",
            is_implicit_in_c=True,
            is_implicit_in_rust=False,
        ))

        # Then: resize integer if needed
        if ptr_int_type != dst:
            self._int_to_int(ptr_int_type, dst, chain)

    def _int_to_ptr(
        self, src: IntType, dst: PointerType, chain: CoercionChain,
    ) -> None:
        # Resize to pointer-width first if needed
        ptr_int_type = IntType(self.pointer_size, Signedness.UNSIGNED)
        if src != ptr_int_type:
            self._int_to_int(src, ptr_int_type, chain)

        chain.append(CoercionStep(
            cast_kind=CastKind.INTTOPTR,
            src_type=ptr_int_type,
            dst_type=dst,
            information_loss=InformationLoss.PROVENANCE,
            description="Integer to pointer (no provenance)",
            is_implicit_in_c=True,
            is_implicit_in_rust=False,
        ))

    # ── Pointer ↔ Pointer ────────────────────────────────────────────────

    def _ptr_to_ptr(
        self, src: PointerType, dst: PointerType, chain: CoercionChain,
    ) -> None:
        chain.append(CoercionStep(
            cast_kind=CastKind.BITCAST,
            src_type=src,
            dst_type=dst,
            information_loss=InformationLoss.NONE,
            description="Pointer type change (same representation)",
            is_implicit_in_c=True,
            is_implicit_in_rust=False,
        ))


# ── Convenience ──────────────────────────────────────────────────────────

def generate_coercion(src: IRType, dst: IRType) -> CoercionChain:
    """Generate a coercion chain from src to dst using default settings."""
    return CoercionGenerator().generate(src, dst)


def coercion_is_safe(src: IRType, dst: IRType) -> bool:
    """Return True if coercing src to dst is guaranteed lossless."""
    return CoercionGenerator().generate(src, dst).is_lossless


def coercion_cost(src: IRType, dst: IRType) -> int:
    """Return the cost of coercing src to dst."""
    return CoercionGenerator().coercion_cost(src, dst)
