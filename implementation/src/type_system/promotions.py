"""
Integer promotion and conversion rules for the Cross-Language Equivalence Verifier.

Implements C99/C11 integer promotion ranks, usual arithmetic conversions,
implicit narrowing detection, and Rust (explicit-only) promotion rules.
Provides cross-language promotion comparison and promotion chain
visualisation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

from ..ir.types import (
    IRType,
    IntType,
    FloatType,
    PointerType,
    VoidType,
    Signedness,
    FloatKind,
    Language,
)


# ── Integer conversion rank (C11 §6.3.1.1) ──────────────────────────────

# The rank determines which type "wins" when two integer types are combined.
# Higher rank = wider or more preferred type.

_C_RANK_TABLE: dict[tuple[int, Signedness], int] = {
    # (width, signedness) -> rank
    # _Bool
    (1, Signedness.UNSIGNED): 1,
    # char / signed char / unsigned char
    (8, Signedness.SIGNED): 2,
    (8, Signedness.UNSIGNED): 2,
    # short / unsigned short
    (16, Signedness.SIGNED): 3,
    (16, Signedness.UNSIGNED): 3,
    # int / unsigned int
    (32, Signedness.SIGNED): 4,
    (32, Signedness.UNSIGNED): 4,
    # long / unsigned long (assuming LP64)
    (64, Signedness.SIGNED): 5,
    (64, Signedness.UNSIGNED): 5,
    # long long / unsigned long long
    (128, Signedness.SIGNED): 6,
    (128, Signedness.UNSIGNED): 6,
}

# C standard type names for display
_C_TYPE_NAMES: dict[tuple[int, Signedness], str] = {
    (1, Signedness.UNSIGNED): "_Bool",
    (8, Signedness.SIGNED): "signed char",
    (8, Signedness.UNSIGNED): "unsigned char",
    (16, Signedness.SIGNED): "short",
    (16, Signedness.UNSIGNED): "unsigned short",
    (32, Signedness.SIGNED): "int",
    (32, Signedness.UNSIGNED): "unsigned int",
    (64, Signedness.SIGNED): "long",
    (64, Signedness.UNSIGNED): "unsigned long",
    (128, Signedness.SIGNED): "long long",
    (128, Signedness.UNSIGNED): "unsigned long long",
}


def c_integer_rank(ty: IntType) -> int:
    """Return the C integer conversion rank for an IntType."""
    key = (ty.width, ty.signedness)
    if key in _C_RANK_TABLE:
        return _C_RANK_TABLE[key]
    # For non-standard widths, rank by width
    return ty.width


def c_type_name(ty: IntType) -> str:
    """Return the C standard name for an IntType."""
    key = (ty.width, ty.signedness)
    return _C_TYPE_NAMES.get(key, f"{'i' if ty.is_signed else 'u'}{ty.width}")


# ── Promotion step ───────────────────────────────────────────────────────

class PromotionKind(Enum):
    """Kind of integer promotion or conversion."""
    INTEGER_PROMOTION = auto()    # C §6.3.1.1: small type -> int
    USUAL_ARITHMETIC = auto()     # C §6.3.1.8: balance two operands
    IMPLICIT_NARROWING = auto()   # Assignment narrowing (warning)
    EXPLICIT_CAST = auto()        # Explicit cast by programmer
    SIGN_CONVERSION = auto()      # Change of signedness
    VARIADIC_PROMOTION = auto()   # Default argument promotion for variadic calls
    RUST_EXPLICIT = auto()        # Rust explicit `as` cast


@dataclass(frozen=True)
class PromotionStep:
    """A single step in a promotion chain."""
    src_type: IntType
    dst_type: IntType
    kind: PromotionKind
    reason: str = ""
    is_safe: bool = True  # True if no information can be lost

    @property
    def widens(self) -> bool:
        return self.dst_type.width > self.src_type.width

    @property
    def narrows(self) -> bool:
        return self.dst_type.width < self.src_type.width

    @property
    def changes_sign(self) -> bool:
        return self.src_type.signedness != self.dst_type.signedness

    def __str__(self) -> str:
        direction = "↑" if self.widens else ("↓" if self.narrows else "=")
        return (
            f"{c_type_name(self.src_type)} {direction} "
            f"{c_type_name(self.dst_type)} [{self.kind.name}]"
        )


# ── Promotion chain ─────────────────────────────────────────────────────

@dataclass
class PromotionChain:
    """A sequence of promotion steps from an original type to a final type."""
    steps: list[PromotionStep] = field(default_factory=list)
    original_type: IntType | None = None
    final_type: IntType | None = None

    @property
    def is_empty(self) -> bool:
        return len(self.steps) == 0

    @property
    def is_identity(self) -> bool:
        return self.original_type == self.final_type

    @property
    def is_safe(self) -> bool:
        return all(s.is_safe for s in self.steps)

    @property
    def has_narrowing(self) -> bool:
        return any(s.narrows for s in self.steps)

    @property
    def has_sign_change(self) -> bool:
        return any(s.changes_sign for s in self.steps)

    @property
    def total_width_change(self) -> int:
        if self.original_type and self.final_type:
            return self.final_type.width - self.original_type.width
        return 0

    def append(self, step: PromotionStep) -> None:
        self.steps.append(step)
        if not self.original_type:
            self.original_type = step.src_type
        self.final_type = step.dst_type

    def visualize(self) -> str:
        """Produce a human-readable visualization of the chain."""
        if not self.steps:
            return "(no promotions)"
        lines = [f"Promotion chain ({len(self.steps)} steps):"]
        for i, step in enumerate(self.steps):
            safe_marker = "✓" if step.is_safe else "⚠"
            lines.append(f"  {i + 1}. {step} {safe_marker}")
        if self.has_narrowing:
            lines.append("  ⚠ WARNING: chain contains narrowing conversion")
        if self.has_sign_change:
            lines.append("  ⚠ WARNING: chain contains sign change")
        return "\n".join(lines)

    def __str__(self) -> str:
        if not self.steps:
            return "(identity)"
        types_str = " → ".join(
            [c_type_name(self.steps[0].src_type)]
            + [c_type_name(s.dst_type) for s in self.steps]
        )
        return types_str


# ── Abstract promotion rules base ────────────────────────────────────────

class PromotionRules:
    """Base class for integer promotion rules."""

    def promote(self, ty: IntType) -> IntType:
        """Apply integer promotion to a single type."""
        raise NotImplementedError

    def usual_arithmetic_conversion(
        self, lhs: IntType, rhs: IntType,
    ) -> Tuple[IntType, PromotionChain, PromotionChain]:
        """Apply usual arithmetic conversions to balance two operands.

        Returns (common_type, lhs_chain, rhs_chain).
        """
        raise NotImplementedError

    def compute_chain(self, src: IntType, dst: IntType) -> PromotionChain:
        """Compute the full promotion chain from src to dst."""
        raise NotImplementedError

    def is_implicit_conversion_allowed(self, src: IntType, dst: IntType) -> bool:
        """Return True if implicit conversion from src to dst is allowed."""
        raise NotImplementedError

    def detect_narrowing(self, src: IntType, dst: IntType) -> bool:
        """Return True if converting src to dst narrows the value."""
        if dst.width < src.width:
            return True
        if dst.width == src.width and src.is_signed and dst.is_unsigned:
            return True  # Sign change can lose negative values
        return False


# ── C promotion rules ────────────────────────────────────────────────────

class CPromotionRules(PromotionRules):
    """C99/C11 integer promotion and usual arithmetic conversion rules."""

    def __init__(self, int_width: int = 32) -> None:
        self.int_width = int_width  # Width of `int` on this platform

    def promote(self, ty: IntType) -> IntType:
        """C11 §6.3.1.1: Integer promotions.

        If an int can represent all values of the original type, the value
        is converted to an int; otherwise, it is converted to unsigned int.
        """
        if ty.width >= self.int_width:
            return ty  # No promotion needed

        int_type = IntType(self.int_width, Signedness.SIGNED)
        # If all values of ty fit in signed int
        if ty.max_value <= int_type.max_value and ty.min_value >= int_type.min_value:
            return int_type
        # Otherwise promote to unsigned int
        return IntType(self.int_width, Signedness.UNSIGNED)

    def usual_arithmetic_conversion(
        self, lhs: IntType, rhs: IntType,
    ) -> Tuple[IntType, PromotionChain, PromotionChain]:
        """C11 §6.3.1.8: Usual arithmetic conversions for integer types.

        1. Apply integer promotions to both operands.
        2. If both have the same type, done.
        3. If both are signed or both unsigned, convert to the wider.
        4. If unsigned rank >= signed rank, convert signed to unsigned.
        5. If signed can represent all values of unsigned, convert to signed.
        6. Otherwise, convert both to the unsigned version of the signed type.
        """
        lhs_chain = PromotionChain()
        rhs_chain = PromotionChain()

        # Step 1: Integer promotions
        lhs_promoted = self.promote(lhs)
        rhs_promoted = self.promote(rhs)

        if lhs_promoted != lhs:
            lhs_chain.append(PromotionStep(
                src_type=lhs, dst_type=lhs_promoted,
                kind=PromotionKind.INTEGER_PROMOTION,
                reason="Integer promotion (C11 §6.3.1.1)",
            ))

        if rhs_promoted != rhs:
            rhs_chain.append(PromotionStep(
                src_type=rhs, dst_type=rhs_promoted,
                kind=PromotionKind.INTEGER_PROMOTION,
                reason="Integer promotion (C11 §6.3.1.1)",
            ))

        lhs_p, rhs_p = lhs_promoted, rhs_promoted

        # Step 2: Same type? done
        if lhs_p == rhs_p:
            lhs_chain.original_type = lhs_chain.original_type or lhs
            lhs_chain.final_type = lhs_p
            rhs_chain.original_type = rhs_chain.original_type or rhs
            rhs_chain.final_type = rhs_p
            return lhs_p, lhs_chain, rhs_chain

        # Step 3: Same signedness
        if lhs_p.signedness == rhs_p.signedness:
            if c_integer_rank(lhs_p) < c_integer_rank(rhs_p):
                common = rhs_p
                lhs_chain.append(PromotionStep(
                    src_type=lhs_p, dst_type=common,
                    kind=PromotionKind.USUAL_ARITHMETIC,
                    reason="Convert to wider same-signed type",
                ))
            else:
                common = lhs_p
                rhs_chain.append(PromotionStep(
                    src_type=rhs_p, dst_type=common,
                    kind=PromotionKind.USUAL_ARITHMETIC,
                    reason="Convert to wider same-signed type",
                ))
            self._finalize_chains(lhs, rhs, common, lhs_chain, rhs_chain)
            return common, lhs_chain, rhs_chain

        # Identify signed and unsigned sides
        if lhs_p.is_unsigned:
            unsigned, signed = lhs_p, rhs_p
            unsigned_is_lhs = True
        else:
            unsigned, signed = rhs_p, lhs_p
            unsigned_is_lhs = False

        # Step 4: unsigned rank >= signed rank
        if c_integer_rank(unsigned) >= c_integer_rank(signed):
            common = unsigned
            if unsigned_is_lhs:
                rhs_chain.append(PromotionStep(
                    src_type=signed, dst_type=common,
                    kind=PromotionKind.SIGN_CONVERSION,
                    reason="Signed → unsigned (unsigned rank >= signed rank)",
                    is_safe=False,
                ))
            else:
                lhs_chain.append(PromotionStep(
                    src_type=signed, dst_type=common,
                    kind=PromotionKind.SIGN_CONVERSION,
                    reason="Signed → unsigned (unsigned rank >= signed rank)",
                    is_safe=False,
                ))
            self._finalize_chains(lhs, rhs, common, lhs_chain, rhs_chain)
            return common, lhs_chain, rhs_chain

        # Step 5: signed type can represent all unsigned values
        if signed.max_value >= unsigned.max_value:
            common = signed
            if unsigned_is_lhs:
                lhs_chain.append(PromotionStep(
                    src_type=unsigned, dst_type=common,
                    kind=PromotionKind.USUAL_ARITHMETIC,
                    reason="Unsigned → signed (signed can represent all unsigned values)",
                ))
            else:
                rhs_chain.append(PromotionStep(
                    src_type=unsigned, dst_type=common,
                    kind=PromotionKind.USUAL_ARITHMETIC,
                    reason="Unsigned → signed (signed can represent all unsigned values)",
                ))
            self._finalize_chains(lhs, rhs, common, lhs_chain, rhs_chain)
            return common, lhs_chain, rhs_chain

        # Step 6: convert both to unsigned version of signed type
        common = IntType(signed.width, Signedness.UNSIGNED)
        if unsigned_is_lhs:
            if lhs_p != common:
                lhs_chain.append(PromotionStep(
                    src_type=lhs_p, dst_type=common,
                    kind=PromotionKind.USUAL_ARITHMETIC,
                    reason="Both → unsigned(signed type)",
                ))
            rhs_chain.append(PromotionStep(
                src_type=rhs_p, dst_type=common,
                kind=PromotionKind.SIGN_CONVERSION,
                reason="Both → unsigned(signed type)",
                is_safe=False,
            ))
        else:
            lhs_chain.append(PromotionStep(
                src_type=lhs_p, dst_type=common,
                kind=PromotionKind.SIGN_CONVERSION,
                reason="Both → unsigned(signed type)",
                is_safe=False,
            ))
            if rhs_p != common:
                rhs_chain.append(PromotionStep(
                    src_type=rhs_p, dst_type=common,
                    kind=PromotionKind.USUAL_ARITHMETIC,
                    reason="Both → unsigned(signed type)",
                ))
        self._finalize_chains(lhs, rhs, common, lhs_chain, rhs_chain)
        return common, lhs_chain, rhs_chain

    def compute_chain(self, src: IntType, dst: IntType) -> PromotionChain:
        chain = PromotionChain(original_type=src, final_type=dst)
        if src == dst:
            return chain

        # First try integer promotion
        promoted = self.promote(src)
        if promoted != src:
            chain.append(PromotionStep(
                src_type=src, dst_type=promoted,
                kind=PromotionKind.INTEGER_PROMOTION,
                reason="Integer promotion",
            ))
            src = promoted

        if src == dst:
            chain.final_type = dst
            return chain

        # Then widening or narrowing
        if src.width < dst.width:
            kind = PromotionKind.USUAL_ARITHMETIC
            is_safe = True
        elif src.width > dst.width:
            kind = PromotionKind.IMPLICIT_NARROWING
            is_safe = False
        else:
            kind = PromotionKind.SIGN_CONVERSION
            is_safe = False

        chain.append(PromotionStep(
            src_type=src, dst_type=dst, kind=kind, is_safe=is_safe,
        ))
        chain.final_type = dst
        return chain

    def is_implicit_conversion_allowed(self, src: IntType, dst: IntType) -> bool:
        return True  # C allows all implicit integer conversions

    def default_argument_promotion(self, ty: IntType) -> IntType:
        """C11 §6.5.2.2/6: Default argument promotions for variadic calls."""
        return self.promote(ty)

    def _finalize_chains(
        self, lhs: IntType, rhs: IntType, common: IntType,
        lhs_chain: PromotionChain, rhs_chain: PromotionChain,
    ) -> None:
        lhs_chain.original_type = lhs_chain.original_type or lhs
        lhs_chain.final_type = common
        rhs_chain.original_type = rhs_chain.original_type or rhs
        rhs_chain.final_type = common


# ── Rust promotion rules ────────────────────────────────────────────────

class RustPromotionRules(PromotionRules):
    """Rust integer type rules: no implicit promotions.

    All integer conversions in Rust must be explicit `as` casts.
    """

    def promote(self, ty: IntType) -> IntType:
        """Rust does not perform implicit promotion."""
        return ty

    def usual_arithmetic_conversion(
        self, lhs: IntType, rhs: IntType,
    ) -> Tuple[IntType, PromotionChain, PromotionChain]:
        """Rust requires both operands to be the same type already."""
        lhs_chain = PromotionChain(original_type=lhs, final_type=lhs)
        rhs_chain = PromotionChain(original_type=rhs, final_type=rhs)

        if lhs == rhs:
            return lhs, lhs_chain, rhs_chain

        # In Rust, this would be a compile error; report what casts are needed
        # but don't auto-promote
        if lhs.width > rhs.width:
            common = lhs
            rhs_chain.append(PromotionStep(
                src_type=rhs, dst_type=lhs,
                kind=PromotionKind.RUST_EXPLICIT,
                reason="Explicit `as` cast required in Rust",
                is_safe=True,
            ))
            rhs_chain.final_type = common
        elif rhs.width > lhs.width:
            common = rhs
            lhs_chain.append(PromotionStep(
                src_type=lhs, dst_type=rhs,
                kind=PromotionKind.RUST_EXPLICIT,
                reason="Explicit `as` cast required in Rust",
                is_safe=True,
            ))
            lhs_chain.final_type = common
        else:
            # Same width, different signedness — error in Rust
            common = lhs  # arbitrary choice
            rhs_chain.append(PromotionStep(
                src_type=rhs, dst_type=lhs,
                kind=PromotionKind.RUST_EXPLICIT,
                reason="Explicit `as` cast required for signedness change in Rust",
                is_safe=False,
            ))
            rhs_chain.final_type = common

        return common, lhs_chain, rhs_chain

    def compute_chain(self, src: IntType, dst: IntType) -> PromotionChain:
        chain = PromotionChain(original_type=src, final_type=dst)
        if src == dst:
            return chain
        chain.append(PromotionStep(
            src_type=src, dst_type=dst,
            kind=PromotionKind.RUST_EXPLICIT,
            reason="Explicit `as` cast required in Rust",
            is_safe=not self.detect_narrowing(src, dst),
        ))
        return chain

    def is_implicit_conversion_allowed(self, src: IntType, dst: IntType) -> bool:
        return src == dst  # No implicit conversions in Rust


# ── Cross-language comparison ────────────────────────────────────────────

@dataclass
class PromotionDifference:
    """A difference in how C and Rust handle a promotion."""
    c_step: PromotionStep | None
    rust_step: PromotionStep | None
    description: str
    is_semantic_diff: bool = True

    def __str__(self) -> str:
        parts = [self.description]
        if self.c_step:
            parts.append(f"  C:    {self.c_step}")
        if self.rust_step:
            parts.append(f"  Rust: {self.rust_step}")
        return "\n".join(parts)


@dataclass
class PromotionComparison:
    """Result of comparing C and Rust promotion behavior for a type pair."""
    c_chain: PromotionChain
    rust_chain: PromotionChain
    differences: list[PromotionDifference] = field(default_factory=list)
    c_common_type: IntType | None = None
    rust_common_type: IntType | None = None

    @property
    def has_semantic_differences(self) -> bool:
        return any(d.is_semantic_diff for d in self.differences)

    @property
    def num_differences(self) -> int:
        return len(self.differences)

    def summary(self) -> str:
        lines = [
            "Promotion Comparison (C vs Rust):",
            f"  C common type:    {c_type_name(self.c_common_type) if self.c_common_type else 'N/A'}",
            f"  Rust common type: {c_type_name(self.rust_common_type) if self.rust_common_type else 'N/A'}",
            f"  C chain:   {self.c_chain}",
            f"  Rust chain: {self.rust_chain}",
            f"  Differences: {self.num_differences}",
        ]
        for d in self.differences:
            lines.append(f"    - {d.description}")
        return "\n".join(lines)

    @classmethod
    def compare(
        cls,
        lhs: IntType,
        rhs: IntType,
        c_rules: CPromotionRules | None = None,
        rust_rules: RustPromotionRules | None = None,
    ) -> "PromotionComparison":
        """Compare how C and Rust would handle a binary operation
        between lhs and rhs types."""
        c_rules = c_rules or CPromotionRules()
        rust_rules = rust_rules or RustPromotionRules()

        c_common, c_lhs_chain, c_rhs_chain = c_rules.usual_arithmetic_conversion(lhs, rhs)
        r_common, r_lhs_chain, r_rhs_chain = rust_rules.usual_arithmetic_conversion(lhs, rhs)

        # Use the lhs chain for comparison; both chains carry the same info
        comparison = cls(
            c_chain=c_lhs_chain,
            rust_chain=r_lhs_chain,
            c_common_type=c_common,
            rust_common_type=r_common,
        )

        # Detect differences
        if c_common != r_common:
            comparison.differences.append(PromotionDifference(
                c_step=c_lhs_chain.steps[-1] if c_lhs_chain.steps else None,
                rust_step=r_lhs_chain.steps[-1] if r_lhs_chain.steps else None,
                description=(
                    f"Common type differs: C={c_type_name(c_common)} vs "
                    f"Rust={c_type_name(r_common)}"
                ),
            ))

        # C implicit promotion where Rust requires explicit cast
        if not c_lhs_chain.is_identity and r_lhs_chain.is_identity:
            comparison.differences.append(PromotionDifference(
                c_step=c_lhs_chain.steps[0] if c_lhs_chain.steps else None,
                rust_step=None,
                description="C implicitly promotes LHS; Rust requires no promotion",
            ))

        # Check if C promotes a type that Rust wouldn't
        for c_step in c_lhs_chain.steps:
            if c_step.kind is PromotionKind.INTEGER_PROMOTION:
                comparison.differences.append(PromotionDifference(
                    c_step=c_step,
                    rust_step=None,
                    description=(
                        f"C implicitly promotes {c_type_name(c_step.src_type)} → "
                        f"{c_type_name(c_step.dst_type)}; Rust does not"
                    ),
                ))

        # Sign conversion differences
        for c_step in c_lhs_chain.steps + c_rhs_chain.steps:
            if c_step.kind is PromotionKind.SIGN_CONVERSION:
                comparison.differences.append(PromotionDifference(
                    c_step=c_step,
                    rust_step=None,
                    description=(
                        f"C implicitly changes sign: "
                        f"{c_type_name(c_step.src_type)} → "
                        f"{c_type_name(c_step.dst_type)}"
                    ),
                ))

        return comparison


# ── Convenience functions ────────────────────────────────────────────────

def c_promote(ty: IntType, int_width: int = 32) -> IntType:
    """Apply C integer promotion to a type."""
    return CPromotionRules(int_width).promote(ty)


def c_usual_conversion(
    lhs: IntType, rhs: IntType, int_width: int = 32,
) -> IntType:
    """Return the common type after C usual arithmetic conversions."""
    common, _, _ = CPromotionRules(int_width).usual_arithmetic_conversion(lhs, rhs)
    return common


def compare_promotions(lhs: IntType, rhs: IntType) -> PromotionComparison:
    """Compare C and Rust promotion behavior for a type pair."""
    return PromotionComparison.compare(lhs, rhs)


def visualize_c_ranks() -> str:
    """Visualize the C integer conversion rank table."""
    lines = ["C Integer Conversion Ranks:", "=" * 50]
    sorted_ranks = sorted(_C_RANK_TABLE.items(), key=lambda x: x[1])
    for (width, sign), rank in sorted_ranks:
        name = _C_TYPE_NAMES.get((width, sign), f"{'i' if sign is Signedness.SIGNED else 'u'}{width}")
        lines.append(f"  Rank {rank}: {name:20s} ({width}-bit {sign.name})")
    return "\n".join(lines)
