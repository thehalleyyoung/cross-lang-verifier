"""
Integer promotion tracking for C↔Rust divergence detection.

C integer promotions (C11 §6.3.1.1) implicitly widen small integer types
(char, short, _Bool, bitfields) to int or unsigned int in arithmetic
expressions.  This creates subtle divergences with Rust's explicit typing
where u8 + u8 stays u8 (with potential overflow) rather than promoting to i32.

Divergence examples:
  - C: (uint8_t)200 + (uint8_t)100 → int promotion → 300 (no overflow)
    Rust: 200u8 + 100u8 → wrapping to 44 (or panic in debug)
  - C: (int16_t)(-1) < (uint16_t)(1) → both promoted to int → true
    Rust: comparison requires explicit cast
  - C: ~(uint8_t)0 → promoted to int → 0xFFFFFF00 (negative as int!)
    Rust: !0u8 → 0xFF (stays u8)

This module tracks the promotion chain so the verifier can model these
divergences precisely.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Dict, List, Optional, Sequence, Tuple

from .types import (
    IRType,
    IntType,
    FloatType,
    Signedness,
    Language,
)


# ── Promotion rules ─────────────────────────────────────────────────────

class PromotionKind(Enum):
    """Kind of integer promotion or conversion."""
    INTEGER_PROMOTION = auto()     # C11 §6.3.1.1: small → int/uint
    USUAL_ARITHMETIC = auto()      # C11 §6.3.1.8: binary operand balancing
    IMPLICIT_NARROWING = auto()    # Assignment to narrower type (lossy)
    EXPLICIT_CAST = auto()         # Explicit (type)expr cast
    SIGN_CONVERSION = auto()       # signed ↔ unsigned (same width)
    FLOAT_TO_INT = auto()          # float → int truncation
    INT_TO_FLOAT = auto()          # int → float (possible precision loss)
    NONE = auto()                  # no promotion needed


@dataclass
class PromotionStep:
    """A single step in a promotion chain."""
    kind: PromotionKind
    from_type: IntType
    to_type: IntType
    is_implicit: bool = True      # True if C compiler inserts silently
    location: str = ""            # source location (file:line)
    preserves_value: bool = True  # False if information is lost
    c_standard_ref: str = ""      # C standard section

    @property
    def widens(self) -> bool:
        return self.to_type.width > self.from_type.width

    @property
    def narrows(self) -> bool:
        return self.to_type.width < self.from_type.width

    @property
    def changes_sign(self) -> bool:
        return self.from_type.signedness != self.to_type.signedness

    def to_dict(self) -> dict:
        return {
            "kind": self.kind.name,
            "from": str(self.from_type),
            "to": str(self.to_type),
            "implicit": self.is_implicit,
            "preserves_value": self.preserves_value,
            "location": self.location,
        }


@dataclass
class PromotionChain:
    """A chain of promotion steps applied to a value in an expression."""
    original_type: IntType
    steps: List[PromotionStep] = field(default_factory=list)
    expression_context: str = ""   # e.g., "binary_add lhs"

    @property
    def final_type(self) -> IntType:
        if not self.steps:
            return self.original_type
        return self.steps[-1].to_type

    @property
    def has_implicit_promotion(self) -> bool:
        return any(s.is_implicit for s in self.steps)

    @property
    def has_narrowing(self) -> bool:
        return any(s.narrows for s in self.steps)

    @property
    def has_sign_change(self) -> bool:
        return any(s.changes_sign for s in self.steps)

    @property
    def divergence_risk(self) -> str:
        """Classify the risk of C/Rust divergence from this chain."""
        if self.has_implicit_promotion and self.original_type.width < 32:
            return "high"
        if self.has_sign_change:
            return "medium"
        if self.has_narrowing:
            return "medium"
        return "low"

    def to_dict(self) -> dict:
        return {
            "original": str(self.original_type),
            "final": str(self.final_type),
            "steps": [s.to_dict() for s in self.steps],
            "divergence_risk": self.divergence_risk,
            "context": self.expression_context,
        }


# ── Promotion divergence report ─────────────────────────────────────────

@dataclass
class PromotionDivergence:
    """A detected integer promotion divergence between C and Rust."""
    chain: PromotionChain
    c_result_type: IntType
    rust_result_type: IntType
    c_behavior: str
    rust_behavior: str
    severity: str  # "critical", "warning", "info"
    description: str
    example_values: Optional[List[int]] = None  # triggering inputs

    def to_dict(self) -> dict:
        return {
            "chain": self.chain.to_dict(),
            "c_result": str(self.c_result_type),
            "rust_result": str(self.rust_result_type),
            "c_behavior": self.c_behavior,
            "rust_behavior": self.rust_behavior,
            "severity": self.severity,
            "description": self.description,
            "example_values": self.example_values,
        }


# ── Promotion tracker ───────────────────────────────────────────────────

I32 = IntType(32, Signedness.SIGNED)
U32 = IntType(32, Signedness.UNSIGNED)
I64 = IntType(64, Signedness.SIGNED)
U64 = IntType(64, Signedness.UNSIGNED)


class IntegerPromotionTracker:
    """Tracks C integer promotions and detects Rust divergences.

    Usage::

        tracker = IntegerPromotionTracker()
        chain = tracker.promote_for_arithmetic(u8_type)
        divs = tracker.detect_binary_divergence(lhs_type, rhs_type, "add")
    """

    def __init__(self, language: Language = Language.C) -> None:
        self._language = language
        self._chains: List[PromotionChain] = []

    @property
    def all_chains(self) -> List[PromotionChain]:
        return list(self._chains)

    # -- C integer promotion (§6.3.1.1) -----------------------------------

    def integer_promote(self, typ: IntType, location: str = "") -> PromotionChain:
        """Apply C integer promotion to a type.

        C11 §6.3.1.1/2: if an int can represent all values of the original
        type, the value is converted to int; otherwise to unsigned int.
        """
        chain = PromotionChain(original_type=typ)

        # Only types narrower than int get promoted
        if typ.width >= 32:
            self._chains.append(chain)
            return chain

        # Can int represent all values?
        if typ.is_signed:
            # signed narrow → int (always fits)
            target = I32
        else:
            # unsigned narrow: fits in int if max_value <= INT32_MAX
            if typ.max_value <= I32.max_value:
                target = I32
            else:
                target = U32

        chain.steps.append(PromotionStep(
            kind=PromotionKind.INTEGER_PROMOTION,
            from_type=typ,
            to_type=target,
            is_implicit=True,
            location=location,
            preserves_value=True,
            c_standard_ref="C11 §6.3.1.1/2",
        ))

        self._chains.append(chain)
        return chain

    # -- Usual arithmetic conversions (§6.3.1.8) --------------------------

    def usual_arithmetic_conversion(
        self,
        lhs: IntType,
        rhs: IntType,
        location: str = "",
    ) -> Tuple[PromotionChain, PromotionChain]:
        """Apply C usual arithmetic conversions for a binary operator.

        Steps:
          1. Integer-promote both operands.
          2. If same type after promotion, done.
          3. If same signedness, convert narrower to wider.
          4. If unsigned rank >= signed rank, convert signed to unsigned.
          5. If signed can represent all unsigned values, convert unsigned to signed.
          6. Otherwise, convert both to unsigned version of signed type.
        """
        lhs_chain = self.integer_promote(lhs, location)
        rhs_chain = self.integer_promote(rhs, location)

        lhs_promoted = lhs_chain.final_type
        rhs_promoted = rhs_chain.final_type

        if lhs_promoted == rhs_promoted:
            return lhs_chain, rhs_chain

        # Same signedness → widen to larger
        if lhs_promoted.signedness == rhs_promoted.signedness:
            if lhs_promoted.width < rhs_promoted.width:
                self._add_conversion(lhs_chain, rhs_promoted,
                                     PromotionKind.USUAL_ARITHMETIC, location)
            elif rhs_promoted.width < lhs_promoted.width:
                self._add_conversion(rhs_chain, lhs_promoted,
                                     PromotionKind.USUAL_ARITHMETIC, location)
            return lhs_chain, rhs_chain

        # Different signedness — identify which is unsigned
        if lhs_promoted.is_unsigned:
            unsigned_chain, signed_chain = lhs_chain, rhs_chain
            unsigned_type, signed_type = lhs_promoted, rhs_promoted
        else:
            unsigned_chain, signed_chain = rhs_chain, lhs_chain
            unsigned_type, signed_type = rhs_promoted, lhs_promoted

        if unsigned_type.width >= signed_type.width:
            # Convert signed to unsigned
            self._add_conversion(signed_chain, unsigned_type.to_unsigned(),
                                 PromotionKind.SIGN_CONVERSION, location)
        elif signed_type.max_value >= unsigned_type.max_value:
            # Signed can represent all unsigned values
            self._add_conversion(unsigned_chain, signed_type,
                                 PromotionKind.USUAL_ARITHMETIC, location)
        else:
            # Convert both to unsigned version of signed type
            target = signed_type.to_unsigned()
            self._add_conversion(signed_chain, target,
                                 PromotionKind.SIGN_CONVERSION, location)
            self._add_conversion(unsigned_chain, target,
                                 PromotionKind.USUAL_ARITHMETIC, location)

        return lhs_chain, rhs_chain

    def _add_conversion(
        self,
        chain: PromotionChain,
        target: IntType,
        kind: PromotionKind,
        location: str,
    ) -> None:
        """Add a conversion step to an existing chain."""
        current = chain.final_type
        if current == target:
            return
        preserves = (
            target.width >= current.width
            and (target.signedness == current.signedness or target.is_signed)
        )
        chain.steps.append(PromotionStep(
            kind=kind,
            from_type=current,
            to_type=target,
            is_implicit=True,
            location=location,
            preserves_value=preserves,
            c_standard_ref="C11 §6.3.1.8",
        ))

    # -- Divergence detection ---------------------------------------------

    def detect_binary_divergence(
        self,
        lhs: IntType,
        rhs: IntType,
        op: str = "add",
        location: str = "",
    ) -> List[PromotionDivergence]:
        """Detect C/Rust divergences for a binary arithmetic operation.

        Args:
            lhs: left-hand operand type
            rhs: right-hand operand type
            op: operation name ("add", "sub", "mul", "div", "mod", "shl", "shr",
                "bitand", "bitor", "bitxor", "lt", "le", "gt", "ge", "eq", "ne")
            location: source location

        Returns:
            List of PromotionDivergence instances.
        """
        divergences: List[PromotionDivergence] = []

        # C: apply usual arithmetic conversions
        lhs_chain, rhs_chain = self.usual_arithmetic_conversion(lhs, rhs, location)
        c_type = lhs_chain.final_type  # both end up at same type

        # Rust: types stay as-is (no implicit promotion)
        # Binary ops require same type; we assume the wider type
        rust_type = lhs if lhs.width >= rhs.width else rhs

        # Detect widening divergence (u8+u8 → int in C, stays u8 in Rust)
        if c_type.width > max(lhs.width, rhs.width):
            examples = self._compute_overflow_examples(lhs, rhs, op)
            divergences.append(PromotionDivergence(
                chain=lhs_chain,
                c_result_type=c_type,
                rust_result_type=rust_type,
                c_behavior=(
                    f"Operands promoted from {lhs}/{rhs} to {c_type}. "
                    f"Arithmetic performed in {c_type.width}-bit, no overflow."
                ),
                rust_behavior=(
                    f"Operands stay as {rust_type}. "
                    f"Arithmetic in {rust_type.width}-bit, may overflow/panic."
                ),
                severity="critical" if examples else "warning",
                description=(
                    f"C integer promotion widens {lhs}/{rhs} → {c_type} before '{op}'. "
                    f"Rust keeps {rust_type}: overflow behavior diverges."
                ),
                example_values=examples,
            ))

        # Detect sign conversion divergence
        if lhs_chain.has_sign_change or rhs_chain.has_sign_change:
            divergences.append(PromotionDivergence(
                chain=lhs_chain if lhs_chain.has_sign_change else rhs_chain,
                c_result_type=c_type,
                rust_result_type=rust_type,
                c_behavior=f"Implicit sign conversion in usual arithmetic conversions.",
                rust_behavior=f"Rust requires explicit `as` cast for sign conversion.",
                severity="warning",
                description=(
                    f"C implicitly converts between signed/unsigned in '{op}'. "
                    "Rust rejects mixed-sign arithmetic without explicit cast."
                ),
            ))

        # Detect unary bitwise NOT divergence for narrow types
        if op in ("bitnot", "neg") and lhs.width < 32:
            promoted = self.integer_promote(lhs, location).final_type
            divergences.append(PromotionDivergence(
                chain=PromotionChain(original_type=lhs),
                c_result_type=promoted,
                rust_result_type=lhs,
                c_behavior=(
                    f"~({lhs})x promotes to {promoted} first: "
                    f"~(uint8_t)0 = 0xFFFFFF00 as int (negative!)"
                ),
                rust_behavior=(
                    f"!x stays {lhs}: !0u8 = 0xFF"
                ),
                severity="critical",
                description=(
                    f"C promotes {lhs} to {promoted} before bitwise NOT. "
                    f"Result has {promoted.width} bits, not {lhs.width}."
                ),
                example_values=[0, lhs.max_value],
            ))

        return divergences

    def detect_comparison_divergence(
        self,
        lhs: IntType,
        rhs: IntType,
        location: str = "",
    ) -> List[PromotionDivergence]:
        """Detect divergences in comparison expressions.

        Classic bug: (int16_t)(-1) < (uint16_t)(1)
          C: both promoted to int → -1 < 1 → true
          Rust: cannot compare i16 and u16 directly
        """
        divergences: List[PromotionDivergence] = []

        if lhs.signedness != rhs.signedness:
            lhs_chain, rhs_chain = self.usual_arithmetic_conversion(lhs, rhs, location)
            c_common = lhs_chain.final_type

            divergences.append(PromotionDivergence(
                chain=lhs_chain,
                c_result_type=c_common,
                rust_result_type=lhs,
                c_behavior=(
                    f"Mixed-sign comparison: {lhs} vs {rhs} → "
                    f"both promoted to {c_common} (C implicit conversion)."
                ),
                rust_behavior=(
                    f"Rust rejects direct comparison of {lhs} and {rhs}. "
                    "Requires explicit cast."
                ),
                severity="warning",
                description=(
                    f"C implicitly converts {lhs} and {rhs} to {c_common} for comparison. "
                    "Sign-dependent comparison results may differ."
                ),
                example_values=[-1, 1, lhs.max_value, rhs.max_value],
            ))

        return divergences

    def _compute_overflow_examples(
        self,
        lhs: IntType,
        rhs: IntType,
        op: str,
    ) -> List[int]:
        """Find example values that overflow in narrow type but not in promoted type."""
        examples = []
        narrow = lhs if lhs.width <= rhs.width else rhs
        wide = I32

        boundary_values = [
            narrow.max_value, narrow.max_value - 1,
            narrow.min_value, narrow.min_value + 1,
            0, 1, -1 if narrow.is_signed else 0,
        ]

        for a in boundary_values:
            if not narrow.contains(a):
                continue
            for b in boundary_values:
                if not narrow.contains(b):
                    continue
                try:
                    if op == "add":
                        result = a + b
                    elif op == "sub":
                        result = a - b
                    elif op == "mul":
                        result = a * b
                    else:
                        continue

                    # Overflows in narrow type but not in wide?
                    if not narrow.contains(result) and wide.contains(result):
                        examples.append(a)
                        examples.append(b)
                        if len(examples) >= 6:
                            return examples
                except (ZeroDivisionError, OverflowError):
                    continue

        return examples

    # -- Summary -----------------------------------------------------------

    def get_all_divergences(self) -> List[PromotionDivergence]:
        """Return all divergences detected so far across all tracked chains."""
        divs = []
        for chain in self._chains:
            if chain.has_implicit_promotion:
                divs.append(PromotionDivergence(
                    chain=chain,
                    c_result_type=chain.final_type,
                    rust_result_type=chain.original_type,
                    c_behavior=f"Implicit promotion from {chain.original_type} to {chain.final_type}",
                    rust_behavior=f"Type stays as {chain.original_type}",
                    severity="warning" if chain.divergence_risk != "high" else "critical",
                    description=f"Implicit C integer promotion: {chain.original_type} → {chain.final_type}",
                ))
        return divs
