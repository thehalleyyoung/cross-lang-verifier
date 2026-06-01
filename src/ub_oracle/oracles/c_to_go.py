"""
Back-compatibility shim for the C -> Go pair (100_STEPS step 37).

The C -> Go oracles are now generated data-driven from the unified
:mod:`~src.ub_oracle.oracles.target_pairs` table (step 39), alongside C -> Swift,
so a new target language is configuration rather than bespoke oracle code. This
module is kept as a stable import path; importing it ensures the target pairs are
registered and exposes the generated Go oracle classes by name.
"""

from __future__ import annotations

from . import target_pairs as _tp

# The Go oracles, generated from the anchor oracles + the Go TargetPack.
GoSignedOverflowOracle = type(_tp.GENERATED[("go", "signed_overflow")])
GoShiftOutOfRangeOracle = type(_tp.GENERATED[("go", "shift_oob")])
GoDivisionByZeroOracle = type(_tp.GENERATED[("go", "div_by_zero")])
GoIntMinDivNeg1Oracle = type(_tp.GENERATED[("go", "intmin_div_neg1")])
GoArrayOutOfBoundsOracle = type(_tp.GENERATED[("go", "array_oob")])

__all__ = [
    "GoSignedOverflowOracle",
    "GoShiftOutOfRangeOracle",
    "GoDivisionByZeroOracle",
    "GoIntMinDivNeg1Oracle",
    "GoArrayOutOfBoundsOracle",
]
