"""
Bit-precise Z3 FP artifact for the floating-point contraction oracle.

This is the Step-134 companion to :mod:`ub_oracle.oracles.floating_point`.
The shipped oracle already confirms ``a*b + c`` against real ``clang`` and
``rustc``.  This module makes the symbolic side a first-class artifact:

* the witness is found in Z3's IEEE-754 binary64 theory with explicit RNE,
* the exact IEEE operand/result bit patterns are retained, and
* :func:`prove_fp_contraction_witness` replays those bits back through Z3 and
  proves that fused single-rounding differs from unfused double-rounding.

The constraints intentionally mirror the operational oracle's historical
constraints.  In particular, the cancellation side-condition keeps the
single-vs-double rounding gap visible in the ``%.17g`` output emitted by the
real compiler harness.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import z3

CLASS_KEY = "fp_contraction"
WIDTH = 64
_CANCELLATION_FACTOR = 2.0 ** 40


def _bits_to_float(bits: int) -> float:
    return struct.unpack("<d", struct.pack("<Q", bits & ((1 << WIDTH) - 1)))[0]


def _model_bits(model: z3.ModelRef, term: z3.FPRef) -> int:
    bv = model.eval(z3.fpToIEEEBV(term), model_completion=True)
    return bv.as_long()


def _fp_from_bits(bits: int) -> z3.FPRef:
    return z3.fpBVToFP(z3.BitVecVal(bits, WIDTH), z3.Float64())


@dataclass(frozen=True)
class FpContractionWitness:
    """Concrete IEEE-754 binary64 witness for FMA contraction sensitivity."""

    a: float
    b: float
    c: float
    a_bits: int
    b_bits: int
    c_bits: int
    fused_bits: int
    unfused_bits: int

    @property
    def fused(self) -> float:
        return _bits_to_float(self.fused_bits)

    @property
    def unfused(self) -> float:
        return _bits_to_float(self.unfused_bits)

    @property
    def inputs(self) -> Dict[str, float]:
        return {"a": self.a, "b": self.b, "c": self.c}

    @property
    def operand_bits(self) -> Dict[str, str]:
        return {
            "a": f"0x{self.a_bits:016x}",
            "b": f"0x{self.b_bits:016x}",
            "c": f"0x{self.c_bits:016x}",
        }

    @property
    def result_bits(self) -> Dict[str, str]:
        return {
            "fused": f"0x{self.fused_bits:016x}",
            "unfused": f"0x{self.unfused_bits:016x}",
        }


@dataclass(frozen=True)
class FpContractionProof:
    class_key: str
    width: int
    proved: bool
    detail: str


def _fp_contraction_terms(prefix: str = "") -> Tuple[z3.FPRef, ...]:
    fp = z3.Float64()
    rm = z3.RNE()
    p = f"{prefix}_" if prefix else ""
    a, b, c = z3.FP(f"{p}a", fp), z3.FP(f"{p}b", fp), z3.FP(f"{p}c", fp)
    prod = z3.fpMul(rm, a, b)
    fused = z3.fpFMA(rm, a, b, c)
    unfused = z3.fpAdd(rm, prod, c)
    return a, b, c, prod, fused, unfused


def fp_contraction_constraints(
    a: z3.FPRef,
    b: z3.FPRef,
    c: z3.FPRef,
    prod: z3.FPRef,
    fused: z3.FPRef,
    unfused: z3.FPRef,
) -> Tuple[z3.BoolRef, ...]:
    """Return the operational constraints used by the FP contraction oracle."""

    fp = z3.Float64()
    rm = z3.RNE()
    finite_normal = []
    for v in (a, b, c):
        finite_normal.extend(
            (
                z3.fpIsNormal(v),
                z3.fpLEQ(z3.fpAbs(v), z3.FPVal(1e3, fp)),
                z3.fpGEQ(z3.fpAbs(v), z3.FPVal(1.0, fp)),
            )
        )
    cancellation_visible = z3.fpLT(
        z3.fpMul(rm, z3.fpAbs(unfused), z3.FPVal(_CANCELLATION_FACTOR, fp)),
        z3.fpAbs(prod),
    )
    return (
        fused != unfused,
        *finite_normal,
        cancellation_visible,
    )


def solve_fp_contraction() -> Optional[FpContractionWitness]:
    """Find a bit-precise binary64 witness for FMA contraction divergence."""

    a, b, c, prod, fused, unfused = _fp_contraction_terms("w")
    solver = z3.Solver()
    solver.add(*fp_contraction_constraints(a, b, c, prod, fused, unfused))
    if solver.check() != z3.sat:
        return None

    model = solver.model()
    a_bits, b_bits, c_bits = (
        _model_bits(model, a),
        _model_bits(model, b),
        _model_bits(model, c),
    )
    fused_bits = _model_bits(model, fused)
    unfused_bits = _model_bits(model, unfused)
    return FpContractionWitness(
        a=_bits_to_float(a_bits),
        b=_bits_to_float(b_bits),
        c=_bits_to_float(c_bits),
        a_bits=a_bits,
        b_bits=b_bits,
        c_bits=c_bits,
        fused_bits=fused_bits,
        unfused_bits=unfused_bits,
    )


def prove_fp_contraction_witness(
    witness: FpContractionWitness,
) -> FpContractionProof:
    """Z3-prove that the stored IEEE bits satisfy the contraction formula.

    This is a concrete certificate check: the operand bits are reconstituted as
    binary64 constants, fused and unfused results are recomputed in Z3 FP theory,
    and the negation of the expected bit-level facts is proved unsatisfiable.
    """

    rm = z3.RNE()
    a, b, c = (
        _fp_from_bits(witness.a_bits),
        _fp_from_bits(witness.b_bits),
        _fp_from_bits(witness.c_bits),
    )
    prod = z3.fpMul(rm, a, b)
    fused = z3.fpFMA(rm, a, b, c)
    unfused = z3.fpAdd(rm, prod, c)
    conditions = z3.And(
        *fp_contraction_constraints(a, b, c, prod, fused, unfused),
        z3.fpToIEEEBV(fused) == z3.BitVecVal(witness.fused_bits, WIDTH),
        z3.fpToIEEEBV(unfused) == z3.BitVecVal(witness.unfused_bits, WIDTH),
    )
    solver = z3.Solver()
    solver.add(z3.Not(conditions))
    res = solver.check()
    if res == z3.unsat:
        return FpContractionProof(
            CLASS_KEY,
            WIDTH,
            True,
            "operand/result IEEE bits satisfy fused != unfused in Z3 FP theory",
        )
    if res == z3.sat:
        return FpContractionProof(
            CLASS_KEY,
            WIDTH,
            False,
            f"bit-level replay rejected witness: {solver.model()}",
        )
    return FpContractionProof(CLASS_KEY, WIDTH, False, f"z3 returned {res}")


def solve_and_prove_fp_contraction() -> FpContractionProof:
    witness = solve_fp_contraction()
    if witness is None:
        return FpContractionProof(CLASS_KEY, WIDTH, False, "no witness found")
    return prove_fp_contraction_witness(witness)
