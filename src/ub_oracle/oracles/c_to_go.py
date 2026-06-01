"""
C -> Go divergence oracles (100_STEPS step 37): the *second language pair*.

This module is the project's central generality proof. Everything that made the
C -> Rust anchor work — the Z3 witness search, the divergence taxonomy, the
re-execution harness, the verifier and its soundness policy — is reused verbatim
here; the **only** thing that changes for a new target language is the small bit
of target-semantics knowledge:

  1. how to *emit* the equivalent program in the target language, and
  2. which return codes count as a language-*defined* outcome.

Each oracle below therefore subclasses its C -> Rust counterpart and overrides
only the Counterexample's target snippet (Go source) and ``target_lang``; the
divergence-finding ``find_divergence`` is inherited unchanged. The harness learns
Go's definedness predicate (rc in {0, 2}: a value, or a runtime panic such as
divide-by-zero / index-out-of-range) via ``RunOutcome.target_outcome_defined``.

Real-compiler ground truth (verified with go1.25): on the *same* witness input
the C program executes UB (UBSan traps) while the Go program is fully defined:

    * signed overflow :  C ``x+c > x`` UB / optimiser-exploited;
                         Go ``x+c`` wraps to a defined value.
    * shift-out-of-range : C ``x<<s`` (s>=width) UB; Go ``x<<uint(s)`` -> 0.
    * division-by-zero   : C ``a/b`` (b==0) UB; Go ``a/b`` -> defined panic (rc 2).
    * INT_MIN / -1       : C signed-division overflow UB; Go wraps to a defined
                           value (Go spec: x/-1 == x for the most-negative x).
    * array OOB          : C ``a[i]`` (i>=len) UB; Go ``a[i]`` -> defined panic.
"""

from __future__ import annotations

from typing import Dict

from ..plugin import register
from ..replay import Counterexample
from .signed_overflow import SignedOverflowOracle, _TYPE_MAP as _SO_TYPES
from .integer_ub import (
    ShiftOutOfRangeOracle,
    DivisionByZeroOracle,
    IntMinDivNeg1Oracle,
)
from .memory_shape import ArrayOutOfBoundsOracle

# width -> (Go type, strconv ParseInt bit-size)
_GO_TYPE = {32: ("int32", 32), 64: ("int64", 64)}


def _go_signed_overflow_src(op: str, c: int, width: int, var: str) -> str:
    gtype, bits = _GO_TYPE[width]
    c_op = "+" if op == "add" else "-"
    cmp = ">" if op == "add" else "<"
    return (
        "package main\n"
        "import (\n\t\"fmt\"\n\t\"os\"\n\t\"strconv\"\n)\n"
        f"func f({var} {gtype}) {gtype} {{\n"
        f"\tif {var} {c_op} {c} {cmp} {var} {{\n\t\treturn 1\n\t}}\n\treturn 0\n}}\n"
        "func main() {\n"
        f"\tv, _ := strconv.ParseInt(os.Args[1], 10, {bits})\n"
        f"\tfmt.Println(f({gtype}(v)))\n"
        "}\n"
    )


def _go_two_arg_src(width: int, expr: str, *, a: str, b: str,
                    btype_bits: int = None) -> str:
    """A two-argument Go program: parse two ints, print ``f(a, b)``.

    ``expr`` is the body of ``f`` (it may reference ``a`` and ``b``)."""
    gtype, bits = _GO_TYPE[width]
    bbits = btype_bits or bits
    return (
        "package main\n"
        "import (\n\t\"fmt\"\n\t\"os\"\n\t\"strconv\"\n)\n"
        f"func f({a} {gtype}, {b} {gtype}) {gtype} {{\n\treturn {expr}\n}}\n"
        "func main() {\n"
        f"\tav, _ := strconv.ParseInt(os.Args[1], 10, {bits})\n"
        f"\tbv, _ := strconv.ParseInt(os.Args[2], 10, {bbits})\n"
        f"\tfmt.Println(f({gtype}(av), {gtype}(bv)))\n"
        "}\n"
    )


def _retarget(ce: Counterexample, go_src: str, note: str) -> Counterexample:
    """Swap a C->Rust counterexample's target to the equivalent Go program,
    preserving the (identical) C source and the witness inputs."""
    ce.target_lang = "go"
    ce.target_snippet = go_src
    ce.divergence_witness = note
    return ce


class GoSignedOverflowOracle(SignedOverflowOracle):
    target_lang = "go"

    def _build_counterexample(self, op, c, width, var, witness) -> Counterexample:
        ce = super()._build_counterexample(op, c, width, var, witness)
        go_src = _go_signed_overflow_src(op, c, width, var)
        c_op = "+" if op == "add" else "-"
        cmp = ">" if op == "add" else "<"
        return _retarget(
            ce, go_src,
            f"C signed {op} overflow at {var}={witness} (UB; optimiser may assume "
            f"`{var} {c_op} {c} {cmp} {var}` always holds), whereas Go's `{var} {c_op} {c}` "
            f"wraps deterministically to a defined value.")


class GoShiftOutOfRangeOracle(ShiftOutOfRangeOracle):
    target_lang = "go"

    def _build(self, width, var, svar, x_val, shift_amt) -> Counterexample:
        ce = super()._build(width, var, svar, x_val, shift_amt)
        go_src = _go_two_arg_src(width, f"{var} << uint({svar})",
                                 a=var, b=svar, btype_bits=32)
        return _retarget(
            ce, go_src,
            f"C `{var} << {svar}` with {svar}={shift_amt} >= width {width} is UB; "
            f"Go masks/widens the count and yields a defined value (0).")


class GoDivisionByZeroOracle(DivisionByZeroOracle):
    target_lang = "go"

    def _build(self, width, op, avar, bvar, a_val, b_val) -> Counterexample:
        ce = super()._build(width, op, avar, bvar, a_val, b_val)
        go_op = "/" if op == "div" else "%"
        go_src = _go_two_arg_src(width, f"{avar} {go_op} {bvar}", a=avar, b=bvar)
        return _retarget(
            ce, go_src,
            f"C `{avar} {go_op} {bvar}` with {bvar}=0 is UB; Go panics "
            f"deterministically at runtime (a defined, observable outcome).")


class GoIntMinDivNeg1Oracle(IntMinDivNeg1Oracle):
    target_lang = "go"

    def _build(self, width, op, avar, bvar, a_val, b_val) -> Counterexample:
        ce = super()._build(width, op, avar, bvar, a_val, b_val)
        go_op = "/" if op == "div" else "%"
        go_src = _go_two_arg_src(width, f"{avar} {go_op} {bvar}", a=avar, b=bvar)
        return _retarget(
            ce, go_src,
            f"C `{avar} {go_op} {bvar}` with {avar}={a_val}, {bvar}={b_val} overflows "
            f"signed division (UB); Go defines x/-1 == x for the most-negative x and "
            f"returns a defined value.")


class GoArrayOutOfBoundsOracle(ArrayOutOfBoundsOracle):
    target_lang = "go"

    def _build(self, length, var, idx) -> Counterexample:
        ce = super()._build(length, var, idx)
        elems = ", ".join(str(10 + k) for k in range(length))
        go_src = (
            "package main\n"
            "import (\n\t\"fmt\"\n\t\"os\"\n\t\"strconv\"\n)\n"
            f"var a = [{length}]int32{{{elems}}}\n"
            f"func f({var} int) int32 {{ return a[{var}] }}\n"
            "func main() {\n"
            f"\t{var}, _ := strconv.Atoi(os.Args[1])\n"
            f"\tfmt.Println(f({var}))\n"
            "}\n"
        )
        return _retarget(
            ce, go_src,
            f"C `a[{var}]` with {var}={idx} on a length-{length} array is UB; "
            f"Go bounds-checks and panics deterministically (a defined outcome).")


# Register the C -> Go pair. These coexist with the C -> Rust anchor oracles:
# the verifier routes a unit to them only when it declares ``target_lang: "go"``.
register(GoSignedOverflowOracle())
register(GoShiftOutOfRangeOracle())
register(GoDivisionByZeroOracle())
register(GoIntMinDivNeg1Oracle())
register(GoArrayOutOfBoundsOracle())
