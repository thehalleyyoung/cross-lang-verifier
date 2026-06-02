"""
Fast-math (``-ffast-math`` / ``-Ofast``) reassociation divergence oracle
(100_STEPS step 107).

IEEE-754 arithmetic is **not** associative: ``(x + y) - x`` is *not* the same
value as ``y`` once rounding is taken into account.  A standard C compilation
honours IEEE and rounds at every step, so for a large ``x`` that swallows a
small ``y`` the result is ``0``.  Under ``-ffast-math`` / ``-Ofast`` the compiler
is licensed to **reassociate** floating-point arithmetic as if it were over the
reals — it may fold ``(x + y) - x`` straight to ``y``.  The *same* C source then
produces a **different observable value** under ``-fno-fast-math`` vs
``-ffast-math``, so the result is not fixed by the program.

Rust and Go, by contrast, **never** auto-reassociate floating arithmetic:
``(x + y) - x`` always evaluates IEEE-strict (fusion/reassociation only via
explicit intrinsics), so each safe-target translation is a single, deterministic,
defined value.  That gap is the cross-language divergence.

The witness ``(x, y)`` is *found* with Z3's floating-point theory (a large ``x``
that exactly swallows a non-zero ``y``, so IEEE-strict yields ``0`` while the
reassociated value is ``y`` — a maximally visible, reproducible gap) and then
re-executed against real ``clang`` (``-fno-fast-math`` vs ``-ffast-math``) and
real ``rustc``/``go`` in ``optimizer_exploited`` mode, reusing the
FP-contraction harness exactly as step 107 asks.
"""

from __future__ import annotations

import struct
from typing import Dict

import z3

from ..catalogue import FAST_MATH_REASSOC, Definedness
from ..plugin import DivergenceOracle, OracleResult, OracleVerdict, register
from ..replay import Counterexample

# clang flag pair whose disagreement evidences the (unspecified) reassociation.
_STRICT = ["-O2", "-fno-fast-math"]
_FAST = ["-O2", "-ffast-math"]


def _fpnum_to_float(model, term) -> float:
    """Convert a Z3 Float64 model value to a Python float via its IEEE bits."""
    bv = model.eval(z3.fpToIEEEBV(term), model_completion=True)
    return struct.unpack("<d", struct.pack("<Q", bv.as_long()))[0]


def _find_reassoc_witness(unit: Dict):
    """Z3-find ``(x, y)`` with ``round(x + y) == x`` (so IEEE-strict
    ``(x + y) - x`` is exactly 0) but ``y != 0`` (so the reassociated value is a
    non-zero ``y``).  Operands are kept finite normals in a clean printable
    range so the re-executed decimals round-trip identically through C/Rust/Go."""
    fp = z3.Float64()
    rm = z3.RNE()
    x, y = z3.FP("x", fp), z3.FP("y", fp)
    solver = z3.Solver()
    # y is entirely swallowed by the addition: strict (x+y)-x rounds to 0.
    solver.add(z3.fpAdd(rm, x, y) == x)
    solver.add(z3.fpGT(z3.fpAbs(y), z3.FPVal(0.0, fp)))   # but y != 0
    for v in (x, y):
        solver.add(z3.fpIsNormal(v))
    # keep x large enough to swallow y yet inside a clean decimal range, and y a
    # small, exactly-representable magnitude.
    solver.add(z3.fpLEQ(z3.fpAbs(x), z3.FPVal(1e16, fp)))
    solver.add(z3.fpGEQ(z3.fpAbs(x), z3.FPVal(1e12, fp)))
    solver.add(z3.fpGEQ(z3.fpAbs(y), z3.FPVal(1.0, fp)))
    solver.add(z3.fpLEQ(z3.fpAbs(y), z3.FPVal(100.0, fp)))
    rng = unit.get("operand_range")
    if rng is not None:
        lo, hi = rng
        for v in (x, y):
            solver.add(z3.fpGEQ(v, z3.FPVal(float(lo), fp)),
                       z3.fpLEQ(v, z3.FPVal(float(hi), fp)))
    if solver.check() != z3.sat:
        return None
    m = solver.model()
    return _fpnum_to_float(m, x), _fpnum_to_float(m, y)


_C_SRC = (
    "#include <stdio.h>\n"
    "#include <stdlib.h>\n"
    "double g(double x, double y){ return (x + y) - x; }\n"
    "int main(int argc, char**argv){\n"
    "    if (argc < 3) return 2;\n"
    "    double x = strtod(argv[1], 0);\n"
    "    double y = strtod(argv[2], 0);\n"
    "    printf(\"%.17g\\n\", g(x, y));\n"
    "    return 0;\n"
    "}\n"
)

_RUST_SRC = (
    "fn g(x: f64, y: f64) -> f64 { (x + y) - x }\n"
    "fn main(){\n"
    "    let v: Vec<f64> = std::env::args().skip(1)\n"
    "        .map(|s| s.parse().unwrap()).collect();\n"
    "    println!(\"{:.17}\", g(v[0], v[1]));\n"
    "}\n"
)

_GO_SRC = (
    "package main\n"
    "import (\n\t\"fmt\"\n\t\"os\"\n\t\"strconv\"\n)\n"
    "func g(x, y float64) float64 { return (x + y) - x }\n"
    "func main() {\n"
    "\tx, _ := strconv.ParseFloat(os.Args[1], 64)\n"
    "\ty, _ := strconv.ParseFloat(os.Args[2], 64)\n"
    "\tfmt.Printf(\"%.17g\\n\", g(x, y))\n"
    "}\n"
)


class _FastMathBase(DivergenceOracle):
    divergence_class = FAST_MATH_REASSOC.key
    source_lang = "c"
    confirmation_mode = "optimizer_exploited"
    optimizer_flag_variants = (_STRICT, _FAST)

    def applies_to(self, unit: Dict) -> bool:
        if unit.get("probe") not in (None, self.divergence_class):
            return False
        return unit.get("kind") == "fp_reassoc"

    def _result(self, unit: Dict, target_src: str) -> OracleResult:
        if not self.applies_to(unit):
            return OracleResult(OracleVerdict.NOT_APPLICABLE, self.divergence_class,
                                detail="unit is not an fp reassociation")
        w = _find_reassoc_witness(unit)
        if w is None:
            return OracleResult(OracleVerdict.NO_DIVERGENCE_FOUND, self.divergence_class,
                                detail="no reassociation-sensitive witness found")
        xv, yv = w
        ce = self._build(xv, yv, target_src)
        return OracleResult(OracleVerdict.DIVERGENT, self.divergence_class,
                            counterexample=ce,
                            detail=f"Z3 FP witness x={xv!r}, y={yv!r}")

    def _build(self, xv: float, yv: float, target_src: str) -> Counterexample:
        return Counterexample(
            divergence_class=self.divergence_class,
            source_lang="c", target_lang=self.target_lang,
            inputs={"x": xv, "y": yv},
            source_snippet=_C_SRC, target_snippet=target_src,
            source_definedness=Definedness.UNSPECIFIED.value,
            divergence_witness=(
                f"C `(x + y) - x` with x={xv!r}, y={yv!r} is reassociation-"
                f"sensitive: `-fno-fast-math` rounds IEEE-strict to 0 while "
                f"`-ffast-math`/`-Ofast` folds it to y={yv!r}; the safe "
                f"{self.target_lang} port never auto-reassociates and is a single "
                f"deterministic, defined value."
            ),
            definedness_witness=(
                "both inputs are finite normal doubles; the divergence is purely "
                "the fast-math reassociation licence, not invalid input."
            ),
        )


class FastMathReassocOracle(_FastMathBase):
    """``(x+y)-x`` IEEE-strict vs ``-ffast-math`` reassociated, vs Rust."""

    target_lang = "rust"

    def find_divergence(self, unit: Dict) -> OracleResult:
        return self._result(unit, _RUST_SRC)


class GoFastMathReassocOracle(_FastMathBase):
    """``(x+y)-x`` IEEE-strict vs ``-ffast-math`` reassociated, vs Go."""

    target_lang = "go"

    def find_divergence(self, unit: Dict) -> OracleResult:
        return self._result(unit, _GO_SRC)


register(FastMathReassocOracle())
register(GoFastMathReassocOracle())
