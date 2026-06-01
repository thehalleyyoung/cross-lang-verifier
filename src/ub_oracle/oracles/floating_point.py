"""
Floating-point contraction divergence oracle (100_STEPS step 19).

The expression ``a*b + c`` is, in C, *licensed to be contracted* into a single
fused multiply-add (one rounding) at the implementation's discretion (C17 6.5p8,
``FP_CONTRACT``).  Whether it is contracted is therefore **unspecified**, so the
result is not uniquely fixed by the language: compiling the same source with
``-ffp-contract=off`` (two roundings) versus ``-ffp-contract=fast`` (one
rounding) yields *different* observable results on the same input.  Rust, by
contrast, never auto-contracts — ``a*b + c`` always rounds twice — so its
translation is a single deterministic, defined value.

This oracle uses Z3's **floating-point theory** to *find* an input ``(a,b,c)``
for which the fused and unfused evaluations actually differ, with a heavy-
cancellation side-constraint so the rounding of ``a*b`` is observable in the
printed result.  The witness is then re-executed with real ``clang`` (off vs
fast) and ``rustc`` to confirm the divergence (``optimizer_exploited`` mode).
"""

from __future__ import annotations

import struct
from typing import Dict, Optional, Tuple

import z3

from ..catalogue import FP_CONTRACTION, Definedness
from ..plugin import DivergenceOracle, OracleResult, OracleVerdict, register
from ..replay import Counterexample

# clang flag pair whose disagreement evidences the (unspecified) contraction.
_OFF = ["-O2", "-ffp-contract=off"]
_FAST = ["-O2", "-ffp-contract=fast"]


def _fpnum_to_float(model, term) -> float:
    """Convert a Z3 Float64 model value to a Python float via its IEEE bits."""
    bv = model.eval(z3.fpToIEEEBV(term), model_completion=True)
    return struct.unpack("<d", struct.pack("<Q", bv.as_long()))[0]


class FpContractionOracle(DivergenceOracle):
    """``a*b + c`` fused (one rounding) vs unfused (two roundings)."""

    divergence_class = FP_CONTRACTION.key
    source_lang = "c"
    target_lang = "rust"
    confirmation_mode = "optimizer_exploited"
    optimizer_flag_variants = (_OFF, _FAST)

    def applies_to(self, unit: Dict) -> bool:
        if unit.get("probe") not in (None, self.divergence_class):
            return False
        return unit.get("kind") == "fp_fma"

    def find_divergence(self, unit: Dict) -> OracleResult:
        if not self.applies_to(unit):
            return OracleResult(OracleVerdict.NOT_APPLICABLE, self.divergence_class,
                                detail="unit is not an fp multiply-add")

        fp = z3.Float64()
        rm = z3.RNE()
        a, b, c = z3.FP("a", fp), z3.FP("b", fp), z3.FP("c", fp)
        prod = z3.fpMul(rm, a, b)
        fused = z3.fpFMA(rm, a, b, c)              # one rounding (contract=fast)
        unfused = z3.fpAdd(rm, prod, c)            # two roundings (contract=off)

        solver = z3.Solver()
        solver.add(fused != unfused)
        # keep operands well-scaled finite normals so the re-executed values are
        # clean (no inf/overflow/denormal surprises).
        for v in (a, b, c):
            solver.add(z3.fpIsNormal(v))
            solver.add(z3.fpLEQ(z3.fpAbs(v), z3.FPVal(1e3, fp)))
            solver.add(z3.fpGEQ(z3.fpAbs(v), z3.FPVal(1.0, fp)))
        # heavy cancellation: |a*b + c| is >= 2^40x smaller than |a*b|, so the
        # rounding error of `a*b` dominates and is visible in the result.
        solver.add(z3.fpLT(z3.fpMul(rm, z3.fpAbs(unfused), z3.FPVal(2.0 ** 40, fp)),
                           z3.fpAbs(prod)))

        if solver.check() != z3.sat:
            return OracleResult(OracleVerdict.NO_DIVERGENCE_FOUND, self.divergence_class,
                                detail="no contraction-sensitive witness found")
        m = solver.model()
        av, bv, cv = (_fpnum_to_float(m, a), _fpnum_to_float(m, b), _fpnum_to_float(m, c))

        ce = self._build(av, bv, cv)
        return OracleResult(OracleVerdict.DIVERGENT, self.divergence_class,
                            counterexample=ce,
                            detail=f"Z3 FP witness a={av!r}, b={bv!r}, c={cv!r}")

    def _build(self, av: float, bv: float, cv: float) -> Counterexample:
        # str(float) in Python 3 is the shortest round-tripping decimal, parsed
        # identically by C strtod and Rust's f64 parser.
        c_src = (
            "#include <stdio.h>\n"
            "#include <stdlib.h>\n"
            "double f(double a, double b, double c){ return a*b + c; }\n"
            "int main(int argc, char**argv){\n"
            "    if (argc < 4) return 2;\n"
            "    double a = strtod(argv[1], 0);\n"
            "    double b = strtod(argv[2], 0);\n"
            "    double c = strtod(argv[3], 0);\n"
            "    printf(\"%.17g\\n\", f(a, b, c));\n"
            "    return 0;\n"
            "}\n"
        )
        rust_src = (
            "fn f(a: f64, b: f64, c: f64) -> f64 { a * b + c }\n"
            "fn main(){\n"
            "    let v: Vec<f64> = std::env::args().skip(1)\n"
            "        .map(|s| s.parse().unwrap()).collect();\n"
            "    println!(\"{:.17}\", f(v[0], v[1], v[2]));\n"
            "}\n"
        )
        return Counterexample(
            divergence_class=self.divergence_class,
            source_lang="c", target_lang="rust",
            inputs={"a": av, "b": bv, "c": cv},
            source_snippet=c_src, target_snippet=rust_src,
            source_definedness=Definedness.UNSPECIFIED.value,
            divergence_witness=(
                f"C `a*b + c` with a={av!r}, b={bv!r}, c={cv!r} is unspecified up to "
                f"FMA contraction: `-ffp-contract=off` and `-ffp-contract=fast` give "
                f"different results, while Rust always rounds twice (deterministic)."
            ),
            definedness_witness=(
                "all three inputs are finite normal doubles; the divergence is "
                "purely the (unspecified) contraction licence, not invalid input."
            ),
        )


register(FpContractionOracle())
