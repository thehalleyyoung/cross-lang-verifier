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

from typing import Dict

from ..catalogue import FP_CONTRACTION, Definedness
from ..plugin import DivergenceOracle, OracleResult, OracleVerdict, register
from ..replay import Counterexample
from ..smt_float import prove_fp_contraction_witness, solve_fp_contraction

# clang flag pair whose disagreement evidences the (unspecified) contraction.
_OFF = ["-O2", "-ffp-contract=off"]
_FAST = ["-O2", "-ffp-contract=fast"]

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

        witness = solve_fp_contraction()
        if witness is None:
            return OracleResult(OracleVerdict.NO_DIVERGENCE_FOUND, self.divergence_class,
                                detail="no contraction-sensitive witness found")
        proof = prove_fp_contraction_witness(witness)
        if not proof.proved:
            return OracleResult(OracleVerdict.NO_DIVERGENCE_FOUND, self.divergence_class,
                                detail=proof.detail)

        ce = self._build(witness.a, witness.b, witness.c)
        return OracleResult(OracleVerdict.DIVERGENT, self.divergence_class,
                            counterexample=ce,
                            detail=(
                                f"Z3 FP witness a={witness.a!r}, b={witness.b!r}, "
                                f"c={witness.c!r}; fused={witness.result_bits['fused']} "
                                f"unfused={witness.result_bits['unfused']}"
                            ))

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
