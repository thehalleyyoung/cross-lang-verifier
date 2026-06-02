"""
Worked **external** divergence-oracle plugin (100_STEPS step 70 — the SDK proof).

This module lives *outside* ``src/ub_oracle/oracles/`` on purpose: it demonstrates
that a third party can add an entirely new C-UB-vs-target-defined divergence class
**without forking the engine** — only by depending on the public plugin SPI
(:mod:`src.ub_oracle.plugin`) and calling :func:`register`.  Importing this module
is all it takes for ``verify_unit`` / ``oracles_for`` to discover and run it.

The class it contributes — *float-to-integer cast overflow* — is genuinely new
(not in the core catalogue) and a textbook C undefined behaviour:

    C    : int f(double x){ return (int)x; }      // (int)1e30 is UB (C17 6.3.1.4)
    Rust : fn f(x:f64)->i32 { x as i32 }          // saturating, defined (>=1.45)

On a witness ``x`` whose magnitude exceeds ``INT_MAX``, the C conversion is
undefined (``-fsanitize=undefined`` traps via ``float-cast-overflow``), while the
Rust ``as`` cast is a *defined, saturating* conversion.  The harness confirms this
in ``trap_vs_defined`` mode against real ``clang`` + ``rustc``.

The witness is *found* with Z3's floating-point theory (a finite normal double
beyond the ``int`` range) rather than hard-coded, so this is a real oracle, not a
fixture — exactly the contract every core oracle honours.

Run the smoke check directly::

    python -m examples.plugins.float_cast_overflow_oracle
"""

from __future__ import annotations

import struct
from typing import Dict

import z3

# Public SPI — the ONLY thing a third-party plugin needs to depend on.
from src.ub_oracle.catalogue import Definedness
from src.ub_oracle.plugin import (
    DivergenceOracle,
    OracleResult,
    OracleVerdict,
    register,
)
from src.ub_oracle.replay import Counterexample

#: a brand-new divergence-class key, owned by this external plugin.
FLOAT_CAST_OVERFLOW = "float_cast_overflow"

_INT_MAX = (1 << 31) - 1


def _fpnum_to_float(model, term) -> float:
    bv = model.eval(z3.fpToIEEEBV(term), model_completion=True)
    return struct.unpack("<d", struct.pack("<Q", bv.as_long()))[0]


class FloatCastOverflowOracle(DivergenceOracle):
    """``(int)x`` (C UB on overflow) vs ``x as i32`` (Rust saturating, defined)."""

    divergence_class = FLOAT_CAST_OVERFLOW
    source_lang = "c"
    target_lang = "rust"
    #: C traps (UBSan) on a perfectly defined double input; Rust is defined.
    confirmation_mode = "trap_vs_defined"

    def applies_to(self, unit: Dict) -> bool:
        if unit.get("probe") not in (None, self.divergence_class):
            return False
        return unit.get("kind") == "float_to_int_cast"

    def find_divergence(self, unit: Dict) -> OracleResult:
        if not self.applies_to(unit):
            return OracleResult(OracleVerdict.NOT_APPLICABLE, self.divergence_class,
                                detail="unit is not a float-to-int cast")

        fp = z3.Float64()
        x = z3.FP("x", fp)
        # Find a finite normal double strictly above INT_MAX so (int)x overflows.
        s = z3.Solver()
        s.add(z3.fpIsNormal(x))
        s.add(z3.fpGT(x, z3.FPVal(float(_INT_MAX) + 1.0, fp)))
        s.add(z3.fpLEQ(x, z3.FPVal(1e18, fp)))  # keep it a clean, parseable magnitude
        if s.check() != z3.sat:
            return OracleResult(OracleVerdict.NO_DIVERGENCE_FOUND, self.divergence_class,
                                detail="no overflowing-cast witness found")
        xv = _fpnum_to_float(s.model(), x)
        ce = self._build(xv)
        return OracleResult(OracleVerdict.DIVERGENT, self.divergence_class,
                            counterexample=ce, detail=f"Z3 FP witness x={xv!r}")

    def _build(self, xv: float) -> Counterexample:
        c_src = (
            "#include <stdio.h>\n"
            "#include <stdlib.h>\n"
            "int f(double x){ return (int)x; }\n"
            "int main(int argc, char**argv){\n"
            "    if (argc < 2) return 2;\n"
            "    double x = strtod(argv[1], 0);\n"
            "    printf(\"%d\\n\", f(x));\n"
            "    return 0;\n"
            "}\n"
        )
        rust_src = (
            "fn f(x: f64) -> i32 { x as i32 }\n"
            "fn main(){\n"
            "    let x: f64 = std::env::args().nth(1).unwrap().parse().unwrap();\n"
            "    println!(\"{}\", f(x));\n"
            "}\n"
        )
        return Counterexample(
            divergence_class=self.divergence_class,
            source_lang="c", target_lang="rust",
            inputs={"x": xv},
            source_snippet=c_src, target_snippet=rust_src,
            source_definedness=Definedness.UNDEFINED.value,
            divergence_witness=(
                f"C `(int)x` with x={xv!r} overflows the `int` range — undefined "
                f"behaviour (C17 6.3.1.4); `-fsanitize=undefined` traps. Rust's "
                f"`x as i32` is a defined saturating conversion to i32::MAX."
            ),
            definedness_witness=(
                f"x={xv!r} is a finite normal double, so the input is fully defined; "
                f"only the C conversion is undefined."
            ),
        )


#: A ready-to-run translation unit for this class (the SDK consumer supplies these).
EXAMPLE_UNIT = {
    "name": "float_cast_demo",
    "kind": "float_to_int_cast",
    "probe": FLOAT_CAST_OVERFLOW,
    "source_lang": "c",
    "target_lang": "rust",
}


# Registering on import is the whole integration step — no engine edits required.
register(FloatCastOverflowOracle())


def _smoke() -> int:
    from src.ub_oracle.verify import verify_unit
    from src.ub_oracle.reexec import toolchain_available, ReexecHarness

    oracle = FloatCastOverflowOracle()
    res = oracle.find_divergence(EXAMPLE_UNIT)
    print("find_divergence:", res.verdict, "-", res.detail)
    report = verify_unit(EXAMPLE_UNIT, harness=ReexecHarness(toolchain_available()))
    print("verify_unit:", report.banner())
    return 0 if res.is_divergent else 1


if __name__ == "__main__":
    raise SystemExit(_smoke())
