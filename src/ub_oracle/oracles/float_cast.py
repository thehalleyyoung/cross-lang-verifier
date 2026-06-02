"""
Float-to-integer out-of-range conversion divergence oracle (100_STEPS step 106).

This is the one genuinely-divergent corner of the C *conversion lattice*. Most
conversions in the lattice agree across mainstream C / Rust / Go toolchains:

    * signed -> unsigned          : C is *defined* (modular, C17 6.3.1.3p2) and
                                    Rust/Go `as` reproduce the same modular value.
    * unsigned -> signed (oob)    : C is *implementation-defined* (6.3.1.3p3) but
                                    on every two's-complement target wraps exactly
                                    like Rust/Go `as`.
    * right-shift of a negative   : C is *implementation-defined* (6.5.7p5) but is
                                    arithmetic on every mainstream target, matching
                                    Rust's signed `>>`.

The corner that *does* diverge is **floating-point -> integer when the rounded
value is outside the destination integer's range**, which is **undefined
behavior** in C (C17 6.3.1.4p1). Real builds bear this out: ``-O0`` yields a
garbage / target-specific value and ``-fsanitize=undefined`` (which includes the
``float-cast-overflow`` check) **traps** deterministically.

The idiomatic *safe* translations never invoke UB here:

    * C    ``(int)x``            (UB when x rounds out of int range; UBSan traps)
    * Rust ``x as i32``          (defined: **saturating** since Rust 1.45)
    * Go   ``int32(x)``          (defined: result is implementation-specified but
                                  not UB and deterministic per build)

So on the same out-of-range input the C program is undefined while the target is
defined and deterministic — confirmed with the harness's ``trap_vs_defined``
mode. The witnessing value is *found* with Z3 (the least-extreme integer-valued
magnitude just past the destination range, honouring any declared range) rather
than hard-coded, exactly like the integer / VLA oracles.
"""

from __future__ import annotations

from typing import Dict, Tuple

import z3

from ..catalogue import FLOAT_CAST_OVERFLOW, Definedness
from ..plugin import DivergenceOracle, OracleResult, OracleVerdict, register
from ..replay import Counterexample

# width -> (C dest integer type, printf format)
_ELT = {
    32: ("int", "%d"),
    64: ("long long", "%lld"),
}
_RUST_ELT = {32: "i32", 64: "i64"}
_GO_ELT = {32: "int32", 64: "int64"}


def _int_bounds(width: int) -> Tuple[int, int]:
    return -(1 << (width - 1)), (1 << (width - 1)) - 1


def _c_cast_program(elt: str, fmt: str, var: str) -> str:
    """A C program that reads a floating value from argv, casts it to the
    destination integer type, and prints the result. The cast is the UB site."""
    return (
        "#include <stdio.h>\n"
        "#include <stdlib.h>\n"
        f"static {elt} g(double {var}){{\n"
        f"    return ({elt}){var};\n"
        "}\n"
        "int main(int argc, char**argv){\n"
        "    if (argc < 2) return 2;\n"
        f"    double {var} = strtod(argv[1], 0);\n"
        f"    printf(\"{fmt}\\n\", g({var}));\n"
        "    return 0;\n"
        "}\n"
    )


def _rust_cast_program(rtype: str, var: str) -> str:
    """The idiomatic safe Rust port: ``as`` is a *saturating* (defined) cast."""
    return (
        f"fn g({var}: f64) -> {rtype} {{\n"
        f"    {var} as {rtype}\n"
        "}\n"
        "fn main(){\n"
        "    let v: Vec<String> = std::env::args().collect();\n"
        f"    let {var}: f64 = v[1].parse().unwrap();\n"
        f"    println!(\"{{}}\", g({var}));\n"
        "}\n"
    )


def _go_cast_program(gtype: str, var: str) -> str:
    """The idiomatic safe Go port: a float->int conversion is defined (the value
    is implementation-specified on overflow, but it never traps)."""
    return (
        "package main\n"
        "import (\n\t\"fmt\"\n\t\"os\"\n\t\"strconv\"\n)\n"
        f"func g({var} float64) {gtype} {{\n"
        f"\treturn {gtype}({var})\n"
        "}\n"
        "func main() {\n"
        f"\t{var}, _ := strconv.ParseFloat(os.Args[1], 64)\n"
        f"\tfmt.Println(g({var}))\n"
        "}\n"
    )


def _find_oob_value(unit: Dict, width: int, var: str) -> Tuple[bool, int]:
    """Z3-find the *least extreme* integer-valued floating input whose rounded
    value lands just past the destination range (typically ``INT_MAX + 1``),
    honouring an optional declared range. The least-extreme witness keeps the
    counterexample clean and exactly reproducible."""
    lo, hi = _int_bounds(width)
    n = z3.Int(var)
    opt = z3.Optimize()
    # just above the representable maximum -> out of range on conversion.
    opt.add(n > hi)
    nr = unit.get("value_range")
    if nr is not None:
        opt.add(n >= int(nr[0]), n <= int(nr[1]))
    opt.minimize(n)  # the out-of-range value closest to the boundary
    if opt.check() != z3.sat:
        return False, 0
    return True, opt.model()[n].as_long()


class _FloatCastBase(DivergenceOracle):
    divergence_class = FLOAT_CAST_OVERFLOW.key
    source_lang = "c"
    confirmation_mode = "trap_vs_defined"

    def applies_to(self, unit: Dict) -> bool:
        if unit.get("probe") not in (None, self.divergence_class):
            return False
        return unit.get("kind") == "float_cast" and unit.get("width", 32) in _ELT


class FloatCastOverflowOracle(_FloatCastBase):
    """C out-of-range float->int cast (UB) vs Rust's defined saturating ``as``."""

    target_lang = "rust"

    def find_divergence(self, unit: Dict) -> OracleResult:
        if not self.applies_to(unit):
            return OracleResult(OracleVerdict.NOT_APPLICABLE, self.divergence_class,
                                detail="unit is not a float->int cast")
        width = unit.get("width", 32)
        var = unit.get("var", "x")
        ok, val = _find_oob_value(unit, width, var)
        if not ok:
            return OracleResult(OracleVerdict.NO_DIVERGENCE_FOUND, self.divergence_class,
                                detail="no out-of-range value in declared range")
        ce = self._build(width, var, val)
        return OracleResult(OracleVerdict.DIVERGENT, self.divergence_class,
                            counterexample=ce,
                            detail=f"Z3 witness {var}={val} (rounds outside the "
                                   f"destination integer range)")

    def _build(self, width: int, var: str, val: int) -> Counterexample:
        elt, fmt = _ELT[width]
        rtype = _RUST_ELT[width]
        c_src = _c_cast_program(elt, fmt, var)
        rust_src = _rust_cast_program(rtype, var)
        return Counterexample(
            divergence_class=self.divergence_class,
            source_lang="c", target_lang="rust",
            inputs={var: val},
            source_snippet=c_src, target_snippet=rust_src,
            source_definedness=Definedness.UNDEFINED.value,
            divergence_witness=(
                f"C `({elt}){var}` with {var}={val}.0 rounds outside the range of "
                f"`{elt}` and is UB (C17 6.3.1.4p1; UBSan `float-cast-overflow` "
                f"traps); the safe Rust port `{var} as {rtype}` is a defined "
                f"saturating cast (a single deterministic value)."
            ),
            definedness_witness=(
                f"{var}={val}.0 is a perfectly valid finite `double`; only the C "
                f"narrowing cast of it to `{elt}` is undefined."
            ),
        )


class GoFloatCastOverflowOracle(_FloatCastBase):
    """C out-of-range float->int cast (UB) vs Go's defined conversion."""

    target_lang = "go"

    def find_divergence(self, unit: Dict) -> OracleResult:
        if not self.applies_to(unit):
            return OracleResult(OracleVerdict.NOT_APPLICABLE, self.divergence_class,
                                detail="unit is not a float->int cast")
        width = unit.get("width", 32)
        var = unit.get("var", "x")
        ok, val = _find_oob_value(unit, width, var)
        if not ok:
            return OracleResult(OracleVerdict.NO_DIVERGENCE_FOUND, self.divergence_class,
                                detail="no out-of-range value in declared range")
        ce = self._build(width, var, val)
        return OracleResult(OracleVerdict.DIVERGENT, self.divergence_class,
                            counterexample=ce,
                            detail=f"Z3 witness {var}={val} (rounds outside the "
                                   f"destination integer range)")

    def _build(self, width: int, var: str, val: int) -> Counterexample:
        elt, fmt = _ELT[width]
        gtype = _GO_ELT[width]
        c_src = _c_cast_program(elt, fmt, var)
        go_src = _go_cast_program(gtype, var)
        return Counterexample(
            divergence_class=self.divergence_class,
            source_lang="c", target_lang="go",
            inputs={var: val},
            source_snippet=c_src, target_snippet=go_src,
            source_definedness=Definedness.UNDEFINED.value,
            divergence_witness=(
                f"C `({elt}){var}` with {var}={val}.0 rounds outside the range of "
                f"`{elt}` and is UB (C17 6.3.1.4p1; UBSan `float-cast-overflow` "
                f"traps); the safe Go port `{gtype}({var})` is a defined "
                f"conversion (implementation-specified but deterministic, never a "
                f"trap)."
            ),
            definedness_witness=(
                f"{var}={val}.0 is a perfectly valid finite `float64`; only the C "
                f"narrowing cast of it to `{elt}` is undefined."
            ),
        )


register(FloatCastOverflowOracle())
register(GoFloatCastOverflowOracle())
