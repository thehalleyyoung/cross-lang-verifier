"""
Variable-length-array (VLA) bound divergence oracle (100_STEPS step 110).

A C variable-length array ``int a[n];`` has **undefined behavior** when the size
expression ``n`` is not a positive value (C17 6.7.6.2p5).  In practice the
optimiser is free to do anything: at ``-O0`` clang typically reserves a wild,
negatively-sized frame and the program *segfaults*; under
``-fsanitize=undefined`` (which includes the ``vla-bound`` check) it **traps**
deterministically.  ``alloca`` with a negative/huge size is the same hazard.

The idiomatic *safe* translations never have a VLA: a dynamically-sized buffer
becomes a heap vector whose length comes from a **checked** conversion of a
signed input.  A negative bound is therefore not undefined — it is a defined,
deterministic **panic**:

    * C   ``int a[n];``                       (UB when n <= 0; UBSan traps)
    * Rust ``if n < 0 {panic} let a = vec![..; n as usize];``  (defined panic)
    * Go  ``a := make([]int64, n)``            (defined ``makeslice`` panic)

This is confirmed with the harness's ``trap_vs_defined`` mode: the UBSan build
traps on the same concrete input on which the target produces a defined outcome.
The witnessing bound is *found* with Z3 (the largest — i.e. least extreme —
negative bound consistent with any declared range) rather than hard-coded, so
the search honours an AI pre-pass's constraints just like the integer oracles.
"""

from __future__ import annotations

from typing import Dict, Tuple

import z3

from ..catalogue import VLA_BOUND, Definedness
from ..plugin import DivergenceOracle, OracleResult, OracleVerdict, register
from ..replay import Counterexample

# width -> (C element type, printf format, strtol fn)
_ELT = {
    32: ("int", "%d", "strtol"),
    64: ("long long", "%lld", "strtoll"),
}
_GO_ELT = {32: "int32", 64: "int64"}


def _c_vla_program(elt: str, fmt: str, scan: str, var: str) -> str:
    """A C program whose hot function declares a VLA of size ``var`` read from
    argv, *uses* every element (so the array cannot be optimised away), and
    prints a checksum."""
    return (
        "#include <stdio.h>\n"
        "#include <stdlib.h>\n"
        f"static {elt} g(int {var}){{\n"
        f"    {elt} a[{var}];\n"
        f"    for (int i = 0; i < {var}; i++) a[i] = ({elt})(i + 1);\n"
        f"    {elt} s = 0;\n"
        f"    for (int i = 0; i < {var}; i++) s += a[i];\n"
        "    return s;\n"
        "}\n"
        "int main(int argc, char**argv){\n"
        "    if (argc < 2) return 2;\n"
        f"    long {var} = {scan}(argv[1], 0, 10);\n"
        f"    printf(\"{fmt}\\n\", g((int){var}));\n"
        "    return 0;\n"
        "}\n"
    )


def _rust_vla_program(rtype: str, var: str) -> str:
    """The idiomatic safe Rust port: a heap vector whose length is a *checked*
    conversion of the signed bound, so a negative bound is a defined panic."""
    return (
        f"fn g({var}: i64) -> {rtype} {{\n"
        f"    if {var} < 0 {{ panic!(\"VLA bound must be non-negative\"); }}\n"
        f"    let len = {var} as usize;\n"
        f"    let mut a = vec![0 as {rtype}; len];\n"
        f"    for i in 0..len {{ a[i] = (i as {rtype}) + 1; }}\n"
        f"    let mut s: {rtype} = 0;\n"
        f"    for i in 0..len {{ s += a[i]; }}\n"
        "    s\n"
        "}\n"
        "fn main(){\n"
        "    let v: Vec<String> = std::env::args().collect();\n"
        f"    let {var}: i64 = v[1].parse().unwrap();\n"
        f"    println!(\"{{}}\", g({var}));\n"
        "}\n"
    )


def _go_vla_program(gtype: str, var: str) -> str:
    """The idiomatic safe Go port: ``make`` rejects a negative length with a
    deterministic ``makeslice`` panic."""
    return (
        "package main\n"
        "import (\n\t\"fmt\"\n\t\"os\"\n\t\"strconv\"\n)\n"
        f"func g({var} int) {gtype} {{\n"
        f"\ta := make([]{gtype}, {var})\n"
        f"\tfor i := 0; i < {var}; i++ {{ a[i] = {gtype}(i) + 1 }}\n"
        f"\tvar s {gtype}\n"
        f"\tfor i := 0; i < {var}; i++ {{ s += a[i] }}\n"
        "\treturn s\n"
        "}\n"
        "func main() {\n"
        f"\t{var}, _ := strconv.Atoi(os.Args[1])\n"
        f"\tfmt.Println(g({var}))\n"
        "}\n"
    )


def _find_negative_bound(unit: Dict, var: str) -> Tuple[bool, int]:
    """Z3-find the *least extreme* non-positive VLA bound (closest to 0, i.e.
    typically -1) consistent with any declared range. Returning the least
    extreme value keeps the witness clean and reproducible."""
    n = z3.BitVec(var, 32)
    opt = z3.Optimize()
    opt.add(n < 0)
    nr = unit.get("bound_range")
    if nr is not None:
        opt.add(n >= z3.BitVecVal(int(nr[0]), 32),
                n <= z3.BitVecVal(int(nr[1]), 32))
    opt.maximize(n)  # the negative value closest to zero
    if opt.check() != z3.sat:
        return False, 0
    raw = opt.model()[n].as_long()
    signed = raw if raw < (1 << 31) else raw - (1 << 32)
    return True, signed


class _VlaBoundBase(DivergenceOracle):
    divergence_class = VLA_BOUND.key
    source_lang = "c"
    confirmation_mode = "trap_vs_defined"

    def applies_to(self, unit: Dict) -> bool:
        if unit.get("probe") not in (None, self.divergence_class):
            return False
        return unit.get("kind") == "vla" and unit.get("width", 32) in _ELT


class VlaBoundOracle(_VlaBoundBase):
    """C VLA with a non-positive bound vs Rust's checked-length heap vector."""

    target_lang = "rust"

    def find_divergence(self, unit: Dict) -> OracleResult:
        if not self.applies_to(unit):
            return OracleResult(OracleVerdict.NOT_APPLICABLE, self.divergence_class,
                                detail="unit is not a VLA")
        width = unit.get("width", 32)
        var = unit.get("var", "n")
        ok, n_val = _find_negative_bound(unit, var)
        if not ok:
            return OracleResult(OracleVerdict.NO_DIVERGENCE_FOUND, self.divergence_class,
                                detail="no non-positive bound in declared range")
        ce = self._build(width, var, n_val)
        return OracleResult(OracleVerdict.DIVERGENT, self.divergence_class,
                            counterexample=ce,
                            detail=f"Z3 witness {var}={n_val} (non-positive VLA bound)")

    def _build(self, width: int, var: str, n_val: int) -> Counterexample:
        elt, fmt, scan = _ELT[width]
        rtype = {32: "i32", 64: "i64"}[width]
        c_src = _c_vla_program(elt, fmt, scan, var)
        rust_src = _rust_vla_program(rtype, var)
        return Counterexample(
            divergence_class=self.divergence_class,
            source_lang="c", target_lang="rust",
            inputs={var: n_val},
            source_snippet=c_src, target_snippet=rust_src,
            source_definedness=Definedness.UNDEFINED.value,
            divergence_witness=(
                f"C `{elt} a[{var}]` with {var}={n_val} <= 0 is UB (C17 6.7.6.2p5; "
                f"UBSan `vla-bound` traps, `-O0` segfaults); the safe Rust port "
                f"sizes a heap `Vec` from a checked conversion and panics "
                f"deterministically (a defined outcome)."
            ),
            definedness_witness=(
                f"{var}={n_val} is a perfectly valid integer input; only the C "
                f"variable-length-array declaration is undefined on it."
            ),
        )


class GoVlaBoundOracle(_VlaBoundBase):
    """C VLA with a non-positive bound vs Go's checked ``make`` length."""

    target_lang = "go"

    def find_divergence(self, unit: Dict) -> OracleResult:
        if not self.applies_to(unit):
            return OracleResult(OracleVerdict.NOT_APPLICABLE, self.divergence_class,
                                detail="unit is not a VLA")
        width = unit.get("width", 32)
        var = unit.get("var", "n")
        ok, n_val = _find_negative_bound(unit, var)
        if not ok:
            return OracleResult(OracleVerdict.NO_DIVERGENCE_FOUND, self.divergence_class,
                                detail="no non-positive bound in declared range")
        ce = self._build(width, var, n_val)
        return OracleResult(OracleVerdict.DIVERGENT, self.divergence_class,
                            counterexample=ce,
                            detail=f"Z3 witness {var}={n_val} (non-positive VLA bound)")

    def _build(self, width: int, var: str, n_val: int) -> Counterexample:
        elt, fmt, scan = _ELT[width]
        gtype = _GO_ELT[width]
        c_src = _c_vla_program(elt, fmt, scan, var)
        go_src = _go_vla_program(gtype, var)
        return Counterexample(
            divergence_class=self.divergence_class,
            source_lang="c", target_lang="go",
            inputs={var: n_val},
            source_snippet=c_src, target_snippet=go_src,
            source_definedness=Definedness.UNDEFINED.value,
            divergence_witness=(
                f"C `{elt} a[{var}]` with {var}={n_val} <= 0 is UB (C17 6.7.6.2p5; "
                f"UBSan `vla-bound` traps); the safe Go port `make([]{gtype}, {var})` "
                f"rejects a negative length with a deterministic `makeslice` panic."
            ),
            definedness_witness=(
                f"{var}={n_val} is a perfectly valid integer input; only the C "
                f"variable-length-array declaration is undefined on it."
            ),
        )


register(VlaBoundOracle())
register(GoVlaBoundOracle())
