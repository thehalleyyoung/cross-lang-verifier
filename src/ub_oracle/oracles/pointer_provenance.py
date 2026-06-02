"""
Pointer-provenance / pointer-arithmetic-overflow divergence oracle
(100_STEPS step 102).

Forming a pointer by adding an integer offset to an array such that the result
lands more than one element past the end — and, in the limit, *overflows the
address space* — is **undefined behavior** in C (C17 6.5.6p8: a pointer may only
range over the elements of the array object plus the one-past-the-end position;
the computed pointer's *provenance* is the original array object).  In practice
``-fsanitize=undefined`` includes the ``pointer-overflow`` check, which **traps**
when ``base + n*sizeof(T)`` overflows; at ``-O0`` the program forms a wild
pointer and reads/derives from garbage.  This is the pointer-provenance hazard
behind ``intptr_t`` round-trips and one-past-the-end arithmetic.

The idiomatic *safe* translation never forms an out-of-provenance pointer.  It
keeps an **index** and accesses through a **checked** operation, so the same
offset becomes a defined, deterministic outcome:

    * C    ``int *p = a; p = p + n; use(p - a);``  (UB; UBSan pointer-overflow traps)
    * Rust ``a.get(n as usize).copied().unwrap_or(SENTINEL)``  (defined value)
    * Go   ``if n < 0 || n >= len(a) {return SENTINEL}; a[n]``  (defined value)

This is confirmed with the harness's ``trap_vs_defined`` mode: the UBSan build
traps on the same concrete input on which the target produces a defined outcome.
The witnessing offset is *found* with Z3 — the least ``n`` whose byte offset
``n*sizeof(T)`` is guaranteed to overflow a 64-bit address space irrespective of
the (run-time, ASLR-randomised) base address, i.e. ``n*sizeof(T) >= 2**64`` —
rather than hard-coded, so the search honours an AI pre-pass's range constraints
exactly like the integer oracles.
"""

from __future__ import annotations

from typing import Dict, Tuple

import z3

from ..catalogue import POINTER_PROVENANCE, Definedness
from ..plugin import DivergenceOracle, OracleResult, OracleVerdict, register
from ..replay import Counterexample

# width -> (C element type, element size in bytes, printf format, strtoll fn)
_ELT = {
    32: ("int", 4, "%ld", "strtoll"),
    64: ("long long", 8, "%ld", "strtoll"),
}
_RUST_ELT = {32: "i32", 64: "i64"}
_GO_ELT = {32: "int32", 64: "int64"}

_ADDR_BITS = 64  # the offset that provably overflows any 64-bit address space


def _c_provenance_program(elt: str, fmt: str, scan: str, var: str) -> str:
    """A C program that forms ``a + n`` (out-of-provenance for large ``n``) and
    *uses* the resulting pointer's value, so it cannot be optimised away."""
    return (
        "#include <stdio.h>\n"
        "#include <stdlib.h>\n"
        f"static long g(long {var}){{\n"
        f"    {elt} a[4] = {{10, 20, 30, 40}};\n"
        f"    {elt} *p = a;\n"
        f"    p = p + {var};          /* UB: pointer arithmetic out of provenance */\n"
        "    return (long)(p - a);\n"
        "}\n"
        "int main(int argc, char**argv){\n"
        "    if (argc < 2) return 2;\n"
        f"    long {var} = {scan}(argv[1], 0, 10);\n"
        f"    printf(\"{fmt}\\n\", g({var}));\n"
        "    return 0;\n"
        "}\n"
    )


def _rust_provenance_program(rtype: str, var: str) -> str:
    """The idiomatic safe Rust port: a *checked* index, never a raw pointer."""
    return (
        f"fn g({var}: i64) -> i64 {{\n"
        f"    let a: [{rtype}; 4] = [10, 20, 30, 40];\n"
        f"    if {var} < 0 {{ return -1; }}\n"
        f"    let i = {var} as usize;\n"
        f"    *a.get(i).unwrap_or(&-1) as i64\n"
        "}\n"
        "fn main(){\n"
        "    let v: Vec<String> = std::env::args().collect();\n"
        f"    let {var}: i64 = v[1].parse().unwrap();\n"
        f"    println!(\"{{}}\", g({var}));\n"
        "}\n"
    )


def _go_provenance_program(gtype: str, var: str) -> str:
    """The idiomatic safe Go port: a bounds-checked index access."""
    return (
        "package main\n"
        "import (\n\t\"fmt\"\n\t\"os\"\n\t\"strconv\"\n)\n"
        f"func g({var} int64) int64 {{\n"
        f"\ta := []{gtype}{{10, 20, 30, 40}}\n"
        f"\tif {var} < 0 || {var} >= int64(len(a)) {{ return -1 }}\n"
        f"\treturn int64(a[{var}])\n"
        "}\n"
        "func main() {\n"
        f"\t{var}, _ := strconv.ParseInt(os.Args[1], 10, 64)\n"
        f"\tfmt.Println(g({var}))\n"
        "}\n"
    )


def _find_overflowing_offset(unit: Dict, elem_size: int, var: str) -> Tuple[bool, int]:
    """Z3-find the *least* positive offset ``n`` whose byte displacement
    ``n*elem_size`` is guaranteed to overflow a 64-bit address space regardless
    of the run-time base address (``n*elem_size >= 2**64``). Minimising keeps the
    witness the cleanest reproducible value (e.g. ``2**62`` for 4-byte ints)."""
    n = z3.Int(var)
    opt = z3.Optimize()
    opt.add(n > 0)
    opt.add(n * elem_size >= (1 << _ADDR_BITS))
    nr = unit.get("offset_range")
    if nr is not None:
        opt.add(n >= int(nr[0]), n <= int(nr[1]))
    opt.minimize(n)
    if opt.check() != z3.sat:
        return False, 0
    return True, opt.model()[n].as_long()


class _ProvenanceBase(DivergenceOracle):
    divergence_class = POINTER_PROVENANCE.key
    source_lang = "c"
    confirmation_mode = "trap_vs_defined"

    def applies_to(self, unit: Dict) -> bool:
        if unit.get("probe") not in (None, self.divergence_class):
            return False
        return unit.get("kind") == "pointer_offset" and unit.get("width", 32) in _ELT


class PointerProvenanceOracle(_ProvenanceBase):
    """C out-of-provenance pointer arithmetic vs Rust's checked index."""

    target_lang = "rust"

    def find_divergence(self, unit: Dict) -> OracleResult:
        if not self.applies_to(unit):
            return OracleResult(OracleVerdict.NOT_APPLICABLE, self.divergence_class,
                                detail="unit is not a pointer-offset")
        width = unit.get("width", 32)
        var = unit.get("var", "n")
        _, esize, _, _ = _ELT[width]
        ok, n_val = _find_overflowing_offset(unit, esize, var)
        if not ok:
            return OracleResult(OracleVerdict.NO_DIVERGENCE_FOUND, self.divergence_class,
                                detail="no overflowing offset in declared range")
        ce = self._build(width, var, n_val)
        return OracleResult(OracleVerdict.DIVERGENT, self.divergence_class,
                            counterexample=ce,
                            detail=f"Z3 witness {var}={n_val} (offset overflows provenance)")

    def _build(self, width: int, var: str, n_val: int) -> Counterexample:
        elt, esize, fmt, scan = _ELT[width]
        rtype = _RUST_ELT[width]
        c_src = _c_provenance_program(elt, fmt, scan, var)
        rust_src = _rust_provenance_program(rtype, var)
        return Counterexample(
            divergence_class=self.divergence_class,
            source_lang="c", target_lang="rust",
            inputs={var: n_val},
            source_snippet=c_src, target_snippet=rust_src,
            source_definedness=Definedness.UNDEFINED.value,
            divergence_witness=(
                f"C `p = a + {var}` with {var}={n_val} forms a pointer whose byte "
                f"offset {var}*{esize} overflows the address space (UB, C17 6.5.6p8; "
                f"UBSan `pointer-overflow` traps, `-O0` derives from a wild pointer); "
                f"the safe Rust port keeps an index and accesses through `a.get(i)`, "
                f"a checked operation that yields a deterministic, defined value."
            ),
            definedness_witness=(
                f"{var}={n_val} is a perfectly valid integer input; only the C "
                f"pointer arithmetic is out of the array's provenance on it."
            ),
        )


class GoPointerProvenanceOracle(_ProvenanceBase):
    """C out-of-provenance pointer arithmetic vs Go's bounds-checked index."""

    target_lang = "go"

    def find_divergence(self, unit: Dict) -> OracleResult:
        if not self.applies_to(unit):
            return OracleResult(OracleVerdict.NOT_APPLICABLE, self.divergence_class,
                                detail="unit is not a pointer-offset")
        width = unit.get("width", 32)
        var = unit.get("var", "n")
        _, esize, _, _ = _ELT[width]
        ok, n_val = _find_overflowing_offset(unit, esize, var)
        if not ok:
            return OracleResult(OracleVerdict.NO_DIVERGENCE_FOUND, self.divergence_class,
                                detail="no overflowing offset in declared range")
        ce = self._build(width, var, n_val)
        return OracleResult(OracleVerdict.DIVERGENT, self.divergence_class,
                            counterexample=ce,
                            detail=f"Z3 witness {var}={n_val} (offset overflows provenance)")

    def _build(self, width: int, var: str, n_val: int) -> Counterexample:
        elt, esize, fmt, scan = _ELT[width]
        gtype = _GO_ELT[width]
        c_src = _c_provenance_program(elt, fmt, scan, var)
        go_src = _go_provenance_program(gtype, var)
        return Counterexample(
            divergence_class=self.divergence_class,
            source_lang="c", target_lang="go",
            inputs={var: n_val},
            source_snippet=c_src, target_snippet=go_src,
            source_definedness=Definedness.UNDEFINED.value,
            divergence_witness=(
                f"C `p = a + {var}` with {var}={n_val} forms a pointer whose byte "
                f"offset {var}*{esize} overflows the address space (UB, C17 6.5.6p8; "
                f"UBSan `pointer-overflow` traps); the safe Go port bounds-checks the "
                f"index and returns a deterministic, defined value."
            ),
            definedness_witness=(
                f"{var}={n_val} is a perfectly valid integer input; only the C "
                f"pointer arithmetic is out of the array's provenance on it."
            ),
        )


register(PointerProvenanceOracle())
register(GoPointerProvenanceOracle())
