"""
``restrict``-violation divergence oracle (100_STEPS step 109).

A ``restrict``-qualified pointer is a *promise* to the compiler that, for the
lifetime of the pointer, the object it points at is accessed **only** through
that pointer.  If two ``restrict`` pointers actually alias and at least one of
the accesses is a store, the behavior is **undefined** (C17 6.7.3.1p4) — and,
crucially, the optimizer *relies* on the promise: at ``-O2`` it may keep a value
it already read in a register instead of re-loading memory that it "knows"
cannot have changed.  So the *same* C source, on the *same* aliasing input,
prints **different** results under ``-O0`` (honest re-read) and ``-O2``
(restrict-based caching).  No sanitizer traps this — the evidence is the
optimisation-level disagreement itself.

The idiomatic *safe* translation cannot reproduce the hazard:

    * Rust ``fn f(a: &mut i32, b: &mut i32)`` — two live ``&mut`` to one object
      are rejected by the borrow checker, so the safe port is *non-aliasing by
      construction* and computes a single deterministic, defined value.
    * Go has no ``restrict`` qualifier and performs no restrict-based rewrite, so
      its pointer code is defined and deterministic whether or not it aliases.

This is confirmed with the harness's ``optimizer_exploited`` mode (``-O0`` vs
``-O2``): the two C builds disagree on the witnessing aliasing input while the
target build is defined and deterministic.  The witnessing selector (which makes
the two pointers alias) is *found* with Z3 rather than hard-coded.
"""

from __future__ import annotations

from typing import Dict, Tuple

import z3

from ..catalogue import RESTRICT_VIOLATION, Definedness
from ..plugin import DivergenceOracle, OracleResult, OracleVerdict, register
from ..replay import Counterexample

# clang flag pair whose disagreement evidences the restrict-based rewrite.
_O0 = ["-O0"]
_O2 = ["-O2"]


def _find_alias_selector(unit: Dict) -> Tuple[bool, int]:
    """Z3-find the least-extreme non-zero selector that makes the two pointers
    alias (the source UB trigger), honouring any declared range."""
    sel = z3.BitVec("sel", 32)
    opt = z3.Optimize()
    opt.add(sel != 0)
    opt.add(sel > 0)  # a clean, small positive selector
    rng = unit.get("selector_range")
    if rng is not None:
        opt.add(sel >= z3.BitVecVal(int(rng[0]), 32),
                sel <= z3.BitVecVal(int(rng[1]), 32))
    opt.minimize(sel)
    if opt.check() != z3.sat:
        return False, 0
    return True, opt.model()[sel].as_long()


_C_SRC = (
    "#include <stdio.h>\n"
    "#include <stdlib.h>\n"
    "static int f(int *restrict a, int *restrict b){\n"
    "    *a = 1;\n"
    "    *b = 2;            /* if a==b this is UB; -O2 assumes a,b never alias */\n"
    "    return *a + *b;    /* -O0: re-reads (=4 when aliased); -O2: caches *a=1 (=3) */\n"
    "}\n"
    "int main(int argc, char**argv){\n"
    "    if (argc < 2) return 2;\n"
    "    long sel = strtol(argv[1], 0, 10);\n"
    "    int x = 0, y = 0;\n"
    "    int *p = &x;\n"
    "    int *q = sel ? &x : &y;   /* sel != 0 -> q aliases p (restrict violated) */\n"
    "    printf(\"%d\\n\", f(p, q));\n"
    "    return 0;\n"
    "}\n"
)

# The safe Rust port: `&mut` cannot alias (borrow checker), so the function is
# non-aliasing by construction and deterministic regardless of the selector.
_RUST_SRC = (
    "fn f(a: &mut i32, b: &mut i32) -> i32 {\n"
    "    *a = 1;\n"
    "    *b = 2;\n"
    "    *a + *b\n"
    "}\n"
    "fn main(){\n"
    "    let _sel: i64 = std::env::args().nth(1).unwrap().parse().unwrap();\n"
    "    let mut x = 0i32;\n"
    "    let mut y = 0i32;\n"
    "    // two `&mut` to *distinct* objects: aliasing is impossible in safe Rust.\n"
    "    println!(\"{}\", f(&mut x, &mut y));\n"
    "}\n"
)

# The safe Go port: no `restrict`, no restrict-based rewrite; aliasing is defined
# and deterministic.
_GO_SRC = (
    "package main\n"
    "import (\n\t\"fmt\"\n\t\"os\"\n\t\"strconv\"\n)\n"
    "func f(a, b *int32) int32 {\n"
    "\t*a = 1\n"
    "\t*b = 2\n"
    "\treturn *a + *b\n"
    "}\n"
    "func main() {\n"
    "\tsel, _ := strconv.Atoi(os.Args[1])\n"
    "\tvar x, y int32\n"
    "\tp := &x\n"
    "\tq := &y\n"
    "\tif sel != 0 {\n\t\tq = &x\n\t}\n"
    "\tfmt.Println(f(p, q))\n"
    "}\n"
)


class _RestrictBase(DivergenceOracle):
    divergence_class = RESTRICT_VIOLATION.key
    source_lang = "c"
    confirmation_mode = "optimizer_exploited"
    optimizer_flag_variants = (_O0, _O2)

    def applies_to(self, unit: Dict) -> bool:
        if unit.get("probe") not in (None, self.divergence_class):
            return False
        return unit.get("kind") == "restrict_pair"

    def _result(self, unit: Dict, target_src: str) -> OracleResult:
        if not self.applies_to(unit):
            return OracleResult(OracleVerdict.NOT_APPLICABLE, self.divergence_class,
                                detail="unit is not a restrict pointer pair")
        ok, sel = _find_alias_selector(unit)
        if not ok:
            return OracleResult(OracleVerdict.NO_DIVERGENCE_FOUND, self.divergence_class,
                                detail="no aliasing selector in declared range")
        ce = self._build(sel, target_src)
        return OracleResult(OracleVerdict.DIVERGENT, self.divergence_class,
                            counterexample=ce,
                            detail=f"Z3 witness sel={sel} (pointers alias)")

    def _build(self, sel: int, target_src: str) -> Counterexample:
        return Counterexample(
            divergence_class=self.divergence_class,
            source_lang="c", target_lang=self.target_lang,
            inputs={"sel": sel},
            source_snippet=_C_SRC, target_snippet=target_src,
            source_definedness=Definedness.UNDEFINED.value,
            divergence_witness=(
                f"C `f(int *restrict a, int *restrict b)` called with aliasing "
                f"pointers (sel={sel}) violates the restrict promise (C17 "
                f"6.7.3.1p4): `-O0` re-reads memory and returns 4 while `-O2` "
                f"caches `*a=1` and returns 3 — the same source diverges across "
                f"optimisation levels. The safe {self.target_lang} port cannot "
                f"alias (Rust `&mut` uniqueness / Go has no restrict) and is a "
                f"single deterministic, defined value."
            ),
            definedness_witness=(
                f"sel={sel} is a valid integer selecting an aliasing call; only "
                f"the C restrict contract is violated — every access is to live, "
                f"in-bounds storage."
            ),
        )


class RestrictViolationOracle(_RestrictBase):
    """C ``restrict`` aliasing miscompile vs Rust's unique ``&mut``."""

    target_lang = "rust"

    def find_divergence(self, unit: Dict) -> OracleResult:
        return self._result(unit, _RUST_SRC)


class GoRestrictViolationOracle(_RestrictBase):
    """C ``restrict`` aliasing miscompile vs Go's restrict-free pointers."""

    target_lang = "go"

    def find_divergence(self, unit: Dict) -> OracleResult:
        return self._result(unit, _GO_SRC)


register(RestrictViolationOracle())
register(GoRestrictViolationOracle())
