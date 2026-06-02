"""
Memory-shape divergence oracles (100_STEPS steps 15-16).

Two classes whose C-vs-Rust divergence is about *memory discipline* rather than
arithmetic:

* **out-of-bounds array access** (``array_oob``) — C ``a[i]`` with ``i`` outside
  the array bounds is UB (UBSan traps); the idiomatic Rust ``a[i]`` is bounds-
  checked and panics deterministically (a defined outcome).  Confirmed with the
  ``trap_vs_defined`` harness mode.

* **strict-aliasing violation** (``strict_aliasing``) — C that writes through two
  pointers of incompatible types to the same storage is UB.  No sanitizer traps
  it, but the optimiser *exploits* it: the very same source produces a different
  result at ``-O0`` vs ``-O2 -fstrict-aliasing``.  That two builds of one
  deterministic program disagree is proof the C result is under-determined, while
  the Rust translation has a single defined value.  Confirmed with the
  ``optimizer_exploited`` harness mode.

Both oracles use Z3 to *find* the witnessing input/values (an out-of-bounds
index; a pair of stored values whose aliased read differs from the strict-
aliasing assumption) rather than hard-coding them.
"""

from __future__ import annotations

from typing import Dict

import z3

from ..catalogue import ARRAY_OOB, STRICT_ALIASING, Definedness
from ..plugin import DivergenceOracle, OracleResult, OracleVerdict, register
from ..replay import Counterexample


class ArrayOutOfBoundsOracle(DivergenceOracle):
    """``a[i]`` where ``i`` is outside the bounds of a fixed-size array."""

    divergence_class = ARRAY_OOB.key
    source_lang = "c"
    target_lang = "rust"
    confirmation_mode = "trap_vs_defined"

    def applies_to(self, unit: Dict) -> bool:
        if unit.get("probe") not in (None, self.divergence_class):
            return False
        return unit.get("kind") == "array_index" and int(unit.get("length", 0)) > 0

    def find_divergence(self, unit: Dict) -> OracleResult:
        if not self.applies_to(unit):
            return OracleResult(OracleVerdict.NOT_APPLICABLE, self.divergence_class,
                                detail="unit is not a fixed-size array index")
        length = int(unit["length"])
        var = unit.get("index_var", "i")

        # Find the smallest in-`int`-range index that is out of bounds: i >= len.
        i = z3.BitVec(var, 32)
        opt = z3.Optimize()
        opt.add(i >= length)          # signed comparison (BitVec operators)
        opt.add(i < (1 << 20))        # keep it small/printable
        opt.minimize(i)
        if opt.check() != z3.sat:  # pragma: no cover - always sat
            return OracleResult(OracleVerdict.NO_DIVERGENCE_FOUND, self.divergence_class)
        idx = opt.model()[i].as_long()

        ce = self._build(length, var, idx)
        return OracleResult(OracleVerdict.DIVERGENT, self.divergence_class,
                            counterexample=ce,
                            detail=f"Z3 witness {var}={idx} (>= length {length})")

    def _build(self, length: int, var: str, idx: int) -> Counterexample:
        elems = ", ".join(str(10 + k) for k in range(length))
        rust_elems = ", ".join(str(10 + k) for k in range(length))
        c_src = (
            "#include <stdio.h>\n"
            "#include <stdlib.h>\n"
            f"int a[{length}] = {{{elems}}};\n"
            f"int f(int {var}){{ return a[{var}]; }}\n"
            "int main(int argc, char**argv){\n"
            "    if (argc < 2) return 2;\n"
            f"    int {var} = atoi(argv[1]);\n"
            f"    printf(\"%d\\n\", f({var}));\n"
            "    return 0;\n"
            "}\n"
        )
        rust_src = (
            f"fn f(a: &[i32; {length}], {var}: usize) -> i32 {{ a[{var}] }}\n"
            "fn main(){\n"
            f"    let a: [i32; {length}] = [{rust_elems}];\n"
            f"    let {var}: usize = std::env::args().nth(1).unwrap().parse().unwrap();\n"
            f"    println!(\"{{}}\", f(&a, {var}));\n"
            "}\n"
        )
        return Counterexample(
            divergence_class=self.divergence_class,
            source_lang="c", target_lang="rust",
            inputs={var: idx},
            source_snippet=c_src, target_snippet=rust_src,
            source_definedness=Definedness.UNDEFINED.value,
            divergence_witness=(
                f"C `a[{var}]` with {var}={idx} on a length-{length} array is UB; "
                f"Rust bounds-checks and panics deterministically (defined)."
            ),
            definedness_witness=(
                f"{var}={idx} is a valid integer index value; only the C array "
                f"access is undefined."
            ),
        )


def _find_pun_pair() -> tuple:
    """Z3-find the smallest non-negative pair ``(A, B)`` with ``A`` differing
    from the low 32 bits of ``B`` (so the aliased read ``(int)B`` differs from
    the strict-aliasing assumption ``A``)."""
    a = z3.BitVec("A", 32)
    b = z3.BitVec("B", 64)
    opt = z3.Optimize()
    opt.add(a != z3.Extract(31, 0, b))  # A differs from the low 32 bits of B
    opt.add(a >= 0)
    opt.add(b >= 0)
    opt.minimize(z3.Concat(z3.BitVecVal(0, 32), a) + b)
    if opt.check() != z3.sat:  # pragma: no cover - always sat
        return None
    m = opt.model()
    return m[a].as_long(), m[b].as_long()


_SA_C_SRC = (
    "#include <stdio.h>\n"
    "int f(int *pi, long *pl){{\n"
    "    *pi = {a};\n"
    "    *pl = {b}L;\n"
    "    return *pi;\n"
    "}}\n"
    "int main(void){{\n"
    "    long storage = 0;\n"
    "    int r = f((int*)&storage, &storage);\n"
    "    printf(\"%d\\n\", r);\n"
    "    return 0;\n"
    "}}\n"
)


class _StrictAliasingBase(DivergenceOracle):
    """Type-punned stores through incompatible pointer types (strict aliasing)."""

    divergence_class = STRICT_ALIASING.key
    source_lang = "c"
    confirmation_mode = "optimizer_exploited"

    def applies_to(self, unit: Dict) -> bool:
        if unit.get("probe") not in (None, self.divergence_class):
            return False
        return unit.get("kind") == "type_pun"

    def _target_src(self, a_val: int, b_val: int) -> str:  # pragma: no cover
        raise NotImplementedError

    def find_divergence(self, unit: Dict) -> OracleResult:
        if not self.applies_to(unit):
            return OracleResult(OracleVerdict.NOT_APPLICABLE, self.divergence_class,
                                detail="unit is not a type-pun")
        pair = _find_pun_pair()
        if pair is None:  # pragma: no cover - always sat
            return OracleResult(OracleVerdict.NO_DIVERGENCE_FOUND, self.divergence_class)
        a_val, b_val = pair
        ce = self._build(a_val, b_val)
        return OracleResult(OracleVerdict.DIVERGENT, self.divergence_class,
                            counterexample=ce,
                            detail=f"Z3 witness A={a_val}, B={b_val}")

    def _build(self, a_val: int, b_val: int) -> Counterexample:
        c_src = _SA_C_SRC.format(a=a_val, b=b_val)
        return Counterexample(
            divergence_class=self.divergence_class,
            source_lang="c", target_lang=self.target_lang,
            inputs={},  # the witness values are baked into the source
            source_snippet=c_src, target_snippet=self._target_src(a_val, b_val),
            source_definedness=Definedness.UNDEFINED.value,
            divergence_witness=(
                f"C punning int*/long* over the same storage (A={a_val}, B={b_val}) "
                f"is a strict-aliasing violation: -O0 returns the aliased value "
                f"while -O2 may assume the int is unchanged and return {a_val}. "
                f"The safe {self.target_lang} translation is a single defined value."
            ),
            definedness_witness=(
                "the program reads/writes only valid, in-bounds storage; the "
                "undefined-ness is purely the type-based aliasing rule."
            ),
        )


class StrictAliasingOracle(_StrictAliasingBase):
    """C int*/long* type-pun vs a defined Rust truncation."""

    target_lang = "rust"

    def _target_src(self, a_val: int, b_val: int) -> str:
        # A defined Rust translation: the storage genuinely holds B, so reading
        # the low int is a defined truncation. Deterministic by construction.
        return (
            "fn main(){\n"
            f"    let _a: i32 = {a_val};\n"
            f"    let b: i64 = {b_val};\n"
            "    let r: i32 = b as i32;\n"
            "    println!(\"{}\", r);\n"
            "}\n"
        )


class GoStrictAliasingOracle(_StrictAliasingBase):
    """C int*/long* type-pun vs a defined Go truncation."""

    target_lang = "go"

    def _target_src(self, a_val: int, b_val: int) -> str:
        # Go has no strict-aliasing rewrite; the idiomatic safe port keeps the
        # value in a typed variable and truncates with a defined conversion.
        return (
            "package main\n"
            "import \"fmt\"\n"
            "func main(){\n"
            f"\tvar b int64 = {b_val}\n"
            "\tr := int32(b)\n"
            "\tfmt.Println(r)\n"
            "}\n"
        )


register(ArrayOutOfBoundsOracle())
register(StrictAliasingOracle())
register(GoStrictAliasingOracle())
