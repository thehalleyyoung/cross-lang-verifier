"""
Sequence-point / unsequenced-modification divergence oracle (100_STEPS step 105).

The C expression ``i++ + i`` modifies ``i`` and reads ``i`` again without an
intervening sequencing edge.  C17 6.5p2 makes that undefined behavior: a compiler
does not have to choose either left-to-right or right-to-left meaning.  Rust, by
contrast, evaluates the explicit translation in a single left-to-right order.

UBSan does not trap this class at runtime, so confirmation is static but still
grounded in the real compiler: clang's ``-Wunsequenced`` must diagnose the exact
source program, and the target binary must compile and run deterministically on
the same witness.
"""

from __future__ import annotations

from typing import Dict, Optional

import z3

from ..catalogue import EVAL_ORDER, Definedness
from ..plugin import DivergenceOracle, OracleResult, OracleVerdict, register
from ..replay import Counterexample

_INT_MIN = -(1 << 31)
_INT_MAX = (1 << 31) - 1
_SUPPORTED_PATTERNS = {"postinc_read_add"}


def _wrap_i32(n: int) -> int:
    n &= (1 << 32) - 1
    return n if n <= _INT_MAX else n - (1 << 32)


def _find_witness(unit: Dict) -> Optional[int]:
    """Find a valid 32-bit ``i`` whose order concretizations differ.

    For ``i++ + i``, a left-to-right concretization yields ``i + (i + 1)`` while a
    read-before-increment concretization yields ``i + i``.  They differ for every
    32-bit input; Z3 is used here to honour optional declared input ranges and to
    pick the smallest-magnitude witness inside that domain.
    """
    lo, hi = _INT_MIN, _INT_MAX
    rng = unit.get("i_range")
    if rng is not None:
        lo, hi = max(lo, int(rng[0])), min(hi, int(rng[1]))
    if lo > hi:
        return None

    i = z3.Int(str(unit.get("var", "i")))
    opt = z3.Optimize()
    opt.add(i >= lo, i <= hi)
    opt.add(2 * i + 1 != 2 * i)
    opt.minimize(z3.If(i >= 0, i, -i))
    opt.minimize(i)
    if opt.check() != z3.sat:
        return None
    return opt.model().eval(i, model_completion=True).as_long()


_C_SRC = (
    "#include <stdio.h>\n"
    "#include <stdlib.h>\n"
    "static int f(int i){ return i++ + i; }\n"
    "int main(int argc, char** argv){\n"
    "    if (argc < 2) return 2;\n"
    "    int i = (int)strtol(argv[1], 0, 10);\n"
    "    printf(\"%d\\n\", f(i));\n"
    "    return 0;\n"
    "}\n"
)

_RUST_SRC = (
    "fn f(mut i: i32) -> i32 {\n"
    "    let left = i;\n"
    "    i = i.wrapping_add(1);\n"
    "    let right = i;\n"
    "    left.wrapping_add(right)\n"
    "}\n"
    "fn main(){\n"
    "    let i: i32 = std::env::args().nth(1).unwrap().parse().unwrap();\n"
    "    println!(\"{}\", f(i));\n"
    "}\n"
)


class SequencePointOracle(DivergenceOracle):
    """C unsequenced read/write UB vs Rust's explicit left-to-right order."""

    divergence_class = EVAL_ORDER.key
    source_lang = "c"
    target_lang = "rust"
    confirmation_mode = "static_ub_vs_defined"

    def applies_to(self, unit: Dict) -> bool:
        if unit.get("probe") not in (None, self.divergence_class):
            return False
        return (
            unit.get("kind") in {"unsequenced", "unsequenced_modify"}
            and unit.get("pattern", "postinc_read_add") in _SUPPORTED_PATTERNS
        )

    def find_divergence(self, unit: Dict) -> OracleResult:
        if not self.applies_to(unit):
            return OracleResult(OracleVerdict.NOT_APPLICABLE, self.divergence_class,
                                detail="unit is not a supported unsequenced pattern")
        witness = _find_witness(unit)
        if witness is None:
            return OracleResult(OracleVerdict.NO_DIVERGENCE_FOUND,
                                self.divergence_class,
                                detail="no 32-bit witness in declared i_range")
        ce = self._build(witness)
        return OracleResult(
            OracleVerdict.DIVERGENT,
            self.divergence_class,
            counterexample=ce,
            detail=f"Z3 witness i={witness}; clang -Wunsequenced proves source UB",
        )

    def _build(self, witness: int) -> Counterexample:
        left_to_right = _wrap_i32(witness + _wrap_i32(witness + 1))
        read_first = _wrap_i32(witness + witness)
        return Counterexample(
            divergence_class=self.divergence_class,
            source_lang="c",
            target_lang="rust",
            inputs={"i": witness},
            source_snippet=_C_SRC,
            target_snippet=_RUST_SRC,
            source_definedness=Definedness.UNDEFINED.value,
            divergence_witness=(
                f"C `i++ + i` at i={witness} modifies and reads `i` without a "
                f"sequencing edge (C17 6.5p2), so the source has undefined "
                f"behavior; clang `-Wunsequenced` diagnoses the real translation "
                f"unit. The Rust port commits to the left-to-right order and "
                f"returns {left_to_right}, while a read-before-increment "
                f"concretization would return {read_first}."
            ),
            definedness_witness=(
                f"i={witness} is a valid `int`/`i32` input, and the Rust program "
                f"uses explicit `wrapping_add`, so the target has one defined, "
                f"deterministic outcome; only the C sequencing rule is violated."
            ),
        )


register(SequencePointOracle())
