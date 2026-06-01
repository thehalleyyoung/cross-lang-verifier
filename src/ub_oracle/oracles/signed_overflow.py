"""
Signed-overflow divergence oracle (100_STEPS steps 13-14) — the anchor plugin.

Given a translation unit describing ``f(x) = x <op> c`` over a C signed integer
type, this oracle uses Z3 to *find* an input ``x`` for which the C computation
signed-overflows (undefined behavior), while the idiomatic Rust translation is
fully defined.  It then emits a re-executable counterexample whose C/Rust source
is generated with the classic "comparison reveals the overflow" pattern so that
the UB is observably consequential:

    C   : int f(int x){ return x + c > x; }   // optimizer may assume always 1
    Rust: fn f(x:i32)->i32{ (x.wrapping_add(c) > x) as i32 }   // defined

On the witness ``x`` the C program's ``-O0`` and ``-O2`` builds disagree (the UB
is exploited by the optimizer) while Rust returns a single defined value — which
the re-execution harness confirms against real compilers.

The Z3 search (not a hard-coded INT_MAX) is what makes this an *oracle* rather
than a fixture: it generalizes to any constant/op/width and would find the
witness even if a human could not.
"""

from __future__ import annotations

from typing import Dict, Optional

import z3

from ..catalogue import SIGNED_OVERFLOW, Definedness
from ..plugin import DivergenceOracle, OracleResult, OracleVerdict, register
from ..replay import Counterexample

# width -> (C type, Rust type)
_TYPE_MAP = {
    32: ("int", "i32"),
    64: ("long long", "i64"),
}

_SUPPORTED_OPS = {"add", "sub"}


def _signed_bounds(width: int):
    return -(1 << (width - 1)), (1 << (width - 1)) - 1


class SignedOverflowOracle(DivergenceOracle):
    divergence_class = SIGNED_OVERFLOW.key
    source_lang = "c"
    target_lang = "rust"

    def applies_to(self, unit: Dict) -> bool:
        return (
            unit.get("kind") == "binop_const"
            and unit.get("op") in _SUPPORTED_OPS
            and unit.get("signed", True)
            and unit.get("width", 32) in _TYPE_MAP
        )

    def find_divergence(self, unit: Dict) -> OracleResult:
        if not self.applies_to(unit):
            return OracleResult(OracleVerdict.NOT_APPLICABLE, self.divergence_class,
                                detail="unit not a signed binop-with-constant")

        op = unit["op"]
        width = unit.get("width", 32)
        c = int(unit["const"])
        var = unit.get("var", "x")

        x = z3.BitVec(var, width)
        c_bv = z3.BitVecVal(c, width)

        # Signed-overflow predicate for the chosen op.
        if op == "add":
            no_of = z3.And(z3.BVAddNoOverflow(x, c_bv, True),
                           z3.BVAddNoUnderflow(x, c_bv))
        else:  # sub
            no_of = z3.And(z3.BVSubNoOverflow(x, c_bv),
                           z3.BVSubNoUnderflow(x, c_bv, True))
        overflows = z3.Not(no_of)

        solver = z3.Solver()
        solver.add(overflows)
        if solver.check() != z3.sat:
            return OracleResult(OracleVerdict.NO_DIVERGENCE_FOUND, self.divergence_class,
                                detail=f"no overflowing input for x {op} {c} at width {width}")

        model = solver.model()
        raw = model[x].as_long()
        lo, hi = _signed_bounds(width)
        witness = raw if raw <= hi else raw - (1 << width)  # interpret as signed

        ce = self._build_counterexample(op, c, width, var, witness)
        return OracleResult(OracleVerdict.DIVERGENT, self.divergence_class,
                            counterexample=ce,
                            detail=f"Z3 witness {var}={witness}")

    def _build_counterexample(self, op: str, c: int, width: int,
                              var: str, witness: int) -> Counterexample:
        ctype, rtype = _TYPE_MAP[width]
        c_op = "+" if op == "add" else "-"
        cmp = ">" if op == "add" else "<"
        wrap = "wrapping_add" if op == "add" else "wrapping_sub"
        scan = "%d" if width == 32 else "%lld"

        c_src = (
            "#include <stdio.h>\n"
            "#include <stdlib.h>\n"
            f"{ctype} f({ctype} {var}){{ return {var} {c_op} {c} {cmp} {var}; }}\n"
            "int main(int argc, char**argv){\n"
            f"    if (argc < 2) return 2;\n"
            f"    {ctype} {var} = ({ctype})strtoll(argv[1], 0, 10);\n"
            f"    printf(\"%d\\n\", (int)f({var}));\n"
            "    return 0;\n"
            "}\n"
        )
        rust_src = (
            f"fn f({var}:{rtype})->{rtype} {{ (({var}.{wrap}({c}) {cmp} {var}) as {rtype}) }}\n"
            "fn main(){\n"
            f"    let {var}:{rtype} = std::env::args().nth(1).unwrap().parse().unwrap();\n"
            f"    println!(\"{{}}\", f({var}));\n"
            "}\n"
        )
        return Counterexample(
            divergence_class=self.divergence_class,
            source_lang="c",
            target_lang="rust",
            inputs={var: witness},
            source_snippet=c_src,
            target_snippet=rust_src,
            source_definedness=Definedness.UNDEFINED.value,
            divergence_witness=(
                f"C signed {op} overflow at {var}={witness} (UB); the optimizer may "
                f"assume `{var} {c_op} {c} {cmp} {var}` is always true, whereas the "
                f"wrapping Rust translation evaluates it on the real wrapped value."
            ),
            definedness_witness=(
                f"{var}={witness} is a valid {width}-bit signed value, so the input "
                f"itself is fully defined; only the C arithmetic is undefined."
            ),
        )


# Register the anchor oracle instance.
register(SignedOverflowOracle())
