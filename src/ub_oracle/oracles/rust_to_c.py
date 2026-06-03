"""
Rust -> C reverse-pair divergence oracle (100_STEPS step 118).

Most built-in pairs start from C UB and show how a safe target gives a defined
outcome. This oracle proves the opposite migration direction matters too:
idiomatic Rust may rely on a *defined panic* that an FFI-oriented C lowering turns
into undefined behaviour. The concrete witness is the unique signed division /
remainder overflow pair, ``INT_MIN`` op ``-1``:

* Rust guarantees that ``i32::MIN / -1`` and ``i32::MIN % -1`` panic
  deterministically. The abort is a language-defined observable outcome.
* C says the same operation is undefined because the quotient is not
  representable in the result type; UBSan traps on the real translated binary.

The witness is found with Z3 rather than hard-coded, and confirmation compiles
the Rust source plus the C target and requires the Rust side to be deterministic
and the C target to trap under UBSan.
"""

from __future__ import annotations

from typing import Dict

import z3

from ..catalogue import INT_MIN_DIV_NEG1, Definedness
from ..plugin import DivergenceOracle, OracleResult, OracleVerdict, register
from ..replay import Counterexample


_TYPE_MAP = {
    32: ("i32", "int32_t", "strtol", "PRId32"),
    64: ("i64", "int64_t", "strtoll", "PRId64"),
}


def _as_signed(raw: int, width: int) -> int:
    hi = (1 << (width - 1)) - 1
    return raw if raw <= hi else raw - (1 << width)


def _rust_src(rtype: str, op: str, avar: str, bvar: str) -> str:
    return (
        f"fn f({avar}: {rtype}, {bvar}: {rtype}) -> {rtype} {{ {avar} {op} {bvar} }}\n"
        "fn main() {\n"
        "    let args: Vec<String> = std::env::args().collect();\n"
        f"    let {avar}: {rtype} = args[1].parse().unwrap();\n"
        f"    let {bvar}: {rtype} = args[2].parse().unwrap();\n"
        f"    println!(\"{{}}\", f({avar}, {bvar}));\n"
        "}\n"
    )


def _c_src(ctype: str, parse: str, fmt_macro: str, op: str, avar: str, bvar: str) -> str:
    return (
        "#include <inttypes.h>\n"
        "#include <stdint.h>\n"
        "#include <stdio.h>\n"
        "#include <stdlib.h>\n"
        f"{ctype} f({ctype} {avar}, {ctype} {bvar}) {{ return {avar} {op} {bvar}; }}\n"
        "int main(int argc, char **argv) {\n"
        "    if (argc < 3) return 2;\n"
        f"    {ctype} {avar} = ({ctype}){parse}(argv[1], 0, 10);\n"
        f"    {ctype} {bvar} = ({ctype}){parse}(argv[2], 0, 10);\n"
        f"    printf(\"%\" {fmt_macro} \"\\n\", f({avar}, {bvar}));\n"
        "    return 0;\n"
        "}\n"
    )


class RustToCIntMinDivOracle(DivergenceOracle):
    """Rust's guaranteed panic vs a C lowering's signed division UB."""

    divergence_class = INT_MIN_DIV_NEG1.key
    source_lang = "rust"
    target_lang = "c"
    confirmation_mode = "source_defined_target_ub"

    def applies_to(self, unit: Dict) -> bool:
        if unit.get("source_lang", "rust") != "rust":
            return False
        if unit.get("target_lang", "c") != "c":
            return False
        if unit.get("probe") not in (None, self.divergence_class):
            return False
        return (
            unit.get("kind") in ("div", "rem")
            and unit.get("signed", True)
            and unit.get("width", 32) in _TYPE_MAP
        )

    def find_divergence(self, unit: Dict) -> OracleResult:
        if not self.applies_to(unit):
            return OracleResult(OracleVerdict.NOT_APPLICABLE, self.divergence_class,
                                detail="unit is not a signed Rust->C div/rem")
        width = int(unit.get("width", 32))
        op_kind = unit.get("kind", "div")
        avar, bvar = unit.get("a", "a"), unit.get("b", "b")

        a = z3.BitVec(avar, width)
        b = z3.BitVec(bvar, width)
        solver = z3.Solver()
        solver.add(b != 0)
        solver.add(z3.Not(z3.BVSDivNoOverflow(a, b)))
        ar, br = unit.get("a_range"), unit.get("b_range")
        if ar is not None:
            solver.add(a >= z3.BitVecVal(int(ar[0]), width),
                       a <= z3.BitVecVal(int(ar[1]), width))
        if br is not None:
            solver.add(b >= z3.BitVecVal(int(br[0]), width),
                       b <= z3.BitVecVal(int(br[1]), width))
        if solver.check() != z3.sat:
            return OracleResult(OracleVerdict.NO_DIVERGENCE_FOUND, self.divergence_class,
                                detail="no signed-division overflow input in range")
        m = solver.model()
        a_val = _as_signed(m[a].as_long(), width)
        b_val = _as_signed(m[b].as_long(), width)
        ce = self._build(width, op_kind, avar, bvar, a_val, b_val)
        return OracleResult(OracleVerdict.DIVERGENT, self.divergence_class,
                            counterexample=ce,
                            detail=f"Z3 witness {avar}={a_val}, {bvar}={b_val}")

    def _build(self, width: int, op_kind: str, avar: str, bvar: str,
               a_val: int, b_val: int) -> Counterexample:
        rtype, ctype, parse, fmt = _TYPE_MAP[width]
        op = "/" if op_kind == "div" else "%"
        return Counterexample(
            divergence_class=self.divergence_class,
            source_lang="rust",
            target_lang="c",
            inputs={avar: a_val, bvar: b_val},
            source_snippet=_rust_src(rtype, op, avar, bvar),
            target_snippet=_c_src(ctype, parse, fmt, op, avar, bvar),
            source_definedness=Definedness.DEFINED.value,
            divergence_witness=(
                f"Rust `{avar} {op} {bvar}` with {avar}={a_val}, {bvar}={b_val} "
                f"is the signed division-overflow pair and therefore panics "
                f"deterministically (a defined observable abort). The C lowering "
                f"performs the same {ctype} operation, where the quotient is not "
                f"representable and UBSan traps it as target-side undefined behavior."
            ),
            definedness_witness=(
                f"{avar}={a_val}, {bvar}={b_val} are valid {width}-bit Rust "
                f"integers; Rust specifies the panic, while C leaves the lowered "
                f"operation undefined."
            ),
        )


register(RustToCIntMinDivOracle())
