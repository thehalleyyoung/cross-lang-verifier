"""
Integer-model divergence oracles (100_STEPS step 18).

Three more C-UB-vs-Rust-defined divergence classes that, unlike signed overflow,
do not silently flip an observable value but instead make the C program execute
*undefined behavior* on a perfectly well-defined input, while the idiomatic Rust
translation has a fully defined, deterministic outcome (a value or a clean
panic).  These are confirmed with the harness's ``trap_vs_defined`` mode:

    * shift-out-of-range :  C ``x << s`` with ``s >= width``  (UB; UBSan traps)
                            Rust ``x.wrapping_shl(s)``          (defined value)
    * division-by-zero   :  C ``a / b`` with ``b == 0``        (UB; UBSan traps)
                            Rust ``a / b``                      (defined panic)
    * INT_MIN / -1       :  C ``a / b`` with the one overflowing pair (UB)
                            Rust ``a / b``                      (defined panic)

Each oracle uses Z3 to *find* the witnessing input from first principles (an
out-of-range shift amount; a zero divisor; the unique signed-division overflow
pair) rather than hard-coding it, so the search generalizes across widths.
"""

from __future__ import annotations

from typing import Dict, Tuple

import z3

from ..catalogue import (
    SHIFT_OUT_OF_RANGE,
    DIVISION_BY_ZERO,
    INT_MIN_DIV_NEG1,
    Definedness,
)
from ..plugin import DivergenceOracle, OracleResult, OracleVerdict, register
from ..replay import Counterexample

# width -> (C type, Rust type, printf/strtol scan format, strtol fn)
_TYPE_MAP = {
    32: ("int", "i32", "%d", "strtol"),
    64: ("long long", "i64", "%lld", "strtoll"),
}


def _as_signed(raw: int, width: int) -> int:
    hi = (1 << (width - 1)) - 1
    return raw if raw <= hi else raw - (1 << width)


def _c_two_arg_main(ctype: str, scan_fn: str, fmt: str, body_decl: str,
                    a: str, b: str, btype: str = None) -> str:
    btype = btype or ctype
    return (
        "#include <stdio.h>\n"
        "#include <stdlib.h>\n"
        f"{body_decl}\n"
        "int main(int argc, char**argv){\n"
        "    if (argc < 3) return 2;\n"
        f"    {ctype} {a} = ({ctype}){scan_fn}(argv[1], 0, 10);\n"
        f"    {btype} {b} = ({btype}){scan_fn}(argv[2], 0, 10);\n"
        f"    printf(\"{fmt}\\n\", f({a}, {b}));\n"
        "    return 0;\n"
        "}\n"
    )


def _rust_two_arg_main(rtype: str, body_decl: str, a: str, b: str,
                       btype: str = None) -> str:
    btype = btype or rtype
    return (
        f"{body_decl}\n"
        "fn main(){\n"
        "    let v: Vec<String> = std::env::args().collect();\n"
        f"    let {a}: {rtype} = v[1].parse().unwrap();\n"
        f"    let {b}: {btype} = v[2].parse().unwrap();\n"
        f"    println!(\"{{}}\", f({a}, {b}));\n"
        "}\n"
    )


class ShiftOutOfRangeOracle(DivergenceOracle):
    """``x << s`` where the shift amount reaches or exceeds the bit width."""

    divergence_class = SHIFT_OUT_OF_RANGE.key
    source_lang = "c"
    target_lang = "rust"
    confirmation_mode = "trap_vs_defined"

    def applies_to(self, unit: Dict) -> bool:
        if unit.get("probe") not in (None, self.divergence_class):
            return False
        return (
            unit.get("kind") == "shift"
            and unit.get("width", 32) in _TYPE_MAP
        )

    def find_divergence(self, unit: Dict) -> OracleResult:
        if not self.applies_to(unit):
            return OracleResult(OracleVerdict.NOT_APPLICABLE, self.divergence_class,
                                detail="unit is not a shift")
        width = unit.get("width", 32)
        var = unit.get("var", "x")
        svar = unit.get("shift_var", "s")
        x_val = int(unit.get("value", 1))

        # Prefer the classic "too large" witness (s >= width), but also cover
        # the other C shift-count UB boundary (s < 0) when the declared range
        # admits only negative counts.  This mirrors the interval pre-pass.
        sr = unit.get("shift_range")
        lo, hi = (int(sr[0]), int(sr[1])) if sr is not None else (-(1 << 15), 1 << 16)
        s = z3.Int(svar)

        def _solve(extra, *, maximize: bool = False):
            opt = z3.Optimize()
            opt.add(s >= lo, s <= hi, extra)
            if maximize:
                opt.maximize(s)
            else:
                opt.minimize(s)
            if opt.check() != z3.sat:
                return None
            return opt.model()[s].as_long()

        shift_amt = _solve(s >= width)
        if shift_amt is None:
            shift_amt = _solve(s < 0, maximize=True)
        if shift_amt is None:
            return OracleResult(OracleVerdict.NO_DIVERGENCE_FOUND, self.divergence_class,
                                detail="no out-of-range shift amount found")

        ce = self._build(width, var, svar, x_val, shift_amt)
        return OracleResult(OracleVerdict.DIVERGENT, self.divergence_class,
                            counterexample=ce,
                            detail=f"Z3 witness {svar}={shift_amt} (outside [0, {width - 1}])")

    def _build(self, width, var, svar, x_val, shift_amt) -> Counterexample:
        ctype, rtype, fmt, scan = _TYPE_MAP[width]
        c_decl = f"{ctype} f({ctype} {var}, int {svar}){{ return {var} << {svar}; }}"
        c_src = _c_two_arg_main(ctype, scan, fmt, c_decl, var, svar, btype="int")
        r_decl = (f"fn f({var}: {rtype}, {svar}: i32) -> {rtype} "
                  f"{{ {var}.wrapping_shl({svar} as u32) }}")
        rust_src = _rust_two_arg_main(rtype, r_decl, var, svar, btype="i32")
        return Counterexample(
            divergence_class=self.divergence_class,
            source_lang="c", target_lang="rust",
            inputs={var: x_val, svar: shift_amt},
            source_snippet=c_src, target_snippet=rust_src,
            source_definedness=Definedness.UNDEFINED.value,
            divergence_witness=(
                f"C `{var} << {svar}` with {svar}={shift_amt} >= width {width} is UB; "
                f"Rust `wrapping_shl` masks the amount and returns a defined value."
            ),
            definedness_witness=(
                f"{var}={x_val}, {svar}={shift_amt} are both valid integers; only the "
                f"C shift operation is undefined."
            ),
        )


class DivisionByZeroOracle(DivergenceOracle):
    """``a / b`` where the divisor is zero."""

    divergence_class = DIVISION_BY_ZERO.key
    source_lang = "c"
    target_lang = "rust"
    confirmation_mode = "trap_vs_defined"

    def applies_to(self, unit: Dict) -> bool:
        if unit.get("probe") not in (None, self.divergence_class):
            return False
        return (
            unit.get("kind") in ("div", "rem")
            and unit.get("width", 32) in _TYPE_MAP
        )

    def find_divergence(self, unit: Dict) -> OracleResult:
        if not self.applies_to(unit):
            return OracleResult(OracleVerdict.NOT_APPLICABLE, self.divergence_class,
                                detail="unit is not a div/rem")
        width = unit.get("width", 32)
        op = unit.get("kind", "div")
        avar, bvar = unit.get("a", "a"), unit.get("b", "b")
        a_val = int(unit.get("dividend", 7))

        b = z3.BitVec(bvar, width)
        solver = z3.Solver()
        solver.add(b == 0)
        br = unit.get("b_range")
        if br is not None:
            solver.add(b >= z3.BitVecVal(int(br[0]), width),
                       b <= z3.BitVecVal(int(br[1]), width))
        if solver.check() != z3.sat:
            return OracleResult(OracleVerdict.NO_DIVERGENCE_FOUND, self.divergence_class,
                                detail="zero divisor excluded by declared range")
        b_val = _as_signed(solver.model()[b].as_long(), width)

        ce = self._build(width, op, avar, bvar, a_val, b_val)
        return OracleResult(OracleVerdict.DIVERGENT, self.divergence_class,
                            counterexample=ce,
                            detail=f"Z3 witness {bvar}={b_val} (zero divisor)")

    def _build(self, width, op, avar, bvar, a_val, b_val) -> Counterexample:
        ctype, rtype, fmt, scan = _TYPE_MAP[width]
        c_op = "/" if op == "div" else "%"
        r_op = "/" if op == "div" else "%"
        c_decl = f"{ctype} f({ctype} {avar}, {ctype} {bvar}){{ return {avar} {c_op} {bvar}; }}"
        c_src = _c_two_arg_main(ctype, scan, fmt, c_decl, avar, bvar)
        r_decl = (f"fn f({avar}: {rtype}, {bvar}: {rtype}) -> {rtype} "
                  f"{{ {avar} {r_op} {bvar} }}")
        rust_src = _rust_two_arg_main(rtype, r_decl, avar, bvar)
        return Counterexample(
            divergence_class=self.divergence_class,
            source_lang="c", target_lang="rust",
            inputs={avar: a_val, bvar: b_val},
            source_snippet=c_src, target_snippet=rust_src,
            source_definedness=Definedness.UNDEFINED.value,
            divergence_witness=(
                f"C `{avar} {c_op} {bvar}` with {bvar}=0 is UB; Rust panics "
                f"deterministically (a defined, observable outcome)."
            ),
            definedness_witness=(
                f"{avar}={a_val}, {bvar}={b_val} are valid integers; only the C "
                f"division is undefined."
            ),
        )


class IntMinDivNeg1Oracle(DivergenceOracle):
    """``a / b`` at the unique signed-division overflow point (INT_MIN / -1)."""

    divergence_class = INT_MIN_DIV_NEG1.key
    source_lang = "c"
    target_lang = "rust"
    confirmation_mode = "trap_vs_defined"

    def applies_to(self, unit: Dict) -> bool:
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
                                detail="unit is not a signed div/rem")
        width = unit.get("width", 32)
        op = unit.get("kind", "div")
        avar, bvar = unit.get("a", "a"), unit.get("b", "b")

        a = z3.BitVec(avar, width)
        b = z3.BitVec(bvar, width)
        solver = z3.Solver()
        solver.add(b != 0)
        # The signed-division-overflow input: where a/b is not representable.
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
                                detail="no signed-division overflow input")
        m = solver.model()
        a_val = _as_signed(m[a].as_long(), width)
        b_val = _as_signed(m[b].as_long(), width)

        ce = self._build(width, op, avar, bvar, a_val, b_val)
        return OracleResult(OracleVerdict.DIVERGENT, self.divergence_class,
                            counterexample=ce,
                            detail=f"Z3 witness {avar}={a_val}, {bvar}={b_val}")

    def _build(self, width, op, avar, bvar, a_val, b_val) -> Counterexample:
        ctype, rtype, fmt, scan = _TYPE_MAP[width]
        c_op = "/" if op == "div" else "%"
        r_op = "/" if op == "div" else "%"
        c_decl = f"{ctype} f({ctype} {avar}, {ctype} {bvar}){{ return {avar} {c_op} {bvar}; }}"
        c_src = _c_two_arg_main(ctype, scan, fmt, c_decl, avar, bvar)
        r_decl = (f"fn f({avar}: {rtype}, {bvar}: {rtype}) -> {rtype} "
                  f"{{ {avar} {r_op} {bvar} }}")
        rust_src = _rust_two_arg_main(rtype, r_decl, avar, bvar)
        return Counterexample(
            divergence_class=self.divergence_class,
            source_lang="c", target_lang="rust",
            inputs={avar: a_val, bvar: b_val},
            source_snippet=c_src, target_snippet=rust_src,
            source_definedness=Definedness.UNDEFINED.value,
            divergence_witness=(
                f"C `{avar} {c_op} {bvar}` with {avar}={a_val}, {bvar}={b_val} overflows "
                f"signed division (UB); Rust panics deterministically (defined)."
            ),
            definedness_witness=(
                f"{avar}={a_val}, {bvar}={b_val} are valid {width}-bit integers; only the "
                f"C division is undefined."
            ),
        )


# Register the three integer-model oracles.
register(ShiftOutOfRangeOracle())
register(DivisionByZeroOracle())
register(IntMinDivNeg1Oracle())
