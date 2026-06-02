"""
Go -> Rust *defined-but-different* divergence oracle (100_STEPS step 120).

Every other pair in the catalogue anchors on a *C* source whose behaviour is
*undefined*: the divergence is "C may do anything / the optimiser exploits it"
versus "the target is defined". This step takes the harder, subtler case the
flagship paper needs to make a general claim: **two memory-safe languages that
are each fully defined yet disagree.** Porting Go to Rust is a real, common
migration, and a faithful-looking port can silently change behaviour even though
*neither* language has any undefined behaviour to blame.

The witness is signed two's-complement division/remainder overflow — the unique
pair ``INT_MIN`` ``op`` ``-1``:

    * **Go** defines integer overflow as modular two's-complement wraparound
      (Go spec, "Integer operators"). ``int32(-2147483648) / -1`` evaluates to a
      defined value, ``-2147483648``; ``% -1`` evaluates to a defined ``0``.
    * **Rust** defines the same expression to **panic** — division (and
      remainder) overflow is a guaranteed, deterministic abort in *every* build
      profile, debug *and* release (it is not subject to the
      ``-C overflow-checks`` relaxation that ordinary ``+``/``*`` are). The
      process exits with code 101 after unwinding.

Both outcomes are language-*defined*; the observable behaviour nonetheless
differs (a printed value vs. a deterministic abort). A program translated from
Go to Rust therefore changes meaning on this input. The witnessing pair is
*found* with Z3 (the signed-division-overflow input, generalising across widths),
not hard-coded, and the divergence is confirmed by the harness's
``defined_divergence`` mode: it builds and runs *both* real programs and checks
that each is defined-and-deterministic while their observable behaviour differs.
No sanitizer and no optimisation-level disagreement is involved — there is no
undefined behaviour on either side.
"""

from __future__ import annotations

from typing import Dict

import z3

from ..catalogue import INT_MIN_DIV_NEG1, Definedness
from ..plugin import DivergenceOracle, OracleResult, OracleVerdict, register
from ..replay import Counterexample

# width -> (Go integer type, Rust integer type, strconv bit size)
_TYPE_MAP = {
    32: ("int32", "i32", 32),
    64: ("int64", "i64", 64),
}


def _as_signed(raw: int, width: int) -> int:
    hi = (1 << (width - 1)) - 1
    return raw if raw <= hi else raw - (1 << width)


def _go_src(gtype: str, bits: int, op: str, avar: str, bvar: str) -> str:
    """A standalone Go program: parse two ints, print ``a op b`` (modular)."""
    return (
        "package main\n"
        "import (\n\t\"fmt\"\n\t\"os\"\n\t\"strconv\"\n)\n"
        f"func f({avar} {gtype}, {bvar} {gtype}) {gtype} {{\n"
        f"\treturn {avar} {op} {bvar}\n"
        "}\n"
        "func main() {\n"
        f"\tp, _ := strconv.ParseInt(os.Args[1], 10, {bits})\n"
        f"\tq, _ := strconv.ParseInt(os.Args[2], 10, {bits})\n"
        f"\tfmt.Println(f({gtype}(p), {gtype}(q)))\n"
        "}\n"
    )


def _rust_src(rtype: str, op: str, avar: str, bvar: str) -> str:
    """The faithful-looking Rust port: the same ``a op b``, which *panics* on the
    overflowing pair (a defined, deterministic abort)."""
    return (
        f"fn f({avar}: {rtype}, {bvar}: {rtype}) -> {rtype} {{ {avar} {op} {bvar} }}\n"
        "fn main(){\n"
        "    let v: Vec<String> = std::env::args().collect();\n"
        f"    let {avar}: {rtype} = v[1].parse().unwrap();\n"
        f"    let {bvar}: {rtype} = v[2].parse().unwrap();\n"
        f"    println!(\"{{}}\", f({avar}, {bvar}));\n"
        "}\n"
    )


class GoToRustIntMinDivOracle(DivergenceOracle):
    """Go's modular ``INT_MIN/-1`` (a defined value) vs Rust's guaranteed panic.

    Covers both division (``/``) and remainder (``%``); the unit's ``kind``
    selects which. Both are defined-but-different between the two safe languages.
    """

    divergence_class = INT_MIN_DIV_NEG1.key
    source_lang = "go"
    target_lang = "rust"
    confirmation_mode = "defined_divergence"

    def applies_to(self, unit: Dict) -> bool:
        if unit.get("source_lang", "go") != "go":
            return False
        if unit.get("target_lang", "rust") != "rust":
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
                                detail="unit is not a signed Go->Rust div/rem")
        width = unit.get("width", 32)
        op_kind = unit.get("kind", "div")
        avar, bvar = unit.get("a", "a"), unit.get("b", "b")

        # Z3-find the signed division/remainder overflow input from first
        # principles: the unique pair whose quotient is not representable.
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

    def _build(self, width, op_kind, avar, bvar, a_val, b_val) -> Counterexample:
        gtype, rtype, bits = _TYPE_MAP[width]
        op = "/" if op_kind == "div" else "%"
        go_src = _go_src(gtype, bits, op, avar, bvar)
        rust_src = _rust_src(rtype, op, avar, bvar)
        # Go's modular outcome on the witnessing pair (a defined value).
        go_value = a_val if op_kind == "div" else 0
        return Counterexample(
            divergence_class=self.divergence_class,
            source_lang="go", target_lang="rust",
            inputs={avar: a_val, bvar: b_val},
            source_snippet=go_src, target_snippet=rust_src,
            source_definedness=Definedness.DEFINED.value,
            divergence_witness=(
                f"Go `{avar} {op} {bvar}` with {avar}={a_val}, {bvar}={b_val} is the "
                f"signed two's-complement overflow pair: Go defines it by modular "
                f"wraparound to {go_value} (a defined value, exit 0), while the "
                f"faithful Rust port `{avar} {op} {bvar}` panics deterministically "
                f"(a defined abort, exit 101). Both languages are fully defined "
                f"here, yet the observable behaviour differs — a Go->Rust port "
                f"silently changes meaning on this input."
            ),
            definedness_witness=(
                f"{avar}={a_val}, {bvar}={b_val} are valid {width}-bit integers; "
                f"neither language has undefined behaviour — the divergence is a "
                f"difference of *defined* semantics (Go modular vs Rust panic)."
            ),
        )


register(GoToRustIntMinDivOracle())
