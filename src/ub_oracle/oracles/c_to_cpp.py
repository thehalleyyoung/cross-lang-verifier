"""
C -> C++ (defined-subset) divergence oracle (100_STEPS step 117).

This is the project's first pair whose *target* is a near-syntactic superset of
the source: the **same byte-identical token** is undefined in C yet well-defined
in C++.  The witnessing construct is a left shift of ``1`` into the sign bit:

    int r = 1 << 31;

In C this is **undefined** (C17 6.5.7p4: the result ``1 * 2**31`` is not
representable in ``int``); ``-fsanitize=undefined`` traps on it and the optimizer
is entitled to assume it never happens.  In **C++20** the very same expression is
**defined**: C++20 mandated two's-complement and specified ``E1 << E2`` to be
``E1 * 2**E2`` reduced modulo ``2**N`` (C++20 [expr.shift]/2), so ``1 << 31``
yields ``INT_MIN`` (``-2147483648``) — a single deterministic value, identical on
``clang++`` and ``g++``, stable across optimisation levels.

So compiling one source text *as C* versus *as C++* produces a genuine
cross-language divergence: the C build executes UB while the C++ build is
defined.  This is confirmed with the standard ``trap_vs_defined`` mode (UBSan
trap on the C side; a defined, deterministic value on the C++ side).  The
witnessing shift amount is *found* with Z3 (the least ``n < width`` that lands a
``1`` on the sign bit), not hard-coded.
"""

from __future__ import annotations

from typing import Dict, Tuple

import z3

from ..catalogue import SIGNED_SHIFT_SIGN_BIT, Definedness
from ..plugin import DivergenceOracle, OracleResult, OracleVerdict, register
from ..replay import Counterexample


def _find_sign_bit_shift(width: int, unit: Dict) -> Tuple[bool, int]:
    """Z3-find the least non-negative shift ``n < width`` whose ``1 << n`` sets
    the sign bit of a ``width``-bit signed int (i.e. ``n == width - 1``), the
    least-extreme witness of the C-undefined / C++-defined shift."""
    n = z3.BitVec("n", 32)
    opt = z3.Optimize()
    opt.add(n >= 0, n < width)
    # 1 << n is unrepresentable in the signed type exactly when it reaches the
    # sign bit: n >= width - 1 (n == width-1 keeps it a pure sign-bit shift,
    # distinct from the shift-amount-out-of-range class).
    opt.add(n >= width - 1)
    rng = unit.get("shift_range")
    if rng is not None:
        opt.add(n >= z3.BitVecVal(int(rng[0]), 32),
                n <= z3.BitVecVal(int(rng[1]), 32))
    opt.minimize(n)
    if opt.check() != z3.sat:
        return False, 0
    return True, opt.model()[n].as_long()


def _c_src(width: int) -> str:
    ctype = "int" if width == 32 else "long long"
    fmt = "%d" if width == 32 else "%lld"
    return (
        "#include <stdio.h>\n"
        "#include <stdlib.h>\n"
        "int main(int argc, char **argv){\n"
        "    if (argc < 2) return 2;\n"
        "    int n = atoi(argv[1]);\n"
        f"    {ctype} one = 1;\n"
        f"    {ctype} r = one << n;   /* C17 6.5.7p4: 1<<(width-1) is UB */\n"
        f"    printf(\"{fmt}\\n\", r);\n"
        "    return 0;\n"
        "}\n"
    )


def _cpp_src(width: int) -> str:
    ctype = "int" if width == 32 else "long long"
    return (
        "#include <cstdio>\n"
        "#include <cstdlib>\n"
        "int main(int argc, char **argv){\n"
        "    if (argc < 2) return 2;\n"
        "    int n = std::atoi(argv[1]);\n"
        f"    {ctype} one = 1;\n"
        f"    {ctype} r = one << n;   /* C++20 [expr.shift]/2: defined modular value */\n"
        "    std::printf(\"%lld\\n\", (long long)r);\n"
        "    return 0;\n"
        "}\n"
    )


class CToCppSignedShiftOracle(DivergenceOracle):
    """C ``1 << (width-1)`` (UB) vs the C++20-defined modular value (INT_MIN)."""

    divergence_class = SIGNED_SHIFT_SIGN_BIT.key
    source_lang = "c"
    target_lang = "cpp"
    confirmation_mode = "trap_vs_defined"

    def applies_to(self, unit: Dict) -> bool:
        if unit.get("probe") not in (None, self.divergence_class):
            return False
        return unit.get("kind") == "sign_bit_shift"

    def find_divergence(self, unit: Dict) -> OracleResult:
        if not self.applies_to(unit):
            return OracleResult(OracleVerdict.NOT_APPLICABLE, self.divergence_class,
                                detail="unit is not a sign-bit-shift unit")
        width = int(unit.get("width", 32))
        ok, n = _find_sign_bit_shift(width, unit)
        if not ok:
            return OracleResult(OracleVerdict.NO_DIVERGENCE_FOUND, self.divergence_class,
                                detail="no sign-bit shift amount in declared range")
        defined_value = -(1 << (width - 1))  # INT_MIN / LLONG_MIN
        ce = Counterexample(
            divergence_class=self.divergence_class,
            source_lang="c", target_lang="cpp",
            inputs={"n": n},
            source_snippet=_c_src(width), target_snippet=_cpp_src(width),
            source_definedness=Definedness.UNDEFINED.value,
            divergence_witness=(
                f"`1 << {n}` on a {width}-bit signed int is undefined in C "
                f"(C17 6.5.7p4: the value is not representable) and UBSan traps; "
                f"the byte-identical token is *defined* in C++20 "
                f"([expr.shift]/2 mandates two's-complement modular shift) and "
                f"deterministically yields {defined_value} on clang++ and g++. "
                f"Compiling one source as C vs C++ diverges across the language "
                f"boundary."
            ),
            definedness_witness=(
                f"n={n} (= width-1) is a valid, in-range shift amount; only the "
                f"C signed-shift representability rule is violated — the C++ side "
                f"is fully defined."
            ),
        )
        return OracleResult(OracleVerdict.DIVERGENT, self.divergence_class,
                            counterexample=ce,
                            detail=f"Z3 witness n={n} (sign-bit shift); "
                                   f"C++20 value={defined_value}")


register(CToCppSignedShiftOracle())
