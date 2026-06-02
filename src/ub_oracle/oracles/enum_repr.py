"""
Out-of-range enum / trap-representation divergence oracle (100_STEPS step 108).

A C enumeration has an implementation-defined integer type wide enough to hold
every enumerator (C17 6.7.2.2p4, 6.2.6.1). Storing a value that is *outside* the
enumerator set but still within that integer type is permitted, and the value is
read back verbatim: ``(enum Color)3`` for ``enum Color { RED, GREEN, BLUE }`` is
simply the integer ``3``.

A safe target enum has **no representation** for such an out-of-range
discriminant — constructing one by transmute would itself be undefined — so the
idiomatic, faithful port matches on the integer and collapses any unknown value
to a default variant (Rust ``match { _ => Default }``, Go ``switch { default }``).
The same input therefore produces a *different*, deterministic value: C keeps
``3`` while the port yields its default (``0``).

Both programs are fully language-defined and deterministic; the observable result
nonetheless differs. The witnessing discriminant (the least value just past the
largest enumerator) is Z3-found within any declared range, and the divergence is
confirmed by the harness's ``defined_divergence`` mode, which builds and runs
*both* real programs and checks that each is defined-and-deterministic while
their printed values differ.
"""

from __future__ import annotations

from typing import Dict, List

import z3

from ..catalogue import ENUM_OUT_OF_RANGE, Definedness
from ..plugin import DivergenceOracle, OracleResult, OracleVerdict, register
from ..replay import Counterexample

_DEFAULT_NAMES = ("RED", "GREEN", "BLUE")


def _names(unit: Dict) -> List[str]:
    n = unit.get("enumerators")
    if n is None:
        return list(_DEFAULT_NAMES)
    if isinstance(n, int):
        return [f"E{i}" for i in range(n)]
    return [str(x) for x in n]


def _find_out_of_range(num: int, unit: Dict) -> int:
    """Z3-find the least non-negative discriminant strictly past the largest
    enumerator (index ``num-1``), honouring any declared range."""
    n = z3.Int("n")
    opt = z3.Optimize()
    opt.add(n >= num)            # out of range: past the last enumerator
    rng = unit.get("value_range")
    if rng is not None:
        opt.add(n >= int(rng[0]), n <= int(rng[1]))
    opt.minimize(n)
    if opt.check() != z3.sat:
        return -1
    return opt.model()[n].as_long()


def _c_src(names: List[str]) -> str:
    enumerators = ", ".join(names)
    return (
        "#include <stdio.h>\n#include <stdlib.h>\n"
        f"enum Color {{ {enumerators} }};\n"
        "int main(int argc, char **argv){\n"
        "    if (argc < 2) return 2;\n"
        "    enum Color c = (enum Color)atoi(argv[1]);\n"
        "    printf(\"%d\\n\", (int)c);\n    return 0;\n}\n"
    )


def _rust_src(names: List[str]) -> str:
    variants = ", ".join(f"{name.capitalize()} = {i}" for i, name in enumerate(names))
    arms = "".join(
        f"        {i} => Color::{name.capitalize()},\n" for i, name in enumerate(names)
    )
    default = names[0].capitalize()
    return (
        "#[derive(Clone, Copy)]\n"
        f"enum Color {{ {variants} }}\n"
        "fn from_int(n: i64) -> Color {\n"
        "    match n {\n"
        f"{arms}"
        f"        _ => Color::{default},\n"
        "    }\n}\n"
        "fn main(){\n"
        "    let n: i64 = std::env::args().nth(1).unwrap().parse().unwrap();\n"
        "    let c = from_int(n);\n"
        "    println!(\"{}\", c as i64);\n}\n"
    )


def _go_src(names: List[str]) -> str:
    const_block = "\t" + names[0] + " Color = iota\n" + "".join(
        f"\t{n}\n" for n in names[1:]
    )
    cases = "".join(f"\tcase {i}:\n\t\treturn {name}\n" for i, name in enumerate(names))
    return (
        "package main\n"
        "import (\n\t\"fmt\"\n\t\"os\"\n\t\"strconv\"\n)\n"
        "type Color int\n"
        "const (\n" + const_block + ")\n"
        "func fromInt(n int) Color {\n\tswitch n {\n"
        + cases
        + "\tdefault:\n\t\treturn " + names[0] + "\n\t}\n}\n"
        "func main() {\n"
        "\tn, _ := strconv.Atoi(os.Args[1])\n"
        "\tfmt.Println(int(fromInt(n)))\n}\n"
    )


class _EnumBase(DivergenceOracle):
    divergence_class = ENUM_OUT_OF_RANGE.key
    source_lang = "c"
    confirmation_mode = "defined_divergence"

    def applies_to(self, unit: Dict) -> bool:
        if unit.get("probe") not in (None, self.divergence_class):
            return False
        if unit.get("target_lang", self.target_lang) != self.target_lang:
            return False
        return unit.get("kind") == "enum_cast"

    def _target_src(self, names) -> str:  # pragma: no cover - overridden
        raise NotImplementedError

    def find_divergence(self, unit: Dict) -> OracleResult:
        if not self.applies_to(unit):
            return OracleResult(OracleVerdict.NOT_APPLICABLE, self.divergence_class,
                                detail="unit is not an enum cast")
        names = _names(unit)
        n = _find_out_of_range(len(names), unit)
        if n < 0:
            return OracleResult(OracleVerdict.NO_DIVERGENCE_FOUND, self.divergence_class,
                                detail="no out-of-range discriminant in declared range")
        ce = self._build(names, n)
        return OracleResult(OracleVerdict.DIVERGENT, self.divergence_class,
                            counterexample=ce, detail=f"Z3 witness n={n}")

    def _build(self, names, n) -> Counterexample:
        return Counterexample(
            divergence_class=self.divergence_class,
            source_lang="c", target_lang=self.target_lang,
            inputs={"n": n},
            source_snippet=_c_src(names),
            target_snippet=self._target_src(names),
            source_definedness=Definedness.IMPLEMENTATION_DEFINED.value,
            divergence_witness=(
                f"n={n} is past the last enumerator (index {len(names) - 1}): C "
                f"stores and reads it verbatim, printing {n}, while the faithful "
                f"{self.target_lang} port has no representation for it and "
                f"collapses to its default variant, printing 0. Both programs are "
                f"defined and deterministic, yet a C->{self.target_lang} port "
                f"silently changes the value on this input."
            ),
            definedness_witness=(
                f"n={n} is a valid integer for the enum's underlying type; "
                f"neither language has undefined behaviour — the divergence is a "
                f"difference of defined semantics (C keeps the raw value; the "
                f"target collapses it)."
            ),
        )


class EnumOutOfRangeOracle(_EnumBase):
    """C's verbatim out-of-range enum value vs Rust's default-collapsing match."""

    target_lang = "rust"

    def _target_src(self, names) -> str:
        return _rust_src(names)


class GoEnumOutOfRangeOracle(_EnumBase):
    """C's verbatim out-of-range enum value vs Go's default-collapsing switch."""

    target_lang = "go"

    def _target_src(self, names) -> str:
        return _go_src(names)


register(EnumOutOfRangeOracle())
register(GoEnumOutOfRangeOracle())
