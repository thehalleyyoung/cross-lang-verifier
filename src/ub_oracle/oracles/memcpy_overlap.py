"""
``memcpy``/overlap divergence oracle (100_STEPS step 103).

C's ``memcpy`` is only defined when the source and destination byte ranges do
not overlap (C17 7.24.2.1p2).  That precondition is easy to violate in real C
ports: a shift within one buffer should have been ``memmove``.  Safe target
translations normally express the operation as a slice move, which is
deterministic and defined under overlap:

    * Rust ``slice::copy_within(src..src+n, dst)`` has memmove semantics.
    * Go ``copy(buf[dst:dst+n], buf[src:src+n])`` is defined for overlapping
      slices.

UBSan does not diagnose this libc precondition.  The confirmation mode therefore
uses a real C build with an executable libc-contract check (plus ASan when the
host runtime reports it), and the witness keeps ``n`` runtime-opaque via
``argv`` so the generated source exercises an actual runtime copy.
"""

from __future__ import annotations

from typing import Dict, Tuple

import z3

from ..catalogue import MEMCPY_OVERLAP, Definedness
from ..plugin import DivergenceOracle, OracleResult, OracleVerdict, register
from ..replay import Counterexample


def _alphabet(length: int) -> str:
    letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789"
    return "".join(letters[i % len(letters)] for i in range(length))


def _c_memcpy_program(length: int) -> str:
    init = _alphabet(length)
    return (
        "#include <stdio.h>\n"
        "#include <stdlib.h>\n"
        "#include <stdint.h>\n"
        "#include <string.h>\n"
        "#ifdef CLV_CHECK_MEMCPY\n"
        "static void *clv_checked_memcpy(void *dst, const void *src, size_t n){\n"
        "    uintptr_t d = (uintptr_t)dst;\n"
        "    uintptr_t s = (uintptr_t)src;\n"
        "    if (n > 0 && d < s + n && s < d + n) {\n"
        "        fprintf(stderr, \"runtime error: memcpy-param-overlap dst=%p src=%p n=%zu\\n\", dst, src, n);\n"
        "        abort();\n"
        "    }\n"
        "    return memmove(dst, src, n);\n"
        "}\n"
        "#define memcpy(d,s,n) clv_checked_memcpy((d),(s),(n))\n"
        "#endif\n"
        f"#define BUFLEN {length}\n"
        "int main(int argc, char** argv){\n"
        "    if (argc < 4) return 2;\n"
        "    long dst_l = strtol(argv[1], 0, 10);\n"
        "    long src_l = strtol(argv[2], 0, 10);\n"
        "    long n_l = strtol(argv[3], 0, 10);\n"
        "    if (dst_l < 0 || src_l < 0 || n_l < 0) return 3;\n"
        "    if (dst_l + n_l > BUFLEN || src_l + n_l > BUFLEN) return 3;\n"
        "    size_t dst = (size_t)dst_l;\n"
        "    size_t src = (size_t)src_l;\n"
        "    size_t n = (size_t)n_l;\n"
        f"    unsigned char buf[BUFLEN + 1] = \"{init}\";\n"
        "    memcpy(buf + dst, buf + src, n);\n"
        "    for (size_t i = 0; i < BUFLEN; i++) putchar((int)buf[i]);\n"
        "    putchar('\\n');\n"
        "    return 0;\n"
        "}\n"
    )


def _rust_memmove_program(length: int) -> str:
    init = _alphabet(length)
    return (
        "fn main() {\n"
        "    let args: Vec<String> = std::env::args().collect();\n"
        "    if args.len() < 4 { std::process::exit(2); }\n"
        "    let dst_i: i64 = args[1].parse().unwrap();\n"
        "    let src_i: i64 = args[2].parse().unwrap();\n"
        "    let n_i: i64 = args[3].parse().unwrap();\n"
        f"    let mut buf = b\"{init}\".to_vec();\n"
        "    if dst_i < 0 || src_i < 0 || n_i < 0 { panic!(\"negative range\"); }\n"
        "    let dst = dst_i as usize;\n"
        "    let src = src_i as usize;\n"
        "    let n = n_i as usize;\n"
        "    if dst + n > buf.len() || src + n > buf.len() { panic!(\"range out of bounds\"); }\n"
        "    buf.copy_within(src..src + n, dst);\n"
        "    println!(\"{}\", String::from_utf8_lossy(&buf));\n"
        "}\n"
    )


def _go_memmove_program(length: int) -> str:
    init = _alphabet(length)
    return (
        "package main\n"
        "import (\n\t\"fmt\"\n\t\"os\"\n\t\"strconv\"\n)\n"
        "func main() {\n"
        "\tif len(os.Args) < 4 { os.Exit(2) }\n"
        "\tdst, _ := strconv.Atoi(os.Args[1])\n"
        "\tsrc, _ := strconv.Atoi(os.Args[2])\n"
        "\tn, _ := strconv.Atoi(os.Args[3])\n"
        f"\tbuf := []byte(\"{init}\")\n"
        "\tif dst < 0 || src < 0 || n < 0 || dst+n > len(buf) || src+n > len(buf) { panic(\"range\") }\n"
        "\tcopy(buf[dst:dst+n], buf[src:src+n])\n"
        "\tfmt.Println(string(buf))\n"
        "}\n"
    )


def _fixed_or_range(opt: z3.Optimize, name: str, unit: Dict,
                    lo: int, hi: int) -> z3.IntNumRef:
    v = z3.Int(name)
    if name in unit:
        opt.add(v == int(unit[name]))
    else:
        rng = unit.get(f"{name}_range")
        if rng is not None:
            lo, hi = int(rng[0]), int(rng[1])
        opt.add(v >= lo, v <= hi)
    return v


def _find_overlap(unit: Dict) -> Tuple[bool, int, int, int, int]:
    """Find a small in-bounds overlapping copy `(dst, src, n, length)`.

    The copy length defaults to at least four bytes because ASan's libc
    interceptor sees a real runtime-sized copy there; callers may tighten ranges
    or pin exact offsets to prove safe/non-overlap cases produce no witness.
    """
    length = int(unit.get("buffer_len", 16))
    if length <= 1:
        return False, 0, 0, 0, length
    min_copy = int(unit.get("min_copy", 4))
    opt = z3.Optimize()
    dst = _fixed_or_range(opt, "dst", unit, 0, length - 1)
    src = _fixed_or_range(opt, "src", unit, 0, length - 1)
    n = _fixed_or_range(opt, "n", unit, min_copy, length - 1)
    opt.add(n > 0)
    opt.add(dst + n <= length, src + n <= length)
    opt.add(dst != src)
    opt.add(dst < src + n, src < dst + n)
    opt.minimize(n)
    opt.minimize(src)
    opt.minimize(dst)
    if opt.check() != z3.sat:
        return False, 0, 0, 0, length
    model = opt.model()
    return (
        True,
        model.eval(dst, model_completion=True).as_long(),
        model.eval(src, model_completion=True).as_long(),
        model.eval(n, model_completion=True).as_long(),
        length,
    )


class _MemcpyOverlapBase(DivergenceOracle):
    divergence_class = MEMCPY_OVERLAP.key
    source_lang = "c"
    confirmation_mode = "libc_contract_trap_vs_defined"

    def applies_to(self, unit: Dict) -> bool:
        if unit.get("probe") not in (None, self.divergence_class):
            return False
        return unit.get("kind") in {"memcpy_overlap", "memcpy_call"}

    def _result(self, unit: Dict, target_src: str) -> OracleResult:
        if not self.applies_to(unit):
            return OracleResult(OracleVerdict.NOT_APPLICABLE, self.divergence_class,
                                detail="unit is not a memcpy call")
        ok, dst, src, n, length = _find_overlap(unit)
        if not ok:
            return OracleResult(OracleVerdict.NO_DIVERGENCE_FOUND, self.divergence_class,
                                detail="no in-bounds overlapping ranges in declared domain")
        ce = self._build(dst, src, n, length, target_src)
        return OracleResult(OracleVerdict.DIVERGENT, self.divergence_class,
                            counterexample=ce,
                            detail=f"Z3 witness dst={dst} src={src} n={n} (ranges overlap)")

    def _build(self, dst: int, src: int, n: int, length: int,
               target_src: str) -> Counterexample:
        return Counterexample(
            divergence_class=self.divergence_class,
            source_lang="c", target_lang=self.target_lang,
            inputs={"dst": dst, "src": src, "n": n},
            source_snippet=_c_memcpy_program(length),
            target_snippet=target_src,
            source_definedness=Definedness.UNDEFINED.value,
            divergence_witness=(
                f"C `memcpy(buf+{dst}, buf+{src}, {n})` copies overlapping byte "
                f"ranges [{dst},{dst+n}) and [{src},{src+n}) in the same object, "
                f"violating C17 7.24.2.1p2. The target {self.target_lang} slice "
                f"operation is memmove-like and therefore deterministic."
            ),
            definedness_witness=(
                f"dst={dst}, src={src}, n={n}, len={length} are all in bounds; "
                f"the only invalid source-side fact is the `memcpy` non-overlap "
                f"precondition."
            ),
        )


class MemcpyOverlapOracle(_MemcpyOverlapBase):
    """C overlapping ``memcpy`` vs Rust ``slice::copy_within``."""

    target_lang = "rust"

    def find_divergence(self, unit: Dict) -> OracleResult:
        length = int(unit.get("buffer_len", 16))
        return self._result(unit, _rust_memmove_program(length))


class GoMemcpyOverlapOracle(_MemcpyOverlapBase):
    """C overlapping ``memcpy`` vs Go's overlap-defined ``copy``."""

    target_lang = "go"

    def find_divergence(self, unit: Dict) -> OracleResult:
        length = int(unit.get("buffer_len", 16))
        return self._result(unit, _go_memmove_program(length))


register(MemcpyOverlapOracle())
register(GoMemcpyOverlapOracle())
