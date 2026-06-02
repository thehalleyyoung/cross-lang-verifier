"""
Bit-field layout & packing divergence oracle (100_STEPS step 112).

A C bit-field group such as ``struct S { unsigned a:3, b:5, c:8; };`` is *packed*
by the implementation into a single addressable storage unit (here one 4-byte
``unsigned``): the struct's ``sizeof`` and its exact in-memory byte image are
fixed by the ABI, **not** by the source text (C17 6.7.2.1p11 makes the
allocation order, alignment and storage unit implementation-defined).

A translator that does not model bit-fields renders each field as an ordinary
integer field — the narrowest faithful choice being one byte per field
(``#[repr(C)]`` ``u8`` in Rust, ``uint8`` in Go). The target struct is therefore
*unpacked*: a different ``sizeof`` **and** a different byte image. Any code that
serialises the struct across an ABI / wire boundary (the whole reason bit-fields
exist) silently changes meaning under the port — even though *both* programs are
fully language-defined and deterministic.

The witness is an assignment of nonzero field values (Z3-found within each
field's declared range), and the divergence is confirmed by the harness's
``defined_divergence`` mode: it builds and runs *both* real programs, requires
each to be defined-and-deterministic, and checks that their observable
``size=…/bytes=…`` output differs. No sanitizer and no optimisation-level
disagreement is involved — the byte images simply differ on real compiled
layout.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import z3

from ..catalogue import BITFIELD_LAYOUT, Definedness
from ..plugin import DivergenceOracle, OracleResult, OracleVerdict, register
from ..replay import Counterexample

# Default field group: (name, bit-width). Each width <= 8 so the value fits the
# faithful per-field byte, and the total <= 32 so the C storage unit is one
# `unsigned`. The packed C image is 4 bytes; the unpacked target image is 3.
_DEFAULT_FIELDS: Tuple[Tuple[str, int], ...] = (("a", 3), ("b", 5), ("c", 8))


def _fields(unit: Dict) -> List[Tuple[str, int]]:
    raw = unit.get("fields", _DEFAULT_FIELDS)
    return [(str(n), int(w)) for n, w in raw]


def _find_field_values(fields: List[Tuple[str, int]], unit: Dict) -> Dict[str, int]:
    """Z3-find nonzero field values (honouring any per-field range) so that the
    packed and unpacked byte images are both populated and observably differ."""
    opt = z3.Optimize()
    bvs: Dict[str, z3.BitVecRef] = {}
    ranges = unit.get("value_ranges", {})
    total = None
    for name, bits in fields:
        v = z3.BitVec(name, 16)
        bvs[name] = v
        lo, hi = 1, (1 << bits) - 1
        r = ranges.get(name)
        if r is not None:
            lo, hi = max(lo, int(r[0])), min(hi, int(r[1]))
        opt.add(v >= z3.BitVecVal(lo, 16), v <= z3.BitVecVal(hi, 16))
        term = z3.ZeroExt(16, v)
        total = term if total is None else total + term
    # A vivid, deterministic witness: maximise the populated bits.
    opt.maximize(total)
    if opt.check() != z3.sat:
        return {}
    m = opt.model()
    return {name: m[bvs[name]].as_long() for name, _ in fields}


def _c_image(fields: List[Tuple[str, int]], vals: Dict[str, int]) -> str:
    """Little-endian byte image of the packed 4-byte `unsigned` storage unit."""
    packed, off = 0, 0
    for name, bits in fields:
        packed |= (vals[name] & ((1 << bits) - 1)) << off
        off += bits
    return (packed & 0xFFFFFFFF).to_bytes(4, "little").hex()


def _target_image(fields: List[Tuple[str, int]], vals: Dict[str, int]) -> str:
    """Byte image of the unpacked one-byte-per-field target struct."""
    return bytes(vals[name] & 0xFF for name, _ in fields).hex()


def _c_src(fields: List[Tuple[str, int]], vals: Dict[str, int]) -> str:
    decls = " ".join(f"unsigned {n}:{w};" for n, w in fields)
    sets = "".join(f"    s.{n} = {vals[n]};\n" for n, _ in fields)
    return (
        "#include <stdio.h>\n#include <string.h>\n"
        f"struct S {{ {decls} }};\n"
        "int main(void){\n"
        "    struct S s; memset(&s, 0, sizeof s);\n"
        f"{sets}"
        "    unsigned char buf[sizeof(struct S)];\n"
        "    memcpy(buf, &s, sizeof s);\n"
        "    printf(\"size=%zu bytes=\", sizeof(struct S));\n"
        "    for (size_t i = 0; i < sizeof(struct S); i++) printf(\"%02x\", buf[i]);\n"
        "    printf(\"\\n\");\n    return 0;\n}\n"
    )


def _rust_src(fields: List[Tuple[str, int]], vals: Dict[str, int]) -> str:
    field_decls = ", ".join(f"{n}: u8" for n, _ in fields)
    inits = ", ".join(f"{n}: {vals[n]}" for n, _ in fields)
    return (
        "#[repr(C)]\n"
        f"struct S {{ {field_decls} }}\n"
        "fn main(){\n"
        f"    let s = S {{ {inits} }};\n"
        "    let p = &s as *const S as *const u8;\n"
        "    let n = std::mem::size_of::<S>();\n"
        "    print!(\"size={} bytes=\", n);\n"
        "    for i in 0..n { unsafe { print!(\"{:02x}\", *p.add(i)); } }\n"
        "    println!();\n}\n"
    )


def _go_src(fields: List[Tuple[str, int]], vals: Dict[str, int]) -> str:
    names = ", ".join(n for n, _ in fields)
    inits = ", ".join(str(vals[n]) for n, _ in fields)
    return (
        "package main\n"
        "import (\n\t\"fmt\"\n\t\"unsafe\"\n)\n"
        f"type S struct {{ {names} uint8 }}\n"
        "func main() {\n"
        f"\ts := S{{{inits}}}\n"
        "\tn := unsafe.Sizeof(s)\n"
        "\tp := (*[1 << 20]byte)(unsafe.Pointer(&s))\n"
        "\tfmt.Printf(\"size=%d bytes=\", n)\n"
        "\tfor i := uintptr(0); i < n; i++ {\n\t\tfmt.Printf(\"%02x\", p[i])\n\t}\n"
        "\tfmt.Println()\n}\n"
    )


class _BitfieldBase(DivergenceOracle):
    divergence_class = BITFIELD_LAYOUT.key
    source_lang = "c"
    confirmation_mode = "defined_divergence"

    def applies_to(self, unit: Dict) -> bool:
        if unit.get("probe") not in (None, self.divergence_class):
            return False
        if unit.get("target_lang", self.target_lang) != self.target_lang:
            return False
        return unit.get("kind") == "bitfield_struct"

    def _target_src(self, fields, vals) -> str:  # pragma: no cover - overridden
        raise NotImplementedError

    def find_divergence(self, unit: Dict) -> OracleResult:
        if not self.applies_to(unit):
            return OracleResult(OracleVerdict.NOT_APPLICABLE, self.divergence_class,
                                detail="unit is not a bit-field struct")
        fields = _fields(unit)
        if sum(w for _, w in fields) > 32 or any(w > 8 for _, w in fields):
            return OracleResult(OracleVerdict.NO_DIVERGENCE_FOUND, self.divergence_class,
                                detail="field group out of modelled range (<=32 total, <=8 each)")
        vals = _find_field_values(fields, unit)
        if not vals:
            return OracleResult(OracleVerdict.NO_DIVERGENCE_FOUND, self.divergence_class,
                                detail="no nonzero field assignment in declared ranges")
        ce = self._build(fields, vals)
        return OracleResult(OracleVerdict.DIVERGENT, self.divergence_class,
                            counterexample=ce,
                            detail="Z3 witness " + ", ".join(f"{n}={vals[n]}" for n, _ in fields))

    def _build(self, fields, vals) -> Counterexample:
        c_img = _c_image(fields, vals)
        t_img = _target_image(fields, vals)
        return Counterexample(
            divergence_class=self.divergence_class,
            source_lang="c", target_lang=self.target_lang,
            inputs=dict(vals),
            source_snippet=_c_src(fields, vals),
            target_snippet=self._target_src(fields, vals),
            source_definedness=Definedness.IMPLEMENTATION_DEFINED.value,
            divergence_witness=(
                f"C packs {', '.join(f'{n}:{w}' for n, w in fields)} into one "
                f"4-byte storage unit -> size=4 bytes={c_img}, while the faithful "
                f"{self.target_lang} field-by-field port is unpacked -> "
                f"size={len(fields)} bytes={t_img}. Both programs are fully "
                f"defined and deterministic, yet their ABI byte image differs: a "
                f"struct serialised across a wire/FFI boundary changes meaning "
                f"under the port."
            ),
            definedness_witness=(
                "Every field value is in its declared range; neither language "
                "has undefined behaviour. The divergence is purely the "
                "implementation-defined C bit-field packing vs the unpacked "
                "target layout."
            ),
        )


class BitfieldLayoutOracle(_BitfieldBase):
    """C packed bit-fields vs Rust's unpacked ``#[repr(C)]`` fields."""

    target_lang = "rust"

    def _target_src(self, fields, vals) -> str:
        return _rust_src(fields, vals)


class GoBitfieldLayoutOracle(_BitfieldBase):
    """C packed bit-fields vs Go's unpacked ``uint8`` struct fields."""

    target_lang = "go"

    def _target_src(self, fields, vals) -> str:
        return _go_src(fields, vals)


register(BitfieldLayoutOracle())
register(GoBitfieldLayoutOracle())
