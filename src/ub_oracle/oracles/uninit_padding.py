"""
Uninitialized-padding read oracle (100_STEPS step 114).

C struct padding is not a field value.  After assigning every declared member,
the bytes that correspond to padding have unspecified values (C17 6.2.6.1p6).
Real code nevertheless often hashes, compares, writes, or sends a whole struct
object representation with ``memcpy``/``write``.  That leaks an indeterminate
padding value into observable behavior.

The oracle models the smallest common migration bug: a naturally padded C struct
is serialized byte-for-byte, while the safe target translation serializes fields
into an explicitly zero-initialized byte buffer.  The emitted C validates the
assumed ABI with ``_Static_assert(sizeof/offsetof)`` before any run.  Ground-truth
confirmation uses MemorySanitizer when available and otherwise clang's real
``-ftrivial-auto-var-init={pattern,zero}`` modes to prove the digest depends
specifically on padding bytes; a runtime switch zeroes padding for the safe
negative control.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, List, Tuple

from ..catalogue import UNINIT_PADDING, Definedness
from ..plugin import DivergenceOracle, OracleResult, OracleVerdict, register
from ..replay import Counterexample


@dataclass(frozen=True)
class _TypeInfo:
    c_type: str
    size: int
    align: int
    rust_type: str
    go_type: str
    go_put: str


_TYPES = {
    "u8": _TypeInfo("uint8_t", 1, 1, "u8", "uint8", ""),
    "u16": _TypeInfo("uint16_t", 2, 2, "u16", "uint16", "PutUint16"),
    "u32": _TypeInfo("uint32_t", 4, 4, "u32", "uint32", "PutUint32"),
    "u64": _TypeInfo("uint64_t", 8, 8, "u64", "uint64", "PutUint64"),
}
_IDENT = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


@dataclass(frozen=True)
class _Field:
    name: str
    typ: str
    value: int

    @property
    def info(self) -> _TypeInfo:
        return _TYPES[self.typ]


@dataclass(frozen=True)
class _Layout:
    offsets: Dict[str, int]
    size: int
    align: int
    padding: Tuple[int, ...]


def _align_up(n: int, align: int) -> int:
    return ((n + align - 1) // align) * align


def _normalize_fields(unit: Dict) -> Tuple[_Field, ...]:
    raw = unit.get("fields")
    if raw is None:
        raw = [
            {"name": "tag", "type": "u8", "value": 7},
            {"name": "value", "type": "u32", "value": 0x01020304},
        ]
    if not isinstance(raw, list) or not raw:
        raise ValueError("fields must be a non-empty list")

    values = unit.get("values", {})
    if values is None:
        values = {}
    if not isinstance(values, dict):
        raise ValueError("values must be a mapping when present")

    out: List[_Field] = []
    seen = set()
    for idx, spec in enumerate(raw):
        if not isinstance(spec, dict):
            raise ValueError(f"fields[{idx}] must be a mapping")
        name = spec.get("name")
        typ = spec.get("type", "u32")
        if not isinstance(name, str) or not _IDENT.match(name):
            raise ValueError(f"invalid field name {name!r}")
        if name in seen:
            raise ValueError(f"duplicate field name {name!r}")
        if typ not in _TYPES:
            raise ValueError(f"unsupported field type {typ!r}")
        raw_value = values.get(name, spec.get("value", idx + 1))
        if isinstance(raw_value, bool) or not isinstance(raw_value, int):
            raise ValueError(f"value for field {name!r} must be an integer")
        mask = (1 << (_TYPES[typ].size * 8)) - 1
        out.append(_Field(name, typ, raw_value & mask))
        seen.add(name)
    return tuple(out)


def _layout(fields: Tuple[_Field, ...], *, packed: bool = False) -> _Layout:
    offset = 0
    max_align = 1
    offsets: Dict[str, int] = {}
    padding: List[int] = []
    for f in fields:
        align = 1 if packed else f.info.align
        max_align = max(max_align, align)
        aligned = _align_up(offset, align)
        padding.extend(range(offset, aligned))
        offsets[f.name] = aligned
        offset = aligned + f.info.size
    size = _align_up(offset, max_align)
    padding.extend(range(offset, size))
    return _Layout(offsets=offsets, size=size, align=max_align,
                   padding=tuple(padding))


def _c_struct_decl(fields: Tuple[_Field, ...], packed: bool) -> str:
    attr = " __attribute__((packed))" if packed else ""
    members = "".join(f"    {f.info.c_type} {f.name};\n" for f in fields)
    return f"struct{attr} P {{\n{members}}};\n"


def _c_source(fields: Tuple[_Field, ...], layout: _Layout, packed: bool) -> str:
    params = ", ".join(f"unsigned long long v{i}" for i, _ in enumerate(fields))
    assigns = "".join(
        f"    p.{f.name} = ({f.info.c_type})v{i};\n"
        for i, f in enumerate(fields)
    )
    parses = []
    args = []
    for i, f in enumerate(fields):
        argn = i + 1
        parses.append(
            f"    unsigned long long v{i} = argc > {argn} ? "
            f"strtoull(argv[{argn}], 0, 10) : {f.value}ull;\n"
        )
        args.append(f"v{i}")
    expose_idx = len(fields) + 1
    static_asserts = [
        f"_Static_assert(sizeof(struct P) == {layout.size}, \"struct size drift\");\n",
        f"_Static_assert(sizeof(struct P) > {sum(f.info.size for f in fields)}, "
        "\"no padding in struct P\");\n",
    ]
    for f in fields:
        static_asserts.append(
            f"_Static_assert(offsetof(struct P, {f.name}) == "
            f"{layout.offsets[f.name]}, \"{f.name} offset drift\");\n"
        )
    return (
        "#include <stddef.h>\n"
        "#include <stdint.h>\n"
        "#include <stdio.h>\n"
        "#include <stdlib.h>\n"
        "#include <string.h>\n"
        + _c_struct_decl(fields, packed)
        + "".join(static_asserts)
        + f"__attribute__((noinline)) static uint32_t digest({params}, int expose_padding){{\n"
        "    struct P p;\n"
        "    if (!expose_padding) memset(&p, 0, sizeof p);\n"
        "#ifdef CLV_ZERO_PADDING\n"
        "    memset(&p, 0, sizeof p);\n"
        "#endif\n"
        + assigns
        + "    unsigned char bytes[sizeof p];\n"
        "    memcpy(bytes, &p, sizeof p);\n"
        "    uint32_t acc = 0;\n"
        "    for (size_t i = 0; i < sizeof bytes; ++i) acc = acc * 131u + bytes[i];\n"
        "    return acc;\n"
        "}\n"
        "int main(int argc, char **argv){\n"
        + "".join(parses)
        + f"    int expose_padding = argc > {expose_idx} ? atoi(argv[{expose_idx}]) : 1;\n"
        f"    printf(\"%u\\n\", digest({', '.join(args)}, expose_padding));\n"
        "    return 0;\n"
        "}\n"
    )


def _rust_source(fields: Tuple[_Field, ...], layout: _Layout) -> str:
    lines = [
        "fn main() {\n",
        "    let args: Vec<String> = std::env::args().collect();\n",
        f"    let mut bytes = [0u8; {layout.size}];\n",
    ]
    for i, f in enumerate(fields):
        argn = i + 1
        off = layout.offsets[f.name]
        lines.append(
            f"    let v{i}: u64 = args.get({argn}).and_then(|s| s.parse().ok()).unwrap_or({f.value});\n"
        )
        if f.info.size == 1:
            lines.append(f"    bytes[{off}] = v{i} as u8;\n")
        else:
            end = off + f.info.size
            lines.append(
                f"    bytes[{off}..{end}].copy_from_slice(&((v{i} as "
                f"{f.info.rust_type}).to_le_bytes()));\n"
            )
    lines += [
        "    let mut acc: u32 = 0;\n",
        "    for b in bytes { acc = acc.wrapping_mul(131).wrapping_add(b as u32); }\n",
        "    println!(\"{}\", acc);\n",
        "}\n",
    ]
    return "".join(lines)


def _go_source(fields: Tuple[_Field, ...], layout: _Layout) -> str:
    lines = [
        "package main\n",
        "import (\n\t\"encoding/binary\"\n\t\"fmt\"\n\t\"os\"\n\t\"strconv\"\n)\n",
        "func main() {\n",
        f"\tbytes := make([]byte, {layout.size})\n",
    ]
    for i, f in enumerate(fields):
        argn = i + 1
        off = layout.offsets[f.name]
        end = off + f.info.size
        lines.append(
            f"\tv{i} := uint64({f.value})\n"
            f"\tif len(os.Args) > {argn} {{ parsed, _ := strconv.ParseUint(os.Args[{argn}], 10, 64); v{i} = parsed }}\n"
        )
        if f.info.size == 1:
            lines.append(f"\tbytes[{off}] = byte(v{i})\n")
        else:
            lines.append(
                f"\tbinary.LittleEndian.{f.info.go_put}(bytes[{off}:{end}], "
                f"{f.info.go_type}(v{i}))\n"
            )
    lines += [
        "\tvar acc uint32\n",
        "\tfor _, b := range bytes { acc = acc*131 + uint32(b) }\n",
        "\tfmt.Println(acc)\n",
        "}\n",
    ]
    return "".join(lines)


def _sources(unit: Dict, target_lang: str) -> Tuple[Counterexample, _Layout]:
    fields = _normalize_fields(unit)
    packed = bool(unit.get("packed", False))
    layout = _layout(fields, packed=packed)
    inputs = {f.name: f.value for f in fields}
    inputs["expose_padding"] = int(unit.get("expose_padding", 1))
    target = {
        "rust": _rust_source,
        "go": _go_source,
    }[target_lang](fields, layout)
    ce = Counterexample(
        divergence_class=UNINIT_PADDING.key,
        source_lang="c",
        target_lang=target_lang,
        inputs=inputs,
        source_snippet=_c_source(fields, layout, packed),
        target_snippet=target,
        source_definedness=Definedness.UNSPECIFIED.value,
        divergence_witness=(
            f"C serializes `sizeof(struct P)` bytes after assigning all {len(fields)} "
            f"field(s).  The compiler-validated layout has padding byte(s) "
            f"{list(layout.padding)}, whose values are unspecified; MSan or the "
            f"pattern-vs-zero auto-init delta observes them in the digest.  The "
            f"{target_lang} translation writes fields into a zero-initialized byte "
            f"buffer, so padding is deterministic."
        ),
        definedness_witness=(
            "All declared fields are initialized from integer inputs before the "
            "object representation is copied; the under-defined source fact is "
            "only the padding bytes that are not C values."
        ),
    )
    return ce, layout


def demo_sources(target_lang: str = "rust") -> Tuple[str, str, Tuple[str, ...], Tuple[str, ...]]:
    """Return the default zoo/demo snippets and witness/safe inputs."""
    ce, _layout = _sources({}, target_lang)
    vals = [str(v) for v in ce.inputs.values()]
    safe = list(vals)
    safe[-1] = "0"
    return ce.source_snippet, ce.target_snippet, tuple(vals), tuple(safe)


class _UninitPaddingBase(DivergenceOracle):
    divergence_class = UNINIT_PADDING.key
    source_lang = "c"
    confirmation_mode = "uninit_padding"

    def applies_to(self, unit: Dict) -> bool:
        if unit.get("probe") not in (None, self.divergence_class):
            return False
        return unit.get("kind") == "uninit_padding"

    def find_divergence(self, unit: Dict) -> OracleResult:
        if not self.applies_to(unit):
            return OracleResult(OracleVerdict.NOT_APPLICABLE, self.divergence_class,
                                detail="unit is not an uninit_padding serialization")
        try:
            ce, layout = _sources(unit, self.target_lang)
        except (KeyError, ValueError) as e:
            return OracleResult(OracleVerdict.NOT_APPLICABLE, self.divergence_class,
                                detail=f"ill-formed uninit_padding unit: {e}")
        if not layout.padding:
            return OracleResult(OracleVerdict.NO_DIVERGENCE_FOUND, self.divergence_class,
                                detail="declared struct layout contains no padding bytes")
        return OracleResult(
            OracleVerdict.DIVERGENT,
            self.divergence_class,
            counterexample=ce,
            detail=f"compiler-asserted layout has padding bytes {list(layout.padding)}",
        )


class UninitPaddingOracle(_UninitPaddingBase):
    """C struct-padding serialization vs Rust zero-padded safe serialization."""

    target_lang = "rust"


class GoUninitPaddingOracle(_UninitPaddingBase):
    """C struct-padding serialization vs Go zero-padded safe serialization."""

    target_lang = "go"


register(UninitPaddingOracle())
register(GoUninitPaddingOracle())
