"""
ABI / interop-divergence checks at FFI boundaries (100_STEPS step 79).

Every other oracle in this project reasons about *values* — what a function
computes on an input. But when translated code talks to its source across an FFI
boundary (the overwhelmingly common migration pattern: a C↔Rust shim, a cgo
wrapper, a Swift C-interop header), an entire second class of divergence appears
that has nothing to do with arithmetic: the two sides can disagree about the
**memory layout** of a shared aggregate — its `size`, its `align`, and the
`offset` of each field. A `struct` passed by pointer whose fields sit at
different offsets on each side is silently misread: a layout divergence is a
memory-safety bug that no value-level oracle can see.

This module decides layout divergence structurally, with the same
ground-truth discipline as the rest of the tool:

* :func:`c_layout` computes the **C ABI** layout of a struct (field offsets,
  total size, alignment) under the LP64 rules shared by x86-64 SysV and AArch64
  for the scalar types we model — fully specified, so it is *exact* and confirmed
  field-by-field against real ``clang`` ``offsetof``/``sizeof``.
* :func:`optimized_layout` models a **layout-optimizing** representation (fields
  reordered by descending alignment to minimize padding) — exactly what a target
  language's *idiomatic, non-FFI* representation is free to do (Rust's default
  ``repr(Rust)`` is the canonical example).
* :func:`abi_divergence` flags an FFI hazard **iff** those two layouts differ
  (in size or any field offset). When they coincide, the declared field order is
  already padding-optimal, so even an optimizing representation cannot permute
  it — the aggregate is interop-safe.

The confirmation closes the loop against three real compilers at once and is
pair-aware via the target-semantics packs' notion of *layout discipline*:

* **Rust** — ``#[repr(C)]`` must reproduce the C layout **exactly** (a positive
  interop-safety result), while the default ``repr(Rust)`` really diverges
  whenever — and only when — :func:`abi_divergence` predicted a hazard.
* **Go** and **C** lay structs out in **declaration order**, so a Go struct
  reproduces the C layout exactly: Go is layout-stable, the hazard is specific to
  representations that reorder.

Because the oracle predicts the *exact* C layout (verified) and only ever flags a
divergence the real default-``repr`` layout is then observed to exhibit, it never
fabricates an interop hazard — it abstains (``interop_safe``) whenever the
declared order is already optimal.
"""

from __future__ import annotations

import os
import subprocess
import tempfile
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple


# --- scalar type model (LP64; x86-64 SysV and AArch64 agree for these) -------
#
# name -> (size, alignment). These are the building blocks of the aggregates we
# reason about; each carries the C spelling, the Rust scalar and the Go scalar so
# the source generators stay data-driven.

@dataclass(frozen=True)
class Scalar:
    name: str
    size: int
    align: int
    c_type: str
    rust_type: str
    go_type: str


_SCALARS: Dict[str, Scalar] = {
    "char":     Scalar("char", 1, 1, "char", "i8", "int8"),
    "uchar":    Scalar("uchar", 1, 1, "unsigned char", "u8", "uint8"),
    "short":    Scalar("short", 2, 2, "short", "i16", "int16"),
    "int":      Scalar("int", 4, 4, "int", "i32", "int32"),
    "longlong": Scalar("longlong", 8, 8, "long long", "i64", "int64"),
    "float":    Scalar("float", 4, 4, "float", "f32", "float32"),
    "double":   Scalar("double", 8, 8, "double", "f64", "float64"),
}


def scalar(name: str) -> Scalar:
    if name not in _SCALARS:
        raise KeyError(f"unknown scalar type {name!r}; known: {sorted(_SCALARS)}")
    return _SCALARS[name]


# --- layouts -----------------------------------------------------------------


@dataclass(frozen=True)
class Field:
    name: str
    type: str  # a key into _SCALARS

    @property
    def scalar(self) -> Scalar:
        return scalar(self.type)


@dataclass(frozen=True)
class Layout:
    """A concrete struct layout: per-field offsets, total size and alignment."""

    order: Tuple[str, ...]            # field names in memory order
    offsets: Dict[str, int]
    size: int
    align: int

    def differs_from(self, other: "Layout") -> bool:
        if self.size != other.size or self.align != other.align:
            return True
        # compare offsets by field name (order-independent: a reorder that keeps
        # every field at the same byte is not a divergence).
        return any(self.offsets.get(n) != other.offsets.get(n)
                   for n in set(self.offsets) | set(other.offsets))


def _round_up(x: int, a: int) -> int:
    return (x + a - 1) // a * a


def _lay_out(fields: List[Field]) -> Layout:
    """Place ``fields`` in the given order under the standard C aggregate rules."""
    offset = 0
    align = 1
    offsets: Dict[str, int] = {}
    for f in fields:
        s = f.scalar
        offset = _round_up(offset, s.align)
        offsets[f.name] = offset
        offset += s.size
        align = max(align, s.align)
    size = _round_up(offset, align) if fields else 0
    return Layout(tuple(f.name for f in fields), offsets, size, max(align, 1))


def c_layout(fields: List[Field]) -> Layout:
    """The C ABI layout: declaration order, natural alignment, tail padding."""
    return _lay_out(fields)


def optimized_layout(fields: List[Field]) -> Layout:
    """A padding-minimizing layout: fields sorted by *descending* alignment.

    This models the freedom a non-FFI target representation has to reorder
    fields (Rust's default ``repr(Rust)`` is the canonical instance). The sort is
    stable so fields of equal alignment keep their declared relative order, which
    matches the simple-struct behaviour of real ``rustc`` default layout.
    """
    ordered = sorted(fields, key=lambda f: -f.scalar.align)
    return _lay_out(ordered)


# --- the divergence decision -------------------------------------------------


class AbiVerdict(Enum):
    INTEROP_HAZARD = "interop_hazard"   # repr(C) and optimized layouts differ
    INTEROP_SAFE = "interop_safe"       # declared order is already optimal

    def __str__(self) -> str:  # pragma: no cover - trivial
        return self.value


@dataclass
class AbiResult:
    verdict: AbiVerdict
    c: Layout
    optimized: Layout
    #: fields whose offset moves between the two layouts (the misread fields).
    moved_fields: List[str] = field(default_factory=list)
    reason: str = ""

    @property
    def is_hazard(self) -> bool:
        return self.verdict is AbiVerdict.INTEROP_HAZARD


def abi_divergence(fields: List[Field]) -> AbiResult:
    """Flag an FFI layout hazard iff the C and optimized layouts differ."""
    cl = c_layout(fields)
    opt = optimized_layout(fields)
    if cl.differs_from(opt):
        moved = sorted(n for n in cl.offsets
                       if cl.offsets.get(n) != opt.offsets.get(n))
        return AbiResult(
            AbiVerdict.INTEROP_HAZARD, cl, opt, moved_fields=moved,
            reason=(f"C layout size={cl.size} offsets={cl.offsets} but an "
                    f"optimizing representation yields size={opt.size} "
                    f"offsets={opt.offsets}; fields {moved} are misread across "
                    f"the FFI boundary"))
    return AbiResult(
        AbiVerdict.INTEROP_SAFE, cl, opt,
        reason=(f"declared order is already padding-optimal "
                f"(size={cl.size}, offsets={cl.offsets}); no reordering possible"))


# --- real-compiler source generators -----------------------------------------


def _struct_fields_decl_c(fields: List[Field]) -> str:
    return " ".join(f"{f.scalar.c_type} {f.name};" for f in fields)


def c_source(fields: List[Field]) -> str:
    """A C program printing ``size off0 off1 …`` for the struct in declared order."""
    decl = _struct_fields_decl_c(fields)
    offs = ", ".join(f"offsetof(struct S,{f.name})" for f in fields)
    fmt = "%zu" + (" %zu" * len(fields))
    return (
        "#include <stdio.h>\n#include <stddef.h>\n"
        f"struct S {{ {decl} }};\n"
        "int main(void){\n"
        f"    printf(\"{fmt}\\n\", sizeof(struct S), {offs});\n"
        "    return 0;\n"
        "}\n"
    )


def rust_source(fields: List[Field]) -> str:
    """A Rust program printing the ``#[repr(C)]`` *and* default-repr layouts."""
    rc = " ".join(f"{f.name}: {f.scalar.rust_type}," for f in fields)
    def line(struct: str) -> str:
        sz = f"std::mem::size_of::<{struct}>()"
        offs = ", ".join(f"std::mem::offset_of!({struct},{f.name})" for f in fields)
        placeholders = "{}" + (" {}" * len(fields))
        return f'    println!("{placeholders}", {sz}, {offs});'
    return (
        f"#[repr(C)]\nstruct ReprC {{ {rc} }}\n"
        f"struct Natural {{ {rc} }}\n"
        "fn main(){\n"
        + line("ReprC") + "\n"
        + line("Natural") + "\n"
        "}\n"
    )


def go_source(fields: List[Field]) -> str:
    """A Go program printing the struct layout (declaration order)."""
    decl = "; ".join(f"{f.name} {f.scalar.go_type}" for f in fields)
    offs = ", ".join(f"unsafe.Offsetof(s.{f.name})" for f in fields)
    verbs = "%d" + (" %d" * len(fields))
    return (
        "package main\n"
        'import ("fmt"; "unsafe")\n'
        f"type S struct {{ {decl} }}\n"
        "func main(){\n"
        "    var s S\n"
        f"    fmt.Printf(\"{verbs}\\n\", unsafe.Sizeof(s), {offs})\n"
        "}\n"
    )


# --- confirmation against real toolchains ------------------------------------


@dataclass
class AbiConfirmation:
    available: bool
    reason: str = ""
    c_size: Optional[int] = None
    c_offsets: Dict[str, int] = field(default_factory=dict)
    rust_reprc_matches_c: Optional[bool] = None
    rust_natural_size: Optional[int] = None
    rust_natural_offsets: Dict[str, int] = field(default_factory=dict)
    rust_natural_diverges: Optional[bool] = None
    go_matches_c: Optional[bool] = None


def _run(argv: List[str], **kw) -> subprocess.CompletedProcess:
    return subprocess.run(argv, capture_output=True, text=True, timeout=120, **kw)


def _parse_layout_line(line: str, names: List[str]) -> Tuple[int, Dict[str, int]]:
    nums = [int(x) for x in line.split()]
    return nums[0], {n: nums[i + 1] for i, n in enumerate(names)}


def confirm_abi(fields: List[Field], cc: str, rustc: Optional[str] = None,
                go: Optional[str] = None) -> AbiConfirmation:
    """Compile & run the C/Rust/Go probes and read back the real layouts."""
    names = [f.name for f in fields]
    with tempfile.TemporaryDirectory() as d:
        # C ground truth
        cp = os.path.join(d, "s.c")
        co = os.path.join(d, "s.out")
        with open(cp, "w") as fh:
            fh.write(c_source(fields))
        r = _run([cc, "-O0", "-o", co, cp])
        if r.returncode != 0:
            return AbiConfirmation(False, reason=f"C compile failed: {r.stderr[:200]}")
        c_size, c_offsets = _parse_layout_line(_run([co]).stdout.strip(), names)

        out = AbiConfirmation(True, c_size=c_size, c_offsets=c_offsets)

        if rustc is not None:
            rp = os.path.join(d, "s.rs")
            ro = os.path.join(d, "r.out")
            with open(rp, "w") as fh:
                fh.write(rust_source(fields))
            r = _run([rustc, "-O", "-o", ro, rp])
            if r.returncode == 0:
                lines = _run([ro]).stdout.strip().splitlines()
                rc_size, rc_offsets = _parse_layout_line(lines[0], names)
                nat_size, nat_offsets = _parse_layout_line(lines[1], names)
                out.rust_reprc_matches_c = (rc_size == c_size and rc_offsets == c_offsets)
                out.rust_natural_size = nat_size
                out.rust_natural_offsets = nat_offsets
                out.rust_natural_diverges = (nat_size != c_size or nat_offsets != c_offsets)
            else:  # pragma: no cover - environment dependent
                out.reason += f" rust compile failed: {r.stderr[:160]}"

        if go is not None:
            gp = os.path.join(d, "m.go")
            go_out = os.path.join(d, "m.out")
            with open(gp, "w") as fh:
                fh.write(go_source(fields))
            env = dict(os.environ)
            env["GOCACHE"] = os.path.join(d, "gocache")
            env["GOPATH"] = os.path.join(d, "gopath")
            r = _run([go, "build", "-o", go_out, gp], env=env)
            if r.returncode == 0:
                g_size, g_offsets = _parse_layout_line(_run([go_out]).stdout.strip(), names)
                out.go_matches_c = (g_size == c_size and g_offsets == c_offsets)
            else:  # pragma: no cover - environment dependent
                out.reason += f" go compile failed: {r.stderr[:160]}"

    return out


# --- ready-made fragments ----------------------------------------------------


def hazard_struct() -> List[Field]:
    """``{char a; int b; char c}`` — declared order wastes padding, so an
    optimizing representation reorders it (the classic FFI hazard)."""
    return [Field("a", "char"), Field("b", "int"), Field("c", "char")]


def safe_struct() -> List[Field]:
    """``{int b; char a; char c}`` — already padding-optimal, so every
    representation agrees (interop-safe)."""
    return [Field("b", "int"), Field("a", "char"), Field("c", "char")]


def uniform_struct() -> List[Field]:
    """``{int x; int y; int z}`` — uniform alignment, no padding to reclaim."""
    return [Field("x", "int"), Field("y", "int"), Field("z", "int")]
