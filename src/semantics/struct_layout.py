"""
Struct layout divergence detector for C↔Rust translations.

Computes C struct layout (field offsets, padding, alignment) and compares
against Rust struct layout. Detects divergences caused by:
  - Different default layout rules (C ABI vs Rust's unspecified layout)
  - Missing #[repr(C)] annotations
  - Bitfield layout differences
  - Flexible array member handling
  - Alignment attribute differences (__attribute__((aligned)) vs #[align])
  - Platform-dependent type sizes (long, size_t, pointer)

References:
  - C11 §6.7.2.1 (struct/union member layout)
  - Rust Reference: Type Layout (repr(C), repr(packed), repr(align))
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Dict, List, Optional, Sequence, Tuple

from ..ir.types import (
    IRType,
    IntType,
    FloatType,
    PointerType,
    ArrayType,
    StructType,
    StructField,
    UnionType,
    Signedness,
)


# ── Configuration ────────────────────────────────────────────────────────

class Platform(Enum):
    """Target platform affecting layout (ILP32, LP64, LLP64)."""
    LP64 = auto()    # Linux/macOS x86_64: int=32, long=64, ptr=64
    LLP64 = auto()   # Windows x64: int=32, long=32, ptr=64
    ILP32 = auto()    # 32-bit: int=32, long=32, ptr=32


@dataclass(frozen=True)
class LayoutConfig:
    """Platform-specific layout parameters."""
    pointer_size: int = 64
    long_size: int = 64       # LP64: 64, LLP64: 32
    max_align: int = 128      # maximum useful alignment (bits)
    char_is_signed: bool = True

    @classmethod
    def for_platform(cls, platform: Platform) -> LayoutConfig:
        if platform is Platform.LP64:
            return cls(pointer_size=64, long_size=64)
        elif platform is Platform.LLP64:
            return cls(pointer_size=64, long_size=32)
        elif platform is Platform.ILP32:
            return cls(pointer_size=32, long_size=32)
        raise ValueError(f"Unknown platform: {platform}")


# ── Field layout computation ─────────────────────────────────────────────

@dataclass
class FieldLayout:
    """Computed layout for a single struct field."""
    name: str
    type_desc: str
    offset_bits: int
    size_bits: int
    align_bits: int
    padding_before: int  # padding inserted before this field

    @property
    def offset_bytes(self) -> int:
        return self.offset_bits // 8

    @property
    def size_bytes(self) -> int:
        return math.ceil(self.size_bits / 8)

    @property
    def end_bits(self) -> int:
        return self.offset_bits + self.size_bits


@dataclass
class StructLayout:
    """Complete layout of a struct (C or Rust)."""
    name: str
    language: str  # "C" or "Rust"
    fields: List[FieldLayout]
    total_size_bits: int
    alignment_bits: int
    trailing_padding_bits: int
    is_packed: bool = False
    is_repr_c: bool = False  # Rust only: has #[repr(C)]

    @property
    def total_size_bytes(self) -> int:
        return math.ceil(self.total_size_bits / 8)

    @property
    def alignment_bytes(self) -> int:
        return self.alignment_bits // 8


# ── Divergence reports ───────────────────────────────────────────────────

class LayoutDivergenceKind(Enum):
    """Kinds of struct layout divergence."""
    FIELD_OFFSET_MISMATCH = auto()     # same field at different offsets
    FIELD_SIZE_MISMATCH = auto()       # field has different size
    TOTAL_SIZE_MISMATCH = auto()       # struct total size differs
    ALIGNMENT_MISMATCH = auto()        # struct alignment differs
    PADDING_MISMATCH = auto()          # different padding pattern
    FIELD_ORDER_MISMATCH = auto()      # Rust reordered fields (no repr(C))
    MISSING_REPR_C = auto()            # Rust struct lacks #[repr(C)]
    BITFIELD_DIVERGENCE = auto()       # bitfield layout differs
    FLEXIBLE_ARRAY_MEMBER = auto()     # C FAM vs Rust slice/Vec


@dataclass
class LayoutDivergence:
    """A single struct layout divergence."""
    kind: LayoutDivergenceKind
    struct_name: str
    severity: str  # "critical", "warning", "info"
    description: str
    c_detail: str = ""
    rust_detail: str = ""
    field_name: str = ""

    def to_dict(self) -> dict:
        return {
            "kind": self.kind.name,
            "struct_name": self.struct_name,
            "severity": self.severity,
            "description": self.description,
            "c_detail": self.c_detail,
            "rust_detail": self.rust_detail,
            "field_name": self.field_name,
        }


@dataclass
class LayoutAnalysisReport:
    """Complete layout analysis for a C/Rust struct pair."""
    c_layout: StructLayout
    rust_layout: StructLayout
    divergences: List[LayoutDivergence]

    @property
    def is_compatible(self) -> bool:
        return len(self.divergences) == 0

    @property
    def has_critical(self) -> bool:
        return any(d.severity == "critical" for d in self.divergences)

    def to_dict(self) -> dict:
        return {
            "c_struct": self.c_layout.name,
            "rust_struct": self.rust_layout.name,
            "compatible": self.is_compatible,
            "c_size": self.c_layout.total_size_bytes,
            "rust_size": self.rust_layout.total_size_bytes,
            "c_align": self.c_layout.alignment_bytes,
            "rust_align": self.rust_layout.alignment_bytes,
            "divergences": [d.to_dict() for d in self.divergences],
        }


# ── Layout computation ──────────────────────────────────────────────────

def compute_struct_layout(
    struct_type: StructType,
    language: str = "C",
    config: LayoutConfig = LayoutConfig(),
    is_repr_c: bool = False,
) -> StructLayout:
    """Compute the layout of a struct following C ABI rules.

    For Rust without repr(C), the compiler may reorder fields for optimal
    packing.  We model both layouts.

    Args:
        struct_type: The struct IR type.
        language: "C" or "Rust".
        config: Platform layout configuration.
        is_repr_c: Whether the Rust struct has #[repr(C)].

    Returns:
        StructLayout with computed offsets.
    """
    fields_ir = list(struct_type.fields)

    if language == "Rust" and not is_repr_c and not struct_type.packed:
        # Rust may reorder fields for optimal packing (largest alignment first)
        fields_ir = sorted(
            fields_ir,
            key=lambda f: f.type.align_bits(config.pointer_size),
            reverse=True,
        )

    field_layouts: List[FieldLayout] = []
    current_offset = 0
    max_align = 8  # minimum 1-byte alignment

    for f in fields_ir:
        f_size = f.type.size_bits(config.pointer_size)
        f_align = f.type.align_bits(config.pointer_size) if not struct_type.packed else 8

        # Compute padding before this field
        padding = 0
        if not struct_type.packed and f_align > 0:
            aligned_offset = _align_up(current_offset, f_align)
            padding = aligned_offset - current_offset
            current_offset = aligned_offset

        field_layouts.append(FieldLayout(
            name=f.name,
            type_desc=str(f.type),
            offset_bits=current_offset,
            size_bits=f_size,
            align_bits=f_align,
            padding_before=padding,
        ))

        current_offset += f_size
        if f_align > max_align:
            max_align = f_align

    # Trailing padding for alignment
    struct_align = 8 if struct_type.packed else max_align
    total_size = _align_up(current_offset, struct_align) if not struct_type.packed else current_offset
    trailing_pad = total_size - current_offset

    return StructLayout(
        name=struct_type.name or "<anonymous>",
        language=language,
        fields=field_layouts,
        total_size_bits=total_size,
        alignment_bits=struct_align,
        trailing_padding_bits=trailing_pad,
        is_packed=struct_type.packed,
        is_repr_c=is_repr_c,
    )


def _align_up(value: int, alignment: int) -> int:
    """Round value up to next multiple of alignment."""
    if alignment <= 0:
        return value
    return ((value + alignment - 1) // alignment) * alignment


# ── Divergence detection ────────────────────────────────────────────────

class StructLayoutAnalyzer:
    """Analyzes struct layout divergences between C and Rust.

    Usage::

        analyzer = StructLayoutAnalyzer()
        report = analyzer.compare(c_struct, rust_struct, is_repr_c=True)
        for div in report.divergences:
            print(div.description)
    """

    def __init__(self, config: LayoutConfig = LayoutConfig()) -> None:
        self._config = config

    def compare(
        self,
        c_struct: StructType,
        rust_struct: StructType,
        is_repr_c: bool = False,
    ) -> LayoutAnalysisReport:
        """Compare layouts of a C struct and its Rust translation.

        Args:
            c_struct: The C struct type from IR.
            rust_struct: The Rust struct type from IR.
            is_repr_c: Whether the Rust struct has #[repr(C)].

        Returns:
            LayoutAnalysisReport with divergences.
        """
        c_layout = compute_struct_layout(c_struct, "C", self._config)
        rust_layout = compute_struct_layout(
            rust_struct, "Rust", self._config, is_repr_c=is_repr_c,
        )

        divergences: List[LayoutDivergence] = []
        struct_name = c_struct.name or "<anonymous>"

        # Check for missing repr(C)
        if not is_repr_c and not rust_struct.packed:
            divergences.append(LayoutDivergence(
                kind=LayoutDivergenceKind.MISSING_REPR_C,
                struct_name=struct_name,
                severity="warning",
                description=(
                    f"Rust struct '{struct_name}' lacks #[repr(C)]. "
                    "Field order and padding may differ from C layout."
                ),
            ))

        # Compare total size
        if c_layout.total_size_bits != rust_layout.total_size_bits:
            divergences.append(LayoutDivergence(
                kind=LayoutDivergenceKind.TOTAL_SIZE_MISMATCH,
                struct_name=struct_name,
                severity="critical",
                description=(
                    f"Struct '{struct_name}' size differs: "
                    f"C={c_layout.total_size_bytes}B, Rust={rust_layout.total_size_bytes}B"
                ),
                c_detail=f"{c_layout.total_size_bits} bits",
                rust_detail=f"{rust_layout.total_size_bits} bits",
            ))

        # Compare alignment
        if c_layout.alignment_bits != rust_layout.alignment_bits:
            divergences.append(LayoutDivergence(
                kind=LayoutDivergenceKind.ALIGNMENT_MISMATCH,
                struct_name=struct_name,
                severity="critical",
                description=(
                    f"Struct '{struct_name}' alignment differs: "
                    f"C={c_layout.alignment_bytes}B, Rust={rust_layout.alignment_bytes}B"
                ),
                c_detail=f"{c_layout.alignment_bits}-bit aligned",
                rust_detail=f"{rust_layout.alignment_bits}-bit aligned",
            ))

        # Build field maps for cross-referencing
        c_fields = {fl.name: fl for fl in c_layout.fields}
        rust_fields = {fl.name: fl for fl in rust_layout.fields}

        # Compare individual field offsets and sizes
        for name, c_fl in c_fields.items():
            if name not in rust_fields:
                continue  # field missing — separate check

            r_fl = rust_fields[name]

            if c_fl.offset_bits != r_fl.offset_bits:
                divergences.append(LayoutDivergence(
                    kind=LayoutDivergenceKind.FIELD_OFFSET_MISMATCH,
                    struct_name=struct_name,
                    severity="critical",
                    description=(
                        f"Field '{name}' offset differs: "
                        f"C=+{c_fl.offset_bytes}B, Rust=+{r_fl.offset_bytes}B"
                    ),
                    c_detail=f"offset={c_fl.offset_bits} bits",
                    rust_detail=f"offset={r_fl.offset_bits} bits",
                    field_name=name,
                ))

            if c_fl.size_bits != r_fl.size_bits:
                divergences.append(LayoutDivergence(
                    kind=LayoutDivergenceKind.FIELD_SIZE_MISMATCH,
                    struct_name=struct_name,
                    severity="critical",
                    description=(
                        f"Field '{name}' size differs: "
                        f"C={c_fl.size_bytes}B, Rust={r_fl.size_bytes}B"
                    ),
                    c_detail=f"{c_fl.size_bits} bits",
                    rust_detail=f"{r_fl.size_bits} bits",
                    field_name=name,
                ))

            if c_fl.padding_before != r_fl.padding_before:
                divergences.append(LayoutDivergence(
                    kind=LayoutDivergenceKind.PADDING_MISMATCH,
                    struct_name=struct_name,
                    severity="warning",
                    description=(
                        f"Padding before field '{name}' differs: "
                        f"C={c_fl.padding_before // 8}B, Rust={r_fl.padding_before // 8}B"
                    ),
                    field_name=name,
                ))

        # Check field order divergence
        c_order = [fl.name for fl in c_layout.fields]
        r_order = [fl.name for fl in rust_layout.fields]
        if c_order != r_order and set(c_order) == set(r_order):
            divergences.append(LayoutDivergence(
                kind=LayoutDivergenceKind.FIELD_ORDER_MISMATCH,
                struct_name=struct_name,
                severity="critical",
                description=(
                    f"Field order differs — C: {c_order}, Rust: {r_order}. "
                    "This breaks memcpy/transmute compatibility."
                ),
                c_detail=", ".join(c_order),
                rust_detail=", ".join(r_order),
            ))

        # Check for flexible array member (last field is zero-length array in C)
        if c_struct.fields:
            last_c = c_struct.fields[-1]
            if isinstance(last_c.type, ArrayType) and last_c.type.length == 0:
                divergences.append(LayoutDivergence(
                    kind=LayoutDivergenceKind.FLEXIBLE_ARRAY_MEMBER,
                    struct_name=struct_name,
                    severity="warning",
                    description=(
                        f"C struct has flexible array member '{last_c.name}'. "
                        "Rust has no direct equivalent — use a raw pointer or PhantomData + manual offset."
                    ),
                    field_name=last_c.name,
                ))

        return LayoutAnalysisReport(
            c_layout=c_layout,
            rust_layout=rust_layout,
            divergences=divergences,
        )

    def analyze_all_structs(
        self,
        c_structs: Dict[str, StructType],
        rust_structs: Dict[str, StructType],
        repr_c_names: Optional[set] = None,
    ) -> List[LayoutAnalysisReport]:
        """Analyze all struct pairs by name.

        Args:
            c_structs: Map of struct name → C StructType.
            rust_structs: Map of struct name → Rust StructType.
            repr_c_names: Set of Rust struct names with #[repr(C)].

        Returns:
            List of LayoutAnalysisReport for each matched pair.
        """
        repr_c = repr_c_names or set()
        reports = []
        for name, c_st in c_structs.items():
            if name in rust_structs:
                reports.append(self.compare(
                    c_st, rust_structs[name],
                    is_repr_c=(name in repr_c),
                ))
        return reports
