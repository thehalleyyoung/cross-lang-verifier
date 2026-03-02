"""
Complete type system for the Cross-Language Equivalence Verifier IR.

Provides types that can represent both C and Rust type systems, with support
for type compatibility checking, promotion rules, size/alignment computation,
lattice operations, and serialization. Handles the differences between C and
Rust semantics (e.g., signed overflow is UB in C but wrapping in Rust).
"""

from __future__ import annotations

import json
import math
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Optional, Sequence


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class Signedness(Enum):
    """Whether an integer type is signed or unsigned."""
    SIGNED = auto()
    UNSIGNED = auto()

    def __str__(self) -> str:
        return "signed" if self is Signedness.SIGNED else "unsigned"


class FloatKind(Enum):
    """IEEE-754 floating-point widths."""
    F32 = 32
    F64 = 64

    def __str__(self) -> str:
        return f"f{self.value}"


class Language(Enum):
    """Source language tag – used when promotion rules differ."""
    C = auto()
    RUST = auto()


class ProvenanceTag(Enum):
    """Pointer provenance model tags (Stacked Borrows-inspired)."""
    UNIQUE = auto()        # &mut in Rust / restrict in C
    SHARED = auto()        # & in Rust / const* in C
    RAW = auto()           # *mut / *const in Rust, plain pointer in C
    UNKNOWN = auto()       # provenance not yet determined


# ---------------------------------------------------------------------------
# Abstract base
# ---------------------------------------------------------------------------

class IRType(ABC):
    """Abstract base for all IR types."""

    @abstractmethod
    def size_bits(self, pointer_size: int = 64) -> int:
        """Return the size of this type in bits."""
        ...

    @abstractmethod
    def align_bits(self, pointer_size: int = 64) -> int:
        """Return the required alignment of this type in bits."""
        ...

    def size_bytes(self, pointer_size: int = 64) -> int:
        """Return the size rounded up to whole bytes."""
        return math.ceil(self.size_bits(pointer_size) / 8)

    def align_bytes(self, pointer_size: int = 64) -> int:
        """Return the alignment in bytes."""
        return math.ceil(self.align_bits(pointer_size) / 8)

    @abstractmethod
    def is_sized(self) -> bool:
        """Return True if the type has a known, finite size."""
        ...

    @abstractmethod
    def __eq__(self, other: object) -> bool: ...

    @abstractmethod
    def __hash__(self) -> int: ...

    @abstractmethod
    def __str__(self) -> str: ...

    def __repr__(self) -> str:
        return str(self)

    # -- Serialization helpers -----------------------------------------------

    @abstractmethod
    def to_dict(self) -> dict:
        """Serialize to a JSON-friendly dictionary."""
        ...

    def to_json(self, **kwargs) -> str:
        """Serialize to JSON string."""
        return json.dumps(self.to_dict(), **kwargs)

    # -- Predicate helpers ---------------------------------------------------

    def is_integer(self) -> bool:
        return isinstance(self, IntType)

    def is_float(self) -> bool:
        return isinstance(self, FloatType)

    def is_pointer(self) -> bool:
        return isinstance(self, PointerType)

    def is_void(self) -> bool:
        return isinstance(self, VoidType)

    def is_aggregate(self) -> bool:
        return isinstance(self, (ArrayType, StructType, UnionType, EnumType))

    def is_function(self) -> bool:
        return isinstance(self, FunctionType)

    def is_numeric(self) -> bool:
        return self.is_integer() or self.is_float()

    def is_scalar(self) -> bool:
        return self.is_numeric() or self.is_pointer()


# ---------------------------------------------------------------------------
# Concrete types
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class VoidType(IRType):
    """The void type – used as return type for procedures."""

    def size_bits(self, pointer_size: int = 64) -> int:
        return 0

    def align_bits(self, pointer_size: int = 64) -> int:
        return 8  # 1-byte alignment by convention

    def is_sized(self) -> bool:
        return False

    def __eq__(self, other: object) -> bool:
        return isinstance(other, VoidType)

    def __hash__(self) -> int:
        return hash("void")

    def __str__(self) -> str:
        return "void"

    def to_dict(self) -> dict:
        return {"kind": "void"}


@dataclass(frozen=True)
class IntType(IRType):
    """Fixed-width integer type (i1 through i128, signed or unsigned).

    Attributes:
        width: bit-width (1, 8, 16, 32, 64, 128 are standard).
        signedness: SIGNED or UNSIGNED.
    """
    width: int
    signedness: Signedness = Signedness.SIGNED

    _STANDARD_WIDTHS = frozenset({1, 8, 16, 32, 64, 128})

    def __post_init__(self):
        if self.width < 1 or self.width > 128:
            raise ValueError(f"IntType width must be 1..128, got {self.width}")

    # -- Size / alignment ---------------------------------------------------

    def size_bits(self, pointer_size: int = 64) -> int:
        return self.width

    def align_bits(self, pointer_size: int = 64) -> int:
        if self.width <= 8:
            return 8
        elif self.width <= 16:
            return 16
        elif self.width <= 32:
            return 32
        elif self.width <= 64:
            return 64
        else:
            return 128

    def is_sized(self) -> bool:
        return True

    # -- Properties ---------------------------------------------------------

    @property
    def is_signed(self) -> bool:
        return self.signedness is Signedness.SIGNED

    @property
    def is_unsigned(self) -> bool:
        return self.signedness is Signedness.UNSIGNED

    @property
    def is_bool(self) -> bool:
        return self.width == 1

    @property
    def max_value(self) -> int:
        if self.is_signed:
            return (1 << (self.width - 1)) - 1
        return (1 << self.width) - 1

    @property
    def min_value(self) -> int:
        if self.is_signed:
            return -(1 << (self.width - 1))
        return 0

    def mask(self) -> int:
        """Bitmask for values of this width."""
        return (1 << self.width) - 1

    def truncate(self, value: int) -> int:
        """Truncate an arbitrary integer to this type's range."""
        value &= self.mask()
        if self.is_signed and (value >> (self.width - 1)) & 1:
            value -= 1 << self.width
        return value

    def contains(self, value: int) -> bool:
        """Return True if *value* is representable in this type."""
        return self.min_value <= value <= self.max_value

    def to_unsigned(self) -> IntType:
        return IntType(self.width, Signedness.UNSIGNED)

    def to_signed(self) -> IntType:
        return IntType(self.width, Signedness.SIGNED)

    def widen(self, target_width: int) -> IntType:
        if target_width < self.width:
            raise ValueError("Cannot widen to a smaller width")
        return IntType(target_width, self.signedness)

    # -- Equality / hashing -------------------------------------------------

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, IntType):
            return NotImplemented
        return self.width == other.width and self.signedness == other.signedness

    def __hash__(self) -> int:
        return hash(("int", self.width, self.signedness))

    def __str__(self) -> str:
        prefix = "i" if self.is_signed else "u"
        return f"{prefix}{self.width}"

    def to_dict(self) -> dict:
        return {
            "kind": "int",
            "width": self.width,
            "signedness": self.signedness.name,
        }


@dataclass(frozen=True)
class FloatType(IRType):
    """IEEE-754 floating-point type (f32 or f64).

    Attributes:
        kind: F32 or F64.
    """
    kind: FloatKind

    def size_bits(self, pointer_size: int = 64) -> int:
        return self.kind.value

    def align_bits(self, pointer_size: int = 64) -> int:
        return self.kind.value

    def is_sized(self) -> bool:
        return True

    @property
    def width(self) -> int:
        return self.kind.value

    @property
    def mantissa_bits(self) -> int:
        return 23 if self.kind is FloatKind.F32 else 52

    @property
    def exponent_bits(self) -> int:
        return 8 if self.kind is FloatKind.F32 else 11

    @property
    def epsilon(self) -> float:
        return 2.0 ** -(self.mantissa_bits)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, FloatType):
            return NotImplemented
        return self.kind == other.kind

    def __hash__(self) -> int:
        return hash(("float", self.kind))

    def __str__(self) -> str:
        return str(self.kind)

    def to_dict(self) -> dict:
        return {"kind": "float", "float_kind": self.kind.name}


@dataclass(frozen=True)
class PointerType(IRType):
    """Pointer type with optional provenance tracking.

    Attributes:
        pointee: the type being pointed to.
        provenance: provenance tag for alias analysis.
        address_space: numeric address space (0 = default).
    """
    pointee: IRType
    provenance: ProvenanceTag = ProvenanceTag.UNKNOWN
    address_space: int = 0

    def size_bits(self, pointer_size: int = 64) -> int:
        return pointer_size

    def align_bits(self, pointer_size: int = 64) -> int:
        return pointer_size

    def is_sized(self) -> bool:
        return True

    @property
    def is_opaque(self) -> bool:
        return isinstance(self.pointee, VoidType)

    def with_provenance(self, prov: ProvenanceTag) -> PointerType:
        return PointerType(self.pointee, prov, self.address_space)

    def strip_provenance(self) -> PointerType:
        return PointerType(self.pointee, ProvenanceTag.UNKNOWN, self.address_space)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, PointerType):
            return NotImplemented
        return (
            self.pointee == other.pointee
            and self.provenance == other.provenance
            and self.address_space == other.address_space
        )

    def __hash__(self) -> int:
        return hash(("ptr", self.pointee, self.provenance, self.address_space))

    def __str__(self) -> str:
        prov = ""
        if self.provenance is not ProvenanceTag.UNKNOWN:
            prov = f" [{self.provenance.name.lower()}]"
        return f"*{self.pointee}{prov}"

    def to_dict(self) -> dict:
        return {
            "kind": "pointer",
            "pointee": self.pointee.to_dict(),
            "provenance": self.provenance.name,
            "address_space": self.address_space,
        }


@dataclass(frozen=True)
class ArrayType(IRType):
    """Fixed-length array type.

    Attributes:
        element: element type.
        length: number of elements (0 means flexible array member in C).
    """
    element: IRType
    length: int

    def __post_init__(self):
        if self.length < 0:
            raise ValueError(f"Array length must be >= 0, got {self.length}")
        if not self.element.is_sized():
            raise ValueError("Array element type must be sized")

    def size_bits(self, pointer_size: int = 64) -> int:
        if self.length == 0:
            return 0
        elem_size = self.element.size_bits(pointer_size)
        elem_align = self.element.align_bits(pointer_size)
        stride = _align_up(elem_size, elem_align)
        return stride * self.length

    def align_bits(self, pointer_size: int = 64) -> int:
        return self.element.align_bits(pointer_size)

    def is_sized(self) -> bool:
        return self.length > 0 and self.element.is_sized()

    @property
    def stride_bits(self) -> int:
        s = self.element.size_bits()
        a = self.element.align_bits()
        return _align_up(s, a)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, ArrayType):
            return NotImplemented
        return self.element == other.element and self.length == other.length

    def __hash__(self) -> int:
        return hash(("array", self.element, self.length))

    def __str__(self) -> str:
        return f"[{self.length} x {self.element}]"

    def to_dict(self) -> dict:
        return {
            "kind": "array",
            "element": self.element.to_dict(),
            "length": self.length,
        }


@dataclass(frozen=True)
class StructField:
    """A single field in a struct type."""
    name: str
    type: IRType
    offset_bits: Optional[int] = None  # computed lazily


@dataclass(frozen=True)
class StructType(IRType):
    """Product (struct/record) type.

    Attributes:
        name: optional struct name (None for anonymous structs).
        fields: ordered sequence of StructField.
        packed: if True, no padding between fields.
    """
    name: Optional[str]
    fields: tuple[StructField, ...]
    packed: bool = False

    def _compute_layout(self, pointer_size: int = 64) -> list[int]:
        """Return a list of field offsets in bits."""
        offsets: list[int] = []
        current = 0
        for f in self.fields:
            if not self.packed:
                align = f.type.align_bits(pointer_size)
                current = _align_up(current, align)
            offsets.append(current)
            current += f.type.size_bits(pointer_size)
        return offsets

    def size_bits(self, pointer_size: int = 64) -> int:
        if not self.fields:
            return 0
        offsets = self._compute_layout(pointer_size)
        last = offsets[-1] + self.fields[-1].type.size_bits(pointer_size)
        if not self.packed:
            overall_align = self.align_bits(pointer_size)
            last = _align_up(last, overall_align)
        return last

    def align_bits(self, pointer_size: int = 64) -> int:
        if not self.fields:
            return 8
        if self.packed:
            return 8
        return max(f.type.align_bits(pointer_size) for f in self.fields)

    def is_sized(self) -> bool:
        return all(f.type.is_sized() for f in self.fields)

    def field_offset(self, index: int, pointer_size: int = 64) -> int:
        """Return the offset (in bits) of the field at *index*."""
        if index < 0 or index >= len(self.fields):
            raise IndexError(f"Field index {index} out of range for struct with {len(self.fields)} fields")
        return self._compute_layout(pointer_size)[index]

    def field_by_name(self, name: str) -> tuple[int, StructField]:
        """Look up a field by name; return (index, field)."""
        for i, f in enumerate(self.fields):
            if f.name == name:
                return i, f
        raise KeyError(f"No field named '{name}' in struct {self.name}")

    @property
    def field_types(self) -> tuple[IRType, ...]:
        return tuple(f.type for f in self.fields)

    @property
    def field_names(self) -> tuple[str, ...]:
        return tuple(f.name for f in self.fields)

    @property
    def num_fields(self) -> int:
        return len(self.fields)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, StructType):
            return NotImplemented
        return (
            self.name == other.name
            and self.fields == other.fields
            and self.packed == other.packed
        )

    def __hash__(self) -> int:
        return hash(("struct", self.name, self.fields, self.packed))

    def __str__(self) -> str:
        inner = ", ".join(f"{f.name}: {f.type}" for f in self.fields)
        prefix = f"%{self.name}" if self.name else ""
        pack = "<packed> " if self.packed else ""
        return f"{prefix}{{{pack}{inner}}}"

    def to_dict(self) -> dict:
        return {
            "kind": "struct",
            "name": self.name,
            "fields": [{"name": f.name, "type": f.type.to_dict()} for f in self.fields],
            "packed": self.packed,
        }


@dataclass(frozen=True)
class UnionType(IRType):
    """Sum / union type (like C union).

    Size is the maximum of all variant sizes; alignment is the maximum
    alignment of all variants.

    Attributes:
        name: optional union name.
        variants: mapping from variant name to its type.
    """
    name: Optional[str]
    variants: tuple[tuple[str, IRType], ...]

    def size_bits(self, pointer_size: int = 64) -> int:
        if not self.variants:
            return 0
        raw = max(t.size_bits(pointer_size) for _, t in self.variants)
        return _align_up(raw, self.align_bits(pointer_size))

    def align_bits(self, pointer_size: int = 64) -> int:
        if not self.variants:
            return 8
        return max(t.align_bits(pointer_size) for _, t in self.variants)

    def is_sized(self) -> bool:
        return all(t.is_sized() for _, t in self.variants)

    def variant_by_name(self, name: str) -> IRType:
        for vn, vt in self.variants:
            if vn == name:
                return vt
        raise KeyError(f"No variant named '{name}' in union {self.name}")

    @property
    def num_variants(self) -> int:
        return len(self.variants)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, UnionType):
            return NotImplemented
        return self.name == other.name and self.variants == other.variants

    def __hash__(self) -> int:
        return hash(("union", self.name, self.variants))

    def __str__(self) -> str:
        inner = " | ".join(f"{n}: {t}" for n, t in self.variants)
        prefix = f"%{self.name}" if self.name else ""
        return f"{prefix}union{{{inner}}}"

    def to_dict(self) -> dict:
        return {
            "kind": "union",
            "name": self.name,
            "variants": [{"name": n, "type": t.to_dict()} for n, t in self.variants],
        }


@dataclass(frozen=True)
class EnumType(IRType):
    """Tagged union / enum type (Rust enum, C tagged union).

    Modeled as a discriminant tag (integer) plus a payload that is the
    maximum-sized variant.  The tag width is chosen to fit the number of
    variants.

    Attributes:
        name: optional enum name.
        variants: ordered sequence of (variant_name, payload_type) pairs.
                  Use VoidType() for unit variants.
        tag_width: bit-width of the discriminant (default: auto).
    """
    name: Optional[str]
    variants: tuple[tuple[str, IRType], ...]
    tag_width: int = 0  # 0 = auto

    def _effective_tag_width(self) -> int:
        if self.tag_width > 0:
            return self.tag_width
        n = len(self.variants)
        if n <= 1:
            return 0
        if n <= 256:
            return 8
        if n <= 65536:
            return 16
        return 32

    def _max_payload_bits(self, pointer_size: int = 64) -> int:
        if not self.variants:
            return 0
        return max(
            (t.size_bits(pointer_size) if t.is_sized() else 0)
            for _, t in self.variants
        )

    def size_bits(self, pointer_size: int = 64) -> int:
        tw = self._effective_tag_width()
        payload = self._max_payload_bits(pointer_size)
        if payload == 0 and tw == 0:
            return 0
        payload_align = max(
            (t.align_bits(pointer_size) for _, t in self.variants if t.is_sized()),
            default=8,
        )
        total = _align_up(tw, payload_align) + payload
        overall_align = self.align_bits(pointer_size)
        return _align_up(total, overall_align)

    def align_bits(self, pointer_size: int = 64) -> int:
        tw = self._effective_tag_width()
        aligns = [tw] if tw > 0 else [8]
        for _, t in self.variants:
            if t.is_sized():
                aligns.append(t.align_bits(pointer_size))
        return max(aligns)

    def is_sized(self) -> bool:
        return all(t.is_sized() or isinstance(t, VoidType) for _, t in self.variants)

    def variant_index(self, name: str) -> int:
        for i, (vn, _) in enumerate(self.variants):
            if vn == name:
                return i
        raise KeyError(f"No variant named '{name}' in enum {self.name}")

    def variant_type(self, name: str) -> IRType:
        for vn, vt in self.variants:
            if vn == name:
                return vt
        raise KeyError(f"No variant named '{name}' in enum {self.name}")

    @property
    def num_variants(self) -> int:
        return len(self.variants)

    @property
    def variant_names(self) -> tuple[str, ...]:
        return tuple(n for n, _ in self.variants)

    @property
    def is_c_like(self) -> bool:
        """True if all variants have VoidType payload (C-style enum)."""
        return all(isinstance(t, VoidType) for _, t in self.variants)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, EnumType):
            return NotImplemented
        return self.name == other.name and self.variants == other.variants

    def __hash__(self) -> int:
        return hash(("enum", self.name, self.variants))

    def __str__(self) -> str:
        inner = " | ".join(
            f"{n}" if isinstance(t, VoidType) else f"{n}({t})"
            for n, t in self.variants
        )
        prefix = f"%{self.name}" if self.name else ""
        return f"{prefix}enum{{{inner}}}"

    def to_dict(self) -> dict:
        return {
            "kind": "enum",
            "name": self.name,
            "variants": [{"name": n, "type": t.to_dict()} for n, t in self.variants],
            "tag_width": self.tag_width,
        }


@dataclass(frozen=True)
class FunctionType(IRType):
    """Function (callable) type.

    Attributes:
        return_type: the return type.
        param_types: ordered parameter types.
        is_variadic: True for C-style variadic functions.
    """
    return_type: IRType
    param_types: tuple[IRType, ...]
    is_variadic: bool = False

    def size_bits(self, pointer_size: int = 64) -> int:
        return 0  # function types are unsized

    def align_bits(self, pointer_size: int = 64) -> int:
        return 0

    def is_sized(self) -> bool:
        return False

    @property
    def arity(self) -> int:
        return len(self.param_types)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, FunctionType):
            return NotImplemented
        return (
            self.return_type == other.return_type
            and self.param_types == other.param_types
            and self.is_variadic == other.is_variadic
        )

    def __hash__(self) -> int:
        return hash(("func", self.return_type, self.param_types, self.is_variadic))

    def __str__(self) -> str:
        params = ", ".join(str(p) for p in self.param_types)
        if self.is_variadic:
            params += ", ..."
        return f"({params}) -> {self.return_type}"

    def to_dict(self) -> dict:
        return {
            "kind": "function",
            "return_type": self.return_type.to_dict(),
            "param_types": [p.to_dict() for p in self.param_types],
            "is_variadic": self.is_variadic,
        }


# ---------------------------------------------------------------------------
# Convenience constructors
# ---------------------------------------------------------------------------

# Standard integer types
I1 = IntType(1, Signedness.SIGNED)
I8 = IntType(8, Signedness.SIGNED)
I16 = IntType(16, Signedness.SIGNED)
I32 = IntType(32, Signedness.SIGNED)
I64 = IntType(64, Signedness.SIGNED)
I128 = IntType(128, Signedness.SIGNED)

U1 = IntType(1, Signedness.UNSIGNED)
U8 = IntType(8, Signedness.UNSIGNED)
U16 = IntType(16, Signedness.UNSIGNED)
U32 = IntType(32, Signedness.UNSIGNED)
U64 = IntType(64, Signedness.UNSIGNED)
U128 = IntType(128, Signedness.UNSIGNED)

# Standard float types
F32 = FloatType(FloatKind.F32)
F64 = FloatType(FloatKind.F64)

VOID = VoidType()


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------

def _align_up(value: int, alignment: int) -> int:
    """Round *value* up to the next multiple of *alignment*."""
    if alignment <= 0:
        return value
    return (value + alignment - 1) // alignment * alignment


# ---------------------------------------------------------------------------
# Type compatibility & promotion
# ---------------------------------------------------------------------------

class TypeCompatibility(Enum):
    """Result of a type compatibility check."""
    IDENTICAL = auto()
    COMPATIBLE = auto()       # implicit conversion exists
    PROMOTABLE = auto()       # promotion is possible
    INCOMPATIBLE = auto()

    @property
    def is_ok(self) -> bool:
        return self is not TypeCompatibility.INCOMPATIBLE


def check_compatibility(a: IRType, b: IRType) -> TypeCompatibility:
    """Check whether two types are compatible for assignment/comparison."""
    if a == b:
        return TypeCompatibility.IDENTICAL

    # Integer widening
    if isinstance(a, IntType) and isinstance(b, IntType):
        if a.signedness == b.signedness and a.width <= b.width:
            return TypeCompatibility.PROMOTABLE
        if a.width < b.width:
            return TypeCompatibility.COMPATIBLE
        if a.width == b.width:
            return TypeCompatibility.COMPATIBLE
        return TypeCompatibility.INCOMPATIBLE

    # Float widening
    if isinstance(a, FloatType) and isinstance(b, FloatType):
        if a.kind.value <= b.kind.value:
            return TypeCompatibility.PROMOTABLE
        return TypeCompatibility.COMPATIBLE

    # Int -> Float
    if isinstance(a, IntType) and isinstance(b, FloatType):
        return TypeCompatibility.COMPATIBLE

    # Pointer compatibility: same pointee or one is void*
    if isinstance(a, PointerType) and isinstance(b, PointerType):
        if a.pointee == b.pointee:
            if a.provenance != b.provenance:
                return TypeCompatibility.COMPATIBLE
            return TypeCompatibility.IDENTICAL
        if isinstance(a.pointee, VoidType) or isinstance(b.pointee, VoidType):
            return TypeCompatibility.COMPATIBLE
        return TypeCompatibility.INCOMPATIBLE

    # Array to pointer decay
    if isinstance(a, ArrayType) and isinstance(b, PointerType):
        if a.element == b.pointee:
            return TypeCompatibility.COMPATIBLE
        return TypeCompatibility.INCOMPATIBLE

    return TypeCompatibility.INCOMPATIBLE


def compute_common_type(a: IRType, b: IRType, lang: Language = Language.C) -> Optional[IRType]:
    """Compute the common type to which *a* and *b* can both be promoted.

    Follows C integer promotion rules when *lang* is C, and Rust semantics
    (no implicit widening) when *lang* is RUST.

    Returns None if no common type exists.
    """
    if a == b:
        return a

    # ---- Integer promotion ------------------------------------------------
    if isinstance(a, IntType) and isinstance(b, IntType):
        if lang is Language.RUST:
            # Rust requires explicit casts – no implicit common type
            return None

        # C promotion: both promoted to at least int (i32)
        wa = max(a.width, 32)
        wb = max(b.width, 32)
        target_width = max(wa, wb)

        # If signedness matches, keep it
        if a.signedness == b.signedness:
            return IntType(target_width, a.signedness)

        # Otherwise, the unsigned type wins if its width >= the signed type
        unsigned, signed = (a, b) if a.is_unsigned else (b, a)
        if unsigned.width >= signed.width:
            return IntType(target_width, Signedness.UNSIGNED)
        if signed.width > unsigned.width:
            return IntType(target_width, Signedness.SIGNED)
        return IntType(target_width, Signedness.UNSIGNED)

    # ---- Float promotion --------------------------------------------------
    if isinstance(a, FloatType) and isinstance(b, FloatType):
        wider = FloatKind.F64 if (a.kind is FloatKind.F64 or b.kind is FloatKind.F64) else FloatKind.F32
        return FloatType(wider)

    # ---- Mixed int / float ------------------------------------------------
    if isinstance(a, IntType) and isinstance(b, FloatType):
        return b
    if isinstance(a, FloatType) and isinstance(b, IntType):
        return a

    # ---- Pointer types ----------------------------------------------------
    if isinstance(a, PointerType) and isinstance(b, PointerType):
        if a.pointee == b.pointee:
            prov = ProvenanceTag.UNKNOWN
            return PointerType(a.pointee, prov, max(a.address_space, b.address_space))
        # If one is void*, result is void*
        if isinstance(a.pointee, VoidType):
            return a
        if isinstance(b.pointee, VoidType):
            return b
        return None

    return None


# ---------------------------------------------------------------------------
# Type lattice operations
# ---------------------------------------------------------------------------

def type_join(a: IRType, b: IRType) -> Optional[IRType]:
    """Compute the *join* (least upper bound) in the type lattice.

    The join represents the smallest type that can contain values of both
    *a* and *b*.
    """
    if a == b:
        return a

    if isinstance(a, IntType) and isinstance(b, IntType):
        width = max(a.width, b.width)
        if a.signedness == b.signedness:
            return IntType(width, a.signedness)
        # Signed + unsigned: need one more bit if widths match
        if a.width == b.width:
            next_width = _next_standard_width(width + 1)
            if next_width is None:
                return None
            return IntType(next_width, Signedness.SIGNED)
        wider = a if a.width > b.width else b
        return IntType(wider.width, Signedness.SIGNED)

    if isinstance(a, FloatType) and isinstance(b, FloatType):
        wider = FloatKind.F64 if a.kind is FloatKind.F64 or b.kind is FloatKind.F64 else FloatKind.F32
        return FloatType(wider)

    if isinstance(a, IntType) and isinstance(b, FloatType):
        return FloatType(FloatKind.F64)
    if isinstance(a, FloatType) and isinstance(b, IntType):
        return FloatType(FloatKind.F64)

    if isinstance(a, PointerType) and isinstance(b, PointerType):
        if a.pointee == b.pointee:
            return PointerType(a.pointee, ProvenanceTag.UNKNOWN)
        return PointerType(VoidType(), ProvenanceTag.UNKNOWN)

    return None


def type_meet(a: IRType, b: IRType) -> Optional[IRType]:
    """Compute the *meet* (greatest lower bound) in the type lattice.

    The meet represents the largest type whose values can be contained
    by both *a* and *b*.
    """
    if a == b:
        return a

    if isinstance(a, IntType) and isinstance(b, IntType):
        width = min(a.width, b.width)
        if a.signedness == b.signedness:
            return IntType(width, a.signedness)
        return IntType(width, Signedness.UNSIGNED)

    if isinstance(a, FloatType) and isinstance(b, FloatType):
        narrower = FloatKind.F32 if a.kind is FloatKind.F32 or b.kind is FloatKind.F32 else FloatKind.F64
        return FloatType(narrower)

    if isinstance(a, IntType) and isinstance(b, FloatType):
        return a
    if isinstance(a, FloatType) and isinstance(b, IntType):
        return b

    return None


def _next_standard_width(minimum: int) -> Optional[int]:
    """Return the smallest standard width >= *minimum*, or None."""
    for w in (8, 16, 32, 64, 128):
        if w >= minimum:
            return w
    return None


# ---------------------------------------------------------------------------
# C ↔ Rust type mapping helpers
# ---------------------------------------------------------------------------

# C standard integer type equivalents
C_CHAR = IntType(8, Signedness.SIGNED)
C_UCHAR = IntType(8, Signedness.UNSIGNED)
C_SHORT = IntType(16, Signedness.SIGNED)
C_USHORT = IntType(16, Signedness.UNSIGNED)
C_INT = IntType(32, Signedness.SIGNED)
C_UINT = IntType(32, Signedness.UNSIGNED)
C_LONG = IntType(64, Signedness.SIGNED)   # assuming LP64
C_ULONG = IntType(64, Signedness.UNSIGNED)
C_LONGLONG = IntType(64, Signedness.SIGNED)
C_ULONGLONG = IntType(64, Signedness.UNSIGNED)
C_SIZE_T = IntType(64, Signedness.UNSIGNED)
C_PTRDIFF_T = IntType(64, Signedness.SIGNED)
C_BOOL = IntType(1, Signedness.UNSIGNED)

# Rust equivalents
RUST_I8 = IntType(8, Signedness.SIGNED)
RUST_U8 = IntType(8, Signedness.UNSIGNED)
RUST_I16 = IntType(16, Signedness.SIGNED)
RUST_U16 = IntType(16, Signedness.UNSIGNED)
RUST_I32 = IntType(32, Signedness.SIGNED)
RUST_U32 = IntType(32, Signedness.UNSIGNED)
RUST_I64 = IntType(64, Signedness.SIGNED)
RUST_U64 = IntType(64, Signedness.UNSIGNED)
RUST_I128 = IntType(128, Signedness.SIGNED)
RUST_U128 = IntType(128, Signedness.UNSIGNED)
RUST_ISIZE = IntType(64, Signedness.SIGNED)
RUST_USIZE = IntType(64, Signedness.UNSIGNED)
RUST_BOOL = IntType(1, Signedness.UNSIGNED)


_C_TO_RUST_MAP: dict[str, IRType] = {
    "char": RUST_I8,
    "unsigned char": RUST_U8,
    "short": RUST_I16,
    "unsigned short": RUST_U16,
    "int": RUST_I32,
    "unsigned int": RUST_U32,
    "long": RUST_I64,
    "unsigned long": RUST_U64,
    "long long": RUST_I64,
    "unsigned long long": RUST_U64,
    "float": F32,
    "double": F64,
    "_Bool": RUST_BOOL,
    "size_t": RUST_USIZE,
    "ptrdiff_t": RUST_ISIZE,
}


def c_type_to_rust(c_type_name: str) -> Optional[IRType]:
    """Map a C type name to its Rust-equivalent IR type."""
    return _C_TO_RUST_MAP.get(c_type_name)


def are_layout_compatible(a: IRType, b: IRType, pointer_size: int = 64) -> bool:
    """Check whether two types have identical memory layout (size + alignment)."""
    if not (a.is_sized() and b.is_sized()):
        return False
    return (
        a.size_bits(pointer_size) == b.size_bits(pointer_size)
        and a.align_bits(pointer_size) == b.align_bits(pointer_size)
    )


# ---------------------------------------------------------------------------
# Deserialization
# ---------------------------------------------------------------------------

def type_from_dict(data: dict) -> IRType:
    """Reconstruct an IRType from its dict serialization."""
    kind = data["kind"]
    match kind:
        case "void":
            return VoidType()
        case "int":
            return IntType(data["width"], Signedness[data["signedness"]])
        case "float":
            return FloatType(FloatKind[data["float_kind"]])
        case "pointer":
            pointee = type_from_dict(data["pointee"])
            prov = ProvenanceTag[data.get("provenance", "UNKNOWN")]
            return PointerType(pointee, prov, data.get("address_space", 0))
        case "array":
            elem = type_from_dict(data["element"])
            return ArrayType(elem, data["length"])
        case "struct":
            fields = tuple(
                StructField(f["name"], type_from_dict(f["type"]))
                for f in data["fields"]
            )
            return StructType(data.get("name"), fields, data.get("packed", False))
        case "union":
            variants = tuple(
                (v["name"], type_from_dict(v["type"]))
                for v in data["variants"]
            )
            return UnionType(data.get("name"), variants)
        case "enum":
            variants = tuple(
                (v["name"], type_from_dict(v["type"]))
                for v in data["variants"]
            )
            return EnumType(data.get("name"), variants, data.get("tag_width", 0))
        case "function":
            ret = type_from_dict(data["return_type"])
            params = tuple(type_from_dict(p) for p in data["param_types"])
            return FunctionType(ret, params, data.get("is_variadic", False))
        case _:
            raise ValueError(f"Unknown type kind: {kind}")


# ---------------------------------------------------------------------------
# Overflow semantics — used by instruction lowering
# ---------------------------------------------------------------------------

class OverflowBehavior(Enum):
    """How integer overflow is handled."""
    WRAP = auto()           # two's complement wrap (Rust default, C unsigned)
    UNDEFINED = auto()      # undefined behavior (C signed default)
    SATURATE = auto()       # clamp to min/max
    TRAP = auto()           # raise a trap / panic (Rust debug mode)
    CHECKED = auto()        # return (result, overflow_flag) pair

    @staticmethod
    def for_language(lang: Language, signed: bool) -> "OverflowBehavior":
        """Return the default overflow behavior for a language/signedness pair."""
        if lang is Language.RUST:
            return OverflowBehavior.WRAP  # release mode
        # C
        if signed:
            return OverflowBehavior.UNDEFINED
        return OverflowBehavior.WRAP


# ---------------------------------------------------------------------------
# Type visitor
# ---------------------------------------------------------------------------

class TypeVisitor:
    """Visitor pattern for IR types — override the methods you need."""

    def visit(self, ty: IRType) -> None:
        match ty:
            case VoidType():
                self.visit_void(ty)
            case IntType():
                self.visit_int(ty)
            case FloatType():
                self.visit_float(ty)
            case PointerType():
                self.visit_pointer(ty)
            case ArrayType():
                self.visit_array(ty)
            case StructType():
                self.visit_struct(ty)
            case UnionType():
                self.visit_union(ty)
            case EnumType():
                self.visit_enum(ty)
            case FunctionType():
                self.visit_function(ty)
            case _:
                self.visit_unknown(ty)

    def visit_void(self, ty: VoidType) -> None: ...
    def visit_int(self, ty: IntType) -> None: ...
    def visit_float(self, ty: FloatType) -> None: ...

    def visit_pointer(self, ty: PointerType) -> None:
        self.visit(ty.pointee)

    def visit_array(self, ty: ArrayType) -> None:
        self.visit(ty.element)

    def visit_struct(self, ty: StructType) -> None:
        for f in ty.fields:
            self.visit(f.type)

    def visit_union(self, ty: UnionType) -> None:
        for _, vt in ty.variants:
            self.visit(vt)

    def visit_enum(self, ty: "EnumType") -> None:
        for _, vt in ty.variants:
            if not isinstance(vt, VoidType):
                self.visit(vt)

    def visit_function(self, ty: FunctionType) -> None:
        self.visit(ty.return_type)
        for p in ty.param_types:
            self.visit(p)

    def visit_unknown(self, ty: IRType) -> None: ...


class TypeCollector(TypeVisitor):
    """Collect all leaf types reachable from a root type."""

    def __init__(self) -> None:
        self.types: list[IRType] = []

    def visit_void(self, ty: VoidType) -> None:
        self.types.append(ty)

    def visit_int(self, ty: IntType) -> None:
        self.types.append(ty)

    def visit_float(self, ty: FloatType) -> None:
        self.types.append(ty)

    def visit_pointer(self, ty: PointerType) -> None:
        self.types.append(ty)
        super().visit_pointer(ty)

    def visit_array(self, ty: ArrayType) -> None:
        self.types.append(ty)
        super().visit_array(ty)

    def visit_struct(self, ty: StructType) -> None:
        self.types.append(ty)
        super().visit_struct(ty)

    def visit_union(self, ty: UnionType) -> None:
        self.types.append(ty)
        super().visit_union(ty)

    def visit_enum(self, ty: "EnumType") -> None:
        self.types.append(ty)
        super().visit_enum(ty)

    def visit_function(self, ty: FunctionType) -> None:
        self.types.append(ty)
        super().visit_function(ty)


def collect_types(ty: IRType) -> list[IRType]:
    """Collect all types reachable from *ty* (including *ty* itself)."""
    c = TypeCollector()
    c.visit(ty)
    return c.types


def contains_pointer(ty: IRType) -> bool:
    """Return True if *ty* contains or is a pointer type."""
    return any(isinstance(t, PointerType) for t in collect_types(ty))
