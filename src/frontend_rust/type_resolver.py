"""
Rust type resolution for the Cross-Language Equivalence Verifier.

Resolves type aliases, determines integer types from suffixes, handles
reference types, raw pointer types, array/slice types, Option/Result,
tuple types, and maps Rust types to IR types.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from .rust_ast import (
    RustType, NeverType, UnitType, PathType, ReferenceType,
    RawPointerType, ArrayType as RustArrayType, SliceType,
    TupleType, FnPointerType, InferredType, ParenType,
    OptionType, ResultType, BoxType,
    StructItem, EnumItem, TypeAliasItem,
    StructField as RustStructField,
    Mutability, Expr, LitExpr,
)


# ---------------------------------------------------------------------------
# Type information
# ---------------------------------------------------------------------------

@dataclass
class RustTypeInfo:
    """Resolved type information."""
    size_bits: int = 0
    align_bits: int = 8
    is_signed: bool = False
    is_integer: bool = False
    is_float: bool = False
    is_bool: bool = False
    is_pointer: bool = False
    is_reference: bool = False
    is_array: bool = False
    is_slice: bool = False
    is_tuple: bool = False
    is_struct: bool = False
    is_enum: bool = False
    is_unit: bool = False
    is_never: bool = False
    is_function: bool = False
    is_option: bool = False
    is_result: bool = False
    is_box: bool = False
    element_count: int = 0
    field_names: list[str] = field(default_factory=list)
    field_types: list[RustType] = field(default_factory=list)

    @property
    def size_bytes(self) -> int:
        return (self.size_bits + 7) // 8

    @property
    def align_bytes(self) -> int:
        return (self.align_bits + 7) // 8


# ---------------------------------------------------------------------------
# Struct layout
# ---------------------------------------------------------------------------

@dataclass
class RustStructLayout:
    """Computed layout for a Rust struct."""
    name: str
    size_bits: int = 0
    align_bits: int = 8
    fields: list["RustFieldLayout"] = field(default_factory=list)
    repr: str = ""  # "C", "transparent", "packed", etc.


@dataclass
class RustFieldLayout:
    """Layout of a single struct field."""
    name: str
    type_name: RustType
    offset_bits: int = 0
    size_bits: int = 0
    align_bits: int = 8


# ---------------------------------------------------------------------------
# Primitive type info
# ---------------------------------------------------------------------------

_PRIMITIVE_TYPES: dict[str, tuple[int, bool, bool, bool]] = {
    # name -> (bits, is_signed, is_integer, is_float)
    "i8": (8, True, True, False),
    "i16": (16, True, True, False),
    "i32": (32, True, True, False),
    "i64": (64, True, True, False),
    "i128": (128, True, True, False),
    "isize": (64, True, True, False),
    "u8": (8, False, True, False),
    "u16": (16, False, True, False),
    "u32": (32, False, True, False),
    "u64": (64, False, True, False),
    "u128": (128, False, True, False),
    "usize": (64, False, True, False),
    "f32": (32, False, False, True),
    "f64": (64, False, False, True),
    "bool": (1, False, False, False),
    "char": (32, False, False, False),  # Unicode scalar value
}


# ---------------------------------------------------------------------------
# RustTypeResolver
# ---------------------------------------------------------------------------

class RustTypeResolver:
    """Resolves Rust types: alias expansion, layout computation, type mapping.

    Usage::

        resolver = RustTypeResolver()
        resolver.register_struct("Point", struct_item)
        info = resolver.resolve(some_type)
    """

    def __init__(self, pointer_size: int = 64) -> None:
        self._pointer_size = pointer_size
        self._type_aliases: dict[str, RustType] = {}
        self._structs: dict[str, StructItem] = {}
        self._enums: dict[str, EnumItem] = {}
        self._struct_layouts: dict[str, RustStructLayout] = {}
        # Update isize/usize based on pointer size
        if pointer_size == 32:
            _PRIMITIVE_TYPES["isize"] = (32, True, True, False)
            _PRIMITIVE_TYPES["usize"] = (32, False, True, False)

    # -------------------------------------------------------------------
    # Registration
    # -------------------------------------------------------------------

    def register_type_alias(self, name: str, aliased: RustType) -> None:
        self._type_aliases[name] = aliased

    def register_struct(self, name: str, item: StructItem) -> None:
        self._structs[name] = item
        self._struct_layouts[name] = self._compute_struct_layout(item)

    def register_enum(self, name: str, item: EnumItem) -> None:
        self._enums[name] = item

    # -------------------------------------------------------------------
    # Type alias resolution
    # -------------------------------------------------------------------

    def resolve_alias(self, ty: RustType, max_depth: int = 32) -> RustType:
        """Fully resolve type aliases."""
        depth = 0
        while isinstance(ty, PathType) and len(ty.segments) == 1 and depth < max_depth:
            name = ty.segments[0]
            if name in self._type_aliases:
                ty = self._type_aliases[name]
                depth += 1
            else:
                break

        if isinstance(ty, ParenType):
            return self.resolve_alias(ty.inner, max_depth)

        return ty

    # -------------------------------------------------------------------
    # Type resolution
    # -------------------------------------------------------------------

    def resolve(self, ty: RustType) -> RustTypeInfo:
        """Resolve a Rust type to its RustTypeInfo."""
        ty = self.resolve_alias(ty)

        if isinstance(ty, NeverType):
            return RustTypeInfo(size_bits=0, is_never=True)

        if isinstance(ty, UnitType):
            return RustTypeInfo(size_bits=0, is_unit=True)

        if isinstance(ty, PathType):
            return self._resolve_path_type(ty)

        if isinstance(ty, ReferenceType):
            return RustTypeInfo(
                size_bits=self._pointer_size,
                align_bits=self._pointer_size,
                is_reference=True,
            )

        if isinstance(ty, RawPointerType):
            return RustTypeInfo(
                size_bits=self._pointer_size,
                align_bits=self._pointer_size,
                is_pointer=True,
            )

        if isinstance(ty, RustArrayType):
            elem_info = self.resolve(ty.element)
            count = self._eval_array_length(ty.length)
            return RustTypeInfo(
                size_bits=elem_info.size_bits * count,
                align_bits=elem_info.align_bits,
                is_array=True,
                element_count=count,
            )

        if isinstance(ty, SliceType):
            # Slice is a fat pointer: (ptr, len)
            return RustTypeInfo(
                size_bits=self._pointer_size * 2,
                align_bits=self._pointer_size,
                is_slice=True,
            )

        if isinstance(ty, TupleType):
            return self._resolve_tuple_type(ty)

        if isinstance(ty, FnPointerType):
            return RustTypeInfo(
                size_bits=self._pointer_size,
                align_bits=self._pointer_size,
                is_function=True,
            )

        if isinstance(ty, InferredType):
            return RustTypeInfo()

        if isinstance(ty, OptionType):
            inner_info = self.resolve(ty.inner)
            # Option<T> may have niche optimization
            return RustTypeInfo(
                size_bits=inner_info.size_bits + 8,  # simplified
                align_bits=max(inner_info.align_bits, 8),
                is_option=True,
            )

        if isinstance(ty, ResultType):
            ok_info = self.resolve(ty.ok_type)
            err_info = self.resolve(ty.err_type)
            return RustTypeInfo(
                size_bits=max(ok_info.size_bits, err_info.size_bits) + 8,
                align_bits=max(ok_info.align_bits, err_info.align_bits, 8),
                is_result=True,
            )

        if isinstance(ty, BoxType):
            return RustTypeInfo(
                size_bits=self._pointer_size,
                align_bits=self._pointer_size,
                is_box=True,
            )

        return RustTypeInfo()

    def _resolve_path_type(self, ty: PathType) -> RustTypeInfo:
        """Resolve a path type."""
        name = ty.name

        # Primitive types
        if name in _PRIMITIVE_TYPES:
            bits, signed, is_int, is_float = _PRIMITIVE_TYPES[name]
            return RustTypeInfo(
                size_bits=bits,
                align_bits=min(bits, 64) if bits > 0 else 8,
                is_signed=signed,
                is_integer=is_int,
                is_float=is_float,
                is_bool=(name == "bool"),
            )

        # str type (unsized)
        if name == "str":
            return RustTypeInfo(size_bits=0, is_slice=True)

        # Option<T>
        if name == "Option" and ty.generic_args:
            inner_info = self.resolve(ty.generic_args[0])
            return RustTypeInfo(
                size_bits=inner_info.size_bits + 8,
                align_bits=max(inner_info.align_bits, 8),
                is_option=True,
            )

        # Result<T, E>
        if name == "Result" and len(ty.generic_args) >= 2:
            ok_info = self.resolve(ty.generic_args[0])
            err_info = self.resolve(ty.generic_args[1])
            return RustTypeInfo(
                size_bits=max(ok_info.size_bits, err_info.size_bits) + 8,
                align_bits=max(ok_info.align_bits, err_info.align_bits, 8),
                is_result=True,
            )

        # Box<T>
        if name == "Box":
            return RustTypeInfo(
                size_bits=self._pointer_size,
                align_bits=self._pointer_size,
                is_box=True,
            )

        # Vec<T>, String, etc. (heap-allocated)
        if name in ("Vec", "String"):
            return RustTypeInfo(
                size_bits=self._pointer_size * 3,  # ptr + len + cap
                align_bits=self._pointer_size,
            )

        # User-defined structs
        if name in self._structs:
            return self._resolve_struct(name)

        # User-defined enums
        if name in self._enums:
            return self._resolve_enum(name)

        # Unknown type
        return RustTypeInfo()

    def _resolve_struct(self, name: str) -> RustTypeInfo:
        """Resolve a struct type."""
        layout = self._struct_layouts.get(name)
        if layout is None:
            return RustTypeInfo(is_struct=True)
        return RustTypeInfo(
            size_bits=layout.size_bits,
            align_bits=layout.align_bits,
            is_struct=True,
            field_names=[f.name for f in layout.fields],
            field_types=[f.type_name for f in layout.fields],
        )

    def _resolve_enum(self, name: str) -> RustTypeInfo:
        """Resolve an enum type."""
        enum = self._enums.get(name)
        if enum is None:
            return RustTypeInfo(is_enum=True)

        # Check repr
        if enum.repr:
            repr_type = enum.repr.strip()
            if repr_type in _PRIMITIVE_TYPES:
                bits, signed, _, _ = _PRIMITIVE_TYPES[repr_type]
                return RustTypeInfo(
                    size_bits=bits,
                    align_bits=min(bits, 64),
                    is_signed=signed,
                    is_integer=True,
                    is_enum=True,
                )

        # Default C-like enum = i32 discriminant
        has_data = any(not v.is_unit for v in enum.variants)
        if not has_data:
            return RustTypeInfo(
                size_bits=32,
                align_bits=32,
                is_signed=True,
                is_integer=True,
                is_enum=True,
            )

        # Data-carrying enum: size is discriminant + max variant size
        max_variant_size = 0
        max_variant_align = 8
        for v in enum.variants:
            for f in v.fields:
                if f.type_ann:
                    fi = self.resolve(f.type_ann)
                    max_variant_size = max(max_variant_size, fi.size_bits)
                    max_variant_align = max(max_variant_align, fi.align_bits)

        disc_size = 8  # smallest possible discriminant
        total_align = max(max_variant_align, disc_size)
        total_size = _align_up(disc_size + max_variant_size, total_align)

        return RustTypeInfo(
            size_bits=total_size,
            align_bits=total_align,
            is_enum=True,
        )

    def _resolve_tuple_type(self, ty: TupleType) -> RustTypeInfo:
        """Resolve a tuple type."""
        if not ty.elements:
            return RustTypeInfo(size_bits=0, is_unit=True, is_tuple=True)

        current_offset = 0
        max_align = 8
        field_types: list[RustType] = []

        for elem in ty.elements:
            info = self.resolve(elem)
            current_offset = _align_up(current_offset, info.align_bits)
            current_offset += info.size_bits
            max_align = max(max_align, info.align_bits)
            field_types.append(elem)

        total_size = _align_up(current_offset, max_align)
        return RustTypeInfo(
            size_bits=total_size,
            align_bits=max_align,
            is_tuple=True,
            field_types=field_types,
        )

    # -------------------------------------------------------------------
    # Struct layout computation
    # -------------------------------------------------------------------

    def _compute_struct_layout(self, item: StructItem) -> RustStructLayout:
        """Compute struct layout."""
        layout = RustStructLayout(name=item.name)

        # Detect repr
        for attr in item.attributes:
            if attr.name == "repr":
                layout.repr = attr.args.strip()

        is_c_repr = "C" in layout.repr
        is_packed = "packed" in layout.repr

        current_offset = 0
        max_align = 8

        for f in item.fields:
            if f.type_ann is None:
                continue
            finfo = self.resolve(f.type_ann)
            f_align = 8 if is_packed else finfo.align_bits

            if not is_packed:
                current_offset = _align_up(current_offset, f_align)

            fl = RustFieldLayout(
                name=f.name,
                type_name=f.type_ann,
                offset_bits=current_offset,
                size_bits=finfo.size_bits,
                align_bits=f_align,
            )
            layout.fields.append(fl)
            current_offset += finfo.size_bits
            max_align = max(max_align, f_align)

        if not is_packed:
            current_offset = _align_up(current_offset, max_align)

        layout.size_bits = current_offset
        layout.align_bits = max_align
        return layout

    # -------------------------------------------------------------------
    # Integer type from suffix
    # -------------------------------------------------------------------

    def type_from_int_suffix(self, suffix: str) -> RustType:
        """Get the Rust type for an integer literal suffix."""
        if suffix in _PRIMITIVE_TYPES:
            return PathType(segments=[suffix])
        return PathType(segments=["i32"])  # default

    def type_from_float_suffix(self, suffix: str) -> RustType:
        """Get the Rust type for a float literal suffix."""
        if suffix == "f32":
            return PathType(segments=["f32"])
        return PathType(segments=["f64"])  # default

    # -------------------------------------------------------------------
    # Type compatibility
    # -------------------------------------------------------------------

    def are_compatible(self, ty1: RustType, ty2: RustType) -> bool:
        """Check if two Rust types are structurally compatible."""
        ty1 = self.resolve_alias(ty1)
        ty2 = self.resolve_alias(ty2)

        if type(ty1) != type(ty2):
            return False

        if isinstance(ty1, PathType) and isinstance(ty2, PathType):
            return ty1.segments == ty2.segments

        if isinstance(ty1, ReferenceType) and isinstance(ty2, ReferenceType):
            return (ty1.mutability == ty2.mutability and
                    self.are_compatible(ty1.referent, ty2.referent))

        if isinstance(ty1, RawPointerType) and isinstance(ty2, RawPointerType):
            return (ty1.mutability == ty2.mutability and
                    self.are_compatible(ty1.pointee, ty2.pointee))

        if isinstance(ty1, TupleType) and isinstance(ty2, TupleType):
            if len(ty1.elements) != len(ty2.elements):
                return False
            return all(
                self.are_compatible(a, b)
                for a, b in zip(ty1.elements, ty2.elements)
            )

        if isinstance(ty1, NeverType) and isinstance(ty2, NeverType):
            return True

        if isinstance(ty1, UnitType) and isinstance(ty2, UnitType):
            return True

        return False

    def sizeof(self, ty: RustType) -> int:
        """Get sizeof in bytes."""
        return self.resolve(ty).size_bytes

    def alignof(self, ty: RustType) -> int:
        """Get alignof in bytes."""
        return self.resolve(ty).align_bytes

    # -------------------------------------------------------------------
    # Helpers
    # -------------------------------------------------------------------

    def _eval_array_length(self, expr: Optional[Expr]) -> int:
        """Try to evaluate an array length expression."""
        if expr is None:
            return 0
        if isinstance(expr, LitExpr) and expr.int_value is not None:
            return expr.int_value
        return 0

    def get_struct_layout(self, name: str) -> Optional[RustStructLayout]:
        return self._struct_layouts.get(name)

    def get_field_index(self, struct_name: str, field_name: str) -> Optional[int]:
        layout = self._struct_layouts.get(struct_name)
        if layout is None:
            return None
        for i, f in enumerate(layout.fields):
            if f.name == field_name:
                return i
        return None


def _align_up(value: int, align: int) -> int:
    if align <= 0:
        return value
    return ((value + align - 1) // align) * align
