"""
C type resolution for the Cross-Language Equivalence Verifier.

Resolves typedefs, computes struct/union layouts, determines integer promotion
types, computes sizeof/alignof, handles platform-specific types (size_t,
ptrdiff_t, etc.), and performs type compatibility checking.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from .c_ast import (
    CType, VoidCType, IntCType, FloatCType, PointerCType, ArrayCType,
    FunctionCType, StructRefCType, UnionRefCType, EnumRefCType,
    TypedefRefCType, QualifiedCType, AtomicCType, TypeofCType,
    TypeQualifier, FieldDecl, StructDecl, UnionDecl, EnumDecl,
    TypedefDecl, Expr, IntLiteral,
)


# ---------------------------------------------------------------------------
# Platform configuration
# ---------------------------------------------------------------------------

@dataclass
class PlatformConfig:
    """Target platform configuration for type layout."""
    pointer_size: int = 64        # bits
    char_is_signed: bool = True
    wchar_size: int = 32          # bits
    long_size: int = 64           # bits (64 on LP64, 32 on LLP64/Windows)
    long_double_size: int = 80    # bits (80 on x86, 64 on ARM)
    max_align: int = 16           # bytes
    endian: str = "little"

    @staticmethod
    def lp64() -> "PlatformConfig":
        """Standard LP64 (Linux/macOS x86-64)."""
        return PlatformConfig(pointer_size=64, long_size=64)

    @staticmethod
    def llp64() -> "PlatformConfig":
        """LLP64 (Windows x86-64)."""
        return PlatformConfig(pointer_size=64, long_size=32)

    @staticmethod
    def ilp32() -> "PlatformConfig":
        """ILP32 (32-bit platforms)."""
        return PlatformConfig(pointer_size=32, long_size=32, long_double_size=96)


# ---------------------------------------------------------------------------
# Resolved type info
# ---------------------------------------------------------------------------

@dataclass
class TypeInfo:
    """Resolved type information with size and alignment."""
    size_bits: int
    align_bits: int
    is_signed: bool = False
    is_integer: bool = False
    is_float: bool = False
    is_pointer: bool = False
    is_array: bool = False
    is_struct: bool = False
    is_union: bool = False
    is_void: bool = False
    is_function: bool = False
    is_enum: bool = False
    element_count: int = 0
    # For struct/union: field layout
    field_offsets: list[int] = field(default_factory=list)
    field_types: list[CType] = field(default_factory=list)
    field_names: list[str] = field(default_factory=list)

    @property
    def size_bytes(self) -> int:
        return (self.size_bits + 7) // 8

    @property
    def align_bytes(self) -> int:
        return (self.align_bits + 7) // 8


# ---------------------------------------------------------------------------
# Struct/Union layout entry
# ---------------------------------------------------------------------------

@dataclass
class StructLayout:
    """Computed layout for a struct or union."""
    name: str
    size_bits: int = 0
    align_bits: int = 8
    is_packed: bool = False
    fields: list[FieldLayout] = field(default_factory=list)


@dataclass
class FieldLayout:
    """Layout of a single struct/union field."""
    name: str
    type_name: CType
    offset_bits: int = 0
    size_bits: int = 0
    align_bits: int = 8
    bitfield_width: Optional[int] = None


# ---------------------------------------------------------------------------
# Integer rank (C11 6.3.1.1)
# ---------------------------------------------------------------------------

def _integer_rank(ty: IntCType) -> int:
    """Return the integer conversion rank of a type."""
    if ty.is_bool:
        return 0
    if ty.is_char:
        return 1
    if ty.is_short:
        return 2
    if ty.is_int:
        return 3
    if ty.is_long:
        return 4
    if ty.is_long_long:
        return 5
    if ty.is_int128:
        return 6
    return 3  # default to int rank


# ---------------------------------------------------------------------------
# C Type Resolver
# ---------------------------------------------------------------------------

class CTypeResolver:
    """Resolves C types: typedef expansion, layout computation, promotion rules.

    Usage::

        resolver = CTypeResolver()
        resolver.register_typedef("size_t", IntCType(is_unsigned=True, is_long=True))
        resolver.register_struct("point", struct_decl)

        info = resolver.resolve(some_type)
        promoted = resolver.integer_promote(some_int_type)
    """

    def __init__(self, platform: PlatformConfig | None = None) -> None:
        self._platform = platform or PlatformConfig.lp64()
        self._typedefs: dict[str, CType] = {}
        self._structs: dict[str, StructDecl] = {}
        self._unions: dict[str, UnionDecl] = {}
        self._enums: dict[str, EnumDecl] = {}
        self._struct_layouts: dict[str, StructLayout] = {}
        self._union_layouts: dict[str, StructLayout] = {}
        self._register_builtin_typedefs()

    def _register_builtin_typedefs(self) -> None:
        """Register platform-specific type aliases."""
        ptr_size = self._platform.pointer_size
        long_size = self._platform.long_size

        # size_t and ssize_t
        if ptr_size == 64:
            self._typedefs["size_t"] = IntCType(is_unsigned=True, is_long=True, is_signed=False)
            self._typedefs["ssize_t"] = IntCType(is_signed=True, is_long=True)
            self._typedefs["ptrdiff_t"] = IntCType(is_signed=True, is_long=True)
            self._typedefs["intptr_t"] = IntCType(is_signed=True, is_long=True)
            self._typedefs["uintptr_t"] = IntCType(is_unsigned=True, is_long=True, is_signed=False)
        else:
            self._typedefs["size_t"] = IntCType(is_unsigned=True, is_int=True, is_signed=False)
            self._typedefs["ssize_t"] = IntCType(is_signed=True, is_int=True)
            self._typedefs["ptrdiff_t"] = IntCType(is_signed=True, is_int=True)
            self._typedefs["intptr_t"] = IntCType(is_signed=True, is_int=True)
            self._typedefs["uintptr_t"] = IntCType(is_unsigned=True, is_int=True, is_signed=False)

        # Fixed-width integer types
        self._typedefs["int8_t"] = IntCType(is_signed=True, is_char=True)
        self._typedefs["int16_t"] = IntCType(is_signed=True, is_short=True)
        self._typedefs["int32_t"] = IntCType(is_signed=True, is_int=True)
        self._typedefs["int64_t"] = IntCType(is_signed=True, is_long_long=True)
        self._typedefs["uint8_t"] = IntCType(is_unsigned=True, is_char=True, is_signed=False)
        self._typedefs["uint16_t"] = IntCType(is_unsigned=True, is_short=True, is_signed=False)
        self._typedefs["uint32_t"] = IntCType(is_unsigned=True, is_int=True, is_signed=False)
        self._typedefs["uint64_t"] = IntCType(is_unsigned=True, is_long_long=True, is_signed=False)

        # Least and fast types
        for prefix in ("int_least", "int_fast"):
            for w, base in ((8, "char"), (16, "short"), (32, "int"), (64, "long_long")):
                name_s = f"{prefix}{w}_t"
                name_u = f"u{prefix}{w}_t"
                kwargs = {f"is_{base}": True}
                self._typedefs[name_s] = IntCType(is_signed=True, **kwargs)
                self._typedefs[name_u] = IntCType(is_unsigned=True, is_signed=False, **kwargs)

        self._typedefs["intmax_t"] = IntCType(is_signed=True, is_long_long=True)
        self._typedefs["uintmax_t"] = IntCType(is_unsigned=True, is_long_long=True, is_signed=False)

        # wchar_t
        wchar_w = self._platform.wchar_size
        if wchar_w == 32:
            self._typedefs["wchar_t"] = IntCType(is_signed=True, is_int=True)
        else:
            self._typedefs["wchar_t"] = IntCType(is_unsigned=True, is_short=True, is_signed=False)

        self._typedefs["char16_t"] = IntCType(is_unsigned=True, is_short=True, is_signed=False)
        self._typedefs["char32_t"] = IntCType(is_unsigned=True, is_int=True, is_signed=False)

        # Bool
        self._typedefs["bool"] = IntCType(is_bool=True)

        # va_list
        self._typedefs["va_list"] = PointerCType(pointee=VoidCType())
        self._typedefs["__va_list_tag"] = StructRefCType(name="__va_list_tag")

    # -------------------------------------------------------------------
    # Registration
    # -------------------------------------------------------------------

    def register_typedef(self, name: str, underlying: CType) -> None:
        """Register a typedef name."""
        self._typedefs[name] = underlying

    def register_struct(self, name: str, decl: StructDecl) -> None:
        """Register a struct definition."""
        self._structs[name] = decl
        if decl.is_definition:
            self._struct_layouts[name] = self._compute_struct_layout(decl)

    def register_union(self, name: str, decl: UnionDecl) -> None:
        """Register a union definition."""
        self._unions[name] = decl
        if decl.is_definition:
            self._union_layouts[name] = self._compute_union_layout(decl)

    def register_enum(self, name: str, decl: EnumDecl) -> None:
        """Register an enum definition."""
        self._enums[name] = decl

    # -------------------------------------------------------------------
    # Typedef resolution
    # -------------------------------------------------------------------

    def resolve_typedef(self, ty: CType, max_depth: int = 32) -> CType:
        """Fully resolve a typedef chain."""
        depth = 0
        while isinstance(ty, TypedefRefCType) and depth < max_depth:
            resolved = self._typedefs.get(ty.name)
            if resolved is None:
                return ty
            ty = resolved
            depth += 1

        if isinstance(ty, QualifiedCType):
            ty = QualifiedCType(
                base=self.resolve_typedef(ty.base, max_depth),
                qualifiers=ty.qualifiers,
            )

        return ty

    def strip_qualifiers(self, ty: CType) -> CType:
        """Remove all type qualifiers."""
        ty = self.resolve_typedef(ty)
        if isinstance(ty, QualifiedCType):
            return self.strip_qualifiers(ty.base)
        return ty

    # -------------------------------------------------------------------
    # Size and alignment computation
    # -------------------------------------------------------------------

    def sizeof(self, ty: CType) -> int:
        """Compute sizeof(type) in bytes."""
        info = self.resolve(ty)
        return info.size_bytes

    def alignof(self, ty: CType) -> int:
        """Compute _Alignof(type) in bytes."""
        info = self.resolve(ty)
        return info.align_bytes

    def sizeof_bits(self, ty: CType) -> int:
        """Compute sizeof(type) in bits."""
        info = self.resolve(ty)
        return info.size_bits

    def resolve(self, ty: CType) -> TypeInfo:
        """Resolve a C type to its TypeInfo (size, alignment, properties)."""
        ty = self.resolve_typedef(ty)

        if isinstance(ty, QualifiedCType):
            return self.resolve(ty.base)

        if isinstance(ty, AtomicCType):
            return self.resolve(ty.base)

        if isinstance(ty, VoidCType):
            return TypeInfo(size_bits=0, align_bits=8, is_void=True)

        if isinstance(ty, IntCType):
            return self._resolve_int_type(ty)

        if isinstance(ty, FloatCType):
            return self._resolve_float_type(ty)

        if isinstance(ty, PointerCType):
            ps = self._platform.pointer_size
            return TypeInfo(size_bits=ps, align_bits=ps, is_pointer=True)

        if isinstance(ty, ArrayCType):
            elem_info = self.resolve(ty.element)
            if ty.size is not None and isinstance(ty.size, IntLiteral):
                count = ty.size.value
                total = elem_info.size_bits * count
                return TypeInfo(
                    size_bits=total,
                    align_bits=elem_info.align_bits,
                    is_array=True,
                    element_count=count,
                )
            return TypeInfo(
                size_bits=0,
                align_bits=elem_info.align_bits,
                is_array=True,
            )

        if isinstance(ty, FunctionCType):
            return TypeInfo(size_bits=0, align_bits=8, is_function=True)

        if isinstance(ty, StructRefCType):
            return self._resolve_struct_type(ty.name)

        if isinstance(ty, UnionRefCType):
            return self._resolve_union_type(ty.name)

        if isinstance(ty, EnumRefCType):
            # Enums are int-sized by default in C
            return TypeInfo(size_bits=32, align_bits=32, is_signed=True, is_integer=True, is_enum=True)

        # Unknown type
        return TypeInfo(size_bits=0, align_bits=8)

    def _resolve_int_type(self, ty: IntCType) -> TypeInfo:
        """Resolve an integer type to its TypeInfo."""
        width = self._int_width(ty)
        is_signed = ty.is_signed and not ty.is_unsigned
        if ty.is_char and not ty.is_signed and not ty.is_unsigned:
            is_signed = self._platform.char_is_signed

        align = min(width, 64)  # Alignment caps at 64 bits for most platforms
        if width <= 8:
            align = 8
        elif width <= 16:
            align = 16
        elif width <= 32:
            align = 32
        elif width <= 64:
            align = 64
        else:
            align = 128

        return TypeInfo(
            size_bits=width,
            align_bits=align,
            is_signed=is_signed,
            is_integer=True,
        )

    def _int_width(self, ty: IntCType) -> int:
        """Get the bit width of an integer type."""
        if ty.is_bool:
            return 1
        if ty.is_char:
            return 8
        if ty.is_short:
            return 16
        if ty.is_int128:
            return 128
        if ty.is_long_long:
            return 64
        if ty.is_long:
            return self._platform.long_size
        return 32  # int

    def _resolve_float_type(self, ty: FloatCType) -> TypeInfo:
        """Resolve a float type to its TypeInfo."""
        if ty.is_float:
            return TypeInfo(size_bits=32, align_bits=32, is_float=True)
        if ty.is_long_double:
            size = self._platform.long_double_size
            return TypeInfo(size_bits=size, align_bits=min(size, 128), is_float=True)
        # double
        return TypeInfo(size_bits=64, align_bits=64, is_float=True)

    def _resolve_struct_type(self, name: str) -> TypeInfo:
        """Resolve a struct type to its TypeInfo."""
        layout = self._struct_layouts.get(name)
        if layout is None:
            return TypeInfo(size_bits=0, align_bits=8, is_struct=True)
        return TypeInfo(
            size_bits=layout.size_bits,
            align_bits=layout.align_bits,
            is_struct=True,
            field_offsets=[f.offset_bits for f in layout.fields],
            field_types=[f.type_name for f in layout.fields],
            field_names=[f.name for f in layout.fields],
        )

    def _resolve_union_type(self, name: str) -> TypeInfo:
        """Resolve a union type to its TypeInfo."""
        layout = self._union_layouts.get(name)
        if layout is None:
            return TypeInfo(size_bits=0, align_bits=8, is_union=True)
        return TypeInfo(
            size_bits=layout.size_bits,
            align_bits=layout.align_bits,
            is_union=True,
            field_offsets=[0] * len(layout.fields),
            field_types=[f.type_name for f in layout.fields],
            field_names=[f.name for f in layout.fields],
        )

    # -------------------------------------------------------------------
    # Struct/Union layout computation
    # -------------------------------------------------------------------

    def _compute_struct_layout(self, decl: StructDecl) -> StructLayout:
        """Compute the layout of a struct."""
        layout = StructLayout(name=decl.name, is_packed=decl.is_packed)
        current_offset = 0
        max_align = 8

        for f in decl.fields:
            if f.type_name is None:
                continue
            finfo = self.resolve(f.type_name)
            f_align = 8 if decl.is_packed else finfo.align_bits

            # Handle bitfields
            bf_width = None
            if f.bitfield_width is not None and isinstance(f.bitfield_width, IntLiteral):
                bf_width = f.bitfield_width.value
                f_size = bf_width
            else:
                f_size = finfo.size_bits

            # Align current offset
            if not decl.is_packed and bf_width is None:
                current_offset = _align_up(current_offset, f_align)

            fl = FieldLayout(
                name=f.name,
                type_name=f.type_name,
                offset_bits=current_offset,
                size_bits=f_size,
                align_bits=f_align,
                bitfield_width=bf_width,
            )
            layout.fields.append(fl)
            current_offset += f_size
            max_align = max(max_align, f_align)

        # Pad struct to alignment
        if not decl.is_packed:
            current_offset = _align_up(current_offset, max_align)

        # Apply alignment override
        if decl.alignment is not None:
            max_align = decl.alignment * 8

        layout.size_bits = current_offset
        layout.align_bits = max_align
        return layout

    def _compute_union_layout(self, decl: UnionDecl) -> StructLayout:
        """Compute the layout of a union."""
        layout = StructLayout(name=decl.name)
        max_size = 0
        max_align = 8

        for f in decl.fields:
            if f.type_name is None:
                continue
            finfo = self.resolve(f.type_name)
            fl = FieldLayout(
                name=f.name,
                type_name=f.type_name,
                offset_bits=0,
                size_bits=finfo.size_bits,
                align_bits=finfo.align_bits,
            )
            layout.fields.append(fl)
            max_size = max(max_size, finfo.size_bits)
            max_align = max(max_align, finfo.align_bits)

        layout.size_bits = _align_up(max_size, max_align)
        layout.align_bits = max_align
        return layout

    # -------------------------------------------------------------------
    # Integer promotions (C11 6.3.1.1)
    # -------------------------------------------------------------------

    def integer_promote(self, ty: CType) -> CType:
        """Apply integer promotion rules (C11 6.3.1.1)."""
        ty = self.resolve_typedef(ty)
        ty = self.strip_qualifiers(ty)

        if not isinstance(ty, IntCType):
            return ty

        # _Bool, char, short all promote to int
        if ty.is_bool or ty.is_char or ty.is_short:
            width = self._int_width(ty)
            # If all values of the original type can be represented as int
            if width < 32:
                return IntCType(is_signed=True, is_int=True)

        # If type rank < int rank, promote to int or unsigned int
        rank = _integer_rank(ty)
        if rank < 3:  # rank of int
            if self._int_width(ty) < 32:
                return IntCType(is_signed=True, is_int=True)
            if ty.is_unsigned:
                return IntCType(is_unsigned=True, is_int=True, is_signed=False)
            return IntCType(is_signed=True, is_int=True)

        return ty

    def usual_arithmetic_conversions(self, ty1: CType, ty2: CType) -> CType:
        """Apply usual arithmetic conversions (C11 6.3.1.8)."""
        ty1 = self.resolve_typedef(ty1)
        ty2 = self.resolve_typedef(ty2)
        ty1 = self.strip_qualifiers(ty1)
        ty2 = self.strip_qualifiers(ty2)

        # If either is long double
        if isinstance(ty1, FloatCType) and ty1.is_long_double:
            return ty1
        if isinstance(ty2, FloatCType) and ty2.is_long_double:
            return ty2

        # If either is double
        if isinstance(ty1, FloatCType) and ty1.is_double:
            return ty1
        if isinstance(ty2, FloatCType) and ty2.is_double:
            return ty2

        # If either is float
        if isinstance(ty1, FloatCType) and ty1.is_float:
            return ty1
        if isinstance(ty2, FloatCType) and ty2.is_float:
            return ty2

        # Integer promotions first
        ty1 = self.integer_promote(ty1)
        ty2 = self.integer_promote(ty2)

        if not isinstance(ty1, IntCType) or not isinstance(ty2, IntCType):
            return ty1  # fallback

        # Same type
        w1 = self._int_width(ty1)
        w2 = self._int_width(ty2)
        r1 = _integer_rank(ty1)
        r2 = _integer_rank(ty2)

        s1 = ty1.is_signed and not ty1.is_unsigned
        s2 = ty2.is_signed and not ty2.is_unsigned

        if s1 == s2:
            # Same signedness: use higher rank
            return ty1 if r1 >= r2 else ty2

        # Different signedness
        unsigned_ty = ty1 if not s1 else ty2
        signed_ty = ty2 if not s1 else ty1
        u_rank = _integer_rank(unsigned_ty)
        s_rank = _integer_rank(signed_ty)

        if u_rank >= s_rank:
            return unsigned_ty

        u_width = self._int_width(unsigned_ty)
        s_width = self._int_width(signed_ty)

        if s_width > u_width:
            return signed_ty

        # Convert both to unsigned version of signed type
        return IntCType(
            is_unsigned=True,
            is_signed=False,
            is_char=signed_ty.is_char,
            is_short=signed_ty.is_short,
            is_int=signed_ty.is_int,
            is_long=signed_ty.is_long,
            is_long_long=signed_ty.is_long_long,
            is_int128=signed_ty.is_int128,
        )

    # -------------------------------------------------------------------
    # Type compatibility checking
    # -------------------------------------------------------------------

    def are_compatible(self, ty1: CType, ty2: CType) -> bool:
        """Check if two C types are compatible (C11 6.2.7)."""
        ty1 = self.resolve_typedef(ty1)
        ty2 = self.resolve_typedef(ty2)
        ty1 = self.strip_qualifiers(ty1)
        ty2 = self.strip_qualifiers(ty2)

        if type(ty1) != type(ty2):
            return False

        if isinstance(ty1, VoidCType):
            return True

        if isinstance(ty1, IntCType) and isinstance(ty2, IntCType):
            return (self._int_width(ty1) == self._int_width(ty2) and
                    (ty1.is_signed == ty2.is_signed or
                     ty1.is_unsigned == ty2.is_unsigned))

        if isinstance(ty1, FloatCType) and isinstance(ty2, FloatCType):
            return ty1.width_bits == ty2.width_bits

        if isinstance(ty1, PointerCType) and isinstance(ty2, PointerCType):
            return self.are_compatible(ty1.pointee, ty2.pointee)

        if isinstance(ty1, ArrayCType) and isinstance(ty2, ArrayCType):
            return self.are_compatible(ty1.element, ty2.element)

        if isinstance(ty1, StructRefCType) and isinstance(ty2, StructRefCType):
            return ty1.name == ty2.name

        if isinstance(ty1, UnionRefCType) and isinstance(ty2, UnionRefCType):
            return ty1.name == ty2.name

        if isinstance(ty1, EnumRefCType) and isinstance(ty2, EnumRefCType):
            return ty1.name == ty2.name

        if isinstance(ty1, FunctionCType) and isinstance(ty2, FunctionCType):
            if not self.are_compatible(ty1.return_type, ty2.return_type):
                return False
            if len(ty1.params) != len(ty2.params):
                return False
            for p1, p2 in zip(ty1.params, ty2.params):
                if p1.type_name and p2.type_name:
                    if not self.are_compatible(p1.type_name, p2.type_name):
                        return False
            return ty1.is_variadic == ty2.is_variadic

        return False

    def is_assignable(self, target: CType, source: CType) -> bool:
        """Check if source type can be assigned to target type."""
        target = self.resolve_typedef(target)
        source = self.resolve_typedef(source)
        target = self.strip_qualifiers(target)
        source = self.strip_qualifiers(source)

        if self.are_compatible(target, source):
            return True

        # Arithmetic types are interconvertible
        t_info = self.resolve(target)
        s_info = self.resolve(source)
        if (t_info.is_integer or t_info.is_float) and (s_info.is_integer or s_info.is_float):
            return True

        # Pointer conversions
        if t_info.is_pointer and s_info.is_pointer:
            return True  # C allows with warning

        # Pointer/integer conversion
        if (t_info.is_pointer and s_info.is_integer) or (t_info.is_integer and s_info.is_pointer):
            return True  # C allows with warning

        # void* conversion
        if isinstance(target, PointerCType) and isinstance(target.pointee, VoidCType):
            return s_info.is_pointer
        if isinstance(source, PointerCType) and isinstance(source.pointee, VoidCType):
            return t_info.is_pointer

        return False

    # -------------------------------------------------------------------
    # Lookup helpers
    # -------------------------------------------------------------------

    def get_struct_layout(self, name: str) -> Optional[StructLayout]:
        """Get the computed layout of a struct."""
        return self._struct_layouts.get(name)

    def get_union_layout(self, name: str) -> Optional[StructLayout]:
        """Get the computed layout of a union."""
        return self._union_layouts.get(name)

    def get_typedef(self, name: str) -> Optional[CType]:
        """Get the underlying type of a typedef."""
        return self._typedefs.get(name)

    def get_field_index(self, struct_name: str, field_name: str) -> Optional[int]:
        """Get the index of a field in a struct."""
        layout = self._struct_layouts.get(struct_name)
        if layout is None:
            return None
        for i, f in enumerate(layout.fields):
            if f.name == field_name:
                return i
        return None

    def get_field_offset(self, struct_name: str, field_name: str) -> Optional[int]:
        """Get the byte offset of a field in a struct."""
        layout = self._struct_layouts.get(struct_name)
        if layout is None:
            return None
        for f in layout.fields:
            if f.name == field_name:
                return f.offset_bits // 8
        return None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _align_up(value: int, align: int) -> int:
    """Round value up to the next multiple of align."""
    if align <= 0:
        return value
    return ((value + align - 1) // align) * align
