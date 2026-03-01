"""
ABI compatibility checker for C/Rust FFI.
Checks struct layout, calling conventions, type sizes, enum representation,
function pointer compatibility, opaque type handling, and platform differences.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple, Any, Union
from enum import Enum, auto
import math


# ---------------------------------------------------------------------------
# Platform / ABI models
# ---------------------------------------------------------------------------

class Platform(Enum):
    LP64 = "LP64"          # Linux/macOS 64-bit
    LLP64 = "LLP64"       # Windows 64-bit
    ILP32 = "ILP32"       # 32-bit systems
    AARCH64 = "AARCH64"   # ARM 64-bit


class CallingConvention(Enum):
    CDECL = "cdecl"
    STDCALL = "stdcall"
    FASTCALL = "fastcall"
    SYSTEM = "system"
    WIN64 = "win64"
    SYSV64 = "sysv64"
    AAPCS = "aapcs"


class Endianness(Enum):
    LITTLE = "little"
    BIG = "big"


@dataclass
class PlatformInfo:
    name: Platform
    pointer_size: int = 8
    int_size: int = 4
    long_size: int = 8
    long_long_size: int = 8
    size_t_size: int = 8
    max_align: int = 16
    endianness: Endianness = Endianness.LITTLE
    char_signed: bool = True

    @classmethod
    def lp64(cls) -> "PlatformInfo":
        return cls(name=Platform.LP64, long_size=8)

    @classmethod
    def llp64(cls) -> "PlatformInfo":
        return cls(name=Platform.LLP64, long_size=4)

    @classmethod
    def ilp32(cls) -> "PlatformInfo":
        return cls(name=Platform.ILP32, pointer_size=4, long_size=4,
                   size_t_size=4, max_align=8)

    @classmethod
    def aarch64(cls) -> "PlatformInfo":
        return cls(name=Platform.AARCH64, max_align=16)


# ---------------------------------------------------------------------------
# Type descriptors
# ---------------------------------------------------------------------------

class TypeKind(Enum):
    PRIMITIVE = auto()
    STRUCT = auto()
    UNION = auto()
    ENUM = auto()
    ARRAY = auto()
    POINTER = auto()
    FUNCTION_POINTER = auto()
    VOID = auto()
    OPAQUE = auto()
    TYPEDEF = auto()


@dataclass
class TypeDescriptor:
    name: str
    kind: TypeKind
    size: Optional[int] = None
    alignment: Optional[int] = None
    fields: List["FieldDescriptor"] = field(default_factory=list)
    element_type: Optional["TypeDescriptor"] = None
    element_count: Optional[int] = None
    pointee_type: Optional["TypeDescriptor"] = None
    return_type: Optional["TypeDescriptor"] = None
    param_types: List["TypeDescriptor"] = field(default_factory=list)
    enum_variants: List["EnumVariant"] = field(default_factory=list)
    is_repr_c: bool = False
    is_packed: bool = False
    is_transparent: bool = False
    typedef_target: Optional["TypeDescriptor"] = None
    language: str = "c"
    attributes: Dict[str, Any] = field(default_factory=dict)

    def effective_alignment(self, platform: PlatformInfo) -> int:
        if self.is_packed:
            return 1
        if self.alignment is not None:
            return self.alignment
        return self._natural_alignment(platform)

    def _natural_alignment(self, platform: PlatformInfo) -> int:
        if self.kind == TypeKind.PRIMITIVE:
            return self.size or 1
        if self.kind == TypeKind.POINTER:
            return platform.pointer_size
        if self.kind == TypeKind.ARRAY:
            if self.element_type:
                return self.element_type.effective_alignment(platform)
        if self.kind in (TypeKind.STRUCT, TypeKind.UNION):
            if not self.fields:
                return 1
            return max(f.type_desc.effective_alignment(platform)
                       for f in self.fields if f.type_desc)
        if self.kind == TypeKind.ENUM:
            return self.size or 4
        return 1

    def compute_size(self, platform: PlatformInfo) -> int:
        if self.size is not None:
            return self.size
        if self.kind == TypeKind.POINTER:
            return platform.pointer_size
        if self.kind == TypeKind.ARRAY:
            if self.element_type and self.element_count:
                return self.element_type.compute_size(platform) * self.element_count
        if self.kind == TypeKind.STRUCT:
            return self._compute_struct_size(platform)
        if self.kind == TypeKind.UNION:
            if not self.fields:
                return 0
            return max(f.type_desc.compute_size(platform)
                       for f in self.fields if f.type_desc)
        if self.kind == TypeKind.ENUM:
            return self.size or 4
        return 0

    def _compute_struct_size(self, platform: PlatformInfo) -> int:
        offset = 0
        max_align = 1
        for fld in self.fields:
            if not fld.type_desc:
                continue
            falign = fld.type_desc.effective_alignment(platform)
            if not self.is_packed:
                padding = (falign - (offset % falign)) % falign
                offset += padding
            fld.offset = offset
            fld.padding_before = (falign - ((offset - (fld.padding_before or 0)) % falign)) % falign if not self.is_packed else 0
            fsize = fld.type_desc.compute_size(platform)
            offset += fsize
            max_align = max(max_align, falign)
        if not self.is_packed:
            tail_padding = (max_align - (offset % max_align)) % max_align
            offset += tail_padding
        return offset


@dataclass
class FieldDescriptor:
    name: str
    type_desc: Optional[TypeDescriptor] = None
    offset: Optional[int] = None
    bit_field_width: Optional[int] = None
    padding_before: Optional[int] = None
    is_flexible_array: bool = False


@dataclass
class EnumVariant:
    name: str
    value: Optional[int] = None
    has_payload: bool = False
    payload_types: List[TypeDescriptor] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Incompatibility report
# ---------------------------------------------------------------------------

class IncompatibilityKind(Enum):
    SIZE_MISMATCH = "size_mismatch"
    ALIGNMENT_MISMATCH = "alignment_mismatch"
    FIELD_ORDER_MISMATCH = "field_order_mismatch"
    FIELD_TYPE_MISMATCH = "field_type_mismatch"
    FIELD_COUNT_MISMATCH = "field_count_mismatch"
    MISSING_REPR_C = "missing_repr_c"
    CALLING_CONVENTION_MISMATCH = "calling_convention_mismatch"
    ENUM_REPRESENTATION_MISMATCH = "enum_representation_mismatch"
    FUNCTION_SIGNATURE_MISMATCH = "function_signature_mismatch"
    OPAQUE_TYPE_ISSUE = "opaque_type_issue"
    PLATFORM_SPECIFIC = "platform_specific"
    PADDING_MISMATCH = "padding_mismatch"
    BIT_FIELD_ISSUE = "bit_field_issue"
    FLEXIBLE_ARRAY_ISSUE = "flexible_array_issue"


@dataclass
class Incompatibility:
    kind: IncompatibilityKind
    c_type: str
    rust_type: str
    description: str
    severity: str = "error"
    suggestion: str = ""
    platform: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "kind": self.kind.value,
            "c_type": self.c_type,
            "rust_type": self.rust_type,
            "description": self.description,
            "severity": self.severity,
            "suggestion": self.suggestion,
            "platform": self.platform,
        }


@dataclass
class ABIReport:
    compatible: bool = True
    incompatibilities: List[Incompatibility] = field(default_factory=list)
    size_mismatches: List[Dict[str, Any]] = field(default_factory=list)
    alignment_issues: List[Dict[str, Any]] = field(default_factory=list)
    suggestions: List[str] = field(default_factory=list)
    types_checked: int = 0
    functions_checked: int = 0
    platform: str = "LP64"
    warnings: List[str] = field(default_factory=list)

    def add_incompatibility(self, incompat: Incompatibility) -> None:
        self.incompatibilities.append(incompat)
        if incompat.severity == "error":
            self.compatible = False
        if incompat.kind == IncompatibilityKind.SIZE_MISMATCH:
            self.size_mismatches.append(incompat.to_dict())
        if incompat.kind == IncompatibilityKind.ALIGNMENT_MISMATCH:
            self.alignment_issues.append(incompat.to_dict())
        if incompat.suggestion:
            self.suggestions.append(incompat.suggestion)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "compatible": self.compatible,
            "incompatibility_count": len(self.incompatibilities),
            "incompatibilities": [i.to_dict() for i in self.incompatibilities],
            "size_mismatches": self.size_mismatches,
            "alignment_issues": self.alignment_issues,
            "suggestions": self.suggestions,
            "types_checked": self.types_checked,
            "functions_checked": self.functions_checked,
            "platform": self.platform,
            "warnings": self.warnings,
        }


# ---------------------------------------------------------------------------
# Primitive type maps
# ---------------------------------------------------------------------------

C_PRIMITIVE_SIZES_LP64: Dict[str, Tuple[int, int]] = {
    "char": (1, 1), "signed char": (1, 1), "unsigned char": (1, 1),
    "short": (2, 2), "unsigned short": (2, 2),
    "int": (4, 4), "unsigned int": (4, 4),
    "long": (8, 8), "unsigned long": (8, 8),
    "long long": (8, 8), "unsigned long long": (8, 8),
    "float": (4, 4), "double": (8, 8),
    "long double": (16, 16),
    "_Bool": (1, 1), "bool": (1, 1),
    "size_t": (8, 8), "ssize_t": (8, 8),
    "ptrdiff_t": (8, 8),
    "int8_t": (1, 1), "uint8_t": (1, 1),
    "int16_t": (2, 2), "uint16_t": (2, 2),
    "int32_t": (4, 4), "uint32_t": (4, 4),
    "int64_t": (8, 8), "uint64_t": (8, 8),
    "intptr_t": (8, 8), "uintptr_t": (8, 8),
    "void*": (8, 8),
}

C_PRIMITIVE_SIZES_LLP64: Dict[str, Tuple[int, int]] = {
    **C_PRIMITIVE_SIZES_LP64,
    "long": (4, 4), "unsigned long": (4, 4),
}

RUST_PRIMITIVE_SIZES: Dict[str, Tuple[int, int]] = {
    "i8": (1, 1), "u8": (1, 1),
    "i16": (2, 2), "u16": (2, 2),
    "i32": (4, 4), "u32": (4, 4),
    "i64": (8, 8), "u64": (8, 8),
    "i128": (16, 16), "u128": (16, 16),
    "f32": (4, 4), "f64": (8, 8),
    "bool": (1, 1), "char": (4, 4),
    "isize": (8, 8), "usize": (8, 8),
    "*mut ()": (8, 8), "*const ()": (8, 8),
    "*mut c_void": (8, 8), "*const c_void": (8, 8),
}

C_TO_RUST_TYPE_MAP: Dict[str, str] = {
    "char": "i8", "signed char": "i8", "unsigned char": "u8",
    "short": "i16", "unsigned short": "u16",
    "int": "i32", "unsigned int": "u32",
    "long": "i64", "unsigned long": "u64",
    "long long": "i64", "unsigned long long": "u64",
    "float": "f32", "double": "f64",
    "_Bool": "bool", "bool": "bool",
    "size_t": "usize", "ssize_t": "isize",
    "ptrdiff_t": "isize",
    "int8_t": "i8", "uint8_t": "u8",
    "int16_t": "i16", "uint16_t": "u16",
    "int32_t": "i32", "uint32_t": "u32",
    "int64_t": "i64", "uint64_t": "u64",
    "intptr_t": "isize", "uintptr_t": "usize",
    "void*": "*mut c_void",
}

# LLP64 overrides
C_TO_RUST_TYPE_MAP_LLP64: Dict[str, str] = {
    **C_TO_RUST_TYPE_MAP,
    "long": "i32", "unsigned long": "u32",
}


# ---------------------------------------------------------------------------
# ABI Checker
# ---------------------------------------------------------------------------

class ABIChecker:
    """Check ABI compatibility between C and Rust types for FFI."""

    def __init__(self, platform: Optional[PlatformInfo] = None):
        self.platform = platform or PlatformInfo.lp64()
        self._report = ABIReport(platform=self.platform.name.value)
        self._c_prim_sizes = (C_PRIMITIVE_SIZES_LP64
                              if self.platform.name != Platform.LLP64
                              else C_PRIMITIVE_SIZES_LLP64)
        self._type_map = (C_TO_RUST_TYPE_MAP
                          if self.platform.name != Platform.LLP64
                          else C_TO_RUST_TYPE_MAP_LLP64)

    def _reset(self) -> None:
        self._report = ABIReport(platform=self.platform.name.value)

    # --- size/alignment helpers ---
    def _c_type_size(self, td: TypeDescriptor) -> int:
        if td.size is not None:
            return td.size
        if td.kind == TypeKind.PRIMITIVE:
            info = self._c_prim_sizes.get(td.name)
            if info:
                return info[0]
        return td.compute_size(self.platform)

    def _c_type_align(self, td: TypeDescriptor) -> int:
        if td.alignment is not None:
            return td.alignment
        if td.kind == TypeKind.PRIMITIVE:
            info = self._c_prim_sizes.get(td.name)
            if info:
                return info[1]
        return td.effective_alignment(self.platform)

    def _rust_type_size(self, td: TypeDescriptor) -> int:
        if td.size is not None:
            return td.size
        if td.kind == TypeKind.PRIMITIVE:
            info = RUST_PRIMITIVE_SIZES.get(td.name)
            if info:
                return info[0]
        return td.compute_size(self.platform)

    def _rust_type_align(self, td: TypeDescriptor) -> int:
        if td.alignment is not None:
            return td.alignment
        if td.kind == TypeKind.PRIMITIVE:
            info = RUST_PRIMITIVE_SIZES.get(td.name)
            if info:
                return info[1]
        return td.effective_alignment(self.platform)

    # --- struct layout ---
    def _compute_c_struct_layout(self, td: TypeDescriptor) -> List[Dict[str, Any]]:
        layout = []
        offset = 0
        for fld in td.fields:
            if not fld.type_desc:
                continue
            falign = self._c_type_align(fld.type_desc)
            if not td.is_packed:
                padding = (falign - (offset % falign)) % falign
                offset += padding
            else:
                padding = 0
            fsize = self._c_type_size(fld.type_desc)
            layout.append({
                "name": fld.name,
                "offset": offset,
                "size": fsize,
                "alignment": falign,
                "padding_before": padding,
                "type": fld.type_desc.name,
            })
            offset += fsize
        return layout

    def _compute_rust_struct_layout(self, td: TypeDescriptor) -> List[Dict[str, Any]]:
        if not td.is_repr_c:
            self._report.add_incompatibility(Incompatibility(
                kind=IncompatibilityKind.MISSING_REPR_C,
                c_type="", rust_type=td.name,
                description=f"Rust struct `{td.name}` is not #[repr(C)]; layout is unspecified",
                severity="error",
                suggestion=f"Add #[repr(C)] to struct `{td.name}`",
            ))
        layout = []
        offset = 0
        for fld in td.fields:
            if not fld.type_desc:
                continue
            falign = self._rust_type_align(fld.type_desc)
            if not td.is_packed:
                padding = (falign - (offset % falign)) % falign
                offset += padding
            else:
                padding = 0
            fsize = self._rust_type_size(fld.type_desc)
            layout.append({
                "name": fld.name,
                "offset": offset,
                "size": fsize,
                "alignment": falign,
                "padding_before": padding,
                "type": fld.type_desc.name,
            })
            offset += fsize
        return layout

    # --- comparison ---
    def _compare_struct_layouts(self, c_type: TypeDescriptor,
                                rust_type: TypeDescriptor) -> None:
        c_layout = self._compute_c_struct_layout(c_type)
        r_layout = self._compute_rust_struct_layout(rust_type)

        if len(c_type.fields) != len(rust_type.fields):
            self._report.add_incompatibility(Incompatibility(
                kind=IncompatibilityKind.FIELD_COUNT_MISMATCH,
                c_type=c_type.name, rust_type=rust_type.name,
                description=f"Field count mismatch: C has {len(c_type.fields)}, Rust has {len(rust_type.fields)}",
                severity="error",
            ))
            return

        for i, (cf, rf) in enumerate(zip(c_layout, r_layout)):
            if cf["offset"] != rf["offset"]:
                self._report.add_incompatibility(Incompatibility(
                    kind=IncompatibilityKind.PADDING_MISMATCH,
                    c_type=c_type.name, rust_type=rust_type.name,
                    description=f"Field `{cf['name']}` offset mismatch: C={cf['offset']}, Rust={rf['offset']}",
                    severity="error",
                    suggestion="Check field ordering and padding; use #[repr(C)] on Rust side",
                ))
            if cf["size"] != rf["size"]:
                self._report.add_incompatibility(Incompatibility(
                    kind=IncompatibilityKind.SIZE_MISMATCH,
                    c_type=f"{c_type.name}.{cf['name']} ({cf['type']})",
                    rust_type=f"{rust_type.name}.{rf['name']} ({rf['type']})",
                    description=f"Field size mismatch: C={cf['size']}, Rust={rf['size']}",
                    severity="error",
                    suggestion=f"Use equivalent types: C `{cf['type']}` -> Rust `{self._type_map.get(cf['type'], '?')}`",
                ))
            if cf["alignment"] != rf["alignment"]:
                self._report.add_incompatibility(Incompatibility(
                    kind=IncompatibilityKind.ALIGNMENT_MISMATCH,
                    c_type=c_type.name, rust_type=rust_type.name,
                    description=f"Field `{cf['name']}` alignment mismatch: C={cf['alignment']}, Rust={rf['alignment']}",
                    severity="warning",
                ))

        c_total = self._c_type_size(c_type)
        r_total = self._rust_type_size(rust_type)
        if c_total != r_total:
            self._report.add_incompatibility(Incompatibility(
                kind=IncompatibilityKind.SIZE_MISMATCH,
                c_type=c_type.name, rust_type=rust_type.name,
                description=f"Total struct size mismatch: C={c_total}, Rust={r_total}",
                severity="error",
            ))

        c_align = self._c_type_align(c_type)
        r_align = self._rust_type_align(rust_type)
        if c_align != r_align:
            self._report.add_incompatibility(Incompatibility(
                kind=IncompatibilityKind.ALIGNMENT_MISMATCH,
                c_type=c_type.name, rust_type=rust_type.name,
                description=f"Struct alignment mismatch: C={c_align}, Rust={r_align}",
                severity="warning",
            ))

        for fld in c_type.fields:
            if fld.bit_field_width is not None:
                self._report.add_incompatibility(Incompatibility(
                    kind=IncompatibilityKind.BIT_FIELD_ISSUE,
                    c_type=c_type.name, rust_type=rust_type.name,
                    description=f"C bit-field `{fld.name}:{fld.bit_field_width}` has no direct Rust equivalent",
                    severity="warning",
                    suggestion="Use explicit bit manipulation in Rust",
                ))
            if fld.is_flexible_array:
                self._report.add_incompatibility(Incompatibility(
                    kind=IncompatibilityKind.FLEXIBLE_ARRAY_ISSUE,
                    c_type=c_type.name, rust_type=rust_type.name,
                    description=f"C flexible array member `{fld.name}[]` needs special handling in Rust",
                    severity="warning",
                    suggestion="Use a zero-length array [T; 0] or PhantomData in Rust",
                ))

    # --- calling conventions ---
    def _check_calling_convention(self, c_cc: CallingConvention,
                                  rust_cc: CallingConvention,
                                  func_name: str) -> None:
        compatible_pairs = {
            (CallingConvention.CDECL, CallingConvention.CDECL),
            (CallingConvention.STDCALL, CallingConvention.STDCALL),
            (CallingConvention.FASTCALL, CallingConvention.FASTCALL),
            (CallingConvention.SYSTEM, CallingConvention.STDCALL),
            (CallingConvention.SYSTEM, CallingConvention.CDECL),
            (CallingConvention.SYSV64, CallingConvention.CDECL),
            (CallingConvention.WIN64, CallingConvention.STDCALL),
        }
        if c_cc == rust_cc:
            return
        if (c_cc, rust_cc) in compatible_pairs or (rust_cc, c_cc) in compatible_pairs:
            self._report.warnings.append(
                f"Function `{func_name}`: calling conventions differ ({c_cc.value} vs {rust_cc.value}) but may be compatible on this platform"
            )
            return
        self._report.add_incompatibility(Incompatibility(
            kind=IncompatibilityKind.CALLING_CONVENTION_MISMATCH,
            c_type=func_name, rust_type=func_name,
            description=f"Calling convention mismatch: C uses {c_cc.value}, Rust uses {rust_cc.value}",
            severity="error",
            suggestion=f'Use extern "{c_cc.value}" in Rust declaration',
        ))

    # --- enum comparison ---
    def _compare_enums(self, c_enum: TypeDescriptor,
                       rust_enum: TypeDescriptor) -> None:
        c_has_payload = any(v.has_payload for v in c_enum.enum_variants)
        r_has_payload = any(v.has_payload for v in rust_enum.enum_variants)

        if not c_has_payload and r_has_payload:
            self._report.add_incompatibility(Incompatibility(
                kind=IncompatibilityKind.ENUM_REPRESENTATION_MISMATCH,
                c_type=c_enum.name, rust_type=rust_enum.name,
                description="C enum is a plain integer, but Rust enum has payload variants (tagged union)",
                severity="error",
                suggestion="Use #[repr(C)] or #[repr(i32)] on Rust enum without payloads for C compatibility",
            ))
            return

        if not c_has_payload and not r_has_payload:
            c_size = self._c_type_size(c_enum)
            r_size = self._rust_type_size(rust_enum)
            if c_size != r_size:
                self._report.add_incompatibility(Incompatibility(
                    kind=IncompatibilityKind.SIZE_MISMATCH,
                    c_type=c_enum.name, rust_type=rust_enum.name,
                    description=f"Enum size mismatch: C={c_size}, Rust={r_size}",
                    severity="error",
                    suggestion="Use #[repr(c_int)] or #[repr(i32)] to match C enum size",
                ))

            c_names = {v.name for v in c_enum.enum_variants}
            r_names = {v.name for v in rust_enum.enum_variants}
            missing_in_rust = c_names - r_names
            extra_in_rust = r_names - c_names
            if missing_in_rust:
                self._report.warnings.append(
                    f"Enum `{c_enum.name}`: C variants not in Rust: {missing_in_rust}"
                )
            if extra_in_rust:
                self._report.warnings.append(
                    f"Enum `{rust_enum.name}`: Rust variants not in C: {extra_in_rust}"
                )

            c_values = {v.name: v.value for v in c_enum.enum_variants if v.value is not None}
            r_values = {v.name: v.value for v in rust_enum.enum_variants if v.value is not None}
            for name in c_names & r_names:
                cv = c_values.get(name)
                rv = r_values.get(name)
                if cv is not None and rv is not None and cv != rv:
                    self._report.add_incompatibility(Incompatibility(
                        kind=IncompatibilityKind.ENUM_REPRESENTATION_MISMATCH,
                        c_type=c_enum.name, rust_type=rust_enum.name,
                        description=f"Enum variant `{name}` value mismatch: C={cv}, Rust={rv}",
                        severity="error",
                    ))

        if not rust_enum.is_repr_c:
            self._report.add_incompatibility(Incompatibility(
                kind=IncompatibilityKind.MISSING_REPR_C,
                c_type=c_enum.name, rust_type=rust_enum.name,
                description=f"Rust enum `{rust_enum.name}` is not #[repr(C)] or #[repr(i32)]",
                severity="error",
                suggestion="Add #[repr(C)] or appropriate integer repr",
            ))

    # --- function pointer ---
    def _compare_function_pointers(self, c_fp: TypeDescriptor,
                                   rust_fp: TypeDescriptor,
                                   context: str = "") -> None:
        if c_fp.return_type and rust_fp.return_type:
            self._compare_types(c_fp.return_type, rust_fp.return_type,
                                f"{context} return type")
        elif bool(c_fp.return_type) != bool(rust_fp.return_type):
            c_ret = c_fp.return_type.name if c_fp.return_type else "void"
            r_ret = rust_fp.return_type.name if rust_fp.return_type else "()"
            if not (c_ret == "void" and r_ret == "()"):
                self._report.add_incompatibility(Incompatibility(
                    kind=IncompatibilityKind.FUNCTION_SIGNATURE_MISMATCH,
                    c_type=c_fp.name, rust_type=rust_fp.name,
                    description=f"Return type mismatch: C={c_ret}, Rust={r_ret}",
                    severity="error",
                ))

        if len(c_fp.param_types) != len(rust_fp.param_types):
            self._report.add_incompatibility(Incompatibility(
                kind=IncompatibilityKind.FUNCTION_SIGNATURE_MISMATCH,
                c_type=c_fp.name, rust_type=rust_fp.name,
                description=f"Parameter count mismatch: C has {len(c_fp.param_types)}, Rust has {len(rust_fp.param_types)}",
                severity="error",
            ))
            return

        for i, (cp, rp) in enumerate(zip(c_fp.param_types, rust_fp.param_types)):
            self._compare_types(cp, rp, f"{context} param {i}")

    # --- opaque types ---
    def _check_opaque_type(self, c_type: TypeDescriptor,
                           rust_type: TypeDescriptor) -> None:
        if c_type.kind == TypeKind.POINTER and c_type.name == "void*":
            if rust_type.kind == TypeKind.POINTER:
                r_pointee = rust_type.pointee_type
                if r_pointee and r_pointee.name in ("c_void", "()", "u8"):
                    return
                else:
                    self._report.warnings.append(
                        f"void* mapped to {rust_type.name}: consider using *mut c_void for type safety"
                    )
            else:
                self._report.add_incompatibility(Incompatibility(
                    kind=IncompatibilityKind.OPAQUE_TYPE_ISSUE,
                    c_type="void*", rust_type=rust_type.name,
                    description="C void* should map to *mut c_void or *const c_void in Rust",
                    severity="error",
                    suggestion="Use std::ffi::c_void",
                ))

    # --- general type comparison ---
    def _compare_types(self, c_type: TypeDescriptor,
                       rust_type: TypeDescriptor, context: str = "") -> None:
        self._report.types_checked += 1

        if c_type.kind == TypeKind.TYPEDEF and c_type.typedef_target:
            self._compare_types(c_type.typedef_target, rust_type, context)
            return

        if c_type.kind == TypeKind.OPAQUE or rust_type.kind == TypeKind.OPAQUE:
            self._check_opaque_type(c_type, rust_type)
            return

        if c_type.kind == TypeKind.POINTER and (c_type.name == "void*" or
                (c_type.pointee_type and c_type.pointee_type.name == "void")):
            self._check_opaque_type(c_type, rust_type)
            return

        if c_type.kind == TypeKind.PRIMITIVE and rust_type.kind == TypeKind.PRIMITIVE:
            expected_rust = self._type_map.get(c_type.name)
            if expected_rust and expected_rust != rust_type.name:
                c_size = self._c_prim_sizes.get(c_type.name, (0, 0))[0]
                r_size = RUST_PRIMITIVE_SIZES.get(rust_type.name, (0, 0))[0]
                if c_size != r_size:
                    self._report.add_incompatibility(Incompatibility(
                        kind=IncompatibilityKind.SIZE_MISMATCH,
                        c_type=f"{c_type.name} ({c_size} bytes)",
                        rust_type=f"{rust_type.name} ({r_size} bytes)",
                        description=f"Type size mismatch in {context}",
                        severity="error",
                        suggestion=f"Use `{expected_rust}` instead of `{rust_type.name}`",
                    ))
                else:
                    self._report.warnings.append(
                        f"Type name mismatch in {context}: C `{c_type.name}` -> Rust `{rust_type.name}` (expected `{expected_rust}`) but sizes match"
                    )
            return

        if c_type.kind == TypeKind.STRUCT and rust_type.kind == TypeKind.STRUCT:
            self._compare_struct_layouts(c_type, rust_type)
            return

        if c_type.kind == TypeKind.ENUM and rust_type.kind == TypeKind.ENUM:
            self._compare_enums(c_type, rust_type)
            return

        if c_type.kind == TypeKind.FUNCTION_POINTER and rust_type.kind == TypeKind.FUNCTION_POINTER:
            self._compare_function_pointers(c_type, rust_type, context)
            return

        if c_type.kind == TypeKind.POINTER and rust_type.kind == TypeKind.POINTER:
            c_ptee = c_type.pointee_type
            r_ptee = rust_type.pointee_type
            if c_ptee and r_ptee:
                self._compare_types(c_ptee, r_ptee, f"{context} pointee")
            return

        if c_type.kind == TypeKind.ARRAY and rust_type.kind == TypeKind.ARRAY:
            if c_type.element_count != rust_type.element_count:
                self._report.add_incompatibility(Incompatibility(
                    kind=IncompatibilityKind.SIZE_MISMATCH,
                    c_type=c_type.name, rust_type=rust_type.name,
                    description=f"Array length mismatch: C={c_type.element_count}, Rust={rust_type.element_count}",
                    severity="error",
                ))
            if c_type.element_type and rust_type.element_type:
                self._compare_types(c_type.element_type, rust_type.element_type,
                                    f"{context} element")
            return

        if c_type.kind == TypeKind.VOID:
            if rust_type.name in ("()", "c_void", "std::ffi::c_void"):
                return

        if c_type.kind != rust_type.kind:
            self._report.add_incompatibility(Incompatibility(
                kind=IncompatibilityKind.FIELD_TYPE_MISMATCH,
                c_type=f"{c_type.name} ({c_type.kind.name})",
                rust_type=f"{rust_type.name} ({rust_type.kind.name})",
                description=f"Type kind mismatch in {context}",
                severity="error",
            ))

    # --- platform differences ---
    def _check_platform_differences(self, c_types: List[TypeDescriptor],
                                    rust_types: List[TypeDescriptor]) -> None:
        sensitive_c_types = {"long", "unsigned long", "size_t", "ssize_t",
                             "ptrdiff_t", "time_t", "off_t"}

        for ct in c_types:
            if ct.kind == TypeKind.STRUCT:
                for fld in ct.fields:
                    if fld.type_desc and fld.type_desc.name in sensitive_c_types:
                        self._report.warnings.append(
                            f"Field `{fld.name}` in `{ct.name}` uses platform-sensitive type `{fld.type_desc.name}`"
                        )
            elif ct.kind == TypeKind.PRIMITIVE and ct.name in sensitive_c_types:
                self._report.warnings.append(
                    f"Type `{ct.name}` has different sizes on LP64 vs LLP64 platforms"
                )

        if self.platform.name == Platform.LLP64:
            for ct in c_types:
                if ct.name in ("long", "unsigned long"):
                    self._report.warnings.append(
                        f"On LLP64 (Windows), `{ct.name}` is 4 bytes, not 8. Ensure Rust uses i32/u32."
                    )

    # --- function comparison ---
    def _compare_functions(self, c_func: Dict[str, Any],
                           rust_func: Dict[str, Any]) -> None:
        self._report.functions_checked += 1
        func_name = c_func.get("name", "?")

        c_cc = CallingConvention(c_func.get("calling_convention", "cdecl"))
        r_cc = CallingConvention(rust_func.get("calling_convention", "cdecl"))
        self._check_calling_convention(c_cc, r_cc, func_name)

        c_params = c_func.get("params", [])
        r_params = rust_func.get("params", [])
        if len(c_params) != len(r_params):
            self._report.add_incompatibility(Incompatibility(
                kind=IncompatibilityKind.FUNCTION_SIGNATURE_MISMATCH,
                c_type=func_name, rust_type=func_name,
                description=f"Parameter count mismatch: C has {len(c_params)}, Rust has {len(r_params)}",
                severity="error",
            ))
        else:
            for i, (cp, rp) in enumerate(zip(c_params, r_params)):
                c_td = cp if isinstance(cp, TypeDescriptor) else TypeDescriptor(
                    name=cp.get("type", "int"), kind=TypeKind.PRIMITIVE)
                r_td = rp if isinstance(rp, TypeDescriptor) else TypeDescriptor(
                    name=rp.get("type", "i32"), kind=TypeKind.PRIMITIVE)
                self._compare_types(c_td, r_td, f"{func_name} param {i}")

        c_ret = c_func.get("return_type")
        r_ret = rust_func.get("return_type")
        if c_ret and r_ret:
            c_td = c_ret if isinstance(c_ret, TypeDescriptor) else TypeDescriptor(
                name=c_ret.get("type", "void") if isinstance(c_ret, dict) else str(c_ret),
                kind=TypeKind.PRIMITIVE)
            r_td = r_ret if isinstance(r_ret, TypeDescriptor) else TypeDescriptor(
                name=r_ret.get("type", "()") if isinstance(r_ret, dict) else str(r_ret),
                kind=TypeKind.PRIMITIVE)
            self._compare_types(c_td, r_td, f"{func_name} return")

    # --- main entry ---
    def check(self, c_types: List[TypeDescriptor],
              rust_types: List[TypeDescriptor],
              c_functions: Optional[List[Dict[str, Any]]] = None,
              rust_functions: Optional[List[Dict[str, Any]]] = None) -> ABIReport:
        self._reset()

        type_pairs = list(zip(c_types, rust_types))
        for c_td, r_td in type_pairs:
            self._compare_types(c_td, r_td)

        self._check_platform_differences(c_types, rust_types)

        if c_functions and rust_functions:
            c_by_name = {f.get("name", ""): f for f in c_functions}
            r_by_name = {f.get("name", ""): f for f in rust_functions}
            for name in c_by_name:
                if name in r_by_name:
                    self._compare_functions(c_by_name[name], r_by_name[name])
                else:
                    self._report.warnings.append(
                        f"C function `{name}` has no Rust binding")

        return self._report

    def check_struct_pair(self, c_struct: TypeDescriptor,
                          rust_struct: TypeDescriptor) -> ABIReport:
        self._reset()
        self._compare_struct_layouts(c_struct, rust_struct)
        return self._report

    def check_enum_pair(self, c_enum: TypeDescriptor,
                        rust_enum: TypeDescriptor) -> ABIReport:
        self._reset()
        self._compare_enums(c_enum, rust_enum)
        return self._report

    def check_function_pair(self, c_func: Dict[str, Any],
                            rust_func: Dict[str, Any]) -> ABIReport:
        self._reset()
        self._compare_functions(c_func, rust_func)
        return self._report

    def suggest_rust_type(self, c_type_name: str) -> str:
        return self._type_map.get(c_type_name, f"/* unknown C type: {c_type_name} */")

    def generate_rust_binding(self, c_func: Dict[str, Any]) -> str:
        name = c_func.get("name", "unknown")
        params = c_func.get("params", [])
        ret = c_func.get("return_type", "void")

        param_strs = []
        for p in params:
            pname = p.get("name", "_") if isinstance(p, dict) else "_"
            ptype = p.get("type", "int") if isinstance(p, dict) else str(p)
            rtype = self.suggest_rust_type(ptype)
            param_strs.append(f"{pname}: {rtype}")

        ret_type = "" if ret == "void" else f" -> {self.suggest_rust_type(ret if isinstance(ret, str) else ret.get('type', 'void'))}"
        params_str = ", ".join(param_strs)
        return f'extern "C" {{ fn {name}({params_str}){ret_type}; }}'
