"""Comprehensive C↔Rust type mapping.

Maps C types to idiomatic Rust equivalents, detects type mismatches between
C and Rust code, and suggests safe Rust wrappers for common C patterns
including void*, function pointers, unions, bitfields, flexible array
members, enums, and strings.
"""

import re
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Set
from enum import Enum


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

class Safety(Enum):
    SAFE = "safe"
    UNSAFE = "unsafe"
    CONDITIONAL = "conditional"


class Confidence(Enum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


@dataclass
class RustType:
    c_type: str
    rust_type: str
    safety: Safety = Safety.SAFE
    requires_lifetime: bool = False
    notes: str = ""
    alternative: Optional[str] = None

    def display(self) -> str:
        s = f"{self.c_type} -> {self.rust_type}"
        if self.alternative:
            s += f" (alt: {self.alternative})"
        if self.notes:
            s += f" [{self.notes}]"
        return s


@dataclass
class TypeMismatch:
    location: str
    c_type: str
    rust_type: str
    expected_rust_type: str
    severity: str  # "error", "warning", "info"
    description: str
    fix: str = ""

    @property
    def id(self) -> str:
        return f"{self.location}:{self.c_type}->{self.rust_type}"


@dataclass
class SafeWrapper:
    c_pattern: str
    c_snippet: str
    wrapper_name: str
    wrapper_code: str
    safety_guarantee: str
    usage_example: str

    def display(self) -> str:
        return f"// {self.safety_guarantee}\n{self.wrapper_code}"


# ---------------------------------------------------------------------------
# Type mapping tables
# ---------------------------------------------------------------------------

# Primitive C -> Rust mappings
PRIMITIVE_MAP: Dict[str, RustType] = {
    "void": RustType("void", "()", notes="unit type"),
    "char": RustType("char", "i8", alternative="u8", notes="C char signedness is platform-dependent"),
    "signed char": RustType("signed char", "i8"),
    "unsigned char": RustType("unsigned char", "u8"),
    "short": RustType("short", "i16"),
    "unsigned short": RustType("unsigned short", "u16"),
    "int": RustType("int", "i32"),
    "unsigned int": RustType("unsigned int", "u32"),
    "long": RustType("long", "i64", alternative="i32", notes="platform-dependent: i32 on Windows, i64 on LP64"),
    "unsigned long": RustType("unsigned long", "u64", alternative="u32", notes="platform-dependent"),
    "long long": RustType("long long", "i64"),
    "unsigned long long": RustType("unsigned long long", "u64"),
    "float": RustType("float", "f32"),
    "double": RustType("double", "f64"),
    "long double": RustType("long double", "f64", notes="Rust has no 80-bit float; use f64 or external crate"),
    "_Bool": RustType("_Bool", "bool"),
    "bool": RustType("bool", "bool"),
    "size_t": RustType("size_t", "usize"),
    "ssize_t": RustType("ssize_t", "isize"),
    "ptrdiff_t": RustType("ptrdiff_t", "isize"),
    "intptr_t": RustType("intptr_t", "isize"),
    "uintptr_t": RustType("uintptr_t", "usize"),
    "int8_t": RustType("int8_t", "i8"),
    "int16_t": RustType("int16_t", "i16"),
    "int32_t": RustType("int32_t", "i32"),
    "int64_t": RustType("int64_t", "i64"),
    "uint8_t": RustType("uint8_t", "u8"),
    "uint16_t": RustType("uint16_t", "u16"),
    "uint32_t": RustType("uint32_t", "u32"),
    "uint64_t": RustType("uint64_t", "u64"),
}

# Pointer type mappings
POINTER_MAP: Dict[str, RustType] = {
    "void*": RustType("void*", "*mut std::ffi::c_void", safety=Safety.UNSAFE,
                       alternative="Box<dyn Any>", notes="opaque pointer"),
    "const void*": RustType("const void*", "*const std::ffi::c_void", safety=Safety.UNSAFE),
    "char*": RustType("char*", "*mut c_char", safety=Safety.UNSAFE,
                       alternative="CString", notes="owned C string"),
    "const char*": RustType("const char*", "*const c_char", safety=Safety.UNSAFE,
                             alternative="&CStr", notes="borrowed C string"),
    "FILE*": RustType("FILE*", "*mut libc::FILE", safety=Safety.UNSAFE,
                       alternative="std::fs::File", notes="use std::io for safe I/O"),
}

# Complex type patterns
_FUNC_PTR_RE = re.compile(
    r"(\w[\w\s\*]*)\s*\(\s*\*\s*(\w+)\s*\)\s*\(([^)]*)\)"
)
_ARRAY_RE = re.compile(r"(\w[\w\s\*]*)\s+(\w+)\s*\[(\d*)\]")
_BITFIELD_RE = re.compile(r"(\w+)\s+(\w+)\s*:\s*(\d+)\s*;")
_UNION_RE = re.compile(r"union\s+(\w+)\s*\{([^}]+)\}", re.DOTALL)
_ENUM_RE = re.compile(r"enum\s+(\w+)\s*\{([^}]+)\}", re.DOTALL)
_STRUCT_RE = re.compile(r"struct\s+(\w+)\s*\{([^}]+)\}", re.DOTALL)
_TYPEDEF_RE = re.compile(r"typedef\s+(.+?)\s+(\w+)\s*;")
_FLEXIBLE_ARRAY_RE = re.compile(r"(\w+)\s+(\w+)\s*\[\s*\]\s*;")

_RUST_STRUCT_RE = re.compile(r"(?:#\[repr\(C\)\]\s*)?struct\s+(\w+)\s*\{([^}]+)\}", re.DOTALL)
_RUST_TYPE_RE = re.compile(r"(\w+)\s*:\s*([^,}]+)")


# ---------------------------------------------------------------------------
# Core mapping functions
# ---------------------------------------------------------------------------

def _normalize_c_type(t: str) -> str:
    """Normalize C type string for lookup."""
    t = t.strip()
    t = re.sub(r"\s+", " ", t)
    t = re.sub(r"\s*\*\s*", "*", t)
    # Remove leading const for lookup, handle separately
    return t


def _map_single_type(c_type: str) -> RustType:
    """Map a single C type to its Rust equivalent."""
    norm = _normalize_c_type(c_type)

    # Direct primitive lookup
    if norm in PRIMITIVE_MAP:
        return PRIMITIVE_MAP[norm]

    # Pointer lookup
    if norm in POINTER_MAP:
        return POINTER_MAP[norm]

    # Const pointer
    if norm.startswith("const ") and norm.endswith("*"):
        base = norm[6:].rstrip("*").strip()
        if base in PRIMITIVE_MAP:
            rust_base = PRIMITIVE_MAP[base].rust_type
            return RustType(norm, f"*const {rust_base}", safety=Safety.UNSAFE,
                           alternative=f"&{rust_base}")
        return RustType(norm, f"*const {base}", safety=Safety.UNSAFE)

    # Mutable pointer to known type
    if norm.endswith("*"):
        base = norm.rstrip("*").strip()
        if base in PRIMITIVE_MAP:
            rust_base = PRIMITIVE_MAP[base].rust_type
            return RustType(norm, f"*mut {rust_base}", safety=Safety.UNSAFE,
                           alternative=f"&mut {rust_base}")
        return RustType(norm, f"*mut {base}", safety=Safety.UNSAFE)

    # Double pointer
    if norm.endswith("**"):
        base = norm.rstrip("*").strip()
        return RustType(norm, f"*mut *mut {base}", safety=Safety.UNSAFE,
                       notes="double pointer — consider Box<Box<T>> or &mut &mut T")

    # Struct pointer
    if norm.startswith("struct "):
        struct_name = norm.replace("struct ", "").rstrip("*").strip()
        if "*" in norm:
            return RustType(norm, f"*mut {struct_name}", safety=Safety.UNSAFE,
                           alternative=f"&mut {struct_name}")
        return RustType(norm, struct_name)

    # Enum type
    if norm.startswith("enum "):
        enum_name = norm.replace("enum ", "").strip()
        return RustType(norm, enum_name, notes="ensure Rust enum has same discriminant values")

    # Union type
    if norm.startswith("union "):
        union_name = norm.replace("union ", "").strip()
        return RustType(norm, union_name, safety=Safety.UNSAFE,
                       notes="use #[repr(C)] union or an enum with variants")

    # Fallback
    return RustType(norm, norm, notes="unmapped type — manual review needed")


def _map_function_pointer(ret_type: str, params: str) -> RustType:
    """Map C function pointer to Rust fn pointer or closure."""
    param_list = [p.strip() for p in params.split(",") if p.strip()]
    rust_params = []
    for p in param_list:
        # Extract type (last token before name is the type, or the whole thing)
        parts = p.rsplit(None, 1)
        if len(parts) >= 1:
            ptype = parts[0] if len(parts) > 1 else p
            mapped = _map_single_type(ptype)
            rust_params.append(mapped.rust_type)
    rust_ret = _map_single_type(ret_type).rust_type
    params_str = ", ".join(rust_params) if rust_params else ""
    if rust_ret == "()" or ret_type.strip() == "void":
        rust_fn = f"extern \"C\" fn({params_str})"
        safe_alt = f"Box<dyn Fn({params_str})>"
    else:
        rust_fn = f"extern \"C\" fn({params_str}) -> {rust_ret}"
        safe_alt = f"Box<dyn Fn({params_str}) -> {rust_ret}>"
    c_repr = f"{ret_type}(*)({params})"
    return RustType(c_repr, rust_fn, safety=Safety.UNSAFE,
                   alternative=safe_alt, notes="function pointer")


def _map_array(base_type: str, size: str) -> RustType:
    """Map C array to Rust array or slice."""
    mapped_base = _map_single_type(base_type)
    if size:
        rust_type = f"[{mapped_base.rust_type}; {size}]"
        c_repr = f"{base_type}[{size}]"
    else:
        # Flexible array member
        rust_type = f"[{mapped_base.rust_type}]"
        c_repr = f"{base_type}[]"
    return RustType(c_repr, rust_type,
                   alternative=f"Vec<{mapped_base.rust_type}>",
                   notes="flexible array member" if not size else "fixed-size array")


def _map_bitfield_struct(fields: List[Tuple[str, str, int]]) -> str:
    """Generate Rust bitfield representation using bitflags or manual packing."""
    lines = ["// Bitfield struct — use bitflags crate or manual packing"]
    lines.append("// Consider using the `bitfield` or `modular-bitfield` crate")
    lines.append("")

    total_bits = sum(bits for _, _, bits in fields)
    backing_type = "u8" if total_bits <= 8 else "u16" if total_bits <= 16 else "u32" if total_bits <= 32 else "u64"

    lines.append(f"#[repr(transparent)]")
    lines.append(f"pub struct Bitfields({backing_type});")
    lines.append("")
    lines.append("impl Bitfields {")

    offset = 0
    for ftype, fname, bits in fields:
        mask = (1 << bits) - 1
        lines.append(f"    /// `{ftype} {fname} : {bits}` — bits [{offset}:{offset + bits})")
        lines.append(f"    pub fn {fname}(&self) -> {backing_type} {{")
        lines.append(f"        (self.0 >> {offset}) & 0x{mask:X}")
        lines.append(f"    }}")
        lines.append(f"    pub fn set_{fname}(&mut self, val: {backing_type}) {{")
        lines.append(f"        self.0 = (self.0 & !(0x{mask:X} << {offset})) | ((val & 0x{mask:X}) << {offset});")
        lines.append(f"    }}")
        offset += bits

    lines.append("}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def map_types(c_types: List[str]) -> List[RustType]:
    """Map a list of C type strings to their Rust equivalents."""
    results: List[RustType] = []
    for ct in c_types:
        ct = ct.strip()
        # Function pointer
        fp = _FUNC_PTR_RE.match(ct)
        if fp:
            results.append(_map_function_pointer(fp.group(1), fp.group(3)))
            continue
        # Array
        arr = _ARRAY_RE.match(ct)
        if arr:
            results.append(_map_array(arr.group(1), arr.group(3)))
            continue
        results.append(_map_single_type(ct))
    return results


def detect_type_mismatches(c_code: str, rust_code: str) -> List[TypeMismatch]:
    """Detect type mismatches between paired C and Rust struct/function definitions."""
    mismatches: List[TypeMismatch] = []

    # Compare struct fields
    c_structs: Dict[str, List[Tuple[str, str]]] = {}
    for m in _STRUCT_RE.finditer(c_code):
        name = m.group(1)
        body = m.group(2)
        fields: List[Tuple[str, str]] = []
        for line in body.strip().splitlines():
            line = line.strip().rstrip(";").strip()
            if not line:
                continue
            # "type name" pattern
            parts = line.rsplit(None, 1)
            if len(parts) == 2:
                fields.append((parts[0].strip(), parts[1].strip()))
        c_structs[name] = fields

    rust_structs: Dict[str, List[Tuple[str, str]]] = {}
    for m in _RUST_STRUCT_RE.finditer(rust_code):
        name = m.group(1)
        body = m.group(2)
        fields = []
        for fm in _RUST_TYPE_RE.finditer(body):
            fields.append((fm.group(1).strip(), fm.group(2).strip().rstrip(",")))
        rust_structs[name] = fields

    for struct_name, c_fields in c_structs.items():
        if struct_name not in rust_structs:
            mismatches.append(TypeMismatch(
                location=f"struct {struct_name}",
                c_type=f"struct {struct_name}",
                rust_type="(missing)",
                expected_rust_type=f"struct {struct_name}",
                severity="error",
                description=f"C struct {struct_name} has no Rust equivalent",
                fix=f"Create #[repr(C)] struct {struct_name} {{ ... }}",
            ))
            continue

        r_fields = rust_structs[struct_name]
        r_field_map = {n: t for n, t in r_fields}

        for c_ftype, c_fname in c_fields:
            if c_fname not in r_field_map:
                mismatches.append(TypeMismatch(
                    location=f"struct {struct_name}::{c_fname}",
                    c_type=c_ftype,
                    rust_type="(missing)",
                    expected_rust_type=_map_single_type(c_ftype).rust_type,
                    severity="error",
                    description=f"Field {c_fname} missing in Rust struct {struct_name}",
                    fix=f"Add `{c_fname}: {_map_single_type(c_ftype).rust_type}` to struct",
                ))
                continue

            expected = _map_single_type(c_ftype)
            actual_rust = r_field_map[c_fname]
            # Normalize for comparison
            exp_norm = expected.rust_type.replace(" ", "")
            act_norm = actual_rust.replace(" ", "")
            if exp_norm != act_norm:
                # Check alternative
                alt_norm = (expected.alternative or "").replace(" ", "")
                if alt_norm and alt_norm == act_norm:
                    mismatches.append(TypeMismatch(
                        location=f"struct {struct_name}::{c_fname}",
                        c_type=c_ftype,
                        rust_type=actual_rust,
                        expected_rust_type=expected.rust_type,
                        severity="info",
                        description=f"Using alternative mapping {actual_rust} instead of {expected.rust_type}",
                    ))
                else:
                    mismatches.append(TypeMismatch(
                        location=f"struct {struct_name}::{c_fname}",
                        c_type=c_ftype,
                        rust_type=actual_rust,
                        expected_rust_type=expected.rust_type,
                        severity="warning",
                        description=f"Type mismatch: C {c_ftype} mapped to {actual_rust}, expected {expected.rust_type}",
                        fix=f"Change to {expected.rust_type}" + (f" or {expected.alternative}" if expected.alternative else ""),
                    ))

    # Check enum discriminant mismatches
    c_enums: Dict[str, List[Tuple[str, Optional[int]]]] = {}
    for m in _ENUM_RE.finditer(c_code):
        name = m.group(1)
        body = m.group(2)
        variants: List[Tuple[str, Optional[int]]] = []
        val = 0
        for line in body.strip().splitlines():
            line = line.strip().rstrip(",").strip()
            if not line:
                continue
            if "=" in line:
                vname, vval = line.split("=", 1)
                val = int(vval.strip(), 0) if vval.strip() else val
                variants.append((vname.strip(), val))
            else:
                variants.append((line, val))
            val += 1
        c_enums[name] = variants

    for enum_name, c_variants in c_enums.items():
        # Check if Rust has matching enum
        rust_enum_match = re.search(
            rf"enum\s+{re.escape(enum_name)}\s*\{{([^}}]+)\}}", rust_code, re.DOTALL
        )
        if not rust_enum_match:
            mismatches.append(TypeMismatch(
                location=f"enum {enum_name}",
                c_type=f"enum {enum_name}",
                rust_type="(missing)",
                expected_rust_type=f"enum {enum_name}",
                severity="error",
                description=f"C enum {enum_name} has no Rust equivalent",
                fix=f"Create #[repr(C)] enum {enum_name} {{ ... }}",
            ))
            continue

        rust_body = rust_enum_match.group(1)
        rust_variants = set()
        for line in rust_body.strip().splitlines():
            line = line.strip().rstrip(",").strip()
            if line:
                vname = line.split("=")[0].strip().split("(")[0].strip()
                rust_variants.add(vname)

        for c_vname, _ in c_variants:
            if c_vname not in rust_variants:
                mismatches.append(TypeMismatch(
                    location=f"enum {enum_name}::{c_vname}",
                    c_type=f"enum variant {c_vname}",
                    rust_type="(missing)",
                    expected_rust_type=c_vname,
                    severity="warning",
                    description=f"Enum variant {c_vname} missing in Rust enum {enum_name}",
                    fix=f"Add {c_vname} variant to enum {enum_name}",
                ))

    return mismatches


def suggest_safe_wrappers(c_code: str) -> List[SafeWrapper]:
    """Suggest safe Rust wrappers for common unsafe C patterns."""
    wrappers: List[SafeWrapper] = []

    # void* pattern -> generic wrapper
    if re.search(r"\bvoid\s*\*", c_code):
        wrappers.append(SafeWrapper(
            c_pattern="void*",
            c_snippet="void* data",
            wrapper_name="TypedPointer",
            wrapper_code=(
                "use std::marker::PhantomData;\n\n"
                "/// Safe wrapper around an opaque void* pointer.\n"
                "pub struct TypedPointer<T> {\n"
                "    ptr: *mut T,\n"
                "    _marker: PhantomData<T>,\n"
                "}\n\n"
                "impl<T> TypedPointer<T> {\n"
                "    pub fn new(val: T) -> Self {\n"
                "        let boxed = Box::new(val);\n"
                "        Self {\n"
                "            ptr: Box::into_raw(boxed),\n"
                "            _marker: PhantomData,\n"
                "        }\n"
                "    }\n\n"
                "    pub fn as_ref(&self) -> &T {\n"
                "        unsafe { &*self.ptr }\n"
                "    }\n\n"
                "    pub fn as_mut(&mut self) -> &mut T {\n"
                "        unsafe { &mut *self.ptr }\n"
                "    }\n"
                "}\n\n"
                "impl<T> Drop for TypedPointer<T> {\n"
                "    fn drop(&mut self) {\n"
                "        unsafe { drop(Box::from_raw(self.ptr)); }\n"
                "    }\n"
                "}\n"
            ),
            safety_guarantee="Type-safe wrapper: prevents type confusion from void* casts",
            usage_example="let ptr = TypedPointer::new(42u32);\nassert_eq!(*ptr.as_ref(), 42);",
        ))

    # Function pointer pattern
    if _FUNC_PTR_RE.search(c_code):
        wrappers.append(SafeWrapper(
            c_pattern="function pointer",
            c_snippet="void (*callback)(void* ctx, int result)",
            wrapper_name="Callback",
            wrapper_code=(
                "/// Type-safe callback wrapper replacing C function pointers.\n"
                "pub struct Callback<F: Fn(i32)> {\n"
                "    func: F,\n"
                "}\n\n"
                "impl<F: Fn(i32)> Callback<F> {\n"
                "    pub fn new(func: F) -> Self {\n"
                "        Self { func }\n"
                "    }\n\n"
                "    pub fn invoke(&self, result: i32) {\n"
                "        (self.func)(result);\n"
                "    }\n"
                "}\n"
            ),
            safety_guarantee="Closure-based: no raw function pointer or void* context needed",
            usage_example='let cb = Callback::new(|r| println!("result: {r}"));\ncb.invoke(42);',
        ))

    # Union pattern
    for m in _UNION_RE.finditer(c_code):
        union_name = m.group(1)
        body = m.group(2)
        fields: List[Tuple[str, str]] = []
        for line in body.strip().splitlines():
            line = line.strip().rstrip(";").strip()
            if line:
                parts = line.rsplit(None, 1)
                if len(parts) == 2:
                    fields.append((parts[0], parts[1]))

        variants = []
        for ftype, fname in fields:
            rust_t = _map_single_type(ftype).rust_type
            variants.append(f"    {fname.capitalize()}({rust_t}),")
        variants_str = "\n".join(variants)

        wrappers.append(SafeWrapper(
            c_pattern=f"union {union_name}",
            c_snippet=f"union {union_name} {{ ... }}",
            wrapper_name=f"{union_name}Safe",
            wrapper_code=(
                f"/// Safe enum replacement for C union {union_name}.\n"
                f"pub enum {union_name}Safe {{\n"
                f"{variants_str}\n"
                f"}}\n"
            ),
            safety_guarantee=f"Tagged union: runtime type safety for union {union_name}",
            usage_example=f"let val = {union_name}Safe::{fields[0][1].capitalize()}(0);" if fields else "",
        ))

    # Bitfield pattern
    bitfields: List[Tuple[str, str, int]] = []
    for m in _BITFIELD_RE.finditer(c_code):
        bitfields.append((m.group(1), m.group(2), int(m.group(3))))
    if bitfields:
        wrapper_code = _map_bitfield_struct(bitfields)
        wrappers.append(SafeWrapper(
            c_pattern="bitfield struct",
            c_snippet="struct { unsigned flag : 1; ... }",
            wrapper_name="Bitfields",
            wrapper_code=wrapper_code,
            safety_guarantee="Packed bitfield: safe accessor methods with masking",
            usage_example="let mut bf = Bitfields(0);\nbf.set_flag(1);\nassert_eq!(bf.flag(), 1);",
        ))

    # Flexible array member
    for m in _FLEXIBLE_ARRAY_RE.finditer(c_code):
        ftype = m.group(1)
        fname = m.group(2)
        rust_elem = _map_single_type(ftype).rust_type
        wrappers.append(SafeWrapper(
            c_pattern="flexible array member",
            c_snippet=f"{ftype} {fname}[];",
            wrapper_name=f"FlexArray{fname.capitalize()}",
            wrapper_code=(
                f"/// Safe wrapper for flexible array member `{fname}`.\n"
                f"pub struct FlexArray{fname.capitalize()} {{\n"
                f"    header: Header,  // fixed-size portion\n"
                f"    data: Vec<{rust_elem}>,\n"
                f"}}\n\n"
                f"impl FlexArray{fname.capitalize()} {{\n"
                f"    pub fn new(header: Header, data: Vec<{rust_elem}>) -> Self {{\n"
                f"        Self {{ header, data }}\n"
                f"    }}\n\n"
                f"    pub fn get(&self, index: usize) -> Option<&{rust_elem}> {{\n"
                f"        self.data.get(index)\n"
                f"    }}\n\n"
                f"    pub fn len(&self) -> usize {{\n"
                f"        self.data.len()\n"
                f"    }}\n"
                f"}}\n"
            ),
            safety_guarantee="Bounds-checked: Vec replaces C flexible array member with checked access",
            usage_example=f"let arr = FlexArray{fname.capitalize()}::new(hdr, vec![1, 2, 3]);\nassert_eq!(arr.len(), 3);",
        ))

    # C string patterns
    if re.search(r"\bchar\s*\*", c_code) or re.search(r"\bstrcpy|strcat|strlen\b", c_code):
        wrappers.append(SafeWrapper(
            c_pattern="C string (char*)",
            c_snippet="char* str / const char* str",
            wrapper_name="SafeString",
            wrapper_code=(
                "use std::ffi::{CStr, CString};\n\n"
                "/// Safe C string interop wrapper.\n"
                "pub struct SafeString {\n"
                "    inner: CString,\n"
                "}\n\n"
                "impl SafeString {\n"
                "    /// Create from a Rust &str (must not contain null bytes).\n"
                "    pub fn from_str(s: &str) -> Result<Self, std::ffi::NulError> {\n"
                "        Ok(Self { inner: CString::new(s)? })\n"
                "    }\n\n"
                "    /// Create from a raw C string pointer (copies data).\n"
                "    pub unsafe fn from_ptr(ptr: *const i8) -> Option<Self> {\n"
                "        if ptr.is_null() {\n"
                "            return None;\n"
                "        }\n"
                "        let cstr = CStr::from_ptr(ptr);\n"
                "        Some(Self { inner: cstr.to_owned() })\n"
                "    }\n\n"
                "    /// Get as Rust &str.\n"
                "    pub fn as_str(&self) -> &str {\n"
                "        self.inner.to_str().unwrap_or(\"\")\n"
                "    }\n\n"
                "    /// Get as raw C pointer for FFI calls.\n"
                "    pub fn as_ptr(&self) -> *const i8 {\n"
                "        self.inner.as_ptr()\n"
                "    }\n\n"
                "    pub fn len(&self) -> usize {\n"
                "        self.inner.as_bytes().len()\n"
                "    }\n"
                "}\n\n"
                "impl std::fmt::Display for SafeString {\n"
                "    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {\n"
                "        write!(f, \"{}\", self.as_str())\n"
                "    }\n"
                "}\n"
            ),
            safety_guarantee="Null-safe, encoding-aware string interop between C and Rust",
            usage_example=(
                'let s = SafeString::from_str("hello").unwrap();\n'
                'assert_eq!(s.as_str(), "hello");\n'
                "// Pass s.as_ptr() to C functions"
            ),
        ))

    # Array access with bounds checking
    if _ARRAY_RE.search(c_code):
        wrappers.append(SafeWrapper(
            c_pattern="C array access",
            c_snippet="arr[i]",
            wrapper_name="BoundedArray",
            wrapper_code=(
                "/// Bounds-checked fixed-size array wrapper.\n"
                "pub struct BoundedArray<T, const N: usize> {\n"
                "    data: [T; N],\n"
                "}\n\n"
                "impl<T: Default + Copy, const N: usize> BoundedArray<T, N> {\n"
                "    pub fn new() -> Self {\n"
                "        Self { data: [T::default(); N] }\n"
                "    }\n\n"
                "    pub fn get(&self, index: usize) -> Option<&T> {\n"
                "        self.data.get(index)\n"
                "    }\n\n"
                "    pub fn set(&mut self, index: usize, val: T) -> bool {\n"
                "        if index < N {\n"
                "            self.data[index] = val;\n"
                "            true\n"
                "        } else {\n"
                "            false\n"
                "        }\n"
                "    }\n\n"
                "    pub fn as_slice(&self) -> &[T] {\n"
                "        &self.data\n"
                "    }\n"
                "}\n"
            ),
            safety_guarantee="Bounds-checked: prevents buffer overflow from unchecked array indexing",
            usage_example="let mut arr = BoundedArray::<i32, 10>::new();\narr.set(0, 42);\nassert_eq!(arr.get(0), Some(&42));",
        ))

    return wrappers
