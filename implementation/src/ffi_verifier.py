"""Verify FFI boundaries between Rust and C.

Checks that Rust FFI bindings match C headers, verifies ABI compatibility
of struct layouts, and detects safety issues in FFI code including
repr(C), #[no_mangle], extern "C", and callback safety.
"""

import re
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Set
from enum import Enum


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

class FFIStatus(Enum):
    OK = "ok"
    MISMATCH = "mismatch"
    MISSING = "missing"
    UNSAFE = "unsafe"
    WARNING = "warning"


@dataclass
class FFIBinding:
    c_name: str
    c_signature: str
    rust_name: Optional[str]
    rust_signature: Optional[str]
    status: FFIStatus
    issues: List[str] = field(default_factory=list)


@dataclass
class FFIResult:
    bindings: List[FFIBinding] = field(default_factory=list)
    total_c_functions: int = 0
    matched: int = 0
    mismatched: int = 0
    missing: int = 0

    @property
    def match_rate(self) -> float:
        return self.matched / self.total_c_functions if self.total_c_functions > 0 else 0.0

    def summary(self) -> str:
        lines = [
            f"FFI Verification: {self.match_rate:.0%} match rate",
            f"  Total C functions: {self.total_c_functions}",
            f"  Matched: {self.matched}",
            f"  Mismatched: {self.mismatched}",
            f"  Missing: {self.missing}",
        ]
        for b in self.bindings:
            if b.issues:
                lines.append(f"  {b.c_name}: {', '.join(b.issues)}")
        return "\n".join(lines)


@dataclass
class ABIField:
    name: str
    type_name: str
    size: int
    alignment: int
    offset: int


@dataclass
class ABIResult:
    compatible: bool
    c_struct: str
    rust_struct: str
    c_size: int = 0
    rust_size: int = 0
    c_alignment: int = 0
    rust_alignment: int = 0
    field_issues: List[str] = field(default_factory=list)
    c_fields: List[ABIField] = field(default_factory=list)
    rust_fields: List[ABIField] = field(default_factory=list)

    def summary(self) -> str:
        status = "✅ Compatible" if self.compatible else "❌ Incompatible"
        lines = [
            f"ABI Check: {self.c_struct} <-> {self.rust_struct}: {status}",
            f"  C size: {self.c_size}, alignment: {self.c_alignment}",
            f"  Rust size: {self.rust_size}, alignment: {self.rust_alignment}",
        ]
        for issue in self.field_issues:
            lines.append(f"  ⚠ {issue}")
        return "\n".join(lines)


class FFISafetyCategory(Enum):
    MISSING_REPR_C = "missing_repr_c"
    MISSING_NO_MANGLE = "missing_no_mangle"
    MISSING_EXTERN_C = "missing_extern_c"
    CALLBACK_SAFETY = "callback_safety"
    NULLABLE_POINTER = "nullable_pointer"
    LIFETIME_ACROSS_FFI = "lifetime_across_ffi"
    PANIC_ACROSS_FFI = "panic_across_ffi"
    THREAD_SAFETY = "thread_safety"
    STRING_ENCODING = "string_encoding"
    ALIGNMENT = "alignment"


@dataclass
class FFISafetyIssue:
    category: FFISafetyCategory
    location: str
    description: str
    severity: str  # "error", "warning", "info"
    fix: str

    @property
    def id(self) -> str:
        return f"{self.category.value}:{self.location}"


# ---------------------------------------------------------------------------
# Type size/alignment tables (LP64 model)
# ---------------------------------------------------------------------------

_C_TYPE_INFO: Dict[str, Tuple[int, int]] = {
    "char": (1, 1), "signed char": (1, 1), "unsigned char": (1, 1),
    "short": (2, 2), "unsigned short": (2, 2),
    "int": (4, 4), "unsigned int": (4, 4), "unsigned": (4, 4),
    "long": (8, 8), "unsigned long": (8, 8),
    "long long": (8, 8), "unsigned long long": (8, 8),
    "float": (4, 4), "double": (8, 8),
    "int8_t": (1, 1), "uint8_t": (1, 1),
    "int16_t": (2, 2), "uint16_t": (2, 2),
    "int32_t": (4, 4), "uint32_t": (4, 4),
    "int64_t": (8, 8), "uint64_t": (8, 8),
    "size_t": (8, 8), "ssize_t": (8, 8),
    "bool": (1, 1), "_Bool": (1, 1),
}

_RUST_TYPE_INFO: Dict[str, Tuple[int, int]] = {
    "i8": (1, 1), "u8": (1, 1),
    "i16": (2, 2), "u16": (2, 2),
    "i32": (4, 4), "u32": (4, 4),
    "i64": (8, 8), "u64": (8, 8),
    "i128": (16, 16), "u128": (16, 16),
    "f32": (4, 4), "f64": (8, 8),
    "isize": (8, 8), "usize": (8, 8),
    "bool": (1, 1),
}

# Any pointer type
_POINTER_SIZE = 8
_POINTER_ALIGN = 8

# ---------------------------------------------------------------------------
# Parsing helpers
# ---------------------------------------------------------------------------

_C_FUNC_DECL_RE = re.compile(
    r"(?:extern\s+)?(?:(?:unsigned|signed|long|short|const|volatile|struct|enum|union)\s+)*"
    r"(\w+)\s*\*?\s+(\w+)\s*\(([^)]*)\)\s*;",
    re.MULTILINE,
)

_RUST_EXTERN_FN_RE = re.compile(
    r'extern\s+"C"\s*\{([^}]+)\}',
    re.DOTALL,
)

_RUST_FN_DECL_RE = re.compile(
    r"(?:pub\s+)?fn\s+(\w+)\s*\(([^)]*)\)\s*(?:->\s*([^;{]+))?\s*;",
    re.MULTILINE,
)

_RUST_EXPORT_FN_RE = re.compile(
    r'#\[no_mangle\]\s*(?:pub\s+)?(?:unsafe\s+)?extern\s+"C"\s+fn\s+(\w+)\s*\(([^)]*)\)\s*(?:->\s*([^\{]+))?\s*\{',
    re.MULTILINE,
)

_C_STRUCT_RE = re.compile(
    r"(?:typedef\s+)?struct\s+(\w+)\s*\{([^}]+)\}",
    re.DOTALL,
)

_RUST_REPR_C_STRUCT_RE = re.compile(
    r"#\[repr\(C\)\]\s*(?:pub\s+)?struct\s+(\w+)\s*\{([^}]+)\}",
    re.DOTALL,
)


def _parse_c_header_functions(header: str) -> Dict[str, Tuple[str, str]]:
    """Parse C header: name -> (return_type, params_str)."""
    funcs: Dict[str, Tuple[str, str]] = {}
    for m in _C_FUNC_DECL_RE.finditer(header):
        ret_type = m.group(1)
        name = m.group(2)
        params = m.group(3).strip()
        funcs[name] = (ret_type, params)
    return funcs


def _parse_rust_extern_functions(rust_code: str) -> Dict[str, Tuple[str, str]]:
    """Parse Rust extern "C" block declarations: name -> (return_type, params_str)."""
    funcs: Dict[str, Tuple[str, str]] = {}
    for block_match in _RUST_EXTERN_FN_RE.finditer(rust_code):
        block = block_match.group(1)
        for m in _RUST_FN_DECL_RE.finditer(block):
            name = m.group(1)
            params = m.group(2).strip()
            ret = (m.group(3) or "()").strip()
            funcs[name] = (ret, params)
    # Also find #[no_mangle] exported functions
    for m in _RUST_EXPORT_FN_RE.finditer(rust_code):
        name = m.group(1)
        params = m.group(2).strip()
        ret = (m.group(3) or "()").strip()
        funcs[name] = (ret, params)
    return funcs


def _get_type_info(type_name: str, table: Dict[str, Tuple[int, int]]) -> Tuple[int, int]:
    """Get (size, alignment) for a type, defaulting to pointer size for pointer types."""
    t = type_name.strip()
    if "*" in t or t.startswith("*"):
        return (_POINTER_SIZE, _POINTER_ALIGN)
    t_clean = re.sub(r"\s+", " ", t).strip()
    if t_clean in table:
        return table[t_clean]
    # Remove qualifiers
    for qual in ("const ", "volatile ", "mut ", "pub "):
        t_clean = t_clean.replace(qual, "")
    t_clean = t_clean.strip()
    if t_clean in table:
        return table[t_clean]
    return (_POINTER_SIZE, _POINTER_ALIGN)  # assume pointer-sized for unknown types


def _compute_struct_layout(
    fields: List[Tuple[str, str]], type_table: Dict[str, Tuple[int, int]]
) -> Tuple[int, int, List[ABIField]]:
    """Compute C-compatible struct layout with padding. Returns (total_size, alignment, field_list)."""
    abi_fields: List[ABIField] = []
    offset = 0
    max_align = 1

    for fname, ftype in fields:
        size, align = _get_type_info(ftype, type_table)
        max_align = max(max_align, align)
        # Pad to alignment
        if offset % align != 0:
            offset += align - (offset % align)
        abi_fields.append(ABIField(
            name=fname, type_name=ftype, size=size, alignment=align, offset=offset
        ))
        offset += size

    # Final padding to struct alignment
    if offset % max_align != 0:
        offset += max_align - (offset % max_align)

    return (offset, max_align, abi_fields)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def verify_ffi_bindings(rust_ffi_code: str, c_header: str) -> FFIResult:
    """Check that Rust FFI declarations match C header function signatures."""
    c_funcs = _parse_c_header_functions(c_header)
    rust_funcs = _parse_rust_extern_functions(rust_ffi_code)
    result = FFIResult(total_c_functions=len(c_funcs))

    # Simple C -> Rust type mapping for parameter comparison
    type_map = {
        "void": "()", "char": "c_char", "int": "c_int", "unsigned": "c_uint",
        "long": "c_long", "float": "f32", "double": "f64",
        "size_t": "usize", "ssize_t": "isize",
        "int8_t": "i8", "int16_t": "i16", "int32_t": "i32", "int64_t": "i64",
        "uint8_t": "u8", "uint16_t": "u16", "uint32_t": "u32", "uint64_t": "u64",
        "bool": "bool", "_Bool": "bool",
    }

    for c_name, (c_ret, c_params) in c_funcs.items():
        if c_name in rust_funcs:
            r_ret, r_params = rust_funcs[c_name]
            issues: List[str] = []

            # Check return type
            expected_ret = type_map.get(c_ret.strip(), c_ret.strip())
            r_ret_clean = r_ret.strip().rstrip(",").strip()
            if expected_ret != r_ret_clean and r_ret_clean != expected_ret:
                # Allow some equivalences
                if not (c_ret.strip() == "void" and r_ret_clean in ("()", "")):
                    issues.append(f"Return type: C '{c_ret}' vs Rust '{r_ret_clean}'")

            # Check parameter count
            c_param_list = [p.strip() for p in c_params.split(",") if p.strip() and p.strip() != "void"]
            r_param_list = [p.strip() for p in r_params.split(",") if p.strip()]
            if len(c_param_list) != len(r_param_list):
                issues.append(f"Parameter count: C has {len(c_param_list)}, Rust has {len(r_param_list)}")

            status = FFIStatus.OK if not issues else FFIStatus.MISMATCH
            result.bindings.append(FFIBinding(
                c_name=c_name, c_signature=f"{c_ret} {c_name}({c_params})",
                rust_name=c_name, rust_signature=f"fn {c_name}({r_params}) -> {r_ret}",
                status=status, issues=issues,
            ))
            if issues:
                result.mismatched += 1
            else:
                result.matched += 1
        else:
            result.bindings.append(FFIBinding(
                c_name=c_name, c_signature=f"{c_ret} {c_name}({c_params})",
                rust_name=None, rust_signature=None,
                status=FFIStatus.MISSING,
                issues=[f"No Rust binding found for C function {c_name}"],
            ))
            result.missing += 1

    return result


def check_abi_compatibility(c_struct: str, rust_repr: str) -> ABIResult:
    """Check ABI compatibility between a C struct definition and Rust #[repr(C)] struct."""
    # Parse C struct
    c_match = _C_STRUCT_RE.search(c_struct)
    if not c_match:
        return ABIResult(compatible=False, c_struct="(unparseable)", rust_struct="",
                        field_issues=["Could not parse C struct definition"])

    c_name = c_match.group(1)
    c_body = c_match.group(2)
    c_fields: List[Tuple[str, str]] = []
    for line in c_body.strip().splitlines():
        line = line.strip().rstrip(";").strip()
        if not line or line.startswith("//") or line.startswith("/*"):
            continue
        parts = line.rsplit(None, 1)
        if len(parts) == 2:
            ftype, fname = parts
            fname = fname.rstrip("[]").strip()
            c_fields.append((fname, ftype.strip()))

    # Parse Rust struct
    r_match = _RUST_REPR_C_STRUCT_RE.search(rust_repr)
    has_repr_c = bool(r_match)
    if not r_match:
        # Try without repr(C)
        r_match2 = re.search(r"(?:pub\s+)?struct\s+(\w+)\s*\{([^}]+)\}", rust_repr, re.DOTALL)
        if not r_match2:
            return ABIResult(compatible=False, c_struct=c_name, rust_struct="(unparseable)",
                            field_issues=["Could not parse Rust struct definition"])
        r_name = r_match2.group(1)
        r_body = r_match2.group(2)
    else:
        r_name = r_match.group(1)
        r_body = r_match.group(2)

    r_fields: List[Tuple[str, str]] = []
    for fm in re.finditer(r"(?:pub\s+)?(\w+)\s*:\s*([^,}]+)", r_body):
        r_fields.append((fm.group(1).strip(), fm.group(2).strip().rstrip(",")))

    # Compute layouts
    c_size, c_align, c_abi = _compute_struct_layout(c_fields, _C_TYPE_INFO)
    r_size, r_align, r_abi = _compute_struct_layout(r_fields, _RUST_TYPE_INFO)

    issues: List[str] = []

    if not has_repr_c:
        issues.append(f"Rust struct {r_name} missing #[repr(C)] — layout is not guaranteed to match C")

    if c_size != r_size:
        issues.append(f"Size mismatch: C={c_size} bytes, Rust={r_size} bytes")

    if c_align != r_align:
        issues.append(f"Alignment mismatch: C={c_align}, Rust={r_align}")

    if len(c_fields) != len(r_fields):
        issues.append(f"Field count mismatch: C has {len(c_fields)}, Rust has {len(r_fields)}")

    # Compare field by field
    for i in range(min(len(c_abi), len(r_abi))):
        cf = c_abi[i]
        rf = r_abi[i]
        if cf.offset != rf.offset:
            issues.append(f"Field '{cf.name}'/'{rf.name}' offset mismatch: C={cf.offset}, Rust={rf.offset}")
        if cf.size != rf.size:
            issues.append(f"Field '{cf.name}'/'{rf.name}' size mismatch: C={cf.size}, Rust={rf.size}")

    compatible = len(issues) == 0

    return ABIResult(
        compatible=compatible,
        c_struct=c_name, rust_struct=r_name if r_match or r_match2 else "(unknown)",
        c_size=c_size, rust_size=r_size,
        c_alignment=c_align, rust_alignment=r_align,
        field_issues=issues,
        c_fields=c_abi, rust_fields=r_abi,
    )


def detect_ffi_safety_issues(rust_code: str) -> List[FFISafetyIssue]:
    """Detect safety issues in Rust FFI code."""
    issues: List[FFISafetyIssue] = []

    # Check for structs used in FFI without repr(C)
    all_structs = set(re.findall(r"(?:pub\s+)?struct\s+(\w+)", rust_code))
    repr_c_structs = set(re.findall(r"#\[repr\(C\)\]\s*(?:pub\s+)?struct\s+(\w+)", rust_code))
    # Find structs referenced in extern blocks
    extern_blocks = re.findall(r'extern\s+"C"\s*\{([^}]+)\}', rust_code, re.DOTALL)
    extern_types: Set[str] = set()
    for block in extern_blocks:
        extern_types.update(re.findall(r"\b([A-Z]\w+)\b", block))
    ffi_structs = all_structs & extern_types
    missing_repr = ffi_structs - repr_c_structs
    for s in missing_repr:
        issues.append(FFISafetyIssue(
            category=FFISafetyCategory.MISSING_REPR_C,
            location=f"struct {s}",
            description=f"Struct {s} used in FFI but missing #[repr(C)]",
            severity="error",
            fix=f"Add #[repr(C)] attribute to struct {s}",
        ))

    # Check for exported functions without #[no_mangle]
    pub_extern_fns = re.findall(
        r'(pub\s+(?:unsafe\s+)?extern\s+"C"\s+fn\s+(\w+))',
        rust_code,
    )
    no_mangle_fns = set(re.findall(
        r'#\[no_mangle\]\s*pub\s+(?:unsafe\s+)?extern\s+"C"\s+fn\s+(\w+)',
        rust_code,
    ))
    for _, fn_name in pub_extern_fns:
        if fn_name not in no_mangle_fns:
            issues.append(FFISafetyIssue(
                category=FFISafetyCategory.MISSING_NO_MANGLE,
                location=f"fn {fn_name}",
                description=f"Exported FFI function {fn_name} missing #[no_mangle]",
                severity="error",
                fix=f'Add #[no_mangle] before pub extern "C" fn {fn_name}',
            ))

    # Check for panic-possible code in extern "C" functions
    for m in _RUST_EXPORT_FN_RE.finditer(rust_code):
        fn_name = m.group(1)
        # Find function body
        start = m.end() - 1
        brace_count = 0
        pos = start
        while pos < len(rust_code):
            if rust_code[pos] == "{":
                brace_count += 1
            elif rust_code[pos] == "}":
                brace_count -= 1
                if brace_count == 0:
                    break
            pos += 1
        body = rust_code[start:pos + 1]

        panic_patterns = [".unwrap()", ".expect(", "panic!(", "unreachable!(", "todo!(", "unimplemented!("]
        for pat in panic_patterns:
            if pat in body:
                issues.append(FFISafetyIssue(
                    category=FFISafetyCategory.PANIC_ACROSS_FFI,
                    location=f"fn {fn_name}",
                    description=f"FFI function {fn_name} may panic via {pat} — unwinding across FFI is UB",
                    severity="error",
                    fix=f"Use std::panic::catch_unwind or replace {pat} with error return",
                ))
                break

    # Check for &str / String parameters in extern "C" functions (should use *const c_char)
    for m in _RUST_EXPORT_FN_RE.finditer(rust_code):
        fn_name = m.group(1)
        params = m.group(2)
        if "&str" in params or "String" in params:
            issues.append(FFISafetyIssue(
                category=FFISafetyCategory.STRING_ENCODING,
                location=f"fn {fn_name}",
                description=f"FFI function {fn_name} uses Rust string types in signature",
                severity="error",
                fix="Use *const c_char or *mut c_char for C-compatible string parameters",
            ))

    # Check for references with lifetimes crossing FFI
    for m in _RUST_EXPORT_FN_RE.finditer(rust_code):
        fn_name = m.group(1)
        params = m.group(2)
        ret = m.group(3) or ""
        if re.search(r"&'?\w*\s+\w+", params) and "&" in ret:
            issues.append(FFISafetyIssue(
                category=FFISafetyCategory.LIFETIME_ACROSS_FFI,
                location=f"fn {fn_name}",
                description=f"FFI function {fn_name} returns a reference — C cannot enforce Rust lifetimes",
                severity="warning",
                fix="Return raw pointer or owned value instead of reference",
            ))

    # Check for function pointers without Option wrapping (nullable in C)
    extern_fn_ptrs = re.findall(
        r'extern\s+"C"\s+fn\s*\([^)]*\)',
        rust_code,
    )
    for fp in extern_fn_ptrs:
        # Check if it's wrapped in Option
        fp_loc = rust_code.find(fp)
        if fp_loc > 0:
            before = rust_code[max(0, fp_loc - 20):fp_loc]
            if "Option<" not in before:
                issues.append(FFISafetyIssue(
                    category=FFISafetyCategory.CALLBACK_SAFETY,
                    location="function pointer",
                    description=f"Function pointer '{fp[:60]}...' not wrapped in Option — NULL callbacks are UB",
                    severity="warning",
                    fix="Wrap in Option<extern \"C\" fn(...)> to handle NULL callbacks safely",
                ))

    # Check for mutable static used across threads
    if re.search(r"static\s+mut\s+\w+", rust_code):
        issues.append(FFISafetyIssue(
            category=FFISafetyCategory.THREAD_SAFETY,
            location="static mut",
            description="Mutable static found — accessing from multiple threads is UB",
            severity="warning",
            fix="Use std::sync::Mutex, AtomicXxx, or thread_local! instead of static mut",
        ))

    return issues
