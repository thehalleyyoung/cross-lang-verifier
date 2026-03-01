"""
API Surface Analyzer for C-to-Rust migration.

Parses C headers and Rust source files, checks ABI compatibility,
generates FFI bindings, and produces migration guides.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Type-mapping tables
# ---------------------------------------------------------------------------

C_TO_RUST_TYPE_MAP: Dict[str, str] = {
    "void": "()",
    "int": "c_int",
    "unsigned int": "c_uint",
    "unsigned": "c_uint",
    "long": "c_long",
    "unsigned long": "c_ulong",
    "long long": "c_longlong",
    "unsigned long long": "c_ulonglong",
    "short": "c_short",
    "unsigned short": "c_ushort",
    "char": "c_char",
    "unsigned char": "c_uchar",
    "signed char": "c_schar",
    "float": "c_float",
    "double": "c_double",
    "size_t": "usize",
    "ssize_t": "isize",
    "int8_t": "i8",
    "int16_t": "i16",
    "int32_t": "i32",
    "int64_t": "i64",
    "uint8_t": "u8",
    "uint16_t": "u16",
    "uint32_t": "u32",
    "uint64_t": "u64",
    "bool": "bool",
    "_Bool": "bool",
    "ptrdiff_t": "isize",
    "intptr_t": "isize",
    "uintptr_t": "usize",
}

RUST_TO_C_TYPE_MAP: Dict[str, str] = {v: k for k, v in C_TO_RUST_TYPE_MAP.items()}
RUST_TO_C_TYPE_MAP.update({
    "i8": "int8_t",
    "i16": "int16_t",
    "i32": "int32_t",
    "i64": "int64_t",
    "u8": "uint8_t",
    "u16": "uint16_t",
    "u32": "uint32_t",
    "u64": "uint64_t",
    "usize": "size_t",
    "isize": "ssize_t",
    "f32": "float",
    "f64": "double",
    "bool": "bool",
    "()": "void",
    "c_int": "int",
    "c_uint": "unsigned int",
    "c_char": "char",
    "c_uchar": "unsigned char",
    "c_long": "long",
    "c_ulong": "unsigned long",
    "c_float": "float",
    "c_double": "double",
})

TYPE_SIZES: Dict[str, int] = {
    "char": 1, "c_char": 1, "i8": 1, "u8": 1, "bool": 1,
    "short": 2, "c_short": 2, "i16": 2, "u16": 2,
    "int": 4, "c_int": 4, "i32": 4, "u32": 4, "float": 4, "c_float": 4,
    "long": 8, "c_long": 8, "i64": 8, "u64": 8, "double": 8, "c_double": 8,
    "size_t": 8, "usize": 8, "isize": 8, "ssize_t": 8,
}


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class CompatLevel(Enum):
    COMPATIBLE = auto()
    BREAKING = auto()
    DEPRECATED = auto()
    ADDED = auto()
    REMOVED = auto()


# ---------------------------------------------------------------------------
# C data classes
# ---------------------------------------------------------------------------

@dataclass
class CFunction:
    name: str
    return_type: str
    parameters: List[Tuple[str, str]]  # (type, name)
    is_variadic: bool = False
    is_static: bool = False
    is_inline: bool = False
    doc_comment: str = ""

    @property
    def signature(self) -> str:
        params = ", ".join(f"{t} {n}" for t, n in self.parameters)
        if self.is_variadic:
            params += ", ..." if params else "..."
        return f"{self.return_type} {self.name}({params})"


@dataclass
class CTypedef:
    name: str
    underlying_type: str
    is_function_pointer: bool = False
    doc_comment: str = ""


@dataclass
class CStruct:
    name: str
    fields: List[Tuple[str, str]]  # (type, name)
    is_opaque: bool = False
    doc_comment: str = ""

    @property
    def estimated_size(self) -> int:
        total = 0
        for ftype, _ in self.fields:
            base = ftype.replace("*", "").strip()
            if "*" in ftype:
                total += 8
            else:
                total += TYPE_SIZES.get(base, 4)
        return total


@dataclass
class CEnum:
    name: str
    variants: List[Tuple[str, Optional[int]]]  # (name, value_or_None)
    doc_comment: str = ""


@dataclass
class CAPISpec:
    functions: List[CFunction] = field(default_factory=list)
    typedefs: List[CTypedef] = field(default_factory=list)
    structs: List[CStruct] = field(default_factory=list)
    enums: List[CEnum] = field(default_factory=list)
    macros: List[Tuple[str, str]] = field(default_factory=list)  # (name, value)

    @property
    def public_symbol_count(self) -> int:
        return (len(self.functions) + len(self.typedefs)
                + len(self.structs) + len(self.enums) + len(self.macros))


# ---------------------------------------------------------------------------
# Rust data classes
# ---------------------------------------------------------------------------

@dataclass
class RustFunction:
    name: str
    return_type: str
    parameters: List[Tuple[str, str]]  # (name, type)
    is_unsafe: bool = False
    is_extern_c: bool = False
    is_pub: bool = True
    doc_comment: str = ""
    generic_params: List[str] = field(default_factory=list)

    @property
    def signature(self) -> str:
        params = ", ".join(f"{n}: {t}" for n, t in self.parameters)
        sig = f"fn {self.name}({params})"
        if self.return_type and self.return_type != "()":
            sig += f" -> {self.return_type}"
        return sig


@dataclass
class RustStruct:
    name: str
    fields: List[Tuple[str, str]]  # (name, type)
    is_pub: bool = True
    derives: List[str] = field(default_factory=list)
    repr: Optional[str] = None
    doc_comment: str = ""


@dataclass
class RustEnum:
    name: str
    variants: List[Tuple[str, Optional[str]]]  # (name, associated_data_or_None)
    is_pub: bool = True
    derives: List[str] = field(default_factory=list)
    repr: Optional[str] = None
    doc_comment: str = ""


@dataclass
class RustAPISpec:
    functions: List[RustFunction] = field(default_factory=list)
    structs: List[RustStruct] = field(default_factory=list)
    enums: List[RustEnum] = field(default_factory=list)
    traits: List[str] = field(default_factory=list)
    type_aliases: List[Tuple[str, str]] = field(default_factory=list)

    @property
    def public_symbol_count(self) -> int:
        return (len(self.functions) + len(self.structs)
                + len(self.enums) + len(self.traits) + len(self.type_aliases))


# ---------------------------------------------------------------------------
# Compatibility / change tracking
# ---------------------------------------------------------------------------

@dataclass
class CompatIssue:
    symbol: str
    level: CompatLevel
    message: str
    suggestion: str = ""

    def __str__(self) -> str:
        tag = self.level.name
        base = f"[{tag}] {self.symbol}: {self.message}"
        if self.suggestion:
            base += f" — suggestion: {self.suggestion}"
        return base


@dataclass
class CompatResult:
    issues: List[CompatIssue] = field(default_factory=list)
    matched_functions: int = 0
    unmatched_c_functions: List[str] = field(default_factory=list)
    unmatched_rust_functions: List[str] = field(default_factory=list)

    @property
    def is_fully_compatible(self) -> bool:
        return all(i.level == CompatLevel.COMPATIBLE for i in self.issues)

    @property
    def breaking_count(self) -> int:
        return sum(1 for i in self.issues if i.level == CompatLevel.BREAKING)

    def summary(self) -> str:
        lines = [f"Matched functions: {self.matched_functions}"]
        lines.append(f"Unmatched C functions: {len(self.unmatched_c_functions)}")
        lines.append(f"Unmatched Rust functions: {len(self.unmatched_rust_functions)}")
        lines.append(f"Total issues: {len(self.issues)}")
        lines.append(f"Breaking: {self.breaking_count}")
        return "\n".join(lines)


@dataclass
class BindingFunction:
    c_name: str
    rust_name: str
    return_type: str
    parameters: List[Tuple[str, str]]
    is_safe_wrapper: bool = False


@dataclass
class APIChange:
    symbol: str
    change_type: CompatLevel
    old_signature: str
    new_signature: str
    description: str


# ---------------------------------------------------------------------------
# Helper: strip C comments and preprocessor lines
# ---------------------------------------------------------------------------

def _strip_c_comments(text: str) -> str:
    text = re.sub(r"//[^\n]*", "", text)
    text = re.sub(r"/\*.*?\*/", "", text, flags=re.DOTALL)
    return text


def _collapse_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def _map_c_type_to_rust(ctype: str) -> str:
    ctype = ctype.strip()
    ptr_depth = ctype.count("*")
    base = ctype.replace("*", "").strip()
    # const qualifier
    is_const = "const" in base
    base = base.replace("const", "").strip()
    rust_base = C_TO_RUST_TYPE_MAP.get(base, base)
    for _ in range(ptr_depth):
        if is_const:
            rust_base = f"*const {rust_base}"
        else:
            rust_base = f"*mut {rust_base}"
    return rust_base


def _map_rust_type_to_c(rtype: str) -> str:
    rtype = rtype.strip()
    if rtype.startswith("*const "):
        inner = _map_rust_type_to_c(rtype[7:])
        return f"const {inner}*"
    if rtype.startswith("*mut "):
        inner = _map_rust_type_to_c(rtype[5:])
        return f"{inner}*"
    return RUST_TO_C_TYPE_MAP.get(rtype, rtype)


# ---------------------------------------------------------------------------
# extract_c_api
# ---------------------------------------------------------------------------

def extract_c_api(header_content: str) -> CAPISpec:
    """Parse public API from C header file content."""
    spec = CAPISpec()
    cleaned = _strip_c_comments(header_content)

    # --- macros (#define NAME value) ---
    for m in re.finditer(
        r"^[ \t]*#define\s+([A-Z_][A-Z0-9_]*)\s+(.+?)$", cleaned, re.MULTILINE
    ):
        name, value = m.group(1), m.group(2).strip()
        if name.startswith("_") or name.endswith("_H"):
            continue
        spec.macros.append((name, value))

    # --- typedefs ---
    # function-pointer typedefs: typedef ret (*name)(params);
    for m in re.finditer(
        r"typedef\s+(.+?)\(\s*\*\s*(\w+)\s*\)\s*\(([^)]*)\)\s*;", cleaned
    ):
        ret_type = _collapse_whitespace(m.group(1))
        name = m.group(2)
        spec.typedefs.append(CTypedef(name=name, underlying_type=ret_type,
                                       is_function_pointer=True))

    # simple typedefs: typedef <type> <name>;
    for m in re.finditer(
        r"typedef\s+((?:(?:struct|enum|unsigned|signed|const|long)\s+)*\w[\w\s\*]*?)"
        r"\s+(\w+)\s*;", cleaned
    ):
        underlying = _collapse_whitespace(m.group(1))
        name = m.group(2)
        already = any(td.name == name for td in spec.typedefs)
        if not already:
            spec.typedefs.append(CTypedef(name=name, underlying_type=underlying))

    # --- enums ---
    for m in re.finditer(
        r"(?:typedef\s+)?enum\s+(\w+)?\s*\{([^}]*)\}\s*(\w+)?\s*;", cleaned
    ):
        name = m.group(1) or m.group(3) or "anonymous_enum"
        body = m.group(2)
        variants: List[Tuple[str, Optional[int]]] = []
        for entry in body.split(","):
            entry = entry.strip()
            if not entry:
                continue
            if "=" in entry:
                vname, vval = entry.split("=", 1)
                vname = vname.strip()
                vval_str = vval.strip()
                try:
                    variants.append((vname, int(vval_str, 0)))
                except ValueError:
                    variants.append((vname, None))
            else:
                variants.append((entry, None))
        spec.enums.append(CEnum(name=name, variants=variants))

    # --- structs ---
    for m in re.finditer(
        r"(?:typedef\s+)?struct\s+(\w+)?\s*\{([^}]*)\}\s*(\w+)?\s*;", cleaned
    ):
        name = m.group(1) or m.group(3) or "anonymous_struct"
        body = m.group(2)
        fields: List[Tuple[str, str]] = []
        for line in body.split(";"):
            line = line.strip()
            if not line:
                continue
            parts = line.rsplit(None, 1)
            if len(parts) == 2:
                ftype = parts[0].strip()
                fname = parts[1].strip().rstrip(";")
                # handle arrays
                arr = re.match(r"(\w+)\[(\d+)\]", fname)
                if arr:
                    fname = arr.group(1)
                    ftype = f"{ftype}[{arr.group(2)}]"
                fields.append((ftype, fname))
        spec.structs.append(CStruct(name=name, fields=fields))

    # opaque struct declarations (struct Foo;)
    for m in re.finditer(r"(?:^|;)\s*struct\s+(\w+)\s*;", cleaned):
        name = m.group(1)
        already = any(s.name == name for s in spec.structs)
        if not already:
            spec.structs.append(CStruct(name=name, fields=[], is_opaque=True))

    # --- functions ---
    # Matches: [static] [inline] return_type name(params);
    func_pat = re.compile(
        r"(?:^|(?<=;))\s*"
        r"((?:(?:static|inline|extern|const|unsigned|signed|long|struct|enum)\s+)*"
        r"[\w\*]+(?:\s*\*)*)"
        r"\s+(\w+)\s*"
        r"\(([^)]*)\)\s*;",
        re.MULTILINE,
    )
    for m in func_pat.finditer(cleaned):
        ret_raw = _collapse_whitespace(m.group(1))
        name = m.group(2)
        params_raw = m.group(3).strip()

        is_static = "static" in ret_raw
        is_inline = "inline" in ret_raw
        ret_type = ret_raw.replace("static", "").replace("inline", "")
        ret_type = ret_type.replace("extern", "").strip()
        ret_type = _collapse_whitespace(ret_type)

        is_variadic = params_raw.endswith("...")
        params_raw_clean = params_raw.replace("...", "").strip().rstrip(",")

        parameters: List[Tuple[str, str]] = []
        if params_raw_clean and params_raw_clean != "void":
            for p in params_raw_clean.split(","):
                p = p.strip()
                if not p:
                    continue
                parts = p.rsplit(None, 1)
                if len(parts) == 2:
                    ptype = parts[0].strip()
                    pname = parts[1].strip()
                    if pname.startswith("*"):
                        ptype += " *"
                        pname = pname.lstrip("*")
                    parameters.append((ptype, pname))
                else:
                    parameters.append((p, ""))

        if is_static:
            continue

        spec.functions.append(
            CFunction(
                name=name,
                return_type=ret_type,
                parameters=parameters,
                is_variadic=is_variadic,
                is_static=is_static,
                is_inline=is_inline,
            )
        )

    return spec


# ---------------------------------------------------------------------------
# extract_rust_api
# ---------------------------------------------------------------------------

def extract_rust_api(rust_content: str) -> RustAPISpec:
    """Parse public API from Rust source content."""
    spec = RustAPISpec()

    # strip line comments (but keep doc comments for later)
    content_no_comments = re.sub(r"(?<!/)//(?!/)[^\n]*", "", rust_content)

    # --- pub fn / pub unsafe fn / pub extern "C" fn ---
    fn_pat = re.compile(
        r'pub\s+(unsafe\s+)?(extern\s+"C"\s+)?fn\s+(\w+)'
        r"(?:<([^>]*)>)?"
        r"\s*\(([^)]*)\)"
        r"(?:\s*->\s*([\w\*&\[\]:,<> ]+))?"
    )
    for m in fn_pat.finditer(content_no_comments):
        is_unsafe = m.group(1) is not None
        is_extern_c = m.group(2) is not None
        name = m.group(3)
        generics_raw = m.group(4)
        params_raw = m.group(5).strip()
        ret_type = (m.group(6) or "()").strip()

        generic_params: List[str] = []
        if generics_raw:
            generic_params = [g.strip() for g in generics_raw.split(",") if g.strip()]

        parameters: List[Tuple[str, str]] = []
        if params_raw:
            for p in params_raw.split(","):
                p = p.strip()
                if not p or p in ("self", "&self", "&mut self"):
                    continue
                if ":" in p:
                    pname, ptype = p.split(":", 1)
                    parameters.append((pname.strip(), ptype.strip()))
                else:
                    parameters.append(("_", p))

        spec.functions.append(
            RustFunction(
                name=name,
                return_type=ret_type,
                parameters=parameters,
                is_unsafe=is_unsafe,
                is_extern_c=is_extern_c,
                generic_params=generic_params,
            )
        )

    # --- pub struct ---
    struct_pat = re.compile(
        r"(?:#\[derive\(([^)]*)\)\]\s*)?"
        r'(?:#\[repr\((\w+)\)\]\s*)?'
        r"pub\s+struct\s+(\w+)\s*\{([^}]*)\}"
    )
    for m in struct_pat.finditer(content_no_comments):
        derives_raw = m.group(1) or ""
        repr_attr = m.group(2)
        name = m.group(3)
        body = m.group(4)

        derives = [d.strip() for d in derives_raw.split(",") if d.strip()]
        fields: List[Tuple[str, str]] = []
        for line in body.split(","):
            line = line.strip()
            if not line:
                continue
            line = re.sub(r"pub\s+", "", line)
            if ":" in line:
                fname, ftype = line.split(":", 1)
                fields.append((fname.strip(), ftype.strip()))

        spec.structs.append(
            RustStruct(name=name, fields=fields, derives=derives, repr=repr_attr)
        )

    # --- pub enum ---
    enum_pat = re.compile(
        r"(?:#\[derive\(([^)]*)\)\]\s*)?"
        r'(?:#\[repr\((\w+)\)\]\s*)?'
        r"pub\s+enum\s+(\w+)\s*\{([^}]*)\}"
    )
    for m in enum_pat.finditer(content_no_comments):
        derives_raw = m.group(1) or ""
        repr_attr = m.group(2)
        name = m.group(3)
        body = m.group(4)

        derives = [d.strip() for d in derives_raw.split(",") if d.strip()]
        variants: List[Tuple[str, Optional[str]]] = []
        for line in body.split(","):
            line = line.strip()
            if not line:
                continue
            if "(" in line:
                vname = line.split("(")[0].strip()
                vdata = line[line.index("("):]
                variants.append((vname, vdata))
            elif "{" in line:
                vname = line.split("{")[0].strip()
                variants.append((vname, "{...}"))
            elif "=" in line:
                vname, vval = line.split("=", 1)
                variants.append((vname.strip(), vval.strip()))
            else:
                variants.append((line, None))

        spec.enums.append(
            RustEnum(name=name, variants=variants, derives=derives, repr=repr_attr)
        )

    # --- pub trait ---
    for m in re.finditer(r"pub\s+trait\s+(\w+)", content_no_comments):
        spec.traits.append(m.group(1))

    # --- pub type aliases ---
    for m in re.finditer(
        r"pub\s+type\s+(\w+)\s*=\s*([^;]+);", content_no_comments
    ):
        spec.type_aliases.append((m.group(1), m.group(2).strip()))

    return spec


# ---------------------------------------------------------------------------
# api_compatibility_check
# ---------------------------------------------------------------------------

def api_compatibility_check(c_api: CAPISpec, rust_api: RustAPISpec) -> CompatResult:
    """Check ABI compatibility between C and Rust APIs."""
    result = CompatResult()

    c_fn_map: Dict[str, CFunction] = {f.name: f for f in c_api.functions}
    rust_fn_map: Dict[str, RustFunction] = {f.name: f for f in rust_api.functions}

    all_c_names = set(c_fn_map.keys())
    all_rust_names = set(rust_fn_map.keys())

    matched = all_c_names & all_rust_names
    result.matched_functions = len(matched)
    result.unmatched_c_functions = sorted(all_c_names - all_rust_names)
    result.unmatched_rust_functions = sorted(all_rust_names - all_c_names)

    for name in sorted(result.unmatched_c_functions):
        result.issues.append(CompatIssue(
            symbol=name, level=CompatLevel.REMOVED,
            message="C function has no corresponding Rust function",
            suggestion=f"Add a Rust extern \"C\" fn {name} or a safe wrapper",
        ))

    for name in sorted(result.unmatched_rust_functions):
        result.issues.append(CompatIssue(
            symbol=name, level=CompatLevel.ADDED,
            message="Rust function has no corresponding C function",
        ))

    for name in sorted(matched):
        cfn = c_fn_map[name]
        rfn = rust_fn_map[name]

        # Check calling convention
        if not rfn.is_extern_c:
            result.issues.append(CompatIssue(
                symbol=name, level=CompatLevel.BREAKING,
                message="Rust function is not extern \"C\" — ABI mismatch",
                suggestion='Add extern "C" to the Rust function declaration',
            ))

        # Check return type compatibility
        expected_rust_ret = _map_c_type_to_rust(cfn.return_type)
        if rfn.return_type != expected_rust_ret and rfn.return_type != cfn.return_type:
            c_size = TYPE_SIZES.get(cfn.return_type.replace("*", "").strip(), -1)
            r_size = TYPE_SIZES.get(rfn.return_type, -2)
            if c_size != r_size:
                result.issues.append(CompatIssue(
                    symbol=name, level=CompatLevel.BREAKING,
                    message=(f"Return type mismatch: C `{cfn.return_type}` "
                             f"vs Rust `{rfn.return_type}`"),
                    suggestion=f"Use `{expected_rust_ret}` as the Rust return type",
                ))
            else:
                result.issues.append(CompatIssue(
                    symbol=name, level=CompatLevel.COMPATIBLE,
                    message=(f"Return types differ in name but match in size: "
                             f"C `{cfn.return_type}` vs Rust `{rfn.return_type}`"),
                ))

        # Check parameter count
        c_param_count = len(cfn.parameters)
        r_param_count = len(rfn.parameters)
        if c_param_count != r_param_count:
            result.issues.append(CompatIssue(
                symbol=name, level=CompatLevel.BREAKING,
                message=(f"Parameter count mismatch: C has {c_param_count}, "
                         f"Rust has {r_param_count}"),
            ))
        else:
            # Check each parameter type
            for idx, ((ctype, cname), (rname, rtype)) in enumerate(
                zip(cfn.parameters, rfn.parameters)
            ):
                expected = _map_c_type_to_rust(ctype)
                if rtype != expected and rtype != ctype:
                    result.issues.append(CompatIssue(
                        symbol=f"{name}::param[{idx}]",
                        level=CompatLevel.BREAKING,
                        message=(f"Parameter type mismatch: C `{ctype} {cname}` "
                                 f"vs Rust `{rname}: {rtype}`, expected `{expected}`"),
                        suggestion=f"Change Rust parameter type to `{expected}`",
                    ))

        # Variadic check
        if cfn.is_variadic:
            result.issues.append(CompatIssue(
                symbol=name, level=CompatLevel.BREAKING,
                message="C function is variadic; Rust does not support variadic extern fns easily",
                suggestion="Use a va_list wrapper or multiple fixed-argument overloads",
            ))

    # Check struct compatibility
    c_struct_map = {s.name: s for s in c_api.structs}
    rust_struct_map = {s.name: s for s in rust_api.structs}
    for sname in sorted(set(c_struct_map) & set(rust_struct_map)):
        cs = c_struct_map[sname]
        rs = rust_struct_map[sname]
        if rs.repr != "C":
            result.issues.append(CompatIssue(
                symbol=sname, level=CompatLevel.BREAKING,
                message="Rust struct missing #[repr(C)] for C ABI compatibility",
                suggestion="Add #[repr(C)] attribute to the struct",
            ))
        if len(cs.fields) != len(rs.fields):
            result.issues.append(CompatIssue(
                symbol=sname, level=CompatLevel.BREAKING,
                message=(f"Field count mismatch: C has {len(cs.fields)}, "
                         f"Rust has {len(rs.fields)}"),
            ))

    return result


# ---------------------------------------------------------------------------
# generate_rust_bindings
# ---------------------------------------------------------------------------

def generate_rust_bindings(c_header: str) -> str:
    """Generate bindgen-style Rust FFI bindings from a C header string."""
    c_api = extract_c_api(c_header)
    lines: List[str] = [
        "// Auto-generated Rust FFI bindings",
        "// Generated by api_surface_analyzer",
        "",
        "#![allow(non_camel_case_types, non_snake_case, non_upper_case_globals)]",
        "",
        "use std::os::raw::*;",
        "",
    ]

    # Type aliases from typedefs
    for td in c_api.typedefs:
        if td.is_function_pointer:
            rust_ret = _map_c_type_to_rust(td.underlying_type)
            lines.append(
                f"pub type {td.name} = Option<unsafe extern \"C\" fn() -> {rust_ret}>;"
            )
        else:
            rust_type = _map_c_type_to_rust(td.underlying_type)
            lines.append(f"pub type {td.name} = {rust_type};")
    if c_api.typedefs:
        lines.append("")

    # Enums
    for enum in c_api.enums:
        lines.append("#[repr(C)]")
        lines.append("#[derive(Debug, Copy, Clone, PartialEq, Eq)]")
        lines.append(f"pub enum {enum.name} {{")
        next_val = 0
        for vname, vval in enum.variants:
            if vval is not None:
                next_val = vval
            lines.append(f"    {vname} = {next_val},")
            next_val += 1
        lines.append("}")
        lines.append("")

    # Structs
    for struct in c_api.structs:
        if struct.is_opaque:
            lines.append("#[repr(C)]")
            lines.append(f"pub struct {struct.name} {{ _opaque: [u8; 0] }}")
            lines.append("")
            continue
        lines.append("#[repr(C)]")
        lines.append("#[derive(Debug, Copy, Clone)]")
        lines.append(f"pub struct {struct.name} {{")
        for ftype, fname in struct.fields:
            rust_type = _map_c_type_to_rust(ftype)
            lines.append(f"    pub {fname}: {rust_type},")
        lines.append("}")
        lines.append("")

    # Extern block for functions
    if c_api.functions:
        lines.append('extern "C" {')
        for fn in c_api.functions:
            rust_ret = _map_c_type_to_rust(fn.return_type)
            params = []
            for ptype, pname in fn.parameters:
                rtype = _map_c_type_to_rust(ptype)
                safe_name = pname if pname else "_"
                if safe_name in ("type", "match", "fn", "let", "mut", "ref", "self"):
                    safe_name = f"r#{safe_name}"
                params.append(f"{safe_name}: {rtype}")
            if fn.is_variadic:
                params.append("...")
            params_str = ", ".join(params)
            ret_str = f" -> {rust_ret}" if rust_ret != "()" else ""
            lines.append(f"    pub fn {fn.name}({params_str}){ret_str};")
        lines.append("}")
        lines.append("")

    # Macro constants
    for mname, mval in c_api.macros:
        try:
            int(mval, 0)
            lines.append(f"pub const {mname}: c_int = {mval};")
        except ValueError:
            try:
                float(mval)
                lines.append(f"pub const {mname}: c_double = {mval};")
            except ValueError:
                if mval.startswith('"'):
                    lines.append(f"// pub const {mname}: &str = {mval};  // string macro")
                else:
                    lines.append(f"// pub const {mname} = {mval};  // complex macro")

    lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# generate_c_wrapper
# ---------------------------------------------------------------------------

def generate_c_wrapper(rust_api: RustAPISpec) -> str:
    """Generate cbindgen-style C header wrappers from Rust API."""
    lines: List[str] = [
        "/* Auto-generated C header from Rust API */",
        "/* Generated by api_surface_analyzer */",
        "",
        "#ifndef RUST_API_H",
        "#define RUST_API_H",
        "",
        "#include <stdint.h>",
        "#include <stdbool.h>",
        "#include <stddef.h>",
        "",
        "#ifdef __cplusplus",
        'extern "C" {',
        "#endif",
        "",
    ]

    # Structs
    for st in rust_api.structs:
        if st.repr == "C":
            lines.append(f"typedef struct {st.name} {{")
            for fname, ftype in st.fields:
                c_type = _map_rust_type_to_c(ftype)
                lines.append(f"    {c_type} {fname};")
            lines.append(f"}} {st.name};")
            lines.append("")
        else:
            lines.append(f"/* Opaque type — no #[repr(C)] */")
            lines.append(f"typedef struct {st.name} {st.name};")
            lines.append("")

    # Enums
    for en in rust_api.enums:
        lines.append(f"typedef enum {en.name} {{")
        for idx, (vname, vdata) in enumerate(en.variants):
            suffix = "," if idx < len(en.variants) - 1 else ""
            if vdata and vdata.strip().isdigit():
                lines.append(f"    {en.name}_{vname} = {vdata}{suffix}")
            else:
                lines.append(f"    {en.name}_{vname} = {idx}{suffix}")
        lines.append(f"}} {en.name};")
        lines.append("")

    # Functions
    for fn in rust_api.functions:
        if fn.generic_params:
            lines.append(f"/* Skipped generic function: {fn.name} */")
            continue
        c_ret = _map_rust_type_to_c(fn.return_type)
        params: List[str] = []
        for pname, ptype in fn.parameters:
            c_type = _map_rust_type_to_c(ptype)
            params.append(f"{c_type} {pname}")
        params_str = ", ".join(params) if params else "void"
        lines.append(f"{c_ret} {fn.name}({params_str});")
    lines.append("")

    # Type aliases
    for alias_name, alias_type in rust_api.type_aliases:
        c_type = _map_rust_type_to_c(alias_type)
        lines.append(f"typedef {c_type} {alias_name};")
    if rust_api.type_aliases:
        lines.append("")

    lines.extend([
        "#ifdef __cplusplus",
        "}",
        "#endif",
        "",
        "#endif /* RUST_API_H */",
        "",
    ])
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# api_migration_guide
# ---------------------------------------------------------------------------

def api_migration_guide(c_api: CAPISpec, rust_api: RustAPISpec) -> str:
    """Generate a human-readable markdown migration guide."""
    compat = api_compatibility_check(c_api, rust_api)
    changes = detect_api_breaking_changes(c_api, rust_api)

    sections: List[str] = [
        "# C-to-Rust API Migration Guide",
        "",
        "## Overview",
        "",
        f"- **C API symbols**: {c_api.public_symbol_count}",
        f"- **Rust API symbols**: {rust_api.public_symbol_count}",
        f"- **Matched functions**: {compat.matched_functions}",
        f"- **Breaking issues**: {compat.breaking_count}",
        "",
    ]

    if compat.unmatched_c_functions:
        sections.append("## Missing Rust Equivalents")
        sections.append("")
        sections.append("The following C functions have no direct Rust counterpart:")
        sections.append("")
        for name in compat.unmatched_c_functions:
            sections.append(f"- `{name}`")
        sections.append("")

    if compat.unmatched_rust_functions:
        sections.append("## New Rust Functions")
        sections.append("")
        sections.append("Functions added in the Rust API that were not in the C API:")
        sections.append("")
        for name in compat.unmatched_rust_functions:
            sections.append(f"- `{name}`")
        sections.append("")

    if changes:
        sections.append("## Breaking Changes")
        sections.append("")
        sections.append("| Symbol | Change | Old | New |")
        sections.append("|--------|--------|-----|-----|")
        for ch in changes:
            sections.append(
                f"| `{ch.symbol}` | {ch.description} "
                f"| `{ch.old_signature}` | `{ch.new_signature}` |"
            )
        sections.append("")

    # Type mapping reference
    sections.append("## Type Mapping Reference")
    sections.append("")
    sections.append("| C Type | Rust Type |")
    sections.append("|--------|-----------|")
    for c_t, r_t in sorted(C_TO_RUST_TYPE_MAP.items()):
        sections.append(f"| `{c_t}` | `{r_t}` |")
    sections.append("")

    # Migration steps
    sections.append("## Recommended Migration Steps")
    sections.append("")
    sections.append("1. Generate raw FFI bindings with `generate_rust_bindings()`.")
    sections.append("2. Run `api_compatibility_check()` to identify ABI issues.")
    sections.append("3. Add `#[repr(C)]` to all structs shared across the FFI boundary.")
    sections.append("4. Wrap unsafe FFI calls with `suggest_safe_wrappers()`.")
    sections.append("5. Write FFI integration tests with `generate_ffi_tests()`.")
    sections.append("6. Incrementally replace C callers with Rust safe API calls.")
    sections.append("")

    if compat.issues:
        sections.append("## All Compatibility Issues")
        sections.append("")
        for issue in compat.issues:
            level_emoji = {
                CompatLevel.BREAKING: "🔴",
                CompatLevel.COMPATIBLE: "🟢",
                CompatLevel.DEPRECATED: "🟡",
                CompatLevel.ADDED: "🔵",
                CompatLevel.REMOVED: "⚪",
            }.get(issue.level, "⚪")
            sections.append(f"- {level_emoji} **{issue.symbol}**: {issue.message}")
            if issue.suggestion:
                sections.append(f"  - 💡 {issue.suggestion}")
        sections.append("")

    return "\n".join(sections)


# ---------------------------------------------------------------------------
# detect_api_breaking_changes
# ---------------------------------------------------------------------------

def detect_api_breaking_changes(
    old_api: CAPISpec, new_api: RustAPISpec
) -> List[APIChange]:
    """Find breaking changes between a C API and its Rust replacement."""
    changes: List[APIChange] = []

    old_fn_map = {f.name: f for f in old_api.functions}
    new_fn_map = {f.name: f for f in new_api.functions}

    # Removed functions
    for name in sorted(set(old_fn_map) - set(new_fn_map)):
        cfn = old_fn_map[name]
        changes.append(APIChange(
            symbol=name,
            change_type=CompatLevel.REMOVED,
            old_signature=cfn.signature,
            new_signature="<removed>",
            description="Function removed in Rust API",
        ))

    # Added functions
    for name in sorted(set(new_fn_map) - set(old_fn_map)):
        rfn = new_fn_map[name]
        changes.append(APIChange(
            symbol=name,
            change_type=CompatLevel.ADDED,
            old_signature="<not present>",
            new_signature=rfn.signature,
            description="New function added in Rust API",
        ))

    # Modified functions
    for name in sorted(set(old_fn_map) & set(new_fn_map)):
        cfn = old_fn_map[name]
        rfn = new_fn_map[name]

        # Return type change
        expected_ret = _map_c_type_to_rust(cfn.return_type)
        if rfn.return_type != expected_ret and rfn.return_type != cfn.return_type:
            changes.append(APIChange(
                symbol=name,
                change_type=CompatLevel.BREAKING,
                old_signature=cfn.signature,
                new_signature=rfn.signature,
                description=f"Return type changed: {cfn.return_type} → {rfn.return_type}",
            ))

        # Parameter count change
        if len(cfn.parameters) != len(rfn.parameters):
            changes.append(APIChange(
                symbol=name,
                change_type=CompatLevel.BREAKING,
                old_signature=cfn.signature,
                new_signature=rfn.signature,
                description=(f"Parameter count changed: "
                             f"{len(cfn.parameters)} → {len(rfn.parameters)}"),
            ))
        else:
            for idx, ((ctype, _cname), (_rname, rtype)) in enumerate(
                zip(cfn.parameters, rfn.parameters)
            ):
                expected = _map_c_type_to_rust(ctype)
                if rtype != expected and rtype != ctype:
                    changes.append(APIChange(
                        symbol=f"{name}[param {idx}]",
                        change_type=CompatLevel.BREAKING,
                        old_signature=f"{ctype}",
                        new_signature=f"{rtype}",
                        description=f"Parameter {idx} type changed",
                    ))

    # Struct changes
    old_struct_map = {s.name: s for s in old_api.structs}
    new_struct_map = {s.name: s for s in new_api.structs}

    for name in sorted(set(old_struct_map) - set(new_struct_map)):
        changes.append(APIChange(
            symbol=name,
            change_type=CompatLevel.REMOVED,
            old_signature=f"struct {name}",
            new_signature="<removed>",
            description="Struct removed in Rust API",
        ))

    for name in sorted(set(old_struct_map) & set(new_struct_map)):
        cs = old_struct_map[name]
        rs = new_struct_map[name]
        if len(cs.fields) != len(rs.fields):
            changes.append(APIChange(
                symbol=name,
                change_type=CompatLevel.BREAKING,
                old_signature=f"struct {name} ({len(cs.fields)} fields)",
                new_signature=f"struct {name} ({len(rs.fields)} fields)",
                description="Struct field count changed",
            ))

    # Enum changes
    old_enum_map = {e.name: e for e in old_api.enums}
    new_enum_map = {e.name: e for e in new_api.enums}

    for name in sorted(set(old_enum_map) - set(new_enum_map)):
        changes.append(APIChange(
            symbol=name,
            change_type=CompatLevel.REMOVED,
            old_signature=f"enum {name}",
            new_signature="<removed>",
            description="Enum removed in Rust API",
        ))

    for name in sorted(set(old_enum_map) & set(new_enum_map)):
        old_variants = {v[0] for v in old_enum_map[name].variants}
        new_variants = {v[0] for v in new_enum_map[name].variants}
        removed = old_variants - new_variants
        if removed:
            changes.append(APIChange(
                symbol=name,
                change_type=CompatLevel.BREAKING,
                old_signature=f"enum {name} with {', '.join(sorted(removed))}",
                new_signature=f"enum {name} without {', '.join(sorted(removed))}",
                description=f"Enum variants removed: {', '.join(sorted(removed))}",
            ))

    return changes


# ---------------------------------------------------------------------------
# generate_ffi_tests
# ---------------------------------------------------------------------------

def generate_ffi_tests(c_api: CAPISpec, rust_api: RustAPISpec) -> str:
    """Generate FFI integration tests for verifying C/Rust interop."""
    lines: List[str] = [
        "// Auto-generated FFI integration tests",
        "// Generated by api_surface_analyzer",
        "",
        "#[cfg(test)]",
        "mod ffi_tests {",
        "    use super::*;",
        "    use std::mem;",
        "    use std::os::raw::*;",
        "",
    ]

    # Size/alignment tests for structs
    c_struct_map = {s.name: s for s in c_api.structs}
    rust_struct_map = {s.name: s for s in rust_api.structs}

    for name in sorted(set(c_struct_map) & set(rust_struct_map)):
        cs = c_struct_map[name]
        rs = rust_struct_map[name]
        if cs.is_opaque:
            continue
        expected_size = cs.estimated_size
        lines.append("    #[test]")
        lines.append(f"    fn test_{name.lower()}_layout() {{")
        lines.append(f"        // C struct has {len(cs.fields)} fields, "
                     f"estimated size {expected_size}")
        lines.append(f"        let rust_size = mem::size_of::<{name}>();")
        lines.append(f"        let rust_align = mem::align_of::<{name}>();")
        lines.append(f"        assert!(rust_size > 0, \"{name} has zero size\");")
        lines.append(f"        assert!(rust_align > 0, \"{name} has zero alignment\");")
        # Check field count matches
        lines.append(f"        // Verify field count: C={len(cs.fields)} "
                     f"Rust={len(rs.fields)}")
        lines.append(f"        assert_eq!({len(cs.fields)}, {len(rs.fields)}, "
                     f"\"field count mismatch for {name}\");")
        lines.append("    }")
        lines.append("")

    # Enum value tests
    c_enum_map = {e.name: e for e in c_api.enums}
    rust_enum_map = {e.name: e for e in rust_api.enums}

    for name in sorted(set(c_enum_map) & set(rust_enum_map)):
        ce = c_enum_map[name]
        lines.append("    #[test]")
        lines.append(f"    fn test_{name.lower()}_values() {{")
        next_val = 0
        for vname, vval in ce.variants:
            if vval is not None:
                next_val = vval
            lines.append(f"        assert_eq!({name}::{vname} as i32, {next_val}, "
                         f"\"enum value mismatch for {vname}\");")
            next_val += 1
        lines.append("    }")
        lines.append("")

    # Function signature smoke tests
    matched_fns = set(f.name for f in c_api.functions) & set(
        f.name for f in rust_api.functions
    )
    for name in sorted(matched_fns):
        cfn = {f.name: f for f in c_api.functions}[name]
        lines.append("    #[test]")
        lines.append(f"    fn test_{name}_exists() {{")
        lines.append(f"        // Verify the symbol is linkable")
        lines.append(f"        let fptr: unsafe extern \"C\" fn(")
        param_types = []
        for ptype, _pname in cfn.parameters:
            param_types.append(f"            {_map_c_type_to_rust(ptype)},")
        for pt in param_types:
            lines.append(pt)
        rust_ret = _map_c_type_to_rust(cfn.return_type)
        ret_str = f" -> {rust_ret}" if rust_ret != "()" else ""
        lines.append(f"        ){ret_str} = {name};")
        lines.append(f"        let _ = fptr;  // suppress unused warning")
        lines.append("    }")
        lines.append("")

    # Roundtrip tests for structs with repr(C)
    for name in sorted(set(c_struct_map) & set(rust_struct_map)):
        rs = rust_struct_map[name]
        if rs.repr != "C" or not rs.fields:
            continue
        lines.append("    #[test]")
        lines.append(f"    fn test_{name.lower()}_roundtrip() {{")
        lines.append(f"        // Verify struct can be zeroed and fields accessed")
        lines.append(f"        let val: {name} = unsafe {{ mem::zeroed() }};")
        for fname, ftype in rs.fields:
            lines.append(f"        let _ = val.{fname};")
        lines.append("    }")
        lines.append("")

    lines.append("}")
    lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# suggest_safe_wrappers
# ---------------------------------------------------------------------------

def suggest_safe_wrappers(rust_bindings: str) -> str:
    """Wrap unsafe FFI bindings in safe Rust API."""
    lines: List[str] = [
        "// Safe wrappers around FFI bindings",
        "// Generated by api_surface_analyzer",
        "",
        "use std::ffi::{CStr, CString};",
        "use std::os::raw::*;",
        "",
    ]

    # Parse extern "C" function declarations from the bindings
    fn_pat = re.compile(
        r"pub\s+fn\s+(\w+)\s*\(([^)]*)\)(?:\s*->\s*([\w\*:&<>, ]+))?\s*;"
    )

    for m in fn_pat.finditer(rust_bindings):
        name = m.group(1)
        params_raw = m.group(2).strip()
        ret_type = (m.group(3) or "()").strip()

        params: List[Tuple[str, str]] = []
        if params_raw and params_raw != "...":
            for p in params_raw.split(","):
                p = p.strip()
                if p == "...":
                    continue
                if ":" in p:
                    pname, ptype = p.split(":", 1)
                    params.append((pname.strip(), ptype.strip()))

        safe_name = f"safe_{name}"
        safe_params: List[str] = []
        call_args: List[str] = []
        needs_result_wrap = False
        needs_string_conv = False
        needs_null_check = False

        for pname, ptype in params:
            ptype = ptype.strip()
            if ptype == "*const c_char":
                safe_params.append(f"{pname}: &str")
                call_args.append(f"{pname}_c.as_ptr()")
                needs_string_conv = True
            elif ptype == "*mut c_char":
                safe_params.append(f"{pname}: &mut String")
                call_args.append(f"{pname}_buf.as_mut_ptr()")
                needs_string_conv = True
            elif ptype.startswith("*const "):
                inner = ptype[7:]
                safe_params.append(f"{pname}: &{inner}")
                call_args.append(f"{pname} as *const {inner}")
            elif ptype.startswith("*mut "):
                inner = ptype[5:]
                safe_params.append(f"{pname}: &mut {inner}")
                call_args.append(f"{pname} as *mut {inner}")
                needs_null_check = True
            else:
                safe_params.append(f"{pname}: {ptype}")
                call_args.append(pname)

        # Determine safe return type
        safe_ret = ret_type
        ret_conversion_pre = ""
        ret_conversion_post = ""
        if ret_type == "*const c_char":
            safe_ret = "Option<String>"
            needs_result_wrap = True
        elif ret_type == "*mut c_char":
            safe_ret = "Option<String>"
            needs_result_wrap = True
        elif ret_type.startswith("*"):
            inner = ret_type.replace("*const ", "").replace("*mut ", "")
            safe_ret = f"Option<&{inner}>"
            needs_null_check = True

        safe_params_str = ", ".join(safe_params)
        ret_str = f" -> {safe_ret}" if safe_ret != "()" else ""

        lines.append(f"/// Safe wrapper for `{name}`.")
        lines.append(f"pub fn {safe_name}({safe_params_str}){ret_str} {{")

        # String conversions
        for pname, ptype in params:
            ptype = ptype.strip()
            if ptype == "*const c_char":
                lines.append(
                    f"    let {pname}_c = CString::new({pname})"
                    f".expect(\"CString conversion failed\");"
                )

        call_args_str = ", ".join(call_args)
        call_expr = f"{name}({call_args_str})"

        if ret_type == "()" or not ret_type:
            lines.append(f"    unsafe {{ {call_expr} }}")
        elif needs_result_wrap and "c_char" in ret_type:
            lines.append(f"    let ptr = unsafe {{ {call_expr} }};")
            lines.append("    if ptr.is_null() {")
            lines.append("        None")
            lines.append("    } else {")
            lines.append("        Some(unsafe { CStr::from_ptr(ptr) }"
                         ".to_string_lossy().into_owned())")
            lines.append("    }")
        elif needs_null_check and ret_type.startswith("*"):
            lines.append(f"    let ptr = unsafe {{ {call_expr} }};")
            lines.append("    if ptr.is_null() {")
            lines.append("        None")
            lines.append("    } else {")
            lines.append("        Some(unsafe { &*ptr })")
            lines.append("    }")
        else:
            lines.append(f"    unsafe {{ {call_expr} }}")

        lines.append("}")
        lines.append("")

    if len(lines) <= 7:
        lines.append("// No extern functions found to wrap.")
        lines.append("")

    return "\n".join(lines)
