"""
C Macro Expander and Migration Module for C-to-Rust conversion.

Handles recursive macro expansion, classification, side-effect detection,
header guard analysis, conditional compilation, and Rust code generation.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Dict, List, Optional, Set, Tuple


class MacroKind(Enum):
    OBJECT_LIKE = auto()
    FUNCTION_LIKE = auto()
    TYPE_ALIAS = auto()
    CONDITIONAL = auto()
    INCLUDE_GUARD = auto()
    STRINGIFICATION = auto()
    TOKEN_PASTE = auto()
    VARIADIC_MACRO = auto()
    X_MACRO = auto()


@dataclass
class MacroDef:
    name: str
    params: Optional[List[str]] = None
    body: str = ""
    kind: MacroKind = MacroKind.OBJECT_LIKE
    line_number: int = 0
    is_variadic: bool = False
    file_origin: str = ""
    dependencies: List[str] = field(default_factory=list)

    @property
    def is_function_like(self) -> bool:
        return self.params is not None


@dataclass
class ExpandedCode:
    original: str
    expanded: str
    macros_used: List[str] = field(default_factory=list)
    expansion_steps: List[str] = field(default_factory=list)
    unresolved: List[str] = field(default_factory=list)


@dataclass
class MacroClassification:
    macro: MacroDef
    kind: MacroKind
    rust_equivalent: str
    confidence: float = 1.0
    notes: str = ""


@dataclass
class MacroSideEffect:
    macro_name: str
    effect_type: str
    description: str
    severity: str = "warning"
    line_number: int = 0
    affected_params: List[str] = field(default_factory=list)


@dataclass
class HeaderGuard:
    guard_symbol: str
    file_name: str
    style: str = "ifndef"
    line_start: int = 0
    line_end: int = 0
    is_pragma_once: bool = False


@dataclass
class MacroExpansionResult:
    input_text: str
    output_text: str
    macro_name: str
    depth: int = 0
    substitutions: Dict[str, str] = field(default_factory=dict)


@dataclass
class MacroConversion:
    original: MacroDef
    rust_code: str
    conversion_kind: str
    warnings: List[str] = field(default_factory=list)
    requires_unsafe: bool = False


@dataclass
class ConditionalBlock:
    condition: str
    directive: str
    body: str
    line_start: int = 0
    line_end: int = 0
    children: List[ConditionalBlock] = field(default_factory=list)
    else_body: Optional[str] = None


# ---------------------------------------------------------------------------
# Internal parsing helpers
# ---------------------------------------------------------------------------

_DEFINE_RE = re.compile(
    r"^\s*#\s*define\s+(\w+)"
    r"(?:\(([^)]*)\))?"
    r"(?:\s+(.*))?$",
    re.MULTILINE,
)

_COND_START_RE = re.compile(
    r"^\s*#\s*(if|ifdef|ifndef)\s+(.*?)\s*$", re.MULTILINE
)
_COND_ELIF_RE = re.compile(r"^\s*#\s*elif\s+(.*?)\s*$", re.MULTILINE)
_COND_ELSE_RE = re.compile(r"^\s*#\s*else\s*$", re.MULTILINE)
_COND_END_RE = re.compile(r"^\s*#\s*endif\s*$", re.MULTILINE)

_PRAGMA_ONCE_RE = re.compile(r"^\s*#\s*pragma\s+once\s*$", re.MULTILINE)

_INCLUDE_RE = re.compile(
    r'^\s*#\s*include\s+[<"]([^>"]+)[>"]', re.MULTILINE
)

_UNDEF_RE = re.compile(r"^\s*#\s*undef\s+(\w+)", re.MULTILINE)


def _parse_macro_defs(c_code: str) -> Dict[str, MacroDef]:
    """Parse all #define directives from C source into MacroDef objects."""
    macros: Dict[str, MacroDef] = {}
    lines = c_code.splitlines()
    i = 0
    while i < len(lines):
        line = lines[i]
        # Handle line continuations
        full_line = line
        while full_line.rstrip().endswith("\\") and i + 1 < len(lines):
            full_line = full_line.rstrip()[:-1] + " " + lines[i + 1].strip()
            i += 1

        m = _DEFINE_RE.match(full_line)
        if m:
            name = m.group(1)
            raw_params = m.group(2)
            body = (m.group(3) or "").strip()

            params: Optional[List[str]] = None
            is_variadic = False
            if raw_params is not None:
                params = [p.strip() for p in raw_params.split(",") if p.strip()]
                if params and params[-1] == "...":
                    is_variadic = True
                    params[-1] = "__VA_ARGS__"
                elif params and params[-1].endswith("..."):
                    is_variadic = True
                    params[-1] = params[-1][:-3].strip()

            kind = _infer_kind(name, params, body, is_variadic, c_code)

            deps = _find_macro_deps(body, set(macros.keys()))

            macro = MacroDef(
                name=name,
                params=params,
                body=body,
                kind=kind,
                line_number=i + 1,
                is_variadic=is_variadic,
                dependencies=deps,
            )
            macros[name] = macro
        i += 1
    return macros


def _infer_kind(
    name: str,
    params: Optional[List[str]],
    body: str,
    is_variadic: bool,
    full_code: str,
) -> MacroKind:
    """Determine the MacroKind from definition characteristics."""
    if is_variadic:
        return MacroKind.VARIADIC_MACRO
    if "#" in body and params:
        if "##" in body:
            return MacroKind.TOKEN_PASTE
        return MacroKind.STRINGIFICATION

    # Check for include-guard pattern
    guard_pat = re.compile(
        r"^\s*#\s*ifndef\s+" + re.escape(name) + r"\s*\n"
        r"\s*#\s*define\s+" + re.escape(name),
        re.MULTILINE,
    )
    if guard_pat.search(full_code) and body == "":
        return MacroKind.INCLUDE_GUARD

    # Check for X-macro pattern (body references another macro for expansion)
    if re.search(r"\bX\s*\(", body):
        return MacroKind.X_MACRO

    # Type-alias heuristic: body is a single type keyword or pointer type
    type_keywords = {
        "int", "char", "short", "long", "float", "double", "void",
        "unsigned", "signed", "uint8_t", "uint16_t", "uint32_t", "uint64_t",
        "int8_t", "int16_t", "int32_t", "int64_t", "size_t", "ssize_t",
        "bool", "BOOL", "BYTE", "WORD", "DWORD",
    }
    stripped = body.rstrip(";").strip()
    if stripped in type_keywords or re.match(r"^(const\s+)?\w+\s*\*?$", stripped):
        if params is None and stripped:
            return MacroKind.TYPE_ALIAS

    if params is not None:
        return MacroKind.FUNCTION_LIKE

    # Conditional-compilation sentinel (value-less define used in #ifdef)
    if body == "":
        return MacroKind.CONDITIONAL

    return MacroKind.OBJECT_LIKE


def _find_macro_deps(body: str, known: Set[str]) -> List[str]:
    """Return list of macro names referenced in *body* that are already known."""
    tokens = re.findall(r"\b([A-Za-z_]\w*)\b", body)
    return [t for t in tokens if t in known]


def _substitute_args(
    body: str, params: List[str], args: List[str]
) -> str:
    """Replace parameter tokens in *body* with supplied *args*."""
    result = body
    mapping: Dict[str, str] = {}
    for idx, p in enumerate(params):
        val = args[idx] if idx < len(args) else ""
        mapping[p] = val

    # Handle stringification (#param)
    for p, v in mapping.items():
        result = re.sub(r"#\s*" + re.escape(p), f'"{v}"', result)

    # Handle token pasting (a ## b)
    result = re.sub(r"\s*##\s*", "", result)

    # Plain substitution
    for p, v in mapping.items():
        result = re.sub(r"\b" + re.escape(p) + r"\b", v, result)

    return result


def _expand_single(
    text: str,
    macros: Dict[str, MacroDef],
    expanding: Set[str],
    depth: int,
    steps: List[str],
    used: List[str],
    max_depth: int = 64,
) -> str:
    """Recursively expand macros in *text*."""
    if depth > max_depth:
        return text

    changed = True
    iterations = 0
    while changed and iterations < 200:
        changed = False
        iterations += 1
        for name, mdef in macros.items():
            if name in expanding:
                continue
            if mdef.is_function_like:
                pattern = re.compile(
                    r"\b" + re.escape(name) + r"\s*\(", re.DOTALL
                )
                match = pattern.search(text)
                while match:
                    start = match.start()
                    paren_start = match.end() - 1
                    args, end = _extract_args(text, paren_start)
                    if args is not None:
                        expanded_body = _substitute_args(
                            mdef.body, mdef.params or [], args
                        )
                        expanding.add(name)
                        expanded_body = _expand_single(
                            expanded_body, macros, expanding,
                            depth + 1, steps, used, max_depth,
                        )
                        expanding.discard(name)
                        text = text[:start] + expanded_body + text[end:]
                        steps.append(
                            f"Expand {name}({', '.join(args)}) -> {expanded_body}"
                        )
                        if name not in used:
                            used.append(name)
                        changed = True
                    match = pattern.search(text, start + len(expanded_body))
            else:
                token_re = re.compile(r"\b" + re.escape(name) + r"\b")
                m = token_re.search(text)
                if m:
                    expanding.add(name)
                    expanded_body = _expand_single(
                        mdef.body, macros, expanding,
                        depth + 1, steps, used, max_depth,
                    )
                    expanding.discard(name)
                    text = token_re.sub(expanded_body, text, count=1)
                    steps.append(f"Expand {name} -> {expanded_body}")
                    if name not in used:
                        used.append(name)
                    changed = True
    return text


def _extract_args(text: str, paren_pos: int) -> Tuple[Optional[List[str]], int]:
    """Extract comma-separated arguments from balanced parentheses."""
    if paren_pos >= len(text) or text[paren_pos] != "(":
        return None, paren_pos
    depth = 0
    current: List[str] = []
    buf: List[str] = []
    i = paren_pos
    while i < len(text):
        ch = text[i]
        if ch == "(":
            if depth > 0:
                buf.append(ch)
            depth += 1
        elif ch == ")":
            depth -= 1
            if depth == 0:
                current.append("".join(buf).strip())
                return current, i + 1
            buf.append(ch)
        elif ch == "," and depth == 1:
            current.append("".join(buf).strip())
            buf = []
        else:
            buf.append(ch)
        i += 1
    return None, paren_pos


# ---------------------------------------------------------------------------
# C type to Rust type mapping
# ---------------------------------------------------------------------------

_C_TO_RUST_TYPE: Dict[str, str] = {
    "int": "i32",
    "unsigned int": "u32",
    "unsigned": "u32",
    "long": "i64",
    "unsigned long": "u64",
    "short": "i16",
    "unsigned short": "u16",
    "char": "i8",
    "unsigned char": "u8",
    "float": "f32",
    "double": "f64",
    "void": "()",
    "bool": "bool",
    "BOOL": "bool",
    "size_t": "usize",
    "ssize_t": "isize",
    "uint8_t": "u8",
    "uint16_t": "u16",
    "uint32_t": "u32",
    "uint64_t": "u64",
    "int8_t": "i8",
    "int16_t": "i16",
    "int32_t": "i32",
    "int64_t": "i64",
    "BYTE": "u8",
    "WORD": "u16",
    "DWORD": "u32",
}


def _c_type_to_rust(c_type: str) -> str:
    """Map a C type string to its Rust equivalent."""
    stripped = c_type.strip().rstrip(";").strip()
    if stripped in _C_TO_RUST_TYPE:
        return _C_TO_RUST_TYPE[stripped]
    if stripped.endswith("*"):
        inner = stripped[:-1].strip()
        rust_inner = _c_type_to_rust(inner)
        return f"*mut {rust_inner}"
    if stripped.startswith("const "):
        inner = stripped[6:].strip()
        if inner.endswith("*"):
            base = inner[:-1].strip()
            rust_base = _c_type_to_rust(base)
            return f"*const {rust_base}"
        return _c_type_to_rust(inner)
    return stripped


def _infer_rust_type_from_value(value: str) -> str:
    """Guess a Rust type from a literal value string."""
    v = value.strip().rstrip(";").strip()
    if v.startswith('"'):
        return "&str"
    if v.startswith("'"):
        return "u8"
    if re.match(r"^0[xX][0-9a-fA-F]+[uU]?[lL]*$", v):
        if v.upper().endswith("ULL"):
            return "u64"
        if v.upper().endswith("UL") or v.upper().endswith("U"):
            return "u32"
        return "i32"
    if re.match(r"^-?\d+\.\d*[fF]?$", v):
        if v.endswith("f") or v.endswith("F"):
            return "f32"
        return "f64"
    if re.match(r"^-?\d+[uU]?[lL]*$", v):
        if v.upper().endswith("ULL"):
            return "u64"
        if v.upper().endswith("UL") or v.upper().endswith("U"):
            return "u32"
        if v.upper().endswith("LL"):
            return "i64"
        if v.upper().endswith("L"):
            return "i64"
        return "i32"
    if v in ("true", "false", "TRUE", "FALSE"):
        return "bool"
    return "i32"


def _sanitize_rust_value(value: str) -> str:
    """Clean C literal suffixes for Rust consumption."""
    v = value.strip().rstrip(";").strip()
    v = re.sub(r"([0-9a-fA-Fx]+)[uU]?[lL]{0,2}$", r"\1", v)
    if v.endswith("f") or v.endswith("F"):
        v = v[:-1]
    return v


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def expand_macros(c_code: str) -> ExpandedCode:
    """Fully expand all macros in *c_code* with recursive substitution."""
    macros = _parse_macro_defs(c_code)

    # Remove preprocessor directives from the working copy
    work = re.sub(r"^\s*#\s*define\s+.*$", "", c_code, flags=re.MULTILINE)
    work = re.sub(r"^\s*#\s*undef\s+.*$", "", work, flags=re.MULTILINE)
    work = re.sub(r"^\s*#\s*include\s+.*$", "", work, flags=re.MULTILINE)
    work = re.sub(r"^\s*#\s*(if|ifdef|ifndef|elif|else|endif).*$", "", work, flags=re.MULTILINE)

    steps: List[str] = []
    used: List[str] = []
    expanded = _expand_single(work, macros, set(), 0, steps, used)

    # Identify unresolved identifiers that look like macros (ALL_CAPS)
    remaining_caps = set(re.findall(r"\b([A-Z_][A-Z0-9_]{2,})\b", expanded))
    unresolved = [tok for tok in remaining_caps if tok not in macros]

    return ExpandedCode(
        original=c_code,
        expanded=expanded.strip(),
        macros_used=used,
        expansion_steps=steps,
        unresolved=sorted(unresolved),
    )


def classify_macros(c_code: str) -> List[MacroClassification]:
    """Classify every macro in *c_code* and suggest a Rust equivalent."""
    macros = _parse_macro_defs(c_code)
    results: List[MacroClassification] = []
    for mdef in macros.values():
        kind = mdef.kind
        rust_eq, confidence, notes = _classification_details(mdef)
        results.append(
            MacroClassification(
                macro=mdef,
                kind=kind,
                rust_equivalent=rust_eq,
                confidence=confidence,
                notes=notes,
            )
        )
    return results


def _classification_details(mdef: MacroDef) -> Tuple[str, float, str]:
    """Return (rust_equivalent, confidence, notes) for a MacroDef."""
    if mdef.kind == MacroKind.INCLUDE_GUARD:
        return "// no equivalent needed (Rust modules)", 1.0, "Include guard"
    if mdef.kind == MacroKind.CONDITIONAL:
        return f'#[cfg(feature = "{mdef.name.lower()}")]', 0.8, "Feature flag"
    if mdef.kind == MacroKind.TYPE_ALIAS:
        rust_t = _c_type_to_rust(mdef.body)
        return f"type {mdef.name} = {rust_t};", 0.95, "Type alias"
    if mdef.kind == MacroKind.STRINGIFICATION:
        return f"macro_rules! {mdef.name.lower()} {{ ... }}", 0.6, "Stringification requires macro_rules!"
    if mdef.kind == MacroKind.TOKEN_PASTE:
        return f"macro_rules! {mdef.name.lower()} {{ ... }}", 0.5, "Token paste requires macro_rules!"
    if mdef.kind == MacroKind.VARIADIC_MACRO:
        return f"macro_rules! {mdef.name.lower()} {{ ... }}", 0.6, "Variadic macro"
    if mdef.kind == MacroKind.X_MACRO:
        return f"macro_rules! {mdef.name.lower()} {{ ... }}", 0.4, "X-macro pattern"
    if mdef.kind == MacroKind.FUNCTION_LIKE:
        params = mdef.params or []
        sig = ", ".join(f"{p}: _" for p in params)
        return f"fn {mdef.name.lower()}({sig}) -> _ {{ {mdef.body} }}", 0.7, "Function-like"
    # OBJECT_LIKE
    rt = _infer_rust_type_from_value(mdef.body)
    val = _sanitize_rust_value(mdef.body)
    return f"const {mdef.name}: {rt} = {val};", 0.9, "Object-like constant"


def macro_to_rust(macro_def: MacroDef) -> str:
    """Convert a single MacroDef to its Rust equivalent string."""
    kind = macro_def.kind

    if kind == MacroKind.INCLUDE_GUARD:
        return f"// Include guard {macro_def.name} has no Rust equivalent (module system handles this)"

    if kind == MacroKind.CONDITIONAL:
        feature = macro_def.name.lower().strip("_")
        return f'#[cfg(feature = "{feature}")]'

    if kind == MacroKind.TYPE_ALIAS:
        rust_type = _c_type_to_rust(macro_def.body)
        return f"pub type {macro_def.name} = {rust_type};"

    if kind == MacroKind.OBJECT_LIKE:
        rust_type = _infer_rust_type_from_value(macro_def.body)
        val = _sanitize_rust_value(macro_def.body)
        return f"pub const {macro_def.name}: {rust_type} = {val};"

    if kind == MacroKind.FUNCTION_LIKE:
        return _convert_function_like(macro_def)

    if kind == MacroKind.STRINGIFICATION:
        return _convert_stringify_macro(macro_def)

    if kind == MacroKind.TOKEN_PASTE:
        return _convert_token_paste_macro(macro_def)

    if kind == MacroKind.VARIADIC_MACRO:
        return _convert_variadic_macro(macro_def)

    if kind == MacroKind.X_MACRO:
        return _convert_x_macro(macro_def)

    return f"// TODO: convert {macro_def.name}"


def _convert_function_like(mdef: MacroDef) -> str:
    """Convert a function-like macro to a Rust inline fn or macro_rules!."""
    params = mdef.params or []
    body = mdef.body.strip()

    # Detect if the body is a simple expression (no statements, no blocks)
    is_simple = ";" not in body and "{" not in body and "do" not in body

    if is_simple and not _has_multiple_eval(mdef):
        param_list = ", ".join(f"{p}: i32" for p in params)
        rust_body = _translate_expr(body)
        return (
            f"#[inline]\n"
            f"pub fn {mdef.name.lower()}({param_list}) -> i32 {{\n"
            f"    {rust_body}\n"
            f"}}"
        )

    # Fall back to macro_rules! for complex or multi-eval macros
    param_names = ", ".join(f"${p}:expr" for p in params)
    rust_body = _translate_expr(body)
    for p in params:
        rust_body = re.sub(r"\b" + re.escape(p) + r"\b", f"${p}", rust_body)
    return (
        f"macro_rules! {mdef.name.lower()} {{\n"
        f"    ({param_names}) => {{\n"
        f"        {rust_body}\n"
        f"    }};\n"
        f"}}"
    )


def _convert_stringify_macro(mdef: MacroDef) -> str:
    params = mdef.params or []
    param_names = ", ".join(f"${p}:expr" for p in params)
    body = mdef.body
    for p in params:
        body = re.sub(r"#\s*" + re.escape(p), f"stringify!(${p})", body)
        body = re.sub(r"\b" + re.escape(p) + r"\b", f"${p}", body)
    return (
        f"macro_rules! {mdef.name.lower()} {{\n"
        f"    ({param_names}) => {{\n"
        f"        {body}\n"
        f"    }};\n"
        f"}}"
    )


def _convert_token_paste_macro(mdef: MacroDef) -> str:
    params = mdef.params or []
    param_names = ", ".join(f"${p}:ident" for p in params)
    body = mdef.body
    # Replace a##b with concat_idents style
    body = re.sub(r"(\w+)\s*##\s*(\w+)", r"paste::paste! { [<\1 \2>] }", body)
    for p in params:
        body = re.sub(r"\b" + re.escape(p) + r"\b", f"${p}", body)
    return (
        f"macro_rules! {mdef.name.lower()} {{\n"
        f"    ({param_names}) => {{\n"
        f"        {body}\n"
        f"    }};\n"
        f"}}"
    )


def _convert_variadic_macro(mdef: MacroDef) -> str:
    params = mdef.params or []
    fixed = [p for p in params if p != "__VA_ARGS__"]
    fixed_names = ", ".join(f"${p}:expr" for p in fixed)
    sep = ", " if fixed_names else ""
    body = mdef.body
    body = body.replace("__VA_ARGS__", "$($rest),*")
    for p in fixed:
        body = re.sub(r"\b" + re.escape(p) + r"\b", f"${p}", body)
    return (
        f"macro_rules! {mdef.name.lower()} {{\n"
        f"    ({fixed_names}{sep}$($rest:tt),*) => {{\n"
        f"        {body}\n"
        f"    }};\n"
        f"}}"
    )


def _convert_x_macro(mdef: MacroDef) -> str:
    return (
        f"// X-macro {mdef.name}: convert manually.\n"
        f"// Original body: {mdef.body}\n"
        f"macro_rules! {mdef.name.lower()} {{\n"
        f"    ($callback:ident) => {{\n"
        f"        // Invoke $callback for each entry\n"
        f"        // $callback!(...);\n"
        f"    }};\n"
        f"}}"
    )


def _translate_expr(expr: str) -> str:
    """Rough translation of C expression syntax to Rust."""
    result = expr.strip()
    # NULL -> std::ptr::null()
    result = re.sub(r"\bNULL\b", "std::ptr::null()", result)
    # sizeof(T) -> std::mem::size_of::<T>()
    result = re.sub(
        r"\bsizeof\s*\(([^)]+)\)",
        lambda m: f"std::mem::size_of::<{_c_type_to_rust(m.group(1))}>()",
        result,
    )
    # (type)expr casts -> expr as type  (simple cases only)
    result = re.sub(
        r"\((\w+)\)\s*(\w+)",
        lambda m: f"{m.group(2)} as {_c_type_to_rust(m.group(1))}",
        result,
    )
    return result


def _has_multiple_eval(mdef: MacroDef) -> bool:
    """Check if any parameter appears more than once in the body."""
    for p in mdef.params or []:
        if len(re.findall(r"\b" + re.escape(p) + r"\b", mdef.body)) > 1:
            return True
    return False


def detect_macro_side_effects(c_code: str) -> List[MacroSideEffect]:
    """Detect macros that may have side effects or unsafe evaluation patterns."""
    macros = _parse_macro_defs(c_code)
    effects: List[MacroSideEffect] = []

    for name, mdef in macros.items():
        body = mdef.body

        # 1. Multiple evaluation of parameters
        if mdef.is_function_like and _has_multiple_eval(mdef):
            for p in mdef.params or []:
                count = len(re.findall(r"\b" + re.escape(p) + r"\b", body))
                if count > 1:
                    effects.append(
                        MacroSideEffect(
                            macro_name=name,
                            effect_type="multiple_evaluation",
                            description=(
                                f"Parameter '{p}' is evaluated {count} times. "
                                f"Passing an expression with side effects (e.g. i++) "
                                f"will cause unexpected behaviour."
                            ),
                            severity="error",
                            line_number=mdef.line_number,
                            affected_params=[p],
                        )
                    )

        # 2. Increment / decrement operators in the body
        if re.search(r"\+\+|--", body):
            effects.append(
                MacroSideEffect(
                    macro_name=name,
                    effect_type="mutation",
                    description="Macro body contains increment/decrement operators.",
                    severity="warning",
                    line_number=mdef.line_number,
                )
            )

        # 3. Assignment inside macro body
        if re.search(r"[^=!<>]=[^=]", body):
            effects.append(
                MacroSideEffect(
                    macro_name=name,
                    effect_type="assignment",
                    description="Macro body contains an assignment operator.",
                    severity="warning",
                    line_number=mdef.line_number,
                )
            )

        # 4. Statement expression (GCC extension)
        if re.search(r"\(\s*\{", body):
            effects.append(
                MacroSideEffect(
                    macro_name=name,
                    effect_type="statement_expression",
                    description="Macro uses GCC statement-expression extension ({...}).",
                    severity="warning",
                    line_number=mdef.line_number,
                )
            )

        # 5. Volatile access
        if re.search(r"\bvolatile\b", body):
            effects.append(
                MacroSideEffect(
                    macro_name=name,
                    effect_type="volatile_access",
                    description="Macro body references volatile memory.",
                    severity="warning",
                    line_number=mdef.line_number,
                )
            )

        # 6. Function call in body (potential side effect)
        if mdef.is_function_like and re.search(r"\b\w+\s*\(", body):
            callee_match = re.search(r"\b(\w+)\s*\(", body)
            if callee_match:
                callee = callee_match.group(1)
                if callee not in ("sizeof", "typeof", "__typeof__", "offsetof"):
                    effects.append(
                        MacroSideEffect(
                            macro_name=name,
                            effect_type="function_call",
                            description=f"Macro body calls function '{callee}'.",
                            severity="info",
                            line_number=mdef.line_number,
                        )
                    )

    return effects


def header_guard_analysis(c_headers: str) -> List[HeaderGuard]:
    """Analyse include guards and #pragma once directives."""
    guards: List[HeaderGuard] = []
    lines = c_headers.splitlines()

    # Detect #pragma once
    for i, line in enumerate(lines):
        if _PRAGMA_ONCE_RE.match(line):
            guards.append(
                HeaderGuard(
                    guard_symbol="",
                    file_name="",
                    style="pragma_once",
                    line_start=i + 1,
                    line_end=i + 1,
                    is_pragma_once=True,
                )
            )

    # Detect #ifndef / #define guard pairs
    ifndef_re = re.compile(r"^\s*#\s*ifndef\s+(\w+)\s*$")
    define_re = re.compile(r"^\s*#\s*define\s+(\w+)\s*$")
    endif_re = re.compile(r"^\s*#\s*endif")

    i = 0
    while i < len(lines):
        m_ifndef = ifndef_re.match(lines[i])
        if m_ifndef and i + 1 < len(lines):
            guard_sym = m_ifndef.group(1)
            m_define = define_re.match(lines[i + 1])
            if m_define and m_define.group(1) == guard_sym:
                # Find matching #endif
                depth = 1
                end_line = i + 2
                while end_line < len(lines) and depth > 0:
                    if re.match(r"^\s*#\s*(if|ifdef|ifndef)\b", lines[end_line]):
                        depth += 1
                    elif endif_re.match(lines[end_line]):
                        depth -= 1
                    end_line += 1
                file_hint = ""
                suffix_match = re.match(r"^_*(\w+?)_H_*$", guard_sym)
                if suffix_match:
                    file_hint = suffix_match.group(1).lower() + ".h"
                guards.append(
                    HeaderGuard(
                        guard_symbol=guard_sym,
                        file_name=file_hint,
                        style="ifndef",
                        line_start=i + 1,
                        line_end=end_line,
                        is_pragma_once=False,
                    )
                )
                i = end_line
                continue
        i += 1
    return guards


def expand_conditional_compilation(c_code: str) -> List[ConditionalBlock]:
    """Parse #if/#ifdef/#elif/#else/#endif chains into ConditionalBlock trees."""
    lines = c_code.splitlines()
    blocks: List[ConditionalBlock] = []
    stack: List[Tuple[ConditionalBlock, int]] = []

    for i, line in enumerate(lines):
        stripped = line.strip()

        # #if / #ifdef / #ifndef
        m_start = re.match(r"^#\s*(if|ifdef|ifndef)\s+(.*?)\s*$", stripped)
        if m_start:
            directive = m_start.group(1)
            condition = m_start.group(2)
            block = ConditionalBlock(
                condition=condition,
                directive=directive,
                body="",
                line_start=i + 1,
            )
            stack.append((block, i))
            continue

        # #elif
        m_elif = re.match(r"^#\s*elif\s+(.*?)\s*$", stripped)
        if m_elif and stack:
            parent, start = stack[-1]
            parent.body = "\n".join(lines[start + 1 : i])
            child = ConditionalBlock(
                condition=m_elif.group(1),
                directive="elif",
                body="",
                line_start=i + 1,
            )
            parent.children.append(child)
            stack[-1] = (child, i)
            continue

        # #else
        if re.match(r"^#\s*else\s*$", stripped) and stack:
            parent, start = stack[-1]
            parent.body = "\n".join(lines[start + 1 : i])
            else_block = ConditionalBlock(
                condition="else",
                directive="else",
                body="",
                line_start=i + 1,
            )
            parent.children.append(else_block)
            stack[-1] = (else_block, i)
            continue

        # #endif
        if re.match(r"^#\s*endif", stripped) and stack:
            current, start = stack.pop()
            current.body = current.body or "\n".join(lines[start + 1 : i])
            current.line_end = i + 1
            if not stack:
                blocks.append(current)
            continue

    return blocks


def convert_all_macros(c_code: str) -> str:
    """Convert all macros in a C source file to Rust equivalents."""
    macros = _parse_macro_defs(c_code)
    output_lines: List[str] = []

    # Collect conditional-compilation directives mapped to cfg attributes
    cond_blocks = expand_conditional_compilation(c_code)
    cfg_map = _build_cfg_map(cond_blocks)

    for name, mdef in macros.items():
        rust = macro_to_rust(mdef)
        if rust:
            output_lines.append(rust)
            output_lines.append("")

    # Non-macro code: expand macros then translate basic syntax
    body_code = re.sub(r"^\s*#.*$", "", c_code, flags=re.MULTILINE).strip()
    if body_code:
        expanded = _expand_single(body_code, macros, set(), 0, [], [])
        translated = _translate_expr(expanded)
        if translated.strip():
            output_lines.append("// --- Expanded code ---")
            output_lines.append(translated)

    # Append cfg attributes derived from conditional compilation
    if cfg_map:
        output_lines.append("")
        output_lines.append("// --- Conditional compilation (cfg) ---")
        for cond, cfg_attr in cfg_map.items():
            output_lines.append(f"// #if {cond}  =>  {cfg_attr}")

    return "\n".join(output_lines)


def _build_cfg_map(blocks: List[ConditionalBlock]) -> Dict[str, str]:
    """Map conditional-compilation conditions to Rust #[cfg(...)] strings."""
    result: Dict[str, str] = {}
    for block in blocks:
        cond = block.condition.strip()
        if block.directive == "ifdef":
            feature = cond.lower().strip("_")
            result[cond] = f'#[cfg(feature = "{feature}")]'
        elif block.directive == "ifndef":
            feature = cond.lower().strip("_")
            result[cond] = f'#[cfg(not(feature = "{feature}"))]'
        elif block.directive == "if":
            cfg_expr = _translate_if_condition(cond)
            result[cond] = f"#[cfg({cfg_expr})]"
        for child in block.children:
            if child.directive == "elif":
                cfg_expr = _translate_if_condition(child.condition)
                result[child.condition] = f"#[cfg({cfg_expr})]"
    return result


def _translate_if_condition(condition: str) -> str:
    """Translate a C preprocessor #if condition to a Rust cfg expression."""
    cond = condition.strip()

    # defined(X) -> feature = "x"
    cond = re.sub(
        r"defined\s*\(\s*(\w+)\s*\)",
        lambda m: f'feature = "{m.group(1).lower()}"',
        cond,
    )
    cond = re.sub(
        r"defined\s+(\w+)",
        lambda m: f'feature = "{m.group(1).lower()}"',
        cond,
    )

    # Logical operators
    cond = re.sub(r"\s*&&\s*", ", ", cond)
    cond = re.sub(r"\s*\|\|\s*", "), any(", cond)
    cond = re.sub(r"!\s*", "not(", cond)

    # Platform-specific heuristics
    cond = re.sub(r'feature = "_win32"', 'target_os = "windows"', cond)
    cond = re.sub(r'feature = "_win64"', 'target_os = "windows"', cond)
    cond = re.sub(r'feature = "__linux__"', 'target_os = "linux"', cond)
    cond = re.sub(r'feature = "__apple__"', 'target_os = "macos"', cond)
    cond = re.sub(r'feature = "__unix__"', 'target_family = "unix"', cond)

    return cond


def detect_unsafe_macros(c_code: str) -> List[MacroDef]:
    """Return macros whose Rust translation would require `unsafe`."""
    macros = _parse_macro_defs(c_code)
    unsafe_macros: List[MacroDef] = []

    unsafe_patterns = [
        (r"\bvolatile\b", "volatile memory access"),
        (r"\*\s*\(", "pointer dereference"),
        (r"\bmalloc\b|\bcalloc\b|\brealloc\b|\bfree\b", "manual memory management"),
        (r"\bmemcpy\b|\bmemset\b|\bmemmove\b", "raw memory operations"),
        (r"\basm\b|\b__asm\b|\b__asm__\b", "inline assembly"),
        (r"->", "pointer member access"),
        (r"\bunion\b", "union access"),
        (r"\bsetjmp\b|\blongjmp\b", "non-local jumps"),
        (r"\bsignal\b", "signal handling"),
        (r"\(\s*void\s*\*\s*\)", "void pointer cast"),
        (r"reinterpret_cast|static_cast", "explicit cast (C++ style)"),
    ]

    for name, mdef in macros.items():
        body = mdef.body
        for pattern, _reason in unsafe_patterns:
            if re.search(pattern, body):
                unsafe_macros.append(mdef)
                break

    return unsafe_macros


def generate_build_rs(c_code: str) -> str:
    """Generate a build.rs file for cfg-based conditional compilation."""
    cond_blocks = expand_conditional_compilation(c_code)
    features: Set[str] = set()
    platform_checks: List[str] = []

    for block in cond_blocks:
        _collect_features(block, features, platform_checks)

    lines: List[str] = [
        "// Auto-generated build.rs for conditional compilation migration",
        "// from C preprocessor directives.",
        "",
        "fn main() {",
    ]

    # Re-export Cargo features so #[cfg(feature = "...")] works
    if features:
        lines.append("    // Re-export features for conditional compilation")
        for feat in sorted(features):
            lines.append(f'    println!("cargo:rustc-cfg=feature=\\"{feat}\\"");')
        lines.append("")

    # Platform-based cfg
    platform_map = {
        "_WIN32": ("windows", 'target_os = "windows"'),
        "_WIN64": ("windows", 'target_os = "windows"'),
        "__linux__": ("linux", 'target_os = "linux"'),
        "__APPLE__": ("macos", 'target_os = "macos"'),
        "__unix__": ("unix", 'target_family = "unix"'),
        "__FreeBSD__": ("freebsd", 'target_os = "freebsd"'),
        "__ANDROID__": ("android", 'target_os = "android"'),
    }

    emitted_platforms: Set[str] = set()
    for check in platform_checks:
        if check in platform_map:
            label, _cfg = platform_map[check]
            if label not in emitted_platforms:
                lines.append(f"    // Platform: {check}")
                lines.append(f'    if cfg!(target_os = "{label}") {{')
                lines.append(f'        println!("cargo:rustc-cfg=feature=\\"{label}\\"");')
                lines.append("    }")
                emitted_platforms.add(label)

    # Environment variable probing
    if features:
        lines.append("")
        lines.append("    // Probe environment variables for feature flags")
        for feat in sorted(features):
            env_var = feat.upper()
            lines.append(f'    if std::env::var("{env_var}").is_ok() {{')
            lines.append(f'        println!("cargo:rustc-cfg=feature=\\"{feat}\\"");')
            lines.append("    }")

    lines.append("}")
    return "\n".join(lines)


def _collect_features(
    block: ConditionalBlock,
    features: Set[str],
    platform_checks: List[str],
) -> None:
    """Recursively collect feature names and platform symbols from condition blocks."""
    cond = block.condition.strip()

    platform_syms = {
        "_WIN32", "_WIN64", "__linux__", "__APPLE__", "__unix__",
        "__FreeBSD__", "__ANDROID__", "__MINGW32__", "__MINGW64__",
    }

    if block.directive in ("ifdef", "ifndef"):
        if cond in platform_syms:
            platform_checks.append(cond)
        else:
            features.add(cond.lower().strip("_"))
    elif block.directive == "if":
        defined_syms = re.findall(r"defined\s*\(?\s*(\w+)\s*\)?", cond)
        for sym in defined_syms:
            if sym in platform_syms:
                platform_checks.append(sym)
            else:
                features.add(sym.lower().strip("_"))

    for child in block.children:
        _collect_features(child, features, platform_checks)
