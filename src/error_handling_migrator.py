"""
Error handling migration module for C-to-Rust migration.

Analyzes C error handling patterns (errno, return codes, setjmp/longjmp,
goto-based cleanup, NULL checks) and generates idiomatic Rust equivalents
using Result<T, E>, custom error enums, and the ? operator.
"""

import re
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Dict, List, Optional, Set, Tuple


class ErrorPatternKind(Enum):
    ERRNO = auto()
    RETURN_CODE = auto()
    SETJMP = auto()
    NULL_CHECK = auto()
    GOTO_ERROR = auto()
    GLOBAL_ERROR = auto()
    CALLBACK_ERROR = auto()
    ASSERT_ABORT = auto()


@dataclass
class ErrorPattern:
    kind: ErrorPatternKind
    source_line: str
    line_number: int
    function_name: str
    description: str
    severity: str = "error"
    context_lines: List[str] = field(default_factory=list)

    def summary(self) -> str:
        return f"[{self.kind.name}] line {self.line_number} in {self.function_name}: {self.description}"


@dataclass
class ReturnCodePattern:
    function_name: str
    return_type: str
    error_value: str
    success_value: str
    check_expression: str
    line_number: int

    def is_negative_convention(self) -> bool:
        try:
            val = int(self.error_value)
            return val < 0
        except ValueError:
            return self.error_value.startswith("-")

    def is_null_convention(self) -> bool:
        return self.error_value.upper() in ("NULL", "0") and self.return_type.endswith("*")


@dataclass
class ErrnoUsage:
    errno_constant: str
    check_context: str
    line_number: int
    enclosing_function: str
    associated_call: str

    def to_rust_io_error_kind(self) -> str:
        mapping = {
            "ENOENT": "ErrorKind::NotFound",
            "EACCES": "ErrorKind::PermissionDenied",
            "EPERM": "ErrorKind::PermissionDenied",
            "EEXIST": "ErrorKind::AlreadyExists",
            "EINTR": "ErrorKind::Interrupted",
            "EAGAIN": "ErrorKind::WouldBlock",
            "EWOULDBLOCK": "ErrorKind::WouldBlock",
            "EPIPE": "ErrorKind::BrokenPipe",
            "ECONNREFUSED": "ErrorKind::ConnectionRefused",
            "ECONNRESET": "ErrorKind::ConnectionReset",
            "ECONNABORTED": "ErrorKind::ConnectionAborted",
            "ENOTCONN": "ErrorKind::NotConnected",
            "EADDRINUSE": "ErrorKind::AddrInUse",
            "EADDRNOTAVAIL": "ErrorKind::AddrNotAvailable",
            "ETIMEDOUT": "ErrorKind::TimedOut",
            "EINVAL": "ErrorKind::InvalidInput",
            "ENOMEM": "ErrorKind::OutOfMemory",
        }
        return mapping.get(self.errno_constant, "ErrorKind::Other")


@dataclass
class SetjmpUsage:
    setjmp_line: int
    longjmp_lines: List[int]
    jmp_buf_name: str
    enclosing_function: str
    error_values: List[str]

    def complexity_score(self) -> int:
        return len(self.longjmp_lines) * 2 + (1 if len(self.error_values) > 1 else 0)


@dataclass
class ErrorContext:
    function_name: str
    patterns: List[ErrorPattern]
    cleanup_labels: List[str]
    resources_allocated: List[str]
    return_type: str
    has_multiple_exit_points: bool

    def needs_drop_impl(self) -> bool:
        return len(self.resources_allocated) > 0 and len(self.cleanup_labels) > 0

    def dominant_pattern(self) -> Optional[ErrorPatternKind]:
        if not self.patterns:
            return None
        counts: Dict[ErrorPatternKind, int] = {}
        for p in self.patterns:
            counts[p.kind] = counts.get(p.kind, 0) + 1
        return max(counts, key=counts.get)


@dataclass
class ErrorPatternMap:
    patterns: List[ErrorPattern] = field(default_factory=list)
    errno_usages: List[ErrnoUsage] = field(default_factory=list)
    return_code_patterns: List[ReturnCodePattern] = field(default_factory=list)
    setjmp_usages: List[SetjmpUsage] = field(default_factory=list)
    contexts: List[ErrorContext] = field(default_factory=list)
    global_error_vars: List[str] = field(default_factory=list)

    def pattern_count(self) -> int:
        return len(self.patterns)

    def has_pattern_kind(self, kind: ErrorPatternKind) -> bool:
        return any(p.kind == kind for p in self.patterns)

    def patterns_by_function(self) -> Dict[str, List[ErrorPattern]]:
        result: Dict[str, List[ErrorPattern]] = {}
        for p in self.patterns:
            result.setdefault(p.function_name, []).append(p)
        return result

    def all_error_codes(self) -> Set[str]:
        codes: Set[str] = set()
        for eu in self.errno_usages:
            codes.add(eu.errno_constant)
        for rc in self.return_code_patterns:
            codes.add(rc.error_value)
        return codes


@dataclass
class RustErrorType:
    name: str
    variants: List[str]
    source_patterns: List[ErrorPatternKind]
    uses_io_error: bool = False
    uses_custom_message: bool = False
    derives: List[str] = field(default_factory=lambda: ["Debug", "Clone"])

    def to_enum_definition(self) -> str:
        lines = []
        derives_str = ", ".join(self.derives)
        lines.append(f"#[derive({derives_str})]")
        lines.append(f"pub enum {self.name} {{")
        for variant in self.variants:
            lines.append(f"    {variant},")
        if self.uses_io_error:
            lines.append("    Io(std::io::Error),")
        if self.uses_custom_message:
            lines.append("    Custom(String),")
        lines.append("}")
        return "\n".join(lines)


@dataclass
class ErrorEnumVariant:
    name: str
    value: Optional[int]
    description: str
    original_define: str

    def to_rust_variant(self) -> str:
        if self.value is not None:
            return f"    /// {self.description}\n    {self.name} = {self.value},"
        return f"    /// {self.description}\n    {self.name},"


@dataclass
class ErrorPropResult:
    all_paths_covered: bool
    missing_paths: List[str]
    extra_paths: List[str]
    c_error_paths: List[str]
    rust_error_paths: List[str]
    warnings: List[str]
    coverage_ratio: float

    def is_equivalent(self) -> bool:
        return self.all_paths_covered and len(self.extra_paths) == 0

    def summary(self) -> str:
        status = "PASS" if self.all_paths_covered else "FAIL"
        return (
            f"[{status}] Coverage: {self.coverage_ratio:.1%}, "
            f"Missing: {len(self.missing_paths)}, Extra: {len(self.extra_paths)}"
        )


@dataclass
class ErrorMigrationResult:
    original_c_code: str
    generated_rust_code: str
    error_types: List[RustErrorType]
    pattern_map: ErrorPatternMap
    propagation_result: Optional[ErrorPropResult]
    notes: List[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

_FUNC_RE = re.compile(
    r"(?:static\s+)?(?:inline\s+)?"
    r"([\w\s\*]+?)\s+(\w+)\s*\(([^)]*)\)\s*\{",
    re.MULTILINE,
)

_ERRNO_CHECK_RE = re.compile(
    r"\b(errno)\s*==\s*(\w+)|"
    r"\b(errno)\s*!=\s*0|"
    r"\bperror\s*\(",
    re.MULTILINE,
)

_ERRNO_SET_RE = re.compile(r"\berrno\s*=\s*(\w+);", re.MULTILINE)

_RETURN_CODE_RE = re.compile(
    r"if\s*\(\s*(\w+)\s*(==|!=|<|<=|>|>=)\s*(-?\d+|NULL)\s*\)",
    re.MULTILINE,
)

_GOTO_RE = re.compile(r"\bgoto\s+(\w+)\s*;", re.MULTILINE)

_LABEL_RE = re.compile(r"^(\w+)\s*:", re.MULTILINE)

_SETJMP_RE = re.compile(r"\bsetjmp\s*\(\s*(\w+)\s*\)", re.MULTILINE)
_LONGJMP_RE = re.compile(r"\blongjmp\s*\(\s*(\w+)\s*,\s*(\w+)\s*\)", re.MULTILINE)

_NULL_CHECK_RE = re.compile(
    r"if\s*\(\s*!?\s*(\w+)\s*(==|!=)\s*NULL\s*\)|"
    r"if\s*\(\s*(\w+)\s*\)|"
    r"if\s*\(\s*!\s*(\w+)\s*\)",
    re.MULTILINE,
)

_DEFINE_ERROR_RE = re.compile(
    r"#define\s+(E\w+|ERR_\w+|ERROR_\w+)\s+(-?\d+|0x[0-9a-fA-F]+)",
    re.MULTILINE,
)

_MALLOC_RE = re.compile(
    r"(\w+)\s*=\s*(?:malloc|calloc|realloc)\s*\(", re.MULTILINE
)

_GLOBAL_ERR_RE = re.compile(
    r"(?:static\s+)?(?:int|char\s*\*)\s+(g_err\w*|last_error\w*|error_code\w*)\s*[;=]",
    re.MULTILINE,
)

_ASSERT_RE = re.compile(r"\bassert\s*\((.+?)\)\s*;", re.MULTILINE)
_ABORT_RE = re.compile(r"\babort\s*\(\s*\)\s*;", re.MULTILINE)

_CALLBACK_ERR_RE = re.compile(
    r"typedef\s+\w+\s*\(\s*\*\s*(\w+error\w*|\w+err\w*)\s*\)",
    re.IGNORECASE | re.MULTILINE,
)


def _find_enclosing_function(c_code: str, position: int) -> str:
    best_name = "<global>"
    best_pos = -1
    for m in _FUNC_RE.finditer(c_code):
        if m.start() <= position and m.start() > best_pos:
            best_name = m.group(2)
            best_pos = m.start()
    return best_name


def _line_number_at(c_code: str, position: int) -> int:
    return c_code[:position].count("\n") + 1


def _extract_function_bodies(c_code: str) -> List[Tuple[str, str, str, int, int]]:
    results: List[Tuple[str, str, str, int, int]] = []
    for m in _FUNC_RE.finditer(c_code):
        ret_type = m.group(1).strip()
        name = m.group(2)
        start = m.end() - 1
        depth = 0
        idx = start
        while idx < len(c_code):
            if c_code[idx] == "{":
                depth += 1
            elif c_code[idx] == "}":
                depth -= 1
                if depth == 0:
                    body = c_code[start : idx + 1]
                    results.append((ret_type, name, body, m.start(), idx + 1))
                    break
            idx += 1
    return results


def _to_pascal_case(name: str) -> str:
    parts = re.split(r"[_\s]+", name)
    return "".join(p.capitalize() for p in parts if p)


def _sanitize_variant_name(raw: str) -> str:
    cleaned = re.sub(r"^(E_|ERR_|ERROR_)", "", raw, flags=re.IGNORECASE)
    return _to_pascal_case(cleaned)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def map_error_patterns(c_code: str) -> ErrorPatternMap:
    """Scan C source code and identify all error handling patterns."""
    result = ErrorPatternMap()
    lines = c_code.split("\n")

    # Errno checks
    for m in _ERRNO_CHECK_RE.finditer(c_code):
        ln = _line_number_at(c_code, m.start())
        func = _find_enclosing_function(c_code, m.start())
        constant = m.group(2) if m.group(2) else "nonzero"
        associated_call = ""
        line_text = lines[ln - 1] if ln <= len(lines) else ""
        prev_lines = lines[max(0, ln - 4) : ln - 1]
        for pl in reversed(prev_lines):
            call_m = re.search(r"(\w+)\s*\(", pl)
            if call_m and call_m.group(1) not in ("if", "while", "for", "switch"):
                associated_call = call_m.group(1)
                break
        eu = ErrnoUsage(
            errno_constant=constant,
            check_context=line_text.strip(),
            line_number=ln,
            enclosing_function=func,
            associated_call=associated_call,
        )
        result.errno_usages.append(eu)
        result.patterns.append(
            ErrorPattern(
                kind=ErrorPatternKind.ERRNO,
                source_line=line_text.strip(),
                line_number=ln,
                function_name=func,
                description=f"errno checked against {constant}",
            )
        )

    # Errno set patterns
    for m in _ERRNO_SET_RE.finditer(c_code):
        ln = _line_number_at(c_code, m.start())
        func = _find_enclosing_function(c_code, m.start())
        constant = m.group(1)
        line_text = lines[ln - 1] if ln <= len(lines) else ""
        eu = ErrnoUsage(
            errno_constant=constant,
            check_context=line_text.strip(),
            line_number=ln,
            enclosing_function=func,
            associated_call="(set)",
        )
        result.errno_usages.append(eu)

    # Return code checks
    for m in _RETURN_CODE_RE.finditer(c_code):
        ln = _line_number_at(c_code, m.start())
        func = _find_enclosing_function(c_code, m.start())
        var_name = m.group(1)
        operator = m.group(2)
        value = m.group(3)
        line_text = lines[ln - 1] if ln <= len(lines) else ""

        if value == "NULL":
            result.patterns.append(
                ErrorPattern(
                    kind=ErrorPatternKind.NULL_CHECK,
                    source_line=line_text.strip(),
                    line_number=ln,
                    function_name=func,
                    description=f"NULL check on {var_name}",
                )
            )
        else:
            error_val = value if operator in ("==", "<=", "<") else "0"
            success_val = "0" if error_val != "0" else value
            rcp = ReturnCodePattern(
                function_name=func,
                return_type="int",
                error_value=error_val,
                success_value=success_val,
                check_expression=line_text.strip(),
                line_number=ln,
            )
            result.return_code_patterns.append(rcp)
            result.patterns.append(
                ErrorPattern(
                    kind=ErrorPatternKind.RETURN_CODE,
                    source_line=line_text.strip(),
                    line_number=ln,
                    function_name=func,
                    description=f"return code check: {var_name} {operator} {value}",
                )
            )

    # NULL checks (pointer style)
    for m in _NULL_CHECK_RE.finditer(c_code):
        var = m.group(1) or m.group(3) or m.group(4)
        if not var:
            continue
        ln = _line_number_at(c_code, m.start())
        func = _find_enclosing_function(c_code, m.start())
        line_text = lines[ln - 1] if ln <= len(lines) else ""
        already = any(
            p.line_number == ln and p.kind == ErrorPatternKind.NULL_CHECK
            for p in result.patterns
        )
        if not already:
            result.patterns.append(
                ErrorPattern(
                    kind=ErrorPatternKind.NULL_CHECK,
                    source_line=line_text.strip(),
                    line_number=ln,
                    function_name=func,
                    description=f"NULL/validity check on {var}",
                )
            )

    # goto-based error handling
    goto_targets: Dict[str, List[int]] = {}
    for m in _GOTO_RE.finditer(c_code):
        label = m.group(1)
        ln = _line_number_at(c_code, m.start())
        goto_targets.setdefault(label, []).append(ln)

    error_labels = {
        lbl
        for lbl in goto_targets
        if re.search(r"err|fail|clean|out|done|exit|end", lbl, re.IGNORECASE)
    }
    for lbl in error_labels:
        for ln in goto_targets[lbl]:
            func = _find_enclosing_function(c_code, 0)
            for m2 in _FUNC_RE.finditer(c_code):
                pos_in_code = 0
                for i, line in enumerate(lines):
                    if i + 1 == ln:
                        pos_in_code = sum(len(l) + 1 for l in lines[:i])
                        break
                if m2.start() <= pos_in_code:
                    func = m2.group(2)
            line_text = lines[ln - 1] if ln <= len(lines) else ""
            result.patterns.append(
                ErrorPattern(
                    kind=ErrorPatternKind.GOTO_ERROR,
                    source_line=line_text.strip(),
                    line_number=ln,
                    function_name=func,
                    description=f"goto {lbl} (error/cleanup jump)",
                )
            )

    # setjmp / longjmp
    jmpbuf_to_setjmp: Dict[str, int] = {}
    for m in _SETJMP_RE.finditer(c_code):
        buf_name = m.group(1)
        ln = _line_number_at(c_code, m.start())
        jmpbuf_to_setjmp[buf_name] = ln

    longjmp_by_buf: Dict[str, List[Tuple[int, str]]] = {}
    for m in _LONGJMP_RE.finditer(c_code):
        buf_name = m.group(1)
        err_val = m.group(2)
        ln = _line_number_at(c_code, m.start())
        longjmp_by_buf.setdefault(buf_name, []).append((ln, err_val))

    for buf_name, setjmp_ln in jmpbuf_to_setjmp.items():
        func = _find_enclosing_function(c_code, 0)
        for m2 in _FUNC_RE.finditer(c_code):
            if _line_number_at(c_code, m2.start()) <= setjmp_ln:
                func = m2.group(2)
        longjmp_info = longjmp_by_buf.get(buf_name, [])
        sju = SetjmpUsage(
            setjmp_line=setjmp_ln,
            longjmp_lines=[info[0] for info in longjmp_info],
            jmp_buf_name=buf_name,
            enclosing_function=func,
            error_values=[info[1] for info in longjmp_info],
        )
        result.setjmp_usages.append(sju)
        line_text = lines[setjmp_ln - 1] if setjmp_ln <= len(lines) else ""
        result.patterns.append(
            ErrorPattern(
                kind=ErrorPatternKind.SETJMP,
                source_line=line_text.strip(),
                line_number=setjmp_ln,
                function_name=func,
                description=f"setjmp/longjmp with buffer {buf_name}",
            )
        )

    # Global error variables
    for m in _GLOBAL_ERR_RE.finditer(c_code):
        var_name = m.group(1)
        result.global_error_vars.append(var_name)
        ln = _line_number_at(c_code, m.start())
        result.patterns.append(
            ErrorPattern(
                kind=ErrorPatternKind.GLOBAL_ERROR,
                source_line=lines[ln - 1].strip() if ln <= len(lines) else "",
                line_number=ln,
                function_name=_find_enclosing_function(c_code, m.start()),
                description=f"global error variable: {var_name}",
            )
        )

    # Callback-based error handling
    for m in _CALLBACK_ERR_RE.finditer(c_code):
        cb_name = m.group(1)
        ln = _line_number_at(c_code, m.start())
        result.patterns.append(
            ErrorPattern(
                kind=ErrorPatternKind.CALLBACK_ERROR,
                source_line=lines[ln - 1].strip() if ln <= len(lines) else "",
                line_number=ln,
                function_name=_find_enclosing_function(c_code, m.start()),
                description=f"error callback typedef: {cb_name}",
            )
        )

    # assert / abort
    for m in _ASSERT_RE.finditer(c_code):
        ln = _line_number_at(c_code, m.start())
        func = _find_enclosing_function(c_code, m.start())
        result.patterns.append(
            ErrorPattern(
                kind=ErrorPatternKind.ASSERT_ABORT,
                source_line=lines[ln - 1].strip() if ln <= len(lines) else "",
                line_number=ln,
                function_name=func,
                description=f"assert({m.group(1).strip()})",
            )
        )
    for m in _ABORT_RE.finditer(c_code):
        ln = _line_number_at(c_code, m.start())
        func = _find_enclosing_function(c_code, m.start())
        result.patterns.append(
            ErrorPattern(
                kind=ErrorPatternKind.ASSERT_ABORT,
                source_line=lines[ln - 1].strip() if ln <= len(lines) else "",
                line_number=ln,
                function_name=func,
                description="abort() call",
            )
        )

    # Build ErrorContext per function
    func_bodies = _extract_function_bodies(c_code)
    for ret_type, fname, body, _start, _end in func_bodies:
        func_patterns = [p for p in result.patterns if p.function_name == fname]
        cleanup_labels = [
            lbl
            for lbl in error_labels
            if re.search(rf"\b{re.escape(lbl)}\s*:", body)
        ]
        resources = [m2.group(1) for m2 in _MALLOC_RE.finditer(body)]
        return_count = len(re.findall(r"\breturn\b", body))
        ctx = ErrorContext(
            function_name=fname,
            patterns=func_patterns,
            cleanup_labels=cleanup_labels,
            resources_allocated=resources,
            return_type=ret_type,
            has_multiple_exit_points=return_count > 1,
        )
        result.contexts.append(ctx)

    return result


def suggest_rust_error_types(c_errors: ErrorPatternMap) -> List[RustErrorType]:
    """Suggest Rust error types based on detected C error patterns."""
    suggestions: List[RustErrorType] = []
    uses_errno = c_errors.has_pattern_kind(ErrorPatternKind.ERRNO)
    uses_null = c_errors.has_pattern_kind(ErrorPatternKind.NULL_CHECK)
    uses_return_code = c_errors.has_pattern_kind(ErrorPatternKind.RETURN_CODE)
    uses_setjmp = c_errors.has_pattern_kind(ErrorPatternKind.SETJMP)

    if uses_errno:
        errno_constants = {eu.errno_constant for eu in c_errors.errno_usages}
        variants = []
        for const in sorted(errno_constants):
            if const == "nonzero":
                variants.append("Unknown")
            else:
                variants.append(_sanitize_variant_name(const))
        suggestions.append(
            RustErrorType(
                name="IoError",
                variants=variants,
                source_patterns=[ErrorPatternKind.ERRNO],
                uses_io_error=True,
                derives=["Debug"],
            )
        )

    if uses_return_code:
        rc_variants: List[str] = []
        seen_values: Set[str] = set()
        for rcp in c_errors.return_code_patterns:
            val = rcp.error_value
            if val not in seen_values:
                seen_values.add(val)
                rc_variants.append(f"Code{val.replace('-', 'Neg')}")
        if not rc_variants:
            rc_variants = ["GenericFailure"]
        suggestions.append(
            RustErrorType(
                name="ReturnCodeError",
                variants=rc_variants,
                source_patterns=[ErrorPatternKind.RETURN_CODE],
                uses_custom_message=True,
                derives=["Debug", "Clone", "PartialEq"],
            )
        )

    if uses_null:
        suggestions.append(
            RustErrorType(
                name="NullPointerError",
                variants=["AllocationFailed", "InvalidReference", "UninitializedPointer"],
                source_patterns=[ErrorPatternKind.NULL_CHECK],
                derives=["Debug", "Clone"],
            )
        )

    if uses_setjmp:
        setjmp_variants = []
        for sju in c_errors.setjmp_usages:
            for ev in sju.error_values:
                vname = f"Jump{_to_pascal_case(ev)}" if not ev.isdigit() else f"JumpCode{ev}"
                if vname not in setjmp_variants:
                    setjmp_variants.append(vname)
        if not setjmp_variants:
            setjmp_variants = ["NonLocalReturn"]
        suggestions.append(
            RustErrorType(
                name="NonLocalError",
                variants=setjmp_variants,
                source_patterns=[ErrorPatternKind.SETJMP],
                uses_custom_message=True,
                derives=["Debug"],
            )
        )

    if c_errors.has_pattern_kind(ErrorPatternKind.GLOBAL_ERROR):
        suggestions.append(
            RustErrorType(
                name="GlobalStateError",
                variants=["InvalidState", "Uninitialized"],
                source_patterns=[ErrorPatternKind.GLOBAL_ERROR],
                uses_custom_message=True,
                derives=["Debug", "Clone"],
            )
        )

    if c_errors.has_pattern_kind(ErrorPatternKind.ASSERT_ABORT):
        suggestions.append(
            RustErrorType(
                name="InvariantViolation",
                variants=["AssertionFailed", "Aborted"],
                source_patterns=[ErrorPatternKind.ASSERT_ABORT],
                uses_custom_message=True,
                derives=["Debug"],
            )
        )

    # If no specific patterns found, provide a generic error type
    if not suggestions:
        suggestions.append(
            RustErrorType(
                name="AppError",
                variants=["Unknown"],
                source_patterns=[],
                uses_custom_message=True,
                derives=["Debug"],
            )
        )

    return suggestions


def generate_error_enum(c_error_codes: List[str]) -> str:
    """Convert a list of C #define error code names into a Rust enum with Display and Error impls."""
    variants: List[ErrorEnumVariant] = []
    for code in c_error_codes:
        m = re.match(r"^(\w+)\s*=\s*(-?\d+|0x[0-9a-fA-F]+)$", code.strip())
        if m:
            name_raw = m.group(1)
            val_str = m.group(2)
            value = int(val_str, 0)
            variant_name = _sanitize_variant_name(name_raw)
            desc = f"Corresponds to C define {name_raw} ({val_str})"
            variants.append(ErrorEnumVariant(name=variant_name, value=value, description=desc, original_define=code))
        else:
            name_raw = code.strip()
            variant_name = _sanitize_variant_name(name_raw)
            desc = f"Corresponds to C define {name_raw}"
            variants.append(ErrorEnumVariant(name=variant_name, value=None, description=desc, original_define=code))

    if not variants:
        return "// No error codes provided\n"

    enum_name = "ErrorCode"
    lines = [
        "use std::fmt;",
        "use std::error::Error;",
        "",
        "#[derive(Debug, Clone, Copy, PartialEq, Eq)]",
        f"pub enum {enum_name} {{",
    ]
    for v in variants:
        lines.append(v.to_rust_variant())
    lines.append("}")
    lines.append("")

    # Display impl
    lines.append(f"impl fmt::Display for {enum_name} {{")
    lines.append("    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {")
    lines.append("        match self {")
    for v in variants:
        lines.append(f'            {enum_name}::{v.name} => write!(f, "{v.description}"),')
    lines.append("        }")
    lines.append("    }")
    lines.append("}")
    lines.append("")

    # Error impl
    lines.append(f"impl Error for {enum_name} {{}}")
    lines.append("")

    # From<i32> impl for numeric variants
    numeric = [v for v in variants if v.value is not None]
    if numeric:
        lines.append(f"impl TryFrom<i32> for {enum_name} {{")
        lines.append("    type Error = i32;")
        lines.append("")
        lines.append(f"    fn try_from(value: i32) -> Result<Self, Self::Error> {{")
        lines.append("        match value {")
        for v in numeric:
            lines.append(f"            {v.value} => Ok({enum_name}::{v.name}),")
        lines.append("            other => Err(other),")
        lines.append("        }")
        lines.append("    }")
        lines.append("}")

    return "\n".join(lines) + "\n"


def detect_error_code_conventions(c_code: str) -> Dict[str, str]:
    """Detect error coding conventions used in C source code."""
    conventions: Dict[str, str] = {}

    # Check for negative return values as errors
    neg_returns = re.findall(r"return\s+(-\d+)\s*;", c_code)
    if neg_returns:
        conventions["negative_return"] = (
            f"Functions return negative values on error: {', '.join(sorted(set(neg_returns)))}"
        )

    # Check for NULL returns
    null_returns = re.findall(r"return\s+NULL\s*;", c_code)
    if null_returns:
        conventions["null_return"] = f"Functions return NULL on error ({len(null_returns)} occurrences)"

    # Check for 0/-1 convention
    zero_success = len(re.findall(r"return\s+0\s*;", c_code))
    neg1_error = len(re.findall(r"return\s+-1\s*;", c_code))
    if zero_success > 0 and neg1_error > 0:
        conventions["zero_success_neg1_error"] = (
            f"Uses 0 for success ({zero_success}x) and -1 for error ({neg1_error}x)"
        )

    # Check for boolean-style (1 success, 0 failure)
    return_1 = len(re.findall(r"return\s+1\s*;", c_code))
    return_0 = zero_success
    if return_1 > 0 and return_0 > 0:
        conventions["boolean_return"] = (
            f"Possible boolean convention: return 1 ({return_1}x), return 0 ({return_0}x)"
        )

    # Check for errno usage
    errno_count = len(re.findall(r"\berrno\b", c_code))
    if errno_count > 0:
        conventions["errno"] = f"Uses errno for error reporting ({errno_count} references)"

    # Check for error output parameters
    err_params = re.findall(r"\w+\s*\([^)]*(?:int\s*\*\s*err|char\s*\*\s*err\w*)[^)]*\)", c_code)
    if err_params:
        conventions["error_out_param"] = (
            f"Uses output parameters for errors ({len(err_params)} functions)"
        )

    # Check for perror / strerror usage
    perror_count = len(re.findall(r"\bperror\s*\(", c_code))
    strerror_count = len(re.findall(r"\bstrerror\s*\(", c_code))
    if perror_count or strerror_count:
        conventions["posix_error_printing"] = (
            f"perror: {perror_count}x, strerror: {strerror_count}x"
        )

    # Check for custom error macros
    err_macros = re.findall(
        r"#define\s+(CHECK|ASSERT|VERIFY|ENSURE|REQUIRE|RETURN_IF_ERROR|BAIL)\w*\s*\(",
        c_code, re.IGNORECASE,
    )
    if err_macros:
        conventions["custom_error_macros"] = (
            f"Custom error macros: {', '.join(sorted(set(err_macros)))}"
        )

    # goto error pattern
    goto_error_count = len(
        re.findall(r"\bgoto\s+(?:err|error|fail|cleanup|out|done)\w*\s*;", c_code, re.IGNORECASE)
    )
    if goto_error_count > 0:
        conventions["goto_cleanup"] = (
            f"Uses goto for error cleanup ({goto_error_count} occurrences)"
        )

    # Check for fprintf(stderr, ...)
    stderr_prints = len(re.findall(r"fprintf\s*\(\s*stderr\s*,", c_code))
    if stderr_prints > 0:
        conventions["stderr_logging"] = f"Logs errors to stderr ({stderr_prints} occurrences)"

    return conventions


def map_errno_to_rust(c_code: str) -> str:
    """Map errno-based error handling in C to Rust std::io::Error usage."""
    output_lines = [
        "use std::io::{self, Error, ErrorKind};",
        "",
    ]

    pattern_map = map_error_patterns(c_code)
    func_bodies = _extract_function_bodies(c_code)

    if not pattern_map.errno_usages and not func_bodies:
        output_lines.append("// No errno usage detected in provided C code.")
        return "\n".join(output_lines) + "\n"

    errno_functions: Dict[str, List[ErrnoUsage]] = {}
    for eu in pattern_map.errno_usages:
        errno_functions.setdefault(eu.enclosing_function, []).append(eu)

    for ret_type, fname, body, _s, _e in func_bodies:
        if fname not in errno_functions:
            continue
        usages = errno_functions[fname]

        # Parse parameters
        param_m = re.search(r"\(([^)]*)\)", c_code[_s:_e])
        params_raw = param_m.group(1).strip() if param_m else ""
        rust_params = _convert_params_to_rust(params_raw)
        rust_ret = _c_return_to_rust_io(ret_type)

        output_lines.append(f"pub fn {fname}({rust_params}) -> io::Result<{rust_ret}> {{")

        # Convert body lines
        body_lines = body.strip("{}").strip().split("\n")
        for bline in body_lines:
            stripped = bline.strip()
            if not stripped or stripped == "{" or stripped == "}":
                continue

            # Convert errno checks to Rust
            errno_check = re.match(
                r"if\s*\(\s*errno\s*==\s*(\w+)\s*\)\s*\{?", stripped
            )
            if errno_check:
                const = errno_check.group(1)
                eu_temp = ErrnoUsage(
                    errno_constant=const, check_context="", line_number=0,
                    enclosing_function=fname, associated_call="",
                )
                kind = eu_temp.to_rust_io_error_kind()
                output_lines.append(
                    f'    return Err(Error::new({kind}, "{const}"));'
                )
                continue

            # Convert function calls that check errno
            call_check = re.match(
                r"(\w+)\s*=\s*(\w+)\s*\(([^)]*)\)\s*;", stripped
            )
            if call_check:
                var = call_check.group(1)
                callee = call_check.group(2)
                args = call_check.group(3)
                if callee in ("open", "read", "write", "close", "fopen", "fread", "fwrite"):
                    output_lines.append(f"    // Migrated from: {stripped}")
                    output_lines.append(f"    let {var} = /* {callee}({args}) mapped to Rust */;")
                else:
                    output_lines.append(f"    {_translate_statement_to_rust(stripped)}")
                continue

            # Convert return with errno
            ret_errno = re.match(r"return\s+(-?\d+)\s*;", stripped)
            if ret_errno:
                val = int(ret_errno.group(1))
                if val < 0 or val == 0:
                    if val < 0:
                        output_lines.append(
                            '    return Err(Error::last_os_error());'
                        )
                    else:
                        if rust_ret == "()":
                            output_lines.append("    Ok(())")
                        else:
                            output_lines.append(f"    Ok({val})")
                else:
                    output_lines.append(f"    Ok({val})")
                continue

            output_lines.append(f"    {_translate_statement_to_rust(stripped)}")

        output_lines.append("}")
        output_lines.append("")

    return "\n".join(output_lines) + "\n"


def convert_goto_error(c_code: str) -> str:
    """Convert goto-based cleanup patterns to Rust using ? operator and Drop."""
    output_lines: List[str] = []
    func_bodies = _extract_function_bodies(c_code)

    if not func_bodies:
        return "// No functions found in provided C code.\n"

    for ret_type, fname, body, _s, _e in func_bodies:
        goto_labels = list(re.finditer(r"^(\w+)\s*:", body, re.MULTILINE))
        gotos = list(_GOTO_RE.finditer(body))
        error_gotos = [
            g for g in gotos
            if re.search(r"err|fail|clean|out|done|exit|end", g.group(1), re.IGNORECASE)
        ]

        if not error_gotos:
            output_lines.append(f"// {fname}: no goto-error pattern detected, skipping")
            output_lines.append("")
            continue

        # Detect resources allocated before goto
        resources = [m.group(1) for m in _MALLOC_RE.finditer(body)]

        # Generate a resource guard struct if needed
        if resources:
            guard_name = f"{_to_pascal_case(fname)}Guard"
            output_lines.append(f"struct {guard_name} {{")
            for res in resources:
                output_lines.append(f"    {res}: Option<*mut u8>,")
            output_lines.append("}")
            output_lines.append("")
            output_lines.append(f"impl Drop for {guard_name} {{")
            output_lines.append("    fn drop(&mut self) {")
            for res in resources:
                output_lines.append(f"        if let Some(ptr) = self.{res} {{")
                output_lines.append(f"            // SAFETY: freeing resource {res}")
                output_lines.append(f"            unsafe {{ libc::free(ptr as *mut libc::c_void); }}")
                output_lines.append("        }")
            output_lines.append("    }")
            output_lines.append("}")
            output_lines.append("")

        # Extract cleanup block from the label
        cleanup_code_lines: List[str] = []
        for lbl_match in goto_labels:
            lbl_name = lbl_match.group(1)
            if not re.search(r"err|fail|clean|out|done|exit|end", lbl_name, re.IGNORECASE):
                continue
            lbl_pos = lbl_match.end()
            remaining = body[lbl_pos:]
            for cl in remaining.split("\n"):
                cl_s = cl.strip()
                if cl_s == "}" or cl_s.startswith("return"):
                    if cl_s.startswith("return"):
                        cleanup_code_lines.append(cl_s)
                    break
                if cl_s and cl_s != "{":
                    cleanup_code_lines.append(cl_s)

        # Build converted function
        param_m = re.search(r"\(([^)]*)\)", c_code[_s:_e])
        params_raw = param_m.group(1).strip() if param_m else ""
        rust_params = _convert_params_to_rust(params_raw)
        rust_ret = _c_return_to_rust_result(ret_type)

        output_lines.append(f"pub fn {fname}({rust_params}) -> Result<{rust_ret}, Box<dyn std::error::Error>> {{")

        if resources:
            guard_name = f"{_to_pascal_case(fname)}Guard"
            output_lines.append(f"    let _guard = {guard_name} {{")
            for res in resources:
                output_lines.append(f"        {res}: None,")
            output_lines.append("    };")

        body_lines = body.strip("{}").strip().split("\n")
        skip_until_label = False
        inside_cleanup = False

        for bline in body_lines:
            stripped = bline.strip()
            if not stripped or stripped == "{" or stripped == "}":
                continue

            # Skip label definitions and their cleanup blocks
            label_match = re.match(r"^(\w+)\s*:", stripped)
            if label_match:
                lbl = label_match.group(1)
                if lbl in {g.group(1) for g in error_gotos}:
                    inside_cleanup = True
                    output_lines.append(f"    // cleanup from label '{lbl}' handled by Drop + ?")
                    continue

            if inside_cleanup:
                if stripped.startswith("return"):
                    inside_cleanup = False
                continue

            # Convert goto to ? operator
            goto_m = re.match(r"if\s*\(.+?\)\s*\{?\s*goto\s+(\w+)\s*;", stripped)
            if goto_m:
                condition = re.search(r"if\s*\((.+?)\)", stripped)
                cond_text = condition.group(1) if condition else "error"
                rust_cond = _translate_condition_to_rust(cond_text)
                output_lines.append(f"    if {rust_cond} {{")
                output_lines.append(f'        return Err("{goto_m.group(1)}".into());')
                output_lines.append("    }")
                continue

            # Convert simple goto
            simple_goto = re.match(r"goto\s+(\w+)\s*;", stripped)
            if simple_goto:
                output_lines.append(f'    return Err("{simple_goto.group(1)}".into());')
                continue

            output_lines.append(f"    {_translate_statement_to_rust(stripped)}")

        if rust_ret == "()":
            output_lines.append("    Ok(())")
        else:
            output_lines.append(f"    Ok(Default::default())")
        output_lines.append("}")
        output_lines.append("")

    return "\n".join(output_lines) + "\n"


def migrate_error_handling(c_code: str) -> str:
    """Generate Rust code with proper Result<T,E> error handling from C source."""
    pattern_map = map_error_patterns(c_code)
    error_types = suggest_rust_error_types(pattern_map)
    conventions = detect_error_code_conventions(c_code)

    output_sections: List[str] = []

    # Header
    output_sections.append("// Auto-generated Rust error handling migration")
    output_sections.append("use std::fmt;")
    output_sections.append("use std::error::Error;")
    if pattern_map.has_pattern_kind(ErrorPatternKind.ERRNO):
        output_sections.append("use std::io::{self, ErrorKind};")
    output_sections.append("")

    # Generate error type definitions
    for et in error_types:
        output_sections.append(et.to_enum_definition())
        output_sections.append("")

        # Display impl
        output_sections.append(f"impl fmt::Display for {et.name} {{")
        output_sections.append("    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {")
        output_sections.append("        match self {")
        for v in et.variants:
            output_sections.append(f'            {et.name}::{v} => write!(f, "{et.name}::{v}"),')
        if et.uses_io_error:
            output_sections.append(f'            {et.name}::Io(e) => write!(f, "IO error: {{e}}"),')
        if et.uses_custom_message:
            output_sections.append(f'            {et.name}::Custom(msg) => write!(f, "{{msg}}"),')
        output_sections.append("        }")
        output_sections.append("    }")
        output_sections.append("}")
        output_sections.append("")
        output_sections.append(f"impl Error for {et.name} {{}}")
        output_sections.append("")

    # Generate From impls for io::Error if needed
    for et in error_types:
        if et.uses_io_error:
            output_sections.append(f"impl From<io::Error> for {et.name} {{")
            output_sections.append(f"    fn from(err: io::Error) -> Self {{")
            output_sections.append(f"        {et.name}::Io(err)")
            output_sections.append("    }")
            output_sections.append("}")
            output_sections.append("")

    # Generate migrated functions
    func_bodies = _extract_function_bodies(c_code)
    primary_error = error_types[0].name if error_types else "AppError"

    for ret_type, fname, body, _s, _e in func_bodies:
        ctx = next((c for c in pattern_map.contexts if c.function_name == fname), None)
        param_m = re.search(r"\(([^)]*)\)", c_code[_s:_e])
        params_raw = param_m.group(1).strip() if param_m else ""
        rust_params = _convert_params_to_rust(params_raw)
        rust_ret = _c_return_to_rust_result(ret_type)

        has_errors = ctx is not None and len(ctx.patterns) > 0
        if has_errors:
            output_sections.append(
                f"pub fn {fname}({rust_params}) -> Result<{rust_ret}, {primary_error}> {{"
            )
        else:
            output_sections.append(f"pub fn {fname}({rust_params}) -> {rust_ret} {{")

        body_lines = body.strip("{}").strip().split("\n")
        inside_cleanup_label = False

        for bline in body_lines:
            stripped = bline.strip()
            if not stripped or stripped in ("{", "}"):
                continue

            # Skip cleanup label blocks
            label_m = re.match(r"^(\w+)\s*:", stripped)
            if label_m and re.search(
                r"err|fail|clean|out|done|exit|end", label_m.group(1), re.IGNORECASE
            ):
                inside_cleanup_label = True
                output_sections.append(f"    // cleanup label '{label_m.group(1)}' converted to Drop/? pattern")
                continue
            if inside_cleanup_label:
                if stripped.startswith("return"):
                    inside_cleanup_label = False
                continue

            # Convert errno checks
            errno_m = re.match(r"if\s*\(\s*errno\s*==\s*(\w+)\s*\)", stripped)
            if errno_m:
                const = errno_m.group(1)
                eu_temp = ErrnoUsage(
                    errno_constant=const, check_context="", line_number=0,
                    enclosing_function=fname, associated_call="",
                )
                kind = eu_temp.to_rust_io_error_kind()
                output_sections.append(
                    f"    // errno == {const} -> io::Error"
                )
                output_sections.append(
                    f'    return Err({primary_error}::Io(io::Error::new({kind}, "{const}")));'
                )
                continue

            # Convert goto
            goto_m = re.match(r"goto\s+(\w+)\s*;", stripped)
            if goto_m:
                output_sections.append(
                    f'    return Err({primary_error}::Custom("{goto_m.group(1)}".to_string()));'
                )
                continue

            # Convert return with error value
            ret_m = re.match(r"return\s+(-\d+)\s*;", stripped)
            if ret_m and has_errors:
                output_sections.append(
                    f'    return Err({primary_error}::Custom("error code {ret_m.group(1)}".to_string()));'
                )
                continue

            ret_null = re.match(r"return\s+NULL\s*;", stripped)
            if ret_null and has_errors:
                output_sections.append(
                    f'    return Err({primary_error}::Custom("null return".to_string()));'
                )
                continue

            ret_ok = re.match(r"return\s+(.+?)\s*;", stripped)
            if ret_ok and has_errors:
                val = ret_ok.group(1).strip()
                if val == "0" and rust_ret == "()":
                    output_sections.append("    return Ok(());")
                else:
                    output_sections.append(f"    return Ok({_translate_expr_to_rust(val)});")
                continue

            # Convert assert
            assert_m = re.match(r"assert\s*\((.+?)\)\s*;", stripped)
            if assert_m:
                cond = _translate_condition_to_rust(assert_m.group(1))
                output_sections.append(f"    debug_assert!({cond});")
                continue

            output_sections.append(f"    {_translate_statement_to_rust(stripped)}")

        if has_errors:
            if rust_ret == "()":
                output_sections.append("    Ok(())")
            else:
                output_sections.append("    Ok(Default::default())")
        output_sections.append("}")
        output_sections.append("")

    return "\n".join(output_sections) + "\n"


def verify_error_propagation(c_code: str, rust_code: str) -> ErrorPropResult:
    """Verify that error paths in C code are preserved in the generated Rust code."""
    c_patterns = map_error_patterns(c_code)

    # Extract C error paths
    c_error_paths: List[str] = []
    for p in c_patterns.patterns:
        path_id = f"{p.function_name}:{p.kind.name}:L{p.line_number}"
        c_error_paths.append(path_id)

    # Extract C return-error sites
    c_error_returns: List[str] = []
    for m in re.finditer(r"return\s+(-\d+|NULL)\s*;", c_code):
        ln = _line_number_at(c_code, m.start())
        func = _find_enclosing_function(c_code, m.start())
        c_error_returns.append(f"{func}:ERROR_RETURN:L{ln}")
    c_error_paths.extend(c_error_returns)

    # Analyze Rust code for error handling sites
    rust_error_paths: List[str] = []
    rust_lines = rust_code.split("\n")

    # Find Result return types
    for i, line in enumerate(rust_lines):
        fn_m = re.search(r"pub\s+fn\s+(\w+)\s*\(.*\)\s*->\s*Result", line)
        if fn_m:
            fname = fn_m.group(1)
            rust_error_paths.append(f"{fname}:RESULT_TYPE:L{i + 1}")

    # Find Err() returns
    for i, line in enumerate(rust_lines):
        err_m = re.search(r"Err\s*\(", line)
        if err_m:
            func = "<unknown>"
            for j in range(i, -1, -1):
                fn_line = re.search(r"pub\s+fn\s+(\w+)", rust_lines[j])
                if fn_line:
                    func = fn_line.group(1)
                    break
            rust_error_paths.append(f"{func}:ERR_RETURN:L{i + 1}")

    # Find ? operator usage
    for i, line in enumerate(rust_lines):
        if re.search(r"\?\s*;|\?\s*$", line):
            func = "<unknown>"
            for j in range(i, -1, -1):
                fn_line = re.search(r"pub\s+fn\s+(\w+)", rust_lines[j])
                if fn_line:
                    func = fn_line.group(1)
                    break
            rust_error_paths.append(f"{func}:QUESTION_OP:L{i + 1}")

    # Compare function-level coverage
    c_funcs_with_errors = {p.split(":")[0] for p in c_error_paths}
    rust_funcs_with_errors = {p.split(":")[0] for p in rust_error_paths if p.split(":")[0] != "<unknown>"}

    missing_paths: List[str] = []
    for cf in c_funcs_with_errors:
        if cf not in rust_funcs_with_errors and cf != "<global>":
            missing_paths.append(f"{cf}: error handling not found in Rust")

    extra_paths: List[str] = []
    for rf in rust_funcs_with_errors:
        if rf not in c_funcs_with_errors:
            extra_paths.append(f"{rf}: error handling added in Rust (no C equivalent)")

    # Coverage ratio
    if c_funcs_with_errors:
        covered = len(c_funcs_with_errors & rust_funcs_with_errors)
        coverage_ratio = covered / len(c_funcs_with_errors)
    else:
        coverage_ratio = 1.0

    # Warnings
    warnings: List[str] = []
    if c_patterns.has_pattern_kind(ErrorPatternKind.SETJMP):
        if "panic" not in rust_code.lower() and "catch_unwind" not in rust_code:
            warnings.append("setjmp/longjmp detected in C but no panic/catch_unwind in Rust")
    if c_patterns.has_pattern_kind(ErrorPatternKind.GLOBAL_ERROR):
        if "static" not in rust_code and "thread_local" not in rust_code:
            warnings.append("Global error state in C but no static/thread_local in Rust")
    if c_patterns.has_pattern_kind(ErrorPatternKind.GOTO_ERROR):
        if "Drop" not in rust_code and "?" not in rust_code:
            warnings.append("goto-based cleanup in C but no Drop/? in Rust")

    all_covered = len(missing_paths) == 0

    return ErrorPropResult(
        all_paths_covered=all_covered,
        missing_paths=missing_paths,
        extra_paths=extra_paths,
        c_error_paths=c_error_paths,
        rust_error_paths=rust_error_paths,
        warnings=warnings,
        coverage_ratio=coverage_ratio,
    )


# ---------------------------------------------------------------------------
# Internal translation helpers
# ---------------------------------------------------------------------------


def _convert_params_to_rust(c_params: str) -> str:
    if not c_params or c_params == "void":
        return ""
    parts = [p.strip() for p in c_params.split(",") if p.strip()]
    rust_parts: List[str] = []
    for part in parts:
        tokens = part.split()
        if len(tokens) < 2:
            rust_parts.append(f"{part}: /* unknown */")
            continue
        name = tokens[-1].lstrip("*")
        c_type = " ".join(tokens[:-1])
        is_ptr = "*" in part
        rust_type = _c_type_to_rust(c_type, is_ptr)
        rust_parts.append(f"{name}: {rust_type}")
    return ", ".join(rust_parts)


def _c_type_to_rust(c_type: str, is_pointer: bool = False) -> str:
    c_type_clean = c_type.strip().rstrip("*").strip()
    type_map = {
        "int": "i32",
        "unsigned int": "u32",
        "unsigned": "u32",
        "long": "i64",
        "unsigned long": "u64",
        "long long": "i64",
        "short": "i16",
        "unsigned short": "u16",
        "char": "i8",
        "unsigned char": "u8",
        "float": "f32",
        "double": "f64",
        "size_t": "usize",
        "ssize_t": "isize",
        "void": "()",
        "bool": "bool",
        "_Bool": "bool",
    }
    rust_type = type_map.get(c_type_clean, c_type_clean)
    if is_pointer:
        if c_type_clean == "char":
            return "&str"
        if c_type_clean == "void":
            return "*mut u8"
        return f"&mut {rust_type}"
    return rust_type


def _c_return_to_rust_io(c_ret: str) -> str:
    c_ret_clean = c_ret.strip()
    if c_ret_clean in ("void", ""):
        return "()"
    if "*" in c_ret_clean:
        return "Vec<u8>"
    type_map = {
        "int": "i32",
        "long": "i64",
        "size_t": "usize",
        "ssize_t": "isize",
        "char": "i8",
    }
    return type_map.get(c_ret_clean, c_ret_clean)


def _c_return_to_rust_result(c_ret: str) -> str:
    c_ret_clean = c_ret.strip()
    if c_ret_clean in ("void", "int", ""):
        return "()"
    if "*" in c_ret_clean:
        return "Vec<u8>"
    return _c_type_to_rust(c_ret_clean)


def _translate_statement_to_rust(c_stmt: str) -> str:
    stmt = c_stmt.rstrip(";").strip()

    # Variable declarations with initialization
    decl_m = re.match(r"([\w\s\*]+?)\s+(\w+)\s*=\s*(.+)", stmt)
    if decl_m:
        c_type_raw = decl_m.group(1).strip()
        var_name = decl_m.group(2)
        init_val = decl_m.group(3).strip().rstrip(";")
        rust_val = _translate_expr_to_rust(init_val)
        return f"let mut {var_name} = {rust_val};"

    # Simple variable declarations
    simple_decl = re.match(r"([\w\s\*]+?)\s+(\w+)\s*;?\s*$", stmt)
    if simple_decl:
        c_type_raw = simple_decl.group(1).strip()
        var_name = simple_decl.group(2)
        is_ptr = "*" in c_type_raw
        rust_type = _c_type_to_rust(c_type_raw, is_ptr)
        return f"let mut {var_name}: {rust_type};"

    # If statements
    if_m = re.match(r"if\s*\((.+?)\)\s*\{?", stmt)
    if if_m:
        cond = _translate_condition_to_rust(if_m.group(1))
        return f"if {cond} {{"

    # Else
    if stmt == "else" or stmt == "} else {":
        return "} else {"

    # For loops
    for_m = re.match(r"for\s*\(\s*(.+?);\s*(.+?);\s*(.+?)\s*\)", stmt)
    if for_m:
        init = for_m.group(1).strip()
        cond = for_m.group(2).strip()
        inc = for_m.group(3).strip()
        return f"// for ({init}; {cond}; {inc})"

    # While loops
    while_m = re.match(r"while\s*\((.+?)\)", stmt)
    if while_m:
        cond = _translate_condition_to_rust(while_m.group(1))
        return f"while {cond} {{"

    # Function calls
    call_m = re.match(r"(\w+)\s*\((.+)\)", stmt)
    if call_m:
        fn_name = call_m.group(1)
        args = call_m.group(2)
        if fn_name == "printf":
            fmt_str = args.split(",")[0].strip().strip('"')
            rest_args = ",".join(args.split(",")[1:]).strip()
            if rest_args:
                return f'println!("{fmt_str}", {rest_args});'
            return f'println!("{fmt_str}");'
        if fn_name == "fprintf":
            parts = args.split(",", 2)
            if len(parts) >= 2 and "stderr" in parts[0]:
                fmt = parts[1].strip().strip('"')
                rest = ",".join(parts[2:]).strip() if len(parts) > 2 else ""
                if rest:
                    return f'eprintln!("{fmt}", {rest});'
                return f'eprintln!("{fmt}");'
        if fn_name == "free":
            return f"drop({args.strip()});"
        if fn_name in ("malloc", "calloc"):
            return f"Vec::with_capacity({args.strip()});"
        return f"{fn_name}({args});"

    # Assignment
    assign_m = re.match(r"(\w+)\s*=\s*(.+)", stmt)
    if assign_m:
        var = assign_m.group(1)
        val = _translate_expr_to_rust(assign_m.group(2).rstrip(";"))
        return f"{var} = {val};"

    return f"// TODO: migrate: {c_stmt}"


def _translate_condition_to_rust(c_cond: str) -> str:
    cond = c_cond.strip()

    # NULL comparisons
    cond = re.sub(r"(\w+)\s*==\s*NULL", r"\1.is_none()", cond)
    cond = re.sub(r"(\w+)\s*!=\s*NULL", r"\1.is_some()", cond)
    cond = re.sub(r"NULL\s*==\s*(\w+)", r"\1.is_none()", cond)
    cond = re.sub(r"NULL\s*!=\s*(\w+)", r"\1.is_some()", cond)

    # Boolean-style: if (!ptr) -> if ptr.is_none()
    cond = re.sub(r"!\s*(\w+)\b(?!\s*\()", r"\1.is_none()", cond)

    # Logical operators
    cond = cond.replace("&&", "&&").replace("||", "||")

    return cond


def _translate_expr_to_rust(c_expr: str) -> str:
    expr = c_expr.strip().rstrip(";")

    # NULL -> None
    if expr == "NULL":
        return "None"

    # Cast expressions
    cast_m = re.match(r"\(\s*[\w\s\*]+\s*\)\s*(.+)", expr)
    if cast_m:
        inner = _translate_expr_to_rust(cast_m.group(1))
        return f"{inner} as _"

    # malloc/calloc
    malloc_m = re.match(r"(?:malloc|calloc)\s*\((.+)\)", expr)
    if malloc_m:
        return f"Vec::with_capacity({malloc_m.group(1).strip()})"

    # sizeof
    expr = re.sub(r"sizeof\s*\(\s*(\w+)\s*\)", r"std::mem::size_of::<\1>()", expr)

    # String literals are fine as-is
    if expr.startswith('"'):
        return expr

    return expr
