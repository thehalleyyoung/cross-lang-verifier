"""Semantic diff between C and Rust code.

Computes semantic (not textual) differences between C and Rust programs,
annotates divergences inline, generates differential test harnesses,
and measures migration completeness.
"""

import re
import os
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Set
from pathlib import Path
from enum import Enum


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

class DiffCategory(Enum):
    MEMORY_MANAGEMENT = "memory_management"
    ERROR_HANDLING = "error_handling"
    INTEGER_SEMANTICS = "integer_semantics"
    STRING_HANDLING = "string_handling"
    CONTROL_FLOW = "control_flow"
    TYPE_COERCION = "type_coercion"
    CONCURRENCY = "concurrency"
    POINTER_ARITHMETIC = "pointer_arithmetic"
    OWNERSHIP = "ownership"
    LIFETIME = "lifetime"
    NULL_HANDLING = "null_handling"
    UNDEFINED_BEHAVIOR = "undefined_behavior"


class Severity(Enum):
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class SemanticDifference:
    category: DiffCategory
    severity: Severity
    c_location: Tuple[int, int]  # (start_line, end_line)
    rust_location: Tuple[int, int]
    c_snippet: str
    rust_snippet: str
    description: str
    recommendation: str = ""

    @property
    def id(self) -> str:
        return f"{self.category.value}:{self.c_location[0]}-{self.rust_location[0]}"


@dataclass
class SemanticDiff:
    differences: List[SemanticDifference] = field(default_factory=list)
    c_functions: List[str] = field(default_factory=list)
    rust_functions: List[str] = field(default_factory=list)
    equivalence_score: float = 0.0

    @property
    def critical_count(self) -> int:
        return sum(1 for d in self.differences if d.severity == Severity.CRITICAL)

    @property
    def by_category(self) -> Dict[DiffCategory, List[SemanticDifference]]:
        result: Dict[DiffCategory, List[SemanticDifference]] = {}
        for d in self.differences:
            result.setdefault(d.category, []).append(d)
        return result

    def summary(self) -> str:
        lines = [f"Semantic Diff: {len(self.differences)} differences found"]
        lines.append(f"  Equivalence score: {self.equivalence_score:.1%}")
        for cat, diffs in self.by_category.items():
            lines.append(f"  {cat.value}: {len(diffs)} differences")
        return "\n".join(lines)


@dataclass
class AnnotatedLine:
    line_number: int
    text: str
    annotations: List[str] = field(default_factory=list)
    severity: Optional[Severity] = None


@dataclass
class AnnotatedSource:
    c_lines: List[AnnotatedLine] = field(default_factory=list)
    rust_lines: List[AnnotatedLine] = field(default_factory=list)

    def render_c(self) -> str:
        return _render_annotated(self.c_lines, "C")

    def render_rust(self) -> str:
        return _render_annotated(self.rust_lines, "Rust")

    def render_side_by_side(self, width: int = 80) -> str:
        out: List[str] = []
        half = width // 2 - 2
        out.append(f"{'C':<{half}} | {'Rust':<{half}}")
        out.append("-" * width)
        max_lines = max(len(self.c_lines), len(self.rust_lines))
        for i in range(max_lines):
            c_text = self.c_lines[i].text if i < len(self.c_lines) else ""
            r_text = self.rust_lines[i].text if i < len(self.rust_lines) else ""
            c_trunc = c_text[:half].ljust(half)
            r_trunc = r_text[:half].ljust(half)
            out.append(f"{c_trunc} | {r_trunc}")
            # annotations for C side
            if i < len(self.c_lines) and self.c_lines[i].annotations:
                for ann in self.c_lines[i].annotations:
                    ann_line = f"  ^ {ann}"[:half].ljust(half)
                    out.append(f"{ann_line} |")
            # annotations for Rust side
            if i < len(self.rust_lines) and self.rust_lines[i].annotations:
                for ann in self.rust_lines[i].annotations:
                    ann_line = f"  ^ {ann}"[:half].ljust(half)
                    padding = " " * half
                    out.append(f"{padding} | {ann_line}")
        return "\n".join(out)


@dataclass
class CompletenessReport:
    total_c_functions: int = 0
    migrated_functions: int = 0
    partially_migrated: int = 0
    unmigrated_functions: List[str] = field(default_factory=list)
    semantic_coverage: float = 0.0
    ub_eliminated: int = 0
    ub_remaining: int = 0
    type_coverage: float = 0.0
    details: Dict[str, Dict] = field(default_factory=dict)

    @property
    def completion_pct(self) -> float:
        if self.total_c_functions == 0:
            return 0.0
        return (self.migrated_functions / self.total_c_functions) * 100.0

    def summary(self) -> str:
        lines = [
            f"Migration Completeness: {self.completion_pct:.1f}%",
            f"  Total C functions: {self.total_c_functions}",
            f"  Fully migrated: {self.migrated_functions}",
            f"  Partially migrated: {self.partially_migrated}",
            f"  Unmigrated: {len(self.unmigrated_functions)}",
            f"  Semantic coverage: {self.semantic_coverage:.1%}",
            f"  Type coverage: {self.type_coverage:.1%}",
            f"  UB eliminated: {self.ub_eliminated}, remaining: {self.ub_remaining}",
        ]
        if self.unmigrated_functions:
            lines.append("  Unmigrated functions:")
            for fn in self.unmigrated_functions[:20]:
                lines.append(f"    - {fn}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Parsing helpers
# ---------------------------------------------------------------------------

_C_FUNC_RE = re.compile(
    r"(?:static\s+|inline\s+|extern\s+)*"
    r"(?:(?:unsigned|signed|long|short|const|volatile|struct|enum|union)\s+)*"
    r"(\w+)\s*\*?\s+"
    r"(\w+)\s*\(([^)]*)\)\s*\{",
    re.MULTILINE,
)

_RUST_FUNC_RE = re.compile(
    r"(?:pub\s+)?(?:unsafe\s+)?(?:extern\s+\"C\"\s+)?fn\s+(\w+)\s*"
    r"(?:<[^>]*>)?\s*\(([^)]*)\)\s*(?:->\s*([^\{]+))?\s*\{",
    re.MULTILINE,
)

_C_MALLOC_RE = re.compile(r"\b(malloc|calloc|realloc|free)\s*\(")
_C_NULL_CHECK_RE = re.compile(r"if\s*\(\s*(\w+)\s*==\s*NULL\s*\)")
_C_OVERFLOW_RE = re.compile(r"(\w+)\s*(\+|\-|\*)\s*(\w+)")
_C_ARRAY_ACCESS_RE = re.compile(r"(\w+)\s*\[\s*([^\]]+)\s*\]")
_C_GOTO_RE = re.compile(r"\bgoto\s+(\w+)\s*;")
_C_CAST_RE = re.compile(r"\(\s*([\w\s\*]+)\s*\)\s*(\w+)")
_RUST_UNWRAP_RE = re.compile(r"\.unwrap\(\)")
_RUST_UNSAFE_RE = re.compile(r"\bunsafe\s*\{")
_RUST_OPTION_RE = re.compile(r"\bOption<")
_RUST_RESULT_RE = re.compile(r"\bResult<")
_RUST_BOX_RE = re.compile(r"\bBox<")
_RUST_VEC_RE = re.compile(r"\bVec<")


def _extract_c_functions(code: str) -> Dict[str, Tuple[str, int, int]]:
    """Extract C functions: name -> (body, start_line, end_line)."""
    functions: Dict[str, Tuple[str, int, int]] = {}
    for m in _C_FUNC_RE.finditer(code):
        name = m.group(2)
        start = code[:m.start()].count("\n") + 1
        brace_count = 0
        pos = m.end() - 1
        while pos < len(code):
            if code[pos] == "{":
                brace_count += 1
            elif code[pos] == "}":
                brace_count -= 1
                if brace_count == 0:
                    break
            pos += 1
        end = code[:pos].count("\n") + 1
        body = code[m.start():pos + 1]
        functions[name] = (body, start, end)
    return functions


def _extract_rust_functions(code: str) -> Dict[str, Tuple[str, int, int]]:
    """Extract Rust functions: name -> (body, start_line, end_line)."""
    functions: Dict[str, Tuple[str, int, int]] = {}
    for m in _RUST_FUNC_RE.finditer(code):
        name = m.group(1)
        start = code[:m.start()].count("\n") + 1
        brace_count = 0
        pos = m.end() - 1
        while pos < len(code):
            if code[pos] == "{":
                brace_count += 1
            elif code[pos] == "}":
                brace_count -= 1
                if brace_count == 0:
                    break
            pos += 1
        end = code[:pos].count("\n") + 1
        body = code[m.start():pos + 1]
        functions[name] = (body, start, end)
    return functions


def _match_functions(
    c_funcs: Dict[str, Tuple[str, int, int]],
    rust_funcs: Dict[str, Tuple[str, int, int]],
) -> List[Tuple[str, Optional[str]]]:
    """Match C functions to Rust counterparts by name similarity."""
    matches: List[Tuple[str, Optional[str]]] = []
    rust_names = set(rust_funcs.keys())
    for c_name in c_funcs:
        # exact match
        if c_name in rust_names:
            matches.append((c_name, c_name))
            rust_names.discard(c_name)
            continue
        # snake_case normalization
        c_snake = re.sub(r"([A-Z])", r"_\1", c_name).lower().strip("_")
        found = False
        for rn in list(rust_names):
            r_snake = re.sub(r"([A-Z])", r"_\1", rn).lower().strip("_")
            if c_snake == r_snake:
                matches.append((c_name, rn))
                rust_names.discard(rn)
                found = True
                break
        if not found:
            matches.append((c_name, None))
    return matches


# ---------------------------------------------------------------------------
# Semantic analysis
# ---------------------------------------------------------------------------

def _analyze_memory_diffs(
    c_body: str, rust_body: str, c_loc: Tuple[int, int], rust_loc: Tuple[int, int]
) -> List[SemanticDifference]:
    diffs: List[SemanticDifference] = []
    c_mallocs = _C_MALLOC_RE.findall(c_body)
    has_box = bool(_RUST_BOX_RE.search(rust_body))
    has_vec = bool(_RUST_VEC_RE.search(rust_body))
    if c_mallocs and not has_box and not has_vec:
        diffs.append(SemanticDifference(
            category=DiffCategory.MEMORY_MANAGEMENT,
            severity=Severity.WARNING,
            c_location=c_loc, rust_location=rust_loc,
            c_snippet=", ".join(c_mallocs),
            rust_snippet="(no Box/Vec found)",
            description="C uses manual allocation but Rust equivalent lacks Box/Vec",
            recommendation="Use Box::new() or Vec for heap allocations",
        ))
    if "free(" in c_body:
        diffs.append(SemanticDifference(
            category=DiffCategory.OWNERSHIP,
            severity=Severity.INFO,
            c_location=c_loc, rust_location=rust_loc,
            c_snippet="free(...)",
            rust_snippet="(automatic drop)",
            description="C explicit free replaced by Rust ownership/Drop",
        ))
    return diffs


def _analyze_null_diffs(
    c_body: str, rust_body: str, c_loc: Tuple[int, int], rust_loc: Tuple[int, int]
) -> List[SemanticDifference]:
    diffs: List[SemanticDifference] = []
    c_nulls = _C_NULL_CHECK_RE.findall(c_body)
    has_option = bool(_RUST_OPTION_RE.search(rust_body))
    if c_nulls and not has_option:
        diffs.append(SemanticDifference(
            category=DiffCategory.NULL_HANDLING,
            severity=Severity.ERROR,
            c_location=c_loc, rust_location=rust_loc,
            c_snippet=f"NULL checks on: {', '.join(c_nulls)}",
            rust_snippet="(no Option<T> found)",
            description="C NULL checks not mapped to Option<T> in Rust",
            recommendation="Use Option<T> or Option<NonNull<T>> for nullable pointers",
        ))
    if "NULL" in c_body and _RUST_UNWRAP_RE.search(rust_body):
        diffs.append(SemanticDifference(
            category=DiffCategory.NULL_HANDLING,
            severity=Severity.WARNING,
            c_location=c_loc, rust_location=rust_loc,
            c_snippet="NULL check",
            rust_snippet=".unwrap()",
            description="C NULL check replaced by .unwrap() which panics instead of returning error",
            recommendation="Use .ok_or() or match/if-let for graceful handling",
        ))
    return diffs


def _analyze_error_handling_diffs(
    c_body: str, rust_body: str, c_loc: Tuple[int, int], rust_loc: Tuple[int, int]
) -> List[SemanticDifference]:
    diffs: List[SemanticDifference] = []
    c_errno = "errno" in c_body or "perror" in c_body
    c_retval = bool(re.search(r"return\s+(-1|NULL|0)\s*;", c_body))
    has_result = bool(_RUST_RESULT_RE.search(rust_body))
    if (c_errno or c_retval) and not has_result:
        diffs.append(SemanticDifference(
            category=DiffCategory.ERROR_HANDLING,
            severity=Severity.WARNING,
            c_location=c_loc, rust_location=rust_loc,
            c_snippet="errno/return-code error handling",
            rust_snippet="(no Result<T,E> found)",
            description="C error handling pattern not mapped to Result<T,E>",
            recommendation="Use Result<T, E> with descriptive error types",
        ))
    return diffs


def _analyze_integer_diffs(
    c_body: str, rust_body: str, c_loc: Tuple[int, int], rust_loc: Tuple[int, int]
) -> List[SemanticDifference]:
    diffs: List[SemanticDifference] = []
    c_casts = _C_CAST_RE.findall(c_body)
    narrowing_types = {"char", "short", "int", "unsigned char", "unsigned short"}
    for cast_type, _ in c_casts:
        cast_type = cast_type.strip()
        if cast_type in narrowing_types and "as " not in rust_body:
            diffs.append(SemanticDifference(
                category=DiffCategory.TYPE_COERCION,
                severity=Severity.WARNING,
                c_location=c_loc, rust_location=rust_loc,
                c_snippet=f"({cast_type}) cast",
                rust_snippet="(no explicit `as` cast)",
                description=f"C narrowing cast to {cast_type} may silently truncate",
                recommendation="Use try_into() or checked conversions",
            ))
            break
    if re.search(r"\bint\b", c_body) and re.search(r"\busize\b", rust_body):
        diffs.append(SemanticDifference(
            category=DiffCategory.INTEGER_SEMANTICS,
            severity=Severity.WARNING,
            c_location=c_loc, rust_location=rust_loc,
            c_snippet="int (signed, typically 32-bit)",
            rust_snippet="usize (unsigned, pointer-width)",
            description="C int mapped to usize changes signedness and width semantics",
            recommendation="Use i32 for direct C int equivalence, usize only for indices",
        ))
    return diffs


def _analyze_control_flow_diffs(
    c_body: str, rust_body: str, c_loc: Tuple[int, int], rust_loc: Tuple[int, int]
) -> List[SemanticDifference]:
    diffs: List[SemanticDifference] = []
    gotos = _C_GOTO_RE.findall(c_body)
    if gotos:
        diffs.append(SemanticDifference(
            category=DiffCategory.CONTROL_FLOW,
            severity=Severity.INFO,
            c_location=c_loc, rust_location=rust_loc,
            c_snippet=f"goto {', '.join(gotos)}",
            rust_snippet="(structured control flow)",
            description="C goto statements restructured in Rust",
        ))
    if "switch" in c_body and "match" not in rust_body:
        diffs.append(SemanticDifference(
            category=DiffCategory.CONTROL_FLOW,
            severity=Severity.WARNING,
            c_location=c_loc, rust_location=rust_loc,
            c_snippet="switch statement",
            rust_snippet="(no match expression)",
            description="C switch not translated to Rust match expression",
            recommendation="Use match with exhaustive patterns",
        ))
    return diffs


def _analyze_string_diffs(
    c_body: str, rust_body: str, c_loc: Tuple[int, int], rust_loc: Tuple[int, int]
) -> List[SemanticDifference]:
    diffs: List[SemanticDifference] = []
    c_str_funcs = re.findall(r"\b(strcpy|strcat|strlen|strcmp|strncpy|sprintf|snprintf)\b", c_body)
    if c_str_funcs and "String" not in rust_body and "&str" not in rust_body:
        diffs.append(SemanticDifference(
            category=DiffCategory.STRING_HANDLING,
            severity=Severity.WARNING,
            c_location=c_loc, rust_location=rust_loc,
            c_snippet=f"C string functions: {', '.join(set(c_str_funcs))}",
            rust_snippet="(no String/&str usage found)",
            description="C string operations not mapped to safe Rust string types",
            recommendation="Use String, &str, or CStr/CString for FFI boundaries",
        ))
    if "sprintf" in c_body or "snprintf" in c_body:
        has_format = "format!" in rust_body or "write!" in rust_body
        if not has_format:
            diffs.append(SemanticDifference(
                category=DiffCategory.STRING_HANDLING,
                severity=Severity.WARNING,
                c_location=c_loc, rust_location=rust_loc,
                c_snippet="sprintf/snprintf",
                rust_snippet="(no format!/write!)",
                description="C format string not replaced with Rust format macro",
                recommendation="Use format!() or write!() macros",
            ))
    return diffs


def _analyze_concurrency_diffs(
    c_body: str, rust_body: str, c_loc: Tuple[int, int], rust_loc: Tuple[int, int]
) -> List[SemanticDifference]:
    diffs: List[SemanticDifference] = []
    c_mutex = bool(re.search(r"\bpthread_mutex_(lock|unlock|init)\b", c_body))
    rust_mutex = "Mutex" in rust_body or "RwLock" in rust_body
    if c_mutex and not rust_mutex:
        diffs.append(SemanticDifference(
            category=DiffCategory.CONCURRENCY,
            severity=Severity.ERROR,
            c_location=c_loc, rust_location=rust_loc,
            c_snippet="pthread_mutex",
            rust_snippet="(no Mutex/RwLock)",
            description="C mutex usage not mapped to Rust synchronization primitives",
            recommendation="Use std::sync::Mutex or RwLock",
        ))
    if "volatile" in c_body and "AtomicBool" not in rust_body and "Atomic" not in rust_body:
        diffs.append(SemanticDifference(
            category=DiffCategory.CONCURRENCY,
            severity=Severity.WARNING,
            c_location=c_loc, rust_location=rust_loc,
            c_snippet="volatile",
            rust_snippet="(no Atomic type)",
            description="C volatile not mapped to Rust atomic types",
            recommendation="Use std::sync::atomic types with appropriate Ordering",
        ))
    return diffs


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def semantic_diff(c_code: str, rust_code: str) -> SemanticDiff:
    """Compute semantic differences between C and Rust code."""
    c_funcs = _extract_c_functions(c_code)
    rust_funcs = _extract_rust_functions(rust_code)
    matches = _match_functions(c_funcs, rust_funcs)

    result = SemanticDiff(
        c_functions=list(c_funcs.keys()),
        rust_functions=list(rust_funcs.keys()),
    )

    analyzers = [
        _analyze_memory_diffs,
        _analyze_null_diffs,
        _analyze_error_handling_diffs,
        _analyze_integer_diffs,
        _analyze_control_flow_diffs,
        _analyze_string_diffs,
        _analyze_concurrency_diffs,
    ]

    matched = 0
    for c_name, rust_name in matches:
        if rust_name is None:
            continue
        matched += 1
        c_body, c_start, c_end = c_funcs[c_name]
        r_body, r_start, r_end = rust_funcs[rust_name]
        c_loc = (c_start, c_end)
        r_loc = (r_start, r_end)
        for analyzer in analyzers:
            result.differences.extend(analyzer(c_body, r_body, c_loc, r_loc))

    total = len(c_funcs)
    if total > 0:
        base_score = matched / total
        penalty = len(result.differences) * 0.02
        result.equivalence_score = max(0.0, min(1.0, base_score - penalty))

    return result


def annotate_divergences(c_code: str, rust_code: str) -> AnnotatedSource:
    """Annotate C and Rust source with inline divergence markers."""
    diff = semantic_diff(c_code, rust_code)
    c_lines_raw = c_code.splitlines()
    rust_lines_raw = rust_code.splitlines()

    c_annotated = [
        AnnotatedLine(line_number=i + 1, text=line)
        for i, line in enumerate(c_lines_raw)
    ]
    rust_annotated = [
        AnnotatedLine(line_number=i + 1, text=line)
        for i, line in enumerate(rust_lines_raw)
    ]

    for d in diff.differences:
        c_start, c_end = d.c_location
        for i in range(max(0, c_start - 1), min(len(c_annotated), c_end)):
            c_annotated[i].annotations.append(
                f"[{d.severity.value.upper()}] {d.description}"
            )
            if c_annotated[i].severity is None or d.severity.value > (c_annotated[i].severity.value or ""):
                c_annotated[i].severity = d.severity

        r_start, r_end = d.rust_location
        for i in range(max(0, r_start - 1), min(len(rust_annotated), r_end)):
            rust_annotated[i].annotations.append(
                f"[{d.severity.value.upper()}] {d.description}"
            )
            if rust_annotated[i].severity is None or d.severity.value > (rust_annotated[i].severity.value or ""):
                rust_annotated[i].severity = d.severity

    return AnnotatedSource(c_lines=c_annotated, rust_lines=rust_annotated)


def generate_test_harness(c_code: str, rust_code: str) -> str:
    """Generate a differential test harness for C and Rust code."""
    c_funcs = _extract_c_functions(c_code)
    rust_funcs = _extract_rust_functions(rust_code)
    matches = _match_functions(c_funcs, rust_funcs)

    parts: List[str] = []
    parts.append("// Auto-generated differential test harness")
    parts.append("// Compile C side:  gcc -o c_test c_test.c")
    parts.append("// Compile Rust side: rustc --edition 2021 rust_test.rs")
    parts.append("")

    # C test driver
    parts.append("// ===== c_test.c =====")
    parts.append('#include <stdio.h>')
    parts.append('#include <stdlib.h>')
    parts.append('#include <string.h>')
    parts.append('#include <stdint.h>')
    parts.append("")
    parts.append("// Paste original C functions here")
    parts.append("")
    parts.append("int main(void) {")

    for c_name, rust_name in matches:
        if rust_name is None:
            continue
        c_body = c_funcs[c_name][0]
        # Extract parameter types from function signature
        sig_match = _C_FUNC_RE.search(c_body)
        if sig_match:
            ret_type = sig_match.group(1)
            params_str = sig_match.group(3).strip()
            parts.append(f"    // Test {c_name}")
            if params_str and params_str != "void":
                param_list = [p.strip() for p in params_str.split(",")]
                args: List[str] = []
                for idx, p in enumerate(param_list):
                    # Generate test values based on type
                    if "int" in p:
                        args.append("42")
                    elif "char*" in p or "char *" in p:
                        args.append('"test"')
                    elif "float" in p or "double" in p:
                        args.append("3.14")
                    elif "*" in p:
                        args.append("NULL")
                    else:
                        args.append("0")
                call_args = ", ".join(args)
            else:
                call_args = ""
            if ret_type != "void":
                parts.append(f'    printf("{c_name}: %d\\n", {c_name}({call_args}));')
            else:
                parts.append(f"    {c_name}({call_args});")
                parts.append(f'    printf("{c_name}: ok\\n");')

    parts.append("    return 0;")
    parts.append("}")
    parts.append("")

    # Rust test driver
    parts.append("// ===== rust_test.rs =====")
    parts.append("// Paste original Rust functions here")
    parts.append("")
    parts.append("fn main() {")

    for c_name, rust_name in matches:
        if rust_name is None:
            continue
        rust_body = rust_funcs[rust_name][0]
        sig_match = _RUST_FUNC_RE.search(rust_body)
        if sig_match:
            params_str = sig_match.group(2).strip()
            ret_type = (sig_match.group(3) or "").strip()
            parts.append(f"    // Test {rust_name}")
            if params_str:
                param_list = [p.strip() for p in params_str.split(",") if p.strip()]
                args = []
                for p in param_list:
                    type_part = p.split(":")[1].strip() if ":" in p else ""
                    if "i32" in type_part or "i64" in type_part or "usize" in type_part:
                        args.append("42")
                    elif "f32" in type_part or "f64" in type_part:
                        args.append("3.14")
                    elif "&str" in type_part:
                        args.append('"test"')
                    elif "String" in type_part:
                        args.append('"test".to_string()')
                    elif "bool" in type_part:
                        args.append("true")
                    elif "Option" in type_part:
                        args.append("None")
                    else:
                        args.append("Default::default()")
                call_args = ", ".join(args)
            else:
                call_args = ""
            if ret_type and ret_type != "()":
                parts.append(f'    println!("{rust_name}: {{:?}}", {rust_name}({call_args}));')
            else:
                parts.append(f"    {rust_name}({call_args});")
                parts.append(f'    println!("{rust_name}: ok");')

    parts.append("}")
    parts.append("")

    # Comparison script
    parts.append("# ===== compare.sh =====")
    parts.append("#!/bin/bash")
    parts.append("set -e")
    parts.append("gcc -o c_test c_test.c -Wall -Wextra")
    parts.append("rustc --edition 2021 rust_test.rs -o rust_test")
    parts.append("C_OUT=$(./c_test)")
    parts.append("RUST_OUT=$(./rust_test)")
    parts.append('if [ "$C_OUT" = "$RUST_OUT" ]; then')
    parts.append('    echo "PASS: Outputs match"')
    parts.append("else")
    parts.append('    echo "FAIL: Outputs differ"')
    parts.append('    diff <(echo "$C_OUT") <(echo "$RUST_OUT")')
    parts.append("fi")

    return "\n".join(parts)


def migration_completeness(c_dir: str, rust_dir: str) -> CompletenessReport:
    """Measure what percentage of C semantics are preserved in Rust migration."""
    c_path = Path(c_dir)
    rust_path = Path(rust_dir)
    report = CompletenessReport()

    all_c_funcs: Dict[str, str] = {}
    for c_file in c_path.rglob("*.c"):
        code = c_file.read_text(errors="replace")
        for name, (body, _, _) in _extract_c_functions(code).items():
            all_c_funcs[name] = body
    for h_file in c_path.rglob("*.h"):
        code = h_file.read_text(errors="replace")
        for name, (body, _, _) in _extract_c_functions(code).items():
            all_c_funcs[name] = body

    all_rust_funcs: Dict[str, str] = {}
    for rs_file in rust_path.rglob("*.rs"):
        code = rs_file.read_text(errors="replace")
        for name, (body, _, _) in _extract_rust_functions(code).items():
            all_rust_funcs[name] = body

    report.total_c_functions = len(all_c_funcs)
    rust_name_set = set(all_rust_funcs.keys())
    rust_snake_set = {re.sub(r"([A-Z])", r"_\1", n).lower().strip("_") for n in rust_name_set}

    for c_name, c_body in all_c_funcs.items():
        c_snake = re.sub(r"([A-Z])", r"_\1", c_name).lower().strip("_")
        if c_name in rust_name_set or c_snake in rust_snake_set:
            rust_body = all_rust_funcs.get(c_name, "")
            if not rust_body:
                for rn, rb in all_rust_funcs.items():
                    if re.sub(r"([A-Z])", r"_\1", rn).lower().strip("_") == c_snake:
                        rust_body = rb
                        break
            has_unsafe = bool(_RUST_UNSAFE_RE.search(rust_body))
            if has_unsafe:
                report.partially_migrated += 1
            else:
                report.migrated_functions += 1

            # UB analysis
            ub_patterns = [r"\bfree\(", r"\bmalloc\(", r"\bstrcpy\(", r"\bgets\("]
            for pat in ub_patterns:
                if re.search(pat, c_body):
                    report.ub_eliminated += 1

            report.details[c_name] = {
                "status": "partial" if has_unsafe else "complete",
                "has_unsafe": has_unsafe,
            }
        else:
            report.unmigrated_functions.append(c_name)

    # Compute coverage metrics
    if report.total_c_functions > 0:
        report.semantic_coverage = (
            (report.migrated_functions + report.partially_migrated * 0.5)
            / report.total_c_functions
        )
    # Type coverage: ratio of Rust functions using safe types
    safe_type_count = 0
    for body in all_rust_funcs.values():
        if (_RUST_OPTION_RE.search(body) or _RUST_RESULT_RE.search(body)
                or _RUST_BOX_RE.search(body) or _RUST_VEC_RE.search(body)):
            safe_type_count += 1
    if all_rust_funcs:
        report.type_coverage = safe_type_count / len(all_rust_funcs)

    return report


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _render_annotated(lines: List[AnnotatedLine], lang: str) -> str:
    """Render annotated source with inline markers."""
    severity_markers = {
        Severity.INFO: "ℹ️",
        Severity.WARNING: "⚠️",
        Severity.ERROR: "❌",
        Severity.CRITICAL: "🔴",
    }
    out: List[str] = [f"=== {lang} Source ==="]
    for line in lines:
        prefix = f"{line.line_number:4d}"
        if line.severity:
            marker = severity_markers.get(line.severity, " ")
            out.append(f"{prefix} {marker} {line.text}")
        else:
            out.append(f"{prefix}    {line.text}")
        for ann in line.annotations:
            out.append(f"          {ann}")
    return "\n".join(out)
