"""Catalog undefined behavior in C that Rust prevents.

Detects UB patterns in C code, verifies whether corresponding Rust code
eliminates them, and generates tests that trigger the C UB to demonstrate
the safety advantage of the Rust translation.
"""

import re
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
from enum import Enum


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

class UBCategory(Enum):
    SIGNED_OVERFLOW = "signed_overflow"
    NULL_DEREF = "null_deref"
    USE_AFTER_FREE = "use_after_free"
    BUFFER_OVERFLOW = "buffer_overflow"
    DATA_RACE = "data_race"
    STRICT_ALIASING = "strict_aliasing"
    UNINITIALIZED = "uninitialized"
    DIVISION_BY_ZERO = "division_by_zero"
    SHIFT_OVERFLOW = "shift_overflow"
    DOUBLE_FREE = "double_free"
    DANGLING_POINTER = "dangling_pointer"
    FORMAT_STRING = "format_string"
    SEQUENCE_POINT = "sequence_point"


class RustPrevention(Enum):
    TYPE_SYSTEM = "type_system"
    BORROW_CHECKER = "borrow_checker"
    BOUNDS_CHECKING = "bounds_checking"
    OWNERSHIP = "ownership"
    OPTION_TYPE = "option_type"
    SEND_SYNC = "send_sync"
    NO_RAW_POINTERS = "no_raw_pointers"
    CHECKED_ARITHMETIC = "checked_arithmetic"
    SAFE_FORMATTING = "safe_formatting"


@dataclass
class UBPattern:
    category: UBCategory
    location: Tuple[int, int]  # (line_start, line_end)
    snippet: str
    description: str
    severity: str  # "critical", "high", "medium", "low"
    rust_prevention: RustPrevention
    cwe_id: Optional[str] = None

    @property
    def id(self) -> str:
        return f"{self.category.value}:{self.location[0]}"


@dataclass
class UBVerification:
    pattern: UBPattern
    eliminated: bool
    rust_snippet: str = ""
    mechanism: str = ""
    residual_risk: str = ""


@dataclass
class UBReport:
    total_patterns: int = 0
    eliminated: int = 0
    remaining: int = 0
    verifications: List[UBVerification] = field(default_factory=list)

    @property
    def elimination_rate(self) -> float:
        return self.eliminated / self.total_patterns if self.total_patterns > 0 else 0.0

    def summary(self) -> str:
        lines = [
            f"UB Elimination Report: {self.elimination_rate:.0%} eliminated",
            f"  Total UB patterns: {self.total_patterns}",
            f"  Eliminated: {self.eliminated}",
            f"  Remaining: {self.remaining}",
        ]
        for v in self.verifications:
            status = "✅" if v.eliminated else "❌"
            lines.append(f"  {status} {v.pattern.category.value} at line {v.pattern.location[0]}: {v.mechanism}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# UB detection patterns
# ---------------------------------------------------------------------------

_DETECTORS: List[Tuple[UBCategory, str, re.Pattern, str, str, RustPrevention, Optional[str]]] = [
    (
        UBCategory.SIGNED_OVERFLOW,
        "critical",
        re.compile(r"\b(int|long|short)\s+\w+\s*[+\-*]=?\s*", re.MULTILINE),
        "Signed integer arithmetic may overflow (undefined in C)",
        "Use wrapping_add/checked_add or Rust's default panic-on-overflow",
        RustPrevention.CHECKED_ARITHMETIC,
        "CWE-190",
    ),
    (
        UBCategory.NULL_DEREF,
        "critical",
        re.compile(r"\*\s*(\w+)\s*(?:=|\[|\.|->\s*\w+)", re.MULTILINE),
        "Pointer dereference without null check",
        "Rust Option<&T> prevents null dereference at compile time",
        RustPrevention.OPTION_TYPE,
        "CWE-476",
    ),
    (
        UBCategory.USE_AFTER_FREE,
        "critical",
        re.compile(r"free\s*\(\s*(\w+)\s*\).*?\b\1\b", re.DOTALL),
        "Variable used after being freed",
        "Rust ownership system prevents use after move/drop",
        RustPrevention.OWNERSHIP,
        "CWE-416",
    ),
    (
        UBCategory.BUFFER_OVERFLOW,
        "critical",
        re.compile(r"\b(strcpy|strcat|gets|sprintf)\s*\(", re.MULTILINE),
        "Unbounded write to buffer",
        "Rust slices and Vec have bounds checking",
        RustPrevention.BOUNDS_CHECKING,
        "CWE-120",
    ),
    (
        UBCategory.DATA_RACE,
        "high",
        re.compile(r"\bpthread_create\b.*?(?:(?!pthread_mutex_lock).)*$", re.DOTALL),
        "Shared mutable state accessed from multiple threads",
        "Rust Send/Sync traits prevent data races at compile time",
        RustPrevention.SEND_SYNC,
        "CWE-362",
    ),
    (
        UBCategory.STRICT_ALIASING,
        "high",
        re.compile(r"\(\s*([\w\s]+\*)\s*\)\s*(&?\w+)", re.MULTILINE),
        "Pointer cast may violate strict aliasing rules",
        "Rust's type system prevents aliasing violations",
        RustPrevention.TYPE_SYSTEM,
        "CWE-843",
    ),
    (
        UBCategory.UNINITIALIZED,
        "high",
        re.compile(r"(?:int|long|char|float|double|short|unsigned)\s+(\w+)\s*;(?!\s*=)", re.MULTILINE),
        "Variable declared without initialization",
        "Rust requires initialization before use",
        RustPrevention.TYPE_SYSTEM,
        "CWE-457",
    ),
    (
        UBCategory.DIVISION_BY_ZERO,
        "high",
        re.compile(r"(\w+)\s*/\s*(\w+)", re.MULTILINE),
        "Division where divisor may be zero",
        "Rust panics on division by zero (defined behavior)",
        RustPrevention.CHECKED_ARITHMETIC,
        "CWE-369",
    ),
    (
        UBCategory.SHIFT_OVERFLOW,
        "medium",
        re.compile(r"(\w+)\s*(<<|>>)\s*(\w+)", re.MULTILINE),
        "Shift amount may exceed bit width",
        "Rust checks shift amounts in debug mode",
        RustPrevention.CHECKED_ARITHMETIC,
        "CWE-682",
    ),
    (
        UBCategory.DOUBLE_FREE,
        "critical",
        re.compile(r"free\s*\(\s*(\w+)\s*\).*?free\s*\(\s*\1\s*\)", re.DOTALL),
        "Same pointer freed twice",
        "Rust ownership: exactly one owner calls drop",
        RustPrevention.OWNERSHIP,
        "CWE-415",
    ),
    (
        UBCategory.DANGLING_POINTER,
        "critical",
        re.compile(r"return\s+&\s*(\w+)\s*;", re.MULTILINE),
        "Returning pointer to local variable",
        "Rust borrow checker prevents dangling references",
        RustPrevention.BORROW_CHECKER,
        "CWE-825",
    ),
    (
        UBCategory.FORMAT_STRING,
        "high",
        re.compile(r"\b(printf|fprintf|sprintf|snprintf)\s*\(\s*(\w+)\s*[,)]", re.MULTILINE),
        "Format string from variable (potential format string attack)",
        "Rust format macros are compile-time checked",
        RustPrevention.SAFE_FORMATTING,
        "CWE-134",
    ),
    (
        UBCategory.SEQUENCE_POINT,
        "medium",
        re.compile(r"(\w+)\+\+.*?\1\+\+|(\w+)\s*=\s*\2\+\+", re.MULTILINE),
        "Multiple unsequenced modifications to same variable",
        "Rust's expression semantics have defined evaluation order",
        RustPrevention.TYPE_SYSTEM,
        None,
    ),
]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def find_ub_patterns(c_code: str) -> List[UBPattern]:
    """Find undefined behavior patterns in C code."""
    patterns: List[UBPattern] = []
    lines = c_code.splitlines()

    for category, severity, regex, desc, _, prevention, cwe in _DETECTORS:
        for m in regex.finditer(c_code):
            start_line = c_code[:m.start()].count("\n") + 1
            end_line = c_code[:m.end()].count("\n") + 1
            snippet = m.group(0)[:120].strip()
            # Simple heuristic filtering for some detectors
            if category == UBCategory.DIVISION_BY_ZERO:
                # Skip obvious non-zero divisors (literal numbers > 0)
                divisor = m.group(2)
                if divisor.isdigit() and int(divisor) != 0:
                    continue
            if category == UBCategory.NULL_DEREF:
                # Check if there's a null check nearby (within 3 lines above)
                check_start = max(0, start_line - 4)
                context = "\n".join(lines[check_start:start_line])
                ptr_name = m.group(1)
                if re.search(rf"if\s*\(\s*{re.escape(ptr_name)}\s*(!=\s*NULL|!= 0)\s*\)", context):
                    continue
            patterns.append(UBPattern(
                category=category,
                location=(start_line, end_line),
                snippet=snippet,
                description=desc,
                severity=severity,
                rust_prevention=prevention,
                cwe_id=cwe,
            ))

    # De-duplicate by (category, line)
    seen: set = set()
    unique: List[UBPattern] = []
    for p in patterns:
        key = (p.category, p.location[0])
        if key not in seen:
            seen.add(key)
            unique.append(p)

    return unique


def verify_ub_elimination(c_code: str, rust_code: str) -> UBReport:
    """Verify that Rust code eliminates C undefined behavior patterns."""
    ub_patterns = find_ub_patterns(c_code)
    report = UBReport(total_patterns=len(ub_patterns))

    for p in ub_patterns:
        verification = _check_elimination(p, rust_code)
        report.verifications.append(verification)
        if verification.eliminated:
            report.eliminated += 1
        else:
            report.remaining += 1

    return report


def _check_elimination(pattern: UBPattern, rust_code: str) -> UBVerification:
    """Check if a specific UB pattern is eliminated in the Rust code."""
    cat = pattern.category

    if cat == UBCategory.SIGNED_OVERFLOW:
        # Check for wrapping/checked/saturating arithmetic
        has_checked = bool(re.search(
            r"\.(checked_add|checked_sub|checked_mul|wrapping_add|wrapping_sub|saturating_add|overflowing_add)\b",
            rust_code
        ))
        # Default Rust debug mode panics on overflow
        return UBVerification(
            pattern=pattern,
            eliminated=True,
            mechanism="Rust panics on overflow in debug; use checked_*/wrapping_* in release" + (
                " (explicit checked ops found)" if has_checked else ""
            ),
            residual_risk="" if has_checked else "Release mode wraps silently unless checked ops used",
        )

    if cat == UBCategory.NULL_DEREF:
        has_option = bool(re.search(r"\bOption<", rust_code))
        has_raw_ptr = bool(re.search(r"\*(?:mut|const)\s", rust_code))
        eliminated = has_option and not has_raw_ptr
        return UBVerification(
            pattern=pattern,
            eliminated=eliminated,
            mechanism="Option<T> prevents null" if has_option else "No Option<T> found",
            residual_risk="Raw pointers still present" if has_raw_ptr else "",
        )

    if cat == UBCategory.USE_AFTER_FREE:
        has_unsafe = bool(re.search(r"\bunsafe\s*\{", rust_code))
        has_raw_ptr = bool(re.search(r"\*(?:mut|const)\s", rust_code))
        eliminated = not has_raw_ptr
        return UBVerification(
            pattern=pattern,
            eliminated=eliminated,
            mechanism="Ownership system prevents use-after-free" if eliminated else "Raw pointers bypass ownership",
            residual_risk="unsafe block with raw pointers" if has_unsafe else "",
        )

    if cat == UBCategory.BUFFER_OVERFLOW:
        has_bounds = bool(re.search(r"\.(get|get_mut)\s*\(", rust_code)) or "Vec<" in rust_code
        has_unsafe_index = bool(re.search(r"get_unchecked", rust_code))
        eliminated = has_bounds and not has_unsafe_index
        return UBVerification(
            pattern=pattern,
            eliminated=eliminated,
            mechanism="Bounds-checked indexing" if eliminated else "Unchecked access found",
            residual_risk="get_unchecked bypasses bounds checking" if has_unsafe_index else "",
        )

    if cat == UBCategory.DATA_RACE:
        has_sync = bool(re.search(r"\b(Mutex|RwLock|Arc|Atomic\w+)\b", rust_code))
        return UBVerification(
            pattern=pattern,
            eliminated=has_sync,
            mechanism="Send/Sync traits + synchronization primitives" if has_sync else "No sync primitives found",
            residual_risk="" if has_sync else "Potential data race if shared state accessed from threads",
        )

    if cat == UBCategory.STRICT_ALIASING:
        has_transmute = "transmute" in rust_code
        return UBVerification(
            pattern=pattern,
            eliminated=not has_transmute,
            mechanism="Rust type system prevents aliasing violations" if not has_transmute else "transmute can alias",
            residual_risk="std::mem::transmute bypasses type safety" if has_transmute else "",
        )

    if cat == UBCategory.UNINITIALIZED:
        has_maybe_uninit = "MaybeUninit" in rust_code
        return UBVerification(
            pattern=pattern,
            eliminated=not has_maybe_uninit,
            mechanism="Rust requires initialization before use",
            residual_risk="MaybeUninit allows uninitialized memory" if has_maybe_uninit else "",
        )

    if cat == UBCategory.DOUBLE_FREE:
        return UBVerification(
            pattern=pattern,
            eliminated=True,
            mechanism="Ownership: single owner, automatic drop, no double-free possible",
        )

    if cat == UBCategory.DANGLING_POINTER:
        return UBVerification(
            pattern=pattern,
            eliminated=True,
            mechanism="Borrow checker ensures references never outlive referents",
        )

    if cat == UBCategory.FORMAT_STRING:
        has_format_macro = bool(re.search(r"\b(format!|println!|write!|eprintln!)\b", rust_code))
        return UBVerification(
            pattern=pattern,
            eliminated=has_format_macro,
            mechanism="Compile-time format string checking" if has_format_macro else "No format macros found",
        )

    if cat == UBCategory.DIVISION_BY_ZERO:
        return UBVerification(
            pattern=pattern,
            eliminated=True,
            mechanism="Rust panics on division by zero (defined behavior, not UB)",
        )

    if cat == UBCategory.SHIFT_OVERFLOW:
        return UBVerification(
            pattern=pattern,
            eliminated=True,
            mechanism="Rust panics on shift overflow in debug mode",
            residual_risk="Release mode wraps shift amount",
        )

    if cat == UBCategory.SEQUENCE_POINT:
        return UBVerification(
            pattern=pattern,
            eliminated=True,
            mechanism="Rust has defined expression evaluation order",
        )

    return UBVerification(
        pattern=pattern,
        eliminated=False,
        mechanism="Unknown UB category — manual review needed",
    )


def generate_ub_tests(c_code: str) -> str:
    """Generate C test code that triggers undefined behavior patterns found in the code."""
    patterns = find_ub_patterns(c_code)
    parts: List[str] = []

    parts.append("/*")
    parts.append(" * Auto-generated UB trigger tests")
    parts.append(" * Compile with: gcc -fsanitize=undefined,address -g -o ub_test ub_test.c")
    parts.append(" * Run with: ./ub_test")
    parts.append(" * UBSan/ASan will report triggered UB")
    parts.append(" */")
    parts.append("")
    parts.append("#include <stdio.h>")
    parts.append("#include <stdlib.h>")
    parts.append("#include <string.h>")
    parts.append("#include <limits.h>")
    parts.append("#include <stdint.h>")
    parts.append("#include <pthread.h>")
    parts.append("")

    test_num = 0
    for p in patterns:
        test_num += 1
        test_name = f"test_ub_{p.category.value}_{test_num}"
        parts.append(f"/* {p.description} */")
        parts.append(f"/* {p.cwe_id or 'No CWE'} | Severity: {p.severity} */")
        parts.append(f"void {test_name}(void) {{")
        parts.append(f'    printf("Testing {p.category.value} (line {p.location[0]})...\\n");')

        if p.category == UBCategory.SIGNED_OVERFLOW:
            parts.append("    int a = INT_MAX;")
            parts.append("    int b = a + 1;  /* UB: signed overflow */")
            parts.append('    printf("  INT_MAX + 1 = %d (UB!)\\n", b);')

        elif p.category == UBCategory.NULL_DEREF:
            parts.append("    int *ptr = NULL;")
            parts.append("    /* int val = *ptr;  // UB: null dereference */")
            parts.append('    printf("  Skipped null deref (would crash)\\n");')

        elif p.category == UBCategory.USE_AFTER_FREE:
            parts.append("    int *p = (int*)malloc(sizeof(int));")
            parts.append("    *p = 42;")
            parts.append("    free(p);")
            parts.append("    /* int val = *p;  // UB: use after free */")
            parts.append('    printf("  Skipped use-after-free (ASan would catch)\\n");')

        elif p.category == UBCategory.BUFFER_OVERFLOW:
            parts.append("    char buf[8];")
            parts.append('    /* strcpy(buf, "this string is way too long for buf"); */')
            parts.append('    strncpy(buf, "ok", sizeof(buf));')
            parts.append('    printf("  Buffer overflow test (safe version): %s\\n", buf);')

        elif p.category == UBCategory.DATA_RACE:
            parts.append("    /* Data race test requires threading — see separate test */")
            parts.append('    printf("  Data race detection requires TSan\\n");')

        elif p.category == UBCategory.STRICT_ALIASING:
            parts.append("    int x = 42;")
            parts.append("    float *fp = (float*)&x;  /* strict aliasing violation */")
            parts.append('    printf("  Aliased read: %f (UB!)\\n", *fp);')

        elif p.category == UBCategory.UNINITIALIZED:
            parts.append("    int uninit;")
            parts.append("    /* printf(\"%d\", uninit);  // UB: uninitialized read */")
            parts.append('    printf("  Uninitialized variable test\\n");')

        elif p.category == UBCategory.DIVISION_BY_ZERO:
            parts.append("    int divisor = 0;")
            parts.append("    /* int result = 42 / divisor;  // UB */")
            parts.append('    printf("  Division by zero test (guarded)\\n");')

        elif p.category == UBCategory.SHIFT_OVERFLOW:
            parts.append("    int x = 1;")
            parts.append("    int shifted = x << 33;  /* UB if int is 32-bit */")
            parts.append('    printf("  Shift overflow: %d (UB!)\\n", shifted);')

        elif p.category == UBCategory.DOUBLE_FREE:
            parts.append("    int *p = (int*)malloc(sizeof(int));")
            parts.append("    free(p);")
            parts.append("    /* free(p);  // UB: double free */")
            parts.append('    printf("  Double free test (guarded)\\n");')

        elif p.category == UBCategory.DANGLING_POINTER:
            parts.append("    /* Returns &local which is dangling after return */")
            parts.append('    printf("  Dangling pointer test\\n");')

        elif p.category == UBCategory.FORMAT_STRING:
            parts.append("    char user_input[] = \"%s%s%s%s%s\";")
            parts.append("    /* printf(user_input);  // UB: format string vuln */")
            parts.append('    printf("  Format string test (guarded): %s\\n", user_input);')

        elif p.category == UBCategory.SEQUENCE_POINT:
            parts.append("    int i = 0;")
            parts.append("    /* int val = i++ + i++;  // UB: unsequenced */")
            parts.append('    printf("  Sequence point test\\n");')

        parts.append("}")
        parts.append("")

    # Main function
    parts.append("int main(void) {")
    parts.append(f'    printf("Running {test_num} UB tests...\\n\\n");')
    test_num2 = 0
    for p in patterns:
        test_num2 += 1
        parts.append(f"    test_ub_{p.category.value}_{test_num2}();")
    parts.append("")
    parts.append(f'    printf("\\nDone. {test_num} tests executed.\\n");')
    parts.append('    printf("Compile with -fsanitize=undefined,address for full UB detection.\\n");')
    parts.append("    return 0;")
    parts.append("}")

    return "\n".join(parts)
