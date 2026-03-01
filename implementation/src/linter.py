"""C code linter for Rust-migration readiness.

Analyses C source code for patterns that require special attention during
migration to Rust.  Covers 30+ rules spanning pointer arithmetic, macro
complexity, undefined-behaviour idioms, GCC extensions, C11/C23 features,
and more.  Each finding includes a severity, description, Rust-equivalent
suggestion, and auto-fix availability flag.
"""

import re
import time
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Set
from enum import Enum


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------

class Severity(Enum):
    """How urgent the finding is for migration."""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class RuleCategory(Enum):
    """Broad category a lint rule belongs to."""
    POINTER = "pointer"
    MEMORY = "memory"
    MACRO = "macro"
    CONTROL_FLOW = "control_flow"
    TYPE_SYSTEM = "type_system"
    CONCURRENCY = "concurrency"
    PORTABILITY = "portability"
    GCC_EXTENSION = "gcc_extension"
    C11_FEATURE = "c11_feature"
    C23_FEATURE = "c23_feature"
    DEPRECATED = "deprecated"
    UNDEFINED_BEHAVIOUR = "undefined_behaviour"


class MigrationDifficulty(Enum):
    """How hard it is to migrate a particular pattern."""
    TRIVIAL = "trivial"
    EASY = "easy"
    MODERATE = "moderate"
    HARD = "hard"
    VERY_HARD = "very_hard"


class AutoFixStatus(Enum):
    """Whether an automated fix is available."""
    AVAILABLE = "available"
    PARTIAL = "partial"
    UNAVAILABLE = "unavailable"


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class LintRule:
    """Definition of a single lint rule."""
    id: str
    name: str
    category: RuleCategory
    severity: Severity
    description: str
    rust_equivalent: str
    auto_fix: AutoFixStatus
    difficulty: MigrationDifficulty
    pattern: str = ""
    enabled: bool = True

    @property
    def display(self) -> str:
        return f"[{self.id}] {self.name} ({self.severity.value})"


@dataclass
class LintFinding:
    """A single finding produced by a lint rule."""
    rule_id: str
    rule_name: str
    category: RuleCategory
    severity: Severity
    description: str
    rust_equivalent: str
    auto_fix: AutoFixStatus
    line_number: int
    column: int = 0
    code_snippet: str = ""
    suggestion: str = ""
    confidence: float = 1.0

    @property
    def id(self) -> str:
        return f"{self.rule_id}:L{self.line_number}"

    @property
    def display(self) -> str:
        return (
            f"L{self.line_number}: [{self.rule_id}] {self.description} "
            f"(severity={self.severity.value})"
        )


@dataclass
class LintReport:
    """Aggregated report for all findings on a single source string."""
    findings: List[LintFinding] = field(default_factory=list)
    total_lines: int = 0
    rules_applied: int = 0
    duration_ms: float = 0.0

    @property
    def critical_count(self) -> int:
        return sum(1 for f in self.findings if f.severity == Severity.CRITICAL)

    @property
    def error_count(self) -> int:
        return sum(1 for f in self.findings if f.severity == Severity.ERROR)

    @property
    def warning_count(self) -> int:
        return sum(1 for f in self.findings if f.severity == Severity.WARNING)

    @property
    def info_count(self) -> int:
        return sum(1 for f in self.findings if f.severity == Severity.INFO)

    @property
    def by_category(self) -> Dict[RuleCategory, List[LintFinding]]:
        result: Dict[RuleCategory, List[LintFinding]] = {}
        for f in self.findings:
            result.setdefault(f.category, []).append(f)
        return result

    @property
    def by_severity(self) -> Dict[Severity, List[LintFinding]]:
        result: Dict[Severity, List[LintFinding]] = {}
        for f in self.findings:
            result.setdefault(f.severity, []).append(f)
        return result

    def summary(self) -> str:
        return (
            f"LintReport: {len(self.findings)} findings "
            f"(C={self.critical_count} E={self.error_count} "
            f"W={self.warning_count} I={self.info_count}) "
            f"in {self.total_lines} lines, {self.duration_ms:.1f}ms"
        )


@dataclass
class MigrationStep:
    """A single step in a migration plan."""
    order: int
    title: str
    description: str
    affected_lines: List[int] = field(default_factory=list)
    difficulty: MigrationDifficulty = MigrationDifficulty.MODERATE
    estimated_hours: float = 0.0
    related_findings: List[str] = field(default_factory=list)
    auto_fixable: bool = False


@dataclass
class MigrationPlan:
    """Full migration plan generated from lint analysis."""
    steps: List[MigrationStep] = field(default_factory=list)
    readiness_score: float = 0.0
    total_findings: int = 0
    critical_blockers: int = 0
    estimated_total_hours: float = 0.0
    summary: str = ""
    category_breakdown: Dict[str, int] = field(default_factory=dict)

    @property
    def step_count(self) -> int:
        return len(self.steps)

    def display(self) -> str:
        lines = [
            f"Migration Plan  (readiness {self.readiness_score:.1f}/100)",
            f"  Findings : {self.total_findings}",
            f"  Blockers : {self.critical_blockers}",
            f"  Est hours: {self.estimated_total_hours:.1f}",
            f"  Steps    : {self.step_count}",
        ]
        for step in self.steps:
            lines.append(
                f"    {step.order}. {step.title} "
                f"[{step.difficulty.value}, ~{step.estimated_hours:.1f}h]"
            )
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Rule definitions
# ---------------------------------------------------------------------------

def _build_rules() -> List[LintRule]:
    """Return the full set of 32 lint rules."""
    return [
        LintRule(
            id="PTR001", name="pointer_arithmetic",
            category=RuleCategory.POINTER, severity=Severity.ERROR,
            description="Raw pointer arithmetic detected",
            rust_equivalent="Use slice indexing or iterator methods",
            auto_fix=AutoFixStatus.PARTIAL,
            difficulty=MigrationDifficulty.MODERATE,
            pattern=r'(?<!\w)(\w+)\s*(?:\+\+|--|\+=|\-=|\+\s+\d+|\-\s+\d+)\s*;',
        ),
        LintRule(
            id="PTR002", name="void_pointer_usage",
            category=RuleCategory.POINTER, severity=Severity.ERROR,
            description="void* usage — erases type information",
            rust_equivalent="Use generics, trait objects (dyn Trait), or *mut u8",
            auto_fix=AutoFixStatus.UNAVAILABLE,
            difficulty=MigrationDifficulty.HARD,
            pattern=r'\bvoid\s*\*',
        ),
        LintRule(
            id="MAC001", name="macro_complexity",
            category=RuleCategory.MACRO, severity=Severity.WARNING,
            description="Complex multi-line or parameterised macro",
            rust_equivalent="Use const fn, inline functions, or macro_rules!",
            auto_fix=AutoFixStatus.PARTIAL,
            difficulty=MigrationDifficulty.MODERATE,
            pattern=r'#\s*define\s+\w+\([^)]*\)',
        ),
        LintRule(
            id="CF001", name="goto_usage",
            category=RuleCategory.CONTROL_FLOW, severity=Severity.ERROR,
            description="goto statement — no direct Rust equivalent",
            rust_equivalent="Use loop labels, break/continue, or Result-based control flow",
            auto_fix=AutoFixStatus.UNAVAILABLE,
            difficulty=MigrationDifficulty.HARD,
            pattern=r'\bgoto\s+\w+\s*;',
        ),
        LintRule(
            id="CF002", name="setjmp_longjmp",
            category=RuleCategory.CONTROL_FLOW, severity=Severity.CRITICAL,
            description="setjmp/longjmp — non-local jumps are UB-prone",
            rust_equivalent="Use Result/Option, panic!/catch_unwind, or anyhow",
            auto_fix=AutoFixStatus.UNAVAILABLE,
            difficulty=MigrationDifficulty.VERY_HARD,
            pattern=r'\b(?:setjmp|longjmp)\s*\(',
        ),
        LintRule(
            id="TYP001", name="variadic_functions",
            category=RuleCategory.TYPE_SYSTEM, severity=Severity.WARNING,
            description="Variadic function (va_list) — type-unsafe",
            rust_equivalent="Use generic functions, trait bounds, or tuples",
            auto_fix=AutoFixStatus.UNAVAILABLE,
            difficulty=MigrationDifficulty.MODERATE,
            pattern=r'\.\.\.\s*\)',
        ),
        LintRule(
            id="PORT001", name="bitfield_portability",
            category=RuleCategory.PORTABILITY, severity=Severity.WARNING,
            description="Bit-field layout is implementation-defined",
            rust_equivalent="Use bitflags crate or manual bit masking",
            auto_fix=AutoFixStatus.PARTIAL,
            difficulty=MigrationDifficulty.MODERATE,
            pattern=r'\b\w+\s*:\s*\d+\s*;',
        ),
        LintRule(
            id="TYP002", name="union_type_punning",
            category=RuleCategory.TYPE_SYSTEM, severity=Severity.ERROR,
            description="Union used for type-punning — UB in C++, fragile in C",
            rust_equivalent="Use transmute (unsafe) or bytemuck crate",
            auto_fix=AutoFixStatus.UNAVAILABLE,
            difficulty=MigrationDifficulty.HARD,
            pattern=r'\bunion\s+\w+\s*\{',
        ),
        LintRule(
            id="MEM001", name="flexible_array_member",
            category=RuleCategory.MEMORY, severity=Severity.ERROR,
            description="Flexible array member — no Rust equivalent struct layout",
            rust_equivalent="Use Vec<T> inside the struct or a DST with custom layout",
            auto_fix=AutoFixStatus.UNAVAILABLE,
            difficulty=MigrationDifficulty.HARD,
            pattern=r'\w+\s+\w+\[\s*\]\s*;',
        ),
        LintRule(
            id="CF003", name="signal_handler",
            category=RuleCategory.CONTROL_FLOW, severity=Severity.CRITICAL,
            description="Signal handler — complex async-signal-safety requirements",
            rust_equivalent="Use signal-hook or ctrlc crate",
            auto_fix=AutoFixStatus.UNAVAILABLE,
            difficulty=MigrationDifficulty.VERY_HARD,
            pattern=r'\bsignal\s*\(\s*SIG\w+',
        ),
        LintRule(
            id="CONC001", name="thread_local_storage",
            category=RuleCategory.CONCURRENCY, severity=Severity.WARNING,
            description="Thread-local storage via __thread or _Thread_local",
            rust_equivalent="Use thread_local! macro or std::thread::LocalKey",
            auto_fix=AutoFixStatus.PARTIAL,
            difficulty=MigrationDifficulty.EASY,
            pattern=r'\b(?:__thread|_Thread_local)\b',
        ),
        LintRule(
            id="PORT002", name="volatile_usage",
            category=RuleCategory.PORTABILITY, severity=Severity.WARNING,
            description="volatile qualifier — semantics differ in Rust",
            rust_equivalent="Use std::ptr::read_volatile / write_volatile",
            auto_fix=AutoFixStatus.PARTIAL,
            difficulty=MigrationDifficulty.MODERATE,
            pattern=r'\bvolatile\b',
        ),
        LintRule(
            id="DEP001", name="register_keyword",
            category=RuleCategory.DEPRECATED, severity=Severity.INFO,
            description="register keyword — ignored by modern compilers",
            rust_equivalent="Remove; Rust compiler handles register allocation",
            auto_fix=AutoFixStatus.AVAILABLE,
            difficulty=MigrationDifficulty.TRIVIAL,
            pattern=r'\bregister\b',
        ),
        LintRule(
            id="PTR003", name="restrict_pointers",
            category=RuleCategory.POINTER, severity=Severity.WARNING,
            description="restrict pointer — aliasing guarantee for optimiser",
            rust_equivalent="Rust references are restrict by default (&mut T)",
            auto_fix=AutoFixStatus.AVAILABLE,
            difficulty=MigrationDifficulty.EASY,
            pattern=r'\brestrict\b',
        ),
        LintRule(
            id="TYP003", name="implicit_int_conversion",
            category=RuleCategory.TYPE_SYSTEM, severity=Severity.WARNING,
            description="Implicit integer width conversion detected",
            rust_equivalent="Use explicit as casts or From/Into traits",
            auto_fix=AutoFixStatus.PARTIAL,
            difficulty=MigrationDifficulty.MODERATE,
            pattern=r'\b(?:short|char)\s+\w+\s*=\s*(?:\w+\s*[\+\-\*\/]|\(\s*int\s*\))',
        ),
        LintRule(
            id="DEP002", name="trigraph_usage",
            category=RuleCategory.DEPRECATED, severity=Severity.INFO,
            description="Trigraph sequence — removed in C23",
            rust_equivalent="Use the actual character directly",
            auto_fix=AutoFixStatus.AVAILABLE,
            difficulty=MigrationDifficulty.TRIVIAL,
            pattern=r'\?\?[=\/\'\(\)!<>\-]',
        ),
        LintRule(
            id="DEP003", name="kr_function_declaration",
            category=RuleCategory.DEPRECATED, severity=Severity.WARNING,
            description="K&R style function declaration without prototypes",
            rust_equivalent="Use standard Rust fn signatures with typed parameters",
            auto_fix=AutoFixStatus.PARTIAL,
            difficulty=MigrationDifficulty.EASY,
            pattern=r'\b\w+\s*\(\s*\)\s*\n\s*(?:int|char|float|double|long|short|unsigned)\s+\w+\s*;',
        ),
        LintRule(
            id="GCC001", name="nested_functions",
            category=RuleCategory.GCC_EXTENSION, severity=Severity.ERROR,
            description="Nested function definition — GCC extension, not standard C",
            rust_equivalent="Use closures (|args| { body }) or nested fn items",
            auto_fix=AutoFixStatus.PARTIAL,
            difficulty=MigrationDifficulty.MODERATE,
            pattern=r'(?:auto|void|int|char|float|double|long)\s+\w+\s*\([^)]*\)\s*\{[^}]*\{',
        ),
        LintRule(
            id="GCC002", name="statement_expressions",
            category=RuleCategory.GCC_EXTENSION, severity=Severity.WARNING,
            description="Statement expression ({ ... }) — GCC extension",
            rust_equivalent="Use block expressions (Rust blocks return a value natively)",
            auto_fix=AutoFixStatus.AVAILABLE,
            difficulty=MigrationDifficulty.EASY,
            pattern=r'\(\s*\{[^}]+\}\s*\)',
        ),
        LintRule(
            id="GCC003", name="typeof_operator",
            category=RuleCategory.GCC_EXTENSION, severity=Severity.INFO,
            description="typeof operator — GCC/C23 extension",
            rust_equivalent="Rust infers types; use explicit type annotations if needed",
            auto_fix=AutoFixStatus.AVAILABLE,
            difficulty=MigrationDifficulty.TRIVIAL,
            pattern=r'\btypeof\s*\(',
        ),
        LintRule(
            id="GCC004", name="computed_goto",
            category=RuleCategory.GCC_EXTENSION, severity=Severity.CRITICAL,
            description="Computed goto (&&label) — GCC extension, no Rust equivalent",
            rust_equivalent="Use match on an enum discriminant or function pointer table",
            auto_fix=AutoFixStatus.UNAVAILABLE,
            difficulty=MigrationDifficulty.VERY_HARD,
            pattern=r'&&\s*\w+\s*[;,\)]|goto\s*\*',
        ),
        LintRule(
            id="MEM002", name="alloca_usage",
            category=RuleCategory.MEMORY, severity=Severity.ERROR,
            description="alloca — stack allocation with no overflow protection",
            rust_equivalent="Use Vec or Box for heap; fixed arrays for stack",
            auto_fix=AutoFixStatus.PARTIAL,
            difficulty=MigrationDifficulty.MODERATE,
            pattern=r'\balloca\s*\(',
        ),
        LintRule(
            id="MEM003", name="vla_usage",
            category=RuleCategory.MEMORY, severity=Severity.ERROR,
            description="Variable-length array — optional in C11, absent in Rust",
            rust_equivalent="Use Vec::with_capacity() or boxed slice",
            auto_fix=AutoFixStatus.PARTIAL,
            difficulty=MigrationDifficulty.MODERATE,
            pattern=r'\b(?:int|char|float|double|long|short|unsigned)\s+\w+\[\s*\w+\s*\]\s*;',
        ),
        LintRule(
            id="TYP004", name="complex_number_support",
            category=RuleCategory.TYPE_SYSTEM, severity=Severity.INFO,
            description="C _Complex / complex.h usage",
            rust_equivalent="Use num-complex crate (Complex<f64>)",
            auto_fix=AutoFixStatus.PARTIAL,
            difficulty=MigrationDifficulty.EASY,
            pattern=r'\b(?:_Complex|_Imaginary)\b|#\s*include\s*<complex\.h>',
        ),
        LintRule(
            id="TYP005", name="decimal_floating_point",
            category=RuleCategory.TYPE_SYSTEM, severity=Severity.INFO,
            description="Decimal floating point type (_Decimal32/64/128)",
            rust_equivalent="Use decimal crate or rust_decimal",
            auto_fix=AutoFixStatus.PARTIAL,
            difficulty=MigrationDifficulty.EASY,
            pattern=r'\b_Decimal(?:32|64|128)\b',
        ),
        LintRule(
            id="CONC002", name="atomic_operations",
            category=RuleCategory.CONCURRENCY, severity=Severity.WARNING,
            description="C11 atomic operations or _Atomic qualifier",
            rust_equivalent="Use std::sync::atomic (AtomicBool, AtomicUsize, etc.)",
            auto_fix=AutoFixStatus.PARTIAL,
            difficulty=MigrationDifficulty.MODERATE,
            pattern=r'\b(?:_Atomic|atomic_\w+)\b|#\s*include\s*<stdatomic\.h>',
        ),
        LintRule(
            id="C11_001", name="generic_selection",
            category=RuleCategory.C11_FEATURE, severity=Severity.WARNING,
            description="_Generic selection expression",
            rust_equivalent="Use trait-based dispatch or macro_rules! with type matching",
            auto_fix=AutoFixStatus.UNAVAILABLE,
            difficulty=MigrationDifficulty.HARD,
            pattern=r'\b_Generic\s*\(',
        ),
        LintRule(
            id="C11_002", name="static_assert",
            category=RuleCategory.C11_FEATURE, severity=Severity.INFO,
            description="_Static_assert or static_assert usage",
            rust_equivalent="Use const { assert!(...) } or static_assertions crate",
            auto_fix=AutoFixStatus.AVAILABLE,
            difficulty=MigrationDifficulty.TRIVIAL,
            pattern=r'\b(?:_Static_assert|static_assert)\s*\(',
        ),
        LintRule(
            id="C11_003", name="alignas_alignof",
            category=RuleCategory.C11_FEATURE, severity=Severity.INFO,
            description="_Alignas / _Alignof usage",
            rust_equivalent="Use #[repr(align(N))] or std::mem::align_of::<T>()",
            auto_fix=AutoFixStatus.AVAILABLE,
            difficulty=MigrationDifficulty.EASY,
            pattern=r'\b(?:_Alignas|_Alignof|alignas|alignof)\s*\(',
        ),
        LintRule(
            id="C11_004", name="noreturn_functions",
            category=RuleCategory.C11_FEATURE, severity=Severity.INFO,
            description="_Noreturn or noreturn function attribute",
            rust_equivalent="Use -> ! (never type) as the return type",
            auto_fix=AutoFixStatus.AVAILABLE,
            difficulty=MigrationDifficulty.TRIVIAL,
            pattern=r'\b(?:_Noreturn|noreturn)\b',
        ),
        LintRule(
            id="CONC003", name="thread_support",
            category=RuleCategory.CONCURRENCY, severity=Severity.WARNING,
            description="C11 threads.h / thrd_create usage",
            rust_equivalent="Use std::thread::spawn or rayon for parallelism",
            auto_fix=AutoFixStatus.PARTIAL,
            difficulty=MigrationDifficulty.MODERATE,
            pattern=r'\bthrd_(?:create|join|detach|exit|sleep)\b|#\s*include\s*<threads\.h>',
        ),
        LintRule(
            id="UB001", name="undefined_behaviour_cast",
            category=RuleCategory.UNDEFINED_BEHAVIOUR, severity=Severity.CRITICAL,
            description="Potentially dangerous cast between incompatible pointer types",
            rust_equivalent="Use safe transmute patterns or bytemuck",
            auto_fix=AutoFixStatus.UNAVAILABLE,
            difficulty=MigrationDifficulty.HARD,
            pattern=r'\(\s*(?:struct\s+)?\w+\s*\*\s*\)\s*(?:&|\w+)',
        ),
    ]


ALL_RULES: List[LintRule] = _build_rules()

_RULE_MAP: Dict[str, LintRule] = {r.id: r for r in ALL_RULES}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _strip_comments(source: str) -> str:
    """Remove C block and line comments while preserving line count."""
    result = re.sub(r'/\*.*?\*/', lambda m: '\n' * m.group(0).count('\n'), source, flags=re.DOTALL)
    result = re.sub(r'//[^\n]*', '', result)
    return result


def _strip_string_literals(source: str) -> str:
    """Replace string and character literals with empty strings."""
    result = re.sub(r'"(?:[^"\\]|\\.)*"', '""', source)
    result = re.sub(r"'(?:[^'\\]|\\.)*'", "''", result)
    return result


def _line_number_of(source: str, pos: int) -> int:
    """Return 1-based line number for a character offset."""
    return source[:pos].count('\n') + 1


def _snippet(source: str, line: int, context: int = 0) -> str:
    """Extract a code snippet around a given 1-based line number."""
    lines = source.splitlines()
    start = max(0, line - 1 - context)
    end = min(len(lines), line + context)
    return '\n'.join(lines[start:end])


def _column_of(source: str, pos: int) -> int:
    """Return 1-based column for a character offset."""
    line_start = source.rfind('\n', 0, pos) + 1
    return pos - line_start + 1


def _count_macro_lines(source: str) -> int:
    """Count lines that are part of preprocessor directives."""
    count = 0
    continuation = False
    for line in source.splitlines():
        stripped = line.strip()
        if continuation or stripped.startswith('#'):
            count += 1
            continuation = stripped.endswith('\\')
        else:
            continuation = False
    return count


def _function_boundaries(source: str) -> List[Tuple[int, int, str]]:
    """Identify top-level function boundaries as (start_line, end_line, name)."""
    results: List[Tuple[int, int, str]] = []
    pattern = re.compile(
        r'^(?:static\s+)?(?:inline\s+)?'
        r'(?:(?:void|int|char|float|double|long|short|unsigned|signed|struct\s+\w+|enum\s+\w+)\s*\*?\s+)'
        r'(\w+)\s*\([^)]*\)\s*\{',
        re.MULTILINE,
    )
    for m in pattern.finditer(source):
        start_line = _line_number_of(source, m.start())
        depth = 1
        pos = m.end()
        while pos < len(source) and depth > 0:
            if source[pos] == '{':
                depth += 1
            elif source[pos] == '}':
                depth -= 1
            pos += 1
        end_line = _line_number_of(source, pos - 1) if pos <= len(source) else start_line
        results.append((start_line, end_line, m.group(1)))
    return results


# ---------------------------------------------------------------------------
# Core linting engine
# ---------------------------------------------------------------------------

def _apply_rule(rule: LintRule, source: str, cleaned: str) -> List[LintFinding]:
    """Apply a single rule against the cleaned source, returning findings."""
    if not rule.enabled or not rule.pattern:
        return []

    findings: List[LintFinding] = []
    try:
        compiled = re.compile(rule.pattern, re.MULTILINE)
    except re.error:
        return []

    for m in compiled.finditer(cleaned):
        line = _line_number_of(source, m.start())
        col = _column_of(source, m.start())
        snippet = _snippet(source, line, context=1)
        findings.append(LintFinding(
            rule_id=rule.id,
            rule_name=rule.name,
            category=rule.category,
            severity=rule.severity,
            description=rule.description,
            rust_equivalent=rule.rust_equivalent,
            auto_fix=rule.auto_fix,
            line_number=line,
            column=col,
            code_snippet=snippet,
            suggestion=rule.rust_equivalent,
            confidence=0.85 if rule.severity in (Severity.INFO, Severity.WARNING) else 0.95,
        ))
    return findings


def lint_for_migration(c_source: str) -> LintReport:
    """Run all migration-readiness lint rules against *c_source*.

    Args:
        c_source: The C source code to analyse.

    Returns:
        A ``LintReport`` containing every finding, summary counts, and
        timing information.
    """
    start = time.monotonic()
    stripped = _strip_string_literals(_strip_comments(c_source))
    total_lines = c_source.count('\n') + 1

    all_findings: List[LintFinding] = []
    rules_applied = 0

    for rule in ALL_RULES:
        if not rule.enabled:
            continue
        rules_applied += 1
        all_findings.extend(_apply_rule(rule, c_source, stripped))

    all_findings.sort(key=lambda f: (f.line_number, f.rule_id))

    elapsed = (time.monotonic() - start) * 1000.0
    return LintReport(
        findings=all_findings,
        total_lines=total_lines,
        rules_applied=rules_applied,
        duration_ms=elapsed,
    )


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

_SEVERITY_WEIGHTS: Dict[Severity, float] = {
    Severity.CRITICAL: 10.0,
    Severity.ERROR: 5.0,
    Severity.WARNING: 2.0,
    Severity.INFO: 0.5,
}

_DIFFICULTY_WEIGHTS: Dict[MigrationDifficulty, float] = {
    MigrationDifficulty.TRIVIAL: 0.5,
    MigrationDifficulty.EASY: 1.0,
    MigrationDifficulty.MODERATE: 2.5,
    MigrationDifficulty.HARD: 5.0,
    MigrationDifficulty.VERY_HARD: 8.0,
}


def migration_readiness_score(c_source: str) -> float:
    """Compute a 0–100 readiness score for migrating *c_source* to Rust.

    A score of 100 means no issues were found; the score decreases with
    the number and severity of findings, normalised by the code length.

    Args:
        c_source: The C source code to analyse.

    Returns:
        A float in [0, 100].
    """
    report = lint_for_migration(c_source)
    if report.total_lines == 0:
        return 100.0

    penalty = 0.0
    for finding in report.findings:
        weight = _SEVERITY_WEIGHTS.get(finding.severity, 1.0)
        rule = _RULE_MAP.get(finding.rule_id)
        diff_weight = 1.0
        if rule is not None:
            diff_weight = _DIFFICULTY_WEIGHTS.get(rule.difficulty, 1.0)
        penalty += weight * diff_weight

    normalised = penalty / max(report.total_lines, 1)
    score = max(0.0, 100.0 - normalised * 10.0)
    return round(score, 2)


# ---------------------------------------------------------------------------
# Rust-equivalent suggestions
# ---------------------------------------------------------------------------

_EXTENDED_SUGGESTIONS: Dict[str, str] = {
    "PTR001": (
        "Replace pointer arithmetic with safe indexing:\n"
        "  // C: *(ptr + i)\n"
        "  // Rust: slice[i]\n"
        "Or use iterators:\n"
        "  for item in slice.iter() { ... }"
    ),
    "PTR002": (
        "Replace void* with generics or trait objects:\n"
        "  // C: void* data;\n"
        "  // Rust: fn process<T>(data: &T) { ... }\n"
        "  // or: fn process(data: &dyn Any) { ... }"
    ),
    "MAC001": (
        "Replace function-like macros with const fn or inline fn:\n"
        "  // C: #define MAX(a,b) ((a)>(b)?(a):(b))\n"
        "  // Rust: fn max<T: Ord>(a: T, b: T) -> T { std::cmp::max(a, b) }\n"
        "For compile-time macros, use macro_rules! { ... }"
    ),
    "CF001": (
        "Replace goto with structured control flow:\n"
        "  // C: goto cleanup;\n"
        "  // Rust: Use ? operator with Result, or 'label: { break 'label; }"
    ),
    "CF002": (
        "Replace setjmp/longjmp with Rust error handling:\n"
        "  // C: if (setjmp(buf)) { handle_error(); }\n"
        "  // Rust: match do_work() { Err(e) => handle_error(e), Ok(v) => v }\n"
        "  // Or: std::panic::catch_unwind(|| { ... })"
    ),
    "TYP001": (
        "Replace variadic functions with type-safe alternatives:\n"
        "  // C: int sum(int count, ...) { va_list ap; ... }\n"
        "  // Rust: fn sum(values: &[i32]) -> i32 { values.iter().sum() }"
    ),
    "PORT001": (
        "Replace bit-fields with the bitflags crate:\n"
        "  // C: struct Flags { unsigned a:1; unsigned b:3; };\n"
        "  // Rust: bitflags! { struct Flags: u8 { const A = 0b0001; } }"
    ),
    "TYP002": (
        "Replace union type-punning with safe transmute:\n"
        "  // C: union { int i; float f; } u; u.i = 42; float r = u.f;\n"
        "  // Rust: let r: f32 = f32::from_bits(42u32);\n"
        "  // Or: use bytemuck::cast::<u32, f32>(42)"
    ),
    "MEM001": (
        "Replace flexible array member with Vec:\n"
        "  // C: struct Packet { int len; char data[]; };\n"
        "  // Rust: struct Packet { len: usize, data: Vec<u8> }"
    ),
    "CF003": (
        "Replace signal() with the signal-hook crate:\n"
        "  // C: signal(SIGINT, handler);\n"
        "  // Rust: signal_hook::iterator::Signals::new(&[SIGINT])?"
    ),
    "CONC001": (
        "Replace __thread with thread_local!:\n"
        "  // C: __thread int counter = 0;\n"
        "  // Rust: thread_local! { static COUNTER: Cell<i32> = Cell::new(0); }"
    ),
    "PORT002": (
        "Replace volatile with atomic or ptr::read_volatile:\n"
        "  // C: volatile int flag;\n"
        "  // Rust: use std::sync::atomic::AtomicI32;\n"
        "  // Or: unsafe { std::ptr::read_volatile(&flag) }"
    ),
    "DEP001": (
        "Simply remove the register keyword:\n"
        "  // C: register int i = 0;\n"
        "  // Rust: let i: i32 = 0;  (compiler handles allocation)"
    ),
    "PTR003": (
        "restrict is the default for &mut T in Rust:\n"
        "  // C: void copy(int *restrict dst, const int *restrict src, int n);\n"
        "  // Rust: fn copy(dst: &mut [i32], src: &[i32]) { ... }"
    ),
    "TYP003": (
        "Use explicit casts in Rust:\n"
        "  // C: short s = some_int;\n"
        "  // Rust: let s: i16 = some_int as i16;\n"
        "  // Or: let s = i16::try_from(some_int)?;"
    ),
    "DEP002": (
        "Replace trigraphs with actual characters:\n"
        "  // C: ??= -> #, ??( -> [, ??) -> ]\n"
        "  // Rust: Trigraphs do not exist; use the characters directly."
    ),
    "DEP003": (
        "Rewrite K&R declarations with explicit parameter types:\n"
        "  // C: int add(a, b) int a; int b; { return a+b; }\n"
        "  // Rust: fn add(a: i32, b: i32) -> i32 { a + b }"
    ),
    "GCC001": (
        "Replace nested functions with closures:\n"
        "  // C (GCC): void outer() { int inner(int x) { return x+1; } }\n"
        "  // Rust: fn outer() { let inner = |x: i32| x + 1; }"
    ),
    "GCC002": (
        "Rust block expressions return values natively:\n"
        "  // C (GCC): int x = ({ int t = f(); t * 2; });\n"
        "  // Rust: let x = { let t = f(); t * 2 };"
    ),
    "GCC003": (
        "Remove typeof; Rust infers types:\n"
        "  // C (GCC): typeof(x) y = x;\n"
        "  // Rust: let y = x;  (or specify the type explicitly)"
    ),
    "GCC004": (
        "Replace computed goto with match on enum:\n"
        "  // C (GCC): void *table[] = {&&L1, &&L2}; goto *table[i];\n"
        "  // Rust: match state { State::L1 => { ... }, State::L2 => { ... } }"
    ),
    "MEM002": (
        "Replace alloca with stack arrays or Vec:\n"
        "  // C: char *buf = alloca(n);\n"
        "  // Rust: let mut buf = vec![0u8; n];"
    ),
    "MEM003": (
        "Replace VLA with Vec:\n"
        "  // C: int arr[n];\n"
        "  // Rust: let arr = vec![0i32; n];"
    ),
    "TYP004": (
        "Use the num-complex crate:\n"
        "  // C: double _Complex z = 1.0 + 2.0*I;\n"
        "  // Rust: let z = Complex::new(1.0, 2.0);"
    ),
    "TYP005": (
        "Use the rust_decimal crate:\n"
        "  // C: _Decimal64 price = 19.99DD;\n"
        "  // Rust: let price = Decimal::from_str(\"19.99\").unwrap();"
    ),
    "CONC002": (
        "Use std::sync::atomic:\n"
        "  // C: _Atomic int counter;\n"
        "  // Rust: let counter = AtomicI32::new(0);"
    ),
    "C11_001": (
        "Replace _Generic with trait dispatch:\n"
        "  // C: _Generic(x, int: \"int\", float: \"float\")\n"
        "  // Rust: implement a trait with specialized impls"
    ),
    "C11_002": (
        "Use const assert:\n"
        "  // C: _Static_assert(sizeof(int)==4, \"need 32-bit int\");\n"
        "  // Rust: const _: () = assert!(std::mem::size_of::<i32>() == 4);"
    ),
    "C11_003": (
        "Use #[repr(align(N))] or std::mem::align_of:\n"
        "  // C: _Alignas(16) int x;\n"
        "  // Rust: #[repr(align(16))] struct Aligned(i32);"
    ),
    "C11_004": (
        "Use the never type:\n"
        "  // C: _Noreturn void die(const char *msg);\n"
        "  // Rust: fn die(msg: &str) -> ! { panic!(\"{}\", msg) }"
    ),
    "CONC003": (
        "Use std::thread:\n"
        "  // C: thrd_create(&t, func, arg);\n"
        "  // Rust: let t = std::thread::spawn(move || func(arg));"
    ),
    "UB001": (
        "Avoid wild casts; use safe transmute:\n"
        "  // C: struct B *b = (struct B *)a_ptr;\n"
        "  // Rust: Use as / From / bytemuck depending on context"
    ),
}


def suggest_rust_equivalent(finding: LintFinding) -> str:
    """Return a detailed Rust-equivalent suggestion for *finding*.

    Args:
        finding: A ``LintFinding`` produced by ``lint_for_migration``.

    Returns:
        A multi-line string with a code-level suggestion.
    """
    extended = _EXTENDED_SUGGESTIONS.get(finding.rule_id)
    if extended is not None:
        return extended
    return finding.rust_equivalent


# ---------------------------------------------------------------------------
# Migration plan generation
# ---------------------------------------------------------------------------

_DIFFICULTY_HOURS: Dict[MigrationDifficulty, float] = {
    MigrationDifficulty.TRIVIAL: 0.1,
    MigrationDifficulty.EASY: 0.5,
    MigrationDifficulty.MODERATE: 2.0,
    MigrationDifficulty.HARD: 5.0,
    MigrationDifficulty.VERY_HARD: 10.0,
}

_PHASE_ORDER: List[Tuple[str, Set[RuleCategory], str]] = [
    (
        "Remove deprecated features",
        {RuleCategory.DEPRECATED},
        "Strip register keywords, trigraphs, and K&R declarations.",
    ),
    (
        "Replace GCC extensions",
        {RuleCategory.GCC_EXTENSION},
        "Rewrite nested functions, statement expressions, typeof, and computed gotos.",
    ),
    (
        "Migrate C11/C23 features",
        {RuleCategory.C11_FEATURE, RuleCategory.C23_FEATURE},
        "Map _Generic, _Static_assert, alignas/alignof, and _Noreturn to Rust idioms.",
    ),
    (
        "Rewrite type-system patterns",
        {RuleCategory.TYPE_SYSTEM},
        "Eliminate void*, union type-punning, variadic functions, and implicit casts.",
    ),
    (
        "Fix pointer and memory issues",
        {RuleCategory.POINTER, RuleCategory.MEMORY},
        "Replace pointer arithmetic, restrict, alloca, VLAs, and flexible array members.",
    ),
    (
        "Restructure control flow",
        {RuleCategory.CONTROL_FLOW},
        "Remove goto/setjmp/longjmp/signal handlers; use Result and iterators.",
    ),
    (
        "Address concurrency patterns",
        {RuleCategory.CONCURRENCY},
        "Migrate atomics, thread-local storage, and C11 threads to Rust equivalents.",
    ),
    (
        "Resolve portability concerns",
        {RuleCategory.PORTABILITY},
        "Fix bit-field layout assumptions and volatile semantics.",
    ),
    (
        "Audit undefined-behaviour casts",
        {RuleCategory.UNDEFINED_BEHAVIOUR},
        "Review and replace dangerous pointer casts with safe alternatives.",
    ),
]


def generate_migration_plan(c_source: str) -> MigrationPlan:
    """Produce a step-by-step migration plan from lint analysis.

    Args:
        c_source: The C source code to analyse.

    Returns:
        A ``MigrationPlan`` with ordered steps, difficulty estimates, and
        a readiness score.
    """
    report = lint_for_migration(c_source)
    score = migration_readiness_score(c_source)

    category_counts: Dict[str, int] = {}
    for finding in report.findings:
        key = finding.category.value
        category_counts[key] = category_counts.get(key, 0) + 1

    steps: List[MigrationStep] = []
    order = 1
    for title, categories, description in _PHASE_ORDER:
        phase_findings = [
            f for f in report.findings if f.category in categories
        ]
        if not phase_findings:
            continue

        affected_lines = sorted({f.line_number for f in phase_findings})
        related_ids = [f.id for f in phase_findings]

        max_difficulty = MigrationDifficulty.TRIVIAL
        for f in phase_findings:
            rule = _RULE_MAP.get(f.rule_id)
            if rule is not None and rule.difficulty.value > max_difficulty.value:
                max_difficulty = rule.difficulty

        auto_fixable = all(
            f.auto_fix == AutoFixStatus.AVAILABLE for f in phase_findings
        )

        est_hours = 0.0
        for f in phase_findings:
            rule = _RULE_MAP.get(f.rule_id)
            diff = rule.difficulty if rule else MigrationDifficulty.MODERATE
            est_hours += _DIFFICULTY_HOURS.get(diff, 2.0)

        steps.append(MigrationStep(
            order=order,
            title=title,
            description=description,
            affected_lines=affected_lines,
            difficulty=max_difficulty,
            estimated_hours=round(est_hours, 1),
            related_findings=related_ids,
            auto_fixable=auto_fixable,
        ))
        order += 1

    total_hours = sum(s.estimated_hours for s in steps)
    critical = report.critical_count

    if critical > 0:
        summary_text = (
            f"Migration is BLOCKED by {critical} critical issue(s). "
            f"Resolve them before proceeding."
        )
    elif score >= 80:
        summary_text = (
            f"Code is largely ready for migration (score {score:.1f}/100). "
            f"Address the remaining {len(report.findings)} findings."
        )
    elif score >= 50:
        summary_text = (
            f"Moderate migration effort required (score {score:.1f}/100). "
            f"Plan for ~{total_hours:.0f} hours of work."
        )
    else:
        summary_text = (
            f"Significant refactoring needed (score {score:.1f}/100). "
            f"Consider an incremental migration strategy."
        )

    return MigrationPlan(
        steps=steps,
        readiness_score=score,
        total_findings=len(report.findings),
        critical_blockers=critical,
        estimated_total_hours=round(total_hours, 1),
        summary=summary_text,
        category_breakdown=category_counts,
    )


# ---------------------------------------------------------------------------
# Batch / multi-file helpers
# ---------------------------------------------------------------------------

def lint_files(file_paths: List[str]) -> Dict[str, LintReport]:
    """Lint multiple C files and return per-file reports.

    Args:
        file_paths: Paths to C source files.

    Returns:
        A dict mapping each path to its ``LintReport``.
    """
    results: Dict[str, LintReport] = {}
    for path in file_paths:
        try:
            with open(path, 'r', encoding='utf-8', errors='replace') as fh:
                source = fh.read()
            results[path] = lint_for_migration(source)
        except OSError as exc:
            report = LintReport()
            report.findings.append(LintFinding(
                rule_id="IO_ERR",
                rule_name="file_read_error",
                category=RuleCategory.PORTABILITY,
                severity=Severity.ERROR,
                description=f"Could not read file: {exc}",
                rust_equivalent="N/A",
                auto_fix=AutoFixStatus.UNAVAILABLE,
                line_number=0,
            ))
            results[path] = report
    return results


def aggregate_reports(reports: Dict[str, LintReport]) -> LintReport:
    """Merge multiple per-file reports into one aggregate report.

    Args:
        reports: Mapping of file path → ``LintReport``.

    Returns:
        A single ``LintReport`` combining all findings.
    """
    combined = LintReport()
    for path, report in reports.items():
        for finding in report.findings:
            tagged = LintFinding(
                rule_id=finding.rule_id,
                rule_name=finding.rule_name,
                category=finding.category,
                severity=finding.severity,
                description=f"[{path}] {finding.description}",
                rust_equivalent=finding.rust_equivalent,
                auto_fix=finding.auto_fix,
                line_number=finding.line_number,
                column=finding.column,
                code_snippet=finding.code_snippet,
                suggestion=finding.suggestion,
                confidence=finding.confidence,
            )
            combined.findings.append(tagged)
        combined.total_lines += report.total_lines
        combined.rules_applied = max(combined.rules_applied, report.rules_applied)
        combined.duration_ms += report.duration_ms
    combined.findings.sort(key=lambda f: f.description)
    return combined


# ---------------------------------------------------------------------------
# Rule query utilities
# ---------------------------------------------------------------------------

def get_rule(rule_id: str) -> Optional[LintRule]:
    """Look up a rule by its identifier.

    Args:
        rule_id: e.g. ``"PTR001"``.

    Returns:
        The matching ``LintRule`` or ``None``.
    """
    return _RULE_MAP.get(rule_id)


def list_rules(
    category: Optional[RuleCategory] = None,
    severity: Optional[Severity] = None,
    enabled_only: bool = True,
) -> List[LintRule]:
    """Return rules matching the given filters.

    Args:
        category: If set, only rules in this category.
        severity: If set, only rules at this severity.
        enabled_only: Skip disabled rules (default ``True``).

    Returns:
        A list of matching ``LintRule`` objects.
    """
    result: List[LintRule] = []
    for rule in ALL_RULES:
        if enabled_only and not rule.enabled:
            continue
        if category is not None and rule.category != category:
            continue
        if severity is not None and rule.severity != severity:
            continue
        result.append(rule)
    return result


def disable_rules(rule_ids: Set[str]) -> int:
    """Disable the given rules so they are skipped during linting.

    Args:
        rule_ids: Set of rule IDs to disable.

    Returns:
        Number of rules actually disabled.
    """
    count = 0
    for rule in ALL_RULES:
        if rule.id in rule_ids and rule.enabled:
            rule.enabled = False
            count += 1
    return count


def enable_rules(rule_ids: Set[str]) -> int:
    """Re-enable previously disabled rules.

    Args:
        rule_ids: Set of rule IDs to enable.

    Returns:
        Number of rules actually enabled.
    """
    count = 0
    for rule in ALL_RULES:
        if rule.id in rule_ids and not rule.enabled:
            rule.enabled = True
            count += 1
    return count


# ---------------------------------------------------------------------------
# Formatting / output
# ---------------------------------------------------------------------------

def format_report_text(report: LintReport) -> str:
    """Format a ``LintReport`` as a human-readable text block.

    Args:
        report: The lint report to format.

    Returns:
        A multi-line string.
    """
    lines: List[str] = [report.summary(), ""]

    by_sev = report.by_severity
    for sev in (Severity.CRITICAL, Severity.ERROR, Severity.WARNING, Severity.INFO):
        group = by_sev.get(sev, [])
        if not group:
            continue
        lines.append(f"--- {sev.value.upper()} ({len(group)}) ---")
        for f in group:
            lines.append(f"  L{f.line_number}: [{f.rule_id}] {f.description}")
            if f.code_snippet:
                for snippet_line in f.code_snippet.splitlines():
                    lines.append(f"    | {snippet_line}")
            lines.append(f"    -> {f.suggestion}")
            lines.append("")
    return "\n".join(lines)


def format_report_json(report: LintReport) -> Dict:
    """Serialise a ``LintReport`` to a JSON-ready dictionary.

    Args:
        report: The lint report to serialise.

    Returns:
        A dict suitable for ``json.dumps()``.
    """
    return {
        "summary": report.summary(),
        "total_lines": report.total_lines,
        "rules_applied": report.rules_applied,
        "duration_ms": round(report.duration_ms, 2),
        "counts": {
            "critical": report.critical_count,
            "error": report.error_count,
            "warning": report.warning_count,
            "info": report.info_count,
            "total": len(report.findings),
        },
        "findings": [
            {
                "rule_id": f.rule_id,
                "rule_name": f.rule_name,
                "category": f.category.value,
                "severity": f.severity.value,
                "line": f.line_number,
                "column": f.column,
                "description": f.description,
                "suggestion": f.suggestion,
                "auto_fix": f.auto_fix.value,
                "confidence": f.confidence,
                "snippet": f.code_snippet,
            }
            for f in report.findings
        ],
    }


def format_plan_json(plan: MigrationPlan) -> Dict:
    """Serialise a ``MigrationPlan`` to a JSON-ready dictionary.

    Args:
        plan: The migration plan to serialise.

    Returns:
        A dict suitable for ``json.dumps()``.
    """
    return {
        "readiness_score": plan.readiness_score,
        "total_findings": plan.total_findings,
        "critical_blockers": plan.critical_blockers,
        "estimated_total_hours": plan.estimated_total_hours,
        "summary": plan.summary,
        "category_breakdown": plan.category_breakdown,
        "steps": [
            {
                "order": s.order,
                "title": s.title,
                "description": s.description,
                "difficulty": s.difficulty.value,
                "estimated_hours": s.estimated_hours,
                "affected_lines": s.affected_lines,
                "auto_fixable": s.auto_fixable,
                "related_finding_count": len(s.related_findings),
            }
            for s in plan.steps
        ],
    }
