"""Multi-language equivalence checking beyond C↔Rust.

Extends XEquiv to support C↔Go, C↔Zig, and provides language-specific
semantic models and common divergence patterns per language pair.
"""

import re
import time
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Set
from enum import Enum

from .api import verify_equivalence, VerificationResult, Divergence, Counterexample


# ---------------------------------------------------------------------------
# Language definitions
# ---------------------------------------------------------------------------

class Language(Enum):
    C = "c"
    RUST = "rust"
    GO = "go"
    ZIG = "zig"


@dataclass
class TypeMapping:
    source_type: str
    target_type: str
    exact: bool = True
    notes: str = ""


@dataclass
class DivergencePattern:
    category: str
    source_lang: Language
    target_lang: Language
    description: str
    source_behavior: str
    target_behavior: str
    severity: str
    example_source: str = ""
    example_target: str = ""
    detection_regex: Optional[str] = None


@dataclass
class LanguagePairConfig:
    source: Language
    target: Language
    type_mappings: List[TypeMapping]
    divergence_patterns: List[DivergencePattern]
    function_regex: str
    param_parser: str  # key into _PARAM_PARSERS


@dataclass
class MultiLangVerificationResult:
    source_lang: Language
    target_lang: Language
    result: VerificationResult
    detected_patterns: List[DivergencePattern]
    warnings: List[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Type mappings per language pair
# ---------------------------------------------------------------------------

C_TO_RUST_TYPES: List[TypeMapping] = [
    TypeMapping("int", "i32"), TypeMapping("unsigned int", "u32"),
    TypeMapping("long", "i64"), TypeMapping("unsigned long", "u64"),
    TypeMapping("short", "i16"), TypeMapping("unsigned short", "u16"),
    TypeMapping("char", "i8"), TypeMapping("unsigned char", "u8"),
    TypeMapping("float", "f32"), TypeMapping("double", "f64"),
    TypeMapping("void", "()", notes="Return type only"),
    TypeMapping("bool", "bool"), TypeMapping("_Bool", "bool"),
    TypeMapping("size_t", "usize"), TypeMapping("ssize_t", "isize"),
    TypeMapping("int8_t", "i8"), TypeMapping("int16_t", "i16"),
    TypeMapping("int32_t", "i32"), TypeMapping("int64_t", "i64"),
    TypeMapping("uint8_t", "u8"), TypeMapping("uint16_t", "u16"),
    TypeMapping("uint32_t", "u32"), TypeMapping("uint64_t", "u64"),
]

C_TO_GO_TYPES: List[TypeMapping] = [
    TypeMapping("int", "int32", notes="Go int is platform-dependent; int32 is safer"),
    TypeMapping("unsigned int", "uint32"),
    TypeMapping("long", "int64"), TypeMapping("unsigned long", "uint64"),
    TypeMapping("short", "int16"), TypeMapping("unsigned short", "uint16"),
    TypeMapping("char", "byte", exact=False, notes="Go byte is uint8; C char may be signed"),
    TypeMapping("unsigned char", "byte"),
    TypeMapping("float", "float32"), TypeMapping("double", "float64"),
    TypeMapping("void", "", notes="No return type in Go"),
    TypeMapping("bool", "bool"), TypeMapping("_Bool", "bool"),
    TypeMapping("size_t", "uint", exact=False),
    TypeMapping("int8_t", "int8"), TypeMapping("int16_t", "int16"),
    TypeMapping("int32_t", "int32"), TypeMapping("int64_t", "int64"),
    TypeMapping("uint8_t", "uint8"), TypeMapping("uint16_t", "uint16"),
    TypeMapping("uint32_t", "uint32"), TypeMapping("uint64_t", "uint64"),
]

C_TO_ZIG_TYPES: List[TypeMapping] = [
    TypeMapping("int", "i32"), TypeMapping("unsigned int", "u32"),
    TypeMapping("long", "i64"), TypeMapping("unsigned long", "u64"),
    TypeMapping("short", "i16"), TypeMapping("unsigned short", "u16"),
    TypeMapping("char", "i8"), TypeMapping("unsigned char", "u8"),
    TypeMapping("float", "f32"), TypeMapping("double", "f64"),
    TypeMapping("void", "void"),
    TypeMapping("bool", "bool"), TypeMapping("_Bool", "bool"),
    TypeMapping("size_t", "usize"), TypeMapping("ssize_t", "isize"),
    TypeMapping("int8_t", "i8"), TypeMapping("int16_t", "i16"),
    TypeMapping("int32_t", "i32"), TypeMapping("int64_t", "i64"),
    TypeMapping("uint8_t", "u8"), TypeMapping("uint16_t", "u16"),
    TypeMapping("uint32_t", "u32"), TypeMapping("uint64_t", "u64"),
]


# ---------------------------------------------------------------------------
# Divergence pattern databases
# ---------------------------------------------------------------------------

C_RUST_PATTERNS: List[DivergencePattern] = [
    DivergencePattern(
        "integer_overflow", Language.C, Language.RUST,
        "Signed integer overflow: C is UB, Rust panics (debug) or wraps (release)",
        "Undefined behavior", "Panic in debug / wrapping in release", "critical",
        "int f(int a, int b) { return a + b; }",
        "fn f(a: i32, b: i32) -> i32 { a + b }",
        r"[+\-*]\s*(?!wrapping_)",
    ),
    DivergencePattern(
        "division_by_zero", Language.C, Language.RUST,
        "Division by zero: C is UB, Rust panics",
        "Undefined behavior", "Panic", "critical",
        detection_regex=r"/\s*\w+",
    ),
    DivergencePattern(
        "shift_semantics", Language.C, Language.RUST,
        "Shift by >= bitwidth: C is UB, Rust panics",
        "Undefined behavior", "Panic", "critical",
        detection_regex=r"<<|>>",
    ),
    DivergencePattern(
        "negation_overflow", Language.C, Language.RUST,
        "Negation of INT_MIN: C is UB, Rust panics",
        "Undefined behavior", "Panic", "critical",
        detection_regex=r"-\s*\w+",
    ),
    DivergencePattern(
        "float_precision", Language.C, Language.RUST,
        "Float precision: C is implementation-defined, Rust is IEEE 754 strict",
        "Implementation-defined", "IEEE 754 strict", "warning",
    ),
    DivergencePattern(
        "unsigned_wrap", Language.C, Language.RUST,
        "Unsigned subtraction: both wrap, but Rust requires explicit wrapping_sub in some contexts",
        "Wraps (defined)", "Wraps (defined)", "info",
    ),
]

C_GO_PATTERNS: List[DivergencePattern] = [
    DivergencePattern(
        "integer_overflow", Language.C, Language.GO,
        "Signed integer overflow: C is UB, Go silently wraps (two's complement)",
        "Undefined behavior", "Silent wrap (two's complement)", "critical",
        "int f(int a, int b) { return a + b; }",
        "func f(a, b int32) int32 { return a + b }",
        r"[+\-*]",
    ),
    DivergencePattern(
        "division_by_zero", Language.C, Language.GO,
        "Division by zero: C is UB, Go panics at runtime",
        "Undefined behavior", "Runtime panic", "critical",
        detection_regex=r"/\s*\w+",
    ),
    DivergencePattern(
        "integer_conversion", Language.C, Language.GO,
        "Go requires explicit integer conversions; C promotes implicitly",
        "Implicit promotion", "Explicit conversion required", "warning",
    ),
    DivergencePattern(
        "shift_semantics", Language.C, Language.GO,
        "Shift by >= bitwidth: C is UB, Go masks shift count (Go 1.13+: shift count must be unsigned)",
        "Undefined behavior", "Masked shift count", "critical",
        detection_regex=r"<<|>>",
    ),
    DivergencePattern(
        "pointer_arithmetic", Language.C, Language.GO,
        "Go has no pointer arithmetic (outside unsafe package)",
        "Raw pointer math", "No pointer arithmetic", "critical",
    ),
    DivergencePattern(
        "null_semantics", Language.C, Language.GO,
        "C NULL vs Go nil: different semantics for zero values",
        "NULL pointer dereference is UB", "nil dereference panics", "warning",
    ),
    DivergencePattern(
        "array_bounds", Language.C, Language.GO,
        "Array out-of-bounds: C is UB, Go panics",
        "Undefined behavior", "Runtime panic", "critical",
    ),
]

C_ZIG_PATTERNS: List[DivergencePattern] = [
    DivergencePattern(
        "integer_overflow", Language.C, Language.ZIG,
        "Signed integer overflow: C is UB, Zig has safety-checked undefined behavior (panic in safe, UB in ReleaseFast)",
        "Undefined behavior", "Safety-checked UB (panic in safe builds)", "critical",
        "int f(int a, int b) { return a + b; }",
        "fn f(a: i32, b: i32) i32 { return a + b; }",
        r"[+\-*]",
    ),
    DivergencePattern(
        "division_by_zero", Language.C, Language.ZIG,
        "Division by zero: C is UB, Zig panics in safe mode",
        "Undefined behavior", "Safety panic", "critical",
    ),
    DivergencePattern(
        "optional_semantics", Language.C, Language.ZIG,
        "Zig uses optionals instead of null pointers",
        "NULL pointer", "Optional type (?T)", "warning",
    ),
    DivergencePattern(
        "comptime_evaluation", Language.C, Language.ZIG,
        "Zig comptime evaluation may differ from C preprocessor/constexpr",
        "Preprocessor macros", "comptime evaluation", "info",
    ),
    DivergencePattern(
        "sentinel_terminated", Language.C, Language.ZIG,
        "C null-terminated strings vs Zig sentinel-terminated slices",
        "char* with \\0 terminator", "[:0]const u8 (sentinel-terminated slice)", "warning",
    ),
]


# ---------------------------------------------------------------------------
# Go function parser
# ---------------------------------------------------------------------------

_GO_FUNC_RE = re.compile(
    r"func\s+(?:\([^)]*\)\s+)?(?P<name>[a-zA-Z_]\w*)\s*"
    r"\((?P<params>[^)]*)\)\s*"
    r"(?P<ret>[^{]*?)\s*\{",
    re.MULTILINE,
)


def _parse_go_params(raw: str) -> List[Tuple[str, str]]:
    raw = raw.strip()
    if not raw:
        return []
    params: List[Tuple[str, str]] = []
    for part in raw.split(","):
        part = part.strip()
        tokens = part.rsplit(None, 1)
        if len(tokens) == 2:
            params.append((tokens[0], tokens[1]))
        elif tokens:
            params.append((tokens[0], ""))
    return params


def extract_go_functions(source: str) -> List[Dict]:
    functions = []
    for m in _GO_FUNC_RE.finditer(source):
        functions.append({
            "name": m.group("name"),
            "params": _parse_go_params(m.group("params")),
            "return_type": m.group("ret").strip(),
            "source": _extract_block(source, m.end() - 1),
            "line": source[:m.start()].count("\n") + 1,
        })
    return functions


# ---------------------------------------------------------------------------
# Zig function parser
# ---------------------------------------------------------------------------

_ZIG_FUNC_RE = re.compile(
    r"(?:pub\s+)?fn\s+(?P<name>[a-zA-Z_]\w*)\s*"
    r"\((?P<params>[^)]*)\)\s*"
    r"(?P<ret>[^{]*?)\s*\{",
    re.MULTILINE,
)


def _parse_zig_params(raw: str) -> List[Tuple[str, str]]:
    raw = raw.strip()
    if not raw:
        return []
    params: List[Tuple[str, str]] = []
    for part in raw.split(","):
        part = part.strip()
        if ":" in part:
            name, typ = part.split(":", 1)
            params.append((name.strip(), typ.strip()))
        elif part:
            params.append((part, ""))
    return params


def extract_zig_functions(source: str) -> List[Dict]:
    functions = []
    for m in _ZIG_FUNC_RE.finditer(source):
        functions.append({
            "name": m.group("name"),
            "params": _parse_zig_params(m.group("params")),
            "return_type": m.group("ret").strip(),
            "source": _extract_block(source, m.end() - 1),
            "line": source[:m.start()].count("\n") + 1,
        })
    return functions


def _extract_block(source: str, brace_pos: int) -> str:
    depth = 0
    i = brace_pos
    while i < len(source):
        if source[i] == "{":
            depth += 1
        elif source[i] == "}":
            depth -= 1
            if depth == 0:
                return source[brace_pos:i + 1]
        i += 1
    return source[brace_pos:]


# ---------------------------------------------------------------------------
# Semantic models
# ---------------------------------------------------------------------------

@dataclass
class IntegerSemantics:
    """Model of integer arithmetic semantics for a language."""
    signed_overflow: str      # "ub", "wrap", "panic", "saturate", "checked_ub"
    unsigned_overflow: str    # "wrap", "panic", "checked_ub"
    division_by_zero: str     # "ub", "panic", "nan"
    shift_overflow: str       # "ub", "mask", "panic", "checked_ub"
    implicit_promotion: bool
    negation_min_overflow: str


@dataclass
class LanguageSemanticModel:
    language: Language
    integers: IntegerSemantics
    null_safety: str          # "unsafe", "option", "nil_panic", "optional"
    array_bounds: str         # "unchecked", "checked_panic", "checked_optional"
    string_model: str         # "null_terminated", "length_prefixed", "slice"


_SEMANTIC_MODELS: Dict[Language, LanguageSemanticModel] = {
    Language.C: LanguageSemanticModel(
        language=Language.C,
        integers=IntegerSemantics(
            signed_overflow="ub", unsigned_overflow="wrap",
            division_by_zero="ub", shift_overflow="ub",
            implicit_promotion=True, negation_min_overflow="ub",
        ),
        null_safety="unsafe", array_bounds="unchecked",
        string_model="null_terminated",
    ),
    Language.RUST: LanguageSemanticModel(
        language=Language.RUST,
        integers=IntegerSemantics(
            signed_overflow="panic", unsigned_overflow="panic",
            division_by_zero="panic", shift_overflow="panic",
            implicit_promotion=False, negation_min_overflow="panic",
        ),
        null_safety="option", array_bounds="checked_panic",
        string_model="slice",
    ),
    Language.GO: LanguageSemanticModel(
        language=Language.GO,
        integers=IntegerSemantics(
            signed_overflow="wrap", unsigned_overflow="wrap",
            division_by_zero="panic", shift_overflow="mask",
            implicit_promotion=False, negation_min_overflow="wrap",
        ),
        null_safety="nil_panic", array_bounds="checked_panic",
        string_model="length_prefixed",
    ),
    Language.ZIG: LanguageSemanticModel(
        language=Language.ZIG,
        integers=IntegerSemantics(
            signed_overflow="checked_ub", unsigned_overflow="checked_ub",
            division_by_zero="checked_ub", shift_overflow="checked_ub",
            implicit_promotion=False, negation_min_overflow="checked_ub",
        ),
        null_safety="optional", array_bounds="checked_panic",
        string_model="slice",
    ),
}


def get_semantic_model(lang: Language) -> LanguageSemanticModel:
    return _SEMANTIC_MODELS[lang]


def compare_semantics(source_lang: Language,
                      target_lang: Language) -> List[str]:
    """Compare integer/safety semantics between two languages and list differences."""
    src = _SEMANTIC_MODELS[source_lang]
    tgt = _SEMANTIC_MODELS[target_lang]
    diffs: List[str] = []

    int_fields = [
        ("signed_overflow", "Signed overflow"),
        ("unsigned_overflow", "Unsigned overflow"),
        ("division_by_zero", "Division by zero"),
        ("shift_overflow", "Shift overflow"),
        ("negation_min_overflow", "Negation of MIN"),
    ]
    for attr, label in int_fields:
        sv = getattr(src.integers, attr)
        tv = getattr(tgt.integers, attr)
        if sv != tv:
            diffs.append(f"{label}: {source_lang.value}={sv}, {target_lang.value}={tv}")

    if src.integers.implicit_promotion != tgt.integers.implicit_promotion:
        diffs.append(f"Implicit promotion: {source_lang.value}={src.integers.implicit_promotion}, "
                     f"{target_lang.value}={tgt.integers.implicit_promotion}")
    if src.null_safety != tgt.null_safety:
        diffs.append(f"Null safety: {source_lang.value}={src.null_safety}, {target_lang.value}={tgt.null_safety}")
    if src.array_bounds != tgt.array_bounds:
        diffs.append(f"Array bounds: {source_lang.value}={src.array_bounds}, {target_lang.value}={tgt.array_bounds}")
    if src.string_model != tgt.string_model:
        diffs.append(f"String model: {source_lang.value}={src.string_model}, {target_lang.value}={tgt.string_model}")

    return diffs


# ---------------------------------------------------------------------------
# Pattern-based static analysis
# ---------------------------------------------------------------------------

def detect_divergence_patterns(source_code: str,
                               source_lang: Language,
                               target_lang: Language) -> List[DivergencePattern]:
    """Scan source code for patterns known to cause divergences with the target language."""
    patterns_db = {
        (Language.C, Language.RUST): C_RUST_PATTERNS,
        (Language.C, Language.GO): C_GO_PATTERNS,
        (Language.C, Language.ZIG): C_ZIG_PATTERNS,
    }
    patterns = patterns_db.get((source_lang, target_lang), [])
    detected: List[DivergencePattern] = []

    for pattern in patterns:
        if pattern.detection_regex:
            if re.search(pattern.detection_regex, source_code):
                detected.append(pattern)
        else:
            # Heuristic: check if the category keyword appears in code context
            keywords = {
                "integer_overflow": ["+", "-", "*"],
                "division_by_zero": ["/", "%"],
                "shift_semantics": ["<<", ">>"],
                "pointer_arithmetic": ["->", ".*", "["],
                "array_bounds": ["["],
                "null_semantics": ["NULL", "nil", "null"],
            }
            kws = keywords.get(pattern.category, [])
            if any(kw in source_code for kw in kws):
                detected.append(pattern)

    return detected


# ---------------------------------------------------------------------------
# Multi-language verification
# ---------------------------------------------------------------------------

def verify_multilang(source_code: str, target_code: str,
                     source_lang: Language, target_lang: Language,
                     timeout_s: float = 120.0,
                     method: str = "hybrid") -> MultiLangVerificationResult:
    """Verify equivalence between source and target code in any supported language pair.

    Currently delegates to XEquiv's core verify_equivalence for the actual SMT/fuzz
    check. Adds language-specific pattern detection and semantic warnings.
    """
    warnings: List[str] = []

    # Semantic comparison warnings
    sem_diffs = compare_semantics(source_lang, target_lang)
    for diff in sem_diffs:
        warnings.append(f"Semantic difference: {diff}")

    # Pattern-based pre-scan
    detected = detect_divergence_patterns(source_code, source_lang, target_lang)

    # Core verification (works for C↔Rust natively; for other pairs we
    # still run it since the SMT/fuzz engine operates on the code structure)
    result = verify_equivalence(source_code, target_code,
                                timeout_s=timeout_s, method=method)

    # Enrich result with pattern-based findings not already in divergences
    existing_cats = {d.category for d in result.divergences}
    for pat in detected:
        if pat.category not in existing_cats and not result.equivalent:
            result.divergences.append(Divergence(
                category=pat.category,
                description=pat.description,
                c_behavior=pat.source_behavior,
                rust_behavior=pat.target_behavior,
                severity=pat.severity,
            ))

    return MultiLangVerificationResult(
        source_lang=source_lang,
        target_lang=target_lang,
        result=result,
        detected_patterns=detected,
        warnings=warnings,
    )


def get_type_mappings(source_lang: Language,
                      target_lang: Language) -> List[TypeMapping]:
    """Return type mapping table for a language pair."""
    mapping_db = {
        (Language.C, Language.RUST): C_TO_RUST_TYPES,
        (Language.C, Language.GO): C_TO_GO_TYPES,
        (Language.C, Language.ZIG): C_TO_ZIG_TYPES,
    }
    return mapping_db.get((source_lang, target_lang), [])


def get_divergence_patterns(source_lang: Language,
                            target_lang: Language) -> List[DivergencePattern]:
    """Return known divergence patterns for a language pair."""
    patterns_db = {
        (Language.C, Language.RUST): C_RUST_PATTERNS,
        (Language.C, Language.GO): C_GO_PATTERNS,
        (Language.C, Language.ZIG): C_ZIG_PATTERNS,
    }
    return patterns_db.get((source_lang, target_lang), [])


def supported_pairs() -> List[Tuple[Language, Language]]:
    """Return list of supported language pairs."""
    return [
        (Language.C, Language.RUST),
        (Language.C, Language.GO),
        (Language.C, Language.ZIG),
    ]
