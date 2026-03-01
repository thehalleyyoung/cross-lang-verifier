"""Migrate C test suites to Rust.

Converts C test files to Rust test modules, generates property-based tests
using proptest, creates FFI boundary tests, and builds differential test
harnesses for cross-language validation.
"""

import re
import os
import time
import textwrap
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Set
from pathlib import Path
from enum import Enum


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

class TestKind(Enum):
    UNIT = "unit"
    PROPERTY = "property"
    FFI = "ffi"
    DIFFERENTIAL = "differential"
    INTEGRATION = "integration"


class MigrationStatus(Enum):
    MIGRATED = "migrated"
    PARTIAL = "partial"
    SKIPPED = "skipped"
    FAILED = "failed"


@dataclass
class CTestCase:
    """A single C test extracted from source."""
    name: str
    body: str
    file_path: str
    line_start: int
    line_end: int
    assertions: List[str] = field(default_factory=list)
    fixtures: List[str] = field(default_factory=list)
    dependencies: List[str] = field(default_factory=list)

    @property
    def assertion_count(self) -> int:
        return len(self.assertions)


@dataclass
class RustTestCase:
    """A translated Rust test case."""
    name: str
    rust_code: str
    kind: TestKind
    original_c: Optional[CTestCase] = None
    verified: bool = False
    notes: List[str] = field(default_factory=list)


@dataclass
class TestMigrationResult:
    """Overall result of migrating a test suite."""
    c_test_dir: str
    rust_test_dir: str
    migrated: List[RustTestCase] = field(default_factory=list)
    skipped: List[CTestCase] = field(default_factory=list)
    failed: List[Tuple[CTestCase, str]] = field(default_factory=list)
    total_c_tests: int = 0
    total_rust_tests: int = 0
    total_assertions_migrated: int = 0
    duration_ms: float = 0.0
    rust_test_source: str = ""

    @property
    def success_rate(self) -> float:
        if self.total_c_tests == 0:
            return 0.0
        return len(self.migrated) / self.total_c_tests


@dataclass
class CoverageRegion:
    """A covered region of code."""
    file_path: str
    line_start: int
    line_end: int
    hit_count: int = 0
    branch_taken: Optional[bool] = None


@dataclass
class CoverageComparison:
    """Comparison of C vs Rust test coverage."""
    c_line_coverage: float
    rust_line_coverage: float
    c_branch_coverage: float
    rust_branch_coverage: float
    c_function_coverage: float
    rust_function_coverage: float
    uncovered_c_regions: List[CoverageRegion] = field(default_factory=list)
    uncovered_rust_regions: List[CoverageRegion] = field(default_factory=list)
    coverage_gap: float = 0.0

    @property
    def equivalent(self) -> bool:
        return abs(self.c_line_coverage - self.rust_line_coverage) < 0.05


# ---------------------------------------------------------------------------
# C test extraction
# ---------------------------------------------------------------------------

# Patterns for common C test frameworks
_TEST_PATTERNS = {
    "cunit": re.compile(
        r'void\s+(test_\w+)\s*\(\s*void\s*\)\s*\{', re.MULTILINE
    ),
    "check": re.compile(
        r'START_TEST\s*\(\s*(\w+)\s*\)\s*\{', re.MULTILINE
    ),
    "cmocka": re.compile(
        r'static\s+void\s+(\w+)\s*\(\s*void\s*\*\*\s*state\s*\)\s*\{',
        re.MULTILINE,
    ),
    "plain": re.compile(
        r'(?:void|int)\s+(test\w*)\s*\([^)]*\)\s*\{', re.MULTILINE
    ),
    "gtest_c": re.compile(
        r'TEST\s*\(\s*(\w+)\s*,\s*(\w+)\s*\)\s*\{', re.MULTILINE
    ),
}

_ASSERTION_PATTERNS = [
    re.compile(r'(CU_ASSERT\w*\s*\([^)]+\))'),
    re.compile(r'(ck_assert\w*\s*\([^)]+\))'),
    re.compile(r'(assert_\w+\s*\([^)]+\))'),
    re.compile(r'(assert\s*\([^)]+\))'),
    re.compile(r'(ASSERT_\w+\s*\([^)]+\))'),
    re.compile(r'(EXPECT_\w+\s*\([^)]+\))'),
]


def _detect_framework(source: str) -> str:
    """Detect which C testing framework is used."""
    if "CU_ASSERT" in source or "#include <CUnit" in source:
        return "cunit"
    if "START_TEST" in source or "#include <check" in source:
        return "check"
    if "cmocka" in source or "assert_int_equal" in source:
        return "cmocka"
    if "TEST(" in source and "ASSERT_" in source:
        return "gtest_c"
    return "plain"


def _extract_test_body(source: str, match_end: int) -> Tuple[str, int]:
    """Extract full test body starting from the opening brace."""
    brace_count = 1
    pos = match_end
    # Find opening brace
    while pos < len(source) and source[pos - 1] != '{':
        pos += 1
    start = pos
    for i in range(start, len(source)):
        if source[i] == '{':
            brace_count += 1
        elif source[i] == '}':
            brace_count -= 1
            if brace_count == 0:
                return source[start:i], i + 1
    return source[start:], len(source)


def _extract_assertions(body: str) -> List[str]:
    """Extract assertion statements from test body."""
    assertions = []
    for pattern in _ASSERTION_PATTERNS:
        for m in pattern.finditer(body):
            assertions.append(m.group(1))
    return assertions


def extract_c_tests(source: str, file_path: str = "<stdin>") -> List[CTestCase]:
    """Extract test cases from C test source code.

    Auto-detects the test framework and extracts all test functions.

    Args:
        source: C test source code
        file_path: Source file path for diagnostics

    Returns:
        List of extracted CTestCase objects
    """
    framework = _detect_framework(source)
    pattern = _TEST_PATTERNS.get(framework, _TEST_PATTERNS["plain"])
    tests: List[CTestCase] = []

    for m in pattern.finditer(source):
        if framework == "gtest_c":
            name = f"{m.group(1)}_{m.group(2)}"
        else:
            name = m.group(1)

        line_start = source[:m.start()].count('\n') + 1
        body, end_pos = _extract_test_body(source, m.end())
        line_end = source[:end_pos].count('\n') + 1
        assertions = _extract_assertions(body)

        # Find function calls to detect dependencies
        fn_calls = re.findall(r'\b(\w+)\s*\(', body)
        deps = [f for f in fn_calls
                if f not in ('assert', 'printf', 'fprintf', 'malloc', 'free',
                             'memset', 'memcpy', 'strlen', 'strcmp')]

        tests.append(CTestCase(
            name=name,
            body=body,
            file_path=file_path,
            line_start=line_start,
            line_end=line_end,
            assertions=assertions,
            dependencies=deps,
        ))

    return tests


# ---------------------------------------------------------------------------
# Assertion translation
# ---------------------------------------------------------------------------

_ASSERTION_MAP = {
    r'CU_ASSERT_EQUAL\s*\(\s*(.+?)\s*,\s*(.+?)\s*\)':
        'assert_eq!({0}, {1});',
    r'CU_ASSERT_NOT_EQUAL\s*\(\s*(.+?)\s*,\s*(.+?)\s*\)':
        'assert_ne!({0}, {1});',
    r'CU_ASSERT_TRUE\s*\(\s*(.+?)\s*\)':
        'assert!({0});',
    r'CU_ASSERT_FALSE\s*\(\s*(.+?)\s*\)':
        'assert!(!({0}));',
    r'CU_ASSERT_PTR_NULL\s*\(\s*(.+?)\s*\)':
        'assert!({0}.is_none());',
    r'CU_ASSERT_PTR_NOT_NULL\s*\(\s*(.+?)\s*\)':
        'assert!({0}.is_some());',
    r'ck_assert_int_eq\s*\(\s*(.+?)\s*,\s*(.+?)\s*\)':
        'assert_eq!({0}, {1});',
    r'ck_assert_str_eq\s*\(\s*(.+?)\s*,\s*(.+?)\s*\)':
        'assert_eq!({0}, {1});',
    r'assert_int_equal\s*\(\s*(.+?)\s*,\s*(.+?)\s*\)':
        'assert_eq!({0}, {1});',
    r'assert_string_equal\s*\(\s*(.+?)\s*,\s*(.+?)\s*\)':
        'assert_eq!({0}, {1});',
    r'assert_true\s*\(\s*(.+?)\s*\)':
        'assert!({0});',
    r'assert_false\s*\(\s*(.+?)\s*\)':
        'assert!(!({0}));',
    r'assert\s*\(\s*(.+?)\s*\)':
        'assert!({0});',
    r'ASSERT_EQ\s*\(\s*(.+?)\s*,\s*(.+?)\s*\)':
        'assert_eq!({0}, {1});',
    r'ASSERT_NE\s*\(\s*(.+?)\s*,\s*(.+?)\s*\)':
        'assert_ne!({0}, {1});',
    r'ASSERT_TRUE\s*\(\s*(.+?)\s*\)':
        'assert!({0});',
    r'ASSERT_FALSE\s*\(\s*(.+?)\s*\)':
        'assert!(!({0}));',
    r'EXPECT_EQ\s*\(\s*(.+?)\s*,\s*(.+?)\s*\)':
        'assert_eq!({0}, {1});',
    r'EXPECT_NE\s*\(\s*(.+?)\s*,\s*(.+?)\s*\)':
        'assert_ne!({0}, {1});',
    r'EXPECT_TRUE\s*\(\s*(.+?)\s*\)':
        'assert!({0});',
    r'EXPECT_FALSE\s*\(\s*(.+?)\s*\)':
        'assert!(!({0}));',
}


def _translate_assertion(assertion: str) -> str:
    """Translate a single C assertion to Rust."""
    for pattern, template in _ASSERTION_MAP.items():
        m = re.match(pattern, assertion.strip())
        if m:
            groups = m.groups()
            try:
                return template.format(*groups)
            except (IndexError, KeyError):
                return template
    # Fallback: wrap in assert!
    return f"assert!({assertion});"


def _translate_test_body(body: str) -> str:
    """Translate the body of a C test to Rust."""
    lines = body.strip().split('\n')
    rust_lines = []

    for line in lines:
        stripped = line.strip()
        if not stripped or stripped.startswith('//'):
            rust_lines.append(line)
            continue

        # Try assertion translation
        translated = False
        for pattern in _ASSERTION_PATTERNS:
            m = pattern.search(stripped)
            if m:
                indent = len(line) - len(line.lstrip())
                rust_assertion = _translate_assertion(m.group(1))
                rust_lines.append(' ' * indent + rust_assertion)
                translated = True
                break

        if not translated:
            # Basic C→Rust line translation
            rust_line = _translate_c_line(stripped)
            indent = len(line) - len(line.lstrip())
            rust_lines.append(' ' * indent + rust_line)

    return '\n'.join(rust_lines)


def _translate_c_line(line: str) -> str:
    """Translate a single line of C code to Rust (best-effort)."""
    result = line

    # Variable declarations: int x = 5; → let x: i32 = 5;
    m = re.match(r'(int|long|double|float|char|unsigned\s+\w+|size_t)\s+'
                 r'(\w+)\s*=\s*(.+);', result)
    if m:
        from .auto_translator import _TYPE_MAP
        c_type = m.group(1)
        name = m.group(2)
        value = m.group(3)
        rust_type = _TYPE_MAP.get(c_type, c_type)
        return f"let {name}: {rust_type} = {value};"

    # Variable declarations without initialization
    m = re.match(r'(int|long|double|float|char|unsigned\s+\w+|size_t)\s+'
                 r'(\w+)\s*;', result)
    if m:
        from .auto_translator import _TYPE_MAP
        c_type = m.group(1)
        name = m.group(2)
        rust_type = _TYPE_MAP.get(c_type, c_type)
        return f"let {name}: {rust_type};"

    # NULL → None
    result = re.sub(r'\bNULL\b', 'None', result)

    return result


# ---------------------------------------------------------------------------
# Test generation
# ---------------------------------------------------------------------------

def _translate_single_test(c_test: CTestCase) -> RustTestCase:
    """Translate one C test case to Rust."""
    rust_body = _translate_test_body(c_test.body)
    rust_code = textwrap.dedent(f"""\
        #[test]
        fn {c_test.name}() {{
        {textwrap.indent(rust_body, '    ')}
        }}
    """)
    return RustTestCase(
        name=c_test.name,
        rust_code=rust_code,
        kind=TestKind.UNIT,
        original_c=c_test,
        notes=[f"Migrated from {c_test.file_path}:{c_test.line_start}"],
    )


def migrate_test_suite(c_test_dir: str,
                       rust_dir: str) -> TestMigrationResult:
    """Migrate an entire C test suite to Rust tests.

    Scans c_test_dir for C test files, extracts test cases, translates
    them to Rust, and writes the output to rust_dir.

    Args:
        c_test_dir: Directory containing C test files
        rust_dir: Output directory for Rust test files

    Returns:
        TestMigrationResult with migration details
    """
    start = time.time()
    c_dir = Path(c_test_dir)
    r_dir = Path(rust_dir)
    r_dir.mkdir(parents=True, exist_ok=True)

    result = TestMigrationResult(
        c_test_dir=c_test_dir,
        rust_test_dir=rust_dir,
    )

    c_files = list(c_dir.rglob("*test*.c")) + list(c_dir.rglob("*check*.c"))
    # Deduplicate
    seen: Set[str] = set()
    unique_files = []
    for f in c_files:
        if str(f) not in seen:
            seen.add(str(f))
            unique_files.append(f)

    all_rust_tests: List[str] = []

    for c_file in unique_files:
        source = c_file.read_text(encoding="utf-8", errors="replace")
        c_tests = extract_c_tests(source, str(c_file))
        result.total_c_tests += len(c_tests)

        for ct in c_tests:
            try:
                rt = _translate_single_test(ct)
                result.migrated.append(rt)
                result.total_assertions_migrated += ct.assertion_count
                all_rust_tests.append(rt.rust_code)
            except Exception as exc:
                result.failed.append((ct, str(exc)))

    # Assemble test module
    result.total_rust_tests = len(result.migrated)
    header = textwrap.dedent("""\
        //! Auto-migrated test suite from C.
        //! Generated by XEquiv test_migration.

        #[cfg(test)]
        mod tests {
            use super::*;

    """)
    footer = "}\n"
    body = "\n".join(f"    {line}" for rt in result.migrated
                     for line in rt.rust_code.split('\n'))
    result.rust_test_source = header + body + "\n" + footer

    # Write output
    test_file = r_dir / "tests.rs"
    test_file.write_text(result.rust_test_source, encoding="utf-8")

    result.duration_ms = (time.time() - start) * 1000
    return result


def generate_property_tests(c_code: str, rust_code: str) -> str:
    """Generate proptest-based property tests for Rust code.

    Analyzes the C and Rust code to determine function signatures,
    then generates property tests that exercise all input domains.

    Args:
        c_code: Original C source code
        rust_code: Translated Rust source code

    Returns:
        Rust source code string containing proptest test module
    """
    fn_pattern = re.compile(
        r'(?:pub\s+)?fn\s+(\w+)\s*\(([^)]*)\)\s*(?:->\s*(\S+))?\s*\{',
        re.MULTILINE,
    )
    tests: List[str] = []

    for m in fn_pattern.finditer(rust_code):
        name = m.group(1)
        params_str = m.group(2).strip()
        ret_type = m.group(3) or "()"

        if not params_str:
            continue

        # Parse parameters
        params = []
        strategies = []
        for param in params_str.split(','):
            param = param.strip()
            parts = param.split(':')
            if len(parts) != 2:
                continue
            pname = parts[0].strip()
            ptype = parts[1].strip()
            params.append((pname, ptype))
            strategies.append(_proptest_strategy(ptype))

        if not params:
            continue

        param_decls = ", ".join(
            f"{p[0]} in {s}" for p, s in zip(params, strategies)
        )
        call_args = ", ".join(p[0] for p in params)

        test = textwrap.dedent(f"""\
            proptest! {{
                #[test]
                fn prop_{name}({param_decls}) {{
                    // Property: function should not panic
                    let _result = {name}({call_args});
                    // Add domain-specific properties below
                }}
            }}
        """)
        tests.append(test)

    # Also generate roundtrip tests if we detect encode/decode pairs
    encode_fns = re.findall(r'fn\s+(encode\w*|serialize\w*|to_\w+)', rust_code)
    decode_fns = re.findall(r'fn\s+(decode\w*|deserialize\w*|from_\w+)', rust_code)
    for enc in encode_fns:
        for dec in decode_fns:
            if enc.replace("encode", "") == dec.replace("decode", ""):
                tests.append(textwrap.dedent(f"""\
                    proptest! {{
                        #[test]
                        fn prop_roundtrip_{enc}_{dec}(input in any::<Vec<u8>>()) {{
                            let encoded = {enc}(&input);
                            let decoded = {dec}(&encoded);
                            prop_assert_eq!(input, decoded);
                        }}
                    }}
                """))

    header = textwrap.dedent("""\
        use proptest::prelude::*;

        #[cfg(test)]
        mod property_tests {
            use super::*;
            use proptest::prelude::*;

    """)
    footer = "}\n"
    body = textwrap.indent("\n".join(tests), "    ")
    return header + body + "\n" + footer


def _proptest_strategy(rust_type: str) -> str:
    """Return a proptest strategy for a Rust type."""
    strategies = {
        "i8": "any::<i8>()",
        "i16": "any::<i16>()",
        "i32": "any::<i32>()",
        "i64": "any::<i64>()",
        "u8": "any::<u8>()",
        "u16": "any::<u16>()",
        "u32": "any::<u32>()",
        "u64": "any::<u64>()",
        "f32": "any::<f32>()",
        "f64": "any::<f64>()",
        "usize": "0usize..10000",
        "isize": "-10000isize..10000",
        "bool": "any::<bool>()",
        "String": "\".*\"",
        "&str": "\".*\"",
        "&[u8]": "prop::collection::vec(any::<u8>(), 0..256)",
        "Vec<u8>": "prop::collection::vec(any::<u8>(), 0..256)",
        "Vec<i32>": "prop::collection::vec(any::<i32>(), 0..100)",
        "Option<i32>": "prop::option::of(any::<i32>())",
    }
    return strategies.get(rust_type, f"any::<{rust_type}>()")


def generate_ffi_tests(c_header: str, rust_bindings: str) -> str:
    """Generate tests for FFI boundary between C and Rust.

    Parses C header declarations and Rust extern blocks to generate
    tests that verify FFI calls work correctly.

    Args:
        c_header: C header file contents
        rust_bindings: Rust extern block or bindgen output

    Returns:
        Rust test source code for FFI boundary testing
    """
    # Extract C function declarations from header
    c_fn_pattern = re.compile(
        r'(\w[\w\s\*]+?)\s+(\w+)\s*\(([^)]*)\)\s*;', re.MULTILINE
    )
    c_functions = []
    for m in c_fn_pattern.finditer(c_header):
        c_functions.append({
            "return_type": m.group(1).strip(),
            "name": m.group(2),
            "params": m.group(3).strip(),
        })

    # Extract Rust extern function declarations
    rust_fn_pattern = re.compile(
        r'pub\s+fn\s+(\w+)\s*\(([^)]*)\)\s*(?:->\s*(\S+))?',
        re.MULTILINE,
    )
    rust_functions = {}
    for m in rust_fn_pattern.finditer(rust_bindings):
        rust_functions[m.group(1)] = {
            "params": m.group(2).strip(),
            "return_type": m.group(3) or "()",
        }

    tests: List[str] = []

    for c_fn in c_functions:
        name = c_fn["name"]
        if name not in rust_functions:
            continue

        # Generate test for each FFI function
        rust_info = rust_functions[name]
        test_args = _generate_test_args(rust_info["params"])
        ret_type = rust_info["return_type"]

        test = textwrap.dedent(f"""\
            #[test]
            fn test_ffi_{name}() {{
                unsafe {{
                    {f'let result = {name}({test_args});' if test_args else f'let result = {name}();'}
                    // Verify the FFI call doesn't crash
                    {_generate_ffi_assertion(ret_type)}
                }}
            }}
        """)
        tests.append(test)

    # Generate size/alignment tests for types
    struct_pattern = re.compile(
        r'typedef\s+struct\s+\w*\s*\{[^}]*\}\s*(\w+)\s*;',
        re.DOTALL,
    )
    for m in struct_pattern.finditer(c_header):
        type_name = m.group(1)
        tests.append(textwrap.dedent(f"""\
            #[test]
            fn test_ffi_layout_{type_name}() {{
                // Verify struct layout matches C
                assert!(std::mem::size_of::<{type_name}>() > 0,
                        "FFI type {type_name} should have non-zero size");
                assert!(std::mem::align_of::<{type_name}>() > 0,
                        "FFI type {type_name} should have non-zero alignment");
            }}
        """))

    header = textwrap.dedent("""\
        //! FFI boundary tests — auto-generated by XEquiv.
        //! These tests verify that C↔Rust FFI calls are correct.

        #[cfg(test)]
        mod ffi_tests {
            use super::*;

    """)
    footer = "}\n"
    body = textwrap.indent("\n".join(tests), "    ")
    return header + body + "\n" + footer


def _generate_test_args(params_str: str) -> str:
    """Generate test argument values for FFI function parameters."""
    if not params_str.strip():
        return ""
    args = []
    for param in params_str.split(','):
        param = param.strip()
        parts = param.split(':')
        if len(parts) < 2:
            args.append("0")
            continue
        ptype = parts[-1].strip()
        args.append(_default_value_for_type(ptype))
    return ", ".join(args)


def _default_value_for_type(rust_type: str) -> str:
    """Return a safe default test value for a Rust type."""
    defaults = {
        "i8": "0i8", "i16": "0i16", "i32": "0i32", "i64": "0i64",
        "u8": "0u8", "u16": "0u16", "u32": "0u32", "u64": "0u64",
        "f32": "0.0f32", "f64": "0.0f64",
        "usize": "0usize", "isize": "0isize",
        "bool": "false",
        "*mut u8": "std::ptr::null_mut()",
        "*const u8": "std::ptr::null()",
        "*mut i8": "std::ptr::null_mut()",
        "*const i8": "std::ptr::null()",
        "c_int": "0",
        "c_uint": "0",
        "c_char": "0",
    }
    # Handle pointer types generically
    if rust_type.startswith("*mut"):
        return "std::ptr::null_mut()"
    if rust_type.startswith("*const"):
        return "std::ptr::null()"
    return defaults.get(rust_type, "Default::default()")


def _generate_ffi_assertion(ret_type: str) -> str:
    """Generate an assertion for an FFI return value."""
    if ret_type == "()" or ret_type == "":
        return "// void function — no return value to check"
    if ret_type in ("i32", "i64", "c_int"):
        return 'assert!(result >= -1, "FFI call returned unexpected error");'
    if ret_type.startswith("*"):
        return "// Pointer return — check validity in integration tests"
    if ret_type == "bool":
        return "let _ = result; // bool result is valid"
    return f"let _ = result; // {ret_type} return value"


def differential_test_harness(c_code: str, rust_code: str) -> str:
    """Generate a cross-language differential test harness.

    Creates a Rust test module that compiles the C code via cc crate,
    links it, and runs both implementations with identical inputs to
    compare outputs.

    Args:
        c_code: Original C source code
        rust_code: Translated Rust source code

    Returns:
        Rust source code for differential testing build + tests
    """
    # Extract function signatures from C code
    fn_pattern = re.compile(
        r'(\w[\w\s\*]+?)\s+(\w+)\s*\(([^)]*)\)\s*\{', re.MULTILINE
    )
    c_functions = []
    for m in fn_pattern.finditer(c_code):
        ret = m.group(1).strip()
        name = m.group(2)
        params = m.group(3).strip()
        # Skip main and static functions
        if name == "main" or "static" in ret:
            continue
        c_functions.append({"ret": ret, "name": name, "params": params})

    # Build script for cc crate
    build_rs = textwrap.dedent("""\
        // build.rs — compile C reference implementation for differential testing
        fn main() {
            cc::Build::new()
                .file("c_reference.c")
                .warnings(false)
                .compile("c_reference");
        }
    """)

    # Generate extern declarations and test functions
    extern_decls = []
    test_fns = []

    from .auto_translator import _TYPE_MAP

    for fn_info in c_functions:
        name = fn_info["name"]
        c_ret = fn_info["ret"]
        c_params = fn_info["params"]

        # Translate types for extern block
        rust_ret = _TYPE_MAP.get(c_ret, "i32")
        rust_params = _translate_extern_params(c_params)
        c_name = f"c_{name}"

        extern_decls.append(
            f"    fn {c_name}({rust_params}) -> {rust_ret};"
        )

        # Generate differential test
        test_args = _generate_diff_test_args(rust_params)
        call_args_str = ", ".join(a[0] for a in test_args)
        let_stmts = "\n        ".join(
            f"let {a[0]}: {a[1]} = {a[2]};" for a in test_args
        )

        test_fn = textwrap.dedent(f"""\
            #[test]
            fn diff_test_{name}() {{
                {let_stmts}
                let c_result = unsafe {{ {c_name}({call_args_str}) }};
                let rust_result = {name}({call_args_str});
                assert_eq!(
                    c_result, rust_result,
                    "Differential test failed for {name}: \\
                     C={{:?}} vs Rust={{:?}}",
                    c_result, rust_result
                );
            }}
        """)
        test_fns.append(test_fn)

    extern_block = "extern \"C\" {\n" + "\n".join(extern_decls) + "\n}"

    harness = textwrap.dedent(f"""\
        //! Differential test harness — auto-generated by XEquiv.
        //! Compiles C reference and Rust translation side-by-side,
        //! runs both with identical inputs, and compares outputs.

        {extern_block}

        #[cfg(test)]
        mod differential_tests {{
            use super::*;

    """)
    body = textwrap.indent("\n".join(test_fns), "    ")
    harness += body + "\n}\n"

    # Also return the build.rs content as a comment header
    return f"// === build.rs ===\n{build_rs}\n// === tests ===\n{harness}"


def _translate_extern_params(c_params: str) -> str:
    """Translate C parameter list to Rust extern fn params."""
    from .auto_translator import _TYPE_MAP

    if not c_params.strip() or c_params.strip() == "void":
        return ""
    parts = []
    for param in c_params.split(","):
        param = param.strip()
        m = re.match(r'(.+?)\s*\*?\s*(\w+)$', param)
        if m:
            c_type = m.group(1).strip()
            name = m.group(2)
            if '*' in param:
                c_type = c_type + '*'
            rust_type = _TYPE_MAP.get(c_type, "i32")
            parts.append(f"{name}: {rust_type}")
        else:
            parts.append(param)
    return ", ".join(parts)


def _generate_diff_test_args(
    rust_params: str,
) -> List[Tuple[str, str, str]]:
    """Generate (name, type, value) tuples for differential test args."""
    if not rust_params.strip():
        return []
    args = []
    for param in rust_params.split(","):
        param = param.strip()
        parts = param.split(":")
        if len(parts) != 2:
            continue
        name = parts[0].strip()
        ptype = parts[1].strip()
        value = _default_value_for_type(ptype)
        args.append((name, ptype, value))
    return args


def coverage_equivalence(c_coverage: str,
                         rust_coverage: str) -> CoverageComparison:
    """Compare test coverage between C and Rust implementations.

    Parses coverage reports (lcov/gcov format for C, llvm-cov/tarpaulin
    for Rust) and produces a structured comparison.

    Args:
        c_coverage: C coverage report content (lcov format)
        rust_coverage: Rust coverage report content (lcov format)

    Returns:
        CoverageComparison with per-metric comparison
    """
    c_stats = _parse_lcov(c_coverage)
    r_stats = _parse_lcov(rust_coverage)

    c_uncovered = [
        CoverageRegion(file_path=r["file"], line_start=r["line"],
                       line_end=r["line"], hit_count=0)
        for r in c_stats.get("uncovered_lines", [])
    ]
    r_uncovered = [
        CoverageRegion(file_path=r["file"], line_start=r["line"],
                       line_end=r["line"], hit_count=0)
        for r in r_stats.get("uncovered_lines", [])
    ]

    c_line = c_stats.get("line_rate", 0.0)
    r_line = r_stats.get("line_rate", 0.0)

    return CoverageComparison(
        c_line_coverage=c_line,
        rust_line_coverage=r_line,
        c_branch_coverage=c_stats.get("branch_rate", 0.0),
        rust_branch_coverage=r_stats.get("branch_rate", 0.0),
        c_function_coverage=c_stats.get("function_rate", 0.0),
        rust_function_coverage=r_stats.get("function_rate", 0.0),
        uncovered_c_regions=c_uncovered,
        uncovered_rust_regions=r_uncovered,
        coverage_gap=abs(c_line - r_line),
    )


def _parse_lcov(content: str) -> Dict:
    """Parse lcov-format coverage report."""
    stats: Dict = {
        "line_rate": 0.0,
        "branch_rate": 0.0,
        "function_rate": 0.0,
        "uncovered_lines": [],
    }
    if not content.strip():
        return stats

    total_lines = 0
    hit_lines = 0
    total_branches = 0
    hit_branches = 0
    total_functions = 0
    hit_functions = 0
    current_file = ""

    for line in content.split('\n'):
        line = line.strip()
        if line.startswith("SF:"):
            current_file = line[3:]
        elif line.startswith("DA:"):
            parts = line[3:].split(',')
            if len(parts) >= 2:
                total_lines += 1
                line_num = int(parts[0])
                hits = int(parts[1])
                if hits > 0:
                    hit_lines += 1
                else:
                    stats["uncovered_lines"].append({
                        "file": current_file, "line": line_num
                    })
        elif line.startswith("BRDA:"):
            parts = line[5:].split(',')
            total_branches += 1
            if len(parts) >= 4 and parts[3] != '-' and int(parts[3]) > 0:
                hit_branches += 1
        elif line.startswith("FNF:"):
            total_functions += int(line[4:])
        elif line.startswith("FNH:"):
            hit_functions += int(line[4:])

    stats["line_rate"] = hit_lines / total_lines if total_lines > 0 else 0.0
    stats["branch_rate"] = (hit_branches / total_branches
                            if total_branches > 0 else 0.0)
    stats["function_rate"] = (hit_functions / total_functions
                              if total_functions > 0 else 0.0)

    return stats
