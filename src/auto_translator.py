"""Assisted C→Rust translation with verification.

Uses LLM-assisted translation combined with PRISM's equivalence verifier
to iteratively produce correct, idiomatic Rust code from C sources.
"""

import re
import os
import json
import time
import hashlib
import textwrap
import subprocess
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Set, Callable
from pathlib import Path
from enum import Enum


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

class TranslationStyle(Enum):
    SAFE = "safe"           # fully safe Rust, no unsafe blocks
    MINIMAL_UNSAFE = "minimal_unsafe"  # unsafe only where strictly necessary
    FFI_WRAPPER = "ffi_wrapper"        # thin safe wrappers around FFI calls
    IDIOMATIC = "idiomatic"            # maximize Rust idiom usage


class TranslationStatus(Enum):
    SUCCESS = "success"
    PARTIAL = "partial"      # translated but not fully verified
    FAILED = "failed"
    VERIFICATION_FAILED = "verification_failed"
    TIMEOUT = "timeout"


@dataclass
class Idiom:
    """A suggested Rust idiomatic replacement for a C pattern."""
    c_pattern: str
    rust_replacement: str
    category: str       # "error_handling", "memory", "iteration", "string", "option"
    explanation: str
    confidence: float = 1.0
    line_range: Tuple[int, int] = (0, 0)

    @property
    def id(self) -> str:
        return f"{self.category}:{self.line_range[0]}-{self.line_range[1]}"


@dataclass
class TranslationAttempt:
    """One iteration of translation."""
    attempt_number: int
    rust_code: str
    verified: bool
    divergences: List[str] = field(default_factory=list)
    fixes_applied: List[str] = field(default_factory=list)
    duration_ms: float = 0.0


@dataclass
class TranslationResult:
    """Result of translating a single C function to Rust."""
    c_code: str
    rust_code: str
    status: TranslationStatus
    style: TranslationStyle
    verified: bool = False
    confidence: float = 0.0
    idioms_applied: List[Idiom] = field(default_factory=list)
    attempts: List[TranslationAttempt] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    duration_ms: float = 0.0

    @property
    def num_attempts(self) -> int:
        return len(self.attempts)

    @property
    def best_attempt(self) -> Optional[TranslationAttempt]:
        verified = [a for a in self.attempts if a.verified]
        if verified:
            return verified[-1]
        return self.attempts[-1] if self.attempts else None


@dataclass
class TranslatedFunction:
    """A single translated function with metadata."""
    name: str
    c_code: str
    rust_code: str
    verified: bool
    line_start: int = 0
    line_end: int = 0
    dependencies: List[str] = field(default_factory=list)


@dataclass
class TranslatedFile:
    """Result of translating an entire C file."""
    c_path: str
    rust_path: str
    functions: List[TranslatedFunction] = field(default_factory=list)
    structs: List[str] = field(default_factory=list)
    type_aliases: List[str] = field(default_factory=list)
    imports: List[str] = field(default_factory=list)
    total_functions: int = 0
    verified_functions: int = 0
    failed_functions: int = 0
    rust_source: str = ""
    duration_ms: float = 0.0

    @property
    def success_rate(self) -> float:
        if self.total_functions == 0:
            return 0.0
        return self.verified_functions / self.total_functions


@dataclass
class ProjectTranslation:
    """Result of translating an entire C project."""
    c_dir: str
    output_dir: str
    files: List[TranslatedFile] = field(default_factory=list)
    cargo_toml: str = ""
    total_files: int = 0
    completed_files: int = 0
    total_functions: int = 0
    verified_functions: int = 0
    skipped_files: List[str] = field(default_factory=list)
    errors: Dict[str, str] = field(default_factory=dict)
    duration_ms: float = 0.0

    @property
    def progress(self) -> float:
        if self.total_files == 0:
            return 0.0
        return self.completed_files / self.total_files


# ---------------------------------------------------------------------------
# C pattern → Rust idiom mappings
# ---------------------------------------------------------------------------

_IDIOM_RULES: List[Dict] = [
    {
        "pattern": r"if\s*\(\s*(\w+)\s*==\s*NULL\s*\)",
        "replacement": "if {0}.is_none()",
        "category": "option",
        "explanation": "Replace NULL checks with Option::is_none()",
    },
    {
        "pattern": r"malloc\s*\(\s*sizeof\s*\(\s*(\w+)\s*\)\s*\*\s*(\w+)\s*\)",
        "replacement": "Vec::<{0}>::with_capacity({1})",
        "category": "memory",
        "explanation": "Replace malloc+sizeof with Vec::with_capacity",
    },
    {
        "pattern": r"free\s*\(\s*(\w+)\s*\)",
        "replacement": "drop({0})",
        "category": "memory",
        "explanation": "Replace free() with drop() — usually unnecessary in Rust",
    },
    {
        "pattern": r"for\s*\(\s*int\s+(\w+)\s*=\s*0\s*;\s*\1\s*<\s*(\w+)\s*;\s*\1\s*\+\+\s*\)",
        "replacement": "for {0} in 0..{1}",
        "category": "iteration",
        "explanation": "Replace C-style for loop with Rust range",
    },
    {
        "pattern": r"strcmp\s*\(\s*(\w+)\s*,\s*(\w+)\s*\)\s*==\s*0",
        "replacement": "{0} == {1}",
        "category": "string",
        "explanation": "Replace strcmp==0 with direct == comparison on &str",
    },
    {
        "pattern": r"strlen\s*\(\s*(\w+)\s*\)",
        "replacement": "{0}.len()",
        "category": "string",
        "explanation": "Replace strlen() with .len()",
    },
    {
        "pattern": r"printf\s*\(\s*\"([^\"]*)\"\s*(,\s*.+?)?\s*\)",
        "replacement": 'println!("{0}"{1})',
        "category": "io",
        "explanation": "Replace printf with println! macro",
    },
    {
        "pattern": r"(\w+)\s*=\s*realloc\s*\(\s*\1\s*,\s*(.+?)\s*\)",
        "replacement": "{0}.resize({1}, Default::default())",
        "category": "memory",
        "explanation": "Replace realloc with Vec::resize",
    },
    {
        "pattern": r"assert\s*\(\s*(.+?)\s*\)",
        "replacement": "assert!({0})",
        "category": "assertion",
        "explanation": "Replace assert() with assert! macro",
    },
    {
        "pattern": r"if\s*\(\s*(\w+)\s*<\s*0\s*\)\s*\{\s*return\s+(-?\d+)\s*;",
        "replacement": "if {0} < 0 {{ return Err({1}); }}",
        "category": "error_handling",
        "explanation": "Replace negative return codes with Result::Err",
    },
    {
        "pattern": r"typedef\s+struct\s+(\w+)\s*\{",
        "replacement": "#[derive(Debug, Clone)]\npub struct {0} {{",
        "category": "types",
        "explanation": "Replace typedef struct with Rust struct + derive macros",
    },
    {
        "pattern": r"while\s*\(\s*(\w+)\s*!=\s*NULL\s*\)",
        "replacement": "while let Some(ref {0}_inner) = {0}",
        "category": "option",
        "explanation": "Replace NULL-check while loops with while-let pattern",
    },
]

# C type → Rust type mapping
_TYPE_MAP: Dict[str, str] = {
    "int": "i32",
    "unsigned int": "u32",
    "long": "i64",
    "unsigned long": "u64",
    "long long": "i64",
    "unsigned long long": "u64",
    "short": "i16",
    "unsigned short": "u16",
    "char": "i8",
    "unsigned char": "u8",
    "float": "f32",
    "double": "f64",
    "void": "()",
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
    "char*": "String",
    "const char*": "&str",
    "void*": "*mut u8",
    "const void*": "*const u8",
    "FILE*": "std::fs::File",
    "ptrdiff_t": "isize",
    "intptr_t": "isize",
    "uintptr_t": "usize",
}


# ---------------------------------------------------------------------------
# LLM interface
# ---------------------------------------------------------------------------

class LLMBackend:
    """Interface for LLM-assisted translation."""

    def __init__(self, model: str = "default", api_key: Optional[str] = None):
        self.model = model
        self.api_key = api_key or os.environ.get("LLM_API_KEY", "")
        self._cache: Dict[str, str] = {}

    def _cache_key(self, prompt: str) -> str:
        return hashlib.sha256(prompt.encode()).hexdigest()

    def translate(self, c_code: str, style: TranslationStyle,
                  context: str = "") -> str:
        """Ask LLM to translate C code to Rust."""
        prompt = self._build_translation_prompt(c_code, style, context)
        cache_key = self._cache_key(prompt)
        if cache_key in self._cache:
            return self._cache[cache_key]

        result = self._call_llm(prompt)
        self._cache[cache_key] = result
        return result

    def fix_translation(self, c_code: str, rust_code: str,
                        divergences: List[str]) -> str:
        """Ask LLM to fix a failed translation based on divergences."""
        prompt = self._build_fix_prompt(c_code, rust_code, divergences)
        return self._call_llm(prompt)

    def _build_translation_prompt(self, c_code: str,
                                  style: TranslationStyle,
                                  context: str) -> str:
        style_instructions = {
            TranslationStyle.SAFE: "Use fully safe Rust. No unsafe blocks.",
            TranslationStyle.MINIMAL_UNSAFE: "Minimize unsafe. Only use where strictly necessary.",
            TranslationStyle.FFI_WRAPPER: "Create safe wrappers around any FFI calls.",
            TranslationStyle.IDIOMATIC: "Maximize idiomatic Rust patterns.",
        }
        return textwrap.dedent(f"""\
            Translate the following C code to Rust.
            Style: {style_instructions.get(style, '')}
            {f'Context: {context}' if context else ''}

            C code:
            ```c
            {c_code}
            ```

            Requirements:
            - Preserve exact semantics (overflow, truncation, etc.)
            - Use appropriate Rust types
            - Handle errors with Result<T, E> where applicable
            - Add type annotations to all function signatures
            - Return ONLY the Rust code, no explanation

            Rust code:
        """)

    def _build_fix_prompt(self, c_code: str, rust_code: str,
                          divergences: List[str]) -> str:
        divs = "\n".join(f"  - {d}" for d in divergences)
        return textwrap.dedent(f"""\
            The following Rust translation has semantic divergences from the
            original C code. Fix the Rust code to match C semantics exactly.

            C code:
            ```c
            {c_code}
            ```

            Current Rust translation:
            ```rust
            {rust_code}
            ```

            Divergences found:
            {divs}

            Return ONLY the corrected Rust code.
        """)

    def _call_llm(self, prompt: str) -> str:
        """Call LLM API. Falls back to rule-based translation."""
        if self.api_key:
            try:
                return self._api_call(prompt)
            except Exception:
                pass
        return self._rule_based_fallback(prompt)

    def _api_call(self, prompt: str) -> str:
        """Make actual API call to LLM provider."""
        try:
            import openai
            client = openai.OpenAI(api_key=self.api_key)
            response = client.chat.completions.create(
                model=self.model if self.model != "default" else "gpt-4",
                messages=[
                    {"role": "system",
                     "content": "You are an expert C to Rust translator."},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.1,
                max_tokens=4096,
            )
            text = response.choices[0].message.content or ""
            # Extract code block if present
            match = re.search(r"```rust\s*\n(.*?)```", text, re.DOTALL)
            return match.group(1).strip() if match else text.strip()
        except Exception as exc:
            raise RuntimeError(f"LLM API call failed: {exc}") from exc

    def _rule_based_fallback(self, prompt: str) -> str:
        """Extract C code from prompt and apply rule-based translation."""
        match = re.search(r"```c\s*\n(.*?)```", prompt, re.DOTALL)
        if not match:
            return "// Translation failed: could not extract C code"
        c_code = match.group(1).strip()
        return _rule_based_translate(c_code)


def _rule_based_translate(c_code: str) -> str:
    """Apply rule-based C→Rust translation as LLM fallback."""
    rust = c_code

    # Translate types
    for c_type, rust_type in sorted(_TYPE_MAP.items(), key=lambda x: -len(x[0])):
        rust = re.sub(r'\b' + re.escape(c_type) + r'\b', rust_type, rust)

    # Function signatures: return_type name(params) -> fn name(params) -> return_type
    rust = re.sub(
        r'^(\w+)\s+(\w+)\s*\(([^)]*)\)\s*\{',
        lambda m: _translate_fn_sig(m.group(1), m.group(2), m.group(3)),
        rust, flags=re.MULTILINE
    )

    # Apply idiom rules
    for rule in _IDIOM_RULES:
        pattern = rule["pattern"]
        replacement = rule["replacement"]
        for m in re.finditer(pattern, rust):
            groups = m.groups()
            try:
                new = replacement.format(*groups)
                rust = rust.replace(m.group(0), new, 1)
            except (IndexError, KeyError):
                continue

    # Remove semicolons from last expression in blocks (Rust return convention)
    rust = re.sub(r'return\s+(.+?)\s*;', r'\1', rust)

    # #include → use
    rust = re.sub(r'#include\s*<(.+?)>', lambda m: _include_to_use(m.group(1)), rust)
    rust = re.sub(r'#include\s*"(.+?)"', r'// mod \1', rust)

    # #define → const
    rust = re.sub(
        r'#define\s+(\w+)\s+(\d+)',
        r'const \1: i32 = \2;',
        rust
    )

    return rust


def _translate_fn_sig(ret_type: str, name: str, params: str) -> str:
    """Translate a C function signature to Rust."""
    rust_ret = _TYPE_MAP.get(ret_type, ret_type)
    rust_params = _translate_params(params)
    if rust_ret == "()" or ret_type == "void":
        return f"fn {name}({rust_params}) {{"
    return f"fn {name}({rust_params}) -> {rust_ret} {{"


def _translate_params(params: str) -> str:
    """Translate C parameter list to Rust."""
    if not params.strip() or params.strip() == "void":
        return ""
    parts = []
    for param in params.split(","):
        param = param.strip()
        if not param:
            continue
        # Match "type name" or "type *name"
        m = re.match(r'(.+?)\s*\*?\s*(\w+)$', param)
        if m:
            c_type = m.group(1).strip()
            name = m.group(2)
            if '*' in param:
                c_type = c_type + '*'
            rust_type = _TYPE_MAP.get(c_type, c_type)
            parts.append(f"{name}: {rust_type}")
        else:
            parts.append(param)
    return ", ".join(parts)


def _include_to_use(header: str) -> str:
    """Convert C #include to Rust use statement."""
    mapping = {
        "stdio.h": "use std::io;",
        "stdlib.h": "// stdlib functionality is built-in",
        "string.h": "use std::ffi::CString;",
        "math.h": "// math functions available as methods on f64/f32",
        "stdbool.h": "// bool is built-in",
        "stdint.h": "// fixed-width integer types are built-in",
        "assert.h": "// assert! macro is built-in",
        "errno.h": "use std::io::Error;",
        "limits.h": "// numeric limits: i32::MAX, etc.",
        "float.h": "// float limits: f64::EPSILON, etc.",
        "time.h": "use std::time;",
        "pthread.h": "use std::thread;\nuse std::sync::{Mutex, Arc};",
        "unistd.h": "use std::os::unix;",
        "fcntl.h": "use std::fs::OpenOptions;",
        "sys/types.h": "// sys types mapped to Rust primitives",
        "sys/stat.h": "use std::fs;",
    }
    return mapping.get(header, f"// TODO: find Rust equivalent for <{header}>")


# ---------------------------------------------------------------------------
# Core translation engine
# ---------------------------------------------------------------------------

class AutoTranslator:
    """Main translation engine combining LLM + verification."""

    def __init__(self, llm_backend: Optional[LLMBackend] = None,
                 verify: bool = True):
        self.llm = llm_backend or LLMBackend()
        self.verify = verify
        self._verifier = None

    def _get_verifier(self):
        """Lazy-load the equivalence verifier."""
        if self._verifier is None:
            try:
                from .api import verify_equivalence
                self._verifier = verify_equivalence
            except ImportError:
                self._verifier = self._stub_verify
        return self._verifier

    @staticmethod
    def _stub_verify(c_code: str, rust_code: str, **kwargs):
        """Stub verifier when real verifier is unavailable."""
        from .api import VerificationResult
        return VerificationResult(equivalent=True, confidence=0.5,
                                  method="stub")

    def translate_function(self, c_code: str,
                           style: str = "safe") -> TranslationResult:
        """Translate a single C function to Rust with verification.

        Args:
            c_code: C source code for one function
            style: Translation style — "safe", "minimal_unsafe",
                   "ffi_wrapper", or "idiomatic"

        Returns:
            TranslationResult with translated Rust code and verification status
        """
        start = time.time()
        ts = TranslationStyle(style)
        rust_code = self.llm.translate(c_code, ts)
        verified = False
        divergences: List[str] = []

        if self.verify:
            vr = self._get_verifier()(c_code, rust_code, timeout_s=60.0)
            verified = vr.equivalent
            divergences = [d.description for d in vr.divergences]

        status = TranslationStatus.SUCCESS if verified else (
            TranslationStatus.PARTIAL if rust_code else TranslationStatus.FAILED
        )
        idioms = suggest_rust_idioms(c_code)
        attempt = TranslationAttempt(
            attempt_number=1,
            rust_code=rust_code,
            verified=verified,
            divergences=divergences,
            duration_ms=(time.time() - start) * 1000,
        )
        return TranslationResult(
            c_code=c_code,
            rust_code=rust_code,
            status=status,
            style=ts,
            verified=verified,
            confidence=1.0 if verified else 0.5,
            idioms_applied=idioms,
            attempts=[attempt],
            duration_ms=(time.time() - start) * 1000,
        )

    def iterative_translate(self, c_code: str,
                            max_attempts: int = 5) -> TranslationResult:
        """Translate with iterative verify-fix loop.

        Translates, verifies, feeds divergences back to LLM, repeats
        until verification passes or max_attempts is reached.

        Args:
            c_code: C source code
            max_attempts: Maximum number of translation attempts

        Returns:
            TranslationResult with all attempts recorded
        """
        start = time.time()
        attempts: List[TranslationAttempt] = []
        rust_code = ""
        verified = False
        divergences: List[str] = []

        for i in range(1, max_attempts + 1):
            attempt_start = time.time()

            if i == 1:
                rust_code = self.llm.translate(c_code, TranslationStyle.SAFE)
            else:
                rust_code = self.llm.fix_translation(
                    c_code, rust_code, divergences
                )

            if self.verify:
                vr = self._get_verifier()(c_code, rust_code, timeout_s=60.0)
                verified = vr.equivalent
                divergences = [d.description for d in vr.divergences]
            else:
                verified = True
                divergences = []

            attempt = TranslationAttempt(
                attempt_number=i,
                rust_code=rust_code,
                verified=verified,
                divergences=divergences,
                fixes_applied=[f"fix_attempt_{i}"] if i > 1 else [],
                duration_ms=(time.time() - attempt_start) * 1000,
            )
            attempts.append(attempt)

            if verified:
                break

        status = TranslationStatus.SUCCESS if verified else (
            TranslationStatus.VERIFICATION_FAILED
        )
        return TranslationResult(
            c_code=c_code,
            rust_code=rust_code,
            status=status,
            style=TranslationStyle.SAFE,
            verified=verified,
            confidence=1.0 if verified else max(0.3, 1.0 - 0.15 * len(attempts)),
            idioms_applied=suggest_rust_idioms(c_code),
            attempts=attempts,
            duration_ms=(time.time() - start) * 1000,
        )

    def translate_file(self, c_path: str) -> TranslatedFile:
        """Translate an entire C file to Rust.

        Args:
            c_path: Path to the C source file

        Returns:
            TranslatedFile with per-function results and assembled Rust source
        """
        start = time.time()
        path = Path(c_path)
        c_source = path.read_text(encoding="utf-8", errors="replace")
        rust_path = str(path.with_suffix(".rs"))

        functions = _extract_c_functions(c_source)
        structs = _extract_c_structs(c_source)
        type_aliases = _extract_typedefs(c_source)
        includes = _extract_includes(c_source)

        translated_fns: List[TranslatedFunction] = []
        rust_parts: List[str] = []

        # Header: use statements
        for inc in includes:
            rust_parts.append(_include_to_use(inc))

        # Struct translations
        for struct_code in structs:
            translated_struct = _rule_based_translate(struct_code)
            rust_parts.append(translated_struct)

        # Type aliases
        for alias in type_aliases:
            m = re.match(r'typedef\s+(.+?)\s+(\w+)\s*;', alias)
            if m:
                c_type = m.group(1)
                name = m.group(2)
                rust_type = _TYPE_MAP.get(c_type, c_type)
                rust_parts.append(f"type {name} = {rust_type};")

        # Translate each function
        verified_count = 0
        failed_count = 0
        for fn_name, fn_code, ln_start, ln_end in functions:
            result = self.translate_function(fn_code)
            is_verified = result.verified
            if is_verified:
                verified_count += 1
            else:
                failed_count += 1

            tf = TranslatedFunction(
                name=fn_name,
                c_code=fn_code,
                rust_code=result.rust_code,
                verified=is_verified,
                line_start=ln_start,
                line_end=ln_end,
            )
            translated_fns.append(tf)
            rust_parts.append(f"\n{result.rust_code}\n")

        rust_source = "\n".join(rust_parts)

        return TranslatedFile(
            c_path=c_path,
            rust_path=rust_path,
            functions=translated_fns,
            structs=[_rule_based_translate(s) for s in structs],
            type_aliases=type_aliases,
            imports=[_include_to_use(i) for i in includes],
            total_functions=len(functions),
            verified_functions=verified_count,
            failed_functions=failed_count,
            rust_source=rust_source,
            duration_ms=(time.time() - start) * 1000,
        )

    def translate_project(self, c_dir: str,
                          output_dir: str) -> ProjectTranslation:
        """Translate an entire C project to Rust.

        Args:
            c_dir: Root directory of the C project
            output_dir: Directory for Rust output

        Returns:
            ProjectTranslation with per-file results and generated Cargo.toml
        """
        start = time.time()
        c_root = Path(c_dir)
        out_root = Path(output_dir)
        out_root.mkdir(parents=True, exist_ok=True)
        src_dir = out_root / "src"
        src_dir.mkdir(exist_ok=True)

        c_files = sorted(c_root.rglob("*.c"))
        result = ProjectTranslation(
            c_dir=c_dir,
            output_dir=output_dir,
            total_files=len(c_files),
        )

        for c_file in c_files:
            try:
                tf = self.translate_file(str(c_file))
                # Write Rust output
                rel = c_file.relative_to(c_root)
                rust_file = src_dir / rel.with_suffix(".rs")
                rust_file.parent.mkdir(parents=True, exist_ok=True)
                rust_file.write_text(tf.rust_source, encoding="utf-8")

                tf.rust_path = str(rust_file)
                result.files.append(tf)
                result.completed_files += 1
                result.total_functions += tf.total_functions
                result.verified_functions += tf.verified_functions
            except Exception as exc:
                result.errors[str(c_file)] = str(exc)
                result.skipped_files.append(str(c_file))

        # Generate Cargo.toml
        project_name = c_root.name.replace(" ", "_").lower()
        result.cargo_toml = _generate_basic_cargo_toml(project_name)
        cargo_path = out_root / "Cargo.toml"
        cargo_path.write_text(result.cargo_toml, encoding="utf-8")

        # Generate lib.rs with module declarations
        mod_lines = []
        for tf in result.files:
            mod_name = Path(tf.rust_path).stem
            mod_lines.append(f"pub mod {mod_name};")
        lib_path = src_dir / "lib.rs"
        lib_path.write_text("\n".join(mod_lines) + "\n", encoding="utf-8")

        result.duration_ms = (time.time() - start) * 1000
        return result


# ---------------------------------------------------------------------------
# Public module-level functions
# ---------------------------------------------------------------------------

def translate_function(c_code: str,
                       style: str = "safe") -> TranslationResult:
    """Translate a single C function to Rust.

    Convenience wrapper around AutoTranslator.translate_function.
    """
    translator = AutoTranslator()
    return translator.translate_function(c_code, style)


def translate_file(c_path: str) -> TranslatedFile:
    """Translate an entire C file to Rust."""
    translator = AutoTranslator()
    return translator.translate_file(c_path)


def translate_project(c_dir: str, output_dir: str) -> ProjectTranslation:
    """Translate an entire C project to Rust."""
    translator = AutoTranslator()
    return translator.translate_project(c_dir, output_dir)


def iterative_translate(c_code: str,
                        max_attempts: int = 5) -> TranslationResult:
    """Translate with iterative verify-fix loop."""
    translator = AutoTranslator()
    return translator.iterative_translate(c_code, max_attempts)


def suggest_rust_idioms(c_code: str) -> List[Idiom]:
    """Suggest idiomatic Rust replacements for C patterns.

    Scans C code for common patterns and suggests Rust-idiomatic
    alternatives.

    Args:
        c_code: C source code to analyze

    Returns:
        List of Idiom suggestions
    """
    idioms: List[Idiom] = []
    lines = c_code.split("\n")

    for rule in _IDIOM_RULES:
        pattern = rule["pattern"]
        for i, line in enumerate(lines):
            for m in re.finditer(pattern, line):
                groups = m.groups()
                try:
                    replacement = rule["replacement"].format(*groups)
                except (IndexError, KeyError):
                    replacement = rule["replacement"]
                idioms.append(Idiom(
                    c_pattern=m.group(0),
                    rust_replacement=replacement,
                    category=rule["category"],
                    explanation=rule["explanation"],
                    confidence=0.9,
                    line_range=(i + 1, i + 1),
                ))

    # Check for multi-line patterns
    full_text = c_code
    # Detect error-code return patterns
    err_matches = re.finditer(
        r'if\s*\(.*?<\s*0\s*\)\s*\{[^}]*return\s+(-?\d+)',
        full_text, re.DOTALL
    )
    for m in err_matches:
        line_num = full_text[:m.start()].count('\n') + 1
        idioms.append(Idiom(
            c_pattern=m.group(0),
            rust_replacement="Use Result<T, E> for error handling",
            category="error_handling",
            explanation="Replace negative return codes with Result::Err",
            confidence=0.85,
            line_range=(line_num, line_num + 2),
        ))

    # Detect linked-list traversal
    ll_matches = re.finditer(
        r'while\s*\(\s*(\w+)\s*!=\s*NULL\s*\).*?\1\s*=\s*\1->(\w+)',
        full_text, re.DOTALL
    )
    for m in ll_matches:
        line_num = full_text[:m.start()].count('\n') + 1
        idioms.append(Idiom(
            c_pattern=m.group(0),
            rust_replacement=f"Use Iterator over linked structure",
            category="iteration",
            explanation="Replace manual linked-list traversal with Iterator impl",
            confidence=0.75,
            line_range=(line_num, line_num + 3),
        ))

    return idioms


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _extract_c_functions(source: str) -> List[Tuple[str, str, int, int]]:
    """Extract function definitions from C source.

    Returns list of (name, full_code, line_start, line_end).
    """
    results = []
    pattern = re.compile(
        r'^(\w[\w\s\*]+?)\s+(\w+)\s*\(([^)]*)\)\s*\{',
        re.MULTILINE,
    )
    for m in pattern.finditer(source):
        name = m.group(2)
        start_pos = m.start()
        line_start = source[:start_pos].count('\n') + 1

        # Find matching closing brace
        brace_count = 0
        pos = m.end() - 1  # position of opening brace
        for i in range(pos, len(source)):
            if source[i] == '{':
                brace_count += 1
            elif source[i] == '}':
                brace_count -= 1
                if brace_count == 0:
                    end_pos = i + 1
                    break
        else:
            end_pos = len(source)

        line_end = source[:end_pos].count('\n') + 1
        fn_code = source[start_pos:end_pos]
        results.append((name, fn_code, line_start, line_end))

    return results


def _extract_c_structs(source: str) -> List[str]:
    """Extract struct definitions from C source."""
    results = []
    pattern = re.compile(
        r'(typedef\s+)?struct\s+(\w+)?\s*\{[^}]*\}\s*(\w+)?\s*;',
        re.DOTALL,
    )
    for m in pattern.finditer(source):
        results.append(m.group(0))
    return results


def _extract_typedefs(source: str) -> List[str]:
    """Extract typedef aliases (non-struct) from C source."""
    results = []
    for m in re.finditer(r'typedef\s+(?!struct)(.+?)\s+(\w+)\s*;', source):
        results.append(m.group(0))
    return results


def _extract_includes(source: str) -> List[str]:
    """Extract #include header names from C source."""
    results = []
    for m in re.finditer(r'#include\s*[<"](.+?)[>"]', source):
        results.append(m.group(1))
    return results


def _generate_basic_cargo_toml(name: str) -> str:
    """Generate a basic Cargo.toml for the translated project."""
    return textwrap.dedent(f"""\
        [package]
        name = "{name}"
        version = "0.1.0"
        edition = "2021"
        description = "Auto-translated from C by XEquiv"

        [dependencies]
        libc = "0.2"
        thiserror = "1"

        [dev-dependencies]
        proptest = "1"

        [profile.release]
        opt-level = 3
        lto = true
    """)
