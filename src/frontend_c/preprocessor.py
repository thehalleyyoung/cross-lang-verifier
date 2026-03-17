"""
Minimal C preprocessor for C2Rust output.

Handles #include resolution (via header map), #define expansion (simple
object-like and function-like macros), #ifdef/#ifndef/#if conditional
compilation, __attribute__ parsing, and _Static_assert handling.

This is intentionally limited to patterns commonly found in C2Rust
transpiler output rather than being a full C preprocessor.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from typing import Callable, Optional


# ---------------------------------------------------------------------------
# Macro definitions
# ---------------------------------------------------------------------------

@dataclass
class MacroParam:
    """A parameter in a function-like macro."""
    name: str
    index: int
    is_variadic: bool = False


@dataclass
class MacroDefinition:
    """A preprocessor macro definition."""
    name: str
    body: str = ""
    params: list[MacroParam] | None = None  # None = object-like
    is_function_like: bool = False
    is_builtin: bool = False
    file: str = ""
    line: int = 0

    @property
    def is_object_like(self) -> bool:
        return not self.is_function_like

    @property
    def param_count(self) -> int:
        if self.params is None:
            return 0
        return len(self.params)

    @property
    def is_variadic(self) -> bool:
        if self.params is None:
            return False
        return any(p.is_variadic for p in self.params)


# ---------------------------------------------------------------------------
# Include resolution
# ---------------------------------------------------------------------------

@dataclass
class HeaderMap:
    """Maps header paths to resolved file paths or synthetic content."""
    search_paths: list[str] = field(default_factory=list)
    overrides: dict[str, str] = field(default_factory=dict)
    system_headers: dict[str, str] = field(default_factory=dict)

    def __post_init__(self):
        self._init_system_headers()

    def _init_system_headers(self) -> None:
        """Register minimal system header stubs for C2Rust output."""
        self.system_headers.setdefault("stdint.h", _STDINT_H)
        self.system_headers.setdefault("stddef.h", _STDDEF_H)
        self.system_headers.setdefault("stdbool.h", _STDBOOL_H)
        self.system_headers.setdefault("stdlib.h", _STDLIB_H)
        self.system_headers.setdefault("string.h", _STRING_H)
        self.system_headers.setdefault("stdio.h", _STDIO_H)
        self.system_headers.setdefault("limits.h", _LIMITS_H)
        self.system_headers.setdefault("assert.h", _ASSERT_H)
        self.system_headers.setdefault("errno.h", _ERRNO_H)
        self.system_headers.setdefault("math.h", _MATH_H)
        self.system_headers.setdefault("stdarg.h", _STDARG_H)
        self.system_headers.setdefault("signal.h", _SIGNAL_H)
        self.system_headers.setdefault("setjmp.h", _SETJMP_H)
        self.system_headers.setdefault("float.h", _FLOAT_H)
        self.system_headers.setdefault("inttypes.h", _INTTYPES_H)

    def resolve(self, path: str, is_system: bool = False) -> Optional[str]:
        """Resolve a header path to its content."""
        # Check overrides first
        if path in self.overrides:
            return self.overrides[path]

        # Check system headers
        basename = os.path.basename(path)
        if basename in self.system_headers:
            return self.system_headers[basename]

        # Search include paths
        for search_dir in self.search_paths:
            full_path = os.path.join(search_dir, path)
            if os.path.isfile(full_path):
                try:
                    with open(full_path, 'r') as f:
                        return f.read()
                except (IOError, OSError):
                    pass

        return None


# ---------------------------------------------------------------------------
# Conditional compilation state
# ---------------------------------------------------------------------------

@dataclass
class _ConditionalState:
    """State for #if/#ifdef/#ifndef processing."""
    active: bool = True           # Is this branch active?
    seen_true: bool = False       # Have we seen a true branch?
    parent_active: bool = True    # Is the parent conditional active?

    @property
    def emitting(self) -> bool:
        return self.active and self.parent_active


# ---------------------------------------------------------------------------
# Preprocessor errors
# ---------------------------------------------------------------------------

class PreprocessorError(Exception):
    """Error during preprocessing."""
    def __init__(self, message: str, file: str = "", line: int = 0):
        self.file = file
        self.line = line
        super().__init__(f"{file}:{line}: {message}" if file else message)


# ---------------------------------------------------------------------------
# CPreprocessor
# ---------------------------------------------------------------------------

class CPreprocessor:
    """Minimal C preprocessor targeting C2Rust output patterns.

    Usage::

        pp = CPreprocessor()
        pp.define("MY_CONST", "42")
        result = pp.preprocess(source, filename="input.c")

    Features:
        - #include resolution with header maps
        - #define object-like and function-like macros
        - #ifdef / #ifndef / #if / #elif / #else / #endif
        - __attribute__ pass-through
        - _Static_assert pass-through
        - Predefined macros for C2Rust output
    """

    def __init__(
        self,
        header_map: HeaderMap | None = None,
        defines: dict[str, str] | None = None,
    ) -> None:
        self._header_map = header_map or HeaderMap()
        self._macros: dict[str, MacroDefinition] = {}
        self._cond_stack: list[_ConditionalState] = []
        self._include_depth = 0
        self._max_include_depth = 64
        self._errors: list[PreprocessorError] = []
        self._included_files: set[str] = set()
        self._pragma_once_files: set[str] = set()
        self._counter = 0

        # Register predefined macros
        self._register_builtins()

        # Register user-provided defines
        if defines:
            for name, value in defines.items():
                self.define(name, value)

    @property
    def errors(self) -> list[PreprocessorError]:
        return list(self._errors)

    def _register_builtins(self) -> None:
        """Register built-in predefined macros."""
        builtins = {
            "__STDC__": "1",
            "__STDC_VERSION__": "201112L",
            "__STDC_HOSTED__": "1",
            "__x86_64__": "1",
            "__LP64__": "1",
            "__SIZEOF_POINTER__": "8",
            "__SIZEOF_INT__": "4",
            "__SIZEOF_LONG__": "8",
            "__SIZEOF_LONG_LONG__": "8",
            "__SIZEOF_SHORT__": "2",
            "__SIZEOF_FLOAT__": "4",
            "__SIZEOF_DOUBLE__": "8",
            "__SIZEOF_LONG_DOUBLE__": "16",
            "__BYTE_ORDER__": "1234",
            "__ORDER_LITTLE_ENDIAN__": "1234",
            "__ORDER_BIG_ENDIAN__": "4321",
            "__CHAR_BIT__": "8",
            "__INT_MAX__": "2147483647",
            "__LONG_MAX__": "9223372036854775807L",
            "__LONG_LONG_MAX__": "9223372036854775807LL",
            "__SHRT_MAX__": "32767",
            "__SCHAR_MAX__": "127",
            "__GNUC__": "4",
            "__GNUC_MINOR__": "2",
            "__GNUC_PATCHLEVEL__": "1",
            "__clang__": "1",
            "__clang_major__": "14",
            "__has_attribute": "",
            "__has_builtin": "",
            "__has_feature": "",
            "__has_extension": "",
            "NULL": "((void*)0)",
            "__func__": '""',
            "__FUNCTION__": '""',
            "__PRETTY_FUNCTION__": '""',
        }
        for name, value in builtins.items():
            self._macros[name] = MacroDefinition(
                name=name, body=value, is_builtin=True,
            )

    def define(self, name: str, value: str = "", params: list[str] | None = None) -> None:
        """Define a macro."""
        if params is not None:
            macro_params = [
                MacroParam(name=p, index=i, is_variadic=(p == "..."))
                for i, p in enumerate(params)
            ]
            self._macros[name] = MacroDefinition(
                name=name, body=value, params=macro_params,
                is_function_like=True,
            )
        else:
            self._macros[name] = MacroDefinition(name=name, body=value)

    def undefine(self, name: str) -> None:
        """Undefine a macro."""
        self._macros.pop(name, None)

    def is_defined(self, name: str) -> bool:
        """Check if a macro is defined."""
        return name in self._macros

    # -------------------------------------------------------------------
    # Main preprocessing
    # -------------------------------------------------------------------

    def preprocess(self, source: str, filename: str = "<input>") -> str:
        """Preprocess C source code, returning the preprocessed text."""
        self._cond_stack = []
        self._include_depth = 0
        return self._process_text(source, filename)

    def _process_text(self, source: str, filename: str) -> str:
        """Process a source text, handling all directives."""
        lines = source.split('\n')
        output_lines: list[str] = []
        line_num = 0

        while line_num < len(lines):
            line = lines[line_num]
            line_num += 1

            # Handle line continuations
            while line.endswith('\\') and line_num < len(lines):
                line = line[:-1] + lines[line_num]
                line_num += 1

            stripped = line.lstrip()

            # Check for preprocessor directive
            if stripped.startswith('#'):
                self._process_directive(stripped, filename, line_num, output_lines)
                continue

            # If in inactive conditional, skip the line
            if not self._is_emitting():
                output_lines.append("")
                continue

            # Expand macros in the line
            expanded = self._expand_macros_in_line(line)
            output_lines.append(expanded)

        return '\n'.join(output_lines)

    def _is_emitting(self) -> bool:
        """Check if we should emit code (not in a false conditional)."""
        if not self._cond_stack:
            return True
        return self._cond_stack[-1].emitting

    # -------------------------------------------------------------------
    # Directive processing
    # -------------------------------------------------------------------

    def _process_directive(
        self, line: str, filename: str, line_num: int,
        output: list[str],
    ) -> None:
        """Process a preprocessor directive."""
        # Remove the #
        content = line.lstrip()[1:].lstrip()

        # Get directive name
        parts = content.split(None, 1)
        if not parts:
            output.append("")
            return

        directive = parts[0]
        rest = parts[1] if len(parts) > 1 else ""

        # Conditional directives are always processed
        if directive in ("ifdef", "ifndef", "if", "elif", "else", "endif"):
            self._process_conditional(directive, rest, filename, line_num)
            output.append("")
            return

        # Skip other directives in inactive branches
        if not self._is_emitting():
            output.append("")
            return

        if directive == "include":
            self._process_include(rest, filename, line_num, output)
        elif directive == "define":
            self._process_define(rest, filename, line_num)
            output.append("")
        elif directive == "undef":
            self._process_undef(rest)
            output.append("")
        elif directive == "pragma":
            self._process_pragma(rest, filename)
            output.append("")
        elif directive == "error":
            self._errors.append(PreprocessorError(
                f"#error {rest}", filename, line_num,
            ))
            output.append("")
        elif directive == "warning":
            output.append("")
        elif directive == "line":
            output.append("")
        else:
            output.append("")

    def _process_conditional(
        self, directive: str, condition: str,
        filename: str, line_num: int,
    ) -> None:
        """Process conditional compilation directives."""
        if directive == "ifdef":
            name = condition.strip()
            parent_active = self._is_emitting()
            active = parent_active and self.is_defined(name)
            self._cond_stack.append(_ConditionalState(
                active=active,
                seen_true=active,
                parent_active=parent_active,
            ))

        elif directive == "ifndef":
            name = condition.strip()
            parent_active = self._is_emitting()
            active = parent_active and not self.is_defined(name)
            self._cond_stack.append(_ConditionalState(
                active=active,
                seen_true=active,
                parent_active=parent_active,
            ))

        elif directive == "if":
            parent_active = self._is_emitting()
            result = self._eval_condition(condition)
            active = parent_active and result
            self._cond_stack.append(_ConditionalState(
                active=active,
                seen_true=active,
                parent_active=parent_active,
            ))

        elif directive == "elif":
            if not self._cond_stack:
                self._errors.append(PreprocessorError(
                    "#elif without matching #if", filename, line_num,
                ))
                return
            state = self._cond_stack[-1]
            if state.seen_true:
                state.active = False
            else:
                result = self._eval_condition(condition)
                active = state.parent_active and result
                state.active = active
                if active:
                    state.seen_true = True

        elif directive == "else":
            if not self._cond_stack:
                self._errors.append(PreprocessorError(
                    "#else without matching #if", filename, line_num,
                ))
                return
            state = self._cond_stack[-1]
            if state.seen_true:
                state.active = False
            else:
                state.active = state.parent_active
                state.seen_true = True

        elif directive == "endif":
            if not self._cond_stack:
                self._errors.append(PreprocessorError(
                    "#endif without matching #if", filename, line_num,
                ))
                return
            self._cond_stack.pop()

    def _eval_condition(self, condition: str) -> bool:
        """Evaluate a #if condition expression (simplified)."""
        cond = condition.strip()

        # Handle defined() operator
        cond = re.sub(
            r'defined\s*\(\s*(\w+)\s*\)',
            lambda m: '1' if self.is_defined(m.group(1)) else '0',
            cond,
        )
        cond = re.sub(
            r'defined\s+(\w+)',
            lambda m: '1' if self.is_defined(m.group(1)) else '0',
            cond,
        )

        # Expand macros in the condition
        cond = self._expand_macros_in_line(cond)

        # Replace remaining identifiers with 0 (per C spec)
        cond = re.sub(r'\b[a-zA-Z_]\w*\b', '0', cond)

        # Remove type suffixes from numbers
        cond = re.sub(r'(\d+)[uUlL]+', r'\1', cond)

        # Evaluate
        try:
            # Replace C operators with Python equivalents
            cond = cond.replace('&&', ' and ')
            cond = cond.replace('||', ' or ')
            cond = cond.replace('!', ' not ')
            # Fix double negation issues
            cond = re.sub(r'not\s*=', '!=', cond)
            result = bool(eval(cond, {"__builtins__": {}}, {}))
            return result
        except Exception:
            return False

    # -------------------------------------------------------------------
    # #include processing
    # -------------------------------------------------------------------

    def _process_include(
        self, path_spec: str, filename: str, line_num: int,
        output: list[str],
    ) -> None:
        """Process an #include directive."""
        path_spec = path_spec.strip()

        # Parse the path
        is_system = False
        if path_spec.startswith('<') and '>' in path_spec:
            path = path_spec[1:path_spec.index('>')]
            is_system = True
        elif path_spec.startswith('"') and path_spec.count('"') >= 2:
            path = path_spec[1:path_spec.rindex('"')]
        else:
            self._errors.append(PreprocessorError(
                f"malformed #include: {path_spec}", filename, line_num,
            ))
            output.append("")
            return

        # Check include depth
        if self._include_depth >= self._max_include_depth:
            self._errors.append(PreprocessorError(
                f"#include depth exceeded ({self._max_include_depth})",
                filename, line_num,
            ))
            output.append("")
            return

        # Check pragma once
        if path in self._pragma_once_files:
            output.append("")
            return

        # Resolve the header
        content = self._header_map.resolve(path, is_system)
        if content is None:
            # For C2Rust output, we often want to just skip unknown headers
            output.append(f"/* #include <{path}> - not resolved */")
            return

        # Recursively preprocess
        self._include_depth += 1
        result = self._process_text(content, path)
        self._include_depth -= 1

        output.append(result)

    # -------------------------------------------------------------------
    # #define processing
    # -------------------------------------------------------------------

    def _process_define(self, content: str, filename: str, line_num: int) -> None:
        """Process a #define directive."""
        content = content.strip()
        if not content:
            return

        # Check for function-like macro: NAME(params) body
        match = re.match(r'(\w+)\(([^)]*)\)\s*(.*)', content, re.DOTALL)
        if match:
            name = match.group(1)
            params_str = match.group(2)
            body = match.group(3).strip()

            params = []
            is_variadic = False
            for i, p in enumerate(params_str.split(',')):
                p = p.strip()
                if p == '...':
                    params.append(MacroParam(name='__VA_ARGS__', index=i, is_variadic=True))
                    is_variadic = True
                elif p:
                    params.append(MacroParam(name=p, index=i))

            self._macros[name] = MacroDefinition(
                name=name,
                body=body,
                params=params,
                is_function_like=True,
                file=filename,
                line=line_num,
            )
            return

        # Object-like macro: NAME body
        parts = content.split(None, 1)
        name = parts[0]
        body = parts[1].strip() if len(parts) > 1 else ""

        self._macros[name] = MacroDefinition(
            name=name,
            body=body,
            file=filename,
            line=line_num,
        )

    def _process_undef(self, content: str) -> None:
        """Process #undef directive."""
        name = content.strip()
        self._macros.pop(name, None)

    def _process_pragma(self, content: str, filename: str) -> None:
        """Process #pragma directive."""
        content = content.strip()
        if content == "once":
            self._pragma_once_files.add(filename)
        # Other pragmas are silently ignored

    # -------------------------------------------------------------------
    # Macro expansion
    # -------------------------------------------------------------------

    def _expand_macros_in_line(self, line: str) -> str:
        """Expand all macros in a line of text."""
        return self._expand(line, set(), 0)

    def _expand(self, text: str, expanding: set[str], depth: int) -> str:
        """Recursively expand macros, avoiding infinite recursion."""
        if depth > 64:
            return text

        result: list[str] = []
        i = 0
        n = len(text)

        while i < n:
            # Skip string literals
            if text[i] in '"\'':
                quote = text[i]
                j = i + 1
                while j < n and text[j] != quote:
                    if text[j] == '\\':
                        j += 1
                    j += 1
                j = min(j + 1, n)
                result.append(text[i:j])
                i = j
                continue

            # Try to match an identifier
            if text[i].isalpha() or text[i] == '_':
                j = i
                while j < n and (text[j].isalnum() or text[j] == '_'):
                    j += 1
                ident = text[i:j]

                if ident in self._macros and ident not in expanding:
                    macro = self._macros[ident]

                    if macro.is_function_like:
                        # Look for opening paren
                        k = j
                        while k < n and text[k] in ' \t':
                            k += 1
                        if k < n and text[k] == '(':
                            args, end = self._parse_macro_args(text, k)
                            body = self._substitute_params(macro, args)
                            new_expanding = expanding | {ident}
                            body = self._expand(body, new_expanding, depth + 1)
                            result.append(body)
                            i = end
                            continue
                    else:
                        new_expanding = expanding | {ident}
                        body = self._expand(macro.body, new_expanding, depth + 1)
                        result.append(body)
                        i = j
                        continue

                result.append(ident)
                i = j
                continue

            result.append(text[i])
            i += 1

        return ''.join(result)

    def _parse_macro_args(self, text: str, start: int) -> tuple[list[str], int]:
        """Parse macro invocation arguments from text[start] = '('."""
        args: list[str] = []
        current: list[str] = []
        depth = 0
        i = start + 1  # skip '('
        n = len(text)

        while i < n:
            ch = text[i]
            if ch == '(' or ch == '[' or ch == '{':
                depth += 1
                current.append(ch)
            elif ch == ')' or ch == ']' or ch == '}':
                if depth > 0:
                    depth -= 1
                    current.append(ch)
                elif ch == ')':
                    args.append(''.join(current).strip())
                    return args, i + 1
                else:
                    current.append(ch)
            elif ch == ',' and depth == 0:
                args.append(''.join(current).strip())
                current = []
            elif ch == '"' or ch == "'":
                quote = ch
                current.append(ch)
                i += 1
                while i < n and text[i] != quote:
                    if text[i] == '\\':
                        current.append(text[i])
                        i += 1
                    if i < n:
                        current.append(text[i])
                        i += 1
                if i < n:
                    current.append(text[i])
            else:
                current.append(ch)
            i += 1

        # Unterminated
        args.append(''.join(current).strip())
        return args, i

    def _substitute_params(self, macro: MacroDefinition, args: list[str]) -> str:
        """Substitute macro parameters with arguments."""
        if macro.params is None:
            return macro.body

        body = macro.body
        for param in macro.params:
            if param.is_variadic:
                # __VA_ARGS__ gets all remaining args
                va_args = ", ".join(args[param.index:])
                body = body.replace("__VA_ARGS__", va_args)
            elif param.index < len(args):
                # Simple textual substitution
                body = re.sub(
                    r'\b' + re.escape(param.name) + r'\b',
                    args[param.index],
                    body,
                )

        # Handle # (stringification) operator
        body = re.sub(r'#\s*(\w+)', lambda m: f'"{m.group(1)}"', body)

        # Handle ## (token paste) operator
        body = re.sub(r'\s*##\s*', '', body)

        return body

    # -------------------------------------------------------------------
    # Real-world C handling: __attribute__, __builtin_*, typeof, GCC exts
    # -------------------------------------------------------------------

    _ATTRIBUTE_RE = re.compile(
        r'__attribute__\s*\(\((?:[^()]*|\((?:[^()]*|\([^()]*\))*\))*\)\)',
    )

    _TYPEOF_RE = re.compile(
        r'__typeof__\s*\(([^)]+)\)|typeof\s*\(([^)]+)\)',
    )

    _BUILTIN_MAP: dict[str, str] = {
        # Arithmetic / overflow builtins → identity or zero stub
        "__builtin_add_overflow": "0",
        "__builtin_sub_overflow": "0",
        "__builtin_mul_overflow": "0",
        "__builtin_sadd_overflow": "0",
        "__builtin_ssub_overflow": "0",
        "__builtin_smul_overflow": "0",
        # Bit-manipulation builtins
        "__builtin_clz": "0",
        "__builtin_ctz": "0",
        "__builtin_popcount": "0",
        "__builtin_parity": "0",
        "__builtin_ffs": "0",
        "__builtin_clzl": "0",
        "__builtin_ctzl": "0",
        "__builtin_popcountl": "0",
        "__builtin_clzll": "0",
        "__builtin_ctzll": "0",
        "__builtin_popcountll": "0",
        # Byte-swap builtins
        "__builtin_bswap16": "0",
        "__builtin_bswap32": "0",
        "__builtin_bswap64": "0",
        # Memory builtins (stub as function declarations are enough)
        "__builtin_memcpy": "memcpy",
        "__builtin_memset": "memset",
        "__builtin_memmove": "memmove",
        "__builtin_memcmp": "memcmp",
        "__builtin_strlen": "strlen",
        "__builtin_strcmp": "strcmp",
        # Floating-point classification builtins
        "__builtin_inff": "(1.0f/0.0f)",
        "__builtin_inf": "(1.0/0.0)",
        "__builtin_nanf": "(0.0f/0.0f)",
        "__builtin_nan": "(0.0/0.0)",
        "__builtin_huge_val": "(1.0/0.0)",
        "__builtin_huge_valf": "(1.0f/0.0f)",
        "__builtin_isnan": "0",
        "__builtin_isinf": "0",
        "__builtin_isfinite": "1",
        # Expect / unreachable (optimization hints, not semantic)
        "__builtin_expect": "",
        "__builtin_unreachable": "((void)0)",
        "__builtin_trap": "((void)0)",
        "__builtin_assume": "((void)0)",
        # Alloca
        "__builtin_alloca": "malloc",
        # Address-of / frame
        "__builtin_return_address": "((void*)0)",
        "__builtin_frame_address": "((void*)0)",
        # Atomic (simplified stubs)
        "__sync_fetch_and_add": "0",
        "__sync_fetch_and_sub": "0",
        "__sync_val_compare_and_swap": "0",
        "__sync_synchronize": "((void)0)",
        "__atomic_load_n": "0",
        "__atomic_store_n": "((void)0)",
        "__atomic_exchange_n": "0",
        "__atomic_compare_exchange_n": "0",
    }

    _GCC_EXTENSION_KEYWORDS = frozenset({
        "__extension__", "__inline__", "__inline", "__volatile__",
        "__volatile", "__signed__", "__signed", "__unsigned__",
        "__const__", "__const", "__restrict__", "__restrict",
        "__asm__", "__asm", "_Alignof", "_Alignas",
        "__alignof__", "__alignof",
    })

    _EXTENSION_REPLACEMENTS: dict[str, str] = {
        "__extension__": "",
        "__inline__": "inline",
        "__inline": "inline",
        "__volatile__": "volatile",
        "__volatile": "volatile",
        "__signed__": "signed",
        "__signed": "signed",
        "__unsigned__": "unsigned",
        "__const__": "const",
        "__const": "const",
        "__restrict__": "restrict",
        "__restrict": "restrict",
        "__asm__": "asm",
        "__asm": "asm",
        "__alignof__": "_Alignof",
        "__alignof": "_Alignof",
    }

    def strip_attributes(self, source: str) -> str:
        """Strip all __attribute__((...)) annotations from source."""
        return self._ATTRIBUTE_RE.sub('', source)

    def replace_typeof(self, source: str) -> str:
        """Replace __typeof__(...) and typeof(...) with int as a stub type.
        Real typeof resolution needs a full type system; this is a best-effort
        stub that lets the parser proceed."""
        return self._TYPEOF_RE.sub('int', source)

    def stub_builtins(self, source: str) -> str:
        """Replace __builtin_* calls with stubs.

        For function-like builtins (e.g. __builtin_expect(x, v) → (x)),
        we handle the argument pass-through.  For simple identifier builtins,
        we replace the name with its stub value.
        """
        # Special case: __builtin_expect(expr, val) → (expr)
        source = re.sub(
            r'__builtin_expect\s*\(\s*([^,]+),\s*[^)]+\)',
            r'(\1)',
            source,
        )
        # Special case: __builtin_offsetof(type, field) → 0
        source = re.sub(
            r'__builtin_offsetof\s*\([^)]*\)',
            '0',
            source,
        )
        # Special case: __builtin_types_compatible_p(t1, t2) → 0
        source = re.sub(
            r'__builtin_types_compatible_p\s*\([^)]*\)',
            '0',
            source,
        )
        # Special case: __builtin_constant_p(x) → 0
        source = re.sub(
            r'__builtin_constant_p\s*\([^)]*\)',
            '0',
            source,
        )
        # Special case: __builtin_choose_expr(c, a, b) → (b) (conservative)
        source = re.sub(
            r'__builtin_choose_expr\s*\([^,]*,\s*[^,]*,\s*([^)]+)\)',
            r'(\1)',
            source,
        )

        # Replace remaining builtins by name (function-call forms)
        for builtin, stub in self._BUILTIN_MAP.items():
            if builtin in source:
                # If stub is a simple value and the builtin is called as function
                pattern = re.escape(builtin) + r'\s*\(([^)]*)\)'
                if stub and not stub.startswith('('):
                    source = re.sub(pattern, f'{stub}(\\1)', source)
                elif stub:
                    source = re.sub(pattern, stub, source)
                else:
                    # Empty stub for hints — pass through first argument
                    source = re.sub(pattern, r'(\1)', source)
                # Also replace standalone identifier uses
                source = re.sub(r'\b' + re.escape(builtin) + r'\b(?!\s*\()', stub or '0', source)

        return source

    def replace_gcc_extensions(self, source: str) -> str:
        """Replace GCC extension keywords with standard C equivalents."""
        for ext, replacement in self._EXTENSION_REPLACEMENTS.items():
            source = re.sub(r'\b' + re.escape(ext) + r'\b', replacement, source)
        return source

    def strip_asm_blocks(self, source: str) -> str:
        """Remove inline assembly blocks: asm(...) and __asm__(...) statements."""
        # Remove asm("...") or asm volatile("...")
        source = re.sub(
            r'\b(?:__)?asm(?:__)?(?:\s+volatile)?\s*\((?:[^()]*|\((?:[^()]*|\([^()]*\))*\))*\)\s*;',
            ';',
            source,
        )
        return source

    def preprocess_real_world(self, source: str, filename: str = "<input>") -> str:
        """Full preprocessing pipeline for real-world C code.

        Applies:
          1. Standard preprocessing (#include, #define, #ifdef, etc.)
          2. __attribute__ stripping
          3. __builtin_* replacement with stubs
          4. typeof → stub type
          5. GCC extension keyword normalization
          6. Inline assembly removal
        """
        # Phase 1: standard CPP
        result = self.preprocess(source, filename)
        # Phase 2: strip attributes
        result = self.strip_attributes(result)
        # Phase 3: stub builtins
        result = self.stub_builtins(result)
        # Phase 4: typeof
        result = self.replace_typeof(result)
        # Phase 5: GCC extensions
        result = self.replace_gcc_extensions(result)
        # Phase 6: asm
        result = self.strip_asm_blocks(result)
        return result


# ---------------------------------------------------------------------------
# Minimal system header stubs
# ---------------------------------------------------------------------------

_STDINT_H = """
typedef signed char int8_t;
typedef short int16_t;
typedef int int32_t;
typedef long long int64_t;
typedef unsigned char uint8_t;
typedef unsigned short uint16_t;
typedef unsigned int uint32_t;
typedef unsigned long long uint64_t;
typedef long intptr_t;
typedef unsigned long uintptr_t;
typedef long long intmax_t;
typedef unsigned long long uintmax_t;
typedef long ptrdiff_t;
"""

_STDDEF_H = """
typedef unsigned long size_t;
typedef long ptrdiff_t;
typedef int wchar_t;
#define NULL ((void*)0)
#define offsetof(type, member) __builtin_offsetof(type, member)
"""

_STDBOOL_H = """
#define bool _Bool
#define true 1
#define false 0
"""

_STDLIB_H = """
typedef unsigned long size_t;
void *malloc(size_t size);
void *calloc(size_t nmemb, size_t size);
void *realloc(void *ptr, size_t size);
void free(void *ptr);
void abort(void);
void exit(int status);
int atoi(const char *nptr);
long atol(const char *nptr);
long strtol(const char *nptr, char **endptr, int base);
unsigned long strtoul(const char *nptr, char **endptr, int base);
int abs(int j);
long labs(long j);
"""

_STRING_H = """
typedef unsigned long size_t;
void *memcpy(void *dest, const void *src, size_t n);
void *memmove(void *dest, const void *src, size_t n);
void *memset(void *s, int c, size_t n);
int memcmp(const void *s1, const void *s2, size_t n);
char *strcpy(char *dest, const char *src);
char *strncpy(char *dest, const char *src, size_t n);
int strcmp(const char *s1, const char *s2);
int strncmp(const char *s1, const char *s2, size_t n);
size_t strlen(const char *s);
char *strcat(char *dest, const char *src);
char *strncat(char *dest, const char *src, size_t n);
char *strchr(const char *s, int c);
char *strrchr(const char *s, int c);
char *strstr(const char *haystack, const char *needle);
"""

_STDIO_H = """
typedef unsigned long size_t;
typedef struct _IO_FILE FILE;
extern FILE *stdin;
extern FILE *stdout;
extern FILE *stderr;
int printf(const char *format, ...);
int fprintf(FILE *stream, const char *format, ...);
int sprintf(char *str, const char *format, ...);
int snprintf(char *str, size_t size, const char *format, ...);
int scanf(const char *format, ...);
FILE *fopen(const char *pathname, const char *mode);
int fclose(FILE *stream);
size_t fread(void *ptr, size_t size, size_t nmemb, FILE *stream);
size_t fwrite(const void *ptr, size_t size, size_t nmemb, FILE *stream);
int feof(FILE *stream);
int ferror(FILE *stream);
int fflush(FILE *stream);
int getchar(void);
int putchar(int c);
int puts(const char *s);
char *fgets(char *s, int size, FILE *stream);
int fputs(const char *s, FILE *stream);
"""

_LIMITS_H = """
#define CHAR_BIT 8
#define SCHAR_MIN (-128)
#define SCHAR_MAX 127
#define UCHAR_MAX 255
#define CHAR_MIN SCHAR_MIN
#define CHAR_MAX SCHAR_MAX
#define SHRT_MIN (-32768)
#define SHRT_MAX 32767
#define USHRT_MAX 65535
#define INT_MIN (-2147483647-1)
#define INT_MAX 2147483647
#define UINT_MAX 4294967295U
#define LONG_MIN (-9223372036854775807L-1L)
#define LONG_MAX 9223372036854775807L
#define ULONG_MAX 18446744073709551615UL
#define LLONG_MIN (-9223372036854775807LL-1LL)
#define LLONG_MAX 9223372036854775807LL
#define ULLONG_MAX 18446744073709551615ULL
"""

_ASSERT_H = """
#define assert(expr) ((void)0)
#define static_assert _Static_assert
"""

_ERRNO_H = """
extern int errno;
#define EDOM 33
#define ERANGE 34
#define EILSEQ 84
"""

_MATH_H = """
double fabs(double x);
float fabsf(float x);
double sqrt(double x);
float sqrtf(float x);
double pow(double x, double y);
float powf(float x, float y);
double log(double x);
float logf(float x);
double exp(double x);
float expf(float x);
double sin(double x);
float sinf(float x);
double cos(double x);
float cosf(float x);
double floor(double x);
float floorf(float x);
double ceil(double x);
float ceilf(float x);
double round(double x);
float roundf(float x);
int isnan(double x);
int isinf(double x);
int isfinite(double x);
#define INFINITY (__builtin_inff())
#define NAN (__builtin_nanf(""))
#define HUGE_VAL (__builtin_huge_val())
"""

_STDARG_H = """
typedef __builtin_va_list va_list;
#define va_start(ap, param) ((void)0)
#define va_end(ap) ((void)0)
#define va_arg(ap, type) ((type)0)
#define va_copy(dest, src) ((void)0)
"""

_SIGNAL_H = """
typedef void (*sighandler_t)(int);
#define SIG_DFL ((sighandler_t)0)
#define SIG_IGN ((sighandler_t)1)
#define SIG_ERR ((sighandler_t)-1)
#define SIGABRT 6
#define SIGFPE 8
#define SIGILL 4
#define SIGINT 2
#define SIGSEGV 11
#define SIGTERM 15
sighandler_t signal(int signum, sighandler_t handler);
int raise(int sig);
"""

_SETJMP_H = """
typedef int jmp_buf[64];
int setjmp(jmp_buf env);
void longjmp(jmp_buf env, int val);
"""

_FLOAT_H = """
#define FLT_MIN 1.17549435e-38F
#define FLT_MAX 3.40282347e+38F
#define FLT_EPSILON 1.19209290e-07F
#define DBL_MIN 2.2250738585072014e-308
#define DBL_MAX 1.7976931348623157e+308
#define DBL_EPSILON 2.2204460492503131e-16
#define FLT_MANT_DIG 24
#define DBL_MANT_DIG 53
#define FLT_RADIX 2
"""

_INTTYPES_H = """
#include <stdint.h>
#define PRId8 "d"
#define PRId16 "d"
#define PRId32 "d"
#define PRId64 "ld"
#define PRIu8 "u"
#define PRIu16 "u"
#define PRIu32 "u"
#define PRIu64 "lu"
#define PRIx32 "x"
#define PRIx64 "lx"
"""
