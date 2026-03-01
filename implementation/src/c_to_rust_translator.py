"""
C-to-Rust source code translator.

Parses C source code via regex-based tokenization and pattern matching,
transforms internal representation, and emits idiomatic Rust code.
"""

import re
from typing import List, Tuple, Dict, Optional


# ---------------------------------------------------------------------------
# TypeTranslator
# ---------------------------------------------------------------------------

class TypeTranslator:
    """Translates C type strings to Rust type strings."""

    PRIMITIVE_MAP: Dict[str, str] = {
        "int": "i32",
        "unsigned int": "u32",
        "signed int": "i32",
        "long": "i64",
        "long long": "i64",
        "unsigned long": "u64",
        "unsigned long long": "u64",
        "short": "i16",
        "unsigned short": "u16",
        "char": "i8",
        "unsigned char": "u8",
        "signed char": "i8",
        "float": "f32",
        "double": "f64",
        "long double": "f64",
        "void": "()",
        "size_t": "usize",
        "ssize_t": "isize",
        "ptrdiff_t": "isize",
        "bool": "bool",
        "_Bool": "bool",
        "int8_t": "i8",
        "int16_t": "i16",
        "int32_t": "i32",
        "int64_t": "i64",
        "uint8_t": "u8",
        "uint16_t": "u16",
        "uint32_t": "u32",
        "uint64_t": "u64",
    }

    HEADER_TYPE_MAP: Dict[str, str] = {
        "FILE": "std::fs::File",
        "FILE*": "std::fs::File",
    }

    def translate_type(self, c_type_str: str) -> str:
        c_type_str = c_type_str.strip()
        if not c_type_str:
            return "()"

        # function pointer: int (*fp)(int, int)
        fp_match = re.match(
            r"(\w[\w\s\*]*?)\s*\(\s*\*\s*(\w*)\s*\)\s*\((.*)\)", c_type_str
        )
        if fp_match:
            return self._translate_function_pointer(fp_match)

        # const pointer: const T*
        const_ptr = re.match(r"const\s+(\w[\w\s]*)\s*\*", c_type_str)
        if const_ptr:
            inner = self.translate_type(const_ptr.group(1).strip())
            return f"&{inner}"

        # void*
        if re.match(r"void\s*\*", c_type_str):
            return "*mut u8"

        # char* – treat as string
        if re.match(r"(const\s+)?char\s*\*", c_type_str):
            if "const" in c_type_str:
                return "&str"
            return "String"

        # T* (other pointer)
        ptr_match = re.match(r"(\w[\w\s]*?)\s*\*", c_type_str)
        if ptr_match:
            inner = self.translate_type(ptr_match.group(1).strip())
            return f"*mut {inner}"

        # T[] or T[N]
        arr_match = re.match(r"(\w[\w\s]*?)\s*\[\s*(\d*)\s*\]", c_type_str)
        if arr_match:
            inner = self.translate_type(arr_match.group(1).strip())
            size = arr_match.group(2)
            if size:
                return f"[{inner}; {size}]"
            return f"Vec<{inner}>"

        # struct X / enum X
        struct_match = re.match(r"struct\s+(\w+)", c_type_str)
        if struct_match:
            return struct_match.group(1)
        enum_match = re.match(r"enum\s+(\w+)", c_type_str)
        if enum_match:
            return enum_match.group(1)

        # FILE*
        if c_type_str in self.HEADER_TYPE_MAP:
            return self.HEADER_TYPE_MAP[c_type_str]

        # strip qualifiers for lookup
        cleaned = re.sub(r"\b(static|extern|register|volatile|restrict)\b", "", c_type_str).strip()
        cleaned = re.sub(r"\s+", " ", cleaned)

        if cleaned in self.PRIMITIVE_MAP:
            return self.PRIMITIVE_MAP[cleaned]

        # If nothing matched keep the name (user-defined type)
        return cleaned


    def _translate_function_pointer(self, m: re.Match) -> str:
        ret_type = self.translate_type(m.group(1).strip())
        params_str = m.group(3).strip()
        if params_str in ("", "void"):
            params = []
        else:
            params = [self.translate_type(p.strip()) for p in params_str.split(",")]
        params_rust = ", ".join(params)
        if ret_type == "()":
            return f"fn({params_rust})"
        return f"fn({params_rust}) -> {ret_type}"


# ---------------------------------------------------------------------------
# ExprTranslator
# ---------------------------------------------------------------------------

class ExprTranslator:
    """Translates C expression strings to Rust expression strings."""

    def __init__(self, type_translator: Optional[TypeTranslator] = None):
        self.tt = type_translator or TypeTranslator()

    def translate_expr(self, expr: str) -> str:
        expr = expr.strip()
        if not expr:
            return expr

        # NULL
        if expr == "NULL":
            return "std::ptr::null_mut()"

        # sizeof(T) or sizeof expr
        sizeof_m = re.match(r"sizeof\s*\(\s*(.+?)\s*\)", expr)
        if sizeof_m:
            inner = sizeof_m.group(1).strip()
            rust_type = self.tt.translate_type(inner)
            return f"std::mem::size_of::<{rust_type}>()"

        # Cast: (type)expr
        cast_m = re.match(r"\((\w[\w\s\*]*)\)\s*(.+)", expr)
        if cast_m:
            target = self.tt.translate_type(cast_m.group(1).strip())
            operand = self.translate_expr(cast_m.group(2).strip())
            return f"{operand} as {target}"

        # Ternary: a ? b : c
        ternary_m = re.match(r"(.+?)\s*\?\s*(.+?)\s*:\s*(.+)", expr)
        if ternary_m:
            cond = self.translate_expr(ternary_m.group(1).strip())
            then_e = self.translate_expr(ternary_m.group(2).strip())
            else_e = self.translate_expr(ternary_m.group(3).strip())
            return f"if {cond} {{ {then_e} }} else {{ {else_e} }}"

        # Post-increment: x++
        post_inc = re.match(r"(\w+)\s*\+\+$", expr)
        if post_inc:
            var = post_inc.group(1)
            return f"{{ let _tmp = {var}; {var} += 1; _tmp }}"

        # Post-decrement: x--
        post_dec = re.match(r"(\w+)\s*--$", expr)
        if post_dec:
            var = post_dec.group(1)
            return f"{{ let _tmp = {var}; {var} -= 1; _tmp }}"

        # Pre-increment: ++x
        pre_inc = re.match(r"^\+\+(\w+)", expr)
        if pre_inc:
            var = pre_inc.group(1)
            return f"{{ {var} += 1; {var} }}"

        # Pre-decrement: --x
        pre_dec = re.match(r"^--(\w+)", expr)
        if pre_dec:
            var = pre_dec.group(1)
            return f"{{ {var} -= 1; {var} }}"

        # Pointer dereference: *p
        deref_m = re.match(r"^\*(\w+)$", expr)
        if deref_m:
            return f"unsafe {{ *{deref_m.group(1)} }}"

        # Address-of: &x
        addr_m = re.match(r"^&(\w+)$", expr)
        if addr_m:
            return f"&{addr_m.group(1)}"

        # Pointer arithmetic: p + n
        ptr_arith = re.match(r"(\w+)\s*\+\s*(\w+)", expr)
        if ptr_arith:
            left = ptr_arith.group(1)
            right = ptr_arith.group(2)
            # We cannot always tell; keep as-is (could be int addition)
            return f"{left} + {right}"

        # Comma operator: (a, b)
        if "," in expr and expr.startswith("(") and expr.endswith(")"):
            inner = expr[1:-1]
            parts = [self.translate_expr(p.strip()) for p in inner.split(",")]
            body = "; ".join(parts[:-1]) + "; " + parts[-1]
            return "{ " + body + " }"

        # Compound assignment handled at stmt level; pass through
        # Array subscript: arr[i] – keep as is
        # String literal – keep as is

        return expr


# ---------------------------------------------------------------------------
# MemoryTranslator
# ---------------------------------------------------------------------------

class MemoryTranslator:
    """Translates C memory-management calls to Rust equivalents."""

    def __init__(self, type_translator: Optional[TypeTranslator] = None):
        self.tt = type_translator or TypeTranslator()

    def translate_memory_call(self, call_name: str, args: List[str]) -> str:
        call_name = call_name.strip()
        args = [a.strip() for a in args]

        if call_name == "malloc":
            return self._translate_malloc(args)
        if call_name == "calloc":
            return self._translate_calloc(args)
        if call_name == "realloc":
            return self._translate_realloc(args)
        if call_name == "free":
            return self._translate_free(args)
        if call_name == "memcpy":
            return self._translate_memcpy(args)
        if call_name == "memset":
            return self._translate_memset(args)
        if call_name == "memmove":
            return self._translate_memmove(args)

        return f"{call_name}({', '.join(args)})"

    # -- helpers -----------------------------------------------------------

    def _parse_sizeof(self, arg: str) -> Optional[str]:
        m = re.match(r"sizeof\s*\(\s*(.+?)\s*\)", arg)
        if m:
            return self.tt.translate_type(m.group(1))
        return None

    def _translate_malloc(self, args: List[str]) -> str:
        if not args:
            return "Vec::new()"
        arg = args[0]
        # n * sizeof(T)
        mul_m = re.match(r"(.+?)\s*\*\s*sizeof\s*\(\s*(.+?)\s*\)", arg)
        if mul_m:
            count = mul_m.group(1).strip()
            rust_t = self.tt.translate_type(mul_m.group(2).strip())
            return f"vec![{rust_t}::default(); {count}]"
        # sizeof(T) * n
        mul_m2 = re.match(r"sizeof\s*\(\s*(.+?)\s*\)\s*\*\s*(.+)", arg)
        if mul_m2:
            rust_t = self.tt.translate_type(mul_m2.group(1).strip())
            count = mul_m2.group(2).strip()
            return f"vec![{rust_t}::default(); {count}]"
        # sizeof(T) alone
        st = self._parse_sizeof(arg)
        if st:
            return f"Box::new({st}::default())"
        return f"Vec::<u8>::with_capacity({arg})"

    def _translate_calloc(self, args: List[str]) -> str:
        if len(args) < 2:
            return "Vec::new()"
        count = args[0]
        st = self._parse_sizeof(args[1])
        if st:
            return f"vec![{st}::default(); {count}]"
        return f"vec![0u8; {count} * {args[1]}]"

    def _translate_realloc(self, args: List[str]) -> str:
        if len(args) < 2:
            return "/* realloc: check manually */"
        ptr = args[0]
        new_size = args[1]
        return f"{ptr}.resize({new_size}, Default::default())"

    def _translate_free(self, args: List[str]) -> str:
        if args:
            return f"drop({args[0]})"
        return "/* free */"

    def _translate_memcpy(self, args: List[str]) -> str:
        if len(args) < 3:
            return "/* memcpy: check manually */"
        dst, src, n = args[0], args[1], args[2]
        return f"{dst}[..{n}].copy_from_slice(&{src}[..{n}])"

    def _translate_memset(self, args: List[str]) -> str:
        if len(args) < 3:
            return "/* memset: check manually */"
        ptr, val, n = args[0], args[1], args[2]
        return f"{ptr}[..{n}].iter_mut().for_each(|b| *b = {val})"

    def _translate_memmove(self, args: List[str]) -> str:
        if len(args) < 3:
            return "/* memmove: check manually */"
        dst, src, n = args[0], args[1], args[2]
        return f"{dst}.copy_within(..{n}, {src})"


# ---------------------------------------------------------------------------
# StringTranslator
# ---------------------------------------------------------------------------

class StringTranslator:
    """Translates C string/stdio calls to Rust equivalents."""

    FMT_SPEC_MAP = {
        "%d": "{}",
        "%i": "{}",
        "%u": "{}",
        "%ld": "{}",
        "%lu": "{}",
        "%lld": "{}",
        "%llu": "{}",
        "%f": "{}",
        "%lf": "{}",
        "%e": "{:e}",
        "%g": "{}",
        "%c": "{}",
        "%s": "{}",
        "%p": "{:p}",
        "%x": "{:x}",
        "%X": "{:X}",
        "%o": "{:o}",
        "%zu": "{}",
        "%zd": "{}",
        "%%": "%",
    }

    def translate_string_call(self, call_name: str, args: List[str]) -> str:
        call_name = call_name.strip()
        args = [a.strip() for a in args]

        dispatch = {
            "printf": self._translate_printf,
            "fprintf": self._translate_fprintf,
            "sprintf": self._translate_sprintf,
            "snprintf": self._translate_snprintf,
            "puts": self._translate_puts,
            "strcmp": self._translate_strcmp,
            "strncmp": self._translate_strncmp,
            "strlen": self._translate_strlen,
            "strcpy": self._translate_strcpy,
            "strncpy": self._translate_strncpy,
            "strcat": self._translate_strcat,
            "strncat": self._translate_strncat,
            "strtol": self._translate_strtol,
            "atoi": self._translate_atoi,
            "atof": self._translate_atof,
            "strchr": self._translate_strchr,
            "strstr": self._translate_strstr,
        }
        fn = dispatch.get(call_name)
        if fn:
            return fn(args)
        return f"{call_name}({', '.join(args)})"

    def convert_printf_format(self, c_fmt: str) -> str:
        result = c_fmt
        # Sort specs by length descending so %ld matches before %d
        for spec in sorted(self.FMT_SPEC_MAP, key=len, reverse=True):
            result = result.replace(spec, self.FMT_SPEC_MAP[spec])
        # Handle width/precision specs like %5d, %.2f
        result = re.sub(r"%(\d+)d", r"{:\1}", result)
        result = re.sub(r"%(\d*\.\d+)f", r"{:\1}", result)
        return result

    # -- helpers -----------------------------------------------------------

    def _format_args(self, fmt_arg: str, rest: List[str]) -> str:
        rust_fmt = self.convert_printf_format(fmt_arg)
        if rest:
            return f'{rust_fmt}, {", ".join(rest)}'
        return rust_fmt

    def _translate_printf(self, args: List[str]) -> str:
        if not args:
            return 'print!("")'
        fmt = args[0]
        rest = args[1:]
        fa = self._format_args(fmt, rest)
        if fmt.endswith('\\n"'):
            fa = self._format_args(fmt.replace('\\n"', '"'), rest)
            return f"println!({fa})"
        return f"print!({fa})"

    def _translate_fprintf(self, args: List[str]) -> str:
        if len(args) < 2:
            return 'eprintln!("")'
        stream = args[0]
        fmt = args[1]
        rest = args[2:]
        fa = self._format_args(fmt, rest)
        if stream == "stderr":
            return f"eprintln!({fa})"
        return f"write!({stream}, {fa})"

    def _translate_sprintf(self, args: List[str]) -> str:
        if len(args) < 2:
            return "String::new()"
        buf = args[0]
        fmt = args[1]
        rest = args[2:]
        fa = self._format_args(fmt, rest)
        return f"{buf} = format!({fa})"

    def _translate_snprintf(self, args: List[str]) -> str:
        if len(args) < 3:
            return "String::new()"
        buf = args[0]
        fmt = args[2]
        rest = args[3:]
        fa = self._format_args(fmt, rest)
        return f"{buf} = format!({fa})"

    def _translate_puts(self, args: List[str]) -> str:
        if args:
            return f"println!({args[0]})"
        return 'println!()'

    def _translate_strcmp(self, args: List[str]) -> str:
        if len(args) >= 2:
            return f"{args[0]} == {args[1]}"
        return "false"

    def _translate_strncmp(self, args: List[str]) -> str:
        if len(args) >= 3:
            return f"{args[0]}[..{args[2]}] == {args[1]}[..{args[2]}]"
        return "false"

    def _translate_strlen(self, args: List[str]) -> str:
        if args:
            return f"{args[0]}.len()"
        return "0"

    def _translate_strcpy(self, args: List[str]) -> str:
        if len(args) >= 2:
            return f"{args[0]} = {args[1]}.to_string()"
        return "String::new()"

    def _translate_strncpy(self, args: List[str]) -> str:
        if len(args) >= 3:
            return f"{args[0]}[..{args[2]}].copy_from_slice(&{args[1]}[..{args[2]}])"
        return "/* strncpy */"

    def _translate_strcat(self, args: List[str]) -> str:
        if len(args) >= 2:
            return f"{args[0]}.push_str({args[1]})"
        return "/* strcat */"

    def _translate_strncat(self, args: List[str]) -> str:
        if len(args) >= 3:
            return f"{args[0]}.push_str(&{args[1]}[..{args[2]}])"
        return "/* strncat */"

    def _translate_strtol(self, args: List[str]) -> str:
        if args:
            return f"{args[0]}.parse::<i64>().unwrap_or(0)"
        return "0i64"

    def _translate_atoi(self, args: List[str]) -> str:
        if args:
            return f"{args[0]}.parse::<i32>().unwrap_or(0)"
        return "0i32"

    def _translate_atof(self, args: List[str]) -> str:
        if args:
            return f"{args[0]}.parse::<f64>().unwrap_or(0.0)"
        return "0.0f64"

    def _translate_strchr(self, args: List[str]) -> str:
        if len(args) >= 2:
            return f"{args[0]}.find({args[1]})"
        return "None"

    def _translate_strstr(self, args: List[str]) -> str:
        if len(args) >= 2:
            return f"{args[0]}.find({args[1]})"
        return "None"


# ---------------------------------------------------------------------------
# ErrorTranslator
# ---------------------------------------------------------------------------

class ErrorTranslator:
    """Translates common C error-handling patterns to Rust idioms."""

    def translate_error_pattern(self, stmts: str) -> str:
        result = stmts

        # if (ptr == NULL) return -1;  →  let ptr = ptr.ok_or(Error::new(...))?;
        null_check = re.compile(
            r"if\s*\(\s*(\w+)\s*==\s*NULL\s*\)\s*return\s+(-?\w+)\s*;"
        )
        for m in null_check.finditer(stmts):
            var = m.group(1)
            old = m.group(0)
            replacement = (
                f'let {var} = {var}.ok_or_else(|| '
                f'std::io::Error::new(std::io::ErrorKind::Other, '
                f'"{var} is null"))?;'
            )
            result = result.replace(old, replacement)

        # if (ret < 0) { perror(...); return ret; }
        perror_pat = re.compile(
            r"if\s*\(\s*(\w+)\s*<\s*0\s*\)\s*\{[^}]*perror\s*\([^)]*\)\s*;[^}]*return\s+\w+\s*;[^}]*\}"
        )
        for m in perror_pat.finditer(result):
            old = m.group(0)
            var = m.group(1)
            replacement = (
                f"if {var} < 0 {{ return Err(std::io::Error::last_os_error()); }}"
            )
            result = result.replace(old, replacement)

        # goto cleanup;  →  early return or ? operator
        result = re.sub(
            r"goto\s+cleanup\s*;",
            "return Err(std::io::Error::last_os_error());",
            result,
        )

        # errno  →  std::io::Error::last_os_error()
        result = result.replace("errno", "std::io::Error::last_os_error()")

        return result


# ---------------------------------------------------------------------------
# HeaderTranslator
# ---------------------------------------------------------------------------

class HeaderTranslator:
    """Translates C preprocessor directives to Rust equivalents."""

    INCLUDE_MAP: Dict[str, str] = {
        "stdio.h": "use std::io::{self, Read, Write};",
        "stdlib.h": "",
        "string.h": "",
        "strings.h": "",
        "math.h": "",
        "ctype.h": "",
        "assert.h": "",
        "stdint.h": "",
        "stdbool.h": "",
        "stddef.h": "",
        "limits.h": "",
        "float.h": "",
        "errno.h": "use std::io;",
        "time.h": "use std::time;",
        "unistd.h": "use std::os::unix;",
        "fcntl.h": "use std::fs;",
        "sys/types.h": "",
        "sys/stat.h": "use std::fs;",
        "pthread.h": "use std::thread;",
        "signal.h": "",
    }

    def __init__(self, type_translator: Optional[TypeTranslator] = None):
        self.tt = type_translator or TypeTranslator()

    def translate_header(self, directive: str) -> str:
        directive = directive.strip()

        # #include <header>
        inc_sys = re.match(r'#\s*include\s*<\s*(.+?)\s*>', directive)
        if inc_sys:
            header = inc_sys.group(1)
            return self.INCLUDE_MAP.get(header, f"// #include <{header}>")

        # #include "header.h"
        inc_local = re.match(r'#\s*include\s*"(.+?)"', directive)
        if inc_local:
            mod_name = inc_local.group(1).replace(".h", "").replace("/", "::")
            return f"mod {mod_name};\nuse {mod_name}::*;"

        # #define CONSTANT value
        define_const = re.match(
            r"#\s*define\s+([A-Z_][A-Z0-9_]*)\s+(.+)", directive
        )
        if define_const:
            name = define_const.group(1)
            value = define_const.group(2).strip()
            rust_type = self._infer_const_type(value)
            return f"const {name}: {rust_type} = {value};"

        # #define MACRO(args) body
        define_macro = re.match(
            r"#\s*define\s+(\w+)\s*\(([^)]*)\)\s+(.+)", directive
        )
        if define_macro:
            name = define_macro.group(1)
            params = define_macro.group(2).strip()
            body = define_macro.group(3).strip()
            rust_params = ", ".join(
                f"{p.strip()}: impl Into<i64>" for p in params.split(",") if p.strip()
            )
            return (
                f"// Macro translation (verify manually):\n"
                f"fn {name.lower()}({rust_params}) -> i64 {{ {body} }}"
            )

        # #ifdef / #ifndef / #endif / #if / #else / #elif – emit as comments
        if re.match(r"#\s*(ifdef|ifndef|if|else|elif|endif|undef|pragma)", directive):
            return f"// {directive}"

        return f"// {directive}"

    def _infer_const_type(self, value: str) -> str:
        if re.match(r'^".*"$', value):
            return "&str"
        if re.match(r"^'.'$", value):
            return "char"
        if re.match(r"^-?\d+\.\d+[fF]?$", value):
            if value.endswith(("f", "F")):
                return "f32"
            return "f64"
        if re.match(r"^0[xX][0-9a-fA-F]+$", value):
            return "u32"
        if re.match(r"^-?\d+$", value):
            return "i32"
        return "i32"


# ---------------------------------------------------------------------------
# StmtTranslator
# ---------------------------------------------------------------------------

class StmtTranslator:
    """Translates C statement strings to Rust statement strings."""

    def __init__(
        self,
        type_translator: Optional[TypeTranslator] = None,
        expr_translator: Optional[ExprTranslator] = None,
        mem_translator: Optional[MemoryTranslator] = None,
        str_translator: Optional[StringTranslator] = None,
    ):
        self.tt = type_translator or TypeTranslator()
        self.et = expr_translator or ExprTranslator(self.tt)
        self.mt = mem_translator or MemoryTranslator(self.tt)
        self.st = str_translator or StringTranslator()

    def translate_stmt(self, stmt: str) -> str:
        stmt = stmt.strip()
        if not stmt:
            return ""

        # Variable declaration: type var = expr;
        decl = re.match(
            r"([\w\s\*]+?)\s+(\w+)\s*=\s*(.+?)\s*;$", stmt
        )
        if decl:
            return self._translate_var_decl(decl)

        # Variable declaration without init: type var;
        decl_no_init = re.match(r"([\w\s\*]+?)\s+(\w+)\s*;$", stmt)
        if decl_no_init:
            c_type = decl_no_init.group(1).strip()
            var = decl_no_init.group(2)
            if c_type in ("int", "unsigned int", "long", "short", "float",
                          "double", "char", "unsigned char", "size_t",
                          "int32_t", "int64_t", "uint32_t", "uint64_t"):
                rust_t = self.tt.translate_type(c_type)
                return f"let mut {var}: {rust_t} = Default::default();"
            rust_t = self.tt.translate_type(c_type)
            return f"let mut {var}: {rust_t};"

        # For loop
        for_m = re.match(
            r"for\s*\(\s*(.+?)\s*;\s*(.+?)\s*;\s*(.+?)\s*\)", stmt
        )
        if for_m:
            return self._translate_for(for_m)

        # While loop
        while_m = re.match(r"while\s*\(\s*(.+?)\s*\)", stmt)
        if while_m:
            cond = self.et.translate_expr(while_m.group(1))
            return f"while {cond}"

        # Do-while (opening)
        if stmt.strip().startswith("do"):
            return "loop"

        # Switch
        switch_m = re.match(r"switch\s*\(\s*(.+?)\s*\)", stmt)
        if switch_m:
            expr = self.et.translate_expr(switch_m.group(1))
            return f"match {expr}"

        # Case
        case_m = re.match(r"case\s+(.+?)\s*:", stmt)
        if case_m:
            val = case_m.group(1).strip()
            return f"{val} =>"

        # Default
        if stmt.strip() == "default:":
            return "_ =>"

        # Break
        if stmt.strip().rstrip(";") == "break":
            return "break;"

        # Continue
        if stmt.strip().rstrip(";") == "continue":
            return "continue;"

        # Return
        ret_m = re.match(r"return\s+(.*?)\s*;$", stmt)
        if ret_m:
            val = self.et.translate_expr(ret_m.group(1))
            return f"return {val};"
        if stmt.strip() == "return;":
            return "return;"

        # Goto
        goto_m = re.match(r"goto\s+(\w+)\s*;", stmt)
        if goto_m:
            label = goto_m.group(1)
            return f"// goto {label} — use labeled block: 'break '{label};'"

        # Label
        label_m = re.match(r"^(\w+)\s*:$", stmt)
        if label_m and label_m.group(1) not in ("default",):
            return f"// label '{label_m.group(1)}:"

        # If / else if / else
        if_m = re.match(r"if\s*\(\s*(.+?)\s*\)$", stmt)
        if if_m:
            cond = self.et.translate_expr(if_m.group(1))
            return f"if {cond}"
        if stmt.strip().startswith("else if"):
            eif_m = re.match(r"else\s+if\s*\(\s*(.+?)\s*\)$", stmt)
            if eif_m:
                cond = self.et.translate_expr(eif_m.group(1))
                return f"else if {cond}"
        if stmt.strip() == "else":
            return "else"

        # Function call statement: func(args);
        call_m = re.match(r"(\w+)\s*\(([^)]*)\)\s*;$", stmt)
        if call_m:
            fname = call_m.group(1)
            raw_args = self._split_args(call_m.group(2))
            mem_funcs = {"malloc", "calloc", "realloc", "free", "memcpy", "memset", "memmove"}
            if fname in mem_funcs:
                return self.mt.translate_memory_call(fname, raw_args) + ";"
            str_funcs = {
                "printf", "fprintf", "sprintf", "snprintf", "puts",
                "strcmp", "strncmp", "strlen", "strcpy", "strncpy",
                "strcat", "strncat", "strtol", "atoi", "atof",
                "strchr", "strstr",
            }
            if fname in str_funcs:
                return self.st.translate_string_call(fname, raw_args) + ";"
            translated_args = ", ".join(self.et.translate_expr(a) for a in raw_args)
            return f"{fname}({translated_args});"

        # Assignment: x = expr;  or compound assignment
        assign_m = re.match(r"(\w+)\s*([+\-*/&|^%]?=)\s*(.+?)\s*;$", stmt)
        if assign_m:
            lhs = assign_m.group(1)
            op = assign_m.group(2)
            rhs = self.et.translate_expr(assign_m.group(3))
            return f"{lhs} {op} {rhs};"

        # Expression statement ending with ;
        if stmt.endswith(";"):
            inner = stmt[:-1].strip()
            return self.et.translate_expr(inner) + ";"

        return f"// UNTRANSLATED: {stmt}"

    # -- helpers -----------------------------------------------------------

    def _translate_var_decl(self, m: re.Match) -> str:
        c_type = m.group(1).strip()
        var = m.group(2)
        init = m.group(3).strip()

        # Check if init is a memory or string call
        call_m = re.match(r"(\w+)\s*\(([^)]*)\)", init)
        if call_m:
            fname = call_m.group(1)
            raw_args = self._split_args(call_m.group(2))
            mem_funcs = {"malloc", "calloc", "realloc"}
            if fname in mem_funcs:
                rhs = self.mt.translate_memory_call(fname, raw_args)
                return f"let mut {var} = {rhs};"
            str_funcs = {"atoi", "atof", "strtol", "strlen", "strchr", "strstr"}
            if fname in str_funcs:
                rhs = self.st.translate_string_call(fname, raw_args)
                return f"let {var} = {rhs};"

        rust_t = self.tt.translate_type(c_type)
        init_translated = self.et.translate_expr(init)
        return f"let mut {var}: {rust_t} = {init_translated};"

    def _translate_for(self, m: re.Match) -> str:
        init = m.group(1).strip()
        cond = m.group(2).strip()
        incr = m.group(3).strip()

        # Detect simple range loop: int i = start; i < end; i++
        range_m = re.match(r"(?:int\s+)?(\w+)\s*=\s*(\w+)", init)
        cond_m = re.match(r"(\w+)\s*<\s*(\w+)", cond)
        incr_m = re.match(r"(\w+)\s*\+\+", incr)

        if range_m and cond_m and incr_m:
            var = range_m.group(1)
            start = range_m.group(2)
            end = cond_m.group(2)
            if var == cond_m.group(1) == incr_m.group(1):
                if start == "0":
                    return f"for {var} in 0..{end}"
                return f"for {var} in {start}..{end}"

        # General for → while
        init_r = self.translate_stmt(init + ";") if not init.endswith(";") else self.translate_stmt(init)
        cond_r = self.et.translate_expr(cond)
        incr_r = self.et.translate_expr(incr)
        return f"{init_r}\nwhile {cond_r} {{\n    // body\n    {incr_r};\n}}"

    @staticmethod
    def _split_args(args_str: str) -> List[str]:
        args: List[str] = []
        depth = 0
        current: List[str] = []
        for ch in args_str:
            if ch == "(" :
                depth += 1
                current.append(ch)
            elif ch == ")":
                depth -= 1
                current.append(ch)
            elif ch == "," and depth == 0:
                args.append("".join(current).strip())
                current = []
            else:
                current.append(ch)
        tail = "".join(current).strip()
        if tail:
            args.append(tail)
        return args


# ---------------------------------------------------------------------------
# RustFormatter
# ---------------------------------------------------------------------------

class RustFormatter:
    """Post-processes generated Rust code for readability."""

    INDENT = "    "

    def format_rust(self, code: str) -> str:
        lines = code.split("\n")
        lines = self._add_use_declarations(lines)
        lines = self._add_derive_macros(lines)
        lines = self._indent(lines)
        lines = self._normalize_blank_lines(lines)
        lines = self._fix_operator_spacing(lines)
        return "\n".join(lines) + "\n"

    # -- helpers -----------------------------------------------------------

    def _indent(self, lines: List[str]) -> List[str]:
        result: List[str] = []
        level = 0
        for raw in lines:
            stripped = raw.strip()
            if not stripped:
                result.append("")
                continue
            # Decrease before printing for closing braces
            if stripped.startswith("}") or stripped.startswith("]"):
                level = max(level - 1, 0)
            result.append(self.INDENT * level + stripped)
            # Increase after opening braces
            open_count = stripped.count("{") + stripped.count("[")
            close_count = stripped.count("}") + stripped.count("]")
            level += open_count - close_count
            # clamp for safety
            if stripped.startswith("}"):
                pass  # already handled
            level = max(level, 0)
        return result

    def _add_use_declarations(self, lines: List[str]) -> List[str]:
        uses: List[str] = []
        other: List[str] = []
        for ln in lines:
            if ln.strip().startswith("use "):
                uses.append(ln.strip())
            else:
                other.append(ln)
        # Deduplicate and sort
        uses = sorted(set(uses))
        if uses:
            return uses + [""] + other
        return other

    def _add_derive_macros(self, lines: List[str]) -> List[str]:
        result: List[str] = []
        i = 0
        while i < len(lines):
            stripped = lines[i].strip()
            if re.match(r"(pub\s+)?struct\s+\w+", stripped):
                # Check if the previous non-blank line is already a derive
                prev_idx = len(result) - 1
                while prev_idx >= 0 and result[prev_idx].strip() == "":
                    prev_idx -= 1
                if prev_idx < 0 or not result[prev_idx].strip().startswith("#[derive"):
                    result.append("#[derive(Debug, Clone, Default)]")
            result.append(lines[i])
            i += 1
        return result

    def _normalize_blank_lines(self, lines: List[str]) -> List[str]:
        result: List[str] = []
        prev_blank = False
        for ln in lines:
            is_blank = ln.strip() == ""
            if is_blank and prev_blank:
                continue
            result.append(ln)
            prev_blank = is_blank
        # Remove trailing blanks
        while result and result[-1].strip() == "":
            result.pop()
        return result

    def _fix_operator_spacing(self, lines: List[str]) -> List[str]:
        result: List[str] = []
        for ln in lines:
            # Ensure spaces around = but not ==, !=, <=, >=
            ln = re.sub(r'(?<!=)(?<!!)(?<!<)(?<!>)=(?!=)', ' = ', ln)
            # Collapse multiple spaces (except leading)
            leading = len(ln) - len(ln.lstrip())
            ln = ln[:leading] + re.sub(r"  +", " ", ln[leading:])
            result.append(ln)
        return result


# ---------------------------------------------------------------------------
# CToRustTranslator  (main entry point)
# ---------------------------------------------------------------------------

class CToRustTranslator:
    """
    Translates a C source file to Rust source code.

    Usage::

        translator = CToRustTranslator()
        rust_code = translator.translate(c_source_string)
    """

    def __init__(self):
        self.tt = TypeTranslator()
        self.et = ExprTranslator(self.tt)
        self.mt = MemoryTranslator(self.tt)
        self.st_tr = StringTranslator()
        self.stmt = StmtTranslator(self.tt, self.et, self.mt, self.st_tr)
        self.err = ErrorTranslator()
        self.hdr = HeaderTranslator(self.tt)
        self.fmt = RustFormatter()

    def translate(self, c_source: str) -> str:
        c_source = self._strip_comments(c_source)
        lines = c_source.split("\n")
        rust_lines: List[str] = []

        i = 0
        while i < len(lines):
            line = lines[i]
            stripped = line.strip()

            if not stripped:
                rust_lines.append("")
                i += 1
                continue

            # Preprocessor directive
            if stripped.startswith("#"):
                rust_lines.append(self.hdr.translate_header(stripped))
                i += 1
                continue

            # Typedef
            typedef_m = re.match(r"typedef\s+(.+?)\s+(\w+)\s*;", stripped)
            if typedef_m:
                old = self.tt.translate_type(typedef_m.group(1))
                new = typedef_m.group(2)
                rust_lines.append(f"type {new} = {old};")
                i += 1
                continue

            # Struct definition
            struct_m = re.match(r"(typedef\s+)?struct\s+(\w+)?\s*\{", stripped)
            if struct_m:
                block, end_i = self._collect_block(lines, i)
                rust_lines.extend(self._translate_struct(block, struct_m.group(2)))
                i = end_i + 1
                continue

            # Enum definition
            enum_m = re.match(r"(typedef\s+)?enum\s+(\w+)?\s*\{", stripped)
            if enum_m:
                block, end_i = self._collect_block(lines, i)
                rust_lines.extend(self._translate_enum(block, enum_m.group(2)))
                i = end_i + 1
                continue

            # Function definition
            func_m = re.match(
                r"([\w\s\*]+?)\s+(\w+)\s*\(([^)]*)\)\s*\{?", stripped
            )
            if func_m and "{" in stripped:
                block, end_i = self._collect_block(lines, i)
                rust_lines.extend(self._translate_function(func_m, block))
                i = end_i + 1
                continue

            # Function prototype / declaration
            proto_m = re.match(
                r"([\w\s\*]+?)\s+(\w+)\s*\(([^)]*)\)\s*;", stripped
            )
            if proto_m:
                rust_lines.append(self._translate_prototype(proto_m))
                i += 1
                continue

            # General statement
            rust_lines.append(self.stmt.translate_stmt(stripped))
            i += 1

        code = "\n".join(rust_lines)
        code = self.err.translate_error_pattern(code)
        return self.fmt.format_rust(code)

    # -- helpers -----------------------------------------------------------

    @staticmethod
    def _strip_comments(source: str) -> str:
        # Remove block comments
        source = re.sub(r"/\*.*?\*/", "", source, flags=re.DOTALL)
        # Remove line comments
        source = re.sub(r"//.*", "", source)
        return source

    @staticmethod
    def _collect_block(lines: List[str], start: int) -> Tuple[List[str], int]:
        depth = 0
        block: List[str] = []
        i = start
        while i < len(lines):
            line = lines[i]
            depth += line.count("{") - line.count("}")
            block.append(line)
            if depth <= 0 and i > start:
                break
            i += 1
        return block, i

    def _translate_struct(self, block: List[str], name: Optional[str]) -> List[str]:
        result: List[str] = []
        # Try to find name from typedef ending
        if not name:
            last = block[-1].strip().rstrip(";").strip().rstrip("}").strip()
            # typedef struct { ... } Name;
            after_brace = block[-1].strip()
            tm = re.search(r"}\s*(\w+)\s*;", after_brace)
            if tm:
                name = tm.group(1)
            else:
                name = "UnnamedStruct"

        result.append(f"#[derive(Debug, Clone, Default)]")
        result.append(f"pub struct {name} {{")

        for line in block[1:-1]:
            stripped = line.strip().rstrip(";").strip()
            if not stripped or stripped == "{" or stripped == "}":
                continue
            # Parse field: type name;
            field_m = re.match(r"([\w\s\*]+?)\s+(\w+)\s*(\[\d+\])?", stripped)
            if field_m:
                ft = field_m.group(1).strip()
                fn_ = field_m.group(2)
                arr = field_m.group(3)
                if arr:
                    ft = ft + " " + arr
                rust_t = self.tt.translate_type(ft)
                result.append(f"    pub {fn_}: {rust_t},")
            else:
                result.append(f"    // {stripped}")

        result.append("}")
        return result

    def _translate_enum(self, block: List[str], name: Optional[str]) -> List[str]:
        result: List[str] = []
        if not name:
            after_brace = block[-1].strip()
            tm = re.search(r"}\s*(\w+)\s*;", after_brace)
            name = tm.group(1) if tm else "UnnamedEnum"

        result.append("#[derive(Debug, Clone, Copy, PartialEq)]")
        result.append(f"pub enum {name} {{")

        for line in block[1:-1]:
            stripped = line.strip().rstrip(",").strip()
            if not stripped or stripped == "{" or stripped == "}":
                continue
            # VARIANT = value
            ev_m = re.match(r"(\w+)\s*=\s*(.+)", stripped)
            if ev_m:
                result.append(f"    {ev_m.group(1)} = {ev_m.group(2)},")
            elif stripped:
                result.append(f"    {stripped},")

        result.append("}")
        return result

    def _translate_function(self, sig_match: re.Match, block: List[str]) -> List[str]:
        ret_type_c = sig_match.group(1).strip()
        fn_name = sig_match.group(2)
        params_c = sig_match.group(3).strip()

        ret_type = self.tt.translate_type(ret_type_c)
        params_rust = self._translate_params(params_c)

        sig = f"pub fn {fn_name}({params_rust})"
        if ret_type and ret_type != "()" and ret_type != "void":
            sig += f" -> {ret_type}"
        sig += " {"

        result: List[str] = [sig]

        # Process the body lines (skip first and last which are { / })
        body_lines = block[1:-1] if len(block) > 2 else []
        # Handle the case where the opening brace is on the first line
        first_stripped = block[0].strip()
        brace_idx = first_stripped.rfind("{")
        after_brace = first_stripped[brace_idx + 1:].strip() if brace_idx >= 0 else ""
        if after_brace:
            body_lines = [after_brace] + body_lines

        # Check last line for content before closing brace
        if block:
            last = block[-1].strip()
            if last != "}":
                before_brace = last[:last.rfind("}")].strip() if "}" in last else last
                if before_brace:
                    body_lines.append(before_brace)

        for bl in body_lines:
            stripped = bl.strip()
            if not stripped:
                result.append("")
                continue
            # do-while detection
            dw_m = re.match(r"}\s*while\s*\(\s*(.+?)\s*\)\s*;", stripped)
            if dw_m:
                cond = self.et.translate_expr(dw_m.group(1))
                result.append(f"    if !({cond}) {{ break; }}")
                result.append("}")
                continue

            translated = self.stmt.translate_stmt(stripped)
            # Add braces for blocks
            if stripped.endswith("{"):
                result.append(f"    {translated} {{")
            elif stripped == "}":
                result.append("}")
            else:
                for tl in translated.split("\n"):
                    result.append(f"    {tl}")

        result.append("}")
        result.append("")
        return result

    def _translate_prototype(self, m: re.Match) -> str:
        ret_type_c = m.group(1).strip()
        fn_name = m.group(2)
        params_c = m.group(3).strip()
        ret_type = self.tt.translate_type(ret_type_c)
        params_rust = self._translate_params(params_c)
        sig = f"pub fn {fn_name}({params_rust})"
        if ret_type and ret_type != "()" and ret_type != "void":
            sig += f" -> {ret_type}"
        return sig + ";"

    def _translate_params(self, params_c: str) -> str:
        params_c = params_c.strip()
        if not params_c or params_c == "void":
            return ""
        parts = StmtTranslator._split_args(params_c)
        rust_params: List[str] = []
        for part in parts:
            part = part.strip()
            if part == "...":
                rust_params.append("/* variadic */")
                continue
            # type name
            pm = re.match(r"([\w\s\*]+?)\s+(\w+)(\[\])?$", part)
            if pm:
                c_type = pm.group(1).strip()
                p_name = pm.group(2)
                if pm.group(3):
                    c_type += "[]"
                rust_t = self.tt.translate_type(c_type)
                rust_params.append(f"{p_name}: {rust_t}")
            else:
                # Just a type, no name
                rust_t = self.tt.translate_type(part)
                rust_params.append(f"_: {rust_t}")
        return ", ".join(rust_params)


# ---------------------------------------------------------------------------
# Convenience wrapper
# ---------------------------------------------------------------------------

def translate_c_to_rust(c_source: str) -> str:
    """Translate a C source code string to Rust source code."""
    return CToRustTranslator().translate(c_source)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python c_to_rust_translator.py <input.c> [output.rs]", file=sys.stderr)
        sys.exit(1)

    input_path = sys.argv[1]
    output_path = sys.argv[2] if len(sys.argv) > 2 else None

    with open(input_path, "r") as fh:
        c_code = fh.read()

    rust_code = translate_c_to_rust(c_code)

    if output_path:
        with open(output_path, "w") as fh:
            fh.write(rust_code)
        print(f"Wrote {output_path}", file=sys.stderr)
    else:
        print(rust_code)
