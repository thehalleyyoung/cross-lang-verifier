"""
Test harness generation for differential fuzzing.

Given C and Rust functions, generates FFI test harnesses that call both
with identical inputs and compare outputs. Handles calling conventions,
struct layout differences, and error handling.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import List, Optional, Dict, Tuple, Any

from ..ir.types import (
    IRType, IntType, FloatType, PointerType, VoidType,
    ArrayType, StructType, Signedness, FloatKind,
)
from ..ir.function import Function


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

class CallingConvention(Enum):
    C = auto()
    RUST = auto()
    CDECL = auto()
    STDCALL = auto()


@dataclass
class HarnessConfig:
    """Configuration for test harness generation."""
    output_dir: str = "harness_output"
    c_compiler: str = "gcc"
    c_flags: List[str] = field(default_factory=lambda: ["-Wall", "-Werror", "-O0", "-g"])
    rust_edition: str = "2021"
    compare_outputs: bool = True
    handle_panics: bool = True
    timeout_seconds: int = 5
    max_test_cases: int = 1000
    c_calling_convention: CallingConvention = CallingConvention.C
    rust_calling_convention: CallingConvention = CallingConvention.RUST


# ---------------------------------------------------------------------------
# Test harness
# ---------------------------------------------------------------------------

@dataclass
class TestHarness:
    """A complete test harness for differential testing."""
    name: str
    c_source: str
    rust_source: str
    c_header: str = ""
    build_script: str = ""
    makefile: str = ""
    c_function_name: str = ""
    rust_function_name: str = ""
    parameter_types: List[Tuple[str, str, str]] = field(default_factory=list)  # (name, c_type, rust_type)
    return_type_c: str = "int"
    return_type_rust: str = "i32"

    def write_to_dir(self, directory: str) -> List[str]:
        """Write all harness files to a directory."""
        os.makedirs(directory, exist_ok=True)
        files: List[str] = []

        c_path = os.path.join(directory, f"{self.name}.c")
        with open(c_path, "w") as f:
            f.write(self.c_source)
        files.append(c_path)

        rust_path = os.path.join(directory, f"{self.name}.rs")
        with open(rust_path, "w") as f:
            f.write(self.rust_source)
        files.append(rust_path)

        if self.c_header:
            h_path = os.path.join(directory, f"{self.name}.h")
            with open(h_path, "w") as f:
                f.write(self.c_header)
            files.append(h_path)

        if self.build_script:
            build_path = os.path.join(directory, "build.sh")
            with open(build_path, "w") as f:
                f.write(self.build_script)
            os.chmod(build_path, 0o755)
            files.append(build_path)

        if self.makefile:
            make_path = os.path.join(directory, "Makefile")
            with open(make_path, "w") as f:
                f.write(self.makefile)
            files.append(make_path)

        return files


# ---------------------------------------------------------------------------
# Type mapping helpers
# ---------------------------------------------------------------------------

def _ir_type_to_c(ty: IRType) -> str:
    """Convert IR type to C type string."""
    if isinstance(ty, IntType):
        if ty.width == 1:
            return "_Bool"
        prefix = "" if ty.signed else "u"
        return f"{prefix}int{ty.width}_t"
    if isinstance(ty, FloatType):
        return "float" if ty.kind == FloatKind.F32 else "double"
    if isinstance(ty, PointerType):
        inner = _ir_type_to_c(ty.pointee) if ty.pointee else "void"
        return f"{inner}*"
    if isinstance(ty, VoidType):
        return "void"
    if isinstance(ty, ArrayType):
        inner = _ir_type_to_c(ty.element_type) if ty.element_type else "int"
        return f"{inner}*"
    if isinstance(ty, StructType):
        return f"struct {ty.name}" if hasattr(ty, 'name') and ty.name else "void*"
    return "int"


def _ir_type_to_rust(ty: IRType) -> str:
    """Convert IR type to Rust type string."""
    if isinstance(ty, IntType):
        if ty.width == 1:
            return "bool"
        prefix = "i" if ty.signed else "u"
        return f"{prefix}{ty.width}"
    if isinstance(ty, FloatType):
        return "f32" if ty.kind == FloatKind.F32 else "f64"
    if isinstance(ty, PointerType):
        inner = _ir_type_to_rust(ty.pointee) if ty.pointee else "u8"
        return f"*const {inner}"
    if isinstance(ty, VoidType):
        return "()"
    if isinstance(ty, ArrayType):
        inner = _ir_type_to_rust(ty.element_type) if ty.element_type else "i32"
        return f"*const {inner}"
    return "i32"


def _ir_type_to_c_ffi(ty: IRType) -> str:
    """Convert IR type to C FFI compatible type."""
    if isinstance(ty, IntType):
        if ty.width <= 8:
            return "c_char" if ty.signed else "c_uchar"
        if ty.width <= 16:
            return "c_short" if ty.signed else "c_ushort"
        if ty.width <= 32:
            return "c_int" if ty.signed else "c_uint"
        return "c_longlong" if ty.signed else "c_ulonglong"
    if isinstance(ty, FloatType):
        return "c_float" if ty.kind == FloatKind.F32 else "c_double"
    if isinstance(ty, PointerType):
        return "*const c_void"
    return "c_int"


def _format_specifier(ty: IRType) -> str:
    """Get printf format specifier for a type."""
    if isinstance(ty, IntType):
        if ty.width <= 32:
            return "%d" if ty.signed else "%u"
        return "%lld" if ty.signed else "%llu"
    if isinstance(ty, FloatType):
        return "%.17g"
    if isinstance(ty, PointerType):
        return "%p"
    return "%d"


def _random_value_c(ty: IRType, var_name: str) -> str:
    """Generate C code for a random value of the given type."""
    if isinstance(ty, IntType):
        return f"rand()"
    if isinstance(ty, FloatType):
        return f"((double)rand() / RAND_MAX * 2.0 - 1.0)"
    if isinstance(ty, PointerType):
        return "NULL"
    return "0"


# ---------------------------------------------------------------------------
# Harness generator
# ---------------------------------------------------------------------------

class HarnessGenerator:
    """
    Generates differential test harnesses for C and Rust functions.
    """

    def __init__(self, config: Optional[HarnessConfig] = None):
        self.config = config or HarnessConfig()

    def generate(
        self,
        c_func: Function,
        rust_func: Function,
        harness_name: Optional[str] = None,
    ) -> TestHarness:
        """Generate a complete test harness."""
        name = harness_name or f"harness_{c_func.name}_{rust_func.name}"

        # Map parameter types
        c_args = list(c_func.arguments)
        rust_args = list(rust_func.arguments)
        min_args = min(len(c_args), len(rust_args))

        param_types: List[Tuple[str, str, str]] = []
        for i in range(min_args):
            ca = c_args[i]
            ra = rust_args[i]
            arg_name = ca.name or ra.name or f"arg_{i}"
            c_type = _ir_type_to_c(ca.type) if ca.type else "int"
            rust_type = _ir_type_to_rust(ra.type) if ra.type else "i32"
            param_types.append((arg_name, c_type, rust_type))

        ret_c = _ir_type_to_c(c_func.return_type) if c_func.return_type else "int"
        ret_rust = _ir_type_to_rust(rust_func.return_type) if rust_func.return_type else "i32"

        c_source = self._generate_c_harness(
            name, c_func.name, param_types, ret_c, c_args,
        )
        rust_source = self._generate_rust_harness(
            name, rust_func.name, c_func.name, param_types, ret_c, ret_rust, c_args,
        )
        c_header = self._generate_c_header(c_func.name, param_types, ret_c)
        build_script = self._generate_build_script(name, c_func.name)
        makefile = self._generate_makefile(name, c_func.name)

        return TestHarness(
            name=name,
            c_source=c_source,
            rust_source=rust_source,
            c_header=c_header,
            build_script=build_script,
            makefile=makefile,
            c_function_name=c_func.name,
            rust_function_name=rust_func.name,
            parameter_types=param_types,
            return_type_c=ret_c,
            return_type_rust=ret_rust,
        )

    def _generate_c_harness(
        self,
        name: str,
        c_func_name: str,
        params: List[Tuple[str, str, str]],
        return_type: str,
        c_args: list,
    ) -> str:
        """Generate C harness source."""
        lines = [
            f"/* Differential test harness: {name} */",
            f"/* Auto-generated by Cross-Language Equivalence Verifier */",
            "",
            "#include <stdio.h>",
            "#include <stdlib.h>",
            "#include <stdint.h>",
            "#include <string.h>",
            "#include <math.h>",
            "#include <signal.h>",
            "#include <setjmp.h>",
            "",
            f'#include "{name}.h"',
            "",
            "static jmp_buf jmp_env;",
            "static volatile int got_signal = 0;",
            "",
            "static void signal_handler(int sig) {",
            "    got_signal = sig;",
            "    longjmp(jmp_env, 1);",
            "}",
            "",
        ]

        # Harness function
        param_decls = ", ".join(f"{c_type} {pname}" for pname, c_type, _ in params)
        if not param_decls:
            param_decls = "void"

        lines.extend([
            f"int harness_run({param_decls}) {{",
            "    signal(SIGSEGV, signal_handler);",
            "    signal(SIGFPE, signal_handler);",
            "    signal(SIGABRT, signal_handler);",
            "",
            "    got_signal = 0;",
            "    if (setjmp(jmp_env) == 0) {",
        ])

        # Call the function
        args_str = ", ".join(pname for pname, _, _ in params)
        if return_type != "void":
            lines.append(f"        {return_type} result = {c_func_name}({args_str});")
            # Print result for comparison
            fmt = _format_specifier(c_args[0].type if c_args else IntType(32, Signedness.SIGNED))
            lines.append(f'        printf("C_RESULT: {fmt}\\n", result);')
            lines.append("        return 0;")
        else:
            lines.append(f"        {c_func_name}({args_str});")
            lines.append('        printf("C_RESULT: void\\n");')
            lines.append("        return 0;")

        lines.extend([
            "    } else {",
            '        printf("C_SIGNAL: %d\\n", got_signal);',
            "        return got_signal;",
            "    }",
            "}",
            "",
        ])

        # Main function for standalone testing
        lines.extend([
            "int main(int argc, char** argv) {",
            "    srand(42);",
            "",
        ])

        # Generate test loop
        lines.append(f"    for (int i = 0; i < {self.config.max_test_cases}; i++) {{")
        for pname, c_type, _ in params:
            lines.append(f"        {c_type} {pname} = ({c_type})rand();")
        args_str = ", ".join(pname for pname, _, _ in params)
        lines.append(f"        harness_run({args_str});")
        lines.append("    }")
        lines.append("")
        lines.append("    return 0;")
        lines.append("}")

        return "\n".join(lines)

    def _generate_rust_harness(
        self,
        name: str,
        rust_func_name: str,
        c_func_name: str,
        params: List[Tuple[str, str, str]],
        ret_c: str,
        ret_rust: str,
        c_args: list,
    ) -> str:
        """Generate Rust harness source."""
        lines = [
            f"//! Differential test harness: {name}",
            f"//! Auto-generated by Cross-Language Equivalence Verifier",
            "",
            "use std::ffi::*;",
            "use std::panic;",
            "",
            "extern \"C\" {",
        ]

        # Declare the C function
        c_param_decls = ", ".join(
            f"{pname}: {_ir_type_to_c_ffi(c_args[i].type) if i < len(c_args) and c_args[i].type else 'c_int'}"
            for i, (pname, _, _) in enumerate(params)
        )
        c_ret_ffi = _ir_type_to_c_ffi(c_args[0].type) if c_args and c_args[0].type else "c_int"
        if ret_c == "void":
            lines.append(f"    fn {c_func_name}({c_param_decls});")
        else:
            lines.append(f"    fn {c_func_name}({c_param_decls}) -> {c_ret_ffi};")

        lines.extend([
            "}",
            "",
        ])

        # Rust wrapper that calls both
        rust_param_decls = ", ".join(f"{pname}: {rust_type}" for pname, _, rust_type in params)
        lines.extend([
            f"fn run_comparison({rust_param_decls}) -> bool {{",
        ])

        # Call C function
        args_str = ", ".join(pname for pname, _, _ in params)
        if ret_c != "void":
            lines.append(f"    let c_result = unsafe {{ {c_func_name}({args_str}) }};")
        else:
            lines.append(f"    unsafe {{ {c_func_name}({args_str}) }};")

        # Call Rust function with panic catching
        lines.extend([
            f"    let rust_result = panic::catch_unwind(|| {{",
            f"        {rust_func_name}({args_str})",
            f"    }});",
            "",
        ])

        if ret_c != "void":
            lines.extend([
                "    match rust_result {",
                "        Ok(r) => {",
                "            let c_val = c_result as i64;",
                "            let r_val = r as i64;",
                "            if c_val != r_val {",
                '                eprintln!("DIVERGENCE: C={}, Rust={}", c_val, r_val);',
                "                return false;",
                "            }",
                "            true",
                "        }",
                "        Err(e) => {",
                '            eprintln!("RUST_PANIC: {:?}", e);',
                "            false",
                "        }",
                "    }",
            ])
        else:
            lines.extend([
                "    match rust_result {",
                "        Ok(_) => true,",
                "        Err(e) => {",
                '            eprintln!("RUST_PANIC: {:?}", e);',
                "            false",
                "        }",
                "    }",
            ])

        lines.extend([
            "}",
            "",
            "fn main() {",
            "    let mut divergences = 0u64;",
            f"    for _ in 0..{self.config.max_test_cases} {{",
        ])

        # Generate random inputs
        for pname, _, rust_type in params:
            if "i32" in rust_type or "i64" in rust_type:
                lines.append(f"        let {pname}: {rust_type} = rand::random();")
            elif "u32" in rust_type or "u64" in rust_type:
                lines.append(f"        let {pname}: {rust_type} = rand::random();")
            elif "f32" in rust_type or "f64" in rust_type:
                lines.append(f"        let {pname}: {rust_type} = rand::random();")
            else:
                lines.append(f"        let {pname}: {rust_type} = Default::default();")

        args_str = ", ".join(pname for pname, _, _ in params)
        lines.extend([
            f"        if !run_comparison({args_str}) {{",
            "            divergences += 1;",
            "        }",
            "    }",
            '    println!("Divergences: {}", divergences);',
            "}",
        ])

        return "\n".join(lines)

    def _generate_c_header(
        self,
        c_func_name: str,
        params: List[Tuple[str, str, str]],
        return_type: str,
    ) -> str:
        """Generate C header file."""
        guard = f"__{c_func_name.upper()}_H__"
        lines = [
            f"#ifndef {guard}",
            f"#define {guard}",
            "",
            "#include <stdint.h>",
            "",
        ]

        param_decls = ", ".join(f"{c_type} {pname}" for pname, c_type, _ in params)
        if not param_decls:
            param_decls = "void"
        lines.append(f"{return_type} {c_func_name}({param_decls});")

        lines.extend([
            "",
            f"#endif /* {guard} */",
        ])
        return "\n".join(lines)

    def _generate_build_script(self, name: str, c_func_name: str) -> str:
        """Generate a build script."""
        c_flags = " ".join(self.config.c_flags)
        return f"""#!/bin/bash
# Build script for differential test harness: {name}
set -e

echo "Building C harness..."
{self.config.c_compiler} {c_flags} -c {name}.c -o {name}_c.o

echo "Building Rust harness..."
rustc --edition {self.config.rust_edition} {name}.rs -o {name}_rust

echo "Linking..."
{self.config.c_compiler} {c_flags} {name}_c.o -o {name}_c_test

echo "Build complete."
echo "Run ./{name}_c_test for C-only test"
echo "Run ./{name}_rust for Rust differential test"
"""

    def _generate_makefile(self, name: str, c_func_name: str) -> str:
        """Generate a Makefile."""
        c_flags = " ".join(self.config.c_flags)
        return f"""# Makefile for differential test harness: {name}
CC = {self.config.c_compiler}
CFLAGS = {c_flags}
RUSTC = rustc
RUST_EDITION = {self.config.rust_edition}

.PHONY: all clean test

all: {name}_c_test {name}_rust

{name}_c.o: {name}.c {name}.h
\t$(CC) $(CFLAGS) -c $< -o $@

{name}_c_test: {name}_c.o
\t$(CC) $(CFLAGS) $< -o $@ -lm

{name}_rust: {name}.rs
\t$(RUSTC) --edition $(RUST_EDITION) $< -o $@

test: all
\t./{name}_c_test
\t./{name}_rust

clean:
\trm -f {name}_c.o {name}_c_test {name}_rust
"""

    def generate_ffi_bridge(
        self,
        c_func: Function,
        rust_func: Function,
    ) -> str:
        """Generate a Rust FFI bridge for calling the C function."""
        c_args = list(c_func.arguments)
        lines = [
            "// FFI bridge for C function",
            "use std::os::raw::*;",
            "",
            "extern \"C\" {",
        ]

        param_strs = []
        for arg in c_args:
            name = arg.name or f"arg_{arg.index}"
            ffi_type = _ir_type_to_c_ffi(arg.type) if arg.type else "c_int"
            param_strs.append(f"{name}: {ffi_type}")

        params = ", ".join(param_strs)
        ret_type = _ir_type_to_c_ffi(c_func.return_type) if c_func.return_type else "c_int"

        if isinstance(c_func.return_type, VoidType):
            lines.append(f"    pub fn {c_func.name}({params});")
        else:
            lines.append(f"    pub fn {c_func.name}({params}) -> {ret_type};")

        lines.extend([
            "}",
            "",
        ])

        # Generate a safe Rust wrapper
        safe_params = []
        for arg in c_args:
            name = arg.name or f"arg_{arg.index}"
            rust_type = _ir_type_to_rust(arg.type) if arg.type else "i32"
            safe_params.append(f"{name}: {rust_type}")

        safe_param_str = ", ".join(safe_params)
        call_args = ", ".join(
            f"{arg.name or f'arg_{arg.index}'} as _" for arg in c_args
        )

        if isinstance(c_func.return_type, VoidType):
            lines.extend([
                f"pub fn call_c_{c_func.name}({safe_param_str}) {{",
                f"    unsafe {{ {c_func.name}({call_args}) }}",
                "}",
            ])
        else:
            rust_ret = _ir_type_to_rust(c_func.return_type) if c_func.return_type else "i32"
            lines.extend([
                f"pub fn call_c_{c_func.name}({safe_param_str}) -> {rust_ret} {{",
                f"    unsafe {{ {c_func.name}({call_args}) as {rust_ret} }}",
                "}",
            ])

        return "\n".join(lines)
