"""Project-level discovery of C and Rust function pairs from build systems.

Reads Cargo.toml and compile_commands.json to auto-discover FFI boundary
functions and match them by name for batch verification.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional


@dataclass
class DiscoveredFunction:
    """A function discovered from project source files."""
    name: str
    file_path: str
    line_number: int
    language: str  # "c" or "rust"
    is_ffi: bool = False
    signature: str = ""


@dataclass
class DiscoveryResult:
    """Result of scanning a project for function pairs."""
    rust_functions: List[DiscoveredFunction] = field(default_factory=list)
    c_functions: List[DiscoveredFunction] = field(default_factory=list)
    matched_pairs: List[tuple] = field(default_factory=list)  # (c_func, rust_func)


# Regex for Rust FFI functions: #[no_mangle] or extern "C"
_NO_MANGLE_RE = re.compile(
    r'#\[no_mangle\]\s*(?:pub\s+)?(?:unsafe\s+)?(?:extern\s+"C"\s+)?fn\s+(\w+)',
    re.MULTILINE,
)
_EXTERN_C_RE = re.compile(
    r'(?:pub\s+)?(?:unsafe\s+)?extern\s+"C"\s+fn\s+(\w+)',
    re.MULTILINE,
)

# Regex for C function definitions
_C_FUNC_RE = re.compile(
    r"(?:^|\n)\s*(?:static\s+|inline\s+|extern\s+)*"
    r"(?:unsigned\s+|signed\s+|const\s+)*"
    r"(?:void|int|long|short|char|float|double|size_t|ssize_t|"
    r"uint\d+_t|int\d+_t|bool)\s*\*?\s+"
    r"(\w+)\s*\([^)]*\)\s*\{",
    re.MULTILINE,
)


def scan_cargo_dir(cargo_dir: str) -> List[DiscoveredFunction]:
    """Read Cargo.toml to find .rs source files, then scan for FFI functions."""
    cargo_toml = os.path.join(cargo_dir, "Cargo.toml")
    if not os.path.isfile(cargo_toml):
        raise FileNotFoundError(f"Cargo.toml not found in {cargo_dir}")

    # Collect all .rs files under src/
    src_dir = os.path.join(cargo_dir, "src")
    rs_files: List[str] = []
    if os.path.isdir(src_dir):
        for root, _dirs, files in os.walk(src_dir):
            for f in files:
                if f.endswith(".rs"):
                    rs_files.append(os.path.join(root, f))

    functions: List[DiscoveredFunction] = []
    for rs_path in sorted(rs_files):
        with open(rs_path) as fh:
            source = fh.read()
        lines = source.split("\n")

        # Find #[no_mangle] functions
        seen: set = set()
        for m in _NO_MANGLE_RE.finditer(source):
            name = m.group(1)
            if name not in seen:
                seen.add(name)
                line_no = source[: m.start()].count("\n") + 1
                functions.append(DiscoveredFunction(
                    name=name, file_path=rs_path, line_number=line_no,
                    language="rust", is_ffi=True,
                ))

        # Find extern "C" functions not already captured
        for m in _EXTERN_C_RE.finditer(source):
            name = m.group(1)
            if name not in seen:
                seen.add(name)
                line_no = source[: m.start()].count("\n") + 1
                functions.append(DiscoveredFunction(
                    name=name, file_path=rs_path, line_number=line_no,
                    language="rust", is_ffi=True,
                ))

    return functions


def scan_compile_commands(compile_commands_path: str) -> List[DiscoveredFunction]:
    """Read compile_commands.json and extract functions from referenced .c files."""
    if not os.path.isfile(compile_commands_path):
        raise FileNotFoundError(f"compile_commands.json not found: {compile_commands_path}")

    with open(compile_commands_path) as fh:
        entries = json.load(fh)

    c_files: List[str] = []
    for entry in entries:
        filepath = entry.get("file", "")
        if filepath.endswith(".c"):
            # Resolve relative paths against the directory field
            if not os.path.isabs(filepath):
                directory = entry.get("directory", os.path.dirname(compile_commands_path))
                filepath = os.path.normpath(os.path.join(directory, filepath))
            if filepath not in c_files:
                c_files.append(filepath)

    return _scan_c_files(c_files)


def scan_c_directory(c_dir: str) -> List[DiscoveredFunction]:
    """Scan a directory for .c files and extract function definitions."""
    c_files: List[str] = []
    for root, _dirs, files in os.walk(c_dir):
        for f in files:
            if f.endswith(".c"):
                c_files.append(os.path.join(root, f))
    return _scan_c_files(sorted(c_files))


def _scan_c_files(c_files: List[str]) -> List[DiscoveredFunction]:
    """Extract function definitions from a list of C files."""
    functions: List[DiscoveredFunction] = []
    for c_path in c_files:
        if not os.path.isfile(c_path):
            continue
        with open(c_path) as fh:
            source = fh.read()
        seen: set = set()
        for m in _C_FUNC_RE.finditer(source):
            name = m.group(1)
            if name not in seen:
                seen.add(name)
                line_no = source[: m.start()].count("\n") + 1
                functions.append(DiscoveredFunction(
                    name=name, file_path=c_path, line_number=line_no,
                    language="c",
                ))
    return functions


def discover_matches(
    rust_functions: List[DiscoveredFunction],
    c_functions: List[DiscoveredFunction],
) -> DiscoveryResult:
    """Match Rust FFI functions to C functions by name."""
    rust_by_name = {f.name: f for f in rust_functions}
    c_by_name = {f.name: f for f in c_functions}

    matched = []
    for name in sorted(rust_by_name.keys() & c_by_name.keys()):
        matched.append((c_by_name[name], rust_by_name[name]))

    return DiscoveryResult(
        rust_functions=rust_functions,
        c_functions=c_functions,
        matched_pairs=matched,
    )


def format_discovery_result(result: DiscoveryResult) -> str:
    """Format discovery result as human-readable text."""
    lines = []

    if result.rust_functions:
        lines.append(f"Rust FFI functions ({len(result.rust_functions)}):")
        for f in result.rust_functions:
            lines.append(f"  {f.name}  ({f.file_path}:{f.line_number})")

    if result.c_functions:
        lines.append(f"\nC functions ({len(result.c_functions)}):")
        for f in result.c_functions:
            lines.append(f"  {f.name}  ({f.file_path}:{f.line_number})")

    if result.matched_pairs:
        lines.append(f"\nMatched pairs ({len(result.matched_pairs)}):")
        for c_fn, rs_fn in result.matched_pairs:
            lines.append(f"  {c_fn.name}: {c_fn.file_path} <-> {rs_fn.file_path}")
    else:
        lines.append("\nNo matched pairs found.")

    return "\n".join(lines)
