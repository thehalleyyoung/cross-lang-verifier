"""Integration with c2rust output for verifying unsafe blocks.

Parses c2rust-generated Rust, identifies unsafe blocks, verifies each
against original C, and suggests safe Rust replacements for
verified-equivalent unsafe code.
"""

import re
import os
import json
import time
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Set
from pathlib import Path

from .api import verify_equivalence, VerificationResult, Divergence


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class UnsafeBlock:
    code: str
    file_path: str
    line_start: int
    line_end: int
    enclosing_function: str
    reason: str  # "raw_pointer", "ffi", "union", "static_mut", "asm", "unknown"
    raw_pointer_ops: List[str] = field(default_factory=list)
    ffi_calls: List[str] = field(default_factory=list)

    @property
    def id(self) -> str:
        return f"{self.file_path}:{self.line_start}-{self.line_end}"


@dataclass
class SafeReplacement:
    original: UnsafeBlock
    safe_code: str
    replacement_strategy: str  # "checked_arithmetic", "slice_indexing", "option_unwrap", etc.
    confidence: float
    explanation: str


@dataclass
class UnsafeVerification:
    block: UnsafeBlock
    c_source: Optional[str]
    verification: Optional[VerificationResult]
    safe_replacement: Optional[SafeReplacement]
    is_proven_safe: bool = False
    notes: str = ""


@dataclass
class C2RustReport:
    total_unsafe_blocks: int
    verified: int
    proven_safe: int
    safe_replacements_available: int
    remaining_unsafe: int
    blocks: List[UnsafeVerification]
    duration_ms: float = 0.0


# ---------------------------------------------------------------------------
# Unsafe block extraction
# ---------------------------------------------------------------------------

_UNSAFE_BLOCK_RE = re.compile(
    r"unsafe\s*\{", re.MULTILINE
)

_RAW_PTR_OPS = re.compile(
    r"\*(?:mut|const)\s+\w+|\.offset\(|\.add\(|\.as_ptr\(|\.as_mut_ptr\("
)

_FFI_CALL_RE = re.compile(
    r"(?:libc|ffi|extern)::\w+|(?:std::ffi::\w+)"
)

_UNSAFE_FN_RE = re.compile(
    r"pub\s+unsafe\s+(?:extern\s+\"C\"\s+)?fn\s+(\w+)"
)


def _find_matching_brace(source: str, start: int) -> int:
    depth = 0
    i = start
    while i < len(source):
        if source[i] == "{":
            depth += 1
        elif source[i] == "}":
            depth -= 1
            if depth == 0:
                return i
        i += 1
    return len(source) - 1


def _classify_unsafe_reason(code: str) -> Tuple[str, List[str], List[str]]:
    raw_ops = _RAW_PTR_OPS.findall(code)
    ffi_calls = _FFI_CALL_RE.findall(code)
    if raw_ops:
        return "raw_pointer", raw_ops, ffi_calls
    if ffi_calls:
        return "ffi", raw_ops, ffi_calls
    if "union" in code:
        return "union", raw_ops, ffi_calls
    if "static mut" in code:
        return "static_mut", raw_ops, ffi_calls
    if "asm!" in code or "llvm_asm!" in code:
        return "asm", raw_ops, ffi_calls
    return "unknown", raw_ops, ffi_calls


def _find_enclosing_fn(source: str, pos: int) -> str:
    """Find the function name enclosing position *pos*."""
    fn_re = re.compile(r"(?:pub\s+)?(?:unsafe\s+)?fn\s+(\w+)")
    best_name = "<top-level>"
    best_pos = -1
    for m in fn_re.finditer(source):
        if m.start() < pos and m.start() > best_pos:
            best_name = m.group(1)
            best_pos = m.start()
    return best_name


def extract_unsafe_blocks(source: str, file_path: str = "<inline>") -> List[UnsafeBlock]:
    blocks: List[UnsafeBlock] = []
    for m in _UNSAFE_BLOCK_RE.finditer(source):
        brace_start = m.end() - 1
        brace_end = _find_matching_brace(source, brace_start)
        code = source[m.start() : brace_end + 1]
        line_start = source[:m.start()].count("\n") + 1
        line_end = source[:brace_end].count("\n") + 1
        reason, raw_ops, ffi_calls = _classify_unsafe_reason(code)
        blocks.append(UnsafeBlock(
            code=code,
            file_path=file_path,
            line_start=line_start,
            line_end=line_end,
            enclosing_function=_find_enclosing_fn(source, m.start()),
            reason=reason,
            raw_pointer_ops=raw_ops,
            ffi_calls=ffi_calls,
        ))
    return blocks


def scan_unsafe_in_project(rust_dir: str) -> List[UnsafeBlock]:
    """Walk a Rust project and extract all unsafe blocks."""
    all_blocks: List[UnsafeBlock] = []
    for root, _, files in os.walk(rust_dir):
        if "target" in root.split(os.sep):
            continue
        for fname in files:
            if not fname.endswith(".rs"):
                continue
            fpath = os.path.join(root, fname)
            with open(fpath) as fh:
                all_blocks.extend(extract_unsafe_blocks(fh.read(), fpath))
    return all_blocks


# ---------------------------------------------------------------------------
# C source mapping — find the original C for a c2rust function
# ---------------------------------------------------------------------------

def _build_c_function_map(c_dir: str) -> Dict[str, Tuple[str, str]]:
    """Map function names to (file_path, source_code)."""
    from .project_scanner import extract_c_functions
    fn_map: Dict[str, Tuple[str, str]] = {}
    for root, _, files in os.walk(c_dir):
        for fname in files:
            if not fname.endswith((".c", ".h")):
                continue
            fpath = os.path.join(root, fname)
            with open(fpath) as fh:
                source = fh.read()
            for fn in extract_c_functions(source, fpath):
                fn_map[fn.name] = (fpath, fn.raw_source)
    return fn_map


def find_c_origin(block: UnsafeBlock, c_function_map: Dict[str, Tuple[str, str]]) -> Optional[str]:
    """Try to find the original C source for an unsafe block's enclosing function."""
    fn_name = block.enclosing_function
    # c2rust often keeps the same name
    if fn_name in c_function_map:
        return c_function_map[fn_name][1]
    # Try without common prefixes/suffixes c2rust adds
    for suffix in ("_0", "_1", "_c", "_impl"):
        stripped = fn_name.rstrip(suffix) if fn_name.endswith(suffix) else None
        if stripped and stripped in c_function_map:
            return c_function_map[stripped][1]
    # Fuzzy: check if any C function name is a substring
    for c_name, (_, src) in c_function_map.items():
        if c_name in fn_name or fn_name in c_name:
            return src
    return None


# ---------------------------------------------------------------------------
# Safe replacement suggestions
# ---------------------------------------------------------------------------

_WRAPPING_ARITH_RE = re.compile(
    r"(\w+)\s*(\+|-|\*)\s*(\w+)"
)

_PTR_DEREF_RE = re.compile(
    r"\*(\w+(?:\.\w+)*)"
)

_PTR_OFFSET_RE = re.compile(
    r"(\w+)\.offset\(([^)]+)\)"
)

_ARRAY_PTR_RE = re.compile(
    r"\*(\w+)\.add\(([^)]+)\)"
)


def suggest_safe_replacement(block: UnsafeBlock) -> Optional[SafeReplacement]:
    """Suggest a safe Rust replacement for an unsafe block if possible."""
    code = block.code
    inner = code[len("unsafe {"):-1].strip() if code.startswith("unsafe") else code

    # Pattern 1: pointer arithmetic → slice indexing
    offset_match = _PTR_OFFSET_RE.search(inner)
    if offset_match and block.reason == "raw_pointer":
        ptr, offset = offset_match.group(1), offset_match.group(2)
        safe = inner.replace(
            offset_match.group(0),
            f"{ptr}_slice[{offset} as usize]"
        )
        safe = safe.replace("unsafe ", "")
        return SafeReplacement(
            original=block,
            safe_code=f"// Requires converting raw pointer to slice first\n{safe}",
            replacement_strategy="slice_indexing",
            confidence=0.7,
            explanation=f"Replace pointer+offset with slice indexing. "
                        f"Ensure `{ptr}` is converted to a slice with known length.",
        )

    # Pattern 2: *ptr.add(n) → slice[n]
    add_match = _ARRAY_PTR_RE.search(inner)
    if add_match:
        ptr, idx = add_match.group(1), add_match.group(2)
        safe = inner.replace(add_match.group(0), f"{ptr}_slice[{idx} as usize]")
        return SafeReplacement(
            original=block,
            safe_code=safe,
            replacement_strategy="slice_indexing",
            confidence=0.7,
            explanation=f"Replace *{ptr}.add({idx}) with bounds-checked slice access.",
        )

    # Pattern 3: wrapping arithmetic — suggest checked/wrapping methods
    arith_matches = list(_WRAPPING_ARITH_RE.finditer(inner))
    if arith_matches and "wrapping" not in inner:
        safe = inner
        for am in arith_matches:
            a, op, b = am.group(1), am.group(2), am.group(3)
            method = {"+": "checked_add", "-": "checked_sub", "*": "checked_mul"}.get(op)
            if method:
                safe = safe.replace(am.group(0), f"{a}.{method}({b}).unwrap()")
        return SafeReplacement(
            original=block,
            safe_code=safe,
            replacement_strategy="checked_arithmetic",
            confidence=0.6,
            explanation="Replace raw arithmetic with checked methods that panic on overflow.",
        )

    # Pattern 4: simple dereference — suggest Option/Result
    deref_matches = _PTR_DEREF_RE.findall(inner)
    if deref_matches and block.reason == "raw_pointer":
        safe = inner
        for ptr_expr in deref_matches:
            safe = safe.replace(f"*{ptr_expr}", f"{ptr_expr}.as_ref().unwrap()")
        return SafeReplacement(
            original=block,
            safe_code=safe,
            replacement_strategy="option_unwrap",
            confidence=0.5,
            explanation="Replace raw pointer dereference with as_ref().unwrap(). "
                        "Consider using Option<&T> in the function signature.",
        )

    return None


# ---------------------------------------------------------------------------
# Main verification pipeline
# ---------------------------------------------------------------------------

def verify_c2rust_output(c_dir: str, rust_dir: str,
                         timeout_s: float = 120.0,
                         method: str = "hybrid") -> C2RustReport:
    """Verify c2rust-generated Rust against original C sources.

    For each unsafe block in the Rust output:
    1. Find the original C function
    2. Verify equivalence
    3. Suggest safe replacement if equivalence is proven
    """
    start = time.time()

    unsafe_blocks = scan_unsafe_in_project(rust_dir)
    c_fn_map = _build_c_function_map(c_dir)

    verifications: List[UnsafeVerification] = []
    proven_safe = 0
    replacements_available = 0
    verified_count = 0

    for block in unsafe_blocks:
        c_source = find_c_origin(block, c_fn_map)
        uv = UnsafeVerification(block=block, c_source=c_source,
                                verification=None, safe_replacement=None)

        if c_source is None:
            uv.notes = "No matching C source found"
            verifications.append(uv)
            continue

        try:
            result = verify_equivalence(c_source, block.code,
                                        timeout_s=timeout_s, method=method)
            uv.verification = result
            verified_count += 1

            if result.equivalent:
                uv.is_proven_safe = True
                proven_safe += 1
                replacement = suggest_safe_replacement(block)
                if replacement:
                    uv.safe_replacement = replacement
                    replacements_available += 1
        except Exception as exc:
            uv.notes = f"Verification error: {exc}"

        verifications.append(uv)

    return C2RustReport(
        total_unsafe_blocks=len(unsafe_blocks),
        verified=verified_count,
        proven_safe=proven_safe,
        safe_replacements_available=replacements_available,
        remaining_unsafe=len(unsafe_blocks) - proven_safe,
        blocks=verifications,
        duration_ms=(time.time() - start) * 1000,
    )


# ---------------------------------------------------------------------------
# Tracking which unsafe blocks are proven safe to remove
# ---------------------------------------------------------------------------

class UnsafeSafetyTracker:
    """Track which unsafe blocks have been verified safe and can be replaced."""

    def __init__(self, state_file: str = ".xequiv-unsafe-tracker.json"):
        self.state_file = state_file
        self._entries: Dict[str, dict] = {}
        self._load()

    def _load(self) -> None:
        if os.path.exists(self.state_file):
            with open(self.state_file) as f:
                self._entries = json.load(f)

    def _save(self) -> None:
        with open(self.state_file, "w") as f:
            json.dump(self._entries, f, indent=2)

    def mark_safe(self, block_id: str, replacement: Optional[SafeReplacement] = None) -> None:
        self._entries[block_id] = {
            "status": "proven_safe",
            "timestamp": time.time(),
            "replacement": replacement.safe_code if replacement else None,
            "strategy": replacement.replacement_strategy if replacement else None,
        }
        self._save()

    def mark_unsafe(self, block_id: str, reason: str = "") -> None:
        self._entries[block_id] = {
            "status": "unsafe",
            "timestamp": time.time(),
            "reason": reason,
        }
        self._save()

    def safe_blocks(self) -> List[str]:
        return [bid for bid, e in self._entries.items() if e["status"] == "proven_safe"]

    def unsafe_blocks(self) -> List[str]:
        return [bid for bid, e in self._entries.items() if e["status"] == "unsafe"]

    def summary(self) -> Dict[str, int]:
        safe = sum(1 for e in self._entries.values() if e["status"] == "proven_safe")
        unsafe = sum(1 for e in self._entries.values() if e["status"] == "unsafe")
        return {"total": len(self._entries), "proven_safe": safe, "unsafe": unsafe}

    def apply_replacements(self, rust_dir: str, dry_run: bool = True) -> List[Tuple[str, str, str]]:
        """Apply safe replacements to Rust source files.

        Returns list of (file_path, original_code, replacement_code) tuples.
        """
        changes: List[Tuple[str, str, str]] = []
        for block_id, entry in self._entries.items():
            if entry["status"] != "proven_safe" or not entry.get("replacement"):
                continue
            # Parse block_id: "file:start-end"
            parts = block_id.rsplit(":", 1)
            if len(parts) != 2:
                continue
            file_path = parts[0]
            line_range = parts[1]
            try:
                start_line, end_line = map(int, line_range.split("-"))
            except ValueError:
                continue
            if not os.path.exists(file_path):
                continue
            with open(file_path) as f:
                lines = f.readlines()
            if start_line < 1 or end_line > len(lines):
                continue
            original = "".join(lines[start_line - 1 : end_line])
            replacement = entry["replacement"]
            changes.append((file_path, original, replacement))
            if not dry_run:
                lines[start_line - 1 : end_line] = [replacement + "\n"]
                with open(file_path, "w") as f:
                    f.writelines(lines)
        return changes


def generate_unsafe_report(report: C2RustReport, format: str = "markdown") -> str:
    """Generate a human-readable report from c2rust verification results."""
    if format == "markdown":
        lines = [
            "# c2rust Unsafe Block Verification Report\n",
            f"| Metric | Count |",
            f"|--------|-------|",
            f"| Total unsafe blocks | {report.total_unsafe_blocks} |",
            f"| Verified | {report.verified} |",
            f"| Proven safe to remove | {report.proven_safe} |",
            f"| Safe replacements available | {report.safe_replacements_available} |",
            f"| Remaining unsafe | {report.remaining_unsafe} |",
            f"| Verification time | {report.duration_ms:.0f}ms |\n",
            "## Block Details\n",
        ]
        for uv in report.blocks:
            icon = "✅" if uv.is_proven_safe else "❌" if uv.verification else "⚠️"
            lines.append(f"### {icon} `{uv.block.enclosing_function}` ({uv.block.file_path}:{uv.block.line_start})\n")
            lines.append(f"- **Reason**: {uv.block.reason}")
            if uv.verification:
                lines.append(f"- **Equivalent**: {uv.verification.equivalent}")
                if uv.verification.divergences:
                    for d in uv.verification.divergences:
                        lines.append(f"  - [{d.severity}] {d.category}: {d.description}")
            if uv.safe_replacement:
                lines.append(f"- **Suggested safe replacement** ({uv.safe_replacement.replacement_strategy}):")
                lines.append(f"  ```rust\n  {uv.safe_replacement.safe_code}\n  ```")
                lines.append(f"  {uv.safe_replacement.explanation}")
            if uv.notes:
                lines.append(f"- **Note**: {uv.notes}")
            lines.append("")
        return "\n".join(lines)
    else:
        lines = [
            f"c2rust Unsafe Verification: {report.proven_safe}/{report.total_unsafe_blocks} proven safe",
            f"  Replacements available: {report.safe_replacements_available}",
            f"  Remaining unsafe: {report.remaining_unsafe}",
        ]
        return "\n".join(lines)
