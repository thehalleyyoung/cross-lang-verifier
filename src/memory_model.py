"""
Memory model verification module for C-to-Rust migration.

Provides regex-based static analysis of C source code to detect common memory
safety issues (leaks, dangling pointers, double-free, use-after-free) and maps
those patterns onto Rust ownership / lifetime / smart-pointer idioms.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Dict, List, Optional, Set, Tuple


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class PointerKind(Enum):
    RAW = auto()
    UNIQUE = auto()
    SHARED = auto()
    WEAK = auto()


class SmartPointerType(Enum):
    BOX = "Box"
    RC = "Rc"
    ARC = "Arc"
    REF_CELL = "RefCell"
    MUTEX = "Mutex"
    COW = "Cow"


class OwnershipKind(Enum):
    OWNED = auto()
    BORROWED = auto()
    MUTABLE_BORROWED = auto()
    SHARED = auto()
    TRANSFERRED = auto()


class LifetimeRelation(Enum):
    OUTLIVES = auto()
    EQUAL = auto()
    SUBSET = auto()
    UNRELATED = auto()


class SeverityLevel(Enum):
    INFO = auto()
    WARNING = auto()
    ERROR = auto()
    CRITICAL = auto()


# ---------------------------------------------------------------------------
# Data classes – allocation tracking
# ---------------------------------------------------------------------------

@dataclass
class AllocationInfo:
    variable: str
    alloc_type: str  # malloc, calloc, realloc, new
    line_number: int
    byte_size: Optional[str] = None
    freed: bool = False
    free_line: Optional[int] = None
    scope_depth: int = 0
    source_expression: str = ""


@dataclass
class AllocationMap:
    allocations: Dict[str, AllocationInfo] = field(default_factory=dict)
    leaks: List[AllocationInfo] = field(default_factory=list)
    total_allocs: int = 0
    total_frees: int = 0
    matched_pairs: List[Tuple[str, int, int]] = field(default_factory=list)

    def leak_count(self) -> int:
        return len(self.leaks)

    def summary(self) -> str:
        return (
            f"Allocations: {self.total_allocs}, Frees: {self.total_frees}, "
            f"Leaks: {self.leak_count()}, Matched pairs: {len(self.matched_pairs)}"
        )


# ---------------------------------------------------------------------------
# Data classes – ownership
# ---------------------------------------------------------------------------

@dataclass
class OwnershipTransfer:
    c_variable: str
    rust_variable: str
    kind: OwnershipKind
    c_line: int
    rust_line: int
    pattern: str = ""
    notes: str = ""


@dataclass
class OwnershipResult:
    transfers: List[OwnershipTransfer] = field(default_factory=list)
    unmapped_c_pointers: List[str] = field(default_factory=list)
    unmapped_rust_owners: List[str] = field(default_factory=list)
    compatible: bool = True
    warnings: List[str] = field(default_factory=list)

    def summary(self) -> str:
        return (
            f"Transfers: {len(self.transfers)}, "
            f"Unmapped C: {len(self.unmapped_c_pointers)}, "
            f"Unmapped Rust: {len(self.unmapped_rust_owners)}, "
            f"Compatible: {self.compatible}"
        )


# ---------------------------------------------------------------------------
# Data classes – safety defects
# ---------------------------------------------------------------------------

@dataclass
class DanglingPointer:
    variable: str
    line_number: int
    scope_ended_line: int
    reason: str
    severity: SeverityLevel = SeverityLevel.ERROR


@dataclass
class DoubleFree:
    variable: str
    first_free_line: int
    second_free_line: int
    severity: SeverityLevel = SeverityLevel.CRITICAL


@dataclass
class UseAfterFree:
    variable: str
    free_line: int
    use_line: int
    use_expression: str
    severity: SeverityLevel = SeverityLevel.CRITICAL


# ---------------------------------------------------------------------------
# Data classes – lifetimes
# ---------------------------------------------------------------------------

@dataclass
class LifetimeAnnotation:
    name: str
    applies_to: str
    relation: LifetimeRelation = LifetimeRelation.UNRELATED
    inferred_from: str = ""


@dataclass
class LifetimeMap:
    annotations: List[LifetimeAnnotation] = field(default_factory=list)
    structs_needing_lifetimes: List[str] = field(default_factory=list)
    functions_needing_lifetimes: List[str] = field(default_factory=list)
    elision_applicable: List[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Data classes – layout / alignment
# ---------------------------------------------------------------------------

@dataclass
class LayoutField:
    name: str
    c_type: str
    rust_type: str
    c_size: int
    rust_size: int
    c_offset: int
    rust_offset: int
    c_alignment: int
    rust_alignment: int


@dataclass
class LayoutDiff:
    fields: List[LayoutField] = field(default_factory=list)
    c_total_size: int = 0
    rust_total_size: int = 0
    compatible: bool = True
    padding_differences: List[str] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)


@dataclass
class AlignmentResult:
    compatible: bool = True
    c_alignments: Dict[str, int] = field(default_factory=dict)
    rust_alignments: Dict[str, int] = field(default_factory=dict)
    mismatches: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Data classes – smart pointer suggestion
# ---------------------------------------------------------------------------

@dataclass
class SmartPointerSuggestion:
    variable: str
    line_number: int
    current_pattern: str
    suggested_type: SmartPointerType
    reason: str
    confidence: float = 0.0


# ---------------------------------------------------------------------------
# Data classes – memory region & overall result
# ---------------------------------------------------------------------------

@dataclass
class MemoryRegion:
    name: str
    start_line: int
    end_line: int
    allocations: List[AllocationInfo] = field(default_factory=list)
    pointer_kind: PointerKind = PointerKind.RAW


@dataclass
class MemorySafetyResult:
    allocation_map: AllocationMap = field(default_factory=AllocationMap)
    ownership_result: OwnershipResult = field(default_factory=OwnershipResult)
    dangling_pointers: List[DanglingPointer] = field(default_factory=list)
    double_frees: List[DoubleFree] = field(default_factory=list)
    use_after_frees: List[UseAfterFree] = field(default_factory=list)
    lifetime_map: LifetimeMap = field(default_factory=LifetimeMap)
    smart_pointer_suggestions: List[SmartPointerSuggestion] = field(
        default_factory=list
    )
    overall_safe: bool = True
    issues: List[str] = field(default_factory=list)
    severity: SeverityLevel = SeverityLevel.INFO

    def summary(self) -> str:
        total_issues = (
            len(self.dangling_pointers)
            + len(self.double_frees)
            + len(self.use_after_frees)
            + self.allocation_map.leak_count()
        )
        return (
            f"Overall safe: {self.overall_safe}, Total issues: {total_issues}, "
            f"Leaks: {self.allocation_map.leak_count()}, "
            f"Dangling: {len(self.dangling_pointers)}, "
            f"Double-free: {len(self.double_frees)}, "
            f"UAF: {len(self.use_after_frees)}"
        )


# ---------------------------------------------------------------------------
# Helpers – C type sizes (LP64 model)
# ---------------------------------------------------------------------------

C_TYPE_SIZES: Dict[str, Tuple[int, int]] = {
    "char": (1, 1),
    "unsigned char": (1, 1),
    "signed char": (1, 1),
    "short": (2, 2),
    "unsigned short": (2, 2),
    "int": (4, 4),
    "unsigned int": (4, 4),
    "unsigned": (4, 4),
    "long": (8, 8),
    "unsigned long": (8, 8),
    "long long": (8, 8),
    "unsigned long long": (8, 8),
    "float": (4, 4),
    "double": (8, 8),
    "long double": (16, 16),
    "size_t": (8, 8),
    "ssize_t": (8, 8),
    "ptrdiff_t": (8, 8),
    "int8_t": (1, 1),
    "int16_t": (2, 2),
    "int32_t": (4, 4),
    "int64_t": (8, 8),
    "uint8_t": (1, 1),
    "uint16_t": (2, 2),
    "uint32_t": (4, 4),
    "uint64_t": (8, 8),
    "bool": (1, 1),
    "_Bool": (1, 1),
    "void*": (8, 8),
}

RUST_TYPE_SIZES: Dict[str, Tuple[int, int]] = {
    "i8": (1, 1),
    "u8": (1, 1),
    "i16": (2, 2),
    "u16": (2, 2),
    "i32": (4, 4),
    "u32": (4, 4),
    "i64": (8, 8),
    "u64": (8, 8),
    "i128": (16, 16),
    "u128": (16, 16),
    "f32": (4, 4),
    "f64": (8, 8),
    "isize": (8, 8),
    "usize": (8, 8),
    "bool": (1, 1),
    "char": (4, 4),
    "*const u8": (8, 8),
    "*mut u8": (8, 8),
}

C_TO_RUST_TYPE: Dict[str, str] = {
    "char": "i8",
    "unsigned char": "u8",
    "signed char": "i8",
    "short": "i16",
    "unsigned short": "u16",
    "int": "i32",
    "unsigned int": "u32",
    "unsigned": "u32",
    "long": "i64",
    "unsigned long": "u64",
    "long long": "i64",
    "unsigned long long": "u64",
    "float": "f32",
    "double": "f64",
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
}


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _numbered_lines(code: str) -> List[Tuple[int, str]]:
    return [(i + 1, line) for i, line in enumerate(code.splitlines())]


def _scope_depth_at(lines: List[Tuple[int, str]], target_line: int) -> int:
    depth = 0
    for lineno, text in lines:
        if lineno > target_line:
            break
        depth += text.count("{") - text.count("}")
    return max(depth, 0)


def _resolve_c_type(raw: str) -> str:
    raw = raw.strip()
    raw = re.sub(r"\s+", " ", raw)
    raw = raw.replace("struct ", "").replace("const ", "").strip()
    if raw.endswith("*"):
        return "void*"
    return raw


def _size_align_for_c(type_str: str) -> Tuple[int, int]:
    resolved = _resolve_c_type(type_str)
    if resolved in C_TYPE_SIZES:
        return C_TYPE_SIZES[resolved]
    if "*" in type_str or resolved == "void*":
        return (8, 8)
    return (4, 4)


def _size_align_for_rust(type_str: str) -> Tuple[int, int]:
    cleaned = type_str.strip()
    if cleaned in RUST_TYPE_SIZES:
        return RUST_TYPE_SIZES[cleaned]
    if cleaned.startswith("*const") or cleaned.startswith("*mut"):
        return (8, 8)
    if cleaned.startswith("Box<") or cleaned.startswith("&"):
        return (8, 8)
    return (4, 4)


def _extract_c_structs(code: str) -> List[Tuple[str, List[Tuple[str, str]]]]:
    structs: List[Tuple[str, List[Tuple[str, str]]]] = []
    pattern = re.compile(
        r"(?:typedef\s+)?struct\s+(\w+)\s*\{([^}]*)\}", re.DOTALL
    )
    for m in pattern.finditer(code):
        name = m.group(1)
        body = m.group(2)
        fields: List[Tuple[str, str]] = []
        for field_line in body.split(";"):
            field_line = field_line.strip()
            if not field_line:
                continue
            parts = field_line.rsplit(None, 1)
            if len(parts) == 2:
                ftype, fname = parts
                fname = fname.strip("*; \t")
                ftype = ftype.strip()
                if "*" in field_line and not ftype.endswith("*"):
                    ftype += "*"
                fields.append((ftype, fname))
        structs.append((name, fields))
    return structs


def _extract_rust_structs(
    code: str,
) -> List[Tuple[str, List[Tuple[str, str]]]]:
    structs: List[Tuple[str, List[Tuple[str, str]]]] = []
    pattern = re.compile(
        r"(?:pub\s+)?struct\s+(\w+)(?:<[^>]*>)?\s*\{([^}]*)\}", re.DOTALL
    )
    for m in pattern.finditer(code):
        name = m.group(1)
        body = m.group(2)
        fields: List[Tuple[str, str]] = []
        for field_line in body.split(","):
            field_line = field_line.strip()
            if not field_line:
                continue
            colon_idx = field_line.find(":")
            if colon_idx == -1:
                continue
            fname = field_line[:colon_idx].strip()
            fname = re.sub(r"^pub\s+", "", fname).strip()
            ftype = field_line[colon_idx + 1 :].strip().rstrip(",")
            fields.append((ftype, fname))
        structs.append((name, fields))
    return structs


def _find_c_pointer_declarations(code: str) -> List[Tuple[int, str, str]]:
    results: List[Tuple[int, str, str]] = []
    decl_re = re.compile(
        r"^\s*(?:(?:const|volatile|static|extern)\s+)*"
        r"([\w\s]+?)\s*\*\s*(\w+)\s*(?:=[^;]*)?\s*;",
        re.MULTILINE,
    )
    for lineno, line in _numbered_lines(code):
        m = decl_re.match(line)
        if m:
            ptype = m.group(1).strip()
            pname = m.group(2).strip()
            results.append((lineno, ptype, pname))
    return results


def _find_rust_references(code: str) -> List[Tuple[int, str, str]]:
    results: List[Tuple[int, str, str]] = []
    ref_re = re.compile(
        r"(?:let\s+(?:mut\s+)?)?(\w+)\s*:\s*(&(?:mut\s+)?\w[\w<>]*)", re.MULTILINE
    )
    for lineno, line in _numbered_lines(code):
        for m in ref_re.finditer(line):
            rname = m.group(1).strip()
            rtype = m.group(2).strip()
            results.append((lineno, rtype, rname))
    return results


def _build_scope_map(code: str) -> Dict[int, Tuple[int, int]]:
    scope_map: Dict[int, Tuple[int, int]] = {}
    stack: List[int] = []
    for lineno, line in _numbered_lines(code):
        for ch in line:
            if ch == "{":
                stack.append(lineno)
            elif ch == "}" and stack:
                open_line = stack.pop()
                scope_map[open_line] = (open_line, lineno)
    return scope_map


def _variables_in_scope(
    code: str, scope_map: Dict[int, Tuple[int, int]], line: int
) -> Set[str]:
    variables: Set[str] = set()
    var_re = re.compile(r"\b(?:int|char|float|double|void|long|short|unsigned)\s*\*?\s*(\w+)")
    for lineno, text in _numbered_lines(code):
        if lineno > line:
            break
        in_valid_scope = True
        for open_l, (_, close_l) in scope_map.items():
            if open_l <= lineno <= close_l and close_l < line:
                in_valid_scope = False
                break
        if in_valid_scope:
            for m in var_re.finditer(text):
                variables.add(m.group(1))
    return variables


# ---------------------------------------------------------------------------
# track_allocations
# ---------------------------------------------------------------------------

def track_allocations(c_code: str) -> AllocationMap:
    alloc_map = AllocationMap()
    lines = _numbered_lines(c_code)

    alloc_re = re.compile(
        r"(\w+)\s*=\s*(?:\(\s*\w[\w\s\*]*\s*\)\s*)?"
        r"(malloc|calloc|realloc|aligned_alloc|posix_memalign)\s*\(([^)]*)\)"
    )
    free_re = re.compile(r"\bfree\s*\(\s*(\w+)\s*\)")

    for lineno, line in lines:
        for m in alloc_re.finditer(line):
            var = m.group(1).strip()
            alloc_type = m.group(2).strip()
            size_expr = m.group(3).strip()
            depth = _scope_depth_at(lines, lineno)
            info = AllocationInfo(
                variable=var,
                alloc_type=alloc_type,
                line_number=lineno,
                byte_size=size_expr,
                scope_depth=depth,
                source_expression=m.group(0),
            )
            alloc_map.allocations[var] = info
            alloc_map.total_allocs += 1

    for lineno, line in lines:
        for m in free_re.finditer(line):
            var = m.group(1).strip()
            if var in alloc_map.allocations:
                info = alloc_map.allocations[var]
                if not info.freed:
                    info.freed = True
                    info.free_line = lineno
                    alloc_map.total_frees += 1
                    alloc_map.matched_pairs.append(
                        (var, info.line_number, lineno)
                    )
            else:
                alloc_map.total_frees += 1

    for var, info in alloc_map.allocations.items():
        if not info.freed:
            alloc_map.leaks.append(info)

    return alloc_map


# ---------------------------------------------------------------------------
# verify_ownership_transfer
# ---------------------------------------------------------------------------

def verify_ownership_transfer(
    c_code: str, rust_code: str
) -> OwnershipResult:
    result = OwnershipResult()

    c_pointers = _find_c_pointer_declarations(c_code)
    rust_refs = _find_rust_references(rust_code)

    c_alloc_map = track_allocations(c_code)

    rust_box_re = re.compile(
        r"(?:let\s+(?:mut\s+)?)?(\w+)\s*(?::\s*Box<[^>]+>)?\s*=\s*Box::new\(",
        re.MULTILINE,
    )
    rust_rc_re = re.compile(
        r"(?:let\s+(?:mut\s+)?)?(\w+)\s*(?::\s*Rc<[^>]+>)?\s*=\s*Rc::new\(",
        re.MULTILINE,
    )
    rust_arc_re = re.compile(
        r"(?:let\s+(?:mut\s+)?)?(\w+)\s*(?::\s*Arc<[^>]+>)?\s*=\s*Arc::new\(",
        re.MULTILINE,
    )
    rust_clone_re = re.compile(r"(\w+)\.clone\(\)")

    rust_owners: Dict[str, Tuple[int, OwnershipKind]] = {}
    for lineno, line in _numbered_lines(rust_code):
        for m in rust_box_re.finditer(line):
            rust_owners[m.group(1)] = (lineno, OwnershipKind.OWNED)
        for m in rust_rc_re.finditer(line):
            rust_owners[m.group(1)] = (lineno, OwnershipKind.SHARED)
        for m in rust_arc_re.finditer(line):
            rust_owners[m.group(1)] = (lineno, OwnershipKind.SHARED)

    for lineno, line in _numbered_lines(rust_code):
        for m in rust_clone_re.finditer(line):
            var = m.group(1)
            if var not in rust_owners:
                rust_owners[var] = (lineno, OwnershipKind.SHARED)

    for rline, rtype, rname in rust_refs:
        if rname not in rust_owners:
            kind = (
                OwnershipKind.MUTABLE_BORROWED
                if "mut" in rtype
                else OwnershipKind.BORROWED
            )
            rust_owners[rname] = (rline, kind)

    matched_c: Set[str] = set()
    matched_r: Set[str] = set()

    for c_line, c_type, c_name in c_pointers:
        best_match: Optional[str] = None
        best_score = 0
        for r_name, (r_line, r_kind) in rust_owners.items():
            score = 0
            if c_name == r_name:
                score += 10
            elif c_name.lower() == r_name.lower():
                score += 7
            elif c_name.replace("_", "") == r_name.replace("_", ""):
                score += 5
            c_lower = c_name.lower()
            r_lower = r_name.lower()
            common = len(set(c_lower) & set(r_lower))
            total = max(len(set(c_lower) | set(r_lower)), 1)
            score += int(5 * common / total)
            if score > best_score:
                best_score = score
                best_match = r_name
        if best_match and best_score >= 5:
            r_line, r_kind = rust_owners[best_match]
            if c_name in c_alloc_map.allocations:
                pattern = "heap-allocated pointer → owned"
            else:
                pattern = "stack pointer → reference"
            transfer = OwnershipTransfer(
                c_variable=c_name,
                rust_variable=best_match,
                kind=r_kind,
                c_line=c_line,
                rust_line=r_line,
                pattern=pattern,
            )
            result.transfers.append(transfer)
            matched_c.add(c_name)
            matched_r.add(best_match)

    for _, _, c_name in c_pointers:
        if c_name not in matched_c:
            result.unmapped_c_pointers.append(c_name)

    for r_name in rust_owners:
        if r_name not in matched_r:
            result.unmapped_rust_owners.append(r_name)

    if result.unmapped_c_pointers:
        result.compatible = False
        result.warnings.append(
            f"{len(result.unmapped_c_pointers)} C pointer(s) have no Rust equivalent"
        )

    return result


# ---------------------------------------------------------------------------
# detect_dangling_pointers
# ---------------------------------------------------------------------------

def detect_dangling_pointers(c_code: str) -> List[DanglingPointer]:
    dangles: List[DanglingPointer] = []
    lines = _numbered_lines(c_code)
    scope_map = _build_scope_map(c_code)

    local_addr_re = re.compile(r"(\w+)\s*=\s*&(\w+)")
    return_local_re = re.compile(r"return\s+&(\w+)")
    local_decl_re = re.compile(
        r"(?:int|char|float|double|long|short|unsigned|struct\s+\w+)\s+(\w+)\s*[;=]"
    )

    local_vars: Dict[str, Tuple[int, int]] = {}
    for lineno, line in lines:
        for m in local_decl_re.finditer(line):
            var = m.group(1)
            depth = _scope_depth_at(lines, lineno)
            scope_end = lineno
            for open_l, (_, close_l) in scope_map.items():
                if open_l <= lineno <= close_l:
                    scope_end = max(scope_end, close_l)
            local_vars[var] = (lineno, scope_end)

    pointer_targets: Dict[str, Tuple[str, int]] = {}
    for lineno, line in lines:
        for m in local_addr_re.finditer(line):
            ptr_name = m.group(1)
            target = m.group(2)
            pointer_targets[ptr_name] = (target, lineno)

    for ptr_name, (target, assign_line) in pointer_targets.items():
        if target in local_vars:
            _, scope_end = local_vars[target]
            use_re = re.compile(rf"\b{re.escape(ptr_name)}\b")
            for lineno, line in lines:
                if lineno > scope_end and use_re.search(line):
                    if "free" not in line and "=" not in line.split(ptr_name)[0][-3:]:
                        dangles.append(
                            DanglingPointer(
                                variable=ptr_name,
                                line_number=lineno,
                                scope_ended_line=scope_end,
                                reason=(
                                    f"'{ptr_name}' points to local '{target}' whose "
                                    f"scope ended at line {scope_end}"
                                ),
                            )
                        )
                        break

    for lineno, line in lines:
        m = return_local_re.search(line)
        if m:
            var = m.group(1)
            if var in local_vars:
                dangles.append(
                    DanglingPointer(
                        variable=var,
                        line_number=lineno,
                        scope_ended_line=lineno,
                        reason=f"Returning address of local variable '{var}'",
                        severity=SeverityLevel.CRITICAL,
                    )
                )

    free_re = re.compile(r"\bfree\s*\(\s*(\w+)\s*\)")
    freed_vars: Dict[str, int] = {}
    for lineno, line in lines:
        m = free_re.search(line)
        if m:
            var = m.group(1)
            freed_vars[var] = lineno

    for var, free_line in freed_vars.items():
        use_re = re.compile(rf"\b{re.escape(var)}\b")
        for lineno, line in lines:
            if lineno > free_line:
                if use_re.search(line) and "free" not in line and "NULL" not in line:
                    if not re.search(rf"{re.escape(var)}\s*=", line):
                        dangles.append(
                            DanglingPointer(
                                variable=var,
                                line_number=lineno,
                                scope_ended_line=free_line,
                                reason=f"'{var}' used after free at line {free_line}",
                                severity=SeverityLevel.CRITICAL,
                            )
                        )
                        break

    return dangles


# ---------------------------------------------------------------------------
# detect_double_free
# ---------------------------------------------------------------------------

def detect_double_free(c_code: str) -> List[DoubleFree]:
    doubles: List[DoubleFree] = []
    lines = _numbered_lines(c_code)
    free_re = re.compile(r"\bfree\s*\(\s*(\w+)\s*\)")

    free_locations: Dict[str, List[int]] = {}
    for lineno, line in lines:
        for m in free_re.finditer(line):
            var = m.group(1)
            free_locations.setdefault(var, []).append(lineno)

    for var, free_lines in free_locations.items():
        if len(free_lines) < 2:
            continue
        null_set_re = re.compile(
            rf"\b{re.escape(var)}\s*=\s*NULL\b"
        )
        realloc_re = re.compile(
            rf"\b{re.escape(var)}\s*=\s*(?:\([^)]*\)\s*)?(?:malloc|calloc|realloc)\b"
        )
        for i in range(len(free_lines) - 1):
            first = free_lines[i]
            second = free_lines[i + 1]
            nullified = False
            reallocated = False
            for lineno, line in lines:
                if first < lineno < second:
                    if null_set_re.search(line):
                        nullified = True
                        break
                    if realloc_re.search(line):
                        reallocated = True
                        break
            if not nullified and not reallocated:
                doubles.append(
                    DoubleFree(
                        variable=var,
                        first_free_line=first,
                        second_free_line=second,
                    )
                )

    return doubles


# ---------------------------------------------------------------------------
# detect_use_after_free
# ---------------------------------------------------------------------------

def detect_use_after_free(c_code: str) -> List[UseAfterFree]:
    uafs: List[UseAfterFree] = []
    lines = _numbered_lines(c_code)
    free_re = re.compile(r"\bfree\s*\(\s*(\w+)\s*\)")

    free_locations: Dict[str, List[int]] = {}
    for lineno, line in lines:
        for m in free_re.finditer(line):
            var = m.group(1)
            free_locations.setdefault(var, []).append(lineno)

    for var, free_lines in free_locations.items():
        for free_line in free_lines:
            use_re = re.compile(
                rf"(?<!\bfree\s*\()\b{re.escape(var)}\b(?!\s*=\s*(?:NULL|0|malloc|calloc|realloc))"
            )
            null_set_re = re.compile(rf"\b{re.escape(var)}\s*=\s*(?:NULL|0)\b")
            realloc_re = re.compile(
                rf"\b{re.escape(var)}\s*=\s*(?:\([^)]*\)\s*)?(?:malloc|calloc|realloc)\b"
            )
            for lineno, line in lines:
                if lineno <= free_line:
                    continue
                if null_set_re.search(line) or realloc_re.search(line):
                    break
                stripped = line.strip()
                if stripped.startswith("free("):
                    continue
                if f"free({var})" in stripped or f"free( {var} )" in stripped:
                    continue
                if use_re.search(line):
                    expr = line.strip()
                    uafs.append(
                        UseAfterFree(
                            variable=var,
                            free_line=free_line,
                            use_line=lineno,
                            use_expression=expr,
                        )
                    )
                    break

    return uafs


# ---------------------------------------------------------------------------
# lifetime_inference
# ---------------------------------------------------------------------------

def lifetime_inference(rust_code: str) -> LifetimeMap:
    lt_map = LifetimeMap()
    lines = _numbered_lines(rust_code)

    fn_re = re.compile(
        r"(?:pub\s+)?fn\s+(\w+)\s*(<[^>]*>)?\s*\(([^)]*)\)\s*(?:->\s*([^\{;]+))?"
    )
    struct_re = re.compile(
        r"(?:pub\s+)?struct\s+(\w+)\s*(?:<([^>]*)>)?\s*\{([^}]*)\}",
        re.DOTALL,
    )

    for m in fn_re.finditer(rust_code):
        fn_name = m.group(1)
        _generics = m.group(2) or ""
        params = m.group(3).strip()
        ret = (m.group(4) or "").strip()

        ref_params: List[str] = []
        for part in params.split(","):
            part = part.strip()
            if "&" in part:
                ref_params.append(part)

        needs_explicit = False
        ref_count = len(ref_params)
        has_self = any("self" in p for p in ref_params)
        returns_ref = "&" in ret

        if returns_ref:
            if ref_count == 0:
                needs_explicit = True
            elif ref_count == 1:
                lt_map.elision_applicable.append(fn_name)
            elif has_self:
                lt_map.elision_applicable.append(fn_name)
            else:
                needs_explicit = True

        if needs_explicit:
            lt_map.functions_needing_lifetimes.append(fn_name)
            lifetime_idx = 0
            for param in ref_params:
                lt_name = f"'{'abcdefghijklmnopqrstuvwxyz'[lifetime_idx % 26]}"
                colon = param.find(":")
                pname = param[:colon].strip() if colon != -1 else param.strip()
                pname = re.sub(r"^(?:mut\s+|&\s*)", "", pname).strip()
                lt_map.annotations.append(
                    LifetimeAnnotation(
                        name=lt_name,
                        applies_to=pname,
                        relation=LifetimeRelation.OUTLIVES,
                        inferred_from=f"parameter of {fn_name}",
                    )
                )
                lifetime_idx += 1
            if returns_ref:
                lt_map.annotations.append(
                    LifetimeAnnotation(
                        name="'a",
                        applies_to=f"{fn_name}::return",
                        relation=LifetimeRelation.SUBSET,
                        inferred_from=f"return type of {fn_name}",
                    )
                )

    for m in struct_re.finditer(rust_code):
        struct_name = m.group(1)
        existing_lifetimes = m.group(2) or ""
        body = m.group(3)

        has_refs = "&" in body
        has_lifetime_params = "'" in existing_lifetimes

        if has_refs and not has_lifetime_params:
            lt_map.structs_needing_lifetimes.append(struct_name)
            ref_fields = re.findall(r"(\w+)\s*:\s*&", body)
            for fname in ref_fields:
                lt_map.annotations.append(
                    LifetimeAnnotation(
                        name="'a",
                        applies_to=f"{struct_name}::{fname}",
                        relation=LifetimeRelation.OUTLIVES,
                        inferred_from=f"reference field in struct {struct_name}",
                    )
                )

    impl_re = re.compile(
        r"impl(?:<([^>]*)>)?\s+(\w+)(?:<([^>]*)>)?\s*\{", re.DOTALL
    )
    for m in impl_re.finditer(rust_code):
        impl_name = m.group(2)
        if impl_name in lt_map.structs_needing_lifetimes:
            has_lt = m.group(1) and "'" in m.group(1)
            if not has_lt:
                lt_map.annotations.append(
                    LifetimeAnnotation(
                        name="'a",
                        applies_to=f"impl {impl_name}",
                        relation=LifetimeRelation.EQUAL,
                        inferred_from=f"impl block for {impl_name} with lifetime fields",
                    )
                )

    return lt_map


# ---------------------------------------------------------------------------
# memory_layout_comparison
# ---------------------------------------------------------------------------

def memory_layout_comparison(
    c_struct: str, rust_struct: str
) -> LayoutDiff:
    diff = LayoutDiff()

    c_structs = _extract_c_structs(c_struct)
    r_structs = _extract_rust_structs(rust_struct)

    if not c_structs or not r_structs:
        diff.notes.append("Could not extract struct definitions from one or both inputs")
        return diff

    c_name, c_fields = c_structs[0]
    r_name, r_fields = r_structs[0]

    c_offset = 0
    r_offset = 0

    c_field_map: Dict[str, Tuple[str, int]] = {}
    for ftype, fname in c_fields:
        c_field_map[fname] = (ftype, len(c_field_map))

    r_field_map: Dict[str, Tuple[str, int]] = {}
    for ftype, fname in r_fields:
        r_field_map[fname] = (ftype, len(r_field_map))

    all_names = list(dict.fromkeys(
        [f for _, f in c_fields] + [f for _, f in r_fields]
    ))

    c_offset = 0
    r_offset = 0

    for fname in all_names:
        c_info = c_field_map.get(fname)
        r_info = r_field_map.get(fname)

        if c_info is None or r_info is None:
            if c_info and not r_info:
                diff.notes.append(f"Field '{fname}' exists in C but not in Rust")
            elif r_info and not c_info:
                diff.notes.append(f"Field '{fname}' exists in Rust but not in C")
            diff.compatible = False
            continue

        c_type_str = c_info[0]
        r_type_str = r_info[0]

        c_sz, c_al = _size_align_for_c(c_type_str)
        r_sz, r_al = _size_align_for_rust(r_type_str)

        c_pad = (c_al - (c_offset % c_al)) % c_al
        c_offset += c_pad
        r_pad = (r_al - (r_offset % r_al)) % r_al
        r_offset += r_pad

        lf = LayoutField(
            name=fname,
            c_type=c_type_str,
            rust_type=r_type_str,
            c_size=c_sz,
            rust_size=r_sz,
            c_offset=c_offset,
            rust_offset=r_offset,
            c_alignment=c_al,
            rust_alignment=r_al,
        )
        diff.fields.append(lf)

        if c_sz != r_sz:
            diff.compatible = False
            diff.padding_differences.append(
                f"Field '{fname}': C size {c_sz} != Rust size {r_sz}"
            )
        if c_offset != r_offset:
            diff.padding_differences.append(
                f"Field '{fname}': C offset {c_offset} != Rust offset {r_offset}"
            )

        c_offset += c_sz
        r_offset += r_sz

    max_c_align = max((f.c_alignment for f in diff.fields), default=1)
    max_r_align = max((f.rust_alignment for f in diff.fields), default=1)
    c_pad_end = (max_c_align - (c_offset % max_c_align)) % max_c_align
    r_pad_end = (max_r_align - (r_offset % max_r_align)) % max_r_align
    diff.c_total_size = c_offset + c_pad_end
    diff.rust_total_size = r_offset + r_pad_end

    if diff.c_total_size != diff.rust_total_size:
        diff.compatible = False
        diff.notes.append(
            f"Total sizes differ: C={diff.c_total_size}, Rust={diff.rust_total_size}"
        )

    return diff


# ---------------------------------------------------------------------------
# alignment_verification
# ---------------------------------------------------------------------------

def alignment_verification(
    c_code: str, rust_code: str
) -> AlignmentResult:
    result = AlignmentResult()

    c_align_re = re.compile(
        r"__attribute__\s*\(\s*\(\s*aligned\s*\(\s*(\d+)\s*\)\s*\)\s*\)\s*"
        r"(?:[\w\s\*]+)\s+(\w+)"
    )
    c_alignas_re = re.compile(r"_Alignas\s*\(\s*(\d+)\s*\)\s+[\w\s\*]+\s+(\w+)")
    c_pragma_re = re.compile(r"#pragma\s+pack\s*\(\s*(?:push\s*,\s*)?(\d+)\s*\)")

    rust_repr_re = re.compile(
        r"#\[repr\((?:C,\s*)?align\((\d+)\)\)\]\s*(?:pub\s+)?struct\s+(\w+)"
    )
    rust_repr_c_re = re.compile(
        r"#\[repr\(C\)\]\s*(?:pub\s+)?struct\s+(\w+)"
    )
    rust_repr_packed_re = re.compile(
        r"#\[repr\((?:C,\s*)?packed(?:\((\d+)\))?\)\]\s*(?:pub\s+)?struct\s+(\w+)"
    )

    pragma_pack: Optional[int] = None
    for m in c_pragma_re.finditer(c_code):
        pragma_pack = int(m.group(1))

    for m in c_align_re.finditer(c_code):
        alignment = int(m.group(1))
        var_name = m.group(2)
        result.c_alignments[var_name] = alignment

    for m in c_alignas_re.finditer(c_code):
        alignment = int(m.group(1))
        var_name = m.group(2)
        result.c_alignments[var_name] = alignment

    for m in rust_repr_re.finditer(rust_code):
        alignment = int(m.group(1))
        struct_name = m.group(2)
        result.rust_alignments[struct_name] = alignment

    for m in rust_repr_packed_re.finditer(rust_code):
        pack_val = int(m.group(1)) if m.group(1) else 1
        struct_name = m.group(2)
        result.rust_alignments[struct_name] = pack_val

    c_structs = _extract_c_structs(c_code)
    for sname, fields in c_structs:
        if sname not in result.c_alignments:
            max_align = 1
            for ftype, _ in fields:
                _, fa = _size_align_for_c(ftype)
                max_align = max(max_align, fa)
            if pragma_pack is not None:
                max_align = min(max_align, pragma_pack)
            result.c_alignments[sname] = max_align

    r_structs = _extract_rust_structs(rust_code)
    for sname, fields in r_structs:
        if sname not in result.rust_alignments:
            max_align = 1
            for ftype, _ in fields:
                _, fa = _size_align_for_rust(ftype)
                max_align = max(max_align, fa)
            result.rust_alignments[sname] = max_align

    for name in set(result.c_alignments) & set(result.rust_alignments):
        c_al = result.c_alignments[name]
        r_al = result.rust_alignments[name]
        if c_al != r_al:
            result.compatible = False
            result.mismatches.append(
                f"'{name}': C alignment={c_al}, Rust alignment={r_al}"
            )

    for name in set(result.c_alignments) - set(result.rust_alignments):
        result.warnings.append(
            f"C type '{name}' (align={result.c_alignments[name]}) has no Rust counterpart"
        )

    for name in set(result.rust_alignments) - set(result.c_alignments):
        result.warnings.append(
            f"Rust type '{name}' (align={result.rust_alignments[name]}) has no C counterpart"
        )

    return result


# ---------------------------------------------------------------------------
# suggest_smart_pointers
# ---------------------------------------------------------------------------

def suggest_smart_pointers(c_code: str) -> List[SmartPointerSuggestion]:
    suggestions: List[SmartPointerSuggestion] = []
    lines = _numbered_lines(c_code)

    alloc_re = re.compile(
        r"(\w+)\s*=\s*(?:\([^)]*\)\s*)?(malloc|calloc|realloc)\s*\(([^)]*)\)"
    )
    free_re = re.compile(r"\bfree\s*\(\s*(\w+)\s*\)")
    ref_count_re = re.compile(r"(\w+)\s*->\s*ref_?count\b")
    atomic_re = re.compile(r"atomic_(?:fetch_add|fetch_sub|load|store)\s*\(\s*&?\s*(\w+)")
    pthread_re = re.compile(r"pthread_(?:mutex|rwlock|cond)")
    shared_re = re.compile(r"(\w+)\s*=\s*(\w+)\s*;")

    alloc_vars: Dict[str, Tuple[int, str]] = {}
    for lineno, line in lines:
        for m in alloc_re.finditer(line):
            var = m.group(1)
            alloc_type = m.group(2)
            alloc_vars[var] = (lineno, alloc_type)

    free_vars: Dict[str, List[int]] = {}
    for lineno, line in lines:
        for m in free_re.finditer(line):
            var = m.group(1)
            free_vars.setdefault(var, []).append(lineno)

    ref_counted_vars: Set[str] = set()
    for lineno, line in lines:
        for m in ref_count_re.finditer(line):
            ref_counted_vars.add(m.group(1))

    atomic_vars: Set[str] = set()
    for lineno, line in lines:
        for m in atomic_re.finditer(line):
            atomic_vars.add(m.group(1))

    has_threading = any(pthread_re.search(line) for _, line in lines)

    shared_targets: Dict[str, Set[str]] = {}
    for lineno, line in lines:
        for m in shared_re.finditer(line):
            dest = m.group(1)
            src = m.group(2)
            if src in alloc_vars and dest != src:
                shared_targets.setdefault(src, set()).add(dest)

    for var, (lineno, alloc_type) in alloc_vars.items():
        if var in ref_counted_vars and has_threading:
            suggestions.append(
                SmartPointerSuggestion(
                    variable=var,
                    line_number=lineno,
                    current_pattern=f"Reference-counted with atomic ops ({alloc_type})",
                    suggested_type=SmartPointerType.ARC,
                    reason="Reference counting with threading detected; Arc provides thread-safe shared ownership",
                    confidence=0.9,
                )
            )
            continue

        if var in ref_counted_vars:
            suggestions.append(
                SmartPointerSuggestion(
                    variable=var,
                    line_number=lineno,
                    current_pattern=f"Reference-counted ({alloc_type})",
                    suggested_type=SmartPointerType.RC,
                    reason="Reference counting detected; Rc provides single-threaded shared ownership",
                    confidence=0.85,
                )
            )
            continue

        if var in shared_targets and len(shared_targets[var]) > 0:
            if has_threading:
                suggestions.append(
                    SmartPointerSuggestion(
                        variable=var,
                        line_number=lineno,
                        current_pattern=f"Shared pointer with threading ({alloc_type})",
                        suggested_type=SmartPointerType.ARC,
                        reason=(
                            f"Pointer shared with {len(shared_targets[var])} other variable(s) "
                            f"in threaded context"
                        ),
                        confidence=0.75,
                    )
                )
            else:
                suggestions.append(
                    SmartPointerSuggestion(
                        variable=var,
                        line_number=lineno,
                        current_pattern=f"Shared pointer ({alloc_type})",
                        suggested_type=SmartPointerType.RC,
                        reason=f"Pointer shared with {len(shared_targets[var])} other variable(s)",
                        confidence=0.7,
                    )
                )
            continue

        if var in free_vars:
            suggestions.append(
                SmartPointerSuggestion(
                    variable=var,
                    line_number=lineno,
                    current_pattern=f"Unique allocation with free ({alloc_type})",
                    suggested_type=SmartPointerType.BOX,
                    reason="Single-owner heap allocation with explicit free → Box",
                    confidence=0.9,
                )
            )
        else:
            suggestions.append(
                SmartPointerSuggestion(
                    variable=var,
                    line_number=lineno,
                    current_pattern=f"Allocation without free ({alloc_type})",
                    suggested_type=SmartPointerType.BOX,
                    reason="Heap allocation without free (potential leak) → Box for automatic cleanup",
                    confidence=0.8,
                )
            )

    mutex_re = re.compile(r"pthread_mutex_(?:lock|unlock)\s*\(\s*&?\s*(\w+)")
    guarded_vars: Set[str] = set()
    for lineno, line in lines:
        for m in mutex_re.finditer(line):
            guarded_vars.add(m.group(1))

    for var in guarded_vars:
        already = any(s.variable == var for s in suggestions)
        if not already:
            suggestions.append(
                SmartPointerSuggestion(
                    variable=var,
                    line_number=0,
                    current_pattern="Mutex-guarded variable",
                    suggested_type=SmartPointerType.MUTEX,
                    reason="Variable protected by pthread_mutex → Mutex<T>",
                    confidence=0.8,
                )
            )

    interior_mut_re = re.compile(
        r"(?:volatile\s+)(\w[\w\s\*]*)\s+(\w+)"
    )
    for lineno, line in lines:
        for m in interior_mut_re.finditer(line):
            var = m.group(2)
            already = any(s.variable == var for s in suggestions)
            if not already:
                suggestions.append(
                    SmartPointerSuggestion(
                        variable=var,
                        line_number=lineno,
                        current_pattern="Volatile variable",
                        suggested_type=SmartPointerType.REF_CELL,
                        reason="Volatile access pattern suggests interior mutability → RefCell or Cell",
                        confidence=0.6,
                    )
                )

    return suggestions


# ---------------------------------------------------------------------------
# verify_memory_safety  (top-level orchestrator)
# ---------------------------------------------------------------------------

def verify_memory_safety(
    c_code: str, rust_code: str
) -> MemorySafetyResult:
    result = MemorySafetyResult()

    result.allocation_map = track_allocations(c_code)
    result.ownership_result = verify_ownership_transfer(c_code, rust_code)
    result.dangling_pointers = detect_dangling_pointers(c_code)
    result.double_frees = detect_double_free(c_code)
    result.use_after_frees = detect_use_after_free(c_code)
    result.lifetime_map = lifetime_inference(rust_code)
    result.smart_pointer_suggestions = suggest_smart_pointers(c_code)

    if result.allocation_map.leaks:
        result.issues.append(
            f"{len(result.allocation_map.leaks)} memory leak(s) detected in C code"
        )
    if result.dangling_pointers:
        result.issues.append(
            f"{len(result.dangling_pointers)} dangling pointer(s) detected"
        )
    if result.double_frees:
        result.issues.append(
            f"{len(result.double_frees)} double-free(s) detected"
        )
    if result.use_after_frees:
        result.issues.append(
            f"{len(result.use_after_frees)} use-after-free(s) detected"
        )
    if not result.ownership_result.compatible:
        result.issues.append("Ownership mapping is incomplete or incompatible")
    if result.lifetime_map.functions_needing_lifetimes:
        fns = ", ".join(result.lifetime_map.functions_needing_lifetimes)
        result.issues.append(f"Functions needing explicit lifetimes: {fns}")
    if result.lifetime_map.structs_needing_lifetimes:
        sts = ", ".join(result.lifetime_map.structs_needing_lifetimes)
        result.issues.append(f"Structs needing lifetime parameters: {sts}")

    has_critical = (
        len(result.double_frees) > 0 or len(result.use_after_frees) > 0
    )
    has_errors = (
        len(result.dangling_pointers) > 0
        or len(result.allocation_map.leaks) > 0
    )
    has_warnings = not result.ownership_result.compatible

    if has_critical:
        result.severity = SeverityLevel.CRITICAL
        result.overall_safe = False
    elif has_errors:
        result.severity = SeverityLevel.ERROR
        result.overall_safe = False
    elif has_warnings:
        result.severity = SeverityLevel.WARNING
        result.overall_safe = False
    else:
        result.severity = SeverityLevel.INFO
        result.overall_safe = True

    return result
