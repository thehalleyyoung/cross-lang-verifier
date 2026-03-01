"""
Migration planner for C-to-Rust code migration.

Analyzes C ASTs to produce comprehensive migration plans including dependency
ordering, type mapping, ownership inference, unsafe boundary detection,
FFI wrapper generation, and risk assessment.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Set, Tuple, Optional, Any
from enum import Enum
from collections import defaultdict, deque


# ---------------------------------------------------------------------------
# Data classes used across the module
# ---------------------------------------------------------------------------

@dataclass
class RustType:
    """Representation of a mapped Rust type."""
    name: str
    is_reference: bool = False
    is_mutable: bool = False
    lifetime: Optional[str] = None
    generic_params: List[str] = field(default_factory=list)
    confidence: float = 1.0

    def to_string(self) -> str:
        prefix = ""
        if self.is_reference:
            lt = f"'{self.lifetime} " if self.lifetime else ""
            mut = "mut " if self.is_mutable else ""
            prefix = f"&{lt}{mut}"
        base = self.name
        if self.generic_params:
            base += f"<{', '.join(self.generic_params)}>"
        return f"{prefix}{base}"

    def __repr__(self) -> str:
        return f"RustType({self.to_string()}, confidence={self.confidence:.2f})"


class OwnershipKind(Enum):
    OWNED = "owned"
    BORROWED = "borrowed"
    BORROWED_MUT = "borrowed_mut"
    BOXED = "boxed"
    RC = "rc"
    ARC = "arc"
    RAW = "raw"


@dataclass
class OwnershipSuggestion:
    variable: str
    kind: OwnershipKind
    reason: str
    confidence: float = 1.0


@dataclass
class UnsafeReason:
    kind: str
    location: str
    description: str
    suggestion: str


@dataclass
class RiskLevel:
    level: str
    score: int
    factors: List[str]

    def __repr__(self) -> str:
        return f"RiskLevel({self.level}, score={self.score}, factors={self.factors})"


@dataclass
class MigrationPlan:
    migration_order: List[str]
    type_mappings: Dict[str, RustType]
    ownership_suggestions: Dict[str, OwnershipSuggestion]
    ffi_wrappers: List[str]
    risk_assessment: Dict[str, RiskLevel]
    estimated_effort: int

    def to_dict(self) -> dict:
        return {
            "migration_order": list(self.migration_order),
            "type_mappings": {
                k: {"rust_type": v.to_string(), "confidence": v.confidence}
                for k, v in self.type_mappings.items()
            },
            "ownership_suggestions": {
                k: {"kind": v.kind.value, "reason": v.reason, "confidence": v.confidence}
                for k, v in self.ownership_suggestions.items()
            },
            "ffi_wrappers": list(self.ffi_wrappers),
            "risk_assessment": {
                k: {"level": v.level, "score": v.score, "factors": v.factors}
                for k, v in self.risk_assessment.items()
            },
            "estimated_effort": self.estimated_effort,
        }

    def summary(self) -> str:
        lines = [
            "=== Migration Plan Summary ===",
            f"Functions to migrate: {len(self.migration_order)}",
            f"Type mappings: {len(self.type_mappings)}",
            f"Ownership suggestions: {len(self.ownership_suggestions)}",
            f"FFI wrappers needed: {len(self.ffi_wrappers)}",
            f"Estimated Rust LOC: {self.estimated_effort}",
            "",
            "--- Risk Breakdown ---",
        ]
        risk_counts: Dict[str, int] = {"low": 0, "medium": 0, "high": 0}
        for rl in self.risk_assessment.values():
            risk_counts[rl.level] = risk_counts.get(rl.level, 0) + 1
        for lvl in ("low", "medium", "high"):
            lines.append(f"  {lvl}: {risk_counts.get(lvl, 0)} functions")
        lines.append("")
        lines.append("--- Migration Order ---")
        for idx, fname in enumerate(self.migration_order, 1):
            risk = self.risk_assessment.get(fname)
            tag = f" [{risk.level}]" if risk else ""
            lines.append(f"  {idx}. {fname}{tag}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# CallGraph
# ---------------------------------------------------------------------------

class CallGraph:
    """Directed graph representing function call relationships."""

    def __init__(self) -> None:
        self._adjacency: Dict[str, Set[str]] = defaultdict(set)
        self._reverse: Dict[str, Set[str]] = defaultdict(set)
        self._nodes: Set[str] = set()

    def add_node(self, name: str) -> None:
        self._nodes.add(name)
        if name not in self._adjacency:
            self._adjacency[name] = set()

    def add_edge(self, caller: str, callee: str) -> None:
        self._nodes.add(caller)
        self._nodes.add(callee)
        self._adjacency[caller].add(callee)
        self._reverse[callee].add(caller)

    def get_dependencies(self, name: str) -> Set[str]:
        return set(self._adjacency.get(name, set()))

    def get_dependents(self, name: str) -> Set[str]:
        return set(self._reverse.get(name, set()))

    def topological_sort(self) -> List[str]:
        in_degree: Dict[str, int] = {n: 0 for n in self._nodes}
        for node in self._nodes:
            for dep in self._adjacency.get(node, set()):
                if dep in in_degree:
                    in_degree[dep] += 1

        queue = deque(sorted(n for n, d in in_degree.items() if d == 0))
        result: List[str] = []

        while queue:
            node = queue.popleft()
            result.append(node)
            for dep in sorted(self._adjacency.get(node, set())):
                if dep in in_degree:
                    in_degree[dep] -= 1
                    if in_degree[dep] == 0:
                        queue.append(dep)

        # If result is shorter than nodes, cycles exist; append remaining
        remaining = [n for n in sorted(self._nodes) if n not in set(result)]
        result.extend(remaining)
        return result

    def find_cycles(self) -> List[List[str]]:
        WHITE, GRAY, BLACK = 0, 1, 2
        color: Dict[str, int] = {n: WHITE for n in self._nodes}
        path: List[str] = []
        cycles: List[List[str]] = []

        def dfs(node: str) -> None:
            color[node] = GRAY
            path.append(node)
            for neighbour in sorted(self._adjacency.get(node, set())):
                if neighbour not in color:
                    continue
                if color[neighbour] == GRAY:
                    idx = path.index(neighbour)
                    cycle = list(path[idx:])
                    cycle.append(neighbour)
                    cycles.append(cycle)
                elif color[neighbour] == WHITE:
                    dfs(neighbour)
            path.pop()
            color[node] = BLACK

        for node in sorted(self._nodes):
            if color[node] == WHITE:
                dfs(node)
        return cycles

    def suggest_cycle_breaks(self) -> List[Tuple[str, str]]:
        cycles = self.find_cycles()
        breaks: List[Tuple[str, str]] = []
        seen_edges: Set[Tuple[str, str]] = set()
        for cycle in cycles:
            best_edge: Optional[Tuple[str, str]] = None
            best_score = -1
            for i in range(len(cycle) - 1):
                a, b = cycle[i], cycle[i + 1]
                if (a, b) in seen_edges:
                    continue
                score = len(self._reverse.get(b, set()))
                if best_edge is None or score < best_score:
                    best_edge = (a, b)
                    best_score = score
            if best_edge and best_edge not in seen_edges:
                breaks.append(best_edge)
                seen_edges.add(best_edge)
        return breaks


# ---------------------------------------------------------------------------
# TypeMapper
# ---------------------------------------------------------------------------

class TypeMapper:
    """Maps C types to Rust equivalents."""

    PRIMITIVE_MAP: Dict[str, str] = {
        "int": "i32",
        "unsigned int": "u32",
        "signed int": "i32",
        "long": "i64",
        "long int": "i64",
        "unsigned long": "u64",
        "unsigned long int": "u64",
        "long long": "i64",
        "unsigned long long": "u64",
        "short": "i16",
        "unsigned short": "u16",
        "signed short": "i16",
        "char": "i8",
        "signed char": "i8",
        "unsigned char": "u8",
        "float": "f32",
        "double": "f64",
        "long double": "f64",
        "void": "()",
        "_Bool": "bool",
        "bool": "bool",
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
    }

    def __init__(self) -> None:
        self._typedef_map: Dict[str, str] = {}
        self._struct_map: Dict[str, List[Tuple[str, str]]] = {}
        self._enum_map: Dict[str, List[str]] = {}

    def register_typedef(self, alias: str, original: str) -> None:
        self._typedef_map[alias] = original

    def register_struct(self, name: str, fields: List[Tuple[str, str]]) -> None:
        self._struct_map[name] = list(fields)

    def register_enum(self, name: str, variants: List[str]) -> None:
        self._enum_map[name] = list(variants)

    def map_type(self, c_type: str, usage_context: Optional[str] = None) -> RustType:
        c_type = c_type.strip()

        # Resolve typedefs first
        resolved = c_type
        depth = 0
        while resolved in self._typedef_map and depth < 10:
            resolved = self._typedef_map[resolved]
            depth += 1

        # Function pointer: int (*)(int, int) or similar
        if "(*)" in c_type or "(*" in c_type:
            return self._map_function_pointer(c_type)

        # Pointer types
        if resolved.endswith("*"):
            return self._map_pointer_type(resolved, usage_context)

        # Array types: int[10], char[]
        if "[" in resolved:
            return self._map_array_type(resolved)

        # Struct types
        if resolved.startswith("struct "):
            struct_name = resolved[7:].strip()
            return RustType(name=struct_name, confidence=0.9)

        # Union types
        if resolved.startswith("union "):
            union_name = resolved[6:].strip()
            return RustType(name=f"{union_name}Union", confidence=0.6)

        # Enum types
        if resolved.startswith("enum "):
            enum_name = resolved[5:].strip()
            return RustType(name=enum_name, confidence=0.95)

        # const qualifier removal
        if resolved.startswith("const "):
            inner = self.map_type(resolved[6:], usage_context)
            inner.confidence *= 0.95
            return inner

        # Primitive mapping
        if resolved in self.PRIMITIVE_MAP:
            return RustType(name=self.PRIMITIVE_MAP[resolved], confidence=1.0)

        # Typedef that was registered (avoid re-resolution since we already resolved above)
        if c_type in self._typedef_map and c_type != resolved:
            aliased = self.map_type(resolved, usage_context)
            aliased.confidence *= 0.9
            return aliased

        # Unknown type -- keep name, low confidence
        return RustType(name=resolved, confidence=0.3)

    def _map_pointer_type(self, c_type: str, usage_context: Optional[str]) -> RustType:
        base = c_type.rstrip("* ").strip()

        # void* -> *mut u8
        if base == "void":
            return RustType(name="*mut u8", confidence=0.5)

        # char* -> String (owned) or &str (borrowed)
        if base in ("char", "const char"):
            if usage_context == "borrowed":
                return RustType(name="str", is_reference=True, confidence=0.8)
            return RustType(name="String", confidence=0.75)

        # Double pointer
        if base.endswith("*"):
            inner = self._map_pointer_type(base, usage_context)
            return RustType(
                name=f"*mut {inner.to_string()}",
                confidence=inner.confidence * 0.6,
            )

        inner = self.map_type(base)
        if usage_context == "borrowed":
            return RustType(
                name=inner.name, is_reference=True, is_mutable=False,
                confidence=inner.confidence * 0.8,
            )
        if usage_context == "borrowed_mut":
            return RustType(
                name=inner.name, is_reference=True, is_mutable=True,
                confidence=inner.confidence * 0.75,
            )
        # Default: Box<T>
        return RustType(name="Box", generic_params=[inner.name], confidence=inner.confidence * 0.7)

    def _map_array_type(self, c_type: str) -> RustType:
        bracket_start = c_type.index("[")
        base = c_type[:bracket_start].strip()
        size_str = c_type[bracket_start + 1 : c_type.index("]")].strip()

        inner = self.map_type(base)

        if size_str:
            try:
                size = int(size_str)
                return RustType(
                    name=f"[{inner.name}; {size}]",
                    confidence=inner.confidence * 0.95,
                )
            except ValueError:
                return RustType(
                    name="Vec", generic_params=[inner.name],
                    confidence=inner.confidence * 0.7,
                )
        # Dynamic / unsized
        return RustType(name="Vec", generic_params=[inner.name], confidence=inner.confidence * 0.8)

    def _map_function_pointer(self, c_type: str) -> RustType:
        tokens = c_type.split("(")
        ret_type_str = tokens[0].strip() if tokens else "void"
        ret_mapped = self.map_type(ret_type_str)

        param_section = ""
        if len(tokens) >= 3:
            param_section = tokens[2].rstrip(")").strip()

        param_types: List[str] = []
        if param_section:
            for p in param_section.split(","):
                p = p.strip().rstrip(")")
                if p:
                    mapped = self.map_type(p)
                    param_types.append(mapped.name)

        params_str = ", ".join(param_types) if param_types else ""
        ret_str = ret_mapped.name if ret_mapped.name != "()" else "()"
        fn_sig = f"fn({params_str}) -> {ret_str}"
        return RustType(name=fn_sig, confidence=0.6)

    def map_struct_fields(self, struct_name: str) -> List[Tuple[str, RustType]]:
        fields = self._struct_map.get(struct_name, [])
        mapped: List[Tuple[str, RustType]] = []
        for field_name, field_type in fields:
            mapped.append((field_name, self.map_type(field_type)))
        return mapped


# ---------------------------------------------------------------------------
# OwnershipAnalyzer
# ---------------------------------------------------------------------------


def _normalize_body(body) -> list:
    """Normalize body field to a list of statement dicts."""
    if isinstance(body, list):
        return body
    if isinstance(body, dict):
        if "stmts" in body:
            return body["stmts"]
        return [body]
    if isinstance(body, str):
        return []
    return []


class OwnershipAnalyzer:
    """Infers Rust ownership semantics from C pointer usage patterns."""

    def __init__(self) -> None:
        self._alloc_functions = {"malloc", "calloc", "realloc", "strdup", "strndup"}
        self._free_functions = {"free"}

    def analyze_ownership(self, function_ast: dict) -> Dict[str, OwnershipSuggestion]:
        suggestions: Dict[str, OwnershipSuggestion] = {}
        body = _normalize_body(function_ast.get("body", []))
        params = function_ast.get("params", [])
        variables = self._collect_pointer_vars(body, params)

        for var_name, info in variables.items():
            kind, reason, confidence = self._infer_ownership(var_name, info, body)
            suggestions[var_name] = OwnershipSuggestion(
                variable=var_name, kind=kind, reason=reason, confidence=confidence
            )
        return suggestions

    def _collect_pointer_vars(
        self, body: list, params: list
    ) -> Dict[str, Dict[str, Any]]:
        variables: Dict[str, Dict[str, Any]] = {}
        for param in params:
            if "*" in param.get("type", ""):
                variables[param["name"]] = {
                    "source": "param",
                    "type": param["type"],
                    "allocated": False,
                    "freed": False,
                    "returned": False,
                    "stored_in_struct": False,
                    "passed_to_callee": False,
                    "mutated": False,
                    "alias_count": 0,
                }

        for stmt in body:
            if not isinstance(stmt, dict): continue
            self._scan_statement(stmt, variables)
        return variables

    def _scan_statement(self, stmt: dict, variables: Dict[str, Dict[str, Any]]) -> None:
        kind = stmt.get("kind", "")

        if kind == "declaration" and "*" in stmt.get("type", ""):
            name = stmt.get("name", "")
            init = stmt.get("init", "")
            is_alloc = any(fn in str(init) for fn in self._alloc_functions)
            variables[name] = {
                "source": "local",
                "type": stmt.get("type", ""),
                "allocated": is_alloc,
                "freed": False,
                "returned": False,
                "stored_in_struct": False,
                "passed_to_callee": False,
                "mutated": False,
                "alias_count": 0,
            }

        if kind == "call":
            func_name = stmt.get("function", "")
            args = stmt.get("args", [])
            if func_name in self._free_functions:
                for arg in args:
                    if arg in variables:
                        variables[arg]["freed"] = True
            else:
                for arg in args:
                    if arg in variables:
                        variables[arg]["passed_to_callee"] = True

        if kind == "return":
            val = stmt.get("value", "")
            if val in variables:
                variables[val]["returned"] = True

        if kind == "assignment":
            lhs = stmt.get("lhs", "")
            rhs = stmt.get("rhs", "")
            if "." in lhs or "->" in lhs:
                if rhs in variables:
                    variables[rhs]["stored_in_struct"] = True
            if rhs in variables:
                variables[rhs]["alias_count"] += 1
            if lhs in variables:
                variables[lhs]["mutated"] = True

        # Recurse into sub-blocks
        for sub in _normalize_body(stmt.get("body", [])):
            self._scan_statement(sub, variables)
        for sub in _normalize_body(stmt.get("else_body", [])):
            self._scan_statement(sub, variables)

    def _infer_ownership(
        self, var_name: str, info: Dict[str, Any], body: list
    ) -> Tuple[OwnershipKind, str, float]:
        # Multiple aliases -> shared ownership
        if info["alias_count"] > 1:
            return (
                OwnershipKind.RC,
                f"'{var_name}' is aliased {info['alias_count']} times; use Rc<T>",
                0.65,
            )

        # Allocated and freed in same function -> owned / Box
        if info["allocated"] and info["freed"]:
            return (
                OwnershipKind.OWNED,
                f"'{var_name}' is allocated and freed locally; use owned value",
                0.9,
            )

        # Allocated but returned (not freed) -> Box<T>
        if info["allocated"] and info["returned"]:
            return (
                OwnershipKind.BOXED,
                f"'{var_name}' is allocated and returned; use Box<T>",
                0.85,
            )

        # Stored in struct -> Box or lifetime
        if info["stored_in_struct"]:
            return (
                OwnershipKind.BOXED,
                f"'{var_name}' stored in struct field; use Box<T> or add lifetime",
                0.7,
            )

        # Param passed to callee but not freed -> borrow
        if info["source"] == "param" and info["passed_to_callee"] and not info["freed"]:
            if info["mutated"]:
                return (
                    OwnershipKind.BORROWED_MUT,
                    f"'{var_name}' passed and mutated; use &mut T",
                    0.8,
                )
            return (
                OwnershipKind.BORROWED,
                f"'{var_name}' passed to callee without modification; use &T",
                0.85,
            )

        # Param not freed, not stored -> borrow
        if info["source"] == "param" and not info["freed"]:
            if info["mutated"]:
                return (
                    OwnershipKind.BORROWED_MUT,
                    f"'{var_name}' is a param that is mutated; use &mut T",
                    0.8,
                )
            return (
                OwnershipKind.BORROWED,
                f"'{var_name}' is a param used read-only; use &T",
                0.85,
            )

        # Allocated but not freed and not returned -> likely leaked / Box
        if info["allocated"] and not info["freed"]:
            return (
                OwnershipKind.BOXED,
                f"'{var_name}' allocated but never freed; suggest Box<T>",
                0.6,
            )

        # Fallback
        return (
            OwnershipKind.RAW,
            f"'{var_name}' usage is ambiguous; keeping as raw pointer",
            0.4,
        )


# ---------------------------------------------------------------------------
# UnsafeBoundaryDetector
# ---------------------------------------------------------------------------

class UnsafeBoundaryDetector:
    """Identifies code that requires ``unsafe`` in Rust."""

    UNSAFE_PATTERNS = {
        "ptr_deref": "raw pointer dereference",
        "ffi_call": "call to external C function",
        "inline_asm": "inline assembly usage",
        "union_access": "union field access",
        "ptr_cast": "pointer type cast",
        "ptr_arith": "pointer arithmetic",
        "variadic": "variadic function call",
    }

    def detect_unsafe_needs(self, function_ast: dict) -> List[UnsafeReason]:
        reasons: List[UnsafeReason] = []
        func_name = function_ast.get("name", "<unknown>")
        body = _normalize_body(function_ast.get("body", []))
        self._scan_for_unsafe(body, func_name, reasons)
        return reasons

    def _scan_for_unsafe(
        self, stmts: list, func_name: str, reasons: List[UnsafeReason]
    ) -> None:
        for stmt in stmts:
            if not isinstance(stmt, dict): continue
            kind = stmt.get("kind", "")
            loc = stmt.get("location", func_name)

            # Pointer dereference
            expr = str(stmt.get("expression", "")) + str(stmt.get("lhs", ""))
            if expr.startswith("*") or "->*" in expr:
                reasons.append(UnsafeReason(
                    kind="ptr_deref", location=loc,
                    description="Raw pointer dereference detected",
                    suggestion="Wrap in unsafe block; add safety comment",
                ))

            # FFI / extern call
            if kind == "call":
                callee = stmt.get("function", "")
                if stmt.get("is_extern", False) or callee.startswith("__"):
                    reasons.append(UnsafeReason(
                        kind="ffi_call", location=loc,
                        description=f"Call to external function '{callee}'",
                        suggestion=f"Create safe wrapper around '{callee}'",
                    ))

            # Inline assembly
            if kind == "asm" or "asm" in str(stmt.get("value", "")):
                reasons.append(UnsafeReason(
                    kind="inline_asm", location=loc,
                    description="Inline assembly block",
                    suggestion="Encapsulate in unsafe fn with clear safety contract",
                ))

            # Union field access
            if kind == "member_access" and stmt.get("is_union", False):
                reasons.append(UnsafeReason(
                    kind="union_access", location=loc,
                    description="Access to union field",
                    suggestion="Consider converting to Rust enum with variants",
                ))

            # Pointer cast
            if kind == "cast" and "*" in str(stmt.get("target_type", "")):
                reasons.append(UnsafeReason(
                    kind="ptr_cast", location=loc,
                    description="Cast between pointer types",
                    suggestion="Use safe transmute alternative or unsafe with justification",
                ))

            # Pointer arithmetic
            if kind in ("binary_op", "assignment"):
                op = stmt.get("op", "")
                if op in ("+", "-", "+=", "-=") and "*" in str(stmt.get("type", "")):
                    reasons.append(UnsafeReason(
                        kind="ptr_arith", location=loc,
                        description="Pointer arithmetic operation",
                        suggestion="Replace with slice indexing or iterator",
                    ))

            # Variadic
            if kind == "call" and stmt.get("is_variadic", False):
                reasons.append(UnsafeReason(
                    kind="variadic", location=loc,
                    description="Variadic function call",
                    suggestion="Use Rust macro or generic function instead",
                ))

            # Recurse into sub-blocks
            for sub in _normalize_body(stmt.get("body", [])):
                self._scan_for_unsafe([sub], func_name, reasons)
            for sub in _normalize_body(stmt.get("else_body", [])):
                self._scan_for_unsafe([sub], func_name, reasons)

    def suggest_safe_wrapper(self, function_ast: dict) -> Optional[str]:
        reasons = self.detect_unsafe_needs(function_ast)
        if not reasons:
            return None
        func_name = function_ast.get("name", "unknown")
        params = function_ast.get("params", [])

        param_list = ", ".join(f"{p.get('name', '_')}: /* mapped */" for p in params)
        lines = [
            f"/// Safe wrapper around `{func_name}`.",
            f"/// # Safety",
        ]
        for r in reasons:
            lines.append(f"/// - {r.description}: {r.suggestion}")
        call_args = ", ".join(p.get("name", "_") for p in params)
        lines.append(f"pub fn {func_name}_safe({param_list}) -> /* mapped */ {{")
        lines.append(f"    unsafe {{ {func_name}({call_args}) }}")
        lines.append("}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# FFIWrapperGenerator
# ---------------------------------------------------------------------------

class FFIWrapperGenerator:
    """Generates FFI glue code for incremental migration."""

    def __init__(self, type_mapper: Optional[TypeMapper] = None) -> None:
        self._type_mapper = type_mapper or TypeMapper()

    def generate_ffi_wrapper(self, c_function: dict) -> str:
        func_name = c_function.get("name", "unknown")
        params = c_function.get("params", [])
        ret_type = c_function.get("return_type", "void")

        rust_ret = self._type_mapper.map_type(ret_type)
        rust_params: List[str] = []
        call_args: List[str] = []

        for p in params:
            p_name = p.get("name", "_")
            p_type = p.get("type", "int")
            mapped = self._type_mapper.map_type(p_type)
            rust_params.append(f"{p_name}: {mapped.to_string()}")
            call_args.append(self._marshal_arg(p_name, p_type, mapped))

        param_str = ", ".join(rust_params)
        ret_str = rust_ret.to_string()
        ret_clause = f" -> {ret_str}" if ret_str != "()" else ""

        lines = [
            'extern "C" {',
            f"    fn {func_name}({param_str}){ret_clause};",
            "}",
            "",
            f"/// Safe Rust wrapper for C function `{func_name}`.",
            f"pub fn {func_name}_rs({param_str}){ret_clause} {{",
        ]

        call_args_str = ", ".join(call_args)
        lines.append(f"    unsafe {{ {func_name}({call_args_str}) }}")
        lines.append("}")
        return "\n".join(lines)

    def generate_c_header(self, rust_function: dict) -> str:
        func_name = rust_function.get("name", "unknown")
        params = rust_function.get("params", [])
        ret_type = rust_function.get("return_type", "()")

        c_ret = self._reverse_map_type(ret_type)
        c_params: List[str] = []
        for p in params:
            p_name = p.get("name", "_")
            p_type = p.get("type", "i32")
            c_type = self._reverse_map_type(p_type)
            c_params.append(f"{c_type} {p_name}")

        param_str = ", ".join(c_params) if c_params else "void"
        guard = func_name.upper() + "_H"
        lines = [
            f"/* Auto-generated C header for Rust function {func_name} */",
            f"#ifndef {guard}",
            f"#define {guard}",
            "",
            "#ifdef __cplusplus",
            'extern "C" {',
            "#endif",
            "",
            f"{c_ret} {func_name}({param_str});",
            "",
            "#ifdef __cplusplus",
            "}",
            "#endif",
            "",
            f"#endif /* {guard} */",
        ]
        return "\n".join(lines)

    def _marshal_arg(self, name: str, c_type: str, rust_type: RustType) -> str:
        if "char*" in c_type.replace(" ", ""):
            return f"{name}.as_ptr()"
        if rust_type.name == "String":
            return f"{name}.as_ptr() as *const i8"
        return name

    def _reverse_map_type(self, rust_type: str) -> str:
        reverse_map: Dict[str, str] = {
            "i8": "char",
            "u8": "unsigned char",
            "i16": "short",
            "u16": "unsigned short",
            "i32": "int",
            "u32": "unsigned int",
            "i64": "long",
            "u64": "unsigned long",
            "f32": "float",
            "f64": "double",
            "bool": "_Bool",
            "usize": "size_t",
            "()": "void",
            "String": "char*",
            "*mut u8": "void*",
        }
        return reverse_map.get(rust_type, rust_type)


# ---------------------------------------------------------------------------
# RiskAssessor
# ---------------------------------------------------------------------------

class RiskAssessor:
    """Scores functions by migration difficulty."""

    HIGH_RISK_KEYWORDS = {
        "goto", "setjmp", "longjmp", "signal", "asm", "__asm__",
        "alloca", "va_start", "va_arg", "va_end",
    }
    MEDIUM_RISK_KEYWORDS = {"malloc", "calloc", "realloc", "free", "memcpy", "memmove"}

    def assess_risk(self, function_ast: dict) -> RiskLevel:
        factors: List[str] = []
        score = 0

        body = _normalize_body(function_ast.get("body", []))
        stmt_count = self._count_statements(body)
        params = function_ast.get("params", [])

        # Function length
        if stmt_count > 100:
            score += 20
            factors.append(f"large function ({stmt_count} statements)")
        elif stmt_count > 50:
            score += 10
            factors.append(f"moderate function ({stmt_count} statements)")

        # Cyclomatic complexity
        complexity = self._cyclomatic_complexity(body)
        if complexity > 20:
            score += 25
            factors.append(f"high cyclomatic complexity ({complexity})")
        elif complexity > 10:
            score += 12
            factors.append(f"moderate cyclomatic complexity ({complexity})")
        elif complexity > 5:
            score += 5
            factors.append(f"mild cyclomatic complexity ({complexity})")

        # Pointer operations
        ptr_ops = self._count_pointer_ops(body)
        if ptr_ops > 10:
            score += 20
            factors.append(f"many pointer operations ({ptr_ops})")
        elif ptr_ops > 3:
            score += 8
            factors.append(f"some pointer operations ({ptr_ops})")

        # Void pointers
        void_ptrs = self._count_pattern(body, "void*") + self._count_pattern(body, "void *")
        if void_ptrs > 0:
            score += 15
            factors.append(f"void pointer usage ({void_ptrs} occurrences)")

        # Function pointers in params
        fn_ptr_params = sum(1 for p in params if "(*)" in p.get("type", ""))
        if fn_ptr_params > 0:
            score += 10
            factors.append(f"function pointer parameters ({fn_ptr_params})")

        # goto statements
        gotos = self._count_kind(body, "goto")
        if gotos > 0:
            score += 20
            factors.append(f"goto usage ({gotos} occurrences)")

        # Union access
        union_accesses = self._count_kind(body, "union_access")
        if union_accesses > 0:
            score += 10
            factors.append(f"union field access ({union_accesses})")

        # Variadic
        if function_ast.get("is_variadic", False):
            score += 15
            factors.append("variadic function")

        # Inline assembly
        asm_count = self._count_kind(body, "asm")
        if asm_count > 0:
            score += 25
            factors.append(f"inline assembly ({asm_count} blocks)")

        # Undefined behaviour patterns
        ub_count = self._count_ub_patterns(body)
        if ub_count > 0:
            score += 15
            factors.append(f"potential undefined behaviour ({ub_count} patterns)")

        # Determine level
        score = min(score, 100)
        if score >= 50:
            level = "high"
        elif score >= 20:
            level = "medium"
        else:
            level = "low"
            if not factors:
                factors.append("pure computation, no significant risk factors")

        return RiskLevel(level=level, score=score, factors=factors)

    def _count_statements(self, stmts: list) -> int:
        count = 0
        for stmt in stmts:
            if not isinstance(stmt, dict): continue
            count += 1
            count += self._count_statements(_normalize_body(stmt.get("body", [])))
            count += self._count_statements(_normalize_body(stmt.get("else_body", [])))
        return count

    def _cyclomatic_complexity(self, stmts: list) -> int:
        complexity = 1
        for stmt in stmts:
            if not isinstance(stmt, dict): continue
            kind = stmt.get("kind", "")
            if kind in ("if", "while", "for", "do_while", "case", "ternary"):
                complexity += 1
            elif kind == "switch":
                complexity += max(len(stmt.get("cases", [])), 1)
            elif kind == "binary_op" and stmt.get("op") in ("&&", "||"):
                complexity += 1
            sub_body = self._cyclomatic_complexity(_normalize_body(stmt.get("body", [])))
            sub_else = self._cyclomatic_complexity(_normalize_body(stmt.get("else_body", [])))
            complexity += max(sub_body - 1, 0)
            complexity += max(sub_else - 1, 0)
        return max(complexity, 1)

    def _count_pointer_ops(self, stmts: list) -> int:
        count = 0
        for stmt in stmts:
            if not isinstance(stmt, dict): continue
            text = str(stmt)
            count += text.count("->")
            count += text.count("*ptr")
            count += text.count("&var")
            if stmt.get("kind") == "cast" and "*" in str(stmt.get("target_type", "")):
                count += 1
            count += self._count_pointer_ops(_normalize_body(stmt.get("body", [])))
            count += self._count_pointer_ops(_normalize_body(stmt.get("else_body", [])))
        return count

    def _count_pattern(self, stmts: list, pattern: str) -> int:
        count = 0
        for stmt in stmts:
            if not isinstance(stmt, dict): continue
            text = str(stmt)
            count += text.count(pattern)
            count += self._count_pattern(_normalize_body(stmt.get("body", [])), pattern)
            count += self._count_pattern(_normalize_body(stmt.get("else_body", [])), pattern)
        return count

    def _count_kind(self, stmts: list, kind: str) -> int:
        count = 0
        for stmt in stmts:
            if not isinstance(stmt, dict): continue
            if stmt.get("kind") == kind:
                count += 1
            count += self._count_kind(_normalize_body(stmt.get("body", [])), kind)
            count += self._count_kind(_normalize_body(stmt.get("else_body", [])), kind)
        return count

    def _count_ub_patterns(self, stmts: list) -> int:
        count = 0
        for stmt in stmts:
            if not isinstance(stmt, dict): continue
            text = str(stmt)
            if stmt.get("kind") == "cast":
                src = str(stmt.get("source_type", ""))
                tgt = str(stmt.get("target_type", ""))
                if ("int" in src and "*" in tgt) or ("*" in src and "int" in tgt):
                    count += 1
            for kw in self.HIGH_RISK_KEYWORDS:
                if kw in text:
                    count += 1
            count += self._count_ub_patterns(_normalize_body(stmt.get("body", [])))
            count += self._count_ub_patterns(_normalize_body(stmt.get("else_body", [])))
        return count


# ---------------------------------------------------------------------------
# TypeDependencyGraph
# ---------------------------------------------------------------------------

class TypeDependencyGraph:
    """Tracks dependencies between C types (structs referencing other structs)."""

    def __init__(self) -> None:
        self._deps: Dict[str, Set[str]] = defaultdict(set)
        self._types: Set[str] = set()

    def add_type(self, name: str, referenced_types: List[str]) -> None:
        self._types.add(name)
        for ref in referenced_types:
            self._deps[name].add(ref)
            self._types.add(ref)

    def topological_sort(self) -> List[str]:
        in_degree: Dict[str, int] = {t: 0 for t in self._types}
        for t, deps in self._deps.items():
            for d in deps:
                if d in in_degree:
                    in_degree[d] += 1

        queue = deque(sorted(t for t, d in in_degree.items() if d == 0))
        result: List[str] = []
        while queue:
            t = queue.popleft()
            result.append(t)
            for dep in sorted(self._deps.get(t, set())):
                if dep in in_degree:
                    in_degree[dep] -= 1
                    if in_degree[dep] == 0:
                        queue.append(dep)

        remaining = sorted(self._types - set(result))
        result.extend(remaining)
        return result

    def find_cycles(self) -> List[List[str]]:
        WHITE, GRAY, BLACK = 0, 1, 2
        color: Dict[str, int] = {t: WHITE for t in self._types}
        path: List[str] = []
        cycles: List[List[str]] = []

        def dfs(node: str) -> None:
            color[node] = GRAY
            path.append(node)
            for dep in sorted(self._deps.get(node, set())):
                if dep not in color:
                    continue
                if color[dep] == GRAY:
                    idx = path.index(dep)
                    cycles.append(path[idx:] + [dep])
                elif color[dep] == WHITE:
                    dfs(dep)
            path.pop()
            color[node] = BLACK

        for t in sorted(self._types):
            if color[t] == WHITE:
                dfs(t)
        return cycles


# ---------------------------------------------------------------------------
# MigrationPlanner (main entry point)
# ---------------------------------------------------------------------------

class MigrationPlanner:
    """Orchestrates the full C-to-Rust migration planning pipeline."""

    def __init__(self) -> None:
        self.type_mapper = TypeMapper()
        self.ownership_analyzer = OwnershipAnalyzer()
        self.unsafe_detector = UnsafeBoundaryDetector()
        self.ffi_generator = FFIWrapperGenerator(self.type_mapper)
        self.risk_assessor = RiskAssessor()

    @staticmethod
    def _type_to_str(t) -> str:
        """Convert a type value (str or dict) to string."""
        if isinstance(t, str):
            return t
        if isinstance(t, dict):
            name = t.get("name", "")
            ptr = t.get("pointer_depth", 0)
            return name + "*" * ptr if ptr else name
        return str(t)

    @staticmethod
    def _normalize_dict_ast(d: dict) -> dict:
        """Normalize an asdict-produced dict to the expected format."""
        result: dict = {
            "functions": [], "typedefs": [], "structs": [],
            "enums": [], "unions": [], "globals": []
        }
        for f in d.get("functions", []):
            fd: dict = {"name": f.get("name", "")}
            rt = f.get("return_type", "void")
            fd["return_type"] = MigrationPlanner._type_to_str(rt)
            params = f.get("params", f.get("parameters", []))
            fd["params"] = []
            for p in (params or []):
                pd: dict = {"name": p.get("name", "")}
                pt = p.get("type_spec", p.get("type", ""))
                pd["type"] = MigrationPlanner._type_to_str(pt)
                pd["pointer_depth"] = (pt.get("pointer_depth", 0)
                                       if isinstance(pt, dict) else 0)
                fd["params"].append(pd)
            body = f.get("body", "")
            fd["body"] = _normalize_body(body)
            fd["is_variadic"] = f.get("is_variadic", False)
            result["functions"].append(fd)
        for key in ("typedefs", "structs", "enums", "unions", "globals"):
            result[key] = d.get(key, [])
        # Also pull type_definitions into structs/enums/typedefs
        for td in d.get("type_definitions", []):
            if isinstance(td, dict):
                kind = td.get("kind", "")
                if "Struct" in kind:
                    result["structs"].append(td)
                elif "Enum" in kind:
                    result["enums"].append(td)
                else:
                    result["typedefs"].append(td)
        return result

    @staticmethod
    def _ast_to_dict(ast_obj) -> dict:
        """Convert a CAST object to dict format."""
        result: dict = {"functions": [], "typedefs": [], "structs": [],
                        "enums": [], "unions": [], "globals": []}
        for f in getattr(ast_obj, 'functions', []):
            fd: dict = {"name": getattr(f, 'name', '')}
            rt = getattr(f, 'return_type', None)
            fd["return_type"] = getattr(rt, 'name', str(rt)) if rt else "void"
            params = getattr(f, 'params', []) or []
            fd["params"] = []
            for p in params:
                pd: dict = {"name": getattr(p, 'name', '')}
                pt = getattr(p, 'type_spec', None)
                pd["type"] = getattr(pt, 'name', str(pt)) if pt else ""
                pd["pointer_depth"] = getattr(pt, 'pointer_depth', 0) if pt else 0
                fd["params"].append(pd)
            body = getattr(f, 'body', None)
            fd["body"] = str(body) if body else ""
            fd["is_variadic"] = getattr(f, 'is_variadic', False)
            result["functions"].append(fd)
        for t in getattr(ast_obj, 'type_definitions', []):
            kind = type(t).__name__
            tname = getattr(t, 'name', '')
            if 'Struct' in kind:
                fields = [{"name": getattr(f, 'name', ''), "type": str(getattr(f, 'type_spec', ''))}
                          for f in getattr(t, 'fields', [])]
                result["structs"].append({"name": tname, "fields": fields})
            elif 'Enum' in kind:
                values = [{"name": getattr(v, 'name', str(v))}
                          for v in getattr(t, 'values', getattr(t, 'variants', []))]
                result["enums"].append({"name": tname, "values": values})
            elif 'Union' in kind:
                result["unions"].append({"name": tname})
            elif 'Typedef' in kind:
                result["typedefs"].append({"name": tname, "type": str(getattr(t, 'target_type', ''))})
        for g in getattr(ast_obj, 'global_vars', []):
            result["globals"].append({"name": getattr(g, 'name', str(g))})
        return result

    def plan(self, c_ast) -> MigrationPlan:
        # Convert AST object to dict if needed
        if not isinstance(c_ast, dict):
            c_ast = self._ast_to_dict(c_ast)
        else:
            # Normalize asdict-produced dicts
            c_ast = self._normalize_dict_ast(c_ast)
        functions = c_ast.get("functions", [])
        typedefs = c_ast.get("typedefs", [])
        structs = c_ast.get("structs", [])
        enums = c_ast.get("enums", [])
        unions = c_ast.get("unions", [])

        # Register types with the mapper
        for td in typedefs:
            self.type_mapper.register_typedef(td.get("name", ""), td.get("type", ""))
        for st in structs:
            fields = [(f.get("name", ""), f.get("type", "")) for f in st.get("fields", [])]
            self.type_mapper.register_struct(st.get("name", ""), fields)
        for en in enums:
            self.type_mapper.register_enum(
                en.get("name", ""), [v.get("name", "") for v in en.get("values", [])]
            )

        # Build call graph and determine migration order
        call_graph = self._build_call_graph(functions)
        type_graph = self._build_type_graph(structs, unions, typedefs)

        # Reverse topological order: leaf functions first
        topo = call_graph.topological_sort()
        topo.reverse()

        # Collect type mappings
        type_mappings = self._collect_type_mappings(functions, structs, enums, unions, typedefs)

        # Ownership analysis across all functions
        all_ownership: Dict[str, OwnershipSuggestion] = {}
        for func in functions:
            ownership = self.ownership_analyzer.analyze_ownership(func)
            for var, sug in ownership.items():
                key = f"{func.get('name', '?')}.{var}"
                all_ownership[key] = sug

        # Generate FFI wrappers
        ffi_wrappers: List[str] = []
        for func in functions:
            wrapper = self.ffi_generator.generate_ffi_wrapper(func)
            ffi_wrappers.append(wrapper)

        # Risk assessment per function
        risk_map: Dict[str, RiskLevel] = {}
        for func in functions:
            fname = func.get("name", "?")
            risk_map[fname] = self.risk_assessor.assess_risk(func)

        # Estimate total effort
        estimated_effort = self._estimate_effort(functions, risk_map)

        return MigrationPlan(
            migration_order=topo,
            type_mappings=type_mappings,
            ownership_suggestions=all_ownership,
            ffi_wrappers=ffi_wrappers,
            risk_assessment=risk_map,
            estimated_effort=estimated_effort,
        )

    def _build_call_graph(self, functions: list) -> CallGraph:
        graph = CallGraph()
        func_names = {f.get("name", "") for f in functions}
        for func in functions:
            name = func.get("name", "")
            graph.add_node(name)
            called = self._extract_calls(func.get("body", []))
            for callee in called:
                if callee in func_names:
                    graph.add_edge(name, callee)
        return graph

    def _extract_calls(self, stmts) -> Set[str]:
        calls: Set[str] = set()
        if not stmts or not isinstance(stmts, list):
            return calls
        for stmt in stmts:
            if not isinstance(stmt, dict):
                continue
            if stmt.get("kind") == "call":
                callee = stmt.get("function", "")
                if callee:
                    calls.add(callee)
            calls.update(self._extract_calls(_normalize_body(stmt.get("body", []))))
            eb = stmt.get("else_body")
            if eb:
                calls.update(self._extract_calls(_normalize_body(eb)))
            args = stmt.get("args_stmts")
            if args:
                calls.update(self._extract_calls(_normalize_body(args)))
        return calls

    def _build_type_graph(
        self, structs: list, unions: list, typedefs: list
    ) -> TypeDependencyGraph:
        graph = TypeDependencyGraph()
        for st in structs:
            name = st.get("name", "")
            refs = self._extract_type_refs(st.get("fields", []))
            graph.add_type(name, refs)
        for un in unions:
            name = un.get("name", "")
            refs = self._extract_type_refs(un.get("fields", []))
            graph.add_type(name, refs)
        for td in typedefs:
            name = td.get("name", "")
            underlying = td.get("type", "")
            base = underlying.replace("*", "").replace("struct ", "").strip()
            if base and base != name:
                graph.add_type(name, [base])
        return graph

    def _extract_type_refs(self, fields: list) -> List[str]:
        refs: List[str] = []
        for f in fields:
            ftype = f.get("type", "")
            base = ftype.replace("*", "").replace("const ", "").strip()
            if "[" in base:
                base = base[: base.index("[")].strip()
            if base.startswith("struct "):
                base = base[7:].strip()
            if base.startswith("enum "):
                base = base[5:].strip()
            if base.startswith("union "):
                base = base[6:].strip()
            if base and base not in TypeMapper.PRIMITIVE_MAP:
                refs.append(base)
        return refs

    def _collect_type_mappings(
        self, functions: list, structs: list, enums: list,
        unions: list, typedefs: list,
    ) -> Dict[str, RustType]:
        mappings: Dict[str, RustType] = {}

        seen_types: Set[str] = set()
        for func in functions:
            ret = func.get("return_type", "void")
            seen_types.add(ret)
            for p in func.get("params", []):
                seen_types.add(p.get("type", "int"))

        for ct in seen_types:
            if ct not in mappings:
                mappings[ct] = self.type_mapper.map_type(ct)

        for st in structs:
            name = st.get("name", "")
            mappings[f"struct {name}"] = RustType(name=name, confidence=0.9)
            for field_name, field_rt in self.type_mapper.map_struct_fields(name):
                key = f"struct {name}::{field_name}"
                mappings[key] = field_rt

        for en in enums:
            name = en.get("name", "")
            mappings[f"enum {name}"] = RustType(name=name, confidence=0.95)

        for un in unions:
            name = un.get("name", "")
            mappings[f"union {name}"] = RustType(name=f"{name}Union", confidence=0.6)

        for td in typedefs:
            alias = td.get("name", "")
            mappings[alias] = self.type_mapper.map_type(alias)

        return mappings

    def _estimate_effort(
        self, functions: list, risk_map: Dict[str, RiskLevel]
    ) -> int:
        total = 0
        for func in functions:
            fname = func.get("name", "?")
            stmt_count = self._count_stmts(_normalize_body(func.get("body", [])))
            risk = risk_map.get(fname)
            if risk and risk.level == "high":
                multiplier = 2.0
            elif risk and risk.level == "medium":
                multiplier = 1.5
            else:
                multiplier = 1.2
            total += int(max(stmt_count, 5) * multiplier)
        return total

    def _count_stmts(self, stmts) -> int:
        if not stmts or not isinstance(stmts, list):
            return 0
        count = 0
        for stmt in stmts:
            if not isinstance(stmt, dict): continue
            count += 1
            count += self._count_stmts(_normalize_body(stmt.get("body", [])))
            count += self._count_stmts(_normalize_body(stmt.get("else_body") or []))
        return count
